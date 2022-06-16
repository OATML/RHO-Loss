# regular imports
from pathlib import Path

import numpy as np
from numpy.lib.function_base import select

# lightning related imports
import pytorch_lightning as pl

import copy

# pytorch related imports
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, QMNIST
from torchvision.transforms.transforms import Resize

from src.datamodules.datasets.sequence_datasets import (
    indices_CIFAR10,
    indices_CIFAR100,
    indices_ImageFolder,
    indices_MNIST,
    indices_TMNIST,
    indices_QMNIST,
    indices_ImageNet,
    indices_AmbiguousMNIST,
    indices_infiMNIST,
)
from src.datamodules.clothing1m import Clothing1M
from src.utils import utils

SCRIPT_DIR = Path(__file__).parent.absolute()
print(f"cwd: {SCRIPT_DIR}")

log = utils.get_logger(__name__)


def _corrupt_dataset(
    dataset,
    label_noise=False,
    input_noise=False,
    structured_noise=False,
    pc_corrupted=0.1,
):
    """
    Corrupt dataset with either input noise (i.e., preserve label, corrupt inputs) or label noise
    (preserve input, corrupt label), or both.


    Args:
        dataset: Input dataset, VisionDataset
        label_noise: Whether to corrupt labels. Boolean.
        input_noise: Whether to corrupt inputs. Boolean.
        pc_corrupted: float 0-1, percentage of points to corrupt.

    Returns: corruption_info dictionary, contains information about the corrupted datapoints.
    """
    if (not label_noise and not input_noise) and not structured_noise:
        print("No corruption requested")
        return {}

    n_dataset = len(dataset)

    n_corrupt = int(n_dataset * pc_corrupted)

    n_classes = len(dataset.classes)
    selected_indices = np.random.choice(
        np.arange(n_dataset), size=n_corrupt, replace=False
    )
    print("pc corrupted:" + str(pc_corrupted))
    if input_noise:  # note: this has not been tested
        # method currently assumes that each input channel is an 8 bit integer i.e., [0, 255]
        if isinstance(dataset, indices_ImageFolder):
            raise NotImplementedError(
                "Input noise does not support current dataset, which is an indices_ImageFolder"
            )

        shape = dataset.data.shape[1:]
        corrupted_shape = torch.Size((n_corrupt, *shape))

        if isinstance(dataset.data, torch.Tensor):
            dataset.data[selected_indices] = torch.randint(
                255,
                size=corrupted_shape,
                device=dataset.data.device,
                dtype=dataset.data.dtype,
            )
        elif isinstance(dataset.data, np.ndarray):
            dataset.data[selected_indices] = np.random.randint(
                255, size=corrupted_shape, dtype=dataset.data.dtype
            )
        else:
            raise NotImplementedError(
                "Only Tensor and ndarray supported for corruption with input noise"
            )
    if structured_noise:
        if (
            isinstance(dataset, indices_AmbiguousMNIST)
            or isinstance(dataset, indices_MNIST)
            or isinstance(dataset, indices_QMNIST)
            or isinstance(dataset, indices_infiMNIST)
        ):
            print("structured_noise")
            convert_back = False
            if isinstance(dataset.targets, np.ndarray):
                old_dtype = dataset.targets.dtype
                dataset.targets = torch.tensor(dataset.targets)
                convert_back = True

            if isinstance(dataset.targets, torch.Tensor):
                n_corrupt = 0
                # if 3 --> 5, 4 --> 5, 9 --> 7
                selected_indices = (3 == dataset.targets).nonzero()[:, 0]
                selected_indices = selected_indices[
                    : int(len(selected_indices) * pc_corrupted)
                ]
                dataset.targets[selected_indices] = (
                    torch.ones(
                        len(selected_indices),
                        device=dataset.targets.device,
                    )
                    * 5
                ).type(dataset.targets.dtype)
                all_selected_indices = selected_indices.numpy()

                selected_indices = (9 == dataset.targets).nonzero()[:, 0]
                selected_indices = selected_indices[
                    : int(len(selected_indices) * pc_corrupted)
                ]
                dataset.targets[selected_indices] = (
                    torch.ones(
                        len(selected_indices),
                        device=dataset.targets.device,
                    )
                    * 7
                ).type(dataset.targets.dtype)
                all_selected_indices = np.append(
                    all_selected_indices, selected_indices.numpy()
                )

                selected_indices = (4 == dataset.targets).nonzero()[:, 0]
                selected_indices = selected_indices[
                    : int(len(selected_indices) * pc_corrupted)
                ]
                dataset.targets[selected_indices] = (
                    torch.ones(
                        len(selected_indices),
                        device=dataset.targets.device,
                    )
                    * 9
                ).type(dataset.targets.dtype)
                all_selected_indices = np.append(
                    all_selected_indices, selected_indices.numpy()
                )

                selected_indices = all_selected_indices
                n_corrupt = len(selected_indices)
                print("n_corrupt:" + str(n_corrupt))
            if convert_back:
                dataset.targets = dataset.targets.numpy().astype(old_dtype)
        else:
            raise NotImplementedError(
                "Only MNIST based datasets supported for corruption with structured noise"
            )

    if label_noise:
        if isinstance(dataset.targets, torch.Tensor):
            dataset.targets[selected_indices] = torch.randint(
                n_classes,
                size=(n_corrupt,),
                device=dataset.targets.device,
                dtype=dataset.targets.dtype,
            )
        elif isinstance(dataset.targets, np.ndarray):
            dataset.targets[selected_indices] = np.random.randint(
                n_classes, size=(n_corrupt,), dtype=dataset.targets.dtype
            )
        elif isinstance(dataset.targets, list):
            target_array = np.array(dataset.targets)
            target_array[selected_indices] = np.random.randint(
                n_classes, size=(n_corrupt,)
            ).tolist()
            dataset.targets = target_array.tolist()  # for CIFAR10
        else:
            raise NotImplementedError(
                "Only Tensor, list and ndarray supported for corruption with label noise"
            )

    corruption_info = {
        "label_noise": label_noise,
        "input_noise": input_noise,
        "structured_noise": structured_noise,
        "pc_corrupted": pc_corrupted,
        "n_corrupt": n_corrupt,
        "corrupted_points": selected_indices.tolist(),
        "corrupted_points_ndarray": selected_indices,
    }

    return corruption_info


class Pad:
    def __call__(self, image):
        import torchvision.transforms.functional as F

        w, h = 28, 28
        max_wh = (
            40  # hard-coded to the downloaded data in tMNIST instead of np.max([w, h])
        )
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)

        return F.pad(image, padding, 0, "constant")


class GoldiproxDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        test_batch_size=None,
        data_dir: str = "data",
        sequence=None,
        shuffle=None,
        trainset_corruption=None,
        valset_corruption=None,
        testset_corruption=None,
        pin_memory=False,
        num_workers=0,
        trainset_data_aug=False,
        valset_data_aug=False,
        valset_fraction=1.0,
        percent_clean=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir  # this is data file
        self.batch_size = batch_size
        if test_batch_size is not None:
            self.test_batch_size = test_batch_size
        else:
            self.test_batch_size = batch_size
        self.sequence = sequence
        self.trainset_corruption = trainset_corruption
        self.trainset_corruption_info = None
        self.valset_corruption = valset_corruption
        self.valset_corruption_info = None
        self.testset_corruption = testset_corruption
        self.testset_corruption_info = None
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.trainset_data_aug = trainset_data_aug
        self.valset_data_aug = valset_data_aug
        self.shuffle = shuffle
        self.valset_fraction = valset_fraction
        # attributes needed for double irrlomo training
        self.indices_train_split_1 = None
        self.indices_train_split_2 = None
        self.indices_train_split_info = None
        self.percent_clean = percent_clean

    def setup(self, stage=None, double_irlomo=False):
        # Assign train/val datasets for use in dataloaders
        self.indices_train = self.indices_train_factory()
        if self.sequence is not None:
            self.indices_train.sequence = self.sequence

        if self.trainset_corruption is not None:
            self.corrupt_trainset(self.trainset_corruption)

        self.indices_val = self.indices_val_factory()

        if self.valset_corruption is not None:
            self.corrupt_valset(self.valset_corruption)

        if double_irlomo:
            # create new indices train; use deepcopy incase we have applied corruption, to maintain corruption
            indices_train_copy = copy.deepcopy(self.indices_train)

            # CIFAR-10 and CIFAR-1000 are subsetted before, and Subset objects don't have .targets attribute
            if isinstance(indices_train_copy, Subset):
                train_targets = []
                for ind, x, y in indices_train_copy:
                    train_targets.append(y)
                train_targets = np.array(train_targets)
            else:
                train_targets = indices_train_copy.targets
                if isinstance(train_targets, list):  # required for CINIC10
                    train_targets = np.array(train_targets)

            unique_targets = np.unique(train_targets).tolist()

            train_split_1 = []
            train_split_2 = []

            # ensure even class balance for the split sets
            for t in unique_targets:
                target_indices = np.flatnonzero(train_targets == t)
                
                # shuffle, because some datasets are ordered (e.g. CINIC-10)
                rng = np.random.default_rng(1)
                target_indices = rng.permutation(target_indices)
                
                target_indices = target_indices.tolist()
                train_split_1.extend(target_indices[: int(len(target_indices) / 2)])
                train_split_2.extend(target_indices[int(len(target_indices) / 2) :])

            self.indices_train_split_1 = Subset(indices_train_copy, train_split_1)
            self.indices_train_split_2 = Subset(indices_train_copy, train_split_2)

            self.indices_train_split_info = {
                "train_split_1_indices": train_split_1,
                "train_split_2_indices": train_split_2,
            }

        # Assign test dataset for use in dataloader(s)
        # do this always
        self.indices_test = self.indices_test_factory()

        if self.testset_corruption is not None:
            self.corrupt_testset(self.testset_corruption)

    def train_dataloader(self):
        # Note that if a sequence is provided one in non-SVP set-up should set shuffle to False.
        return DataLoader(
            self.indices_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_split_dataloaders(self):
        return [
            DataLoader(
                self.indices_train_split_1,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ),
            DataLoader(
                self.indices_train_split_2,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ),
        ]

    def val_dataloader(self):
        return DataLoader(
            self.indices_val,
            shuffle=True,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.indices_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def corrupt_trainset(self, corruption_info):
        if self.indices_train is None:
            raise AttributeError("You must call dm.setup first")

        log.warning(
            "You are corrupting the training dataset. Be warned that this introduces stochasticity; as such"
            "you should use the irreducible loss model, rather than saved irreducible losses"
        )
        self.trainset_corruption_info = _corrupt_dataset(
            self.indices_train,
            corruption_info["label_noise"],
            corruption_info["input_noise"],
            corruption_info["structured_noise"],
            corruption_info["pc_corrupted"],
        )

    def corrupt_valset(self, corruption_info):
        if self.indices_val is None:
            raise AttributeError("You must call dm.setup first")

        log.warning("You are corrupting the validation dataset.")
        self.valset_corruption_info = _corrupt_dataset(
            self.indices_val,
            corruption_info["label_noise"],
            corruption_info["input_noise"],
            corruption_info["structured_noise"],
            corruption_info["pc_corrupted"],
        )

    def corrupt_testset(self, corruption_info):
        if self.indices_test is None:
            raise AttributeError("You must call dm.setup first in test mode")

        log.warning("You are corrupting the test dataset.")
        self.testset_corruption_info = _corrupt_dataset(
            self.indices_test,
            corruption_info["label_noise"],
            corruption_info["input_noise"],
            corruption_info["structured_noise"],
            corruption_info["pc_corrupted"],
        )

    def percentage_corrupted(self, global_index, set="train"):
        """
        Computes what percentage of the global_index points were corrupted.

        Args:
            global_index: selected global indices

        Returns:
            percentage corrupted, if corruption was applied. Else none.

        """

        if set == "train":
            corruption_info = self.trainset_corruption_info
        elif set == "validation":
            corruption_info = self.valset_corruption_info
        elif set == "test":
            corruption_info = self.testset_corruption_info
        else:
            raise ValueError(
                "Percentage corrupted set must be one of train, validation, or test"
            )

        if corruption_info is None:
            return None

        return np.in1d(
            global_index.cpu().numpy(), corruption_info["corrupted_points_ndarray"]
        ).mean()


class TMNISTDataModule(GoldiproxDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.075,), (0.258,))]
        )
        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), Pad(), transforms.Normalize((0.075,), (0.258,))]
        )

        self.num_workers = 4  # FIXME later to accept config settings

        self.indices_train_factory = lambda: indices_TMNIST(
            root=f"{SCRIPT_DIR}/tmnist/tmnist_train2.npy", transform=self.transform
        )
        self.indices_val_factory = lambda: indices_TMNIST(
            root=f"{SCRIPT_DIR}/tmnist/tmnist_val2.npy", transform=self.transform
        )
        self.indices_test_factory = lambda: indices_QMNIST(
            self.data_dir, "test50k", compat=True, transform=self.transform_test
        )
        if self.trainset_data_aug:
            log.warning(
                "Trainset data augmentation turned on, but this is not implemented for TMNIST"
            )
        if self.valset_data_aug:
            log.warning(
                "Valset data augmentation turned on, but this is not implemented for TMNIST"
            )


class QMNISTDataModule(GoldiproxDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.indices_train_factory = lambda: indices_MNIST(
            self.data_dir, train=True, transform=self.transform
        )

        self.indices_val_factory = lambda: indices_QMNIST(
            self.data_dir,
            "test50k",
            download=True,
            compat=True,
            transform=self.transform,
        )

        self.indices_test_factory = lambda: indices_MNIST(
            self.data_dir, train=False, transform=self.transform
        )
        if self.trainset_data_aug:
            log.warning(
                "Trainset data augmentation turned on, but this is not implemented for QMNIST"
            )
        if self.valset_data_aug:
            log.warning(
                "Valset data augmentation turned on, but this is not implemented for QMNIST"
            )


class CIFAR10DataModule(GoldiproxDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.data_augmented_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainplusvalset_size = 50000
        train_subset = list(range(0, trainplusvalset_size, 2))
        val_subset = list(range(1, trainplusvalset_size, 2))

        # if only a part of the val subset should be used
        if self.valset_fraction < 1:

            # all of this is only to make sure that the random subset is not very unbalanced
            temp_train_and_valset = indices_CIFAR10(
                self.data_dir,
                train=True,
                transform=self.transform
                if not self.valset_data_aug
                else self.data_augmented_transform,
            )

            targets = np.array(temp_train_and_valset.targets)
            unique_targets = np.unique(targets).tolist()

            reduced_val_subset = []
            for t in unique_targets:
                target_indices = np.flatnonzero(targets == t).tolist()
                target_indices_in_valset = list(
                    set(target_indices) & set(val_subset)
                )  # take targets that are in the validation subset only
                reduced_val_subset.extend(
                    target_indices_in_valset[: int(len(target_indices_in_valset) * self.valset_fraction)]
                )

            val_subset = reduced_val_subset

        log.info(f"Training set has {len(train_subset)} datapoints")
        log.info(f"Validation set has {len(val_subset)} datapoints")
        assert len(set(val_subset) & set(train_subset)) == 0  # ensure that the train and validation subset are disjoint

        if self.sequence is None:
            # if no sequence is given use a subset of CIFAR for train/val
            # otherwise use sequence given
            self.indices_train_factory = lambda: Subset(
                indices_CIFAR10(
                    self.data_dir,
                    train=True,
                    transform=self.transform
                    if not self.trainset_data_aug
                    else self.data_augmented_transform,
                ),
                train_subset,
            )
        else:
            self.indices_train_factory = lambda: indices_CIFAR10(
                self.data_dir,
                train=True,
                transform=self.transform,
                sequence=self.sequence,
            )

        self.indices_val_factory = lambda: Subset(
            indices_CIFAR10(
                self.data_dir,
                train=True,
                transform=self.transform
                if not self.valset_data_aug
                else self.data_augmented_transform,
            ),
            val_subset,
        )
        self.indices_test_factory = lambda: indices_CIFAR10(
            self.data_dir, train=False, transform=self.transform
        )
        if self.trainset_data_aug:
            log.info("Trainset data augmentation turned on")
        if self.valset_data_aug:
            log.info("Valset data augmentation turned on")

    def corrupt_trainset(self, corruption_info):
        """
        Standard corruption function, except operate on the underlying data from the subset option returned by the
        indices_train_factory(), because we made our own subsets for CIFAR. Hence this function has been overriden.
        """
        if self.indices_train is None:
            raise AttributeError("You must call dm.setup first")

        log.warning(
            "You are corrupting the training dataset. Be warned that this introduces stochasticity; as such"
            "you should use the irreducible loss model, rather than saved irreducible losses"
        )
        if self.sequence is not None:
            self.trainset_corruption_info = _corrupt_dataset(
                self.indices_train,
                corruption_info["label_noise"],
                corruption_info["input_noise"],
                corruption_info["structured_noise"],
                corruption_info["pc_corrupted"],
            )
        else:
            self.trainset_corruption_info = _corrupt_dataset(
                self.indices_train.dataset,
                corruption_info["label_noise"],
                corruption_info["input_noise"],
                corruption_info["structured_noise"],
                corruption_info["pc_corrupted"],
            )

    def corrupt_valset(self, corruption_info):
        """
        Standard corruption function, except operate on the underlying data from the subset option returned by the
        indices_train_factory(), because we made our own subsets for CIFAR. Hence this function has been overriden.
        """
        if self.indices_val is None:
            raise AttributeError("You must call dm.setup first")

        log.warning("You are corrupting the validation dataset.")
        if self.sequence is not None:
            self.valset_corruption_info = _corrupt_dataset(
                self.indices_val,
                corruption_info["label_noise"],
                corruption_info["input_noise"],
                corruption_info["structured_noise"],
                corruption_info["pc_corrupted"],
            )
        else:
            self.valset_corruption_info = _corrupt_dataset(
                self.indices_val.dataset,
                corruption_info["label_noise"],
                corruption_info["input_noise"],
                corruption_info["structured_noise"],
                corruption_info["pc_corrupted"],
            )


class CIFAR100DataModule(GoldiproxDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (3, 32, 32)
        self.num_classes = 100
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.data_augmented_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainplusvalset_size = 50000
        train_subset = list(range(0, trainplusvalset_size, 2))
        val_subset = list(range(1, trainplusvalset_size, 2))

        # if only a part of the val subset should be used
        if self.valset_fraction < 1:
            # all of this is only to make sure that the random subset is not very unbalanced
            temp_train_and_valset = indices_CIFAR100(
                self.data_dir,
                train=True,
                transform=self.transform
                if not self.valset_data_aug
                else self.data_augmented_transform,
            )

            targets = np.array(temp_train_and_valset.targets)
            unique_targets = np.unique(targets).tolist()

            reduced_val_subset = []
            for t in unique_targets:
                target_indices = np.flatnonzero(targets == t).tolist()
                target_indices_in_valset = list(
                    set(target_indices) & set(val_subset)
                )  # take targets that are in the validation subset only
                reduced_val_subset.extend(
                    target_indices_in_valset[: int(len(target_indices_in_valset) * self.valset_fraction)]
                )

            val_subset = reduced_val_subset

        log.info(f"Training set has {len(train_subset)} datapoints")
        log.info(f"Validation set has {len(val_subset)} datapoints")
        assert len(set(val_subset) & set(train_subset)) == 0  # ensure that the train and validation subset are disjoint

        if self.sequence is None:
            # if no sequence is given use a subset of CIFAR for train/val
            # otherwise use sequence given
            self.indices_train_factory = lambda: Subset(
                indices_CIFAR100(
                    self.data_dir,
                    train=True,
                    transform=self.transform
                    if not self.trainset_data_aug
                    else self.data_augmented_transform,
                ),
                train_subset,
            )
        else:
            self.indices_train_factory = lambda: indices_CIFAR100(
                self.data_dir,
                train=True,
                transform=self.transform,
                sequence=self.sequence,
            )

        self.indices_val_factory = lambda: Subset(
            indices_CIFAR100(
                self.data_dir,
                train=True,
                transform=self.transform
                if not self.valset_data_aug
                else self.data_augmented_transform,
            ),
            val_subset,
        )
        self.indices_test_factory = lambda: indices_CIFAR100(
            self.data_dir, train=False, transform=self.transform
        )

    def corrupt_trainset(self, corruption_info):
        """
        Standard corruption function, except operate on the underlying data from the subset option returned by the
        indices_train_factory(), because we made our own subsets for CIFAR. Hence this function has been overriden.
        """
        if self.indices_train is None:
            raise AttributeError("You must call dm.setup first")

        log.warning(
            "You are corrupting the training dataset. Be warned that this introduces stochasticity; as such"
            "you should use the irreducible loss model, rather than saved irreducible losses"
        )
        if self.sequence is not None:
            self.trainset_corruption_info = _corrupt_dataset(
                self.indices_train,
                corruption_info["label_noise"],
                corruption_info["input_noise"],
                corruption_info["structured_noise"],
                corruption_info["pc_corrupted"],
            )
        else:
            self.trainset_corruption_info = _corrupt_dataset(
                self.indices_train.dataset,
                corruption_info["label_noise"],
                corruption_info["input_noise"],
                corruption_info["structured_noise"],
                corruption_info["pc_corrupted"],
            )

    def corrupt_valset(self, corruption_info):
        """
        Standard corruption function, except operate on the underlying data from the subset option returned by the
        indices_train_factory(), because we made our own subsets for CIFAR. Hence this function has been overriden.
        """
        if self.indices_val is None:
            raise AttributeError("You must call dm.setup first")

        log.warning("You are corrupting the validation dataset.")
        if self.sequence is not None:
            self.valset_corruption_info = _corrupt_dataset(
                self.indices_val,
                corruption_info["label_noise"],
                corruption_info["input_noise"],
                corruption_info["structured_noise"],
                corruption_info["pc_corrupted"],
            )
        else:
            self.valset_corruption_info = _corrupt_dataset(
                self.indices_val.dataset,
                corruption_info["label_noise"],
                corruption_info["input_noise"],
                corruption_info["structured_noise"],
                corruption_info["pc_corrupted"],
            )


class CINIC10DataModule(GoldiproxDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.47889522, 0.47227842, 0.43047404),
                    (0.24205776, 0.23828046, 0.25874835),
                ),
            ]
        )
        self.data_augmented_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.47889522, 0.47227842, 0.43047404),
                    (0.24205776, 0.23828046, 0.25874835),
                ),
            ]
        )

        self.data_dir = self.data_dir + "/CINIC"

        self.indices_train_factory = lambda: indices_ImageFolder(
            self.data_dir + "/train",
            transform=self.transform
            if not self.trainset_data_aug
            else self.data_augmented_transform,
        )

        # if only a part of the val subset should be used
        if self.valset_fraction < 1:
            # all of this is only to make sure that the random subset is not very unbalanced
            temp_valset = indices_ImageFolder(
                self.data_dir + "/valid",
                transform=self.transform
                if not self.valset_data_aug
                else self.data_augmented_transform,
            )

            targets = np.array(temp_valset.targets)
            unique_targets = np.unique(targets).tolist()

            val_subset = []
            for t in unique_targets:
                target_indices = np.flatnonzero(targets == t)
                np.random.shuffle(
                    target_indices
                )  # shuffle for CINIC, bc CINIC images are from CIFAR and ImageNet and might be sorted
                target_indices = target_indices.tolist()
                val_subset.extend(
                    target_indices[: int(len(target_indices) * self.valset_fraction)]
                )

            self.indices_val_factory = lambda: Subset(
                indices_ImageFolder(
                    self.data_dir + "/valid",
                    transform=self.transform
                    if not self.valset_data_aug
                    else self.data_augmented_transform,
                ),
                val_subset,
            )
        else:
            self.indices_val_factory = lambda: indices_ImageFolder(
                self.data_dir + "/valid",
                transform=self.transform
                if not self.valset_data_aug
                else self.data_augmented_transform,
            )

        self.indices_test_factory = lambda: indices_ImageFolder(
            self.data_dir + "/test", transform=self.transform
        )
        if self.trainset_data_aug:
            log.info("Trainset data augmentation turned on")
        if self.valset_data_aug:
            log.info("Valset data augmentation turned on")


class Clothing1MDataModule(GoldiproxDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (3, 224, 224)
        self.num_classes = 14
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.data_augmented_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.indices_train_factory = lambda: Clothing1M(
            root=self.data_dir,
            mode="train",
            percent_clean=self.percent_clean,
            transform=self.transform
            if not self.trainset_data_aug
            else self.data_augmented_transform,
        )

        self.indices_val_factory = lambda: Clothing1M(
            self.data_dir,
            mode="val",
            transform=self.transform
            if not self.valset_data_aug
            else self.data_augmented_transform,
        )

        self.indices_test_factory = lambda: Clothing1M(
            self.data_dir, mode="test", transform=self.transform
        )

        if self.trainset_data_aug:
            log.info("Trainset data augmentation turned on")
        if self.valset_data_aug:
            log.info("Valset data augmentation turned on")
    
    def percentage_clean(self, selected_global_indices, set="train"):
        if set == "train":
            clean_indicator = self.indices_train.indicate_clean(selected_global_indices)
            return np.mean(clean_indicator)
        elif set == "validation":
            return self.indices_val.indicate_clean(selected_global_indices).mean()
        elif set == "test":
            return self.indices_test.indicate_clean(selected_global_indices).mean()


class ImageNetDataModule(GoldiproxDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (3, 224, 224)
        self.num_classes = 1000
        self.image_size = 224
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.num_samples = 1281167 - self.num_imgs_per_val_class * self.num_classes
        self.data_augmented_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(224 + 32),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if self.trainset_data_aug:
            log.info("Trainset data augmentation turned on")
        if self.valset_data_aug:
            log.info("Valset data augmentation turned on")
        self.indices_train_factory = lambda: indices_ImageNet(
            self.data_dir,
            split="train",
            transform=self.transform
            if not self.trainset_data_aug
            else self.data_augmented_transform,
        )

        # if only a part of the val subset should be used
        if self.valset_fraction < 1:
            # all of this is only to make sure that the random subset is not very unbalanced
            temp_valset = lambda: indices_ImageNet(
                self.data_dir,
                split="val",
                transform=self.transform
                if not self.valset_data_aug
                else self.data_augmented_transform,
            )

            targets = np.array(temp_valset.targets)
            unique_targets = np.unique(targets).tolist()

            val_subset = []
            for t in unique_targets:
                target_indices = np.flatnonzero(targets == t).tolist()
                val_subset.extend(
                    target_indices[: int(len(target_indices) * self.valset_fraction)]
                )

            self.indices_val_factory = lambda: Subset(
                indices_ImageNet(
                    self.data_dir,
                    split="val",
                    transform=self.transform
                    if not self.valset_data_aug
                    else self.data_augmented_transform,
                ),
                val_subset,
            )
        else:
            self.indices_val_factory = lambda: indices_ImageNet(
                self.data_dir,
                split="val",
                transform=self.transform
                if not self.valset_data_aug
                else self.data_augmented_transform,
            )
        self.indices_test_factory = self.indices_val_factory


class CINIC10RelevanceDataModule(GoldiproxDatamodule):
    """
    Like CINIC10, but the validation and test set only have the first five classes. The train set has all classes.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.47889522, 0.47227842, 0.43047404),
                    (0.24205776, 0.23828046, 0.25874835),
                ),
            ]
        )

        self.data_augmented_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.47889522, 0.47227842, 0.43047404),
                    (0.24205776, 0.23828046, 0.25874835),
                ),
            ]
        )

        self.data_dir = self.data_dir + "/CINIC"

        validation_dataset = indices_ImageFolder(
            self.data_dir + "/valid", transform=self.transform
        )

        valid_subset = list(np.nonzero(np.array(validation_dataset.targets) < 4.5)[0])

        test_dataset = indices_ImageFolder(
            self.data_dir + "/test", transform=self.transform
        )

        test_subset = list(np.nonzero(np.array(test_dataset.targets) < 4.5)[0])

        self.indices_train_factory = lambda: indices_ImageFolder(
            self.data_dir + "/train",
            transform=self.transform
            if not self.trainset_data_aug
            else self.data_augmented_transform,
        )

        self.indices_val_factory = lambda: Subset(
            indices_ImageFolder(
                self.data_dir + "/valid",
                transform=self.transform
                if not self.valset_data_aug
                else self.data_augmented_transform,
            ),
            valid_subset,
        )

        self.indices_test_factory = lambda: Subset(
            indices_ImageFolder(self.data_dir + "/test", transform=self.transform),
            test_subset,
        )
        if self.trainset_data_aug:
            log.info("Trainset data augmentation turned on")
        if self.valset_data_aug:
            log.info("Valset data augmentation turned on")

    def percentage_targets_relevant(self, targets):
        return (targets < 5).cpu().numpy().mean()


class AmbiMNISTDataModule(GoldiproxDatamodule):
    """
    train from ambi mnist
    val from QMNIST test50k
    test from mnist test
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.transform_ambi = transforms.Compose(
            [transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.indices_train_factory = lambda: indices_AmbiguousMNIST(
            self.data_dir, train=True, transform=self.transform_ambi
        )
        self.indices_val_factory = lambda: indices_QMNIST(
            self.data_dir,
            "test50k",
            download=True,
            compat=True,
            transform=self.transform,
        )
        self.indices_test_factory = lambda: indices_MNIST(
            self.data_dir, train=False, transform=self.transform
        )
        if self.trainset_data_aug:
            log.warning(
                "Trainset data augmentation turned on, but this is not implemented for QMNIST"
            )
        if self.valset_data_aug:
            log.warning(
                "Valset data augmentation turned on, but this is not implemented for QMNIST"
            )


class infiMNISTDataModule(GoldiproxDatamodule):
    """
    train from infinite mnist/mnist 8m
    val from mnist train
    test from mnist test
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.transform_ambi = transforms.Compose(
            [transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.indices_train_factory = lambda: indices_infiMNIST(
            self.data_dir, transform=self.transform
        )

        self.indices_val_factory = lambda: indices_MNIST(
            self.data_dir, train=True, transform=self.transform
        )

        self.indices_test_factory = lambda: indices_MNIST(
            self.data_dir, train=False, transform=self.transform
        )
        if self.trainset_data_aug:
            log.warning(
                "Trainset data augmentation turned on, but this is not implemented for infiMNIST"
            )
        if self.valset_data_aug:
            log.warning(
                "Valset data augmentation turned on, but this is not implemented for infiMNIST"
            )


class DirtyClothing1MDataModule(GoldiproxDatamodule):
    def __init__(self, val_size=100000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (3, 224, 224)
        self.num_classes = 14
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.data_augmented_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        state = np.random.get_state()
        np.random.seed(0)
        all_set = np.arange(1061883)
        val_subset = np.random.choice(all_set, size=val_size, replace=False)
        # train_subset = np.setdiff1d(all_set,val_subset)
        np.random.set_state(state)
        
        self.indices_train_factory = lambda: Clothing1M(
            root=self.data_dir,
            mode="dirty_train",
            transform=self.transform
            if not self.trainset_data_aug
            else self.data_augmented_transform,
        )

        self.indices_val_factory = lambda: Subset(Clothing1M(
            self.data_dir,
            mode="dirty_train",
            transform=self.transform
            if not self.valset_data_aug
            else self.data_augmented_transform,
        ), val_subset)

        self.indices_test_factory = lambda: Clothing1M(
            self.data_dir,
            mode="test",
            transform=self.transform
        )

        if self.trainset_data_aug:
            log.info("Trainset data augmentation turned on")
        if self.valset_data_aug:
            log.info("Valset data augmentation turned on")

class NoisyOnlyClothing1MDataModule(GoldiproxDatamodule):
    def __init__(self, val_size=100000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = (3, 224, 224)
        self.num_classes = 14
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.data_augmented_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        state = np.random.get_state()
        np.random.seed(0)
        all_set = np.arange(1061883)
        val_subset = np.random.choice(all_set, size=val_size, replace=False)
        train_subset = np.setdiff1d(all_set,val_subset)
        np.random.set_state(state)
        
        self.indices_train_factory = lambda: Subset(Clothing1M(
            root=self.data_dir,
            mode="noisy_train",
            transform=self.transform
            if not self.trainset_data_aug
            else self.data_augmented_transform,
        ), train_subset)

        self.indices_val_factory = lambda: Subset(Clothing1M(
            self.data_dir,
            mode="noisy_train",
            transform=self.transform
            if not self.valset_data_aug
            else self.data_augmented_transform,
        ), val_subset)

        self.indices_test_factory = lambda: Clothing1M(
            self.data_dir,
            mode="test",
            transform=self.transform
        )

        if self.trainset_data_aug:
            log.info("Trainset data augmentation turned on")
        if self.valset_data_aug:
            log.info("Valset data augmentation turned on")
