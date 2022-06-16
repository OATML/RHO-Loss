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


# def _corrupt_dataset(
#     dataset,
#     label_noise=False,
#     input_noise=False,
#     structured_noise=False,
#     pc_corrupted=0.1,
# ):
#     """
#     Corrupt dataset with either input noise (i.e., preserve label, corrupt inputs) or label noise
#     (preserve input, corrupt label), or both.


#     Args:
#         dataset: Input dataset, VisionDataset
#         label_noise: Whether to corrupt labels. Boolean.
#         input_noise: Whether to corrupt inputs. Boolean.
#         pc_corrupted: float 0-1, percentage of points to corrupt.

#     Returns: corruption_info dictionary, contains information about the corrupted datapoints.
#     """
#     if (not label_noise and not input_noise) and not structured_noise:
#         print("No corruption requested")
#         return {}

#     n_dataset = len(dataset)

#     n_corrupt = int(n_dataset * pc_corrupted)

#     n_classes = len(dataset.classes)
#     selected_indices = np.random.choice(
#         np.arange(n_dataset), size=n_corrupt, replace=False
#     )
#     print("pc corrupted:" + str(pc_corrupted))
#     if input_noise:  # note: this has not been tested
#         # method currently assumes that each input channel is an 8 bit integer i.e., [0, 255]
#         if isinstance(dataset, indices_ImageFolder):
#             raise NotImplementedError(
#                 "Input noise does not support current dataset, which is an indices_ImageFolder"
#             )

#         shape = dataset.data.shape[1:]
#         corrupted_shape = torch.Size((n_corrupt, *shape))

#         if isinstance(dataset.data, torch.Tensor):
#             dataset.data[selected_indices] = torch.randint(
#                 255,
#                 size=corrupted_shape,
#                 device=dataset.data.device,
#                 dtype=dataset.data.dtype,
#             )
#         elif isinstance(dataset.data, np.ndarray):
#             dataset.data[selected_indices] = np.random.randint(
#                 255, size=corrupted_shape, dtype=dataset.data.dtype
#             )
#         else:
#             raise NotImplementedError(
#                 "Only Tensor and ndarray supported for corruption with input noise"
#             )
#     if structured_noise:
#         if (
#             isinstance(dataset, indices_AmbiguousMNIST)
#             or isinstance(dataset, indices_MNIST)
#             or isinstance(dataset, indices_QMNIST)
#             or isinstance(dataset, indices_infiMNIST)
#         ):
#             print("structured_noise")
#             convert_back = False
#             if isinstance(dataset.targets, np.ndarray):
#                 old_dtype = dataset.targets.dtype
#                 dataset.targets = torch.tensor(dataset.targets)
#                 convert_back = True

#             if isinstance(dataset.targets, torch.Tensor):
#                 n_corrupt = 0
#                 # if 3 --> 5, 4 --> 5, 9 --> 7
#                 selected_indices = (3 == dataset.targets).nonzero()[:, 0]
#                 selected_indices = selected_indices[
#                     : int(len(selected_indices) * pc_corrupted)
#                 ]
#                 dataset.targets[selected_indices] = (
#                     torch.ones(
#                         len(selected_indices),
#                         device=dataset.targets.device,
#                     )
#                     * 5
#                 ).type(dataset.targets.dtype)
#                 all_selected_indices = selected_indices.numpy()

#                 selected_indices = (9 == dataset.targets).nonzero()[:, 0]
#                 selected_indices = selected_indices[
#                     : int(len(selected_indices) * pc_corrupted)
#                 ]
#                 dataset.targets[selected_indices] = (
#                     torch.ones(
#                         len(selected_indices),
#                         device=dataset.targets.device,
#                     )
#                     * 7
#                 ).type(dataset.targets.dtype)
#                 all_selected_indices = np.append(
#                     all_selected_indices, selected_indices.numpy()
#                 )

#                 selected_indices = (4 == dataset.targets).nonzero()[:, 0]
#                 selected_indices = selected_indices[
#                     : int(len(selected_indices) * pc_corrupted)
#                 ]
#                 dataset.targets[selected_indices] = (
#                     torch.ones(
#                         len(selected_indices),
#                         device=dataset.targets.device,
#                     )
#                     * 9
#                 ).type(dataset.targets.dtype)
#                 all_selected_indices = np.append(
#                     all_selected_indices, selected_indices.numpy()
#                 )

#                 selected_indices = all_selected_indices
#                 n_corrupt = len(selected_indices)
#                 print("n_corrupt:" + str(n_corrupt))
#             if convert_back:
#                 dataset.targets = dataset.targets.numpy().astype(old_dtype)
#         else:
#             raise NotImplementedError(
#                 "Only MNIST based datasets supported for corruption with structured noise"
#             )

#     if label_noise:
#         if isinstance(dataset.targets, torch.Tensor):
#             dataset.targets[selected_indices] = torch.randint(
#                 n_classes,
#                 size=(n_corrupt,),
#                 device=dataset.targets.device,
#                 dtype=dataset.targets.dtype,
#             )
#         elif isinstance(dataset.targets, np.ndarray):
#             dataset.targets[selected_indices] = np.random.randint(
#                 n_classes, size=(n_corrupt,), dtype=dataset.targets.dtype
#             )
#         elif isinstance(dataset.targets, list):
#             target_array = np.array(dataset.targets)
#             target_array[selected_indices] = np.random.randint(
#                 n_classes, size=(n_corrupt,)
#             ).tolist()
#             dataset.targets = target_array.tolist()  # for CIFAR10
#         else:
#             raise NotImplementedError(
#                 "Only Tensor, list and ndarray supported for corruption with label noise"
#             )

#     corruption_info = {
#         "label_noise": label_noise,
#         "input_noise": input_noise,
#         "structured_noise": structured_noise,
#         "pc_corrupted": pc_corrupted,
#         "n_corrupt": n_corrupt,
#         "corrupted_points": selected_indices.tolist(),
#         "corrupted_points_ndarray": selected_indices,
#     }

#     return corruption_info


# class Pad:
#     def __call__(self, image):
#         import torchvision.transforms.functional as F

#         w, h = 28, 28
#         max_wh = (
#             40  # hard-coded to the downloaded data in tMNIST instead of np.max([w, h])
#         )
#         hp = int((max_wh - w) / 2)
#         vp = int((max_wh - h) / 2)
#         padding = (hp, vp, hp, vp)

#         return F.pad(image, padding, 0, "constant")


# class GoldiproxDatamodule(pl.LightningDataModule):
#     def __init__(
#         self,
#         batch_size,
#         test_batch_size=None,
#         data_dir: str = "data",
#         sequence=None,
#         shuffle=None,
#         trainset_corruption=None,
#         valset_corruption=None,
#         testset_corruption=None,
#         pin_memory=False,
#         num_workers=0,
#         trainset_data_aug=False,
#         valset_data_aug=False,
#         valset_fraction=1.0,
#         trainsetsplit = True, #whether the trainset should be split into train set and holdout set (for IL model training). Only relevant for CIFAR10/100
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.data_dir = data_dir  # this is data file
#         self.batch_size = batch_size
#         if test_batch_size is not None:
#             self.test_batch_size = test_batch_size
#         else:
#             self.test_batch_size = batch_size
#         self.sequence = sequence
#         self.trainset_corruption = trainset_corruption
#         self.trainset_corruption_info = None
#         self.valset_corruption = valset_corruption
#         self.valset_corruption_info = None
#         self.testset_corruption = testset_corruption
#         self.testset_corruption_info = None
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.trainset_data_aug = trainset_data_aug
#         self.valset_data_aug = valset_data_aug
#         self.shuffle = shuffle
#         self.valset_fraction = valset_fraction
#         self.trainsetsplit = trainsetsplit
#         # attributes needed for double irrlomo training
#         self.indices_train_split_1 = None
#         self.indices_train_split_2 = None
#         self.indices_train_split_info = None


    # def setup(self, stage=None, double_irlomo=False):
    #     # Assign train/val datasets for use in dataloaders
    #     self.indices_train = self.indices_train_factory()
    #     if self.sequence is not None:
    #         self.indices_train.sequence = self.sequence

    #     if self.trainset_corruption is not None:
    #         self.corrupt_trainset(self.trainset_corruption)

    #     self.indices_val = self.indices_val_factory()

#         if self.valset_corruption is not None:
#             self.corrupt_valset(self.valset_corruption)

#         if double_irlomo:
#             # create new indices train; use deepcopy incase we have applied corruption, to maintain corruption
#             indices_train_copy = copy.deepcopy(self.indices_train)

#             # CIFAR-10 and CIFAR-1000 are subsetted before, and Subset objects don't have .targets attribute
#             if isinstance(indices_train_copy, Subset):
#                 train_targets = []
#                 for ind, x, y in indices_train_copy:
#                     train_targets.append(y)
#                 train_targets = np.array(train_targets)
#             else:
#                 train_targets = indices_train_copy.targets
#                 if isinstance(train_targets, list):  # required for CINIC10
#                     train_targets = np.array(train_targets)

#             unique_targets = np.unique(train_targets).tolist()

#             train_split_1 = []
#             train_split_2 = []

#             # ensure even class balance for the split sets
#             for t in unique_targets:
#                 target_indices = np.flatnonzero(train_targets == t)
                
#                 # shuffle, because some datasets are ordered (e.g. CINIC-10)
#                 rng = np.random.default_rng(1)
#                 target_indices = rng.permutation(target_indices)
                
#                 target_indices = target_indices.tolist()
#                 train_split_1.extend(target_indices[: int(len(target_indices) / 2)])
#                 train_split_2.extend(target_indices[int(len(target_indices) / 2) :])

#             self.indices_train_split_1 = Subset(indices_train_copy, train_split_1)
#             self.indices_train_split_2 = Subset(indices_train_copy, train_split_2)

#             self.indices_train_split_info = {
#                 "train_split_1_indices": train_split_1,
#                 "train_split_2_indices": train_split_2,
#             }

#         # Assign test dataset for use in dataloader(s)
#         # do this always
#         self.indices_test = self.indices_test_factory()

#         if self.testset_corruption is not None:
#             self.corrupt_testset(self.testset_corruption)

#     def train_dataloader(self):
#         # Note that if a sequence is provided one in non-SVP set-up should set shuffle to False.
#         return DataLoader(
#             self.indices_train,
#             batch_size=self.batch_size,
#             shuffle=self.shuffle,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#         )

#     def train_split_dataloaders(self):
#         return [
#             DataLoader(
#                 self.indices_train_split_1,
#                 batch_size=self.batch_size,
#                 shuffle=self.shuffle,
#                 num_workers=self.num_workers,
#                 pin_memory=self.pin_memory,
#             ),
#             DataLoader(
#                 self.indices_train_split_2,
#                 batch_size=self.batch_size,
#                 shuffle=self.shuffle,
#                 num_workers=self.num_workers,
#                 pin_memory=self.pin_memory,
#             ),
#         ]

#     def val_dataloader(self):
#         return DataLoader(
#             self.indices_val,
#             shuffle=True,
#             batch_size=self.test_batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.indices_test,
#             batch_size=self.test_batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#         )

#     def corrupt_trainset(self, corruption_info):
#         if self.indices_train is None:
#             raise AttributeError("You must call dm.setup first")

#         log.warning(
#             "You are corrupting the training dataset. Be warned that this introduces stochasticity; as such"
#             "you should use the irreducible loss model, rather than saved irreducible losses"
#         )
#         self.trainset_corruption_info = _corrupt_dataset(
#             self.indices_train,
#             corruption_info["label_noise"],
#             corruption_info["input_noise"],
#             corruption_info["structured_noise"],
#             corruption_info["pc_corrupted"],
#         )

#     def corrupt_valset(self, corruption_info):
#         if self.indices_val is None:
#             raise AttributeError("You must call dm.setup first")

#         log.warning("You are corrupting the validation dataset.")
#         self.valset_corruption_info = _corrupt_dataset(
#             self.indices_val,
#             corruption_info["label_noise"],
#             corruption_info["input_noise"],
#             corruption_info["structured_noise"],
#             corruption_info["pc_corrupted"],
#         )

#     def corrupt_testset(self, corruption_info):
#         if self.indices_test is None:
#             raise AttributeError("You must call dm.setup first in test mode")

#         log.warning("You are corrupting the test dataset.")
#         self.testset_corruption_info = _corrupt_dataset(
#             self.indices_test,
#             corruption_info["label_noise"],
#             corruption_info["input_noise"],
#             corruption_info["structured_noise"],
#             corruption_info["pc_corrupted"],
#         )

#     def percentage_corrupted(self, global_index, set="train"):
#         """
#         Computes what percentage of the global_index points were corrupted.

#         Args:
#             global_index: selected global indices

#         Returns:
#             percentage corrupted, if corruption was applied. Else none.

#         """

#         if set == "train":
#             corruption_info = self.trainset_corruption_info
#         elif set == "validation":
#             corruption_info = self.valset_corruption_info
#         elif set == "test":
#             corruption_info = self.testset_corruption_info
#         else:
#             raise ValueError(
#                 "Percentage corrupted set must be one of train, validation, or test"
#             )

#         if corruption_info is None:
#             return None

#         return np.in1d(
#             global_index.cpu().numpy(), corruption_info["corrupted_points_ndarray"]
#         ).mean()

import datasets
from transformers import AutoTokenizer
class GLUEDataModule(pl.LightningDataModule):

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "idx",
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        no_test_set_avail,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        shuffle = None,
        sequence = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.shuffle=shuffle
        self.no_test_set_avail = no_test_set_avail

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        self.sequence = sequence

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        if self.no_test_set_avail:
            train_set = self.dataset["train"].select(range(0, len(self.dataset["train"]), 2))
        else:
            train_set = self.dataset["train"]

        
        # if there is a sequence, i.e. in the main/large model training, override the dataset instance to only return elements from the core set
        if self.sequence is not None:
            setattr(train_set, "sequence", self.sequence)
            setattr(train_set, "idx", train_set["idx"])

            def patch_instance(instance):
                """Create a new class derived from instance, override its relevant method.
                Then set instance type to the new class. 
                I have to implement it like this because you can't directly monkey patch magic methods on the instance level."""
                class _(type(instance)):
                    def __getitem__(self, key):
                        """Can be used to index columns (by string names) or rows (by integer index or iterable of indices or bools)."""
                        if not isinstance(key, str):
                            key_temp = torch.tensor(self.sequence[key])
                            key = [loc_idx for loc_idx, idx in enumerate(train_set.idx) if idx in key_temp]

                            if len(key) == 1:
                                key = key[0]

                        return self._getitem(
                            key,
                        )
                    def __len__(self):
                        return len(self.sequence)

                instance.__class__ = _
                
                return

            patch_instance(train_set)

        return DataLoader(train_set, batch_size=self.train_batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            if self.no_test_set_avail:
                val_set = self.dataset["train"].select(range(1, len(self.dataset["train"]), 2))
            else:
                val_set = self.dataset["validation"]
            return DataLoader(val_set, batch_size=self.eval_batch_size, shuffle=True)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, shuffle=True) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            if self.no_test_set_avail:
                return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
            else:
                return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features