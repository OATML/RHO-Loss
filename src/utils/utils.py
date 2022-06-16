import logging
import os
import warnings
from typing import List, Sequence
import subprocess

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn as nn
import numpy as np
import tqdm

from src.datamodules.clothing1m import Clothing1M

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "selection_method",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
        "optimizer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]
    if "selection_method" in config:
        hparams["selection_method"] = config["selection_method"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


### -------------------------------------------------------------------
# utils for temporarily masking a config dict so that hydra won't automatically
# instatiate it and instead passes it through for later instatiation.


def mask_config(config):
    """Mask config from hydra instantiation function by removing "_target_" key."""
    if config is not None:
        with open_dict(
            config
        ):  # this is needed to be able to edit omegaconf structured config dicts
            if "_target_" in config.keys():
                config["target"] = config.pop("_target_")
    return config


def unmask_config(config):
    """Re-introduce "_target_" key so that hydra instantiation function can be used"""
    if config is not None:
        if "target" in config.keys():
            config["_target_"] = config.pop("target")
    return config


### -------------------------------------------------------------------
### utils for precomputing the irreducible loss just once
### and then reusing it across all main model runs


def compute_losses_with_sanity_checks(dataloader, model, device=None):
    """Compute losses for full dataset.

    (I did not implement this with
    trainer.predict() because the trainset returns (index, x, y) and does not
    readily fit into the forward method of the irreducible loss model.)

    Returns:
        losses: Tensor, losses for the full dataset, sorted by <globa_index> (as
        returned from the train data loader). losses[global_index] is the
        irreducible loss of the datapoint with that global index. losses[idx] is
        nan if idx is not part of the dataset.
        targets: Tensor, targets of the datsets, sorted by <index> (as
        returned from the train data loader). Also see above.This is just used to verify that
        the data points are indexed in the same order as they were in this
        function call, when <losses> is used.
    """
    if isinstance(dataloader.dataset, Clothing1M):
        if device is None:
            device = model.device
        else:
            model.to(device)
        print("Computing irreducible loss full training dataset.")
        idx_of_control_images = [1, 3, 10, 30, 100, 300, 1000, 3000]
        control_images = [0] * len(idx_of_control_images)
        losses = torch.zeros(len(dataloader.dataset)).type(torch.FloatTensor)
        targets = torch.zeros(len(dataloader.dataset)).type(torch.LongTensor)
        prediction_correct = []
        
        with torch.inference_mode():
            for idx, x, target in tqdm.tqdm(dataloader):
                idx, x, target = idx, x.to(device), target.to(device)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, target, reduction="none")
                losses[idx] = loss.cpu()
                targets[idx] = target.cpu()
                prediction_correct.append(torch.eq(torch.argmax(logits, dim=1), target).cpu())

                for (id, image) in zip(idx, x):
                    if id in idx_of_control_images:
                        local_index = idx_of_control_images.index(id)
                        control_images[local_index] = image.cpu()

        acc = torch.cat(prediction_correct, dim=0)
        acc = acc.type(torch.FloatTensor).mean()
        average_loss = losses.mean()

        log.info(
            f"Accuracy of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {acc:.3f}\n"
            f"Average loss of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {average_loss:.3f}"
        )

        output = {
            "irreducible_losses": losses,
            "sorted_targets": targets,
            "idx_of_control_images": idx_of_control_images,
            "control_images": control_images,
            "heldout_accuracy": acc,
            "heldout_average_loss": average_loss,
        }

        return output
    else:
        print("Computing irreducible loss full training dataset.")
        idx_of_control_images = [1, 3, 10, 30, 100, 300, 1000, 3000]
        control_images = [0] * len(idx_of_control_images)
        losses = []
        idxs = []
        targets = []
        prediction_correct = []

        with torch.inference_mode():
            for idx, x, target in dataloader:
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, target, reduction="none")
                losses.append(loss)
                idxs.append(idx)
                targets.append(target)
                prediction_correct.append(torch.eq(torch.argmax(logits, dim=1), target))

                for (id, image) in zip(idx, x):
                    if id in idx_of_control_images:
                        local_index = idx_of_control_images.index(id)
                        control_images[local_index] = image

        acc = torch.cat(prediction_correct, dim=0)
        acc = acc.type(torch.FloatTensor).mean()
        average_loss = torch.cat(losses, dim=0).type(torch.FloatTensor).mean()

        log.info(
            f"Accuracy of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {acc:.3f}\n"
            f"Average loss of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {average_loss:.3f}"
        )

        losses_temp = torch.cat(losses, dim=0)
        idxs = torch.cat(idxs, dim=0)
        targets_temp = torch.cat(targets, dim=0)

        max_idx = idxs.max()

        losses = torch.tensor(
            [float("nan")] * (max_idx + 1), dtype=losses_temp.dtype
        )  # losses[global_index] is the irreducible loss of the datapoint with that global index. losses[idx] is nan if idx is not part of the dataset.
        targets = torch.zeros(max_idx + 1, dtype=targets_temp.dtype)
        losses[idxs] = losses_temp
        targets[idxs] = targets_temp

        output = {
            "irreducible_losses": losses,
            "sorted_targets": targets,
            "idx_of_control_images": idx_of_control_images,
            "control_images": control_images,
            "heldout_accuracy": acc,
            "heldout_average_loss": average_loss,
        }

        return output

def verify_correct_dataset_order(
    dataloader,
    sorted_target,
    idx_of_control_images,
    control_images,
    dont_compare_control_images=False,
):
    """Roughly checks that a dataloader is sorted in the same order as the
    precomputed losses. Concretely, does two checks: 1) that the labels used for
    computing the irreducible losses are in the same order as those returned by
    the dataloader. 2) That a handful of example images is identical across the
    ones used for computing the irreducble loss and the ones returned by the
    dataloader.

    Args:
        dataloader: a PyTorch dataloader, usually the training dataloader in our
        current setting.
        sorted_target, idx_of_control_images, control_images: those were saved
        as controls when pre-computing the irreducible loss.
        dont_compare_control_images: bool. Set to True if you don't want to compare
        control images (required if there is trainset augmentation)
    """
    print(
        "Verifying that the dataset order is compatible with the order of the precomputed losses."
    )
    if isinstance(dataloader.dataset, Clothing1M):
        for idx in np.random.choice(range(len(dataloader.dataset)), 10000, replace=False):
            _, _, target = dataloader.dataset[idx]
            assert torch.equal(target, sorted_target[idx]), "Unequal Images. Order of dataloader is not consistent with order used when precomputing irreducible losses. Can't use precomputed losses. Either ask Jan, or use the irreducible loss model directly ('irreducible_loss-generator: irreducible_loss_model.yaml' in the config.). Note that the latter is probably slower."
            
        if not dont_compare_control_images:
            for i, idx in enumerate(idx_of_control_images):
                _, image, _ = dataloader.dataset[idx]
                assert torch.equal(image, control_images[i]), "Unequal Images. Order of dataloader is not consistent with order used when precomputing irreducible losses. Can't use precomputed losses. Either ask Jan, or use the irreducible loss model directly ('irreducible_loss-generator: irreducible_loss_model.yaml' in the config.). Note that the latter is probably slower."
    else:
        for idx, x, target in dataloader:
            assert torch.equal(
                target, sorted_target[idx]
            ), "Unequal labels. Order of dataloader is not consistent with order used when precomputing irreducible losses. Can't use precomputed losses. Either ask Jan, or use the irreducible loss model directly ('irreducible_loss-generator: irreducible_loss_model.yaml' in the config.). Note that the latter is probably slower."
            if not dont_compare_control_images:
                for id, image in zip(idx, x):
                    if id in idx_of_control_images:
                        assert torch.equal(
                            image, control_images[idx_of_control_images.index(id)]
                        ), "Unequal Images. Order of dataloader is not consistent with order used when precomputing irreducible losses. Can't use precomputed losses. Either ask Jan, or use the irreducible loss model directly ('irreducible_loss-generator: irreducible_loss_model.yaml' in the config.). Note that the latter is probably slower."






def save_repo_status(path):
    """Save current commit hash and uncommitted changes to output dir."""

    with open(os.path.join(path, "git_commit.txt"), "w+") as f:
        subprocess.run(["git", "rev-parse", "HEAD"], stdout=f)

    with open(os.path.join(path, "git_commit.txt"), "r") as f:
        commit = f.readline()

    with open(os.path.join(path, "workspace_changes.diff"), "w+") as f:
        subprocess.run(["git", "diff"], stdout=f)
    
    return commit
