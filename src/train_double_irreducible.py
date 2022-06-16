import copy
from typing import List, Optional
import os

import numpy as np

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
import torch
import torch.nn as nn

from src.utils import utils
from src.models.OneModel import OneModel

log = utils.get_logger(__name__)

import pdb


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup(double_irlomo=True)

    # create datamodule no train aug
    datamodule_notrainaug: LightningDataModule = hydra.utils.instantiate(config.datamodule, trainset_data_aug=False)
    datamodule_notrainaug.setup()

    # Init lightning model

    pl_model_factory = lambda: hydra.utils.instantiate(
        config=config.model,
        optimizer_config=utils.mask_config(
            config.get("optimizer", None)
        ),  # When initialising the optimiser, you need to pass it the model parameters. As we haven't initialised the model yet, we cannot initialise the optimizer here. Thus, we need to pass-through the optimizer-config, to initialise it later. However, hydra.utils.instantiate will instatiate everything that looks like a config (if _recursive_==True, which is required here bc OneModel expects a model argument). Thus, we "mask" the optimizer config from hydra, by modifying the dict so that hydra no longer recognises it as a config.
        scheduler_config=utils.mask_config(
            config.get("scheduler", None)
        ),  # see line above
        datamodule=datamodule,
        _convert_="partial",
    )

    log.info(f"Instantiating model <{config.model._target_}>")
    pl_model_1: LightningModule = pl_model_factory()
    pl_model_2: LightningModule = pl_model_factory()

    # Init lightning callbacks
    def callbacks_factory(filename):
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")

                    if (
                        cb_conf._target_
                        == "pytorch_lightning.callbacks.ModelCheckpoint"
                    ):
                        callbacks.append(
                            hydra.utils.instantiate(cb_conf, filename=filename)
                        )
                    else:
                        callbacks.append(hydra.utils.instantiate(cb_conf))

        return callbacks

    callbacks_1 = callbacks_factory("model_1_epoch_{epoch:03d}")
    callbacks_2 = callbacks_factory("model_2_epoch_{epoch:03d}")

    def logger_factory(suffix):
        # Init lightning loggers
        logger: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    if (
                        lg_conf._target_
                        == "pytorch_lightning.loggers.wandb.WandbLogger"
                        and lg_conf.name
                    ):
                        logger.append(
                            hydra.utils.instantiate(
                                lg_conf, name=f"{lg_conf.name}_{suffix}"
                            )
                        )
                    else:
                        logger.append(hydra.utils.instantiate(lg_conf))
        return logger

    logger_1 = logger_factory("model_1")
    logger_2 = logger_factory("model_2")

    trainer_factory = lambda callbacks, logger: hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer_1: Trainer = trainer_factory(callbacks_1, logger_1)
    trainer_2: Trainer = trainer_factory(callbacks_2, logger_2)

    # Send config to all lightning loggers
    log.info("Logging hyperparameters!")
    trainer_1.logger.log_hyperparams(config)
    trainer_2.logger.log_hyperparams(config)

    (
        train_split_1_dataloader,
        train_split_2_dataloader,
    ) = datamodule.train_split_dataloaders()
    train_split_1_indices = datamodule.indices_train_split_info["train_split_1_indices"]
    train_split_2_indices = datamodule.indices_train_split_info["train_split_2_indices"]

    # Train the model
    log.info("Starting training!")
    trainer_1.fit(
        pl_model_1,
        train_dataloader=train_split_1_dataloader,
        val_dataloaders=datamodule.val_dataloader(),
    )
    trainer_2.fit(
        pl_model_2,
        train_dataloader=train_split_2_dataloader,
        val_dataloaders=datamodule.val_dataloader(),
    )

    model_1 = OneModel.load_from_checkpoint(
        trainer_1.checkpoint_callback.best_model_path
    )
    model_1.eval()  # not sure this is actually needed when using model(x), but hopefully won't hurt either.
    irreducible_loss_and_checks_1 = utils.compute_losses_with_sanity_checks(
        dataloader=datamodule_notrainaug.train_dataloader(), model=model_1
    )

    model_2 = OneModel.load_from_checkpoint(
        trainer_2.checkpoint_callback.best_model_path
    )
    model_2.eval()  # not sure this is actually needed when using model(x), but hopefully won't hurt either.
    irreducible_loss_and_checks_2 = utils.compute_losses_with_sanity_checks(
        dataloader=datamodule_notrainaug.train_dataloader(), model=model_2
    )

    irreducible_loss_and_checks_merged = copy.deepcopy(irreducible_loss_and_checks_1)
    # average the heldout accuracy and loss for the merged model
    irreducible_loss_and_checks_merged["heldout_accuracy"] = (
        0.5 * irreducible_loss_and_checks_1["heldout_accuracy"]
        + 0.5 * irreducible_loss_and_checks_2["heldout_accuracy"]
    )
    irreducible_loss_and_checks_merged["heldout_average_loss"] = (
        0.5 * irreducible_loss_and_checks_1["heldout_average_loss"]
        + 0.5 * irreducible_loss_and_checks_2["heldout_average_loss"]
    )
    irreducible_loss_and_checks_merged["irreducible_losses"][
        train_split_1_indices
    ] = irreducible_loss_and_checks_2["irreducible_losses"][train_split_1_indices]
    irreducible_loss_and_checks_merged["irreducible_losses"][
        train_split_2_indices
    ] = irreducible_loss_and_checks_1["irreducible_losses"][train_split_2_indices]

    path = os.path.join(
        os.path.dirname(trainer_1.checkpoint_callback.best_model_path),
        "irred_losses_and_checks_double_irrlomo.pt",
    )  # save irred losses in same directory as model checkpoint
    torch.save(irreducible_loss_and_checks_merged, path)
    log.info(
        f"Average Heldout Acc:{irreducible_loss_and_checks_merged['heldout_accuracy']}\n"
        f"Average Heldout Loss:{irreducible_loss_and_checks_merged['heldout_average_loss']}"
    )
    log.info(f"Best checkpoint irred_losses_path:\n{path}")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=pl_model_1,
        datamodule=datamodule,
        trainer=trainer_1,
        callbacks=callbacks_1,
        logger=logger_2,
    )

    utils.finish(
        config=config,
        model=pl_model_2,
        datamodule=datamodule,
        trainer=trainer_2,
        callbacks=callbacks_2,
        logger=logger_2,
    )

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return (
            trainer_1.callback_metrics[optimized_metric]
            + trainer_2.callback_metrics[optimized_metric]
        )
