"""
Trains SvP Model using the training set (from the datamodule) as the train set. It then uses a selection method to obtain a coreset sequence and then trains the model on the core-set
"""
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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.utils import utils
from src.models.OneModel import OneModel
from src.models.SVPModels import ForgettingEventsModel, OneModel_SVP
from src.utils.svp_utils import get_coreset

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

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    callbacks.append(EarlyStopping(monitor="val_loss_epoch"))
    # Init lightning model
    if config.model.pretrained_proxy:
        log.info(
            f"Instantiating Pre-Trained SVP Proxy Model <{config.model.proxy_model._target_}>"
        )
        proxy_model = hydra.utils.instantiate(config.model.proxy_model)
    else:
        log.info(f"Instantiating SVP Proxy Model <{config.model.proxy_model._target_}>")
        proxy_model_nn: nn.Module = hydra.utils.instantiate(config.model.proxy_model)
        if config.selection_method == "forgetting":
            proxy_model = ForgettingEventsModel(
                proxy_model_nn,
                datamodule,
                optimizer_config=config.get("optimizer", None),
                scheduler_config=config.get("scheduler", None),  # see line above
            )
        else:
            proxy_model = OneModel_SVP(
                proxy_model_nn,
                optimizer_config=config.get("optimizer", None),
                scheduler_config=config.get("scheduler", None),
            )

        # Init lightning loggers
        logger: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(
                        hydra.utils.instantiate(
                            lg_conf, prefix="proxy", _convert_="partial"
                        )
                    )

        # Init lightning trainer
        log.info(f"Instantiating trainer for Proxy <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=logger,
            max_epochs=config.proxy_max_epochs,
            _convert_="partial",
        )
        # Send config to all lightning loggers
        log.info("Logging hyperparameters!")
        trainer.logger.log_hyperparams(config)
        # Train the model
        # Train the model
        if config.eval_set == "val":
            val_dataloader = datamodule.val_dataloader()
        elif config.eval_set == "test":
            val_dataloader = datamodule.test_dataloader()
            log.warning(
                "Using the test set as the validation dataloader. This is for final figures in the paper"
            )

        log.info("Starting SVP Proxy training!")
        trainer.fit(
            proxy_model,
            train_dataloader=datamodule.train_dataloader(),
            val_dataloaders=val_dataloader,
        )
        # Evaluate model on test set, using the best model achieved during training
        if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            trainer.test(test_dataloaders=datamodule.test_dataloader())

    log.info(f"Selecting Core-Set>")
    sequence = get_coreset(
        config.selection_method,
        proxy_model,
        datamodule.train_dataloader(),
        config.percent_train,
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, sequence=sequence, _convert_="partial"
    )
    datamodule.setup()

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer for Proxy <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    # Send config to all lightning loggers
    log.info("Logging hyperparameters!")
    trainer.logger.log_hyperparams(config)

    # Init lightning model
    log.info(f"Instantiating SVP Large Model <{config.model.large_model._target_}>")
    large_model_nn: nn.Module = hydra.utils.instantiate(config.model.large_model)
    large_model = OneModel_SVP(
        large_model_nn,
        optimizer_config=config.get("optimizer", None),
        scheduler_config=config.get("scheduler", None),
    )

    log.info("Starting SVP Large Model training!")
    trainer.fit(
        large_model,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloaders=datamodule.test_dataloader(),
    )

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=large_model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
