import pdb
from typing import List, Optional

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

    # init irreducible loss generator (precomputed losses, or irreducible loss
    # model)
    irreducible_loss_generator = hydra.utils.instantiate(
        config.irreducible_loss_generator
    )

    # If precomputed losses are used, verify that the sorting
    # of the precomputes losses matches the dataset
    if type(irreducible_loss_generator) is dict:
        # instantiate a separate datamodule, so that the main datamodule is
        # instantiated with the same random seed whether or not the precomputed
        # losses are used
        datamodule_temp = hydra.utils.instantiate(config.datamodule)
        datamodule_temp.setup()
        utils.verify_correct_dataset_order(
            dataloader=datamodule_temp.train_dataloader(),
            sorted_target=irreducible_loss_generator["sorted_targets"],
            idx_of_control_images=irreducible_loss_generator["idx_of_control_images"],
            control_images=irreducible_loss_generator["control_images"],
            dont_compare_control_images=config.datamodule.get(
                "trainset_data_aug", False
            ),  # cannot compare images from irreducible loss model training run with those of the current run if there is trainset augmentation
        )

        del datamodule_temp

        irreducible_loss_generator = irreducible_loss_generator["irreducible_losses"]

        # Set seed again, so that the main datamodule is instantiated with the
        # same random seed whether or not the precomputed losses are used
        if "seed" in config:
            seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    # init selection method
    log.info(f"Instantiating selection method <{config.selection_method._target_}>")
    selection_method = hydra.utils.instantiate(config.selection_method)

    # Init lightning model
    log.info(f"Instantiating models")
    pl_model: LightningModule = hydra.utils.instantiate(
        config.model,
        selection_method=selection_method,
        irreducible_loss_generator=irreducible_loss_generator,
        datamodule=datamodule,
        optimizer_config=utils.mask_config(
            config.get("optimizer", None)
        ),  # When initialising the optimiser, you need to pass it the model parameters. As we haven't initialised the model yet, we cannot initialise the optimizer here. Thus, we need to pass-through the optimizer-config, to initialise it later. However, hydra.utils.instantiate will instatiate everything that looks like a config (if _recursive_==True, which is required here bc OneModel expects a model argument). Thus, we "mask" the optimizer config from hydra, by modifying the dict so that hydra no longer recognises it as a config.
        _convert_="partial",
    )

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send config to all lightning loggers
    log.info("Logging hyperparameters!")
    trainer.logger.log_hyperparams(config)

    if config.eval_set == "val":
        val_dataloader = datamodule.val_dataloader()
    elif config.eval_set == "test":
        val_dataloader = datamodule.test_dataloader()
        log.warning(
            "Using the test set as the validation dataloader. This is for final figures in the paper"
        )
    
    trainer.validate(model=pl_model, dataloaders=val_dataloader)
#     # Train the model
#     log.info("Starting training!")
#     trainer.fit(
#         pl_model,
#         train_dataloaders=datamodule.train_dataloader(),
#         val_dataloaders=val_dataloader,
#     )

#     # Evaluate model on test set, using the best model achieved during training
#     if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
#         log.info("Starting testing!")
#         trainer.test(test_dataloaders=datamodule.test_dataloader())

#     # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=pl_model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

#     # Print path to best checkpoint
#     log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

#     # Return metric score for hyperparameter optimization
#     optimized_metric = config.get("optimized_metric")
#     if optimized_metric:
#         return trainer.callback_metrics[optimized_metric]
