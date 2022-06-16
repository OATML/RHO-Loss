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

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    pl_model: LightningModule = hydra.utils.instantiate(
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

    # Train the model
    log.info("Starting training!")
    trainer.fit(
        pl_model,
        train_dataloaders=datamodule.val_dataloader(),
        val_dataloaders=datamodule.test_dataloader(),
    )

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(test_dataloaders=datamodule.test_dataloader())

    ## Use this if you want to compute the irred losses for a model you have already trained
    # trainer.checkpoint_callback.best_model_path = "/path/to/model"

    def evaluate_and_save_model_from_checkpoint_path(checkpoint_path, name):
        # Compute irreducible loss for the whole trainset with the best model
        model = OneModel.load_from_checkpoint(checkpoint_path)
        model.eval()  # not sure this is actually needed when using model(x), but hopefully won't hurt either.
        irreducible_loss_and_checks = utils.compute_losses_with_sanity_checks(
            dataloader=datamodule.train_dataloader(), model=model
        )

        path = os.path.join(
            os.path.dirname(trainer.checkpoint_callback.best_model_path),
            name,
        )  # save irred losses in same directory as model checkpoint
        torch.save(irreducible_loss_and_checks, path)

        return path

    saved_path = evaluate_and_save_model_from_checkpoint_path(
        trainer.checkpoint_callback.best_model_path, "irred_losses_and_checks.pt"
    )

    log.info(f"Using monitor: {trainer.checkpoint_callback.monitor}")

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    log.info(f"Best checkpoint irred_losses_path:\n{saved_path}")

    # i.e., if we have checkpointed every epoch, for this model, assume that we also want to save the degraded models
    # I suppose sometimes this won't be true, but this seemed an easy way of doing this.
    if trainer.checkpoint_callback.save_top_k == -1:
        log.info("Checkpoint setup to save every model; saving degraded models")
        path_scores_dict = trainer.checkpoint_callback.best_k_models
        paths, scores = zip(*[(p, s.cpu()) for p, s in path_scores_dict.items()])

        scores_vector = np.array(scores)
        interp_scores = list(np.linspace(np.min(scores), np.max(scores), 5))

        delete_checkpoint = [True] * len(scores)

        for i, interp_score in enumerate(interp_scores):
            nearest_score = int(np.argmin(np.abs(scores_vector - interp_score)))
            path = paths[nearest_score]
            log.info(
                f"Degraded model {i+1} has val score [{trainer.checkpoint_callback.monitor}]: {scores_vector[nearest_score]}"
            )

            saved_path = evaluate_and_save_model_from_checkpoint_path(
                path, f"irred_losses_and_checks_degraded_{i+1}.pt"
            )
            log.info(f"Degraded model {i+1} irred_losses_path:\n{saved_path}")
            delete_checkpoint[nearest_score] = False

        [os.remove(p) for p, dc in zip(paths, delete_checkpoint) if dc]

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=pl_model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
