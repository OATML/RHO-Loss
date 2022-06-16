# regular imports
import math
import numbers

import numpy as np
import hydra

# lightning related imports
import pytorch_lightning as pl

# pytorch related imports
import torch
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F

from src.datamodules.datamodules import CINIC10RelevanceDataModule
from src.utils.utils import unmask_config


class OneModel(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer_config,
        scheduler_config,
        datamodule=None,
        percent_train=1.0,
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.datamodule = datamodule

    def forward(self, x):
        x = self.model(x)
        return x

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        _, data, target = batch

        batch_size = len(data)

        if self.hparams.percent_train < 1:
            selected_batch_size = int(batch_size * self.hparams.percent_train)
            selected_minibatch = torch.randperm(len(data))[:selected_batch_size]
            data = data[selected_minibatch]
            target = target[selected_minibatch]

        logits = self.model(data)
        loss = self.loss(logits, target)

        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        if isinstance(self.datamodule, CINIC10RelevanceDataModule):
            self.log(
                "percentage_relevant",
                self.datamodule.percentage_targets_relevant(target),
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        _, data, target = batch
        logits = self.model(data)
        loss = self.loss(logits, target)

        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("val_loss_epoch", loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_acc_epoch", acc, on_epoch=True, logger=True, sync_dist=True)

        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        _, data, target = batch
        logits = self.model(data)
        loss = self.loss(logits, target)

        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("test_loss_epoch", loss, on_epoch=True, logger=True)
        self.log("test_acc_epoch", acc, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            config=unmask_config(self.optimizer_config),
            params=self.model.parameters(),
            _convert_="partial",
        )

        if self.scheduler_config is None:
            return [optimizer]
        else:
            scheduler = hydra.utils.instantiate(
                unmask_config(self.scheduler_config),
                optimizer=optimizer,
                _convert_="partial",
            )
            return [optimizer], [scheduler]
