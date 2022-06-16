# regular imports
import math
import numbers
import hydra

import numpy as np

# lightning related imports
import pytorch_lightning as pl

# pytorch related imports
import torch
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F

import pdb
from src.models.OneModel import OneModel


class OneModel_SVP(OneModel):
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            config=self.optimizer_config,
            params=self.model.parameters(),
            _convert_="partial",
        )

        if self.scheduler_config is None:
            return [optimizer]
        else:
            scheduler = hydra.utils.instantiate(
                self.scheduler_config, optimizer=optimizer, _convert_="partial"
            )
            return [optimizer], [scheduler]


class ForgettingEventsModel(OneModel_SVP):
    def __init__(self, model, datamodule, optimizer_config, scheduler_config):
        super().__init__(model, optimizer_config, scheduler_config)

        self.datamodule = datamodule
        self.dataset_size = len(self.datamodule.train_dataloader().dataset)

    def on_fit_start(self) -> None:
        self.correct = np.zeros(self.dataset_size, dtype=np.int64)
        self.n_forgotten = np.zeros(self.dataset_size, dtype=np.float32)
        self.was_correct = np.zeros(self.dataset_size, dtype=np.bool)

    def on_fit_end(self) -> None:
        self.n_forgotten[~self.was_correct] = np.inf

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        index, data, target = batch
        logits = self.model(data)
        loss = self.loss(logits, target)

        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        correct_batch = target.eq(preds).cpu().numpy().astype(np.int64)

        transitions = correct_batch - self.correct[index.cpu().numpy()]
        self.correct[index.cpu().numpy()] = correct_batch
        self.was_correct[index.cpu().numpy()] |= correct_batch.astype(np.bool)
        self.n_forgotten[index.cpu().numpy()[transitions == -1]] += 1.0

        acc = accuracy(preds, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        return loss
