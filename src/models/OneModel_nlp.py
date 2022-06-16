# regular imports
import math
import numbers
from typing import Optional

import numpy as np
import hydra

# lightning related imports
import pytorch_lightning as pl

# pytorch related imports
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F


from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


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
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        # eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.datamodule = datamodule

    def forward(self, **inputs):
        return self.model(**inputs)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        # _, inputs = batch
        _ = batch.pop("idx") # pop the "global index" from the batch dict
        inputs = batch
        target = inputs["labels"]

        # batch_size = len(data)

        # if self.hparams.percent_train < 1:
        #     selected_batch_size = int(batch_size * self.hparams.percent_train)
        #     selected_minibatch = torch.randperm(len(data))[:selected_batch_size]
        #     data = data[selected_minibatch]
        #     target = target[selected_minibatch]

        logits = self.model(**inputs)[1]
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
        # _, inputs = batch
        _ = batch.pop("idx")  # pop the "global index" from the batch dict
        inputs = batch
        target = inputs["labels"]

        logits = self.model(**inputs)[1]
        loss = self.loss(logits, target)

        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("val_loss_epoch", loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_acc_epoch", acc, on_epoch=True, logger=True, sync_dist=True)

        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        # _, inputs = batch
        _ = batch.pop("idx")  # pop the "global index" from the batch dict
        inputs = batch
        target = inputs["labels"]

        logits = self.model(**inputs)[1]
        loss = self.loss(logits, target)

        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("test_loss_epoch", loss, on_epoch=True, logger=True)
        self.log("test_acc_epoch", acc, on_epoch=True, logger=True)

        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - val_dataloader() is called after setup() by default
        val_loader = self.datamodule.val_dataloader()

        # Calculate total stepsliner        
        tb_size = self.datamodule.eval_batch_size * max(1, self.trainer.gpus)
        ab_size = tb_size * self.trainer.accumulate_grad_batches
        self.total_steps = int((len(val_loader.dataset) / ab_size) * float(self.trainer.max_epochs))


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.total_steps,
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer]#, [scheduler]
    # def configure_optimizers(self):
    #     optimizer = hydra.utils.instantiate(
    #         config=unmask_config(self.optimizer_config),
    #         params=self.model.parameters(),
    #         _convert_="partial",
    #     )

    #     if self.scheduler_config is None:
    #         return [optimizer]
    #     elif self.scheduler_config.get("target")=="custom_hardcoded_warmup_cosine_lr": # this is super clumsy and dumb but I only need this once to train the irred loss model with the same hypers that I found on the internet and just want to be done with it 
    #         from src.models.pretrained.resnets.schduler import WarmupCosineLR
    #         total_steps = self.scheduler_config["max_epochs"] * self.scheduler_config["len_train_dataloader"]
    #         scheduler = {
    #             "scheduler": WarmupCosineLR(
    #                 optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
    #             ),
    #             "interval": "step",
    #             "name": "learning_rate",
    #         }
    #         return [optimizer], [scheduler]
    #     else:
    #         scheduler = hydra.utils.instantiate(
    #             unmask_config(self.scheduler_config),
    #             optimizer=optimizer,
    #             _convert_="partial",
    #         )
    #         return [optimizer], [scheduler]
