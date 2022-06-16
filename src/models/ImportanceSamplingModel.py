import hydra
import numpy as np

# lightning related imports
import pytorch_lightning as pl

import hydra

# pytorch related imports
import torch
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F

from src.datamodules.datamodules import CINIC10RelevanceDataModule
from src.utils.utils import unmask_config


class ImportanceSamplingModel(pl.LightningModule):
    """
    Standard Importance Sampling Baseline implement, excluding threshold that decides whether to use importance sampling
    or not (tau threshold).

    Currently only technique implemented is the "upper bound" i.e., last layer gradient norm (preactivation), used in
    https://arxiv.org/abs/1803.00942.

    Version currently has:
    * limited logging; only accuracy, not selected point identities or sequence
    * no ability to use a proxy method
    * implemented directly in this function, rather than as a selection method that could be used elsewhere
    -- these things could be improved, certaintly, but I wasn't sure if it was worth it at this point.
    """

    def __init__(
        self,
        model,
        optimizer_config,
        scheduler_config,
        gradient_norm_calc="upper_bound",
        learning_rate=1e-3,
        percent_train=0.1,
        datamodule=None,
    ):
        super().__init__()

        # turn off PL's automatic optimisation so we can optimise per GoldiProx algorithm
        self.automatic_optimization = False

        # log hyperparameters
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss(
            reduction="none"
        )  # return per example losses here.

        # Send nn.Modules to appropriate CUDA/XLA device
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.datamodule = datamodule

    def forward(self, x):
        x = self.model(x)  # x are the logits here.
        return x

    def training_step(self, batch, batch_idx):
        optimiser = self.optimizers()
        optimiser.zero_grad()

        global_index, data, target = batch
        batch_size = len(data)
        selected_batch_size = int(batch_size * self.hparams.percent_train)

        logits = self.model(data)
        _, num_classes = logits.shape

        # logits are effectively the last layer preactivations; they are used for the gradient upper bound penalty
        if self.hparams.gradient_norm_calc == "upper_bound":
            # loss is a batch_size tensor; we need to the gradient of the loss with respect to each of the logits
            # logits of one input doesn't affect loss of the other—we can just sum the loss, and use autograd wrt logits
            # retrain the graph because we will need it later when we do the manual backward step

            # computation using autograd: not necessary, this gradient is available in closed form.
            # g_i = torch.autograd.grad(loss.sum(), logits, retain_graph=True)[0]
            # g_i_norm = torch.norm(g_i, dim=-1).numpy()
            # Note: I have verified using autograd is the same as the below, analytical code
            with torch.inference_mode():
                probs = F.softmax(logits, dim=1)
                one_hot_targets = F.one_hot(target, num_classes=num_classes)

                # Note: I think it's probably inefficient to detach, pass to cpu, sample on the CPU as opposed to doing all
                # operations on the GPU, but I'm not sure about this, so left it as it is.
                g_i_norm = (
                    torch.norm(probs - one_hot_targets, dim=-1).detach().cpu().numpy()
                )  # dim=-1 is the last dimension i.e., norm across class labels for each example

        else:
            raise NotImplementedError

        p_i = g_i_norm / np.sum(g_i_norm)

        batch_indices = np.random.choice(
            np.arange(batch_size), size=selected_batch_size, replace=True, p=p_i
        )

        selected_p_i = p_i[batch_indices]

        loss = self.loss(logits, target)
        selected_loss = loss[batch_indices]

        w_i = 1.0 / (batch_size * selected_p_i)

        # detach weight gradients, because the weights would be affected by the parameters, just to make
        # sure we avoid this, I added stop gradient
        weighted_loss = (
            torch.tensor(w_i).to(selected_loss.device).detach() * selected_loss
        ).mean()

        self.manual_backward(weighted_loss)
        optimiser.step()

        pc_corrupted = self.datamodule.percentage_corrupted(global_index[batch_indices])
        if pc_corrupted:  # returns None if no corruption was applied
            self.log(
                "percentage_selected_corrupted",
                pc_corrupted,
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("train_loss", loss.mean(), on_step=True, on_epoch=True, logger=True)
        self.log(
            "loss_used_for_backward",
            weighted_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("max_p_i", np.max(p_i), on_step=True, on_epoch=True, logger=True)
        self.log(
            "num_unique_points",
            np.unique(batch_indices).size,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        if isinstance(self.datamodule, CINIC10RelevanceDataModule):
            self.log(
                "percentage_relevant",
                self.datamodule.percentage_targets_relevant(target),
                on_step=True,
                on_epoch=True,
                logger=True,
            )

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        _, data, target = batch

        # validation metrics
        logits = self.model(data)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, target)
        loss = self.loss(logits, target)
        self.log(
            "val_loss_epoch",
            loss.mean(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "val_acc_epoch", acc, on_epoch=True, logger=True, prog_bar=True
        )

        return loss.mean()

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        _, data, target = batch
        logits = self.model(data)
        loss = self.loss(logits, target).mean()

        # validation metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss.mean()

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
