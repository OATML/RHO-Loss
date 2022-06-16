import hydra
import numpy as np

# lightning related imports
import pytorch_lightning as pl

import hydra

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
        percent_train=0.1,
        datamodule=None,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        # eval_splits: Optional[list] = None,
        **kwargs,
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

    def forward(self, **inputs):
        x = self.model(**inputs)  
        return x[1] # these are the logits.

    def training_step(self, batch, batch_idx):
        optimiser = self.optimizers()
        optimiser.zero_grad()

        # global_index, data, target = batch
        global_index = batch.pop("idx") # pop the "global index" from the batch dict
        inputs = batch
        target = inputs["labels"]

        batch_size = len(target)
        selected_batch_size = int(batch_size * self.hparams.percent_train)

        with torch.inference_mode():

            logits = self.model(**inputs)[1]
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

        global_index = global_index[batch_indices]
        inputs = {k: v[batch_indices] for k,v in inputs.items()}
        target = target[batch_indices]
        logits = self.model(**inputs)[1]

        selected_loss = self.loss(logits, target)

        w_i = 1.0 / (batch_size * selected_p_i)

        # detach weight gradients, because the weights would be affected by the parameters, just to make
        # sure we avoid this, I added stop gradient
        weighted_loss = (
            torch.tensor(w_i).to(selected_loss.device).detach() * selected_loss
        ).mean()

        self.manual_backward(weighted_loss)
        optimiser.step()

        # pc_corrupted = self.datamodule.percentage_corrupted(global_index[batch_indices])
        # if pc_corrupted:  # returns None if no corruption was applied
        #     self.log(
        #         "percentage_selected_corrupted",
        #         pc_corrupted,
        #         on_step=True,
        #         on_epoch=True,
        #         logger=True,
        #     )

        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("train_loss", selected_loss.mean(), on_step=True, on_epoch=True, logger=True)
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

        # if isinstance(self.datamodule, CINIC10RelevanceDataModule):
        #     self.log(
        #         "percentage_relevant",
        #         self.datamodule.percentage_targets_relevant(target),
        #         on_step=True,
        #         on_epoch=True,
        #         logger=True,
        #     )

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        _ = batch.pop("idx")  # pop the "global index" from the batch dict
        inputs = batch
        target = inputs["labels"]

        # validation metrics
        logits = self.model(**inputs)[1]
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
        _ = batch.pop("idx")  # pop the "global index" from the batch dict
        inputs = batch
        target = inputs["labels"]

        logits = self.model(**inputs)[1]
        loss = self.loss(logits, target).mean()

        # validation metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss.mean()

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