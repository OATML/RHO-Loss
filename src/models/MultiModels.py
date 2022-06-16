# regular imports
import math
import numbers

import numpy as np

# lightning related imports
import pytorch_lightning as pl

# pytorch related imports
import torch
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
import numpy as np

from src.curricula.utils_bald import enable_dropout
from src.models.OneModel import OneModel
from src.utils.utils import unmask_config
import hydra

from src.datamodules.datamodules import CINIC10RelevanceDataModule, Clothing1MDataModule
from src.curricula.selection_methods import reducible_loss_selection, irreducible_loss_selection, gradnorm_ub_selection, ce_loss_selection, uniform_selection
import pdb

class MultiModels(pl.LightningModule):
    def __init__(
        self,
        large_model,
        irreducible_loss_generator=None,
        proxy_model=None,
        selection_method=None,
        optimizer_config=None,
        percent_train=0.1,
        update_irreducible=False,
        detailed_logging=False,
        num_mc=10,  # number of MC samples if using BALD
        datamodule=None,
        repetition_logging=False,
        parallel_implementation=False,
        parallel_skip=False,
        selection_train_mode=True,
        track_all_selection=False,
    ):
        """
        PyTorch Lightning Module for GoldiProx.
        Args:
            large_model: nn.Module, large model in goldiprox setting
            irreducible_loss_generator: Tensor or nn.Module
             Tensor: with irreducible losses for train set, ordered by <index> (see datamodules)
             nn.Module: irreducible loss model
            proxy_model: nn.Module, a model that acts as proxy for large_model
            selection_method: callable class, selection method. current available options include: reducible loss, irreducible loss, ce_loss, uniform, bald
            learning_rate: float, learning rate for all models
            percent_train: float [0-1], the percent of each batch to train on
            update_irreducible: bool, update irreducible loss model with respect to training data
            detailed_logging: bool, detailed loggin
            repetition_logging: bool, enables loging how many of the points are repeated
            selection_train_mode: bool; whether the selection is done with the
                model in .train mode or .eval mode. This influences batch norm
                behaviour and dropout layers. Defaults to True.
        """
        super().__init__()

        # turn off PL's automatic optimisation so we can optimise per GoldiProx algorithm
        self.automatic_optimization = False

        # log and save hyperparameters
        self.save_hyperparameters()
        # saved to self.hparams
        # self.detailed_logging = detailed_logging
        # self.learning_rate = learning_rate
        # self.percent_train = percent_train
        # self.update_irreducible = update_irreducible

        # should be defined or instantiated by hydra
        self.selection_method = selection_method
        self.large_model = large_model
        self.proxy_model = proxy_model
        self.irreducible_loss_generator = irreducible_loss_generator
        self.optimizer_config = optimizer_config
        self.datamodule = datamodule
        # if self.irreducible_loss_model_path is not None:
        #     self.irreducible_loss_model =

        # loss function
        self.loss = nn.CrossEntropyLoss(reduction="none")

        # For recording sequence used in training
        self.sequence = np.asarray([])

        self.detailed_log = []
        self.repetition_logging = repetition_logging

        # store stale gradients here. Only used if parallel_implementation is true
        self.saved_batch = None
        self.current_batch = None

        if track_all_selection:
            self.all_selection_methods = [
                reducible_loss_selection(),
                gradnorm_ub_selection(),
                ce_loss_selection(),
                irreducible_loss_selection(),
                uniform_selection()
                ]
            self.all_selection_method_names = [
                "redloss",
                "gradnorm",
                "ce_loss",
                "irred_loss",
                "uniform"
                ]

    def forward(self, x):
        x = self.large_model(x)
        return x

    def training_step(self, batch, batch_idx):
        global_index, data, target = batch
        batch_size = len(data)
        selected_batch_size = max(1, int(batch_size * self.hparams.percent_train))

        if self.hparams.selection_train_mode:
            self.large_model.train()
        else:
            self.large_model.eval() # switch to eval mode to compute selection
        ### Selection Methods
        selected_indices, metrics_to_log, irreducible_loss = self.selection_method(
            selected_batch_size=selected_batch_size,
            data=data,
            target=target,
            global_index=global_index,
            large_model=self.large_model,
            irreducible_loss_generator=self.irreducible_loss_generator,
            proxy_model=self.proxy_model,
            current_epoch=self.current_epoch,  # not used by all methods, but needed for annealing
            num_classes=self.datamodule.num_classes
        )  # irreducible_loss will be None if the selection_method does not involve
        # irreducible_loss computation (e.g. uniform, CE loss selection)
        self.large_model.train()  # switch to eval mode to compute selection

        pc_corrupted = self.datamodule.percentage_corrupted(
            global_index[selected_indices]
        )
        if pc_corrupted:  # returns None if no corruption was applied
            self.log(
                "selected_percentage_corrupted",
                pc_corrupted,
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        if isinstance(self.datamodule, Clothing1MDataModule) and self.hparams.track_all_selection:
            for name, selection_method in zip(self.all_selection_method_names, self.all_selection_methods):
                selected_indices_method, _, _ = selection_method(
                    selected_batch_size=selected_batch_size,
                    data=data,
                    target=target,
                    global_index=global_index,
                    large_model=self.large_model,
                    irreducible_loss_generator=self.irreducible_loss_generator,
                    proxy_model=self.proxy_model,
                    current_epoch=self.current_epoch,  # not used by all methods, but needed for annealing
                    num_classes=self.datamodule.num_classes
                    )
                self.log(
                    "percentage_clean_"+name,
                    self.datamodule.percentage_clean(global_index[selected_indices_method]),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )



        # build sequence
        self.sequence = np.append(
            self.sequence, global_index[selected_indices].cpu().numpy()
        )
        data, target = data[selected_indices], target[selected_indices]

        if self.hparams.parallel_implementation:
            self.current_batch = (data, target)

            # save the current selected batch
            if self.saved_batch is None:
                self.saved_batch = (
                    self.current_batch
                )  # for the first step, use the selected points from the first batch. reused in the next step

                if self.hparams.parallel_skip:
                    return  # skip step

            data, target = self.saved_batch  # load the stale batch
            self.saved_batch = self.current_batch  # save the current batch

        # repetition logging made optional because it requires no shuffling (and
        # also currently fails with CIFAR)
        if self.repetition_logging:
            self.log_repetition(data)

        if self.proxy_model is not None:
            opt_large_model, opt_proxy_model = self.optimizers()

            opt_proxy_model.zero_grad()
            logits = self.proxy_model(data)

            proxy_model_loss = self.loss(logits, target)
            self.manual_backward(proxy_model_loss.mean())
            opt_proxy_model.step()

            # logging
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            proxy_model_acc = accuracy(preds, target)
            self.log(
                "proxy_train_loss",
                proxy_model_loss.mean(),
                on_step=True,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "proxy_train_acc",
                proxy_model_acc,
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        # Note from Jan about the following if statement: the irreducible losses
        # are already computed in the selection_method. I did not change
        # anything in this if-statement, though, because I don't know if we even
        # use it.
        elif self.hparams.update_irreducible:
            opt_large_model, opt_irreducible_model = self.optimizers()

            opt_irreducible_model.zero_grad()
            logits = self.irreducible_loss_generator(data)
            irreducible_loss = self.loss(logits, target)
            self.manual_backward(irreducible_loss.mean())
            opt_irreducible_model.step()

            # logging
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            irreducible_acc = accuracy(preds, target)
            self.log(
                "IrLoMo_train_loss",
                irreducible_loss.mean(),
                on_step=True,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "IrLoMo_train_acc",
                irreducible_acc,
                on_step=True,
                on_epoch=True,
                logger=True,
            )


        else:
            opt_large_model = self.optimizers()

        opt_large_model.zero_grad()
        logits = self.large_model(data)
        loss = self.loss(logits, target)
        self.manual_backward(loss.mean())
        opt_large_model.step()

        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("train_loss", loss.mean(), on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        if isinstance(self.datamodule, CINIC10RelevanceDataModule):
            self.log(
                "percentage_relevant",
                self.datamodule.percentage_targets_relevant(target),
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        detailed_only_keys = metrics_to_log["detailed_only_keys"]
        metrics_to_log[
            "step"
        ] = (
            self.global_step
        )  # add step to the logging, also might help us concretely cross-corelate exact point in time.

        # batch statistics summary logging, depending on the metric that we ended up using.
        for key, value in metrics_to_log.items():
            if key in detailed_only_keys:
                continue

            if isinstance(value, np.ndarray):
                for percentile in [2.5, 25, 50, 75, 97.5]:
                    v = np.percentile(value, percentile)
                    self.log(
                        f"{key}_{percentile}",
                        v,
                        on_step=True,
                        on_epoch=True,
                        logger=True,
                    )

            elif isinstance(value, numbers.Number):
                self.log(key, value, on_step=True, on_epoch=True, logger=True)

        # unclear to me quite how inefficient this will be. We can use the lightning profiler :~)
        if self.hparams.detailed_logging:
            self.detailed_log.append(metrics_to_log)

        if self.proxy_model is not None:

            # track correlation and covariance between proxy_model and big model
            spearman = self.spearman_correlation(proxy_model_loss, loss)
            self.log(
                "spearman_proxy_loss_large_loss",
                spearman,
                on_step=True,
                on_epoch=True,
                logger=True,
            )
            cov = np.cov(
                proxy_model_loss.detach().cpu().numpy(), loss.detach().cpu().numpy()
            )[0, 1]
            self.log(
                "cov_proxy_loss_large_loss",
                cov,
                on_step=True,
                on_epoch=True,
                logger=True,
            )

            # track standard deviations
            std = torch.std(proxy_model_loss)
            self.log("std_proxy_loss", std, on_step=True, on_epoch=True, logger=True)

            std = torch.std(loss)
            self.log("std_large_loss", std, on_step=True, on_epoch=True, logger=True)

            if irreducible_loss is not None:

                spearman = self.spearman_correlation(
                    proxy_model_loss - irreducible_loss, loss - irreducible_loss
                )
                self.log(
                    "spearman_proxy_redloss_large_redloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                # Correlations between the same metrics on different models

                spearman = self.spearman_correlation(
                    proxy_model_loss - irreducible_loss, loss - irreducible_loss
                )
                self.log(
                    "spearman_proxy_redloss_large_redloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                cov = np.cov(
                    proxy_model_loss.detach().cpu().numpy()
                    - irreducible_loss.detach().cpu().numpy(),
                    loss.detach().cpu().numpy()
                    - irreducible_loss.detach().cpu().numpy(),
                )[0, 1]
                self.log(
                    "cov_proxy_redloss_large_redloss",
                    cov,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                )

                # Correlations between different metrics

                spearman = self.spearman_correlation(
                    proxy_model_loss - irreducible_loss, loss
                )
                self.log(
                    "spearman_proxy_redloss_large_loss",
                    spearman,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                )

                spearman = self.spearman_correlation(proxy_model_loss, irreducible_loss)
                self.log(
                    "spearman_proxy_loss_irrloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                spearman = self.spearman_correlation(
                    proxy_model_loss - irreducible_loss, irreducible_loss
                )
                self.log(
                    "spearman_proxy_redloss_irrloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                spearman = self.spearman_correlation(loss, irreducible_loss)
                self.log(
                    "spearman_large_loss_irrloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                spearman = self.spearman_correlation(
                    loss - irreducible_loss, irreducible_loss
                )
                self.log(
                    "spearman_large_redloss_irrloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                # Standard deviations
                std = torch.std(loss - irreducible_loss)
                self.log(
                    "std_large_redloss", std, on_step=True, on_epoch=False, logger=True
                )

                std = torch.std(proxy_model_loss - irreducible_loss)
                self.log(
                    "std_proxy_redloss", std, on_step=True, on_epoch=False, logger=True
                )

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        _, data, target = batch

        if self.selection_method.bald:
            self.large_model.eval()
            enable_dropout(self.large_model)
            predictions = torch.zeros(
                (self.hparams.num_mc, len(data), 10), device=self.device
            )
            for i in range(self.hparams.num_mc):
                predictions[i] = self.large_model(data)
            predictions = predictions.transpose(0, 1)
            logits = torch.logsumexp(predictions, dim=1) - math.log(self.hparams.num_mc)
            loss = self.loss(logits, target)

        else:
            logits = self.large_model(data)
            loss = self.loss(logits, target)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, target)
        self.log(
            "val_loss_epoch",
            loss.mean(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        if self.hparams.update_irreducible:

            # logging
            logits = self.irreducible_loss_generator(data)
            irlomo_loss = self.loss(logits, target)
            preds = torch.argmax(logits, dim=1)
            irlomo_acc = accuracy(preds, target)
            self.log(
                "irlomo_val_loss",
                irlomo_loss.mean(),
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
            self.log(
                "irlomo_val_acc",
                irlomo_acc,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
            
        if self.proxy_model is not None:
            logits = self.proxy_model(data)
            proxy_loss = self.loss(logits, target)
            preds = torch.argmax(logits, dim=1)
            proxy_acc = accuracy(preds, target)
            self.log(
                "proxy_val_loss_epoch",
                proxy_loss.mean(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
            self.log(
                "proxy_val_acc_epoch",
                proxy_acc,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        return loss.mean()

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        _, data, target = batch
        logits = self.large_model(data)
        loss = self.loss(logits, target).mean()

        # validation metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss.mean()

    def configure_optimizers(self):
        if self.proxy_model is not None:
            opt_large_model = hydra.utils.instantiate(
                config=unmask_config(self.optimizer_config),
                params=self.large_model.parameters(),
                _convert_="partial",
            )
            opt_proxy_model = hydra.utils.instantiate(
                config=unmask_config(self.optimizer_config),
                params=self.proxy_model.parameters(),
                _convert_="partial",
            )
            return [opt_large_model, opt_proxy_model]
        if self.hparams.update_irreducible:
            opt_large_model = hydra.utils.instantiate(
                config=unmask_config(self.optimizer_config),
                params=self.large_model.parameters(),
                _convert_="partial",
            )
            opt_irreducible_model = hydra.utils.instantiate(
                config=unmask_config(self.optimizer_config),
                params=self.irreducible_loss_generator.parameters(),
                _convert_="partial",
            )
            for param_group in opt_irreducible_model.param_groups:
                param_group['lr'] = param_group['lr']/100
            return [opt_large_model, opt_irreducible_model]
        else:
            optimizer = hydra.utils.instantiate(
                config=unmask_config(self.optimizer_config),
                params=self.large_model.parameters(),
                _convert_="partial",
            )
            return [optimizer]

    def _get_ranks(self, x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp, device=x.device)
        ranks[tmp] = torch.arange(len(x), device=x.device)
        return ranks

    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor):
        """Compute correlation between 2 1-D vectors
        Args:
            x: Shape (N, )
            y: Shape (N, )
        """
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        x_rank = self._get_ranks(x)
        y_rank = self._get_ranks(y)

        n = x.size(0)
        upper = 6 * torch.sum((x_rank - y_rank).pow(2))
        down = n * (n ** 2 - 1.0)
        return 1.0 - (upper / down)

    def log_repetition(self, data):
        """Measure repetition of selected points in previous epochs. Given current indices selected, logs what
        percentage of them were also selected exactly 1, 5, and 20 epochs ago. Requires shuffle=False."""
        assert self.datamodule.hparams.shuffle == False
        selected_batch_size = int(len(data))
        train_set_size = self.datamodule.indices_train.sequence.size
        epoch_size = int(train_set_size * self.hparams.percent_train)
        selected_indices_now = self.sequence[-selected_batch_size:]

        selected_indices_1_epoch_ago = self.sequence[
            -1 * epoch_size - selected_batch_size : -1 * epoch_size
        ]
        selected_indices_5_epoch_ago = self.sequence[
            -5 * epoch_size - selected_batch_size : -5 * epoch_size
        ]
        selected_indices_20_epoch_ago = self.sequence[
            -20 * epoch_size - selected_batch_size : -20 * epoch_size
        ]

        perct_repeated_1_epoch_ago = np.intersect1d(
            selected_indices_now, selected_indices_1_epoch_ago
        ).size / float(selected_batch_size)
        perct_repeated_5_epoch_ago = np.intersect1d(
            selected_indices_now, selected_indices_5_epoch_ago
        ).size / float(selected_batch_size)
        perct_repeated_20_epoch_ago = np.intersect1d(
            selected_indices_now, selected_indices_20_epoch_ago
        ).size / float(selected_batch_size)

        self.log(
            "perct_idx_repeated_from_1_epoch_ago",
            perct_repeated_1_epoch_ago,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "perct_idx_repeated_from_5_epoch_ago",
            perct_repeated_5_epoch_ago,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "perct_idx_repeated_from_20_epoch_ago",
            perct_repeated_20_epoch_ago,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
