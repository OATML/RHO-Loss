import os, sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.pretrained.resnets.module import CIFAR10Module

# from src.models.pretrained.resnets.resnet import resnet18
from src.models.pretrained.resnets.resnet import resnet18
from src.models.pretrained.resnets.module import CIFAR10Module
from src.datamodules.datamodules import (
    CIFAR10DataModule,
    CINIC10DataModule,
    QMNISTDataModule,
)
from src.models.OneModel import OneModel
from src.utils import utils


def main():
    datamodule = CIFAR10DataModule(
        data_dir="/Users/jbrauner/repos/goldiprox-hydra/data",
        batch_size=320,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    # # datamodule = QMNISTDataModule(
    # #     data_dir="/Users/jbrauner/repos/goldiprox-hydra/data",
    # #     batch_size=320,
    # #     num_workers=4,
    # #     pin_memory=True,
    # #     shuffle=False,
    # # )

    datamodule.setup()

    # Using one of our models
    # best_model = OneModel.load_from_checkpoint(
    #     "/Users/jbrauner/repos/goldiprox-hydra/outputs/2021-09-17/18-55-08/checkpoints/epoch_000.ckpt"
    # )
    # best_model = OneModel.load_from_checkpoint(
    # "/Users/jbrauner/repos/goldiprox-hydra/outputs/2021-09-09/15-25-00/checkpoints/epoch_009.ckpt"
    # )

    # Pretrained model
    hparams = {
        "classifier": "resnet18",
        "max_epochs": 100,
        "learning_rate": 1e-2,
        "weight_decay": 1e-2,
        "pretrained": True,
    }
    best_model = CIFAR10Module(**hparams)

    best_model.eval()  # not sure this is actually needed when using model(x), but hopefully won't hurt either.

    irreducible_loss_and_checks = utils.compute_losses_with_sanity_checks(
        dataloader=datamodule.train_dataloader(), model=best_model
    )

    save_dir = "/Users/jbrauner/repos/goldiprox-hydra/outputs/pretrained_Resnet_test"  # make sure this directory actually exists :-)
    path = os.path.join(
        save_dir, "irred_losses_and_checks.pt"
    )  # save irred losses in same directory as model checkpoint
    torch.save(irreducible_loss_and_checks, path)


if __name__ == "__main__":
    main()
