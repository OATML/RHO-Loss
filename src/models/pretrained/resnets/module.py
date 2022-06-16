import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy

# from cifar10_models.densenet import densenet121, densenet161, densenet169
# from cifar10_models.googlenet import googlenet
# from cifar10_models.inception import inception_v3
# from cifar10_models.mobilenetv2 import mobilenet_v2
from .resnet_from_internet import resnet18, resnet34, resnet50
# from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .schduler import WarmupCosineLR

all_classifiers = {
    # "vgg11_bn": vgg11_bn(),
    # "vgg13_bn": vgg13_bn(),
    # "vgg16_bn": vgg16_bn(),
    # "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    # "densenet121": densenet121(),
    # "densenet161": densenet161(),
    # "densenet169": densenet169(),
    # "mobilenet_v2": mobilenet_v2(),
    # "googlenet": googlenet(),
    # "inception_v3": inception_v3(),
}


class CIFAR10Module(pl.LightningModule):
    def __init__(self, classifier, learning_rate, weight_decay, max_epochs, pretrained):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers[self.hparams.classifier]
        self.model = self.model(pretrained)

    def forward(self, images):
        predictions = self.model(images)
        return predictions

    def training_step(self, batch, batch_nb):
        global_idx, images, labels = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        global_idx, images, labels = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        global_idx, images, labels = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
