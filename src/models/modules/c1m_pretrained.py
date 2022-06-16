import torchvision
import torch
import torch.nn as nn

def resnet18_imagenet(pretrained=True, classes=14):
    model = torchvision.models.resnet18(pretrained=pretrained, num_classes=1000)
    model.fc = nn.Linear(512, classes, bias=True)
    return model

def resnet34_imagenet(pretrained=True, classes=14):
    model = torchvision.models.resnet18(pretrained=pretrained, num_classes=1000)
    model.fc = nn.Linear(512, classes, bias=True)
    return model

def resnet50_imagenet(pretrained=True, classes=14):
    model = torchvision.models.resnet50(pretrained=pretrained, num_classes=1000)
    model.fc = nn.Linear(2048, classes, bias=True)
    return model

def densenet121_imagenet(pretrained=True, classes=14):
    model = torchvision.models.densenet121(pretrained=pretrained, num_classes=1000)
    model.classifier = nn.Linear(1024, classes, bias=True)
    return model

def mobilenet_v2_imagenet(pretrained=True, classes=14):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained, num_classes=1000)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                     nn.Linear(model.last_channel, classes),)
    return model

def inception_v3_imagenet(pretrained=True, classes=14):
    model = torchvision.models.inception_v3(pretrained=pretrained, num_classes=1000, aux_logits=False)
    model.fc = nn.Linear(2048, classes, bias=True)
    return model

def googlenet_imagenet(pretrained=True, classes=14):
    model = torchvision.models.googlenet(pretrained=pretrained, num_classes=1000, aux_logits=False)
    model.fc = nn.Linear(1024, classes, bias=True)
    return model
