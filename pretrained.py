# -*- coding: utf-8 -*-
import os, sys, pdb
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from timm import create_model as creat
class MyModule(nn.Module):
    def __init__(self, old_module):
        super(MyModule, self).__init__()
        self.stem = old_module.features[0:16]
        self.conv0 = old_module.features[16:23]
        self.conv1 = old_module.features[23:30]
        self.conv2 = old_module.features[30:]
        self.avgpool = old_module.avgpool
        self.fc = old_module.classifier

    def forward(self, x):
        x = self.stem(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
def pretrained_model(net_type,depth,num_classes):
    if net_type == "vgg":
        if depth == "16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif depth == "13":
            model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        elif depth == "19":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        else:
            return None
        num_ftrs = model.classifier[6].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_ftrs, num_classes))
        model.classifier = nn.Sequential(*feature_model)
        new_model = MyModule(model)
    else:
        return None

    return new_model
