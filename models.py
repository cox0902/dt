from typing import *

import torch
from torch import nn
import torchvision


class VisModel(nn.Module):

    def __init__(self, model: Literal["resnet", "resnext"] = "resnet", load_weight: bool = False, copy_weight: bool = False,  
                 use_logits: bool = False):
        super(VisModel, self).__init__()
        self.use_logits = use_logits

        if model == "resnet":
            if load_weight:
                resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            else:
                resnet = torchvision.models.resnet50()
        elif model == "resnext":
            if load_weight:
                resnet = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
            else:
                resnet = torchvision.models.resnext50_32x4d()
        else:
            assert False, model
        
        old_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(in_channels=4, out_channels=old_conv1.out_channels, kernel_size=old_conv1.kernel_size, 
                              stride=old_conv1.stride, padding=old_conv1.padding, bias=old_conv1.bias)

        if load_weight:
            new_conv1.weight.data[:, :old_conv1.in_channels, :, :] = old_conv1.weight.data.clone()
            if copy_weight:
                new_conv1.weight.data[:, old_conv1.in_channels:, :, :] = old_conv1.weight.data.mean(dim=1, keepdim=True)
        else:
            assert not copy_weight

        resnet.conv1 = new_conv1
        sequential = [
            nn.Linear(in_features=2048, out_features=1),
        ]
        if not use_logits:
            sequential.append(nn.Sigmoid())
        resnet.fc = nn.Sequential(*sequential)
        self.resnet = resnet
        
    def forward(self, batch):
        logits = self.resnet(batch["image"])
        predicts = torch.sigmoid(logits) if self.use_logits else logits
        return logits, predicts, batch["target"]

    def predict(self, batch):
        with torch.no_grad():
            fextractor = nn.Sequential(*list(self.resnet.children())[:-1])
            classifier = self.resnet.fc
            features = fextractor(batch["image"]).squeeze()
            predicts = torch.sigmoid(classifier(features))
        return features, predicts
