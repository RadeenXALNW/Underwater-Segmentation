import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=16):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes
        self.resnet = ResNet50()
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.resnet(x)
        output = self.aspp(feature_map)
        output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=False)
        return output
