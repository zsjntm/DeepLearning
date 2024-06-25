import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import get_conv_norm_activation, get_conv_norm
from models.fcn import FCN
from torchvision import models


class BasicBlock(nn.Module):
    def __init__(self, channels, output_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.output_relu = output_relu

    def forward(self, x):
        x = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + x
        if self.output_relu:
            return self.relu(x)
        else:
            return x


class ECMA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_global = get_conv_norm(in_channels, in_channels, 1)
        self.conv_fuse = get_conv_norm_activation(in_channels * 2, out_channels, 3, 1, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        context = self.conv_global(self.gap(x))
        return self.conv_fuse(torch.cat([x, F.interpolate(context, (h, w), mode='nearest')], dim=-3))


def build():
    r18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    state_dict = r18.layer4[1].state_dict()
    last_block = BasicBlock(512, False)
    last_block.load_state_dict(state_dict)

    r18.layer4 = nn.Sequential(r18.layer4[0], last_block)
    backbone = nn.Sequential(*list(r18.children())[:-2])
    ecma = ECMA(512, 256)
    backbone.add_module('ecma', ecma)
    return FCN(backbone, 256, 19)


if __name__ == '__main__':
    # r18 = models.resnet18()
    # print(r18)
    model = build()
    print(model)
    print(model(torch.rand(3, 3, 224, 224)).shape)
