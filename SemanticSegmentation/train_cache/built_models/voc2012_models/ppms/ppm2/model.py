from models import fcn
from models.modules import get_conv_norm_activation, OrderedDict
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, bins=[1, 2, 3, 6]):
        super().__init__()

        branches_n = len(bins)
        branches = []
        for bin in bins:
            lis = [
                ('pool', nn.AdaptiveAvgPool2d(bin)),
                ('cbr', get_conv_norm_activation(in_channels, in_channels // branches_n, 1))
            ]
            branches.append(nn.Sequential(OrderedDict(lis)))
        self.branches = nn.ModuleList(branches)
        self.conv_fuse = get_conv_norm_activation(in_channels + (in_channels // branches_n) * branches_n, out_channels, 3, 1, 1)


    def forward(self, x):
        _, _, h, w= x.shape
        branches_feats = [x]
        for branch in self.branches:
            branches_feats.append(F.interpolate(branch(x), (h, w), mode='bilinear', antialias=True, align_corners=False))
        return self.conv_fuse(torch.cat(branches_feats, dim=-3))

def build():
    r18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(r18.children())[:-2])
    ppm = PPM(512, 256)
    backbone.add_module('ppm', ppm)

    model = fcn.FCN(backbone, 256, 21)
    return model


if __name__ == '__main__':

    model = build().eval()
    print(model)
    print(model(torch.rand(1, 3, 224, 224)).shape)