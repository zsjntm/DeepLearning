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

def build():
    r18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    state_dict = r18.layer4[1].state_dict()
    last_block = BasicBlock(512, False)
    last_block.load_state_dict(state_dict)

    r18.layer4 = nn.Sequential(r18.layer4[0], last_block)
    backbone = nn.Sequential(*list(r18.children())[:-2])
    return FCN(backbone, 512, 21)

if __name__ == '__main__':
    # r18 = models.resnet18()
    # print(r18)
    model = build()
    print(model)
    print(model(torch.rand(1,3,224,224)).shape)
