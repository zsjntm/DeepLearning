from models import modules, fcn
from torchvision import models
import torch.nn as nn


def build():
    convnext = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(convnext.children())[:-2])

    model = fcn.FCN(backbone, 1280, 21)
    return model


if __name__ == '__main__':
    import torch

    model = build().eval()
    print(model)
    print(model(torch.rand(1, 3, 224, 224)).shape)