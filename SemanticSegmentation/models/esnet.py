try:
    from .__init__ import torch, nn, F, torchvision
    from .modules import ECM, get_conv_norm_activation, SegModel
except:
    from __init__ import torch, nn, F, torchvision
    from modules import ECM, get_conv_norm_activation, SegModel


class SFFM(nn.Module):
    def __init__(self, spatial_in_channels, spatial_out_channels, context_channels, out_channels):
        super().__init__()

        self.conv_spatial = get_conv_norm_activation(spatial_in_channels, spatial_out_channels, 3, 1, 1)
        self.conv_fuse = get_conv_norm_activation(context_channels + spatial_out_channels, out_channels, 3, 1, 1)

    def forward(self, spatial, context):
        _, _, h, w = spatial.shape
        spatial = self.conv_spatial(spatial)
        return self.conv_fuse(torch.cat([spatial, F.interpolate(context, (h, w), mode='bilinear')], dim=-3))


class EncoderR18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            r18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            r18 = torchvision.models.resnet18()
        self.stem = nn.Sequential(r18.conv1, r18.bn1, r18.relu, r18.maxpool)
        self.stage1 = r18.layer1
        self.stage2 = r18.layer2
        self.stage3 = r18.layer3
        self.stage4 = r18.layer4

        self.spatial_channels = 128
        self.context_channels = 512

    def forward(self, x):
        spatial = self.stage2(self.stage1(self.stem(x)))
        context = self.stage4(self.stage3(spatial))
        return spatial, context


class ESNet(SegModel):
    def __init__(self, encoder, context_module='default', out_channels=256, cls_n=21):
        super().__init__(256, cls_n)

        self.encoder = encoder
        self.context_module = ECM(encoder.context_channels,
                                  encoder.context_channels // 2) if context_module == 'default' else context_module
        self.sffm = SFFM(encoder.spatial_channels, encoder.spatial_channels // 2, encoder.context_channels // 2,
                         out_channels)

    def forward(self, x):
        _, _, h, w = x.shape
        spatial, context = self.encoder(x)
        context = self.context_module(context)
        fuse = self.sffm(spatial, context)
        return super().forward(fuse, (h, w))


def build_esnet(backbone='resnet18', pretrained=True, cls_n=21):
    return ESNet(EncoderR18(pretrained), 'default', 256, cls_n)


if __name__ == '__main__':
    model = build_esnet()
    x = torch.rand(3, 3, 32, 32)
    print(model)
    print(model(x).shape)
