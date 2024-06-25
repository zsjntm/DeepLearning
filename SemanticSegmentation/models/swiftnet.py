try:
    from .__init__ import torch, nn, F, torchvision
    from .modules import get_norm_activation_conv, BasicBlock, SegModel, get_conv_norm_activation, OrderedDict
except:
    from __init__ import torch, nn, F, torchvision
    from modules import get_norm_activation_conv, BasicBlock, SegModel, get_conv_norm_activation, OrderedDict


class UpSample(nn.Module):
    def __init__(self, in_channels, lateral_channels):
        super().__init__()

        self.conv_lateral = get_norm_activation_conv(lateral_channels, in_channels, 1)
        self.conv_fuse = get_norm_activation_conv(in_channels, in_channels, 3, 1, 1)

    def forward(self, x, lateral_x):
        _, _, h, w = lateral_x.shape
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=False, antialias=False) + self.conv_lateral(
            lateral_x)
        return self.conv_fuse(x)


class Decoder(nn.Module):
    def __init__(self, decode_channels=128, lateral_channels=[64, 128, 256]):
        super().__init__()

        self.upsample1 = UpSample(decode_channels, lateral_channels[0])
        self.upsample2 = UpSample(decode_channels, lateral_channels[1])
        self.upsample3 = UpSample(decode_channels, lateral_channels[2])

    def forward(self, x, lateral_xs):
        x = self.upsample3(x, lateral_xs[2])
        x = self.upsample2(x, lateral_xs[1])
        x = self.upsample1(x, lateral_xs[0])
        return x


class EncoderR18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            r18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            r18 = torchvision.models.resnet18()

        self.stem = nn.Sequential(*list(r18.children())[:4])

        self.layer1 = nn.Sequential(r18.layer1[0], BasicBlock(64, False))
        self.layer1[1].load_state_dict(r18.layer1[1].state_dict())

        self.layer2 = nn.Sequential(r18.layer2[0], BasicBlock(128, False))
        self.layer2[1].load_state_dict(r18.layer2[1].state_dict())

        self.layer3 = nn.Sequential(r18.layer3[0], BasicBlock(256, False))
        self.layer3[1].load_state_dict(r18.layer3[1].state_dict())

        self.layer4 = nn.Sequential(r18.layer4[0], BasicBlock(512, False))
        self.layer4[1].load_state_dict(r18.layer4[1].state_dict())

        self.relu = nn.ReLU(True)
        self.feat_channels = 512
        self.lateral_channels = [64, 128, 256]

    def forward(self, x):
        lateral_x1 = self.layer1(self.stem(x))
        lateral_x2 = self.layer2(self.relu(lateral_x1))
        lateral_x3 = self.layer3(self.relu(lateral_x2))
        x4 = self.layer4(self.relu(lateral_x3))
        return x4, (lateral_x1, lateral_x2, lateral_x3)


class SPP(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=512, out_channels=128, bins=(1, 2, 4, 8)):
        super().__init__()
        self.conv_in = get_conv_norm_activation(in_channels, hidden_channels, 1)

        branches_n = len(bins)
        self.branches = []
        for bin in bins:
            self.branches.append(nn.Sequential(
                OrderedDict([
                    ('pool', nn.AdaptiveAvgPool2d(bin)),
                    ('conv', get_conv_norm_activation(hidden_channels, hidden_channels // branches_n, 1)),
                ])
            ))
        self.branches = nn.ModuleList(self.branches)

        self.conv_out = get_conv_norm_activation(hidden_channels + hidden_channels // branches_n * branches_n,
                                                 out_channels, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv_in(x)
        out = [x]
        for branch in self.branches:
            out.append(F.interpolate(branch(x), (h, w), mode='bilinear'))
        return self.conv_out(torch.cat(out, dim=-3))


class SwiftNet(SegModel):
    def __init__(self, encoder, context_module, decode_channels=128, cls_n=21, seg_head='default'):
        super().__init__(decode_channels, cls_n, seg_head)

        self.encoder = encoder
        self.context_module = context_module
        self.decoder = Decoder(decode_channels, encoder.lateral_channels)

    def forward(self, x):
        _, _, h, w = x.shape
        x, (x1, x2, x3) = self.encoder(x)
        x = self.context_module(x)
        x = self.decoder(x, (x1, x2, x3))
        return super().forward(x, (h, w))

def build_swiftnet(backbone='resnet18', pretrained=True, context_module='default', decode_channels=128, cls_n=21):
    if backbone == 'resnet18':
        encoder = EncoderR18(pretrained)
    if context_module == 'default':
        context_module = SPP(encoder.feat_channels)

    swiftnet = SwiftNet(encoder, context_module, decode_channels, cls_n)
    return swiftnet


if __name__ == '__main__':
    model = build_swiftnet()
    x = torch.rand(3, 3, 32, 32)
    print(model)
    print(model(x).shape)