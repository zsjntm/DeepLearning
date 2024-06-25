try:
    from .__init__ import torch, nn, F, torchvision
    from .modules import get_norm_activation_conv, BasicBlock, SegModel, get_conv_norm_activation, OrderedDict, get_conv_norm
except:
    from __init__ import torch, nn, F, torchvision
    from modules import get_norm_activation_conv, BasicBlock, SegModel, get_conv_norm_activation, OrderedDict, get_conv_norm


class ARM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.cbr = get_conv_norm_activation(in_channels, out_channels, kernel_size, stride, padding)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cb = get_conv_norm(out_channels, out_channels, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cbr(x)
        attn = self.sigmoid(self.cb(self.gap(x)))
        return x * attn


class FFM(nn.Module):
    def __init__(self, in_channels, fuse_channels, fuse_kernel_size=(1, 1), stride=1, padding=0):
        super().__init__()
        self.fuse_cbr = get_conv_norm_activation(in_channels, fuse_channels, fuse_kernel_size, stride, padding)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cb = get_conv_norm(fuse_channels, fuse_channels, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        fuse = self.fuse_cbr(torch.cat([x1, x2], dim=-3))

        attn = self.sigmoid(self.cb(self.gap(fuse)))
        return fuse + fuse * attn

class BiSeNet(SegModel):
    def __init__(self, SP=True, cls_n=21):
        super().__init__(256, cls_n)

        res18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(res18.conv1, res18.bn1, res18.relu, res18.maxpool)
        self.stage1 = res18.layer1
        self.stage2 = res18.layer2
        self.stage3 = res18.layer3
        self.stage4 = res18.layer4

        self.arm16 = ARM(256, 128)
        self.arm32 = ARM(512, 128)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap_cbr = get_conv_norm_activation(512, 128, (1, 1))
        self.cbr_32 = get_conv_norm_activation(128, 128, (3, 3), padding=(1, 1))
        self.cbr_16 = get_conv_norm_activation(128, 128, (3, 3), padding=(1, 1))

        if SP == True:
            self.SP = nn.Sequential()
            self.SP.add_module('cbr1', get_conv_norm_activation(3, 64, (7, 7), 2, (3, 3)))
            self.SP.add_module('cbr2', get_conv_norm_activation(64, 64, (3, 3), 2, (1, 1)))
            self.SP.add_module('cbr3', get_conv_norm_activation(64, 64, (3, 3), 2, (1, 1)))
            self.SP.add_module('cbr4', get_conv_norm_activation(64, 128, (1, 1)))

            self.ffm = FFM(256, 256, (1, 1))
        else:
            self.SP = None

    def forward(self, x):
        _, _, h, w = x.shape
        feature16 = self.stage3(self.stage2(self.stage1(self.stem(x))))
        _, _, h16, w16 = feature16.shape
        feature32 = self.stage4(feature16)
        context = self.gap(feature32)

        feature32_up2 = self.cbr_32(F.interpolate(self.arm32(feature32) + self.gap_cbr(context), (h16, w16), mode='bilinear', align_corners=True))

        if self.SP is None:
            feature16_up2 = self.cbr_16(
                interpolate(self.arm16(feature16) + feature32_up2, scale_factor=2., mode='bilinear', align_corners=True))
            return feature16_up2

        else:
            spatial = self.SP(x)
            _, _, h8, w8 = spatial.shape
            feature16_up2 = self.cbr_16(
                F.interpolate(self.arm16(feature16) + feature32_up2, (h8, w8), mode='bilinear', align_corners=True))
            seg_feature = self.ffm(feature16_up2, spatial)
            # print('jntm......')
            return super().forward(seg_feature, (h, w))

if __name__ == '__main__':
    bisenet = BiSeNet()
    x = torch.rand(3, 3, 32, 32)
    print(bisenet(x).shape)