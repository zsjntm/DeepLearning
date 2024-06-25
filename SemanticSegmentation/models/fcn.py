try:
    from .__init__ import torch, nn, F, torchvision
    from .modules import SegModel
except:
    from __init__ import torch, nn, F, torchvision
    from modules import SegModel


class FCN(SegModel):
    def __init__(self, backbone, out_channels, cls_n, seg_head='default'):
        """
        :param backbone:
        :param out_channels: backbone输出的特征图的通道数
        :param cls_n:
        :param seg_head:
        """
        super().__init__(out_channels, cls_n, seg_head)
        self.backbone = backbone

    def forward(self, x):
        _, _, x_h, x_w = x.shape
        seg_feature = self.backbone(x)
        return super().forward(seg_feature, (x_h, x_w))


def build_fcn(backbone='resnet18', pretrained=True, cls_n=21):
    if backbone == 'resnet18':
        out_channels = 512
        if pretrained:
            backbone = nn.Sequential(
                *list(
                    torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).children())[
                 :-2])
        else:
            backbone = nn.Sequential(*list(torchvision.models.resnet18().children())[:-2])

    return FCN(backbone, out_channels, cls_n)


if __name__ == '__main__':
    '''FCN'''
    # from torchvision.models import resnet18
    # r18 = nn.Sequential(*list(resnet18().children())[:-2])
    # fcn_r18 = FCN(r18, 512, 21)
    #
    # x = torch.rand(1, 3, 224, 224)
    # print(fcn_r18)
    # print(fcn_r18(x).shape)

    '''build_fcn'''
    # fcn_r18 = build_fcn()
    # x = torch.rand(1, 3, 224, 224)
    # print(fcn_r18)
    # print(fcn_r18(x).shape)
