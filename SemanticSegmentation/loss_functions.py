import torch.nn.functional as F


class Cross_Entorpy:
    def __init__(self, ignore_index=-100, reduction='mean'):
        """
        :param ignore_index: 这里的默认值为F.cross_entropy的ignore_index的默认值
        :param reduction: 这里的默认值为F.cross_entropy的reduction的默认值
        """
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, outputs, targets):
        """
        :param outputs: (b, cls_n, h, w) logits
        :param targets: (b, 1, h, w) indices
        :return:
        """
        outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, outputs.shape[-3])  # (b * h * w, cls_n)
        targets = targets.flatten().contiguous()  # (b * h * w, )
        loss = F.cross_entropy(outputs, targets, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss


if __name__ == '__main__':
    import torch

    outputs = torch.randn(15, 10, 3, 3)
    targets = torch.randint(low=0, high=11, size=(15, 1, 3, 3), dtype=torch.int64)
    cce = Cross_Entorpy(ignore_index=10, reduction='sum')
    print(cce(outputs, targets))
