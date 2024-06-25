try:
    from .__init__ import torch, nn, F
except:
    from __init__ import torch, nn, F


class PixelCounter(nn.Module):
    def __init__(self, cls_n, border_index):
        super().__init__()

        self.cls_n = cls_n
        self.border_index = border_index

        self.register_buffer('nii', torch.zeros(cls_n, dtype=torch.int64))
        self.register_buffer('ni_', torch.zeros(cls_n, dtype=torch.int64))
        self.register_buffer('n_i', torch.zeros(cls_n, dtype=torch.int64))

    def count(self, outputs, targets):
        """
        统计一个批量中的三种像素点的数量
        :param outputs: logits or prob (batch_num, cls_n, h, w)
        :param targets: indices (batch_num, 1, h, w)
        :return:
        """

        masks = targets != self.border_index
        targets = F.one_hot(targets * masks, self.cls_n).type(torch.uint8).squeeze().permute(0, 3, 1, 2).contiguous() * masks
        torch.cuda.empty_cache()
        outputs = F.one_hot(outputs.argmax(dim=-3), self.cls_n).type(torch.uint8).permute(0, 3, 1, 2).contiguous()
        torch.cuda.empty_cache()
        # print(targets.dtype, outputs.dtype)
        self.nii = self.nii + (targets * outputs).sum(dim=(0, 2, 3))
        self.ni_ = self.ni_ + targets.sum(dim=(0, 2, 3))
        self.n_i = self.n_i + (outputs * masks).sum(dim=(0, 2, 3))

    def reset(self):
        self.nii = self.nii - self.nii
        self.ni_ = self.ni_ - self.ni_
        self.n_i = self.n_i - self.n_i

    def get_ious(self):
        return self.nii / (self.ni_ + self.n_i - self.nii)

    def get_mIoU(self):
        return self.get_ious().mean()


if __name__ == '__main__':
    outputs = torch.randn(15, 10, 3, 3)
    targets = torch.randint(low=0, high=11, size=(15, 1, 3, 3), dtype=torch.int64)
    # print(targets)

    pixel_counter = PixelCounter(10, 10)
    pixel_counter.count(outputs, targets)
    print(pixel_counter.nii, pixel_counter.ni_, pixel_counter.n_i)
