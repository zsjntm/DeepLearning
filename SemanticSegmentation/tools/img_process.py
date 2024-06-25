try:
    from .__init__ import torch, np
except:
    from __init__ import torch, np


class Tensor2img:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape(3, 1, 1).contiguous()
        self.std = torch.tensor(std).reshape(3, 1, 1).contiguous()

    def __call__(self, tensor, stdz=True):
        """
        :param tensor: (3, h, w) torch.float32 [0, 1](标准化前)
        :param stdz: 是否经过标准化
        :return: RGB
        """
        tensor = tensor.to('cpu')
        if stdz:
            tensor = tensor * self.std + self.mean
        tensor *= 255

        img = tensor.numpy().transpose(1, 2, 0)  # (h, w, 3)
        img = np.clip(img, 0, 255).astype('uint8')
        return img
