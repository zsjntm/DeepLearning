from datasets.cityscapes import Cityscapes
from datasets.transforms import BaseTransform
import sys
from pathlib import Path
from configurations import PROGRAM_DIR

def build():
    root = PROGRAM_DIR / 'data/cityscapes'
    transform = BaseTransform((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset = Cityscapes(root, 'val', transform)
    return dataset


if __name__ == '__main__':
    dataset = build()
    print(len(dataset))
    img, target = dataset[0]
    print(img.shape, img.dtype, type(img), (img.min(), img.max()))
    print(target.shape, target.dtype, type(target), (target.min(), target.max()))

    from tools.img_process import Tensor2img
    tensor2img = Tensor2img((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    img = tensor2img(img)
    target = target.numpy()[0]

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    plt.imshow(target)
    plt.show()
