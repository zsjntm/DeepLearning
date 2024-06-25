from datasets.voc2012 import VOC2012
from datasets.transforms import VanillaTransform
import sys
from pathlib import Path
from configurations import PROGRAM_DIR

def build():
    root = PROGRAM_DIR / 'data/voc2012'

    transform = VanillaTransform((320, 480), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset = VOC2012(root, 'train', transform)
    return dataset

if __name__ == '__main__':
    dataset = build()
    img, target = dataset[2]
    print(img.shape, img.dtype, type(img), (img.min(), img.max()))
    print(target.shape, target.dtype, type(target), (target.min(), target.max()))
    print(len(dataset))

    from tools.img_process import Tensor2img
    tensor2img = Tensor2img((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = tensor2img(img)
    target = target.numpy()[0]

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    plt.imshow(target)
    plt.show()
