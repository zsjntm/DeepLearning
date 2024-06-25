from datasets.cityscapes import Cityscapes
from datasets.transforms import TrainTransformV1
import sys
from pathlib import Path
from configurations import PROGRAM_DIR


def build():
    root = PROGRAM_DIR / 'data/cityscapes'
    transform = TrainTransformV1((256, 512), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), ratio=(1, 4), flip_p=0.5)
    dataset = Cityscapes(root, 'train', transform)
    return dataset


if __name__ == '__main__':
    dataset = build()
    print(len(dataset))
    img, target = dataset[0]
    print(img.shape, img.dtype)
    print(target.shape, target.dtype)

    from tools.vis import vis_seg_map
    import matplotlib.pyplot as plt
    from tools.img_process import Tensor2img
    tensor2img = Tensor2img((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = tensor2img(img)
    plt.imshow(img)
    plt.show()
    # vis_seg_map(target, 'target', 'cityscapes', True)

