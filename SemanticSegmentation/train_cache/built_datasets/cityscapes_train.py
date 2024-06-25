from datasets.cityscapes import Cityscapes
from datasets.transforms import VanillaTransform
import sys
from pathlib import Path
from configurations import PROGRAM_DIR

def build():
    root = PROGRAM_DIR / 'data/cityscapes'
    transform = VanillaTransform(1024, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset = Cityscapes(root, 'train', transform)
    return dataset

if __name__ == '__main__':
    dataset = build()
    print(len(dataset))
    img, target = dataset[0]
    print(img.shape, img.dtype)
    print(target.shape, target.dtype)