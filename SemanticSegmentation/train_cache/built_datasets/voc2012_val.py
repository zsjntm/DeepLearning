from datasets.voc2012 import VOC2012
from datasets.transforms import ResizedBaseTransform
import sys
from pathlib import Path
from configurations import PROGRAM_DIR

def build():
    root = PROGRAM_DIR / 'data/voc2012'

    dataset = VOC2012(root, 'val', ResizedBaseTransform((320, 480), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return dataset

if __name__ == '__main__':
    dataset = build()
    img, target = dataset[2]
    print(img.shape, img.dtype, type(img), (img.min(), img.max()))
    print(target.shape, target.dtype, type(target), (target.min(), target.max()))
    print(len(dataset))
