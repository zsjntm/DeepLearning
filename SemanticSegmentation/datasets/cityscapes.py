try:
    from .__init__ import Path
    from .base_dataset import BaseDataset
except:
    from __init__ import Path
    from base_dataset import BaseDataset


class Cityscapes(BaseDataset):
    def __init__(self, root, split='train', transform=None):
        super(Cityscapes, self).__init__(transform)

        self.root = Path(root)
        imgs_dir = self.root / 'leftImg8bit' / split
        targets_dir = self.root / 'gtFine' / split

        # 初始化图片名、标签名列表
        for city in imgs_dir.iterdir():
            for img in city.iterdir():
                target = targets_dir / city.name / '{}_{}_{}_gtFine_labelTrainIds.png'.format(
                    *(img.name.split('_')[:3]))
                # print(img, target)
                self.imgs.append(img)
                self.targets.append(target)

        self.cls_n = 19
        self.border_index = 255


if __name__ == '__main__':
    '''Cityscapes'''
    root = r'../data/cityscapes'
    train_set = Cityscapes(root)
    print(len(train_set))
    img, target = train_set[425]
    print(img, target)

    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(target)
    # plt.show()

    # import numpy as np
    # import pandas as pd
    # resut = np.array([], dtype='uint8')
    # for i in range(2975):
    #     _, target = train_set[i]
    #     target = np.asarray(target)
    #     resut = np.concatenate([resut, target.flatten()])
    #     # print(resut, resut.dtype, resut.shape, 1024*2048)
    #     # break
    #
    #     if i % 50 == 0:
    #         resut = pd.unique(resut)
    #         resut.sort()
    #         print(i, resut)
    # print()
    # resut = pd.unique(resut)
    # resut.sort()
    # print(resut)  # 0-18共19个类，255为边界，不参与计算与评估；车牌类别被并入到了car类别中
