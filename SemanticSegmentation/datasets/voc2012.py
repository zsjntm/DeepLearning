try:
    from .__init__ import Path, Image
    from .base_dataset import BaseDataset
except:
    from __init__ import Path, Image
    from base_dataset import BaseDataset


class VOC2012(BaseDataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__(transform)

        self.root = Path(root)
        input_images_dir = self.root / split / 'input_images'
        label_images_dir = self.root / split / 'label_images'

        for images_path, labels_path in zip(input_images_dir.iterdir(), label_images_dir.iterdir()):
            self.imgs.append(images_path)
            self.targets.append(labels_path)

        self.cls_n = 21
        self.border_index = 255

if __name__ == '__main__':
    dataset = VOC2012(r'../data/voc2012')
    img, target = dataset[0]
    print(img, target)

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    plt.imshow(target)
    plt.show()