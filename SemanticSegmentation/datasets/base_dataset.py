try:
    from .__init__ import torch, Image
except:
    from __init__ import torch, Image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        """
        图片和标签都能被正常加载为PIL
        """
        super().__init__()
        self.imgs = []  # 保存图片路径的Path对象
        self.targets = []  # 保存标签路径的Path对象
        self.transform = transform  # 输入是一对PIL

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        return self.pull_item(item)

    def load_img_target(self, i):
        """
        根据索引i，将一对图片和标签加载为PIL
        :param i:
        :return:
        """
        img = Image.open(self.imgs[i])
        target = Image.open(self.targets[i])
        return img, target

    def pull_item(self, i):
        """
        将根据索引i加载的一对PIL用transform处理(transform不为None)
        :param i:
        :return:
        """
        img, target = self.load_img_target(i)
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target
