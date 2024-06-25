try:
    from .__init__ import DataLoader, torch, time
    from .pixel_counter import PixelCounter
except:
    from __init__ import DataLoader, torch, time
    from pixel_counter import PixelCounter


class SemanticSegmentationEvaluator:
    def __init__(self, dataset, loss_function):
        """
        :param dataset:
        :param loss_function: reduction为'sum'
        """
        self.dataset = dataset
        self.loss_function = loss_function
        self.pixel_counter = PixelCounter(self.dataset.cls_n, self.dataset.border_index)
        self.total_loss = 0  # 数据集中所有像素损失之和(排除边界点)
        self.pixels_num = 0  # 数据集中所有像素的数量(排除边界点)

    @torch.no_grad()
    def evaluate(self, model, bsize=25, num_workers=0, device='cuda', verbose=1):
        all_t = time.time()

        if verbose == 1:
            print('---------------------------------------------------------------------------------------------------')
            print('validate preparing... ...')
        model.eval().to(device)
        self.pixel_counter.to(device)

        # 重置pixel_counter, total_loss, pixels_num
        self.pixel_counter.reset()
        self.total_loss, self.pixels_num = 0, 0

        data_loader = DataLoader(self.dataset, batch_size=bsize, shuffle=False, num_workers=num_workers)
        if verbose == 1:
            print('batch_num:{}'.format(len(data_loader)))
            print('validate start... ...')
        batch_start_time = time.time()
        for batch_iter, (imgs, targets) in enumerate(data_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)  # model输出为logits

            # 统计边界外的三类像素点，以及边界外每个像素的损失
            self.pixel_counter.count(outputs, targets)
            batch_sum_loss = self.loss_function(outputs, targets)
            batch_pixels_num = (targets != self.pixel_counter.border_index).sum()
            self.total_loss += batch_sum_loss
            self.pixels_num += batch_pixels_num

            if verbose == 1:
                print('batch_iter:{} batch_mean_pixel_loss:{:.4f} batch_time:{:.2f}s'.format(batch_iter,
                                                                                       batch_sum_loss / batch_pixels_num,
                                                                                       time.time() - batch_start_time))
            batch_start_time = time.time()

        mIoU = self.pixel_counter.get_mIoU().item()
        loss = (self.total_loss / self.pixels_num).item()
        if verbose == 1:
            print('mIoU:{:.2f}%\nmean_pixel_loss:{:.4f}\nall_time:{:.2f}min'.format(mIoU * 100,
                                                                                    loss,
                                                                                    (time.time() - all_t) / 60))
            print('---------------------------------------------------------------------------------------------------')
        return {
            'mIoU': mIoU,  # 只统计了边界外的像素
            'loss': loss,  # 模型在整个数据集上的损失(只统计了边界外的像素)
            'pixel_counter': self.pixel_counter
        }


if __name__ == '__main__':
    from train_cache.built_datasets import voc2012_val
    dataset = voc2012_val.build()
    from loss_functions import Cross_Entorpy
    loss_function = Cross_Entorpy(255, 'sum')

    evaluator = SemanticSegmentationEvaluator(dataset, loss_function)



    """测试evaluator"""
    from dl.groceries import load_model
    model_dir = r'D:\DLWS\PROGRAMS\VOC2012SS\models\R18S\FCN_R18'
    state_dict_path = r'D:\DLWS\PROGRAMS\VOC2012SS\results\R18S\FCN_R18_lr_0_001_bsize_32\model.pth'
    model = load_model(model_dir, state_dict_path)
    results = evaluator.evaluate(model, 128, 0)
    print(results)
