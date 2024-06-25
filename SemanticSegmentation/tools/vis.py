try:
    from .__init__ import torch, plt, DataLoader, Path, sys, Image, np, PROGRAM_DIR, cv2, time
    from .img_process import Tensor2img
except:
    from __init__ import torch, plt, DataLoader, Path, sys, Image, np, PROGRAM_DIR, cv2, time
    from img_process import Tensor2img

palette_foreground = [
    128, 64,128,
    244, 35,232,
     70, 70, 70,
    102,102,156,
    190,153,153,
    153,153,153,
    250,170, 30,
    220,220,  0,
    107,142, 35,
    152,251,152,
     70,130,180,
    220, 20, 60,
    255,  0,  0,
      0,  0,142,
      0,  0, 70,
      0, 60,100,
      0, 80,100,
      0,  0,230,
    119, 11, 32,
]
palette_padding = [0] * (768 - 19 * 3 - 3)
palette_border = [0, 0, 0]
PALETTE_CITYSCAPES = palette_foreground + palette_padding + palette_border

def vis_history(history_path, show=True):
    history = torch.load(history_path)
    if show == True:
        plt.title('train and val losses')
        xs = range(1, len(history['train_losses']) + 1)
        plt.plot(xs, history['train_losses'], 'bo', label='train_loss')
        plt.plot(xs, history['val_losses'], 'b', label='val_loss')
        plt.legend()
        plt.show()

        plt.title('val mIoUs')
        plt.plot(xs, history['val_mIoUs'])
        plt.show()

    return history


def vis_histories(histories, colors=None, labels=None, type='train_losses', clip_min=None, clip_max=999, start_epoch=1):
    if labels is None:
        labels = range(len(histories))  # 默认为[0, 1, ...]

    if colors is None:
        colors = [None for _ in labels]  # 默认为[None, None, ...]

    for i, history in enumerate(histories):
        label, color = labels[i], colors[i]

        xs = range(1, len(history[type]) + 1)[start_epoch - 1: ]
        ys = np.clip(history[type], clip_min, clip_max)[start_epoch - 1: ]
        plt.plot(xs, ys, color=color, label=label)
    plt.legend()
    plt.title(type)
    plt.show()


def vis_seg_map(map, type='output', mapping='voc2012', show=True):
    """
    :param map: torch.tensor
    :param type: option: 'output': (cls_n, h, w), 'target': (1, h, w)
    :param mapping: 采用哪个数据集的类别到颜色的映射, option: 'voc2012', 'cityscapes'
    :return: 色彩掩码图 RGB
    """

    if type == 'output':
        map = map.argmax(dim=-3)  # (h, w)

    if type == 'target':
        map = map.squeeze()  # (h, w)

    if mapping == 'voc2012':
        img_path = PROGRAM_DIR / 'data/voc2012/train/label_images/2007_000032.png'
        palette = Image.open(img_path).getpalette()

    if mapping == 'cityscapes':
        palette = PALETTE_CITYSCAPES

    map = map.detach().clone().to('cpu').numpy().astype('uint8')  # (h, w) 'uint8'
    pil = Image.fromarray(map, mode='P')
    pil.putpalette(palette)
    result = np.asarray(pil.convert('RGB'))

    if show:
        plt.imshow(result)
        plt.show()

    return result


@torch.no_grad()
def seg_dataset(model, dataset, results_dir, tensor2img='default', mapping='voc2012', bsize=256, num_workers=0, margin=5, device='cuda',
                verbose=1):
    """
    :param model:
    :param dataset: option: 'voc2012_val'
    :param results_dir:
    :param tensor2img: 将数据集的tensor形式的img转换为RGB形式, option: 'default': IN1k的mean, std解标准化, 再*255；或其他函数
    :param mapping: 采用哪个数据集的类别到颜色的映射, option: 'voc2012', 'cityscapes'
    :param bsize:
    :param num_workers:
    :param device:
    :param verbose:
    :return:
    """
    all_t = time.time()
    model.eval().to(device)
    if tensor2img == 'default':
        tensor2img = Tensor2img((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # make results_dir
    results_dir = Path(results_dir)
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
        if verbose == 1:
            print('dir is not exist, and make it successfully')

    data_loader = DataLoader(dataset, bsize, False, num_workers=num_workers)
    img_idx = 0
    batch_time = time.time()
    for batch_iter, (imgs, targets) in enumerate(data_loader):
        outputs = model(imgs.to(device))

        for img, output, target in zip(imgs, outputs, targets):
            _, h, w = img.shape
            padding = np.zeros((h, margin, 3), dtype='uint8')
            img = tensor2img(img)
            output = vis_seg_map(output, 'output', mapping, False)
            target = vis_seg_map(target, 'target', mapping, False)

            result = np.concatenate([img, padding, output, padding, target], axis=-2)
            cv2.imwrite(str(results_dir / dataset.imgs[img_idx].name), result[:, :, ::-1])
            img_idx += 1
        if verbose == 1:
            print('batch_iter:{} batch_time:{:.2f}s'.format(batch_iter, time.time() - batch_time))
        batch_time = time.time()

    if verbose == 1:
        print('all_time:{:.2f}min'.format((time.time() - all_t) / 60))
