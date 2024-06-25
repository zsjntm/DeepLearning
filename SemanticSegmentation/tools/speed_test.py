try:
    from .__init__ import torch, time
except:
    from __init__ import torch, time


def get_fps(model, test_size=(1024, 2048), repeat_times=15, device='cuda'):
    """

    :param model:
    :param test_size:
    :param repeat_times: 计算fps时，前向传播的次数
    :param device:
    :return:
    """
    model.eval().to(device)
    with torch.no_grad():
        # 先前向传播一次，预热
        input = torch.randn(1, 3, *test_size, device=device)
        logits = model(input)
        torch.cuda.synchronize()  # 确保GPU计算完成

        # 多次前向传播统计fps
        t_start = time.time()
        for _ in range(repeat_times):
            logits = model(input)
            torch.cuda.synchronize()
        spend_time = time.time() - t_start
        fps = repeat_times / spend_time
        return fps


if __name__ == '__main__':
    # import torchvision
    #
    # model = torchvision.models.resnet18()
    # print('resnet18 fps: {}'.format(get_fps(model)))
    #
    # print('resnet50 fps: {}'.format(get_fps(torchvision.models.resnet50())))
    # print('resnet101 fps:{}'.format(get_fps(torchvision.models.resnet101())))
    # print('convnext-t fps:{}'.format(get_fps(torchvision.models.convnext_tiny())))
    # print('efficientnet-b0 fps:{}'.format(get_fps(torchvision.models.efficientnet_b0())))

    from tools.model_tools import load_model
    swift = load_model(r'../train_cache/built_models/cityscapes_models/swiftnet')
    print('swiftnet fps:{}'.format(get_fps(swift)))
