import torch
import sys

def load_model(model_dir, state_dict_path=None, device='cuda'):
    """
    加载模型到device，并赋予其保存的权重，返回的为eval模式
    :param model_dir: 包含一个model.py文件，且该py文件包含build()这个函数。
    :param state_dict_path: model的state_dict的路径
    :param device:
    :return:
    """

    # 加载模型结构
    sys.path.insert(0, model_dir)
    import model
    result = model.build()
    del sys.path[0]
    del sys.modules['model']

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        result.load_state_dict(state_dict)

    return result.to(device).eval()