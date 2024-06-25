from models.esnet import build_esnet

def build():
    return build_esnet('resnet18', True, 21)

if __name__ == '__main__':
    model = build()
    import torch
    x = torch.rand(3,3,32,32)
    print(model(x).shape)