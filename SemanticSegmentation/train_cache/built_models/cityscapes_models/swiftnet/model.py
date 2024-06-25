from models.swiftnet import build_swiftnet

def build():
    return build_swiftnet(cls_n=19)

if __name__ == '__main__':
    import torch
    model = build()
    x = torch.rand(3, 3, 32, 32)
    print(model)
    print(model(x).shape)