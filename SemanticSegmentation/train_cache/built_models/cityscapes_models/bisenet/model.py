from models.bisenet import BiSeNet

def build():
    return BiSeNet(True, 19)

if __name__ == '__main__':
    bisenet = build()
    import torch
    x = torch.rand(3, 3, 32, 32)
    print(bisenet(x).shape)