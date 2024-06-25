from models.bisenet import BiSeNet

def build():
    return BiSeNet(True, 21)

if __name__ == '__main__':
    model = build()
    import torch
    x = torch.rand(3,3,32,32)
    print(model(x).shape)