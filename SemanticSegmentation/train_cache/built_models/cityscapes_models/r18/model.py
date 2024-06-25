from models.fcn import build_fcn

def build():
    return build_fcn('resnet18', True, 19)