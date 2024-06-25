from evaluate.evaluator import SemanticSegmentationEvaluator
from datasets.voc2012 import VOC2012
from datasets.transforms import ResizedBaseTransform
from loss_functions import Cross_Entorpy
from configurations import PROGRAM_DIR


def build():
    root = PROGRAM_DIR / 'data/voc2012'
    transform = ResizedBaseTransform((320, 480), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset = VOC2012(root, 'train', transform)
    loss_function = Cross_Entorpy(dataset.border_index, 'sum')

    evaluator = SemanticSegmentationEvaluator(dataset, loss_function)
    return evaluator

if __name__ == '__main__':
    evaluator = build()

    from dl.groceries import load_model
    model_dir = r'../../../VOC2012SS/models/R18S/FCN_R18'
    state_dict_path = r'../../results/tmp/model.pth'
    model = load_model(model_dir, state_dict_path)

    evaluator.evaluate(model, 256, 0)