import argparse, time
from pathlib import Path

import loss_functions
from torch import optim
from tools.model_tools import load_model
from evaluate.evaluator import SemanticSegmentationEvaluator
from train.trainer import SemanticSegmentationTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training model by args')
    parser.add_argument('--model_dir', type=str,
                        help='there is a model.py which has a \'build()\' function in this dir', dest='model_dir')
    parser.add_argument('--train_dataset', type=str,
                        help='train_cache.built_datasets.[dataset], which has a \'build()\' function',
                        dest='train_dataset')
    parser.add_argument('--val_dataset', type=str,
                        help='train_cache.built_datasets.[dataset], which has a \'build()\' function',
                        dest='val_dataset')
    parser.add_argument('--results_dir', type=str, dest='results_dir')

    parser.add_argument('--lr', type=float, dest='lr')
    parser.add_argument('--bsize', type=int, dest='bsize')
    parser.add_argument('--num_workers', type=int, dest='num_workers')
    parser.add_argument('--epoch', type=int, dest='epoch')

    parser.add_argument('--use_autocast', default=True, type=bool, help='use autocast or not, default is True',
                        dest='use_autocast')
    parser.add_argument('--device', default='cuda', type=str, help='training device, default is cuda', dest='device')
    args = parser.parse_args()

    all_t = time.time()
    # 接受输入的参数
    model_dir, results_dir = args.model_dir, args.results_dir
    exec('from train_cache.built_datasets.{} import build as build_train_set'.format(args.train_dataset))
    exec('from train_cache.built_datasets.{} import build as build_val_set'.format(args.val_dataset))
    train_dataset, val_dataset = build_train_set(), build_val_set()

    lr, bsize, num_workers, epoch = args.lr, args.bsize, args.num_workers, args.epoch
    use_autocast, device = args.use_autocast, args.device

    # 初始化model，训练配置
    model = load_model(model_dir, device=device)
    optimizer = optim.SGD(model.parameters(), lr, 0.9)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=999999)

    train_loss_function = loss_functions.Cross_Entorpy(train_dataset.border_index, 'mean')
    val_loss_function = loss_functions.Cross_Entorpy(val_dataset.border_index, 'sum')

    evaluator = SemanticSegmentationEvaluator(val_dataset, val_loss_function)
    trainer = SemanticSegmentationTrainer(model, train_dataset, train_loss_function, optimizer, lr_scheduler, evaluator,
                                          bsize, num_workers, True, use_autocast, results_dir)  # 如果有检查点，则交给trainer负责

    trainer.start(epoch, device)
    print('spend all time:{:.2f}min'.format((time.time() - all_t) / 60))
