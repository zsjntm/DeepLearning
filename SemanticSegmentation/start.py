import os, time

# os.system('python train.py --help')
# os.system(
#     'python train.py --model_dir train_cache/built_models/voc2012_models/r18 --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/r18 --lr 0.001 --bsize 32 --num_workers 10 --epoch 46')

"have a try on cityscapes"
# # 探索cityscapes的超参，用fcn_r18
# t = time.time()
# lrs = [0.1, 0.01, 0.001, 0.001]  # 选0.01
# bsize = 32
# for lr in lrs:
#     os.system('python train.py --model_dir train_cache/built_models/cityscapes_models/r18 --train_dataset cityscapes_train --val_dataset cityscapes_val --results_dir results/cityscapes/r18_hyper_search/lr={} --lr {} --bsize {} --num_workers 1 --epoch 2'.format(lr, lr, bsize))
#     # break
# print('hyper parameter search spend time:{:.2f}min'.format((time.time() - t) / 60))
#
# # 接着训lr=0.01
# t = time.time()
# lr = 0.01
# bsize = 32
# os.system('python train.py --model_dir train_cache/built_models/cityscapes_models/r18 --train_dataset cityscapes_train --val_dataset cityscapes_val --results_dir results/cityscapes/r18_hyper_search/lr={} --lr {} --bsize {} --num_workers 1 --epoch 40'.format(lr, lr, bsize))
# print('spend time:{:.2f}min'.format((time.time() - t) / 60))  # 一个epoch只包括训练：3.67min

"enlarge bsize of fcn_r18 on voc2012"
# t = time.time()
# lrs = [
#     # 0.1,
#     0.01,  # 选这个，接着训
#     # 0.001,
# ]
# bsize = 256
# for lr in lrs:
#     os.system(
#         'python train.py --model_dir train_cache/built_models/voc2012_models/r18 --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/r18_large_bsize_hyper_search/lr={} --lr {} --bsize {} --num_workers 1 --epoch 99'.format(lr, lr, bsize))
# print('hyper parameter search time:{:.2f}min'.format((time.time() - t) / 60))

"unknown"
# r18_ecm
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/r18_ecm --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/r18_ecm --lr 0.01 --bsize 256 --num_workers 1 --epoch 10')
# r18_ppm
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/r18_ppm --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/r18_ppm --lr 0.01 --bsize 256 --num_workers 1 --epoch 60')
# convnext
# for lr in [0.01, 0.001]:
#     os.system('python train.py --model_dir train_cache/built_models/voc2012_models/convnext_t --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/convnext_t_hyper_search/lr={} --lr {} --bsize 96 --num_workers 1 --epoch 2'.format(lr, lr))

# # efficientnet_b0
# for lr in [
#     # 1,
#     0.1,
#     0.01,
#     # 0.001,
# ]:
#     print('********************************lr={}*****************************************'.format(lr))
#     os.system('python train.py --model_dir train_cache/built_models/voc2012_models/efficientnet_b0 --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/efficientnet_b0_hyper_search/lr={} --lr {} --bsize 128 --num_workers 1 --epoch 30'.format(lr, lr))

"align_corner, antialias, upsample_type in ppm"
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/ppms/ppm1 --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/ppms/ppm1 --lr 0.01 --bsize 256 --num_workers 1 --epoch 20')
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/ppms/ppm2 --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/ppms/ppm2 --lr 0.01 --bsize 256 --num_workers 1 --epoch 20')
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/ppms/ppm3 --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/ppms/ppm3 --lr 0.01 --bsize 256 --num_workers 1 --epoch 30')

"drop last relu of r18 to get features whose value range is R, inspired by resnet_v2"
# r18a: r18最后一个block输出不要激活
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/r18a --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/r18a --lr 0.01 --bsize 256 --num_workers 1 --epoch 29')

# r18a + ecm改进系列, (也是resnet_v2原理)
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/r18a_ecma --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/r18a_ecma --lr 0.01 --bsize 256 --num_workers 1 --epoch 30')
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/r18a_ecmb --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/r18a_ecmb --lr 0.01 --bsize 256 --num_workers 1 --epoch 30')
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/r18a_ecm --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/r18a_ecm --lr 0.01 --bsize 256 --num_workers 1 --epoch 30')

# BRC 改进ECM
# os.system('python train.py --model_dir train_cache/built_models/voc2012_models/r18a_ecmc --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/r18a_ecmc --lr 0.01 --bsize 256 --num_workers 1 --epoch 10')

"ECM try on cityscapes"
# # cityscapes进一步尝试，加入了改进的ECM模块，采用了改进的backbone
# os.system('python train.py --model_dir train_cache/built_models/cityscapes_models/r18a_ecmb --train_dataset cityscapes_train --val_dataset cityscapes_val --results_dir results/cityscapes/r18a_ecmb --lr 0.01 --bsize 32 --num_workers 1 --epoch 42')

"cityscapes' new training recipes"
# # recipe1
# lrs = [
#     # 1,
#     # 0.1,
#     0.01,  # choose 0.01
#     # 0.001,
#     # 0.0001
# ]
# for lr in lrs:
#     os.system(
#         'python train.py --model_dir train_cache/built_models/cityscapes_models/r18 --train_dataset cityscapes_train_recipe1 --val_dataset cityscapes_val --results_dir results/cityscapes/r18_recipe1_hyper_search/lr={} --lr {} --bsize 64 --num_workers 2 --epoch 45'.format(
#             lr, lr))
#     print('lr={} training finished ... ...'.format(lr))  # 一个epoch只包括训练：1.8min

# # recipe2
# lrs = [
#     # 1,
#     # 0.1,
#     0.01,  # choose 0.01
#     # 0.001,
#     # 0.0001
# ]
# for lr in lrs:
#     os.system('python train.py --model_dir train_cache/built_models/cityscapes_models/r18 --train_dataset cityscapes_train_recipe2 --val_dataset cityscapes_val --results_dir results/cityscapes/r18_recipe2_hyper_search/lr={} --lr {} --bsize 256 --num_workers 3 --epoch 16'.format(lr, lr))
#     print('lr={} training finished ... ...'.format(lr))

"compare to classic in voc2012"
# # swiftnet
# print('swiftnet train start ... ...')
# lrs = [
#     # 0.1,
#     0.01,  # 选这个，接着训
#     # 0.001,
# ]
# bsize = 224
# for lr in lrs:
#     os.system(
#         'python train.py --model_dir train_cache/built_models/voc2012_models/swiftnet --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/swiftnet/lr={} --lr {} --bsize {} --num_workers 1 --epoch 99'.format(lr, lr, bsize))
#
# # bisenet
# print('bisenet train start ... ...')
# lrs = [
#     # 0.1,
#     0.01,  # 选这个，接着训
#     # 0.001,
# ]
# bsize = 224
# for lr in lrs:
#     os.system(
#         'python train.py --model_dir train_cache/built_models/voc2012_models/bisenet --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/bisenet/lr={} --lr {} --bsize {} --num_workers 1 --epoch 99'.format(lr, lr, bsize))
#
# # pidnet
# print('pidnet train start ... ...')
# lrs = [
#     # 0.1,
#     0.01,  # 选这个，接着训
#     # 0.001,
# ]
# bsize = 160
# for lr in lrs:
#     os.system(
#         'python train.py --model_dir train_cache/built_models/voc2012_models/pidnet --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/pidnet/lr={} --lr {} --bsize {} --num_workers 1 --epoch 99'.format(lr, lr, bsize))
# # time.sleep(100)
#
# # esnet
# print('esnet train start ... ...')
# lrs = [
#     # 0.1,
#     0.01,  # 选这个，接着训
#     # 0.001,
# ]
# bsize = 256
# for lr in lrs:
#     os.system(
#         'python train.py --model_dir train_cache/built_models/voc2012_models/esnet --train_dataset voc2012_train --val_dataset voc2012_val --results_dir results/voc2012/esnet/lr={} --lr {} --bsize {} --num_workers 1 --epoch 99'.format(lr, lr, bsize))

"compare to classic in cityscapes"
# all recipe1

# # swiftnet
# lrs = [
#     # 1,
#     # 0.1,
#     0.01,  # choose 0.01
#     # 0.001,
#     # 0.0001
# ]
# for lr in lrs:
#     os.system(
#         'python train.py --model_dir train_cache/built_models/cityscapes_models/swiftnet --train_dataset cityscapes_train_recipe1 --val_dataset cityscapes_val --results_dir results/cityscapes/swiftnet/lr={} --lr {} --bsize 64 --num_workers 2 --epoch 31'.format(
#             lr, lr))
#     print('lr={} training finished ... ...'.format(lr))


# bisenet
lrs = [
    # 1,
    # 0.1,
    0.01,  # choose 0.01
    # 0.001,
    # 0.0001
]
for lr in lrs:
    os.system(
        'python train.py --model_dir train_cache/built_models/cityscapes_models/bisenet --train_dataset cityscapes_train_recipe1 --val_dataset cityscapes_val --results_dir results/cityscapes/bisenet/lr={} --lr {} --bsize 56 --num_workers 2 --epoch 31'.format(
            lr, lr))
    print('lr={} training finished ... ...'.format(lr))


# pidnet
lrs = [
    # 1,
    # 0.1,
    0.01,  # choose 0.01
    # 0.001,
    # 0.0001
]
for lr in lrs:
    os.system(
        'python train.py --model_dir train_cache/built_models/cityscapes_models/pidnet --train_dataset cityscapes_train_recipe1 --val_dataset cityscapes_val --results_dir results/cityscapes/pidnet/lr={} --lr {} --bsize 48 --num_workers 2 --epoch 60'.format(
            lr, lr))
    print('lr={} training finished ... ...'.format(lr))

# esnet
lrs = [
    # 1,
    # 0.1,
    0.01,  # choose 0.01
    # 0.001,
    # 0.0001
]
for lr in lrs:
    os.system(
        'python train.py --model_dir train_cache/built_models/cityscapes_models/esnet --train_dataset cityscapes_train_recipe1 --val_dataset cityscapes_val --results_dir results/cityscapes/esnet/lr={} --lr {} --bsize 64 --num_workers 2 --epoch 59'.format(
            lr, lr))
    print('lr={} training finished ... ...'.format(lr))


