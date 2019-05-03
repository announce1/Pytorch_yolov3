import argparse
import time

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

# Hyperparameters
# 0.861      0.956      0.936      0.897       1.51      10.39     0.1367    0.01057    0.01181     0.8409     0.1287   0.001028     -3.441     0.9127  0.0004841
hyp = {'k': 10.39,  # loss multiple  损失倍数
       'xy': 0.1367,  # xy loss fraction  xy损失分数
       'wh': 0.01057,  # wh loss fraction
       'cls': 0.01181,  # cls loss fraction
       'conf': 0.8409,  # conf loss fraction
       'iou_t': 0.1287,  # iou target-anchor training threshold
       'lr0': 0.001028,  # initial learning rate
       'lrf': -3.441,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.9127,  # SGD momentum
       'weight_decay': 0.0004841,  # optimizer weight decay
       }


# 0.856       0.95      0.935      0.887        1.3      8.488     0.1081    0.01351    0.01351     0.8649        0.1      0.001         -3        0.9     0.0005
# hyp = {'k': 8.4875,  # loss multiple
#        'xy': 0.108108,  # xy loss fraction
#        'wh': 0.013514,  # wh loss fraction
#        'cls': 0.013514,  # cls loss fraction
#        'conf': 0.86486,  # conf loss fraction
#        'iou_t': 0.1,  # iou target-anchor training threshold
#        'lr0': 0.001,  # initial learning rate
#        'lrf': -3.,  # final learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.9,  # SGD momentum
#        'weight_decay': 0.0005,  # optimizer weight decay
#        }


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
                     # batch_size = 64 ,则117263张图片需要 117263/64 = 1832.234375个batch_size
                     # 也就是说一个epoch需要1832.234375个batch_size
                     # 273个epoch需要1832.234375 * 273 = 500199.984375 个batch_size
        batch_size=16,
        accumulate=1, # 积累  逐渐增长
        multi_scale=False,  # False 表示一个batch中的图片不允许多scale（320-608）
        freeze_backbone=False,
        transfer=False  # Transfer learning (train only YOLO layers)
):
    init_seeds()
    weights = 'weights' + os.sep  # windows下  os.sep代表 \\,  weights = 'weights\\'
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()

    # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    # 如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率
    # 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    torch.backends.cudnn.benchmark = True  # 增加程序的运行效率

    # Configure run
    data_dict = parse_data_cfg(data_cfg)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg, img_size).to(device)

    # Optimizer优化器
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    # epochs代表训练需要的epoch总数，x代表当前是第几个epoch
    lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # torch.optim.lr_scheduler 提供了几种方法来根据epoches的数量调整学习率。
    # 第一个函数参数是我们的原始优化器（例如SGD等）
    # lr_lambda是一个用于计算学习率的Lambda函数
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch= - 1)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')

    # 是否导入预训练的model，resume  true  导入，False  不导入
    if resume:  # Load previously saved model
        pass
    else:  # Initialize model with backbone (optional)
        cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

    # Dataset 读取图片和标签
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size, augment=True)
    # print(dataset) → <utils.datasets.LoadImagesAndLabels object at 0x7f225b75a780>

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=False,  # False 不打乱数据的排序
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
    # dataloader → torch.from_numpy(img), labels_out, img_path, (h, w)

    # print(dataloader) → <torch.utils.data.dataloader.DataLoader object at 0x7f7a5164fd68>

    # Start training
    t, t0 = time.time(), time.time()

    # 这是个什么神奇用法？？？？
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    # model_info(model)
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches

    for epoch in range(start_epoch, epochs):
        # model.train() tells your model that you are training the model
        model.train() # 让model变成训练模式
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # scheduler.step()是对lr进行调整，非常重要
        scheduler.step()

        mloss = torch.zeros(5).to(device)  # mean losses
        # dataloader → torch.from_numpy(img), labels_out, img_path, (h, w)
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            # print(imgs.shape) → torch.Size([16, 3, 416, 416])
            # print(targets.shape) → torch.Size([139, 6])  139  不固定  是一个batch中物体的数量
            # 6的内容
            # 0 标记的物体在一个batch中的图片编号 0-15
            #
            imgs = imgs.to(device)
            targets = targets.to(device)# targets的详细见自己的CSDN博客
            nt = len(targets)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                # 对burn_in而言,实际 learning\_rate = lr*(batch/burn\_in)**4 ,
                # lr为预设的0.001,burn_in就是1000,batch就是当前的batch数.
                # 这1000主要为了稳定网络.
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                # optimizer通过param_group来管理参数组.param_group中保存了参数组及其对应的学习率,动量等等.
                # 所以我们可以通过更改param_group[‘lr’]的值来更改对应参数组的学习率
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs) # print(len(pred))
                               # print(pred[0].shape) → torch.Size([16, 3, 13, 13, 85])
                               # print(pred[1].shape) → torch.Size([16, 3, 26, 26, 85])
                               # print(pred[2].shape) → torch.Size([16, 3, 52, 52, 85])

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            # loss.backward()获得所有parameter的gradient。
            loss.backward()
            # 然后optimizer存了这些parameter的指针，step()根据这些parameter的gradient对parameter的值进行更新
            # Accumulate gradient for x batches before optimizing
            optimizer.step()       # 更新模型
            optimizer.zero_grad()  # 所有参数的梯度清零

            # 更新跟踪指标的平均值
            mloss = (mloss * i + loss_items) / (i + 1)

            # Print batch results
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, nb - 1), *mloss, nt, time.time() - t)
            t = time.time()
            print(s)

        # 至此，一个epoch结束
        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 5)) or epoch == epochs - 1:
            with torch.no_grad():
                results = test.test(cfg, data_cfg, batch_size=batch_size, img_size=img_size, model=model,
                                    conf_thres=0.1)

        # Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss

        # Save training results
        save = (not opt.nosave) or (epoch == epochs - 1)
        if save:
            # 保存模型参数，优化器参数以及epoch
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == test_loss:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    dt = (time.time() - t0) / 3600
    print('%g epochs completed in %.3f hours.' % (epoch - start_epoch, dt))
    return results


def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    with open('evolve.txt', 'a') as f:
        f.write(c + b + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=2, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='run hyperparameter evolution')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    # Train
    results = train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume or opt.transfer,
        transfer=opt.transfer,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
    )

    # Evolve hyperparameters (optional)
    if opt.evolve:
        best_fitness = results[2]  # use mAP for fitness

        # Write mutation results
        print_mutation(hyp, results)

        gen = 50  # generations to evolve
        for _ in range(gen):

            # Mutate hyperparameters
            old_hyp = hyp.copy()
            init_seeds(seed=int(time.time()))
            s = [.2, .2, .2, .2, .2, .3, .2, .2, .02, .3]
            for i, k in enumerate(hyp.keys()):
                x = (np.random.randn(1) * s[i] + 1) ** 1.1  # plt.hist(x.ravel(), 100)
                hyp[k] = hyp[k] * float(x)  # vary by about 30% 1sigma

            # Clip to limits
            keys = ['iou_t', 'momentum', 'weight_decay']
            limits = [(0, 0.90), (0.80, 0.95), (0, 0.01)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Normalize loss components (sum to 1)
            keys = ['xy', 'wh', 'cls', 'conf']
            s = sum([v for k, v in hyp.items() if k in keys])
            for k in keys:
                hyp[k] /= s

            # Determine mutation fitness
            results = train(
                opt.cfg,
                opt.data_cfg,
                img_size=opt.img_size,
                resume=opt.resume or opt.transfer,
                transfer=opt.transfer,
                epochs=opt.epochs,
                batch_size=opt.batch_size,
                accumulate=opt.accumulate,
                multi_scale=opt.multi_scale,
            )
            mutation_fitness = results[2]

            # Write mutation results
            print_mutation(hyp, results)

            # Update hyperparameters if fitness improved
            if mutation_fitness > best_fitness:
                # Fitness improved!
                print('Fitness improved!')
                best_fitness = mutation_fitness
            else:
                hyp = old_hyp.copy()  # reset hyp to

            # # Plot results
            # import numpy as np
            # import matplotlib.pyplot as plt
            #
            # a = np.loadtxt('evolve.txt')
            # x = a[:, 3]
            # fig = plt.figure(figsize=(14, 7))
            # for i in range(1, 10):
            #     plt.subplot(2, 5, i)
            #     plt.plot(x, a[:, i + 5], '.')
