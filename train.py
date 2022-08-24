import argparse
import random

import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import Metric
from build import *


def run(args):

    rank = args.local_rank
    bs = args.bs
    nw = args.workers
    lr = args.lr
    epochs = args.epochs
    pv = args.period_val
    pp = args.period_print

    device = torch.device(
        f'cuda:{max(rank,0)}' if torch.cuda.is_available() else 'cpu')

    # model
    model = build_model(choice='cdp_Unet', encoder_name="resnet34",
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=2,
                        siam_encoder=True,
                        fusion_form='concat',)
    model = model.to(device)

    # TODO(shinian) load checkpoint
    # TODO(shinian) ema model
    if rank > -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],
                                                          output_device=rank, find_unused_parameters=False)

    # pipeline
    train_pipeline = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2), ],
        additional_targets={'image1': 'image'}
    )
    val_pipeline = None

    # dataloader
    train_set = build_dataset(choice='CommonDataset',
                              metafile="/Users/shinian/proj/data/stb/train.txt",
                              data_root="/Users/shinian/proj/data/stb/",
                              pipeline=train_pipeline,

                              )
    val_set = build_dataset(choice='CommonDataset',
                            metafile="/Users/shinian/proj/data/stb/val.txt",
                            data_root="/Users/shinian/proj/data/stb/",
                            pipeline=val_pipeline,)

    train_sampler = DistributedSampler(train_set) if rank > -1 else None

    train_loader = DataLoader(dataset=train_set,
                              pin_memory=True,
                              batch_size=bs,
                              num_workers=nw,
                              shuffle=False,
                              drop_last=True,
                              sampler=train_sampler)

    val_loader = DataLoader(dataset=val_set,
                            pin_memory=True,
                            batch_size=bs,
                            num_workers=nw,
                            shuffle=False,
                            drop_last=False,
                            sampler=None)

    # loss
    criterion = build_loss(choice='CrossEntropyLoss')

    # optim
    optimizer = build_optimizer(model, choice='Adam', lr=lr, weight_decay=0.0)

    # lr_scheduler
    lr_scheduler = build_scheduler(optimizer, choice='MultiStepLR',
                                   milestones=[int(epochs*0.6), int(epochs*0.85)], gamma=0.1)

    # metric
    metric_train = Metric(init_metric={'f1': 0.0, 'iou': 0.0})
    metric_val = Metric(init_metric={'f1': 0.0, 'iou': 0.0})

    # training
    for epoch in range(epochs):

        if rank > -1:
            train_sampler.set_epoch(epoch)

        metric_train.reset()

        model.train()
        for batch, data in enumerate(train_loader):
            img1 = data[0].to(device)
            img2 = data[1].to(device)
            label = data[2].to(device)

            optimizer.zero_grad()
            pred = model(img1, img2)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            output = pred.argmax(dim=1)
            metric_train(output, label)

            # FIXME(shinian) gather output/loss/metric on different ranks in DDP
            if batch > 0 and batch % pp == 0 and rank < 1:
                print(f"e:{epoch:2d}/{epochs:2d} | b:{batch:3d}/{len(train_loader)} | " +
                      f"loss:{loss.item():.3e} | {str(metric_train)}")

        lr_scheduler.step()
        metric_train.calculate(local=False)

        # TODO(shinian) perform validation

    return


def init(args):
    seed = args.seed
    rank = args.local_rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not (rank > -1):
        return

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',)

    if torch.distributed.get_rank() == 0:
        pass  # TODO(shinian) do something only in master rannk

    return


def main():
    parser = argparse.ArgumentParser(
        description='train')

    parser.add_argument('-seed',  type=int, default=1234,
                        help='global random seed')
    parser.add_argument('-bs',  type=int, default=16,
                        help='batch size on each gpu')
    parser.add_argument('-lr',  type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('-nw', '--workers',  type=int, default=2,
                        help='number of workers for each gpu')
    parser.add_argument('-ne', '--epochs',  type=int, default=30,
                        help='number of epochs')
    parser.add_argument('-pv', '--period_val',  type=int, default=1,
                        help='perform validation every period')
    parser.add_argument('-pp', '--period_print',  type=int, default=10,
                        help='print log every period (batch)')

    parser.add_argument('--local_rank', type=int,
                        default=-1, help='local rank for ddp')

    args = parser.parse_args()

    init(args)
    run(args)


if __name__ == '__main__':
    main()
