import argparse
import os
import random

import albumentations as A
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from build import *
from utils import Metric, ModelEma


def run(args):

    rank = args.local_rank
    bs = args.bs
    nw = args.workers
    lr = args.lr
    pv = args.period_val
    pp = args.period_print
    epochs = args.epochs
    work_dir = args.work_dir
    ema = args.ema
    decay_ema = args.decay_ema

    os.system(f"mkdir -p {work_dir}")

    device = torch.device(
        f'cuda:{max(rank,0)}' if torch.cuda.is_available() else 'cpu')

    # model
    model = build_model(choice='cdp_UnetPlusPlus', encoder_name="timm-efficientnet-b0",
                        encoder_weights="noisy-student",
                        in_channels=3,
                        classes=2,
                        siam_encoder=True,
                        fusion_form='concat',)
    model = model.to(device)



    # TODO(shinian) load checkpoint
    # TODO(shinian) ema model

    if ema:
        model_ema = ModelEma(model, decay=decay_ema)
    

    if rank > -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],
                                                          output_device=rank, find_unused_parameters=True)

    if rank < 1:
        logger.add(os.path.join(work_dir,"train.log"),serialize=True)
        logger.info(args)
        logger.info(model)

    # pipeline
    train_pipeline = A.Compose([
        A.RandomResizedCrop(width=384, height=384),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),],
        additional_targets={'image1': 'image'})
    val_pipeline = A.Compose([
        A.HorizontalFlip(p=0.0),],
        additional_targets={'image1': 'image'})

    # dataloader
    train_set = build_dataset(choice='CommonDataset',
                              metafile="/Users/shinian/proj/data/stb/train.debug.txt",
                              data_root="/Users/shinian/proj/data/stb/",
                              pipeline=train_pipeline,

                              )
    val_set = build_dataset(choice='CommonDataset',
                            metafile="/Users/shinian/proj/data/stb/val.debug.txt",
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
    optimizer = build_optimizer(
        model, choice='Adam', lr=lr, weight_decay=0)

    # lr_scheduler
    # lr_scheduler = build_scheduler(optimizer, choice='MultiStepLR',
    #                                milestones=[int(epochs*0.6), int(epochs*0.85)], gamma=0.1)
    lr_scheduler = build_scheduler(optimizer, choice='CosineAnnealingLR',
                                   T_max=epochs+1, eta_min=1e-5)

    # metric
    metric_train = Metric(init_metric={'f1': 0.0, 'iou': 0.0})
    metric_val = Metric(init_metric={'f1': 0.0, 'iou': 0.0})
    if ema:
        metric_val_ema = Metric(init_metric={'f1': 0.0, 'iou': 0.0})

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
            if ema:
                model_ema.update(model)

            output = pred.argmax(dim=1)
            metric_train(output, label)

            # FIXME(shinian) gather output/loss/metric on different ranks in DDP
            if ((batch+1) % pp == 0 or (batch+1) == len(train_loader)) and rank < 1:
                local = batch != len(train_loader)-1
                state = "T" if local else "E"
                pstr = f"[{state}] e:{epoch+1:2d}/{epochs:2d} | b:{batch+1:3d}/{len(train_loader)} | " + \
                       f"lr: {lr_scheduler.get_last_lr()[0]:.3e} | " + \
                       f"{metric_train.print(local)} | " + \
                       f"loss:{loss.item():.3e}"
                # print(pstr)
                logger.info(pstr)

        lr_scheduler.step()
        metric_train.calculate(local=False)

        # TODO(shinian) perform validation

        if ((epoch+1) % pv == 0 or (epoch+1) == epochs) and rank < 1:
            save_dict = {}
            local = False
            res, loss = validate(
                model, val_loader, criterion, metric_val, device)
            pstr =  f"[V] e:{epoch+1:2d}/{epochs:2d} | b:{len(val_loader):3d}/{len(val_loader)} | " + \
                    f"lr: {lr_scheduler.get_last_lr()[0]:.3e} | " + \
                    f"{metric_val.print(local)} | " + \
                    f"loss:{loss.item():.3e}"
            logger.info(pstr)
            save_dict['state_dict'] = (model.module if hasattr(model, 'module') else model).state_dict()
            save_name = f"epoch_{epoch+1}_{metric_val.print(local,sep0='_',sep1='_')}"
            
            if ema:
                res, loss = validate(
                model_ema.module, val_loader, criterion, metric_val_ema, device)
                pstr =  f"[A] e:{epoch+1:2d}/{epochs:2d} | b:{len(val_loader):3d}/{len(val_loader)} | " + \
                        f"lr: {lr_scheduler.get_last_lr()[0]:.3e} | " + \
                        f"{metric_val_ema.print(local)} | " + \
                        f"loss:{loss.item():.3e}"
                logger.info(pstr)
                save_dict['state_dict_ema'] = (model_ema.module if hasattr(model_ema, 'module') else model_ema).state_dict()
                save_name += f"_{metric_val.print(local,sep0='_',sep1='_')}"

            save_name += ".pth"
            save_path = os.path.join(work_dir, save_name)
            torch.save(save_dict, save_path)
            
    return


@torch.no_grad()
def validate(model, val_loader, criterion, metric_val, device):

    model.eval()
    loss_all = 0.0
    for batch, data in enumerate(val_loader):
        img1 = data[0].to(device)
        img2 = data[1].to(device)
        label = data[2].to(device)
        pred = model(img1, img2)
        loss_all += criterion(pred, label)
        output = pred.argmax(dim=1)
        metric_val(output, label)
    loss = loss_all / len(val_loader)
    res = metric_val.calculate(local=False)
    return res, loss


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
    parser.add_argument('-wd', '--work_dir',  type=str, default='../work_dirs/',
                        help='work dirs to save logs and ckpts')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use exponential moving average while training')
    parser.add_argument('-de','--decay_ema', type=float, default=0.9998,
                        help='decay factor for ema')
    parser.add_argument('--local_rank', type=int,
                        default=-1, help='local rank for ddp')

    args = parser.parse_args()

    init(args)
    run(args)


if __name__ == '__main__':
    main()
