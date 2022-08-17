import argparse
import random

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import OrderedDistributedSampler
from build import *


def run(args):

    rank = args.local_rank
    bs = args.bs
    nw = args.workers
    lr = args.lr
    epochs = args.epochs
    pv = args.period_val

    device = torch.device(
        f'cuda:{max(rank,0)}' if torch.cuda.is_available() else 'cpu')

    # model
    model = build_model(choice='BiT')
    # TODO(shinian) load checkpoint
    # TODO(shinian) ema model
    if rank > -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],
                                                          output_device=rank, find_unused_parameters=False)

    # dataloader
    train_set = build_dataset(choice='LEVIRCD', metafile="")
    val_set = build_dataset(choice='LEVIRCD', metafile="")

    train_sampler = DistributedSampler(train_set) if rank > -1 else None
    val_sampler = OrderedDistributedSampler(val_set) if rank > -1 else None
    
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
                            sampler=val_sampler)

    # loss
    criterion = build_loss(choice='BCEWithLogitsLoss')

    # optim
    optimizer = build_optimizer(model, choice='Adam', lr=lr)

    # lr_scheduler
    lr_scheduler = build_scheduler(optimizer, choice='MultiStepLR',
                                   milestones=[int(epochs*0.6), int(epochs*0.85)], gamma=0.1)

    # training
    for epoch in range(epochs):
        if rank > -1:
            train_sampler.set_epoch(epoch)

        model.train()
        for batch, data in enumerate(train_loader):
            img1 = data[0].to(device)
            img2 = data[1].to(device)
            label = data[2].to(device)

            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # FIXME(shinian) replace it with log
            print(f"{epoch}/{epochs}|{batch}/{len(train_loader)}|{loss.item()}")

        lr_scheduler.step()

        # TODO(shinian) perform validation

    return


def init(args):
    seed = args.seed
    rank = args.local_rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',)

    if torch.distributed.get_rank() == 0:
        pass  # TODO(shinian) do something only in master rannk


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

    parser.add_argument('--local_rank', type=int,
                        default=-1, help='local rank for ddp')

    args = parser.parse_args()

    init(args)
    run(args)


if __name__ == '__main__':
    main()
