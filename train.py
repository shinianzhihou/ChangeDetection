import argparse
import random

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import cfg
from utils import OrderedDistributedSampler,Metric
from build import *


def train(cfg, args):

    rank = args.local_rank
    bs = args.bs
    nw = args.workers
    lr = args.lr
    epochs = args.epochs

    device = torch.device(
        f'cuda:{max(rank,0)}' if torch.cuda.is_available() else 'cpu')

    # model
    model = build_model(cfg, choice='')
    # TODO(shinian) load checkpoint
    # TODO(shinian) ema model
    if rank > -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],
                                                          output_device=rank, find_unused_parameters=False)

    # dataloader
    train_set = build_dataset(cfg, choice='')
    val_set = build_dataset(cfg, choice='')
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
    loss = build_loss(cfg, choice='')

    # optim
    optimizer = build_optimizer(cfg, model, choice='Adam', lr=lr)

    # lr_scheduler
    lr_scheduler = build_scheduler(cfg, optimizer, choice='')

    # training

    for epoch in range(epochs):
        if rank > -1:
            train_sampler.set_epoch(epoch)

        model.train()
        for batch, data in enumerate(train_loader):
            # TODO(shinian) complete



    return


def init_env(cfg, args):

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',)

    if torch.distributed.get_rank() == 0:
        pass  # TODO(shinian) do something only in master rannk


def main():
    parser = argparse.ArgumentParser(
        description='train')

    parser.add_argument('-cfg',  type=str, default='configs/default.yaml',
                        metavar='FILE', help='path to config file')
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

    parser.add_argument('--local_rank', type=int,
                        default=-1, help='local rank for ddp')

    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config options using the command-line')

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    init_env(cfg, args)
    train(cfg, args)


if __name__ == '__main__':
    main()
