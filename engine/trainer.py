import argparse
import os
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

from configs import cfg
from data import Lab
from model import SiameseUnet_conc, FLSiameseUnet_conc, CAFLSiameseUnet_conc
from solver.transforms import *
from utils.checkpoints import lcwo, scwo
from utils.configs import States
from utils.metric import get_metric, update_metric
from utils.eval import eval_model


def train_epoch(
    cfg,
    states,
    train_loader,
    model,
    optimizer,
    criterion,
    scheduler,
    writer,
    test_loader=None
    ):

    curren_epoch = states.curren_epoch
    num_epoch = states.num_epoch

    num_batch = train_loader.__len__()
    device = states.device

    bar_format = '{desc}{percentage:3.0f}%|[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(train_loader,bar_format=bar_format)

    for batch, data in enumerate(train_loader):
        states.step("current_batch")
        current_batch = states.current_batch

        model.train()
        img1, img2, gt = [img.to(device) for img in data]
        optimizer.zero_grad()
        output, floss = model(img1, img2)
        bceloss = criterion(output, gt)
        loss = bceloss+floss if states.add_floss else bceloss
        loss.backward()
        optimizer.step()
        scheduler.step()

        print_str = " | ".join([
            "feature loss:%5s" % (states.add_floss),
            "epoch:%3d/%3d" % (curren_epoch, num_epoch),
            "batch:%3d/%3d" % (batch, num_batch),
            "loss:%.3f" % (loss.item()),
            "bceloss:%.3f" % (bceloss.item()),
            "floss:%.3e" % (floss.item()),
        ])
        pbar.set_description(print_str, refresh=True)

        if cfg.TENSORBOARD.USE_TENSORBOARD:
            prefix = "train/"
            scalar = {
                "bceloss:" : bceloss.item(),
                "floss": floss.item(),
                "loss": loss.item(),
                "lr": scheduler.get_lr()[0],
            }

            images = {
                "image1": img1,
                "image2": img2,
                "gt": gt[:, 1:2, :, :],
                "output": torch.argmax(output, dim=1, keepdim=True)
            }

            for key, value in scalar.items():
                writer.add_scalar(prefix+key, value, current_batch)

            if not current_batch % cfg.SOLVER.IMAGE_EVERY:
                step = current_batch // cfg.SOLVER.IMAGE_EVERY
                for key, value in images.items():
                    writer.add_images(prefix+key, value, step)
                    
            if not current_batch % cfg.SOLVER.METRIC_EVERY:
                step = current_batch // cfg.SOLVER.METRIC_EVERY
                out_tensor = torch.argmax(output, dim=1, keepdim=True).type_as(output)
                gt_tensor = gt[:,1,:,:]
                metric = get_metric(out_tensor,gt_tensor)
                metric_values = [metric.get(key).item() for key in cfg.EVAL.METRIC]
                metric_scalar = dict(zip(cfg.EVAL.METRIC,metric_values))
                writer.add_scalars("train/metric",metric_scalar,step)
                
            if not current_batch % cfg.SOLVER.TEST_EVERY and test_loader:
                step = current_batch // cfg.SOLVER.TEST_EVERY
                metric = eval_model(model,test_loader,cfg,writer,step)
                num_update,metric = update_metric(states.best_metric,metric)
                if num_update > 0:
                    states.update("best_metric",metric)
                    cp_name = "%s_%s_epoch_%d_batch_%d.pt" % (
                        time.strftime("%Y-%m-%d-%H-%M"),
                        model._get_name(),
                        curren_epoch,
                        batch
                    )
                    cp_path = os.path.join(cfg.SOLVER.CHECKPOINT_PATH, cp_name)
                    scwo(cp_path, model, optimizer)
                

        if cfg.SOLVER.USE_CHECKPOINT:
            if not current_batch % cfg.SOLVER.CHECKPOINT_PERIOD:
                cp_name = "%s_%s_epoch_%d_batch_%d.pt" % (
                    time.strftime("%Y-%m-%d-%H-%M"),
                    model._get_name(),
                    curren_epoch,
                    batch
                )
                cp_path = os.path.join(cfg.SOLVER.CHECKPOINT_PATH, cp_name)
                scwo(cp_path, model, optimizer)