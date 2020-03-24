import os
import time

import torch
from tqdm import tqdm

from utils.checkpoints import lcwo, scwo
from utils.eval import eval_model
from utils.metric import get_metric, update_metric


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

    bar_format = '{desc}{percentage:3.0f}%|[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(train_loader,bar_format=bar_format)

    for batch, data in enumerate(pbar):
        states.step("current_batch")
        current_batch = states.current_batch

        model.train()
        img1, img2, gt = [img.to(cfg.MODEL.DEVICE) for img in data]
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        scheduler.step()

        print_str = " | ".join([
            "epoch:%3d/%3d" % (states.current_epoch, cfg.SOLVER.NUM_EPOCH),
            "batch:%3d/%3d" % (batch, train_loader.__len__()),
            "loss:%.3f" % (loss.item()),
        ])
        pbar.set_description(print_str, refresh=True)

        # TODO(SNian) : make it optional 
        if writer:
            writer.add_scalar("train/loss", loss.item(), current_batch)
            writer.add_scalar("train/lr", scheduler.get_lr()[0], current_batch)



        if test_loader and not current_batch % cfg.SOLVER.TEST_PERIOD:
            step = current_batch // cfg.SOLVER.TEST_PERIOD
            del img1,img2,gt
            metric = eval_model(model,test_loader,cfg,writer,step)
            writer.add_scalars("train/metric/",metric,step)
            if cfg.SOLVER.TEST_BETTER_SAVE:
                num_update,metric = update_metric(states.best_metric,metric)
                if num_update > 0:
                    states.update("best_metric",metric)
                    scwo_epoch_batch(states.current_epoch,batch,cfg.CHECKPOINT.PATH,model,optimizer)

        if cfg.BUILD.USE_CHECKPOINT and not current_batch % cfg.CHECKPOINT.PERIOD:
            scwo_epoch_batch(states.current_epoch,batch,cfg.CHECKPOINT.PATH,model,optimizer)


def scwo_epoch_batch(epoch,batch,root,model,optimizer):
    '''Save checkpoint with specific name.'''
    cp_name = "%s_%s_epoch_%d_batch_%d.pt" % (
        time.strftime("%Y-%m-%d-%H-%M"),
        model._get_name(),
        epoch,
        batch
    )
    scwo(os.path.join(root, cp_name), model, optimizer)
