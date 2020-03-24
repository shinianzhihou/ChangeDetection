import os
import argparse

from configs import cfg

from build import (
    build_dataloader,
    build_model,
    build_loss,
    build_optimizer,
    build_scheduler,
    build_tensorboad,
    build_checkpoint
)
from engine.trainer import train_epoch
from utils.states import States


def run_train(cfg):
    states = States(cfg)

    train_loader = build_dataloader(cfg)
    test_loader = build_dataloader(cfg, test=True) if cfg.BUILD.TEST_WHEN_TRAIN else None
    model = build_model(cfg).to(cfg.MODEL.DEVICE)
    optimizer = build_optimizer(cfg, model)
    model,optimizer,states = build_checkpoint(cfg, model, optimizer,states)
    criterion = build_loss(cfg)
    scheduler = build_scheduler(cfg, optimizer, max_iters=train_loader.__len__())
    writer = build_tensorboad(cfg)

    for epoch in range(cfg.SOLVER.NUM_EPOCH):
        states.update("current_epoch",epoch)
        train_epoch(
            cfg,
            states,
            train_loader,
            model,
            optimizer,
            criterion,
            scheduler,
            writer,
            test_loader,
        )
        



def main():
    parser = argparse.ArgumentParser(
        description="easy2train for Change Detection")

    parser.add_argument(
        "-cfg",
        "--config_file",
        default="configs/homo/default.yaml",
        metavar="FILE",
        help="Path to config file",
        type=str,
    )

    parser.add_argument(
        "-se",
        "--skip_eval",
        help="Do not eval the models(checkpoints)",
        action="store_true",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    run_train(cfg)


if __name__ == "__main__":
    main()
