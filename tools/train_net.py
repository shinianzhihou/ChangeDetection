import argparse

from configs import cfg

from build.dataloader import build_dataloader
from build.model import build_model
from build.loss import build_loss
from utils.configs import States


def run_train(cfg):
    states = States(cfg)

    train_loader = build_dataloader(cfg)
    if cfg.SOLVER.TEST_WHEN_TRAIN:
        train_loader = build_dataloader(cfg,test=True)

    model = build_model(cfg)
    criterion = build_loss(cfg)
    # TODO(SNain) : Complete `build_optimizer` and `lr_scheduler`
    # optimizer = build_optimizer(cfg)
    # scheduler = build_scheduler(cfg)

    

    num_batch = train_loader.__len__()



def main():
    parser = argparse.ArgumentParser(
        description="easy2train for Change Detection")

    parser.add_argument(
        "-cfgr",
        "--config_root",
        default="configs/heterogeneous_change_detection",
        help="Path to config file root",
        type=str
    )

    parser.add_argument(
        "-dcfg",
        "--default_config_file",
        default="default.yaml",
        help="Path to default config file",
        type=str,
    )

    parser.add_argument(
        "-cfg",
        "--config_file",
        default="ca_fl_siamese_unet.yaml",
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
    yaml_default_file = os.path.join(
        args.config_root, args.default_config_file)
    yaml_file = os.path.join(args.config_root, args.config_file)
    cfg.merge_from_file(yaml_default_file)
    cfg.merge_from_file(yaml_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)
    run_train(cfg)


if __name__ == "__main__":
    main()
