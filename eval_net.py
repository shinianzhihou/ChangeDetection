import os
import argparse

import pandas as pd

from configs import cfg
from utils.checkpoints import lcwo
from utils.eval import eval_model
from build import (
    build_dataloader,
    build_model,

)

def run_eval(cfg,save_max_imgs=False):
    test_loader = build_dataloader(cfg, test=True)
    model = build_model(cfg).to(cfg.MODEL.DEVICE)
    cp_paths = get_cp_paths(cfg)
    test_res = pd.DataFrame(columns=["checkpoint"]+cfg.EVAL.METRIC)
    for idx,cp_path in enumerate(cp_paths):
        if model._get_name() not in cp_path:
            continue
        model = lcwo(cp_path,model)
        metric = eval_model(model,test_loader,cfg)
        save_value = [cp_path] + [v.item() for k,v in metric.items()]
        test_res.loc[test_res.shape[0]] = save_value

    test_res.to_csv(os.path.join(cfg.EVAL.SAVE_PATH,cfg.EVAL.SAVE_NAME),index=False)

    if cfg.EVAL.SAVE_IMAGES:
        max_checkpoint = test_res.loc[test_res[cfg.EVAL.SAVE_BY_METRIC].idxmax()].to_dict()
        model = lcwo(max_checkpoint["checkpoint"],model)
        eval_model(model,test_loader,cfg,save_imgs=True)
    return test_res

def get_cp_paths(cfg):
    cp_root = cfg.EVAL.CHECKPOINTS_PATH
    cp_names = os.listdir(cfg.EVAL.CHECKPOINTS_PATH)
    cp_paths = [os.path.join(cp_root,cp_name) for cp_name in cp_names]
    
    return cp_paths

def main():

    parser = argparse.ArgumentParser(
        description="eval models from checkpoints.")

    parser.add_argument(
        "-cfg",
        "--config_file",
        default="configs/homo/default.yaml",
        metavar="FILE",
        help="Path to config file",
        type=str,
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
    run_eval(cfg)


if __name__ == "__main__":
    main()