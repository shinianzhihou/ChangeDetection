from torch.optim import SGD

#TODO(SNian) : add some optimizers, sunch as Adam...

def build_optimizer(cfg, model):
    ocfg = cfg.BUILD.OPTIMIZER

    optimizer_map = {
        "SGD": SGD(
            params=model.parameters(),
            lr = cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        ),
    }

    return optimizer_map[ocfg.CHOICE]