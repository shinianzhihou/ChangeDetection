from torch.optim.lr_scheduler import CosineAnnealingLR

from solver.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR

# TODO(SNian) : add WarmupMultiStepLR...

def build_scheduler(cfg, optimizer,max_iters=1000):
    scfg = cfg.BUILD.LR_SCHEDULER
    sscfg = cfg.SOLVER.LR_SCHEDULER

    scheduler_map = {
        "WarmupCosineLR": get_wclr(sscfg,optimizer,max_iters)
    }

    assert scfg.CHOICE in scheduler_map.keys()

    return scheduler_map[scfg.CHOICE]

def get_wclr(sscfg, optimizer, max_iters):
    assert isinstance(max_iters,int)
    return WarmupCosineLR(
        optimizer=optimizer,
        max_iters=max_iters,
        warmup_factor=sscfg.WARMUP_FACTOR,
        warmup_iters=sscfg.WARMUP_ITERS,
        warmup_method=sscfg.WARMUP_METHOD,
        last_epoch=sscfg.LAST_EPOCH,
    )