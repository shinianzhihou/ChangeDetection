import torch.optim.lr_scheduler as lr_schedulers

def build_scheduler(cfg,optimizer, choice='', **kwargs):
    scfg = cfg.build.lr_scheduler
    choice = choice if choice else scfg.choice

    if hasattr(lr_scheduler, choice):
        lr_scheduler = getattr(lr_schedulers, choice)(optimizer, **kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return lr_scheduler

