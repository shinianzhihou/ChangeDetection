import torch.optim.lr_scheduler as lr_scheduler

def build_scheduler(cfg,optimizer, choice='', **kwargs):
    scfg = cfg.build.lr_scheduler
    choice = choice if choice else scfg.choice

    if hasattr(lr_scheduler, choice):
        scheduler = getattr(lr_scheduler, choice)(optimizer, **kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return scheduler

