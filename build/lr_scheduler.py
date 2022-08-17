import torch.optim.lr_scheduler as lr_schedulers

def build_scheduler(optimizer, choice, **kwargs):

    if hasattr(lr_scheduler, choice):
        lr_scheduler = getattr(lr_schedulers, choice)(optimizer, **kwargs)
    else:
        raise NotImplementedError(f"{choice}")

    return lr_scheduler

