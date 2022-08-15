import torch.nn as nn

import loss as mylosses


def build_loss(cfg, choice='', **kwargs):
    lcfg = cfg.build.loss
    choice = choice if choice else lcfg.choice
    
    if hasattr(mylosses,choice):
        loss = getattr(mylosses, choice)(**kwargs)
    elif hasattr(nn,choice):
        loss = getattr(nn, choice)(**kwargs)    
    else:
        raise NotImplementedError(f"{choice}")

    return loss
