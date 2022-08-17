import torch.nn as nn

import loss as mylosses


def build_loss(choice, **kwargs):
    
    if hasattr(mylosses,choice):
        loss = getattr(mylosses, choice)(**kwargs)
    elif hasattr(nn,choice):
        loss = getattr(nn, choice)(**kwargs)    
    else:
        raise NotImplementedError(f"{choice}")

    return loss
