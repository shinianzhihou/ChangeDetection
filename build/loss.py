import torch
from torch.nn import BCELoss, BCEWithLogitsLoss

# TODO(SNian) : add focal loss

def build_loss(cfg):
    lcfg = cfg.BUILD.LOSS

    loss_map = {
        "BCELoss" : BCELoss(reduction=lcfg.REDUCTION),
        "BCEWithLogitsLoss" : BCEWithLogitsLoss(reduction=lcfg.REDUCTION,\
            pos_weight=get_binary_weight(cfg))
    }

    assert lcfg.CHOICE in loss_map.keys()

    return loss_map[lcfg.CHOICE]

def get_binary_weight(cfg):
    device = cfg.MODEL.DEVICE
    lcfg = cfg.BUILD.LOSS
    value = lcfg.POS_WEIGHT

    if not lcfg.USE_POS_WEIGHT:
        return None
    elif isinstance(value,(int,float)):
        pos_weight = [1/lcfg.POS_WEIGHT, lcfg.POS_WEIGHT]
    elif isinstance(value, list):
        pos_weight = value
    else:
        raise TypeError("`int`, `float` or `list` for `pos_weight`")

    return torch.tensor(pos_weight).view(2, 1, 1).to(device)
