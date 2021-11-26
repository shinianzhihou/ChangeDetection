"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES, build_loss



@LOSSES.register_module()
class MultiLoss(nn.Module):
    """MultiLoss
     This loss consists of multi loss with different weights.
     
     Args:
         losses (list): A list of configs for different losses.
         weights (list[float]): The weights for different losses.
    """
    def __init__(self,losses,weights=None):
        super().__init__()
        assert losses, f"{losses} cannot be None or empty." 
        self.losses = [build_loss(_) for _ in losses]
        self.weights = weights if weights else [1.0/len(losses) for i in losses]
    
    def forward(self,pred,target,**kwargs):
        weights = self.weights
        for idx,loss in enumerate(self.losses):
            if idx == 0:
                res = weights[idx] * loss(pred,target,**kwargs)
            else:
                res += weights[idx] * loss(pred,target,**kwargs)
        return res
                
