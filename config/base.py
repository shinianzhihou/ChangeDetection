import os
from yacs.config import CfgNode as cfg_node

_c = cfg_node()

## base 
_c.base = cfg_node()


## build 
_c.build = cfg_node()
### dataset
_c.build.dataset = cfg_node()
_c.build.dataset.choice = ""
### loss
_c.build.loss = cfg_node()
_c.build.loss.choice = ""
### model
_c.build.model = cfg_node()
_c.build.model.choice = ""
### lr_scheduler
_c.build.lr_scheduler = cfg_node()
_c.build.lr_scheduler.choice = ""
### optimizer
_c.build.optimizer = cfg_node()
_c.build.optimizer.choice = ""
_c.build.optimizer.base_lr = 1e-3
_c.build.optimizer.weight_decay = 0






