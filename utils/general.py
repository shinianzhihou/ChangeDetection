from collections import defaultdict

import torch

def merge_kwargs_into_cfg(cfg,kwargs):
    ret = defaultdict()
    for k,v in cfg.items():
        ret[k] = v
    for k,v in kwargs.items():
        ret[k] = v
    return ret

def load_checkpoint(path, ret_state):
    ckpt = torch.load(path,map_location='cpu')
    for item,state in ckpt.items():
        if item in ret_state:
            new_state = {k.replace('module.',''):v for k,v in state.items() \
                    # if 'sam12.conv3.' not in kk\
                    # and 'sam23.conv3.' not in kk 
                    } if item=='state_dict' else state
            ret_state[item].load_state_dict(new_state,strict=False)
            print(f"loaded {item} in {path}.")
        else:
            print(f"extra {item} in {path}.")
    return ret_state
