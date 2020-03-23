import os
import time
from tensorboardX import SummaryWriter

def build_tensorboad(cfg):
    if not cfg.BUILD.USE_TENSORBOARD:
        return None
    tcfg = cfg.TENSORBOARD
    ID = cfg.BUILD.MODEL.CHOICE if tcfg.ID == "" else tcfg.ID
    name = "%s/%s" % ( time.strftime("%Y-%m-%d-%H-%M"),ID)
    path = os.path.join(tcfg.PATH, name)
    writer = SummaryWriter(path)
    writer.add_text("config", str(cfg), 0)
    return writer

