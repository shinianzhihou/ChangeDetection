import torch
import torch.nn as nn


from configs import Config
from utils import logging

cf = Config()
test_batch = 100 # 100 or 200
test_epoch = 100
