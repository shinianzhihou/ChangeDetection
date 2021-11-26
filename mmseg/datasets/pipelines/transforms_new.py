import albumentations as A
from albumentations import DualTransform
from albumentations.pytorch import ToTensorV2
from numpy import random

from ..builder import PIPELINES

PIPELINES.register_module(module=ToTensorV2)
