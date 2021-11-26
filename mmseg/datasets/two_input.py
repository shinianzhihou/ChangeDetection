
import os
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from PIL import Image

from .builder import DATASETS, build_pipeline



class BaseDataset(Dataset):

    CLASSES = ["unchanged","changed"]

    PALETTE = [[0,0,0],[255,255,255]]

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def process(self, image0, image1, mask):
        # TODO(snian) Also make the number of masks configurable
        if self.pipeline:
            augmented = self.pipeline(image=image0,image1=image1, mask=mask)
            image0 = augmented['image']
            image1 = augmented['image1']
            mask = augmented['mask']
        return image0, image1, mask


@DATASETS.register_module()
class TwoInputDataset(BaseDataset):
    """A general dataset class for two input.

    meta_file:
        image_000_a.jpg image_000_b.jpg gt_000.png
        image_001_a.jpg image_001_b.jpg gt_001.png
        image_002_a.jpg image_002_b.jpg gt_002.png
        ...
    
    Args:
        meta_file (str): Path to meta_file.
        pipeline (list[dict]): Processing pipeline.
        data_root (str): Data root for images.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        c255_t1_in_mask (bool): If True, convert 255 to 1 in mask

    """
    def __init__(self,
                 meta_file,
                 data_root='',
                 pipeline=None,
                 test_mode=False,
                 sep=' ',
                 imdecode_backend='cv2',
                 stack_images=True,
                 c255_t1_in_mask=True,
                ):
        super().__init__()
        self.data_root = data_root
        self.pipeline = build_pipeline(pipeline)
        self.test_mode = test_mode
        self.data = open(meta_file).readlines()
        self.length = len(self.data)
        self.sep = sep
        self.imdecode_backend = imdecode_backend
        self.stack_images = stack_images
        self.c255_t1_in_mask = c255_t1_in_mask
    
    def read_images(self,paths,choice='cv2'):
        if choice=='cv2':
            images = [cv2.imread(path,-1) for path in paths]
            images[:2] = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images[:2]]
        elif choice=='pillow':
            images = [np.array(Image.open(path)) for path in paths]
        else:
            raise NotImplementedError(f"{choice} not in cv2 or pillow !!!")
             
        if self.c255_t1_in_mask and images[-1].max()>1:
            images[-1] = images[-1]/255
        images[-1] = images[-1].astype('int64')

        return images

    def _post_process(self,image0,image1,mask):
        if self.stack_images:
            return{
                'img':torch.stack([image0,image1],dim=0),
                'img_metas':{
#                     'ori_shape':(224, 224, 3),
#                     'img_shape':(224, 224, 3),
#                     'pad_shape':(224, 224, 3),
                    },
                'gt_semantic_seg':mask,
            }
            # return torch.stack([image0,image1],dim=0), mask
        else:
            return image0,image1,mask


    def __getitem__(self,idx):
        paths = self.data[idx].strip().split(self.sep)
        image0,image1,mask = self.read_images(
            [os.path.join(self.data_root,path) for path in paths],choice=self.imdecode_backend)
        image0,image1,mask = self.process(image0,image1,mask)

        return self._post_process(image0,image1,mask)

    def __len__(self):
        return self.length

    
