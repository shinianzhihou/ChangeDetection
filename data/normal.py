import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from solver.transforms import *


class Normal(Dataset):
    def __init__(self, csv_path, data_trans=None, mask=True, test=False):
        super(Normal, self).__init__()
        self.data = pd.read_csv(csv_path)
        self.data_trans = data_trans
        self.mask = mask
        self.test = test

    def __getitem__(self, index):
        path = self.data.loc[index]
        img_list = [cv2.imread(path["img1"]), cv2.imread(
            path["img2"]), cv2.imread(path["gt"])]
        img_list[-1] = img_list[-1][:, :, 0:1]  # ground-truth
        if self.data_trans is None:
            tensor_list = self._test_trans(img_list) if self.test else self._transforms(img_list)
        else:
            tensor_list = self.data_trans(img_list)

        return self._operater(tensor_list)

    def __len__(self):
        return len(self.data)

    def _transforms(self, imgs):
        data_trans = transforms.Compose([
            ToPILImage(),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                # RandomRotation(180,expand=False,fill=0),
                RandomRotation(180, expand=False),
            ],
                p=0.8),
            ToTensor(),
        ])
        return data_trans(imgs)

    def _test_trans(self,imgs):
        return transforms.Compose([
            ToPILImage(),
            ToTensor(),
        ])(imgs)

    def _operater(self, tensor_list):
        img1, img2, gt = tensor_list
        if self.mask:
            gt = torch.cat((1.0-gt, gt), dim=0)  # unchanged,changed
        return img1, img2, gt
