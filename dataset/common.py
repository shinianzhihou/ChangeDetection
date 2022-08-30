import os

import cv2
import torchvision
from torch.utils.data import Dataset


class CommonDataset(Dataset):
    """A common dataset class for two input.
    metafile:
        image_000_a.jpg image_000_b.jpg gt_000.png xxx
        image_001_a.jpg image_001_b.jpg gt_001.png xxx
        image_002_a.jpg image_002_b.jpg gt_002.png xxx
        ...

    Args:
        metafile (str): Path to meta_file.
        pipeline (list[dict]): Processing pipeline.
        data_root (str): Data root for images.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        sep (str): Sep in metafile
        c255t1_in_mask (bool): Convert 255 in mask into 1
    """

    def __init__(self,
                 metafile,
                 data_root='',
                 pipeline=None,
                 test_mode=False,
                 sep='\t',
                 c255t1_in_mask=False,
                 ):
        super().__init__()
        self.metafile = metafile
        self.data_root = data_root
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.sep = sep
        self.c255t1_in_mask = c255t1_in_mask
        self.data = self.get_data()
        self.tensor = torchvision.transforms.ToTensor()

    def get_data(self):
        data = []
        lines = open(self.metafile).readlines()
        for line in lines:
            s = line.strip().split(self.sep)
            assert len(s) >= 2, f"wrong when processing {line}"
            s[:2] = [os.path.join(self.data_root, item) for item in s[:2]]
            if not self.test_mode:
                s[2] = os.path.join(self.data_root, s[2])
            data.append(s)
        return data

    def read_images(self, paths):
        images = [cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
                  for path in paths]
        return images

    def read_mask(self, path):
        mask = cv2.imread(path, -1)
        if self.c255t1_in_mask and mask.max() > 1:
            mask /= 255
        mask = mask.astype('int64')
        return mask

    def process(self, image0, image1, mask=None):
        # import pdb; pdb.set_trace()
        if self.pipeline == None:
            return image0, image1, mask

        augmented = self.pipeline(image=image0, image1=image1, mask=mask)
        image0 = self.tensor(augmented['image'])
        image1 = self.tensor(augmented['image1'])
        if self.test_mode:
            return image0, image1
        else:
            mask = self.tensor(augmented['mask'])[0].long()
            return image0, image1, mask

    def __getitem__(self, idx):
        items = self.data[idx]
        image0, image1 = self.read_images(items[:2])
        mask = None if self.test_mode else self.read_mask(items[2])
        return self.process(image0, image1, mask)

    def __len__(self):
        return len(self.data)
