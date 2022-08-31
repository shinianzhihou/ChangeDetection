import argparse
import os
import random

import cv2
import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from build import *
from utils import load_checkpoint



@torch.no_grad()
def run(args):

    cd_ckpt_path = args.cd_ckpt_path
    save_dir = args.save_dir
    device = args.device
    factor = args.factor
    decoder_attention_type = args.decoder_attention_type

    os.system(f"mkdir -p {save_dir}")

    device = torch.device(device)

    # model
    model = build_model(choice='cdp_UnetPlusPlus', encoder_name="timm-efficientnet-b2",
                        encoder_weights="noisy-student",
                        decoder_attention_type=decoder_attention_type,
                        in_channels=3,
                        classes=2,
                        siam_encoder=True,
                        fusion_form='concat',)
    model = model.to(device)
    load_checkpoint(cd_ckpt_path, {"state_dict":model})
    model.eval()



    # pipeline
    test_pipeline = A.Compose([
        A.HorizontalFlip(p=0.0), ],
        additional_targets={'image1': 'image'})

    # dataloader
    test_set = build_dataset(choice='CommonDataset',
                            metafile="val.txt",
                            data_root="data/cd/stb/",
                            test_mode=True,
                            pipeline=test_pipeline,)


    test_loader = DataLoader(dataset=test_set,
                            pin_memory=True,
                            batch_size=1,
                            num_workers=0,
                            shuffle=False,
                            drop_last=False,
                            sampler=None)

    # inference
    for batch, data in enumerate(test_loader):
        img1 = data[0].to(device)
        img2 = data[1].to(device)

        pred = model(img1, img2)
        output = pred.argmax(dim=1).cpu().numpy()[0]*factor
        save_name = test_set.get_file_name(batch,suffix='.png')
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, output)




def main():
    parser = argparse.ArgumentParser(
        description='train')


    parser.add_argument('-ccp', '--cd_ckpt_path',  type=str, default=None,
                        help='change detection ckpt path')
    parser.add_argument('-sd', '--save_dir', type=str, default=None,
                        help='dir for saving images')
    parser.add_argument('-de', '--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('-f', '--factor', type=int, default=1,
                        help='factor*output in cv2.imwrite')
    parser.add_argument('-dat', '--decoder_attention_type', type=str, default=None,
                        help='decoder attention type')

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
