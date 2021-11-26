import sys
code_root = '/workspace/mmcd'
data_root = sys.argv[1]
save_root = sys.argv[2]
# save_root = f'{code_root}/results'
meta_file = f'{code_root}/work_dirs/cd_stb/meta_files/test.txt'
config_file = f'{code_root}/work_dirs/cd_stb/upernet_hr40_576x576_stb/upernet_hr40_576x576_stb.py'
checkpoint_file = f'{code_root}/work_dirs/cd_stb/upernet_hr40_576x576_stb/iter_180000.pth'
device = 'cuda'
# device = 'cpu'


import sys,os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0,f'{code_root}/')

import os
import glob
import cv2

import torch
import torch.nn.functional as F
from tqdm import tqdm
from mmseg.datasets import TwoInputDataset,build_dataloader
from mmseg.apis import init_segmentor

def prepare_meta_file():
    As = glob.glob(os.path.join(data_root,'A/*'))
    Bs = glob.glob(os.path.join(data_root,'B/*'))
    As.sort()
    Bs.sort()
    with open(meta_file,'w') as f:
        for a,b in zip(As,Bs):
            f.write(f'{a}\t{b}\t{a}\n')


def base_forward(model,x):
    y = model.encode_decode(x,{})
    return y

def forward(model,x,tta=False): # x=data['img']
    if not tta:
        return base_forward(model,x)
    origin_x = x.clone()
    
    y = base_forward(model,x)
    
    x_ = origin_x.flip(3)
    y_ = base_forward(model,x_).flip(2)
    y += F.softmax(y_,dim=1)
    
    x_ = origin_x.flip(4)
    y_ = base_forward(model,x_).flip(3)
    y += F.softmax(y_,dim=1)
    
    x_ = origin_x.transpose(3, 4).flip(4)
    y_ = base_forward(model,x_).flip(3).transpose(2, 3)
    y += F.softmax(y_,dim=1)
    
    x_ = origin_x.flip(4).transpose(3, 4)
    y_ = base_forward(model,x_).transpose(2, 3).flip(3)
    y += F.softmax(y_,dim=1)
    
    x_ = origin_x.flip(3).flip(4)
    y_ = base_forward(model,x_).flip(3).flip(2)
    y += F.softmax(y_,dim=1)
    
    return y/6.0

def main():
    model = init_segmentor(config_file, checkpoint_file, device=device)
    model = model.eval()
    test_pipeline = [
        dict(type='Normalize',mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        dict(type='ToTensorV2')
    ]

    dataset = TwoInputDataset(meta_file=meta_file,sep='\t',pipeline=test_pipeline,data_root=data_root)
    dataloader = build_dataloader(dataset,1,1,shuffle=False)

    with torch.no_grad():
        for idx,data in tqdm(enumerate(dataloader),total=len(dataset)):
            for k,v in data.items():
                try:
                    data[k] = v.to(device)
                except:
                    pass
            ori_h, ori_w = data['img'].shape[-2:]
            
            y = forward(model,data['img'],tta=False)
            y = torch.nn.functional.interpolate(y, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
            out = y.argmax(dim=1)
            line = dataset.data[idx]
            save_name = os.path.basename(line.strip().split('\t')[-1]).split('.')[0]+'.png'
            save_path = os.path.join(save_root,save_name)
            cv2.imwrite(save_path,out.squeeze().detach().cpu().numpy()*1)


if __name__ == "__main__":
    prepare_meta_file()
    main()
