code_root = '/workspace/code/mmcd' # change it to /path/to/code_root
data_root = '/workspace/dataset/stb' # change it to /path/to/data_root

import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0,f'{code_root}/')

import os
import cv2

import torch
import torchvision.models as tvmodels
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Resize
from tqdm import tqdm
from skimage import morphology


from mmseg.datasets import TwoInputDataset,build_dataloader
from mmseg.apis import init_segmentor, inference_segmentor, single_gpu_test
from mmseg.utils import Metric


def base_forward(model,x):
    y = model.encode_decode(x,{})
    return y

def forward(model,x,tta=False): # x=data['img']
    if not tta:
        return base_forward(model,x)
    outs = []
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

flag_save = True # 保存结果
flag_pre = True # resize
flag_post = False
flag_debug = False
tta = True # 打开TTA
t22t1 = False
cal_metric = False
save_submit = True # 保存提交格式的文件

for isp in [576]:
    
    
    name = 'upernet_hr40_256x256_stb'
    name = 'upernet_hr40_512x512_stb'
    name = 'ocrnet_hr40_512x512_stb'
    name = 'upernet_hr40_576x576_stb'
    # meta_file = f'{data_root}/stb/val.v1.txt'
    meta_file = f'{code_root}/work_dirs/cd_stb/meta_files/test.txt'
    config_file = f'{code_root}/work_dirs/cd_stb/{name}/{name}.py'
    save_root = f'{code_root}/work_dirs/cd_stb/results_{name}_{isp}/'
#     save_root = f'{code_root}/work_dirs/cd_stb/results_val_{name}_{isp}/'
    checkpoint_root = f'{code_root}/work_dirs/cd_stb/{name}/'


    f = open(f'{code_root}/work_dirs/cd_stb/result.txt','a')
    f.write('='*88+'\n')
    f.write(f'{config_file}\t{meta_file}\t{checkpoint_root}\n')
    f.write('epc\tpr\tre\tf1\tiou\tmiou\toa\tkappa\n')
    if flag_save:
        os.system(f"mkdir -p {save_root}")
    print('epc\tpr\tre\tf1\tiou\tmiou\toa\tkappa\n')
#     for epc in range(160000,162000,2000):
#     for epc in range(96000,122000,2000):
#     for epc in [80000]:
#     for epc in range(118000,162000,2000):
#     for epc in range(5000,202000,5000):
    for epc in [180000]:
    
        checkpoint_file = os.path.join(checkpoint_root,f'iter_{epc}.pth')
        if not os.path.exists(checkpoint_file):
            print(f'{checkpoint_file} not exists.')
            continue

        model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
        model = model.eval()

        test_pipeline = [
#             dict(type='Resize',height=256,width=256,p=1.0)
            dict(type='Normalize',mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            dict(type='ToTensorV2'),
        ]

        dataset = TwoInputDataset(meta_file=meta_file,sep='\t',pipeline=test_pipeline,data_root=data_root)
        dataloader = build_dataloader(dataset,1,1,shuffle=False)

        metric = Metric(name='sn')
        reisp = Resize([isp,isp])
        
        up512 = torch.nn.Upsample(size=(512,512), mode='bilinear',align_corners=False)
        up513 = torch.nn.Upsample(size=(513,513), mode='bilinear',align_corners=False)
        up256 = torch.nn.Upsample(size=(256,256), mode='bilinear',align_corners=False)
        
        with torch.no_grad():
            for idx,data in tqdm(enumerate(dataloader),total=len(dataset)):
                for k,v in data.items():
                    try:
                        data[k] = v.cuda()
                    except:
                        pass
                if t22t1:
                    data['img'] = data['img'][:,[1,0],...]
                if flag_pre:
                    data['img'] = reisp(data['img'].squeeze()).unsqueeze(0)
                    y = forward(model,data['img'],tta=tta)
                    y = up512(y)
                else:
                    y = forward(model,data['img'],tta=tta)
                    

                if flag_post:
                    out = y.argmax(dim=1)
                    imo = out.clone().cpu().squeeze().numpy().astype(bool)
                    imo = morphology.remove_small_holes(imo,area_threshold=1000,connectivity=8)
                    imo = imo.astype(np.uint8)
                    gtt = data['gt_semantic_seg'].cpu().squeeze().numpy()
                    
                else:
                    out = y.argmax(dim=1)
                    imo = out
                    gtt = data['gt_semantic_seg']
                
                if cal_metric:
                    metric(imo,gtt)
                
                if flag_save:
                    line = dataset.data[idx]
                    save_name = os.path.basename(line.strip().split('\t')[-1]).split('.')[0]+'.png'
                    save_path = os.path.join(save_root,save_name)
                    factor = 1 if save_submit else 255
                    cv2.imwrite(save_path,out.squeeze().detach().cpu().numpy()*factor)
                if flag_debug:
                    if idx==10:
                        break


        if cal_metric:
            local=False
            oa = metric.oa(local=local)
            kappa = metric.kappa(local=local)
            pr = metric.pr(local=local)
            re = metric.re(local=local)
            f1 = metric.f1(local=local)
            iou = metric.iou(local=local)
            miou = metric.miou(local=local)
            print(f'{isp}\t{epc}',pr.item(),re.item(),f1.item(),iou.item(),miou.item(),oa.item(),kappa.item(),sep='\t')
            f.write(f'{isp}\t{epc}\t{pr.item()}\t{re.item()}\t{f1.item()}\t{iou.item()}\t{miou.item()}\t{oa.item()}\t{kappa.item()}-----({idx})\n')
        f.flush()
    f.write('='*88+'\n')
    f.close()