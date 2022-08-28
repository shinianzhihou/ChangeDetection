import os
import argparse
import tqdm
import cv2

import numpy as np




def together_show(args):
    file = args.meta_file
    sr = args.save_root
    dr = args.data_root
    lines = open(file).readlines()
    for line in tqdm.tqdm(lines,total=len(lines)):
        s = line.strip().split(args.sep)[:2]
        paths = [os.path.join(dr, p) for p in s]
        imgs = [cv2.imread(p) for p in paths]
        vis = np.concatenate(imgs, axis=1)
        bname = os.path.basename(paths[-1])
        sname = os.path.join(sr, bname)
        cv2.imwrite(sname, vis)
        

def vis_results(args):
    file = args.meta_file
    sr = args.save_root
    dr = args.data_root
    lines = open(file).readlines()
    for line in tqdm.tqdm(lines,total=len(lines)):
        s = line.strip().split(args.sep)[:3]
        paths = [os.path.join(dr, p) for p in s]
        imgs = [cv2.imread(p) for p in paths[:2]]
        mask = cv2.imread(paths[2],-1)
        mask[mask>0] = 255
        zero = np.zeros_like(mask)
        mask = np.stack([zero,zero,mask],axis=-1)
        vis0 = np.concatenate(imgs,axis=1)
        vis1 = np.concatenate([imgs[0]*0.6+mask*0.4, imgs[1]*0.6+mask*0.4],axis=1)
        vis = np.concatenate([vis0,vis1],axis=0)
        bname = os.path.basename(paths[-1])
        sname = os.path.join(sr, bname)
        cv2.imwrite(sname, vis)




def main():
    parser = argparse.ArgumentParser(
        description='vis')
    parser.add_argument('-mf','--meta_file', type=str, help='path to meta file, such as train.txt and val.txt')
    parser.add_argument('--sep', type=str, default='\t', help='sep in meta file')
    parser.add_argument('-sr','--save_root', type=str, help='where to save vis')
    parser.add_argument('-dr','--data_root', type=str, help='data root in meta file')
    parser.add_argument('-c','--choice', type=int, default=0, help='choice for visualization')
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.mkdir(args.save_root)

    if args.choice == 0:
        together_show(args)
    elif args.choice == 1:
        vis_results(args)



if __name__=="__main__":
    main()
