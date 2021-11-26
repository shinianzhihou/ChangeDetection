import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool


def fun(idx):
    path_imgs = [v[idx] for v in paths_imgs]
    name = os.path.basename(path_imgs[0])
    save_path = os.path.join(save_root,name)
    imgs = [Image.open(v) for v in path_imgs]
    fig = plt.figure(figsize=(len(imgs)*4,4))
    for jdx,img in enumerate(imgs):
        cmap = None if len(img.split())>1 else 'gray'
        plt.subplot(1,len(imgs),jdx+1);plt.imshow(img,cmap=cmap);plt.tight_layout();plt.axis('off')
    plt.savefig(save_path,pad_inches=0,bbox_inches='tight',transparent=True,dpi=600)
    plt.close(fig)

# if __name__=='__main__':
import sys
paths_str = sys.argv[1] # python tools/together_show.py path1,path2,path3 save_root
save_root = sys.argv[2]
num_workers = int(sys.argv[3])
if not os.path.exists(save_root): os.mkdir(save_root)
paths = paths_str.split(',')
paths_imgs = [sorted(glob.glob(os.path.join(path,'*'))) for path in paths]
length = len(paths_imgs)
print(len(_) for _ in paths_imgs)
with Pool(num_workers) as p:
    p.map(fun,range(3))



