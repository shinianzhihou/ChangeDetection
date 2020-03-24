import os
from bisect import bisect_right

import cv2
import pandas as pd

from configs import cfg


def img_to_dataset(img_path,step,x_range,y_range,size=112,channel=1):
    '''Divide the img into dataset.

    Args:  
        img_path(string): the path of the image to be divided.  
        step(int): step of slide window.  
        x_range(list): test image(not be divided) <-- [x_start,x_end]  
        y_range(list): test image(not be divided) <-- [y_start,y_end]  
        size(int): size of divided images, default 112  
        channel(int): number of image channels, default 1
    '''
    dataset = []
    img_raw = cv2.imread(img_path)[:,:,0:channel]
    h,w,_ = img_raw.shape
    check = lambda t,t_: False if bisect_right(t_,t+1)==1 or bisect_right(t_,t+size-1)==1 else True

    # train
    for x in range(0,w,step):
        for y in range(0,h,step):
            if not(check(x,x_range) or check(y,y_range)): # or, not and
                continue
            if x+size>w or y+size>h:
                continue
            img_divided = img_raw[y:y+size,x:x+size,:]
            img_d_name = "_x_%d_y_%d.png"%(x,y)
            img_d_path = img_path.replace(".png",img_d_name)
            cv2.imwrite(img_d_path,img_divided)
            dataset.append({
                "source":img_path,
                "step":step,
                "type":"train",
                "size":size,
                "path":img_d_path,
                "img":img_divided,
            })
    
    # test
    test_path = img_path.replace(".png","_test.png")
    test_img = img_raw[y_range[0]:y_range[1],x_range[0]:x_range[1],:]
    cv2.imwrite(test_path,test_img)
    dataset.append({
        "source":img_path,
        "step":step,
        "type":"test",
        "size":size,
        "path":test_path,
        "img":test_img,
    })

    return dataset