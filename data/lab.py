import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset

from solver.transforms import *

class Lab(Dataset):
  '''Lab Dataloader.'''
  def __init__(self,csv_path,data_trans=None,mask=True,choice="none"):
    super(Lab,self).__init__()
    self.data_p = pd.read_csv(csv_path)
    self.get_data(choice)
    self.data_trans = data_trans
    self.mask = mask
    
  def __getitem__(self,index):
    path_list = self.data.loc[index]
    img_list = [cv2.imread(path)[:,:,0:1] for path in path_list]
    if self.data_trans is None:
      tensor_list = self._heter_transforms(img_list)
    else:
      tensor_list = self.data_trans(img_list)

    return self._operater(tensor_list)

  def __len__(self):
    return len(self.data)  

  def _heter_transforms(self,imgs):
    data_trans = transforms.Compose([
      ToPILImage(),
      RandomHorizontalFlip(p=0.5),
      RandomVerticalFlip(p=0.5),
      transforms.RandomApply([
          RandomRotation(180,expand=False,fill=0),                      
        ],
        p=0.8),
      ToTensor(),
    ])
    return data_trans(imgs)

  def _operater(self,tensor_list):
    img1,img2,gt = tensor_list
    if self.mask:
      gt = torch.cat((1.0-gt,gt),dim=0) # unchanged,changed
    return img1,img2,gt


  def get_data(self,choice):
    if choice=="change":
      self.data = self.data_p.loc[self.data_p.p_change>0,["img1","img2","gt"]]
    elif choice=="unchange":
      self.data = self.data_p.loc[self.data_p.p_change==0,["img1","img2","gt"]]
    else:
      self.data = self.data_p.loc[:,["img1","img2","gt"]]
    self.data = self.data.reset_index(drop=True)
  
  def get_raw_item_by_idx(self,index):
    path_list = self.data.loc[index]
    img_list = [cv2.imread(path)[:,:,0:1] for path in path_list]
    return img_list


if __name__=="__main__":
  import matplotlib.pyplot as plt

  from configs import cfg

  yaml_root = "configs/heterogeneous_change_detection"
  yaml_name = "default.yaml"
  yaml_file = os.path.join(yaml_root,yaml_name)
  cfg.merge_from_file(yaml_file)
  lab = Lab(cfg.DATASETS.TRAIN_CSV)

  img1,img2,gt = lab.__getitem__(56)
  img1_t,img2_t,gt_t = ToPILImage()([img1,img2,gt[1,:,:]])

  img1,img2,gt = lab.get_raw_item_by_idx(56)

  plt.subplot(2,3,1)
  plt.imshow(img1[:,:,0],cmap='gray')
  plt.subplot(2,3,2)
  plt.imshow(img2[:,:,0],cmap='gray')
  plt.subplot(2,3,3)
  plt.imshow(gt[:,:,0],cmap='gray')

  plt.subplot(2,3,4)
  plt.imshow(img1_t,cmap='gray')
  plt.subplot(2,3,5)
  plt.imshow(img2_t,cmap='gray')
  plt.subplot(2,3,6)
  plt.imshow(gt_t,cmap='gray')

  plt.show()
