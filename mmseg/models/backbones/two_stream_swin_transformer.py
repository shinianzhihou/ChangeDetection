import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

from torch import einsum
from einops import rearrange, repeat
import timm.models as tms

from ..builder import BACKBONES



@BACKBONES.register_module()
class TwoStreamSwinTransformer(nn.Module):
    arch_dict = {
        
    }
    def __init__(self,variant,pretrained=False):
        super().__init__()
        if variant == 'swin_large_patch4_window12_384_in22k':
            model = tms.swin_large_patch4_window12_384_in22k(pretrained=pretrained)

            self.img_shapes = [48,24,12,12]
            self.channel_shapes = [384,768,1536,1536]
        elif variant == 'swin_base_patch4_window7_224_in22k':
            model = tms.swin_base_patch4_window7_224_in22k(pretrained=pretrained)
            self.img_shapes = [28,14,7,7]
            self.channel_shapes = [384,768,1536,1536]
        elif variant == 'swin_base_patch4_window12_384_in22k':
            model = tms.swin_base_patch4_window12_384_in22k(pretrained=pretrained)
            self.img_shapes = [48,24,12,12]
            self.channel_shapes = [256,512,1024,1024]
        
        else:
            raise NotImplementedError
            
            
        self.patch_embed = model.patch_embed
        self.layers = model.layers
        self.absolute_pos_embed = model.absolute_pos_embed
        self.pos_drop = model.pos_drop
    
    def basic_forward(self,x):
        out = []
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for idx,layer in enumerate(self.layers):
            x = layer(x)
            b,hw,d = x.shape
            h = w = self.img_shapes[idx]
            out.append(x.view(b,d,h,w))
        return out
    
    def forward(self,x):
        b,n,c,h,w = x.shape
        x0,x1 = x.chunk(dim=1,chunks=2)
        x0,x1 = x0.squeeze(1),x1.squeeze(1)
        
        ys_0 = self.basic_forward(x0)
        ys_1 = self.basic_forward(x1)
#         for i in ys_0:
#             print(i.shape)
        return [i-j for i,j in zip(ys_0,ys_1)]
            
    def init_weights(self, pretrained=None):
        pass

