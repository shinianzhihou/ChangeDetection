import torch
import torch.nn as nn
import timm.models as tmodels


from ..builder import BACKBONES
from .fightingcv.attention.CBAM import CBAMBlock

# @BACKBONES.register_module()
# class SiameseEfficientNet(nn.Module):
#     def __init__(self, 
#                  name,
#                  fusion,
#                  bb_pre=['conv_stem','bn1','act1'],
#                  bb_mid=['blocks'],
#                  **kwargs):
#         super().__init__()
#         assert fusion in ['diff','conc']
#         assert  name.find('efficientnet') != -1, 'Must be efficientnet_*'
#         self.conv1x1 = kwargs.pop('conv1x1',False)
#         self.attention = kwargs.pop('attention',False)
#         conv1x1_ins = kwargs.pop('conv1x1_in',[])
#         conv1x1_outs = kwargs.pop('conv1x1_out',[])
#         self.in_index = kwargs.pop('in_index',[])
#         if self.conv1x1:
#             self.conv1x1s = nn.ModuleList([nn.Conv2d(in_c,out_c,1) \
#                                            for in_c,out_c in \
#                                            zip(conv1x1_ins,conv1x1_outs)])
#         if self.attention:
#             self.attentions = nn.ModuleList([CBAMBlock(channel=in_c,reduction=16,kernel_size=7) \
#                                            for in_c,out_c in \
#                                            zip(conv1x1_ins,conv1x1_outs)])
        
#         bb = tmodels.create_model(name,**kwargs) # backbone
#         self.fusion = fusion
#         self.bb_pre_m = nn.ModuleList([getattr(bb,name) for name in bb_pre])
#         self.bb_mid_m = nn.ModuleList([getattr(bb,name) for name in bb_mid])
    
#     def base_forward(self,x):
#         ys = []
#         for m in self.bb_pre_m:
#             x = m(x)
#         for m in self.bb_mid_m:
#             for n in m:
#                 x = n(x)
#                 ys.append(x)
#         return ys
    
#     def forward(self,x):
#         b,n,c,h,w = x.shape
#         xs = [x[:,i,...] for i in range(n)]
#         ys = [self.base_forward(_) for _ in xs]
#         if self.fusion == 'diff':
#             return [j-i for i,j in zip(ys[0],ys[1])]
# #             return [torch.abs(j-i) for i,j in zip(ys[0],ys[1])]
        
#         elif self.fusion == 'conc':
            
#             if self.conv1x1:
#                 zs = []
#                 for index in range(len(ys[0])):
#                     tmp = torch.cat([ys[1][index],ys[0][index]],dim=1)
#                     if index in self.in_index:
#                         idx = self.in_index.index(index)                        
#                         zs.append(self.conv1x1s[idx](tmp))
#                     else:
#                         zs.append(tmp)
#                 return zs 
#             elif self.attention:
#                 zs = []
#                 for index in range(len(ys[0])):
#                     tmp = torch.cat([ys[1][index],ys[0][index]],dim=1)
#                     if index in self.in_index:
#                         idx = self.in_index.index(index)                        
#                         zs.append(self.attentions[idx](tmp))
#                     else:
#                         zs.append(tmp)
#                 return zs 
                    
#             else:
#                 return [torch.cat([j,i],dim=1) for i,j in zip(ys[0],ys[1])]
                    
#         else:
#             raise NotImplementedError
            
#     def init_weights(self, pretrained=None):
#         pass 
        


@BACKBONES.register_module()
class SiameseEfficientNet(nn.Module):
    def __init__(self, 
                 name,
                 fusion,
                 bb_pre=['conv_stem','bn1','act1'],
                 bb_mid=['blocks'],
                 **kwargs):
        super().__init__()
        assert fusion in ['diff','conc']
        assert  name.find('efficientnet') != -1, 'Must be efficientnet_*'
        self.fusion = fusion
        self.conv1x1 = kwargs.pop('conv1x1',False)
        self.attention = kwargs.pop('attention',False)
        self.non_siamese = kwargs.pop('non_siamese',False)
        conv1x1_ins = kwargs.pop('conv1x1_in',[])
        conv1x1_outs = kwargs.pop('conv1x1_out',[])
        self.in_index = kwargs.pop('in_index',[])
        if self.conv1x1:
            self.conv1x1s = nn.ModuleList([nn.Conv2d(in_c,out_c,1) \
                                           for in_c,out_c in \
                                           zip(conv1x1_ins,conv1x1_outs)])
        if self.attention:
            self.attentions = nn.ModuleList([CBAMBlock(channel=in_c,reduction=16,kernel_size=7) \
                                           for in_c,out_c in \
                                           zip(conv1x1_ins,conv1x1_outs)])
        if self.non_siamese:
            bb1 = tmodels.create_model(name,**kwargs) # backbone
            bb2 = tmodels.create_model(name,**kwargs)
            self.bb1_pre_m = nn.ModuleList([getattr(bb1,name) for name in bb_pre])
            self.bb1_mid_m = nn.ModuleList([getattr(bb1,name) for name in bb_mid])
            self.bb2_pre_m = nn.ModuleList([getattr(bb2,name) for name in bb_pre])
            self.bb2_mid_m = nn.ModuleList([getattr(bb2,name) for name in bb_mid])
        else:
            bb = tmodels.create_model(name,**kwargs) # backbone
            self.bb_pre_m = nn.ModuleList([getattr(bb,name) for name in bb_pre])
            self.bb_mid_m = nn.ModuleList([getattr(bb,name) for name in bb_mid])
    
    def base_forward(self,x,idx=0):
        ys = []
        if self.non_siamese:
            if idx == 0: 
                for m in self.bb1_pre_m:
                    x = m(x)
                for m in self.bb1_mid_m:
                    for n in m:
                        x = n(x)
                        ys.append(x)
            else:
                for m in self.bb2_pre_m:
                    x = m(x)
                for m in self.bb2_mid_m:
                    for n in m:
                        x = n(x)
                        ys.append(x)
        else:
            for m in self.bb_pre_m:
                x = m(x)
            for m in self.bb_mid_m:
                for n in m:
                    x = n(x)
                    ys.append(x)
        return ys
    
    def forward(self,x):
        b,n,c,h,w = x.shape
        xs = [x[:,i,...] for i in range(n)]
        if self.non_siamese:
            ys = [self.base_forward(_,idx) for idx,_ in enumerate(xs)]
        else:
            ys = [self.base_forward(_) for _ in xs]
        if self.fusion == 'diff':
            return [j-i for i,j in zip(ys[0],ys[1])]
#             return [torch.abs(j-i) for i,j in zip(ys[0],ys[1])]
        
        elif self.fusion == 'conc':
            
            if self.conv1x1:
                zs = []
                for index in range(len(ys[0])):
                    tmp = torch.cat([ys[1][index],ys[0][index]],dim=1)
                    if index in self.in_index:
                        idx = self.in_index.index(index)                        
                        zs.append(self.conv1x1s[idx](tmp))
                    else:
                        zs.append(tmp)
                return zs 
            elif self.attention:
                zs = []
                for index in range(len(ys[0])):
                    tmp = torch.cat([ys[1][index],ys[0][index]],dim=1)
                    if index in self.in_index:
                        idx = self.in_index.index(index)                        
                        zs.append(self.attentions[idx](tmp))
                    else:
                        zs.append(tmp)
                return zs 
                    
            else:
                return [torch.cat([j,i],dim=1) for i,j in zip(ys[0],ys[1])]
                    
        else:
            raise NotImplementedError
            
    def init_weights(self, pretrained=None):
        pass 
        
