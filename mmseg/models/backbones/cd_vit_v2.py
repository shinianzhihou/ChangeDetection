import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

from torch import einsum
from einops import rearrange, repeat

from ..builder import BACKBONES


def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(
            t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(
            t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # combine heads out
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    """
    Build transformer encoder
    """

    def __init__(self,
                 depth=4,
                 dim=256,
                 dim_head=256,
                 heads=8,
                 attn_dropout=0.0,
                 ff_dropout=0.0):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head=dim_head,
                                       heads=heads, dropout=attn_dropout)),
                PreNorm(dim, Attention(dim, dim_head=dim_head,
                                       heads=heads, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
            ]))

    def forward(self,x):
        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f) + x
            x = ff(x) + x

        return x


class Backbone(nn.Module):
    """
    Backbone before transformer
    """
    def __init__(self,choice='resnet18', pretrained=False, **kwargs):
        super().__init__()
        
        if choice == 'resnet18':
            model = tvmodels.resnet18(pretrained=pretrained, **kwargs)
        elif choice == 'resnet50':
            model = tvmodels.resnet50(pretrained=pretrained, **kwargs)
        else:
            raise NotImplementedError
        self.layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
#             model.maxpool,
        )
        self.layers = nn.Sequential(
            model.layer1,
            model.layer2,
        )
        
    def forward(self,x):
        x = self.layer0(x)
        x = self.layers(x)
        return x
    

def make_backbone(*args, **kwargs):
    return Backbone(*args, **kwargs)


def make_transformer_encoder(*args, **kwargs):
    return TransformerEncoder(*args, **kwargs)


@BACKBONES.register_module()
class CDVitV2(nn.Module):
    """
    Build vision transformer for change detection
    """

    def __init__(self, backbone_choice='resnet18',
                 num_images=2,
                 image_size=224,
                 feature_size=28,
                 patch_size=4,
                 in_channels=128,
                 out_channels=32,
                 encoder_dim=512,
                 encoder_heads=8,
                 encoder_dim_heads=64,
                 encoder_depth=4,
                 attn_dropout=0.0,
                 ff_dropout=0.0,
                 ):
        super().__init__()

        patch_dim = out_channels * patch_size ** 2
        num_patches = (feature_size // patch_size) ** 2
        num_positions = num_images * num_patches
        
        self.backbone = make_backbone(choice=backbone_choice,
                                      pretrained=True)
        self.encoder = nn.ModuleList([])
        for _ in range(encoder_depth):
            self.encoder.append(nn.ModuleList([
                PreNorm(encoder_dim, Attention(encoder_dim, dim_head=encoder_dim_heads,
                                       heads=encoder_heads, dropout=attn_dropout)),
                PreNorm(encoder_dim, Attention(encoder_dim, dim_head=encoder_dim_heads,
                                       heads=encoder_heads, dropout=attn_dropout)),
                PreNorm(encoder_dim, FeedForward(encoder_dim, dropout=ff_dropout))
            ]))
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        # used to reduce feature dim
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=1)

        self.to_path_embedding = nn.Identity()  # used to reduce token dim
        self.cls_token = nn.Parameter(torch.randn(1, encoder_dim))
        self.pos_emb = nn.Embedding(num_positions + 1, encoder_dim)
        self.to_out = nn.Sequential(
            nn.Upsample(size=image_size, mode='bilinear',align_corners=False)
        )
        self._preprocess()

    def _preprocess(self):
        pass

    def forward(self, x):  # should return a tuple
        """
        x : batch, num_imgs, channel, h, w
        """
        b, n, c, h, w = x.shape
        device = x.device
        # x = x.view(-1,c,h,w)
        x = rearrange(x, 'b n c h w -> (b n) c h w')
        feature = self.backbone(x)  # bn c' h' w'
        # print('f0',feature.shape)
        feature = self.conv1x1(feature)
        # print('f0-1',feature.shape)
        feature = rearrange(feature, '(b n) c h w -> b n c h w', b=b)  # bn c' h' w'
        # print('f1',feature.shape)
        feature = rearrange(feature, 'b n c (h1 p1) (w1 p2) -> b (n h1 w1) (p1 p2 c)',
                            p1=self.patch_size, p2=self.patch_size)
        # print('f2',feature.shape)
        tokens = self.to_path_embedding(feature)
#         tokens = feature

        # print('tokens',tokens.shape)
        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        # print('cls_token',cls_token.shape)
        x = torch.cat((cls_token, tokens), dim=1)
        # print('x1',x.shape)
        x += self.pos_emb(torch.arange(x.shape[1], device=device))  # ?
        # print('x2',x.shape)
        for (time_attn, spatial_attn, ff) in self.encoder:
            x = time_attn(x, 'b (n p) d', '(b p) n d', p=self.num_patches) + x
            x = spatial_attn(x, 'b (n p) d', '(b n) p d', n=n) + x
            x = ff(x) + x
        # print('x3',x.shape)
        h = int(math.sqrt((x.shape[1]-1)//2))
        feature = rearrange(x[:, 1:], 'b (n h w) (p1 p2 c) -> b n c (h p1) (w p2)',
                            n=n, h=h, w=h, p1=self.patch_size, p2=self.patch_size)
        feature = rearrange(feature, 'b n c h w -> b (n c) h w')
        ret_list = []
        out = self.to_out(feature)
        ret_list.append(out)
        return ret_list
    
    
        
    def init_weights(self, pretrained=None):
        pass


if __name__=="__main__":
    model = CDVit()
    a = torch.ones(8,2,3,224,224)
    out = model(a)
    print(a.shape,len(out),out[0].shape)