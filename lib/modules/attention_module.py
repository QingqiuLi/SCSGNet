import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.modules.dcn import DeformUnfold
from lib.modules.layers import *
from utils.utils import *


class TAG(nn.Module):
    def __init__(self, in_channels, channel, depth=3, kernel_size=3):
        super(TAG, self).__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 3, 3, 1, 1), #kernel=3,padding=1,stride=1
            nn.BatchNorm2d(in_channels // 3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.foregound_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 3, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.background_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 3, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.conv_in = conv((in_channels // 3) * 3, channel, 1)
        self.conv_in2 = conv(in_channels, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 3 if kernel_size == 3 else 1)

    def forward(self, x, pred):
        pred = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(pred)

        # boundary
        p = fg - .5
        bd_att = .5 - torch.abs(p)
        bd_x = x * bd_att
        # foregound
        fg_att = torch.clip(p, 0, 1)
        fg_x = x * fg_att
        # background
        bg_att = torch.clip(-p, 0, 1)
        bg_x = x * bg_att

        foregound_out = self.foregound_conv(fg_x)
        background_out = self.background_conv(bg_x)
        boundary_out = self.boundary_conv(bd_x)

        out = torch.cat([foregound_out, background_out, boundary_out], dim=1)
        out = self.conv_in(out)       # (in_channels // 3) * 3  -> channel
        for conv_mid in self.conv_mid:
            out = F.relu(conv_mid(out))
        out = self.conv_out(out)
        out = out + pred
        return x,out

    
class DLCC(nn.Module):
    # x : input feature maps( B X C X W X H)
    def __init__(self, dim, num_heads=8,  qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # sample
        self.num_samples = 9
        self.conv_offset = nn.Linear(self.head_dim, self.num_samples * 2, bias=qkv_bias)
        self.unfold = DeformUnfold(kernel_size=3, padding=1, dilation=1)

    def forward(self, x):
        B, C, W, H = x.shape
        N=H*W
        x=((x.permute(0,2,3,1)).reshape(B,C,W*H)).reshape(B,W*H,C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        offset = self.conv_offset(x.reshape(B, N, self.num_heads, self.head_dim)).permute(0, 2, 3, 1).reshape(B * self.num_heads, self.num_samples * 2, H, W)

        k = k.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, H, W)
        v = v.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, H, W)
        k = self.unfold(k, offset).transpose(1, 2).reshape(B, self.num_heads, N, self.head_dim, self.num_samples)
        v = self.unfold(v, offset).reshape(B, self.num_heads, self.head_dim, self.num_samples, N).permute(0, 1, 4, 3, 2)

        attn = torch.matmul(q.unsqueeze(3), k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x=x.reshape(B,W,H,C).permute(0,3,1,2)

        return x