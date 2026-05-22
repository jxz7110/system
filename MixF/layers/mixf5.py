import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers.model_layers import *
from layers.mixF_layers import *
from layers.mixF import *
from layers.chain_mixF import *


class easyBackbone5(nn.Module):
    def __init__(self, configs, revin = False, affine = True, subtract_last = False):
        super().__init__()
        self.nvals = configs.input_channels_len
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.revin = revin
        self.head_dropout = configs.head_dropout
        if self.revin: self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.W_X = nn.Linear(self.patch_len, self.d_model)
        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 1) + 1
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=self.patch_num, d_model=self.d_model)
        self.mixF = easyMixF(self.nvals, self.d_model, self.n_heads, self.e_layers, self.patch_num, self.dropout)
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.pred_len),
            nn.Dropout(self.head_dropout)
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, int(self.pred_len * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.pred_len * 2), self.pred_len),
            nn.Dropout(self.head_dropout)
        )
        self.dropout = nn.Dropout(self.dropout)
    def forward(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        #patch
        x = x.permute(0, 2, 1)          #x: [batch, nvars, seq_len]
        x = self.padding_patch_layer(x) #填充边缘数据
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) #x: [batch, nvars, patch_num, patch_size]
        
        #position_embedding
        x = self.W_X(x)
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) #x: [batch * nvars, patch_num, d_model]
        x = self.dropout(x)
        u = self.head0(x)

        # multi-head attention部分，进行多频处理
        x = self.mixF(x + self.W_pos)
        
        x = self.head1(x)
        x = u + x
        x = torch.reshape(x, (bs, nvars, -1))
        x = x.permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x
    

class easyMixF(nn.Module):
    def __init__(self, nvals, d_model, n_heads, e_layers, patch_num, head_dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers =e_layers
        self.multi_heads = nn.ModuleList([])
        for _ in range(self.e_layers):
            self.multi_heads.append(MixLayer(nvals=nvals, d_model=d_model, num_heads=n_heads, patch_num=patch_num, head_dropout=head_dropout))

    def forward(self, x):
        for multi_head in self.multi_heads:
            output= multi_head(x)
        return output


class MixLayer(nn.Module):
    def __init__(self, nvals, d_model, num_heads, patch_num, head_dropout):
        super().__init__()
        self.Resnet =  nn.Sequential(
            mixF(nvals=nvals, d_model=d_model, patch_num=patch_num, num_heads=num_heads, head_dropout=head_dropout),
            #因为这部分是做的d_model维度进行处理，所以采用d_model的正则化
            nn.BatchNorm2d(nvals),
            nn.GELU(),
            
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(nvals, nvals,kernel_size=1),
            nn.BatchNorm2d(nvals),
            nn.GELU(),
        )
    def forward(self, x):
        x = x +self.Resnet(x)                  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                 # x: [batch * n_val, patch_num, d_model]
        return x
    
class mixF(nn.Module):
    def __init__(self, nvals, d_model, patch_num, num_heads=8, window_size=8, head_dropout=0):
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(nvals, nvals, kernel_size=window_size, groups=nvals, padding='same'),
            nn.GELU(),
            nn.BatchNorm2d(nvals),
        )
        self.heads = int(num_heads)
        self.d_model = d_model
        self.new_d_model = patch_num * d_model
        self.q_linear = nn.Linear(self.new_d_model, self.new_d_model)
        self.k_linear = nn.Linear(self.new_d_model, self.new_d_model)
        self.v_linear = nn.Linear(self.new_d_model, self.new_d_model)
        self.dim = int(self.new_d_model/num_heads)
        self.scale = self.dim ** -0.5
                                 

    def forward(self, x):
        x = self.conv2(x)
        B, N, P, D = x.shape #x: [bacth , nvars, patch_num, d_model]
        x = x.reshape(B, N, P*D)
        q=self.q_linear(x).reshape(B, N, self.heads, self.dim).permute(0, 2, 1, 3)
        k=self.k_linear(x).reshape(B, N, self.heads, self.dim).permute(0, 2, 1, 3)
        v=self.v_linear(x).reshape(B, N, self.heads, self.dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1))*self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(1, 2).reshape(B, N, P, D)
        return attn #[bacth * nvars, patch_num, d_model]
    