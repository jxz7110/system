import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers.model_layers import *
from layers.mixF_layers import *
from layers.mixF import *
from layers.chain_mixF import *


class easyBackbone(nn.Module):
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
        self.mixF = easyMixF(self.d_model, self.n_heads, self.e_layers, self.patch_num, self.head_dropout)
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
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) #x: [batch * nvars, patch_num, d_model]
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
    def __init__(self, d_model, n_heads, e_layers, patch_num, head_dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers =e_layers

        self.multi_heads = nn.ModuleList([])
        for _ in range(self.e_layers):
            self.multi_heads.append(MixLayer(d_model=d_model, num_heads=n_heads, patch_num=patch_num, head_dropout=head_dropout))

    def forward(self, x):
        for multi_head in self.multi_heads:
            output= multi_head(x)
        return output


class MixLayer(nn.Module):
    def __init__(self, d_model, num_heads, patch_num, head_dropout):
        super().__init__()
        self.Resnet =  nn.Sequential(
            mixF(d_model=d_model, num_heads=num_heads, head_dropout=head_dropout),
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2),
            nn.GELU(),
            
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(patch_num,patch_num,kernel_size=1),
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2),
            nn.GELU(),
        )
    def forward(self, x):
        x = x +self.Resnet(x)                  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                 # x: [batch * n_val, patch_num, d_model]
        return x




class mixF(nn.Module):
    def __init__(self, d_model, num_heads=8, window_size=2, head_dropout=0, alph = 0.2):
        super().__init__()
        self.l_heads = int(num_heads * alph)
        self.h_heads = int(num_heads - self.l_heads)
        self.ws = window_size
        self.d_model = d_model
        #high frenquent
        # self.h_attention = HiLo(dim=d_model, num_heads=self.h_heads)
        self.h_attention = MultiHeadAttention(dim_k=d_model, dim_q=d_model, dim_v=d_model, heads=self.h_heads, d_model=d_model)
        
        #low frenquent
        self.low_dim = d_model // window_size
        self.l_attention = nn.Sequential(Transpose(1, 2),
                                      nn.Conv1d(d_model, self.low_dim, kernel_size=window_size, groups=self.low_dim, padding='same'),
                                      Transpose(1, 2),
                                      nn.GELU(),
                                      MultiHeadAttention(dim_k=self.low_dim, dim_q=self.low_dim, dim_v=self.low_dim, heads=self.l_heads, d_model=self.low_dim),
                                      nn.Linear(self.low_dim, self.d_model),
                                      nn.GELU(),
                                      nn.Dropout(head_dropout))
        # self.l_attention = HiLo(dim=self.low_dim, num_heads=self.l_heads)
        self.mix = nn.Sequential(
                                nn.Linear(d_model, d_model),
                                nn.GELU(),
                                Transpose(1, 2),
                                nn.BatchNorm1d(d_model),
                                Transpose(1, 2),
                                 )

    def forward(self, x):
        B, P, D = x.shape #x: [bacth * nvars, patch_num, d_model]
        if self.h_heads != 0:
            high_x = self.h_attention(x)
        if self.l_heads != 0:
            low_x = self.l_attention(x)
        x = high_x + low_x
        x = self.mix(high_x + low_x)
        return x #[bacth * nvars, patch_num, d_model]