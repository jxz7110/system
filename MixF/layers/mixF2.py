import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers.model_layers import *
from layers.mixF_layers import *
from layers.mixF import *
from layers.chain_mixF import *


class easyBackbone2(nn.Module):
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
        self.mixF = easyMixF(self.d_model, self.n_heads, self.e_layers, self.patch_num, self.dropout)
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
            
            mixF(d_model=d_model, patch_num=patch_num, num_heads=num_heads, head_dropout=head_dropout),
            #因为这部分是做的d_model维度进行处理，所以采用d_model的正则化
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2),
            nn.GELU(),
            
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(patch_num, patch_num,kernel_size=1),
            nn.BatchNorm1d(patch_num),
            nn.GELU(),
        )
    def forward(self, x):
        x = x +self.Resnet(x)                  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                 # x: [batch * n_val, patch_num, d_model]
        return x




class mixF(nn.Module):
    def __init__(self, d_model, patch_num, num_heads=8, window_size=8, head_dropout=0, alpha=0.2, beta=0.5):
        super().__init__()
        self.l_heads = int(num_heads * alpha)
        self.h_heads = int(num_heads - self.l_heads)
        self.ws = window_size
        self.d_model = d_model
        #high frenquent
        self.h_attention = MixFAttention(patch_num=patch_num, d_model=d_model, window_size=window_size, beta=beta, num_heads=self.h_heads)
        # self.h_attention = HiLo(dim=d_model, en_dim=patch_num ,num_heads=self.h_heads, window_size=4)
        # self.h_attention = MultiHeadAttention(dim_k=d_model, dim_q=d_model, dim_v=d_model, heads=self.h_heads, d_model=d_model)
        #low frenquent
        self.low_dim = d_model // window_size
        self.l_attention = nn.Sequential(nn.Conv1d(patch_num, patch_num, kernel_size=window_size, groups=patch_num, padding='same'),
                                      nn.GELU(),
                                      MultiHeadAttention(dim_k=d_model, dim_q=d_model, dim_v=d_model, heads=self.l_heads, d_model=d_model),
                                      nn.Linear(d_model, d_model),
                                      nn.GELU(),
                                      nn.Dropout(head_dropout))
        self.mix = nn.Sequential(nn.Linear(d_model, d_model),
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
    







class MixFAttention(nn.Module):
    def __init__(self, patch_num, d_model, window_size=8, beta=0.5, num_heads=8):
        super().__init__()

        #self-attention heads in low patch_size
        self.l_heads = int(num_heads * beta)
        assert d_model % self.l_heads == 0, f"d_model {d_model} should be divided by l_heads {self.l_heads}."
        self.l_dim = int(d_model/self.l_heads)
        self.l_scale = self.l_dim ** -0.5
        self.l_conv1d = nn.Sequential(Transpose(1, 2),
                                    nn.Conv1d(d_model, d_model, kernel_size=window_size, padding='same', groups=d_model),
                                    Transpose(1, 2)
                                    )
        self.l_q_linear = nn.Linear(d_model, d_model)
        self.l_k_linear = nn.Linear(d_model, d_model)
        self.l_v_linear = nn.Linear(d_model, d_model)
        #self-attention heads in high patch_size
        self.h_heads = num_heads - self.l_heads
        assert d_model % self.h_heads == 0, f"d_model {d_model} should be divided by h_heads {self.h_heads}."
        self.h_dim = int(d_model/self.h_heads)
        self.h_scale = self.h_dim ** -0.5
        self.h_q_linear = nn.Linear(d_model, d_model)
        self.h_k_linear = nn.Linear(d_model, d_model)
        self.h_v_linear = nn.Linear(d_model, d_model)
        self.ws = window_size
        self.hl_mix = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.GELU(),
                                    Transpose(1, 2),
                                    nn.BatchNorm1d(d_model),
                                    Transpose(1, 2),
                                    )
    def l_attention(self, x):
        B, P, dim = x.shape
        q = x # [batch*features, patch_size, d_model]
        low_x = self.l_conv1d(x) # [batch*features, patch_size/ws, d_model]
        q = self.l_q_linear(q).reshape(B, P, self.l_heads, self.l_dim).permute(0, 2, 1, 3) # [batch*features, l_heads, patch_size, d_model/l_heads]
        k = self.l_k_linear(low_x).reshape(B, -1, self.l_heads, self.l_dim).permute(0, 2, 1, 3)# [batch*features, l_heads, patch_size/ws, d_model/l_heads]
        v = self.l_v_linear(low_x).reshape(B, -1, self.l_heads, self.l_dim).permute(0, 2, 1, 3) # [batch*features, l_heads, patch_size/ws, d_model/l_heads]
        attn = (q @ k.transpose(-2, -1)) * self.l_scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(1, 2).reshape(B, -1, dim) # [batch*features, patch_size, d_model]
        return attn
    
    def h_attention(self, x):
        B, P, dim = x.shape # [batch*features, patch_size, d_model]
        q = self.h_v_linear(x).reshape(B, P, self.h_heads, self.h_dim).permute(0, 2, 1, 3) # [batch*features, h_heads, patch_size, d_model/h_heads]
        k = self.h_v_linear(x).reshape(B, P, self.h_heads, self.h_dim).permute(0, 2, 1, 3) # [batch*features, h_heads, patch_size, d_model/h_heads]
        v = self.h_v_linear(x).reshape(B, P, self.h_heads, self.h_dim).permute(0, 2, 1, 3) # [batch*features, h_heads, patch_size, d_model/h_heads]
        attn = (q @ k.transpose(-2, -1)) * self.h_scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(1, 2).reshape(B, -1, dim)
        return attn

    

    def forward(self, x): # [batch*features, patch_size, d_model]
        l_out = self.l_attention(x) 
        h_out = self.h_attention(x) 
        return self.hl_mix(l_out + h_out)



class MultiHeadAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads

        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

        # 定义K, Q, V的权重矩阵
        # 多头注意力中K、Q、V的线性层具有相同输入和输出尺寸是一种常见且实用的设计选择！！！
        self.k_linear = nn.Linear(d_model, d_model)
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        # 分头后的维度
        self.d_token = d_model // heads
        # 定义输出权重矩阵
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 计算batch大小
        batch, n, dim_q = x.shape
        # 如果条件为 True，则程序继续执行；如果条件为 False，则程序抛出一个 AssertionError 异常，并停止执行。
        assert dim_q == self.dim_q, f'{x.shape},{self.dim_q},{self.dim_k},{self.dim_v}'  # 确保输入维度与初始化时的dim_q一致

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # 线性变换后的Q, K, V，然后分割成多个头
        q = self.q_linear(q).view(batch, -1, self.heads, self.d_token)
        k = self.k_linear(k).view(batch, -1, self.heads, self.d_token)
        v = self.v_linear(v).view(batch, -1, self.heads, self.d_token)
        # 转置调整维度，以计算注意力分数
        q = q.transpose(1, 2) # 形状变为 [batch, heads, seq_len, d_token]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算自注意力分数
        scores = self.attention(q, k, v, self.d_token)

        # 调整形状以进行拼接
        scores = scores.transpose(1, 2).contiguous().view(batch, -1, self.d_model)

        # 通过输出权重矩阵进行线性变换
        output = self.out(scores)
        return output

    @staticmethod
    def attention(q, k, v, d_token):
        # 计算注意力分数 (q @ k^T) / sqrt(d_token)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_token)
        # 应用softmax归一化（沿着最后一个维度（dim=-1））
        attn = F.softmax(scores, dim=-1)
        # 计算加权的V
        output = torch.matmul(attn, v)
        return output





