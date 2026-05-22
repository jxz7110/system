from typing import Callable, Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
import numpy as np
from layers.model_layers import *
from layers.mixF_layers import *
from layers.mixF import *
from layers.chain_mixF import *

class hardBackbone(nn.Module):
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
        self.mixF = CnnBlock(self.d_model, self.n_heads, self.e_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.d_model*2),
            nn.GELU(),
            nn.Linear(self.d_model*2, self.pred_len)
        )
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

        # multi-head attention部分，进行多频处理
        x = self.mixF(x + self.W_pos)
        
        x = self.head(x)
        x = torch.reshape(x, (bs, nvars, -1))
        x = x.permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, 'norm')
        return x
    
class CnnBlock(nn.Module):
    def __init__(self, d_model, n_heads, e_layers, res_attention = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers =e_layers

        self.layers = nn.ModuleList([mixFBlock(d_model=d_model, num_heads=n_heads, res_attention=res_attention) for i in range(e_layers)])
        self.res_attention = res_attention

    def forward(self, x):
        output = x
        high_score = None
        low_score = None
        if self.res_attention:
            for mod in self.layers: 
                output, high_score, low_score = mod(output, high_prev=high_score, low_prev=low_score)
            return output
        else:
            for mod in self.layers: 
                output = mod(output)
            return output



class mixFBlock(nn.Module):
    def __init__(self, d_model, num_heads, window_size=2, alpha=0.2, res_attention=False, d_ff=128,
                 attn_dropout=0.2, dropout=0.3, bias=True, activation='relu', pre_norm=False,
                 norm='BatchNorm'):
        super().__init__()
        self.res_attention = res_attention
        self.high_layer = TSTEncoderLayer(d_model, n_heads=num_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=False) 
        low_dim = d_model // window_size
        self.high2low = nn.Sequential(Transpose(1, 2),
                                        nn.Conv1d(d_model, low_dim, kernel_size=window_size, groups=low_dim, padding='same'),
                                        Transpose(1, 2),
                                        )
        self.low_layer = TSTEncoderLayer(low_dim, n_heads=num_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=False)
        self.output_mix_layer = nn.Sequential(nn.Linear(low_dim + d_model, d_model*2),
                                                nn.GELU(),
                                                nn.Linear(d_model*2, d_model),
                                                )
    def forward(self, x, high_prev:Optional[Tensor]=None, low_prev:Optional[Tensor]=None):
        if self.res_attention:
            high_output, high_score = self.high_layer(x, high_prev)
            low_x = self.high2low(x)
            low_output, low_score = self.low_layer(low_x, low_prev)
            # output = self.output_mix_layer(torch.cat((high_output,low_output), -1))
            return high_output, high_score, low_score
        else:
            high_output = self.high_layer(x)
            low_x = self.high2low(x)
            low_output = self.low_layer(low_x)
            return self.output_mix_layer(torch.cat((high_output,low_output), -1))


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights