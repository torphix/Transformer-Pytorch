import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxAttention(nn.Module):
    '''
    softmax(Q * K.T / d_k) * V
    '''

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask):
        '''
        Encoder:
            q, k, v are equal in encoder
            mask is for padding
        Decoder:
            q is decoder output
            k, v is encoder output
            mask is padding + lookahead
        '''
        attn = F.softmax((q @ k.transpose(2,3)) / 
                         math.sqrt(k.shape[-1]), dim=-1)
        if mask is not None:
            attn = attn.masked_fill(mask==False, -torch.inf)
        out = attn @ v
        return out, attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, in_d, n_heads):
        super().__init__()
        self.n_heads = n_heads

        self.k_linear = nn.Linear(in_d, n_heads*in_d)
        self.v_linear = nn.Linear(in_d, n_heads*in_d)
        self.q_linear = nn.Linear(in_d, n_heads*in_d)

        self.attention = SoftmaxAttention()
        self.out_layer = nn.Linear(n_heads*in_d, in_d)

        self.dropout = nn.Dropout(p=0.4)
        self.normalize = nn.LayerNorm(in_d)

    def _map_to_heads(self, x):
        '''
        x:[BS, L, N*Heads]
        out: [BS*Heads, L, N]
        '''
        BS, L, N = x.size()
        N = N // self.n_heads
        x = x.view(BS, L, self.n_heads, N)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _map_from_heads(self, x):
        '''
        x: [BS*Heads, L, N]
        out: [BS, L, N*Heads]
        '''
            
        BS, H, L, N = x.size()
        x = x.view(BS, self.n_heads, L, N)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(BS, L, N * self.n_heads)
        return x

    def forward(self, q, k, v, mask=None):
        '''
        Encoder:
            q, k, v are equal in encoder
            mask is for padding
        Decoder:
            q is decoder output
            k, v is encoder output
            mask is padding + lookahead
        '''
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = self._map_to_heads(q)
        k = self._map_to_heads(k)
        v = self._map_to_heads(v)
        
        if mask is not None:
            # [BS, 1, Q_Len, K_len] -> [BS, N_heads, Q_len, K_len]
            mask = mask.repeat(1, self.n_heads, 1, 1)
        else: mask = None

        qkv, attn = self.attention(q, k, v, mask)
        
        qkv = self._map_from_heads(qkv)
        out = self.out_layer(qkv)
        # Regularization
        out = self.normalize(self.dropout(out))
        return out, attn
