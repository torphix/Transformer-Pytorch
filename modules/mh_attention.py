import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttention(nn.Module):
    '''
    softmax(Q * K.T / d_k) * V
    '''

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v, mask=None):
        '''
        q, k, v are equal in encoder
        q, k are encoder output in decoder
        v is decoder embedding in decoder
        mask is for padding
        '''
        attn = F.softmax((q @ k.transpose(1, 2)) / math.sqrt(self.d_k), dim=-1)
        if mask is not None:
            attn.masked_fill(mask, -torch.inf)
        out = attn @ v
        return out, attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, in_d, d_q, d_k, d_v, n_heads):
        super().__init__()
        self.n_heads = n_heads

        self.k_linear = nn.Linear(in_d, n_heads*d_k)
        self.v_linear = nn.Linear(in_d, n_heads*d_v)
        self.q_linear = nn.Linear(in_d, n_heads*d_q)

        self.attention = SoftmaxAttention(d_k)
        self.out_layer = nn.Linear(n_heads*d_v, in_d)

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
        x = x.view(BS * self.n_heads, L, N)
        return x

    def _map_from_heads(self, x):
        '''
        x: [BS*Heads, L, N]
        out: [BS, L, N*Heads]
        '''
        BS, L, N = x.size()
        BS = BS // self.n_heads
        x = x.view(BS, self.n_heads, L, N)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(BS, L, N * self.n_heads)
        return x

    def forward(self, q, k, v):
        '''
        q, k, v are equal in encoder
        q, k are encoder output in decoder
        v is decoder embedding in decoder
        '''
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = self._map_to_heads(q)
        k = self._map_to_heads(k)
        v = self._map_to_heads(v)

        qkv, attn = self.attention(q, k, v)
        qkv = self._map_from_heads(qkv)
        out = self.out_layer(qkv)
        # Regularization
        out = self.normalize(self.dropout(out))
        return out


k, q, v = torch.randn((4, 57, 128)), torch.randn(
    (4, 57, 128)), torch.randn((4, 57, 128))


mha = MultiHeadedAttention(128, 256, 256, 256, 8)

out = mha(q, k, v)
print(out.size())
