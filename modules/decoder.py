import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, in_d, n_heads):
        super().__init__()
        self.dec_self_attn = MultiHeadedAttention(in_d, n_heads)
        self.regularisation = nn.Sequential(
            nn.LayerNorm(in_d),
            nn.Dropout(0.2))
        self.mha = MultiHeadedAttention(in_d, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(in_d, in_d),
            nn.LayerNorm(in_d),
            nn.Dropout(0.2),
            nn.ReLU())
        
    def forward(self, tgt, src, tgt_mask=None, tgt_src_mask=None):
        '''
        tgt_mask = lookahead mask prevention
        tgt_src_mask = combined src & tgt padding masks 
        '''
        tgt, attn = self.dec_self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.regularisation(tgt)
        x, attn = self.mha(tgt, src, src, tgt_src_mask)
        x = self.ff(x)
        return x, attn


class Decoder(nn.Module):
    def __init__(self, n_blocks, in_d, n_heads, pad_token):
        super().__init__()
        self.pad_token = pad_token
        self.layers = nn.ModuleList()
        
        for i in range(n_blocks):
            self.layers.append(
                DecoderBlock(in_d, n_heads))
        
    def get_pad_mask(self, src_lens, tgt_lens):
        # Returns: [BS, 1, max_tgt_len, max_src_len]
        batch_size = src_lens.shape[0]
        max_tgt_len = torch.max(tgt_lens).item()
        max_src_len = torch.max(src_lens).item()
        ids = torch.arange(
            0, max_tgt_len, requires_grad=False).unsqueeze(0).expand(batch_size, -1)
        mask = ids <= src_lens.unsqueeze(1)
        tgt_mask = mask.unsqueeze(-1).repeat(1, 1, max_src_len)
        ids = torch.arange(
            0, max_src_len, requires_grad=False).unsqueeze(0).expand(batch_size, -1)
        src_mask = ids <= src_lens.unsqueeze(1)
        src_mask = src_mask.unsqueeze(1).repeat(1, max_tgt_len, 1)
        mask = tgt_mask & src_mask
        return mask.unsqueeze(1)

    def get_lookahead_mask(self, q, k):
        q_len, k_len = q.shape[1], k.shape[1]
        mask = torch.tril(torch.ones(q_len, k_len)).bool()
        return mask
    
    def forward(self, tgt, src, src_lens=None, tgt_lens=None):
        '''Mask should be pad + lookahead mask'''
        tgt_mask = self.get_pad_mask(tgt_lens, tgt_lens)
        lookahead_mask = self.get_lookahead_mask(tgt, tgt)
        tgt_mask = tgt_mask * lookahead_mask
        tgt_src_mask = self.get_pad_mask(src_lens, tgt_lens)
        res = tgt
        for layer in self.layers:
            tgt, attn = layer(tgt, src, tgt_mask, tgt_src_mask)
            tgt += res
            res = tgt
        return tgt, attn
