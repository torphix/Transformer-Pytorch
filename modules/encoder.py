import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_d, n_heads):
        super().__init__()
        self.mha = MultiHeadedAttention(in_d, n_heads)
        self.regularisation = nn.Sequential(
            nn.LayerNorm(in_d),
            nn.Dropout(0.2))
        self.ff = nn.Sequential(
            nn.Linear(in_d, in_d),
            nn.LayerNorm(in_d),
            nn.ReLU())
        
    def forward(self, q, k, v, pad_mask=None):
        '''
        Mask should be pad mask for encoder
        & pad + lookahead mask for decoder
        '''
        x, attn = self.mha(q, k, v, pad_mask)
        x = self.regularisation(x)
        x = self.ff(x)
        return x, attn
    
    
class Encoder(nn.Module):
    def __init__(self, n_blocks, in_d, n_heads, pad_token):
        super().__init__()
        self.pad_token = pad_token
        self.layers = nn.ModuleList()
        for i in range(n_blocks):
            self.layers.append(
                EncoderBlock(in_d, n_heads))
        
    def get_pad_mask(self, lengths=None):
        if lengths is None: return 
        # Assumes pad token == 0
        batch_size = lengths.shape[0]
        max_len = torch.max(lengths).item()
        ids = torch.arange(
            0, max_len, requires_grad=False).unsqueeze(0).expand(batch_size, -1)
        mask = ids <= lengths.unsqueeze(1).expand(-1, max_len)
        mask = mask.unsqueeze(1).repeat(1, max_len, 1)
        mask = mask & mask.transpose(1,2)
        return mask.unsqueeze(1)
        
    def forward(self, x, lengths=None):
        mask = self.get_pad_mask(lengths)
        res = x
        for layer in self.layers:
            x, attn = layer(x, x, x, mask)
            x += res
            res = x
        return x, attn
