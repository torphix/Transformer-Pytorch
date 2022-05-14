

class Transformer(nn.Module):
    def __init__(self, enc_in_d, enc_n_blocks, enc_n_heads,
                 dec_in_d, dec_n_blocks, dec_n_heads, pad_token):
        super().__init__()
        self.encoder = Encoder(enc_n_blocks, enc_in_d, enc_n_heads, pad_token=pad_token)
        self.decoder = Decoder(dec_n_blocks, dec_in_d, dec_n_heads, pad_token=pad_token)
        
        self.fc_out = nn.Linear(config['decoder']['in_d'], config['out_d'])
    
    def forward(self, src, tgt, src_lens, tgt_lens, return_attn=False):
        '''
        Lens are used to compute the padding and look ahead masks
        Note this transformer has been designed with the idea
        that embedding has already taken place and expects tensors:
        - src: [BS, L, N], - tgt: [BS, L, N] as well as a list of lengths
        for correct padding of each corresponding tensor
        '''
        enc_out, enc_attn = self.encoder(src, src_lens)
        dec_out, dec_attn = self.decoder(tgt, enc_out, src_lens, tgt_lens)
        if return_attn:
            return enc_out, enc_attn, dec_out, dec_attn
        else:
            return enc_out, dec_out
