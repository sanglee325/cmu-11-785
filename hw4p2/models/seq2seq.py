import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, vocab_size, encoder_dim=144, decoder_dim=512, embed_dim=256, key_value_size=128):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_dim)
        self.decoder = Decoder(vocab_size, decoder_dim, embed_dim)

    def forward(self, x, x_len, y=None, mode='train'):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, mode=mode)
        return predictions