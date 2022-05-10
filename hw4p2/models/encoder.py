import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from pBLSTM import pBLSTM

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128, key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        self.pBLSTMs = nn.Sequential(
            pBLSTM(hidden_dim*4, hidden_dim),
            pBLSTM(hidden_dim*4, hidden_dim),
            pBLSTM(hidden_dim*4, hidden_dim),
            pBLSTM(hidden_dim*4, hidden_dim)
        )

        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens):
        rnn_inp = pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)

        outputs, _ = self.lstm(rnn_inp)

        outputs = self.pBLSTMs(outputs)

        linear_input, encoder_lens = pad_packed_sequence(outputs, batch_first=True)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)
        return keys, value, encoder_lens
