import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        x_padded, x_lens = pad_packed_sequence(x, batch_first=True)
        x_lens = x_lens.to(DEVICE)

        x_padded = x_padded[:, :(x_padded.size(1) // 2) * 2, :] # (B, T, dim)

        x_reshaped = x_padded.reshape(x_padded.size(0), x_padded.size(1) // 2, x_padded.size(2) * 2)
        x_lens = x_lens // 2

        x_packed = pack_padded_sequence(x_reshaped, lengths=x_lens, batch_first=True, enforce_sorted=False)


        out, _ = self.blstm(x_packed)
        return out
