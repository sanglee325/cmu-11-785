import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from torchsummaryX import summary

class Network(nn.Module):

    def __init__(self,input_size=13, hidden_size=256, num_layers=3, num_classes=41): 

        super(Network, self).__init__()

        # self.embedding = 
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.2, bidirectional=True, batch_first=True)

        linear_hidden_size = hidden_size * 2
        self.classification = nn.Sequential(
                    nn.Linear(linear_hidden_size,2048),
                    nn.Dropout(0.1),
                    nn.Linear(2048, num_classes)
                    )

    def forward(self, x, X_lens):
        
        packed_input = pack_padded_sequence(x, X_lens, batch_first=True, enforce_sorted=False)

        out1, (out2, out3) = self.lstm(packed_input) # TODO: Pass packed input to self.lstm
        out, lengths = pad_packed_sequence(out1)
        
        out = self.classification(out) 
        out = F.log_softmax(out, dim=2) 

        return out, lengths # TODO: Need to return 2 variables



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    model = Network().to(device)
    print(model)