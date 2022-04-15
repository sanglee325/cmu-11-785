import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from torchsummaryX import summary

class Network(nn.Module):

    def __init__(self,input_size=13, hidden_size=256, num_layers=4, num_classes=41): 

        super(Network, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv1d(input_size,hidden_size, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Conv1d(hidden_size,hidden_size, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(hidden_size,hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size,hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.2),
            nn.GELU()
        )
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=0.2, bidirectional=True, batch_first=True)

        linear_hidden_size = hidden_size * 2
        self.classification = nn.Sequential(
                    nn.Linear(linear_hidden_size,2048),
                    nn.Dropout(p=0.1),
                    nn.GELU(),
                    nn.Linear(2048, num_classes)
                )

    def forward(self, x, X_lens):
        emb = self.embedding(x.permute(0,2,1)).permute(0,2,1)
        packed_input = pack_padded_sequence(emb, X_lens//4, batch_first=True, enforce_sorted=False)

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