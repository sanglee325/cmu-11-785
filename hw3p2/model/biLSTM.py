import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from torchsummaryX import summary
from data.phonemes import *


class Network(nn.Module):

    def __init__(self,input_size=13, hidden_size=256, num_layers=4, num_classes=41): # You can add any extra arguments as you wish

        super(Network, self).__init__()

        # Embedding layer converts the raw input into features which may (or may not) help the LSTM to learn better 
        # For the very low cut-off you dont require an embedding layer. You can pass the input directly to the  LSTM
        # self.embedding = 
        
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)# TODO: # Create a single layer, uni-directional LSTM with hidden_size = 256
        # Use nn.LSTM() Make sure that you give in the proper arguments as given in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        self.lstm = nn.LSTM(input_size,hidden_size=hidden_size,
                            num_layers=num_layers,dropout=0.2, bidirectional=True)
        self.classification = nn.Linear(hidden_size, num_classes)# TODO: Create a single classification layer using nn.Linear()

    def forward(self, x, X_lens): # TODO: You need to pass atleast 1 more parameter apart from self and x
        # x is returned from the dataloader. So it is assumed to be padded with the help of the collate_fn
        packed_input = pack_padded_sequence(x, X_lens, batch_first=True, enforce_sorted=False)# TODO: Pack the input with pack_padded_sequence. Look at the parameters it requires

        out1, (out2, out3) = self.lstm(packed_input) # TODO: Pass packed input to self.lstm
        # As you may see from the LSTM docs, LSTM returns 3 vectors. Which one do you need to pass to the next function?
        out, lengths = pad_packed_sequence(out1) # TODO: Need to 'unpack' the LSTM output using pad_packed_sequence

        out = self.classification(out) # TODO: Pass unpacked LSTM output to the classification layer
        out = F.log_softmax(out, dim=2) # Optional: Do log softmax on the output. Which dimension?

        return out, lengths # TODO: Need to return 2 variables



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    model = Network().to(device)
    print(model)
    summary(model) # x and lx are from the previous cell