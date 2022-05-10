import torch
import torch.nn as nn

from models.attention import Attention

import sys
sys.path.append('../') 

from config import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Decoder(nn.Module):
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=embed_dim + key_value_size, hidden_size=decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)
    
        self.attention = Attention()     
        self.vocab_size = vocab_size
        self.character_prob = nn.Linear(key_value_size*2, vocab_size) #: d_v -> vocab_size
        self.key_value_size = key_value_size
        
        self.character_prob.weight = self.embedding.weight

    def forward(self, key, value, encoder_len, y=None, mode='train'):
        '''
        Args:
            key :(B, T, d_k) - Output of the Encoder (possibly from the Key projection layer)
            value: (B, T, d_v) - Output of the Encoder (possibly from the Value projection layer)
            y: (B, text_len) - Batch input of text with text_length
            mode: Train or eval mode for teacher forcing
        Return:
            predictions: the character perdiction probability 
        '''

        B, key_seq_max_len, key_value_size = key.shape

        if mode == 'train':
            max_len =  y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len = 600

        mask = encoder_len
        mask = mask.to(DEVICE)
        
        predictions = []
        prediction = torch.full((B,), fill_value=0, device=DEVICE)
        hidden_states = [None, None] 
        
        # TODO: Initialize the context
        context = value[:, 0, :]

        attention_plot = [] # this is for debugging

        for i in range(max_len):
            if mode == 'train':
                if i == 0:
                    start_char = torch.zeros(BATCH_SIZE, dtype=torch.long).fill_(letter2index['<sos>']).to(DEVICE)
                    char_embed = self.embedding(start_char)
                else:
                    # Use ground truth
                    char_embed = char_embeddings[:, i-1, :]
            else:
                if i == 0:
                    start_char = torch.zeros(BATCH_SIZE, dtype=torch.long).fill_(letter2index['<sos>']).to(DEVICE)
                    char_embed = self.embedding(start_char)
                else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))

            # what vectors should be concatenated as a context?
            y_context = torch.cat([char_embed, context], dim=1)
            # context and hidden states of lstm 1 from the previous time step should be fed
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])

            # hidden states of lstm1 and hidden states of lstm2 from the previous time step should be fed
            hidden_states[1] = self.lstm2(hidden_states[0][0], hidden_states[1]) # output (h_1, c_1)
            # What then is the query?
            query = hidden_states[1][0]
            
            # Compute attention from the output of the second LSTM Cell
            context, attention = self.attention(query, key, value, mask)
            # We store the first attention of this batch for debugging
            attention_plot.append(attention[0].detach().cpu())
            
            # What should be concatenated as the output context?
            output_context = torch.cat([query, context], dim=1)
            prediction = self.character_prob(output_context)
            # store predictions
            predictions.append(prediction.unsqueeze(1))
        
        # Concatenate the attention and predictions to return
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)
        return predictions, attentions