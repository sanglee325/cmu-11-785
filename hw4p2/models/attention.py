import torch
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)

        mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
        mask = mask.to(DEVICE)

        energy.masked_fill_(mask, -1e9)
        attention = nn.functional.softmax(energy, dim=1)
        output = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return output, attention