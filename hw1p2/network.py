import torch
import torch.nn as nn

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO: Please try different architectures
        in_size = 13
        layers = [
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 40)
        ]
        self.laysers = nn.Sequential(*layers)

    def forward(self, A0):
        x = self.laysers(A0)
        return x