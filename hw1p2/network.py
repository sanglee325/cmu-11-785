import torch
import torch.nn as nn

class Network(torch.nn.Module):
    def __init__(self, input_size=13):
        super(Network, self).__init__()
        # TODO: Please try different architectures
        in_size = 693

        self.layer1 = nn.Linear(in_size, 256)
        self.layer1_bn = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 256)
        self.layer2_bn = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 256)
        self.layer3_bn = nn.BatchNorm1d(256)
        self.layer4 = nn.Linear(256, 256)
        self.layer4_bn = nn.BatchNorm1d(256)
        self.layer5 = nn.Linear(256, 256)
        self.layer5_bn = nn.BatchNorm1d(256)
        self.layer6 = nn.Linear(256, 128)
        self.layer6_bn = nn.BatchNorm1d(128)
        self.layer7 = nn.Linear(128, 64)
        self.layer7_bn = nn.BatchNorm1d(64)
        self.layer8 = nn.Linear(64, 40)
        self.layer8_bn = nn.BatchNorm1d(40)
        
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer1_bn(self.layer1(inputs)))
        x = self.relu(self.layer2_bn(self.layer2(x)))
        x = self.relu(self.layer3_bn(self.layer3(x)))
        x = self.relu(self.layer4_bn(self.layer4(x)))
        x = self.relu(self.layer5_bn(self.layer5(x)))
        x = self.relu(self.layer6_bn(self.layer6(x)))
        x = self.relu(self.layer7_bn(self.layer7(x)))
        x = self.layer8_bn(self.layer8(x))
        return x