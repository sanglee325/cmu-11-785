import torch
import torch.nn as nn

class Network1024(torch.nn.Module):
    def __init__(self, input_size):
        super(Network1024, self).__init__()
        # TODO: Please try different architectures

        self.layer1 = nn.Linear(input_size, 1024)
        self.layer1_bn = nn.BatchNorm1d(1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer2_bn = nn.BatchNorm1d(1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer3_bn = nn.BatchNorm1d(1024)
        self.layer4 = nn.Linear(1024, 1024)
        self.layer4_bn = nn.BatchNorm1d(1024)
        self.layer5 = nn.Linear(1024, 512)
        self.layer5_bn = nn.BatchNorm1d(512)
        self.layer6 = nn.Linear(512, 256)
        self.layer6_bn = nn.BatchNorm1d(256)
        self.layer7 = nn.Linear(256, 128)
        self.layer7_bn = nn.BatchNorm1d(128)
        self.layer8 = nn.Linear(128, 40)
        self.layer8_bn = nn.BatchNorm1d(40)
        
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer1_bn(self.layer1(inputs)))
        x = self.relu(self.layer2_bn(self.layer2(x)))
        x = self.relu(self.layer3_bn(self.layer3(x)))
        x = self.relu(self.layer4_bn(self.layer4(x)))
        x = self.dropout(x)
        x = self.relu(self.layer5_bn(self.layer5(x)))
        x = self.relu(self.layer6_bn(self.layer6(x)))
        x = self.relu(self.layer7_bn(self.layer7(x)))
        x = self.dropout(x)
        x = self.relu(self.layer8_bn(self.layer8(x)))
        return x


class Network2048(torch.nn.Module):
    def __init__(self, input_size):
        super(Network2048, self).__init__()

        self.layer1 = nn.Linear(input_size, 2048)
        self.layer1_bn = nn.BatchNorm1d(2048)
        self.layer2 = nn.Linear(2048, 2048)
        self.layer2_bn = nn.BatchNorm1d(2048)
        self.layer3 = nn.Linear(2048, 2048)
        self.layer3_bn = nn.BatchNorm1d(2048)
        self.layer4 = nn.Linear(2048, 2048)
        self.layer4_bn = nn.BatchNorm1d(2048)
        self.layer5 = nn.Linear(2048, 2048)
        self.layer5_bn = nn.BatchNorm1d(2048)
        self.layer6 = nn.Linear(2048, 2048)
        self.layer6_bn = nn.BatchNorm1d(2048)
        self.layer7 = nn.Linear(2048, 512)
        self.layer7_bn = nn.BatchNorm1d(512)
        self.layer8 = nn.Linear(512, 40)
        self.layer8_bn = nn.BatchNorm1d(40)
        
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer1_bn(self.layer1(inputs)))
        x = self.dropout(x)
        x = self.relu(self.layer2_bn(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.layer3_bn(self.layer3(x)))
        x = self.dropout(x)
        x = self.relu(self.layer4_bn(self.layer4(x)))
        x = self.dropout(x)
        x = self.relu(self.layer5_bn(self.layer5(x)))
        x = self.dropout(x)
        x = self.relu(self.layer6_bn(self.layer6(x)))
        x = self.dropout(x)
        x = self.relu(self.layer7_bn(self.layer7(x)))
        x = self.dropout(x)
        x = self.relu(self.layer8_bn(self.layer8(x)))
        x = self.dropout(x)

        return x
