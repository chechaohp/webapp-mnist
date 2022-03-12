import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self,in_channels, out_channels, n_out, k_size):
        super().__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, k_size)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(5408,100)
        self.fc2 = nn.Linear(100,n_out)

    def forward(self,x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x