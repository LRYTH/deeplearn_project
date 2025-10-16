from torch import nn
import torch

class Linear(nn.Module):
    def __init__(self, input, out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input, out)

    def forward(self, x):
        x = self.linear(x)
        return x