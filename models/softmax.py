import torch
from torch import nn

class Softmax(nn.Module):
    def __init__(self, input, out):
        super(Softmax, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input, out),
        )

    def forward(self, x):
        x = self.model(x)
        return x