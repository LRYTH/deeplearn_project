import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )


    def forward(self, x):
        x = self.model(x)
        return x

class MLP_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, config):
        super(MLP_dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(hidden_size, output_size),
        )


    def forward(self, x):
        x = self.model(x)
        return x