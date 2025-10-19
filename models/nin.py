
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

class NinNet(nn.Module):
    def __init__(self):
        super(NinNet, self).__init__()
        self.model = nn.Sequential(
            nin_block(1, 96, 11, 4, 0),
            nn.MaxPool2d(3, 2),
            nin_block(96, 256, 5, 1, 2),
            nn.MaxPool2d(3, 2),
            nin_block(256, 384, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.Dropout(0.5),
            nin_block(384, 10, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.model(x)
        return x