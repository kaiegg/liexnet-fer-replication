import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eca import ECAModule

class LiExBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(LiExBlock, self).__init__()

        # Sublayer 1: Conv2D -> ReLU -> BN
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Sublayer 2: Depthwise Conv2D -> ReLU -> BN
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Sublayer 3: ECA Module
        self.eca = ECAModule(channels=out_channels)

        # Sublayer 4: Pointwise 1x1 Conv -> MaxPool2D
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        x = F.relu(self.dwconv(x))
        x = self.bn2(x)

        x = self.eca(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x
