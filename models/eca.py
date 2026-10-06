import math
import torch
import torch.nn as nn

class ECAModule(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super(ECAModule, self).__init__()
        # Calculate adaptive kernel size k
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 != 0 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)
