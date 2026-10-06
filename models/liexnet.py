import torch
import torch.nn as nn
import torch.nn.functional as F
from models.liex_block import LiExBlock

class LiExNet(nn.Module):
    def __init__(self, num_classes: int = 7):
        super(LiExNet, self).__init__()

        self.block1 = LiExBlock(in_channels=3, out_channels=24)
        self.block2 = LiExBlock(in_channels=24, out_channels=48)
        self.block3 = LiExBlock(in_channels=48, out_channels=72)
        self.block4 = LiExBlock(in_channels=72, out_channels=96)

        # Feature map size at Block 4 output is 8x8 with 96 channels (for 128x128 input)
        self.flatten_dim = 96 * 8 * 8  # 6144
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout = nn.Dropout(p=0.30)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
