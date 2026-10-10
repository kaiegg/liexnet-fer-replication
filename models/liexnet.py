import torch
import torch.nn as nn
from models.liex_block import LiExBlock


class LiExNet(nn.Module):
    def __init__(self, num_classes=7):
        super(LiExNet, self).__init__()

        # 4 Feature Extraction Blocks
        self.block1 = LiExBlock(in_channels=3, out_channels=24)
        self.block2 = LiExBlock(in_channels=24, out_channels=48)
        self.block3 = LiExBlock(in_channels=48, out_channels=72)
        self.block4 = LiExBlock(in_channels=72, out_channels=96)

        # Global Average Pooling
        # Compresses spatial feature maps from 4x16 down to 1x1 across 96 channels
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Lightweight Classifier Head
        # Matrix size reduces from (6144 x 256) -> (96 x 256)
        self.fc1 = nn.Linear(96, 256)
        self.dropout = nn.Dropout(p=0.30)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Feature Extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Spatial Reduction & Flattening
        x = self.global_pool(x)  # Shape: [Batch, 96, 1, 1]
        x = torch.flatten(x, 1)  # Shape: [Batch, 96]

        # Classification Head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Sanity check for shape and model footprint
    model = LiExNet(num_classes=7)
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)

    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Output Logits Shape: {output.shape}")
    print(f"Total Trainable Parameters: {total_params:,}")
