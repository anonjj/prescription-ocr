"""
Spatial Transformer Network (STN) for learnable geometric rectification.

Placed before the CNN backbone, the STN learns to undo perspective distortion,
rotation, and slant that are common in phone-camera prescription images.

Reference: Jaderberg et al., "Spatial Transformer Networks", NeurIPS 2015.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMG_HEIGHT, IMG_WIDTH


class STN(nn.Module):
    """
    Lightweight Spatial Transformer Network.

    Localization network: 3 conv layers → FC → 6-parameter affine transform.
    Initialized to identity so the model starts as a pass-through.

    Input:  (B, 1, H, W)
    Output: (B, 1, H, W) — geometrically rectified
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Localization network — small CNN to predict affine params
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),  # fixed spatial size
        )

        # Regression head: 128 * 4 * 8 = 4096 → 6 affine parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 6),
        )

        # Initialize to identity transform: [[1,0,0],[0,1,0]]
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 1, H, W)

        Returns:
            Rectified tensor (B, 1, H, W)
        """
        # Predict affine parameters
        features = self.localization(x)
        features = features.view(features.size(0), -1)
        theta = self.fc_loc(features)
        theta = theta.view(-1, 2, 3)

        # Apply spatial transform
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        rectified = F.grid_sample(x, grid, align_corners=False,
                                  padding_mode='border')
        return rectified


if __name__ == "__main__":
    # Quick shape test
    stn = STN(in_channels=1)
    x = torch.randn(2, 1, IMG_HEIGHT, IMG_WIDTH)
    out = stn(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    n_params = sum(p.numel() for p in stn.parameters() if p.requires_grad)
    print(f"STN params: {n_params:,}")
