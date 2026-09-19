"""CNN baseline model for jet-image top tagging.

Architecture follows the image-based approach described in the README
(Section 3, Approach 1), in the spirit of the DeepTop CNN tagger
(Kasieczka et al., 2017, arXiv:1707.08966): a stack of small
convolutional blocks with batch norm and max pooling, followed by a
dense head with dropout. The input is a single-channel 40x40 jet image
(pT-weighted eta-phi histogram, log-compressed and standardized).
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> (optional) MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class JetImageCNN(nn.Module):
    """A compact CNN for binary top-vs-QCD jet tagging.

    Parameters
    ----------
    img_size : int
        Side length of the (square) jet image, default 40.
    in_channels : int
        Number of image channels (1 for the pT-weighted image).
    """

    def __init__(self, img_size: int = 40, in_channels: int = 1, dropout: float = 0.2):
        super().__init__()
        self.img_size = img_size

        # Three conv blocks with 2x2 pooling: 40 -> 20 -> 10 -> 5.
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, kernel=3, pool=True),   # 40 -> 20
            ConvBlock(32, 64, kernel=3, pool=True),            # 20 -> 10
            ConvBlock(64, 128, kernel=3, pool=True),           # 10 -> 5
        )
        feat_side = img_size // 8  # after three 2x2 pools
        feat_dim = 128 * feat_side * feat_side

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # raw logit; use BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x).squeeze(-1)


def build_model(img_size: int = 40, device: str | torch.device = "cpu") -> JetImageCNN:
    model = JetImageCNN(img_size=img_size)
    return model.to(device)
