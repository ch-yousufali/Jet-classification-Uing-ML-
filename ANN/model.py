"""ANN (MLP) baseline model for top-quark jet tagging.

Input is the flattened raw constituent feature vector:
200 constituents x [E, PX, PY, PZ] = 800 features per jet,
zero-padded and standardized. The model is a plain feed-forward
network (dense layers + batch norm + dropout) — the simplest
possible neural-net baseline, serving as the "shallow learning"
comparison point against the CNN and (later) GNN models.
"""

from __future__ import annotations

import torch
from torch import nn


class DenseBlock(nn.Module):
    """Linear -> BatchNorm -> ReLU -> Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class JetFeatureANN(nn.Module):
    """Fully-connected classifier on the 800-dim constituent feature vector.

    Parameters
    ----------
    in_features : int
        Input dimension (default 800 = 200 constituents x 4).
    hidden : tuple of int
        Hidden layer widths.
    dropout : float
        Dropout probability in each dense block.
    """

    def __init__(
        self,
        in_features: int = 800,
        hidden: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_features
        for h in hidden:
            layers.append(DenseBlock(dim, h, dropout))
            dim = h
        layers.append(nn.Linear(dim, 1))  # raw logit; use BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_ann(
    in_features: int = 800,
    hidden: tuple[int, ...] = (512, 256, 128),
    device: str | torch.device = "cpu",
    dropout: float = 0.2,
) -> JetFeatureANN:
    model = JetFeatureANN(in_features=in_features, hidden=hidden, dropout=dropout)
    return model.to(device)
