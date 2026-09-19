"""Smoke test for one training step (src.train.run_epoch).

Uses synthetic data so it runs anywhere without downloading the dataset.
Verifies that a forward + backward + optimizer step actually reduces the
loss and produces a valid AUC.

Run with:

    .venv/bin/python -m pytest tests/test_train.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from CNN.model import build_model
from CNN.train import run_epoch


def _synthetic_loader(n: int = 64, img_size: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Make signal jets slightly brighter so the model can learn *something*.
    x = rng.normal(0, 1, (n, 1, img_size, img_size)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.float32)
    x[y == 1] += 0.5
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=16, shuffle=True)


class TestRunEpoch:
    def test_train_step_runs_and_returns_auc(self):
        device = torch.device("cpu")
        model = build_model(img_size=40, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        loader = _synthetic_loader(n=64)
        loss, acc, auc = run_epoch(model, loader, opt, criterion, device, desc="train")
        assert np.isfinite(loss)
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= auc <= 1.0

    def test_eval_step_no_grad(self):
        device = torch.device("cpu")
        model = build_model(img_size=40, device=device)
        criterion = nn.BCEWithLogitsLoss()
        loader = _synthetic_loader(n=32)
        loss, acc, auc = run_epoch(model, loader, None, criterion, device, desc="val")
        assert np.isfinite(loss)
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= auc <= 1.0

    def test_loss_decreases_over_steps(self):
        # A few train epochs on separable synthetic data should reduce loss.
        device = torch.device("cpu")
        model = build_model(img_size=40, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        loader = _synthetic_loader(n=128)
        first, *_ = run_epoch(model, loader, opt, criterion, device)
        second, *_ = run_epoch(model, loader, opt, criterion, device)
        assert second < first
