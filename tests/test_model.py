"""Unit tests for the CNN model (src.model).

These tests do not need the dataset or a GPU; they only check tensor
shapes, parameter counts, and forward-pass behaviour on random input.
Run with:

    .venv/bin/python -m pytest tests/test_model.py -v
"""

from __future__ import annotations

import pytest
import torch

from CNN.model import JetImageCNN, build_model
from CNN.model import ConvBlock


# --------------------------------------------------------------------------- #
# ConvBlock
# --------------------------------------------------------------------------- #


class TestConvBlock:
    def test_output_shape_with_pool(self):
        block = ConvBlock(1, 16, kernel=3, pool=True)
        x = torch.randn(2, 1, 40, 40)
        out = block(x)
        # 40 -> 20 after 2x2 maxpool
        assert out.shape == (2, 16, 20, 20)

    def test_output_shape_without_pool(self):
        block = ConvBlock(1, 16, kernel=3, pool=False)
        x = torch.randn(2, 1, 40, 40)
        out = block(x)
        assert out.shape == (2, 16, 40, 40)

    def test_relu_output_nonnegative(self):
        block = ConvBlock(1, 8, kernel=3, pool=False)
        block.eval()
        x = torch.randn(4, 1, 12, 12)
        with torch.no_grad():
            out = block(x)
        assert torch.all(out >= 0)


# --------------------------------------------------------------------------- #
# JetImageCNN
# --------------------------------------------------------------------------- #


class TestJetImageCNN:
    def test_forward_default_shape(self):
        model = JetImageCNN(img_size=40)
        x = torch.randn(8, 1, 40, 40)
        out = model(x)
        # single logit per jet, squeezed
        assert out.shape == (8,)

    def test_forward_custom_img_size(self):
        model = JetImageCNN(img_size=32)
        x = torch.randn(4, 1, 32, 32)
        out = model(x)
        assert out.shape == (4,)

    def test_forward_single_batch(self):
        model = JetImageCNN(img_size=40)
        x = torch.randn(1, 1, 40, 40)
        out = model(x)
        assert out.shape == (1,)

    def test_output_is_finite(self):
        model = JetImageCNN(img_size=40)
        model.eval()
        x = torch.randn(4, 1, 40, 40)
        with torch.no_grad():
            out = model(x)
        assert torch.all(torch.isfinite(out))

    def test_param_count_reasonable(self):
        # The architecture has ~913k params; assert it is in a sane range
        # so silent regressions (e.g. a dropped layer) are caught.
        model = JetImageCNN(img_size=40)
        n = sum(p.numel() for p in model.parameters())
        assert 800_000 < n < 1_100_000

    def test_has_batchnorm(self):
        model = JetImageCNN(img_size=40)
        bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert len(bn_modules) == 3  # one per ConvBlock

    def test_has_dropout(self):
        model = JetImageCNN(img_size=40)
        dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(dropout_modules) == 1

    def test_gradients_flow(self):
        model = JetImageCNN(img_size=40)
        x = torch.randn(4, 1, 40, 40)
        y = torch.randint(0, 2, (4,)).float()
        loss = torch.nn.BCEWithLogitsLoss()(model(x), y)
        loss.backward()
        # every parameter should have a gradient
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient for {name}"

    def test_build_model_moves_to_device(self):
        model = build_model(img_size=40, device="cpu")
        x = torch.randn(2, 1, 40, 40)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2,)

    def test_eval_mode_is_deterministic(self):
        model = JetImageCNN(img_size=40)
        model.eval()
        x = torch.randn(4, 1, 40, 40)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)
