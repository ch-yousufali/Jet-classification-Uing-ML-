"""Tests for the ANN (MLP) baseline: model + feature pipeline."""

import numpy as np
import pytest
import torch

from ANN.data import N_FEATURES, build_feature_matrix, _wrap_phi
from ANN.model import DenseBlock, JetFeatureANN, build_ann


class TestWrapPhi:
    def test_wraps_into_range(self):
        x = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -5 * np.pi])
        out = _wrap_phi(x)
        # convention: result in [-pi, pi)
        assert np.all(out >= -np.pi) and np.all(out < np.pi)

    def test_small_angle_preserved(self):
        assert _wrap_phi(np.array([0.5]))[0] == pytest.approx(0.5)


class TestBuildFeatureMatrix:
    def test_shape(self):
        raw = np.zeros((4, N_FEATURES), dtype=np.float32)
        assert build_feature_matrix(raw).shape == (4, N_FEATURES)

    def test_all_zero_jet_gives_zeros(self):
        raw = np.zeros((2, N_FEATURES), dtype=np.float32)
        assert np.all(build_feature_matrix(raw) == 0.0)

    def test_padded_constituents_zeroed(self):
        raw = np.zeros((1, N_FEATURES), dtype=np.float32)
        raw[0, :4] = [50.0, 30.0, 10.0, 20.0]  # one real constituent
        X = build_feature_matrix(raw)
        assert np.all(X[0, 4:] == 0.0)
        assert X[0, 0] > 0  # log1p(pT) nonzero

    def test_finite_output(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(0, 50, (8, N_FEATURES)).astype(np.float32)
        assert np.isfinite(build_feature_matrix(raw)).all()


class TestDenseBlock:
    def test_output_shape(self):
        block = DenseBlock(64, 32)
        x = torch.randn(8, 64)
        assert block(x).shape == (8, 32)


class TestJetFeatureANN:
    def test_forward_shape(self):
        model = JetFeatureANN(in_features=N_FEATURES)
        x = torch.randn(16, N_FEATURES)
        assert model(x).shape == (16,)

    def test_forward_single(self):
        # BatchNorm1d requires batch>1 in train mode; single-sample
        # inference runs in eval mode (running stats).
        model = JetFeatureANN()
        model.eval()
        x = torch.randn(1, N_FEATURES)
        with torch.no_grad():
            assert model(x).shape == (1,)

    def test_output_finite(self):
        model = JetFeatureANN()
        x = torch.randn(32, N_FEATURES)
        assert torch.isfinite(model(x)).all()

    def test_param_count_reasonable(self):
        model = JetFeatureANN()
        n = sum(p.numel() for p in model.parameters())
        assert 100_000 < n < 2_000_000

    def test_gradients_flow(self):
        model = JetFeatureANN()
        x = torch.randn(4, N_FEATURES)
        model(x).sum().backward()
        for p in model.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_eval_deterministic(self):
        model = JetFeatureANN()
        model.eval()
        x = torch.randn(4, N_FEATURES)
        with torch.no_grad():
            assert torch.equal(model(x), model(x))

    def test_build_ann_device(self):
        model = build_ann(device="cpu")
        assert next(model.parameters()).device.type == "cpu"
