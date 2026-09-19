"""Unit tests for the data layer (src.data).

Two flavours of tests:

  1. Pure tests of jet-image construction and kinematics that need no
     downloaded data (run anywhere, fast).
  2. Dataset tests that require the Zenodo HDF5 files to be present in
     `data/`. These are skipped automatically if the files are missing,
     so `pytest` works on a fresh checkout without a 1.4 GB download.

Run with:

    .venv/bin/python -m pytest tests/test_data.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from CNN import (
    DatasetConfig,
    build_jet_images,
    load_split_arrays,
)
from CNN.jet_image import _delta_phi

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")


def _has(split: str) -> bool:
    return os.path.exists(os.path.join(DATA_DIR, f"{split}.h5"))


# --------------------------------------------------------------------------- #
# _delta_phi
# --------------------------------------------------------------------------- #


class TestDeltaPhi:
    def test_zero(self):
        out = _delta_phi(np.array([0.0]), np.array([0.0]))
        assert np.isclose(out[0], 0.0)

    def test_wraps_around_pi(self):
        # phi1 = pi - eps, phi2 = -pi + eps -> the short way around is
        # negative (going from -pi+eps up to pi-eps is +2eps, so the
        # signed delta phi1-phi2 wraps to -2eps).
        eps = 1e-3
        out = _delta_phi(np.array([np.pi - eps]), np.array([-np.pi + eps]))
        assert np.isclose(out[0], -2 * eps, atol=1e-6)

    def test_wraps_around_minus_pi(self):
        eps = 1e-3
        out = _delta_phi(np.array([-np.pi + eps]), np.array([np.pi - eps]))
        assert np.isclose(out[0], 2 * eps, atol=1e-6)

    def test_in_range(self):
        phi1 = np.linspace(-np.pi, np.pi, 50)
        phi2 = np.linspace(np.pi, -np.pi, 50)
        out = _delta_phi(phi1, phi2)
        assert np.all(out >= -np.pi - 1e-6)
        assert np.all(out <= np.pi + 1e-6)


# --------------------------------------------------------------------------- #
# build_jet_images (synthetic input, no download needed)
# --------------------------------------------------------------------------- #


def _toy_jets(n: int = 5, n_const: int = 200):
    rng = np.random.default_rng(0)
    E = rng.uniform(10, 300, (n, n_const)).astype(np.float32)
    PX = rng.uniform(-100, 100, (n, n_const)).astype(np.float32)
    PY = rng.uniform(-100, 100, (n, n_const)).astype(np.float32)
    PZ = rng.uniform(-100, 100, (n, n_const)).astype(np.float32)
    pt = np.sqrt(PX**2 + PY**2)
    pt_safe = np.where(pt > 0, pt, 1.0)
    Eta = np.arcsinh(PZ / pt_safe).astype(np.float32)
    Phi = np.arctan2(PY, PX).astype(np.float32)
    return E, PX, PY, PZ, Eta, Phi


class TestBuildJetImages:
    def test_output_shape(self):
        E, PX, PY, PZ, Eta, Phi = _toy_jets(n=7)
        imgs = build_jet_images(E, PX, PY, PZ, Eta, Phi, img_size=40)
        assert imgs.shape == (7, 1, 40, 40)
        assert imgs.dtype == np.float32

    def test_custom_img_size(self):
        E, PX, PY, PZ, Eta, Phi = _toy_jets(n=3)
        imgs = build_jet_images(E, PX, PY, PZ, Eta, Phi, img_size=24)
        assert imgs.shape == (3, 1, 24, 24)

    def test_per_image_standardized(self):
        E, PX, PY, PZ, Eta, Phi = _toy_jets(n=4)
        imgs = build_jet_images(E, PX, PY, PZ, Eta, Phi, img_size=40)
        # each image has approximately mean 0 (standardization step)
        means = imgs.mean(axis=(1, 2, 3))
        assert np.allclose(means, 0.0, atol=1e-4)

    def test_handles_all_zero_jet(self):
        # a jet with all-zero constituents should not crash
        n = 2
        E = np.zeros((n, 200), np.float32)
        PX = np.zeros((n, 200), np.float32)
        PY = np.zeros((n, 200), np.float32)
        PZ = np.zeros((n, 200), np.float32)
        Eta = np.zeros((n, 200), np.float32)
        Phi = np.zeros((n, 200), np.float32)
        imgs = build_jet_images(E, PX, PY, PZ, Eta, Phi, img_size=40)
        assert imgs.shape == (n, 1, 40, 40)
        assert np.all(np.isfinite(imgs))

    def test_output_is_finite(self):
        E, PX, PY, PZ, Eta, Phi = _toy_jets(n=5)
        imgs = build_jet_images(E, PX, PY, PZ, Eta, Phi, img_size=40)
        assert np.all(np.isfinite(imgs))


# --------------------------------------------------------------------------- #
# load_split_arrays + JetImageDataset (need downloaded data)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has("val"), reason="data/val.h5 not present")
class TestLoadSplitArrays:
    def test_shapes_and_label_balance(self):
        path = os.path.join(DATA_DIR, "val.h5")
        E, PX, PY, PZ, Eta, Phi, y = load_split_arrays(path, max_events=500)
        assert E.shape == (500, 200)
        assert PX.shape == (500, 200)
        assert Eta.shape == (500, 200)
        assert y.shape == (500,)
        assert set(np.unique(y).tolist()).issubset({0.0, 1.0})

    def test_eta_phi_finite_for_active_constituents(self):
        path = os.path.join(DATA_DIR, "val.h5")
        E, PX, PY, PZ, Eta, Phi, y = load_split_arrays(path, max_events=100)
        active = E > 0
        assert np.all(np.isfinite(Eta[active]))
        assert np.all(np.isfinite(Phi[active]))
        # phi is in [-pi, pi]
        assert np.all(Phi[active] >= -np.pi - 1e-4)
        assert np.all(Phi[active] <= np.pi + 1e-4)

    def test_energy_dominates_momentum(self):
        # E^2 >= PX^2 + PY^2 + PZ^2 for on-shell particles
        path = os.path.join(DATA_DIR, "val.h5")
        E, PX, PY, PZ, Eta, Phi, y = load_split_arrays(path, max_events=100)
        active = E > 0
        m2 = E[active] ** 2 - (PX[active] ** 2 + PY[active] ** 2 + PZ[active] ** 2)
        # allow tiny numerical tolerance
        assert np.all(m2 > -1.0)


@pytest.mark.skipif(not _has("val"), reason="data/val.h5 not present")
class TestJetImageDataset:
    def test_dataset_len_and_item(self):
        cfg = DatasetConfig(data_dir=str(DATA_DIR), img_size=40, max_events=200)
        ds = __import__("CNN", fromlist=["JetImageDataset"]).JetImageDataset(
            "val", cfg
        )
        assert len(ds) == 200
        x, y = ds[0]
        assert x.shape == (1, 40, 40)
        assert x.dtype.is_floating_point
        assert y.ndim == 0
