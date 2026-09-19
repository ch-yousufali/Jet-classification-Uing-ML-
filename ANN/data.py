"""Feature loading for the ANN baseline.

The ANN consumes per-constituent kinematic features built from the raw
four-momenta: each jet row in the Zenodo HDF5 files stores
`values_block_0` = 200 constituents x [E, PX, PY, PZ] interleaved (800
floats) followed by 4 truth values (dropped). The label is
`values_block_1[:, 1]` (is_signal_new).

Per constituent we compute the jet-relative features
    [ log1p(pT), eta_i - eta_jet, delta_phi(phi_i - phi_jet), log1p(E) ]
where (eta_jet, phi_jet) is the pT-weighted jet axis. These are
translation-invariant (relative to the jet axis) — a much easier input
for a flat MLP than raw lab-frame momenta. Inactive (zero-pT, padded)
constituents are masked to all-zero feature rows.

Preprocessing, fit on the TRAIN split only:
  per-feature standardization (mean 0, std 1) over the 800-dim vector.

Processed feature matrices are cached to `.npz` under `cache_dir` so
re-runs and evaluation skip the HDF5 read entirely. Cache files are
versioned (annv2_*) since the feature pipeline changed.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import numpy as np
import tables  # PyTables: native reader for the Zenodo HDF5 format
import torch
from torch.utils.data import Dataset

from .download import download_split

N_CONST = 200
N_FEATURES = 800  # 4 features x 200 constituents
FEAT_VERSION = "v2"  # bump when the feature pipeline changes (cache key)


@dataclass
class AnnDatasetConfig:
    data_dir: str = "data"
    max_events: int | None = None  # cap number of jets loaded
    cache_dir: str | None = None   # if set, processed features are cached
    augment: bool = False          # random d_eta/d_phi sign flips (train only)


def load_raw_features(
    path: str, max_events: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read one Zenodo split -> (X_raw, y): X_raw (N, 800) interleaved
    [E,PX,PY,PZ] per constituent, y (N,)."""
    with tables.open_file(path) as f:
        tbl = f.get_node("/table/table")
        n = tbl.nrows
        if max_events is not None:
            n = min(n, max_events)
        arr = tbl.read(0, n)  # structured array

    X = arr["values_block_0"][:, :N_FEATURES].astype(np.float32)
    y = arr["values_block_1"][:, 1].astype(np.float32)  # is_signal_new
    del arr
    gc.collect()
    return X, y


def _wrap_phi(dphi: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into (-pi, pi]."""
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def build_feature_matrix(X_raw: np.ndarray) -> np.ndarray:
    """(N,800) interleaved [E,PX,PY,PZ] -> (N,800) jet-relative features.

    Per constituent: [log1p(pT), d_eta, d_phi, log1p(E)] where the deltas
    are taken w.r.t. the pT-weighted jet axis. Padded constituents
    (pT == 0) produce all-zero feature rows.
    """
    X4 = X_raw.reshape(-1, N_CONST, 4).astype(np.float64)
    E, PX, PY, PZ = X4[..., 0], X4[..., 1], X4[..., 2], X4[..., 3]

    pt = np.sqrt(PX * PX + PY * PY)
    p = np.sqrt(PX * PX + PY * PY + PZ * PZ)
    # eta = arctanh(pz / |p|); clip the ratio for numerical safety.
    ratio = np.clip(PZ / np.maximum(p, 1e-8), -0.999999, 0.999999)
    eta = np.arctanh(ratio)
    phi = np.arctan2(PY, PX)

    # Jet axis: pT-weighted centroid (phi via unit-vector average).
    pt_sum = pt.sum(axis=1)
    pt_sum = np.where(pt_sum == 0.0, 1.0, pt_sum)
    eta_j = (pt * eta).sum(axis=1) / pt_sum
    phi_j = np.arctan2((pt * np.sin(phi)).sum(axis=1),
                       (pt * np.cos(phi)).sum(axis=1))

    d_eta = eta - eta_j[:, None]
    d_phi = _wrap_phi(phi - phi_j[:, None])
    mask = (pt > 0.0)[..., None]

    feats = np.stack(
        [np.log1p(pt), d_eta, d_phi, np.log1p(np.clip(E, 0.0, None))],
        axis=2,
    )
    feats = (feats * mask).astype(np.float32)
    return feats.reshape(len(X_raw), N_FEATURES)


def feature_stats(
    path: str, max_events: int | None = None, chunk_size: int = 50000
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over the processed features, chunked over rows.

    Reads the HDF5 table in row chunks so memory stays bounded regardless
    of split size.
    """
    s1 = np.zeros(N_FEATURES, dtype=np.float64)
    s2 = np.zeros(N_FEATURES, dtype=np.float64)
    cnt = 0
    with tables.open_file(path) as f:
        tbl = f.get_node("/table/table")
        n = tbl.nrows
        if max_events is not None:
            n = min(n, max_events)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            raw = tbl.read(start, end)["values_block_0"][:, :N_FEATURES]
            chunk = build_feature_matrix(raw).astype(np.float64)
            s1 += chunk.sum(axis=0)
            s2 += (chunk * chunk).sum(axis=0)
            cnt += end - start
            del raw, chunk
            gc.collect()
    mean = s1 / cnt
    var = s2 / cnt - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def stats_cache_path(cache_dir: str, max_events: int | None) -> str:
    n = max_events if max_events is not None else "all"
    return os.path.join(cache_dir, f"ann{FEAT_VERSION}_stats_n{n}.npz")


def load_or_compute_stats(
    path: str, cache_dir: str | None, max_events: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached train-split feature stats, or compute and cache them."""
    if cache_dir:
        sp = stats_cache_path(cache_dir, max_events)
        if os.path.exists(sp):
            cached = np.load(sp)
            print(f"[ann-data] loaded feature stats from {sp}")
            return cached["mean"], cached["std"]
    print("[ann-data] computing train-split feature stats ...")
    mean, std = feature_stats(path, max_events)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(stats_cache_path(cache_dir, max_events), mean=mean, std=std)
        print(f"[ann-data] cached feature stats -> {stats_cache_path(cache_dir, max_events)}")
    return mean, std


class JetFeatureDataset(Dataset):
    """PyTorch dataset yielding (feature_vector, label) for the ANN.

    Features are the jet-relative constituent features standardized with
    the provided (train-split) mean/std, held in memory: ~N * 800 * 4
    bytes (~640 MB for 200k jets).
    """

    def __init__(
        self,
        split: str,
        cfg: AnnDatasetConfig | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.split = split
        self.cfg = cfg or AnnDatasetConfig()
        path = download_split(split, self.cfg.data_dir)
        cache_path = self._cache_path()

        if cache_path and os.path.exists(cache_path):
            print(f"[ann-data] {split}: loading cached features from {cache_path}")
            cached = np.load(cache_path)
            self.features = cached["X"]
            self.labels = cached["labels"]
        else:
            print(f"[ann-data] Loading {split} features from {path} ...")
            X, y = load_raw_features(path, self.cfg.max_events)
            # Build features in row chunks — the float64 intermediates for
            # the whole split at once would OOM (~8 GB at 404k rows).
            chunk = 100000
            for s in range(0, len(X), chunk):
                X[s:s + chunk] = build_feature_matrix(X[s:s + chunk])
            if mean is not None and std is not None:
                X -= mean
                X /= std
            self.features = np.ascontiguousarray(X)
            self.labels = np.asarray(y, dtype=np.float32).reshape(-1)
            del X, y
            gc.collect()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.savez(cache_path, X=self.features, labels=self.labels)
                print(f"[ann-data] cached features -> {cache_path}")

        n = len(self.labels)
        pos = int(self.labels.sum())
        print(f"[ann-data] {self.split}: {pos} top / {n - pos} qcd")

    def _cache_path(self) -> str | None:
        if not self.cfg.cache_dir:
            return None
        n = self.cfg.max_events if self.cfg.max_events is not None else "all"
        return os.path.join(
            self.cfg.cache_dir, f"ann{FEAT_VERSION}_{self.split}_n{n}.npz"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        if self.cfg.augment:
            # Random sign flips of d_eta and d_phi — jet physics is
            # symmetric under both reflections (feature layout per
            # constituent: [log_pT, d_eta, d_phi, log_E]).
            x = x.copy()
            if np.random.rand() < 0.5:
                x[1::4] = -x[1::4]
            if np.random.rand() < 0.5:
                x[2::4] = -x[2::4]
        x = torch.from_numpy(x)
        y = torch.tensor(self.labels[idx])
        return x, y
