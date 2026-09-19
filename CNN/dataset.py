"""PyTorch Dataset wrapping the Top Quark Tagging jet images."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .download import download_split
from .jet_image import IMG_SIZE, build_jet_images_streaming


@dataclass
class DatasetConfig:
    data_dir: str = "data"
    img_size: int = IMG_SIZE
    max_events: int | None = None  # cap number of jets loaded (for smoke tests)
    cache_dir: str | None = None   # if set, built images are cached to .npz
    augment: bool = False          # random eta/phi image flips (train only)


class JetImageDataset(Dataset):
    """PyTorch dataset that yields (jet_image, label) tensors.

    Jet images are built once from the constituent four-momenta and cached
    in memory (~N * img_size^2 * 4 bytes; e.g. ~1.3 GB for 200k jets at
    img_size=40). If `cache_dir` is set, the built images + labels are also
    written to a `.npz` on disk so later runs skip the build entirely.
    """

    def __init__(self, split: str, cfg: DatasetConfig | None = None):
        self.split = split
        self.cfg = cfg or DatasetConfig()
        path = download_split(split, self.cfg.data_dir)
        cache_path = self._cache_path()

        if cache_path and os.path.exists(cache_path):
            print(f"[data] {split}: loading cached images from {cache_path}")
            cached = np.load(cache_path)
            self.images = cached["images"]
            self.labels = cached["labels"]
        else:
            print(f"[data] Loading {split} from {path} ...")
            # Stream the HDF5 in chunks — constituents are never fully
            # resident, so peak memory is just the output images (~300 MB
            # scratch on top), enabling 800k-1.2M builds without OOM.
            self.images, y = build_jet_images_streaming(
                path, max_events=self.cfg.max_events, img_size=self.cfg.img_size
            )
            n = len(y)
            print(f"[data] Built {n} jet images ({self.images.nbytes / 1e9:.2f} GB)")
            self.labels = np.asarray(y, dtype=np.float32).reshape(-1)
            del y
            gc.collect()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.savez(cache_path, images=self.images, labels=self.labels)
                print(f"[data] cached images -> {cache_path}")

        n = len(self.labels)
        pos = int(self.labels.sum())
        print(f"[data] {self.split}: {pos} top / {n - pos} qcd")

    def _cache_path(self) -> str | None:
        if not self.cfg.cache_dir:
            return None
        n = self.cfg.max_events if self.cfg.max_events is not None else "all"
        name = f"{self.split}_n{n}_img{self.cfg.img_size}.npz"
        return os.path.join(self.cfg.cache_dir, name)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx]
        if self.cfg.augment:
            # Random reflections across the eta and phi axes — jet physics
            # is symmetric under both, so each flip is a valid new sample
            # (x has shape (1, eta, phi)).
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        y = torch.tensor(self.labels[idx])
        return x, y
