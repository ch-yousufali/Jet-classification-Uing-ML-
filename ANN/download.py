"""Download utilities for the Top Quark Tagging Reference Dataset.

Files are fetched from Zenodo (record 2603256) into a local `data/`
directory on first use and cached thereafter.
"""

from __future__ import annotations

import os
import urllib.request

# Zenodo download URLs for the three splits.
ZENODO_BASE = "https://zenodo.org/record/2603256/files"
SPLIT_URLS = {
    "train": f"{ZENODO_BASE}/train.h5",
    "val": f"{ZENODO_BASE}/val.h5",
    "test": f"{ZENODO_BASE}/test.h5",
}


def download_split(split: str, data_dir: str) -> str:
    """Download one split's HDF5 file from Zenodo if not already present.

    Returns the local path to the file. `split` must be one of
    train/val/test.
    """
    if split not in SPLIT_URLS:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_URLS)}")
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, f"{split}.h5")
    if os.path.exists(dest):
        print(f"[data] {dest} already exists, skipping download.")
        return dest
    url = SPLIT_URLS[split]
    print(f"[data] Downloading {split} from {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[data] Done ({os.path.getsize(dest) / 1e9:.2f} GB).")
    return dest


def ensure_splits(data_dir: str, splits=("train", "val", "test")) -> dict:
    """Download all requested splits and return {split: path}."""
    return {s: download_split(s, data_dir) for s in splits}
