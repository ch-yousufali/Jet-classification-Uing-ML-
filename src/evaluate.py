"""Evaluate a trained CNN checkpoint on the test split.

Example:

    .venv/bin/python -m src.evaluate --ckpt checkpoints/cnn_best.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .data import DatasetConfig, JetImageDataset
from .model import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN on the test split.")
    parser.add_argument("--ckpt", default="checkpoints/cnn_best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Directory for cached jet images (.npz). Empty string disables.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    img_size = ckpt["args"].get("img_size", 40)
    print(f"[eval] loading checkpoint from {args.ckpt} (val_auc={ckpt['val_auc']:.4f})")

    model = build_model(img_size=img_size, device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cfg = DatasetConfig(
        data_dir=args.data_dir, img_size=img_size,
        max_events=args.max_events, cache_dir=args.cache_dir or None,
    )
    test_ds = JetImageDataset("test", cfg)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1.0 / (1.0 + np.exp(-logits))

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, (probs > 0.5).astype(int))
    fpr, tpr, _ = roc_curve(labels, probs)
    # Background rejection at 30% signal efficiency (standard top-tagging metric).
    idx = np.searchsorted(tpr, 0.3)
    rej_at_30 = 1.0 / max(fpr[idx], 1e-12) if idx < len(fpr) else float("inf")

    print(f"[eval] test AUC          = {auc:.4f}")
    print(f"[eval] test accuracy     = {acc:.4f}")
    print(f"[eval] 1/eps_B @ eps_S=0.3 = {rej_at_30:.2f}")


if __name__ == "__main__":
    main()
