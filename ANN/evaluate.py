"""Evaluate a trained ANN checkpoint on the test split.

Example:

    .venv/bin/python -m ann.evaluate --ckpt checkpoints/ann_best.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .data import AnnDatasetConfig, JetFeatureDataset, load_or_compute_stats
from .model import build_ann


def main():
    parser = argparse.ArgumentParser(description="Evaluate ANN on the test split.")
    parser.add_argument("--ckpt", default="checkpoints/ann_best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Directory for cached features (.npz). Empty string disables.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"[ann-eval] loading checkpoint from {args.ckpt} (val_auc={ckpt['val_auc']:.4f})")

    # Rebuild the same feature standardization used at train time:
    # stats are keyed by the train split's --max-events stored in ckpt args.
    train_max = ckpt["args"].get("max_events")
    train_path = os.path.join(args.data_dir, "train.h5")
    mean, std = load_or_compute_stats(train_path, args.cache_dir or None, train_max)

    in_features = ckpt["model_state"]["net.0.block.0.weight"].shape[1]
    hidden = tuple(
        int(h) for h in ckpt["args"].get("hidden", "512,256,128").split(",")
    )
    model = build_ann(
        in_features=in_features,
        hidden=hidden,
        device=device,
        dropout=ckpt["args"].get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cfg = AnnDatasetConfig(
        data_dir=args.data_dir,
        max_events=args.max_events,
        cache_dir=args.cache_dir or None,
    )
    test_ds = JetFeatureDataset("test", cfg, mean=mean, std=std)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = expit(logits)

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, (probs > 0.5).astype(int))
    fpr, tpr, _ = roc_curve(labels, probs)
    # Background rejection at 30% signal efficiency (standard top-tagging metric).
    idx = np.searchsorted(tpr, 0.3)
    rej_at_30 = 1.0 / max(fpr[idx], 1e-12) if idx < len(fpr) else float("inf")

    print(f"[ann-eval] test AUC          = {auc:.4f}")
    print(f"[ann-eval] test accuracy     = {acc:.4f}")
    print(f"[ann-eval] 1/eps_B @ eps_S=0.3 = {rej_at_30:.2f}")


if __name__ == "__main__":
    main()
