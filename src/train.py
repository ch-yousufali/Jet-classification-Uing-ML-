"""Training and evaluation for the CNN jet-tagging baseline.

Example (smoke test on a small subset, CPU):

    .venv/bin/python -m src.train --max-events 4000 --epochs 3 --batch-size 256

Full run (needs train.h5 + val.h5 downloaded):

    .venv/bin/python -m src.train --epochs 20 --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import DatasetConfig, JetImageDataset, ensure_splits
from .model import build_model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "train",
) -> tuple[float, float]:
    """Run one epoch. Returns (mean loss, AUC). Optimizer=None -> eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
    mean_loss = total_loss / max(n, 1)
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, logits) if len(np.unique(labels)) == 2 else float("nan")
    return mean_loss, auc


def main():
    parser = argparse.ArgumentParser(description="Train CNN jet-tagging baseline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--img-size", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Cap jets per split (smoke test). None = full dataset.",
    )
    parser.add_argument(
        "--max-val-events",
        type=int,
        default=None,
        help="Cap val/test jets separately. Falls back to --max-events if unset.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Directory to cache built jet images (.npz). Empty string disables.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop if val AUC has not improved for this many epochs (0 = off).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which splits to download (default train+val).",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if omitted).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train] device = {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Download requested splits.
    ensure_splits(args.data_dir, splits=args.splits)
    cache_dir = args.cache_dir or None
    cfg = DatasetConfig(
        data_dir=args.data_dir, img_size=args.img_size,
        max_events=args.max_events, cache_dir=cache_dir,
    )
    val_cfg = DatasetConfig(
        data_dir=args.data_dir, img_size=args.img_size,
        max_events=args.max_val_events if args.max_val_events is not None else args.max_events,
        cache_dir=cache_dir,
    )

    train_ds = JetImageDataset("train", cfg)
    val_ds = JetImageDataset("val", val_cfg)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(img_size=args.img_size, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    epochs_since_best = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(
            model, train_loader, optimizer, criterion, device, desc="train"
        )
        va_loss, va_auc = run_epoch(
            model, val_loader, None, criterion, device, desc="val"
        )
        dt = time.time() - t0
        print(
            f"[train] epoch {epoch:02d}/{args.epochs} "
            f"loss={tr_loss:.4f} auc={tr_auc:.4f} | "
            f"val_loss={va_loss:.4f} val_auc={va_auc:.4f} ({dt:.1f}s)",
            flush=True,
        )
        history.append(
            dict(epoch=epoch, train_loss=tr_loss, train_auc=tr_auc,
                 val_loss=va_loss, val_auc=va_auc)
        )
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            epochs_since_best = 0
            ckpt = os.path.join(args.ckpt_dir, "cnn_best.pt")
            torch.save(
                dict(epoch=epoch, model_state=model.state_dict(),
                     val_auc=va_auc, args=vars(args)),
                ckpt,
            )
            print(f"[train] saved best model -> {ckpt} (val_auc={va_auc:.4f})", flush=True)
        else:
            epochs_since_best += 1
            if args.early_stopping_patience and epochs_since_best >= args.early_stopping_patience:
                print(
                    f"[train] early stopping: no val AUC improvement for "
                    f"{args.early_stopping_patience} epochs",
                    flush=True,
                )
                break

    # Save history.
    hist_path = os.path.join(args.ckpt_dir, "cnn_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] best val AUC = {best_val_auc:.4f}; history -> {hist_path}")


if __name__ == "__main__":
    main()
