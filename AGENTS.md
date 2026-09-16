# Project notes for agents

## Project
Jet classification (top-quark tagging) on the Top Quark Tagging Reference
Dataset (Zenodo 2603256) using a CNN baseline on jet images. PyTorch.

## Environment
- Python venv at `.venv/` (PEP 668 blocks system pip on this machine).
- Install: `.venv/bin/pip install -r requirements.txt`
- No GPU detected; runs on CPU. Code auto-detects CUDA if present.

## Commands
- Smoke test (downloads train.h5 + val.h5, ~1.4 GB; caps each split to 4000 jets):
  `.venv/bin/python -m src.train --max-events 4000 --epochs 3 --batch-size 256 --splits train val`
- Full training (downloads train.h5 + val.h5, ~1.4 GB):
  `.venv/bin/python -m src.train --epochs 20 --batch-size 512`
- Evaluate on test split (downloads test.h5, ~0.35 GB):
  `.venv/bin/python -m src.evaluate --ckpt checkpoints/cnn_best.pt`
- Smoke evaluate (no extra download if test.h5 present):
  `.venv/bin/python -m src.evaluate --ckpt checkpoints/cnn_best.pt --max-events 4000`

## Dataset
- HDF5 files from https://zenodo.org/record/2603256 (train.h5, val.h5, test.h5).
- Format: PyTables (`tables` package required, not plain h5py). Each row in
  `/table/table` is a structured array with:
  - `values_block_0` (804,) float32 = 200 constituent four-momenta stored
    interleaved `[E, PX, PY, PZ]` (800 values) + truth top-quark 4-vec (4, zero for QCD).
  - `values_block_1` (2,) int64 = `[ttv, is_signal_new]`; label = index 1 (1 = top, 0 = QCD).
  - `index` int64 row id.
- Eta and Phi are NOT stored; computed from (PX, PY, PZ) in `src.data.load_split_arrays`.
- Downloaded automatically into `data/` by `src.data.download_split`.
- Sizes: train.h5 ~1.04 GB (1.2M jets), val.h5 ~0.35 GB (400k), test.h5 ~0.35 GB (400k).

## Verified
- Smoke test passes: 3 epochs / 4000 jets -> val AUC 0.9274, test AUC 0.9247.
- 100k-jet training: 20 epochs / 100k train + 20k val -> best val AUC 0.9731
  (epoch 8), test AUC 0.9725, test accuracy 0.9152, 1/eps_B @ eps_S=0.3 = 410.
- 200k-jet training (2026-09-16): `--max-events 200000 --max-val-events 40000
  --epochs 20 --batch-size 512 --early-stopping-patience 5` -> early stopped
  at epoch 13, best val AUC 0.9739 (epoch 8). Evaluated on FULL 404k test set:
  test AUC 0.9751, accuracy 0.9194, 1/eps_B @ eps_S=0.3 = 401.42.
  Data build now takes ~2 min (chunked processing + arithmetic binning) and
  images are cached to data/cache/*.npz. Peak RAM ~2.5 GB — no OOM.
- Full 1.2M training not yet run (CPU-only, 16 GB RAM; should now fit with
  chunked build but would be slow — est. several hours of training).

## Layout
- `src/data/` — package: `download.py` (Zenodo fetch), `jet_image.py`
  (raw array loading + jet-image construction), `dataset.py`
  (JetImageDataset + DatasetConfig). `__init__.py` re-exports public API.
- `src/model/` — package: `cnn.py` (ConvBlock + JetImageCNN),
  `build.py` (build_model helper). `__init__.py` re-exports public API.
- `src/train.py` — training loop with AUC tracking + checkpointing
- `src/evaluate.py` — test-split evaluation (AUC, accuracy, 1/eps_B @ eps_S=0.3)
- `tests/` — pytest suite: `test_model.py`, `test_data.py`, `test_train.py`.
  Run with `.venv/bin/python -m pytest tests/ -v`. Data-dependent tests
  skip themselves if `data/*.h5` are absent.
