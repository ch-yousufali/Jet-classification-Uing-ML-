# Project notes for agents

## Project
Jet classification (top-quark tagging) on the Top Quark Tagging Reference
Dataset (Zenodo 2603256). Two baselines, each in its own self-contained
package: `CNN/` (jet-image conv net) and `ANN/` (MLP on raw constituent
features). PyTorch. Both read the same `data/` splits.

## Environment
- Python venv at `.venv/` (PEP 668 blocks system pip on this machine).
- Install: `.venv/bin/pip install -r requirements.txt`
- No GPU detected; runs on CPU. Code auto-detects CUDA if present.

## Commands
- CNN smoke test (caps each split to 4000 jets):
  `.venv/bin/python -m CNN.train --max-events 4000 --epochs 3 --batch-size 256 --splits train val`
- CNN 200k training (200 epochs max + early stopping):
  `.venv/bin/python -m CNN.train --max-events 200000 --max-val-events 40000 --epochs 200 --batch-size 512 --early-stopping-patience 5`
- CNN evaluate on test split:
  `.venv/bin/python -m CNN.evaluate --ckpt checkpoints/cnn_best.pt`
- ANN 200k training (200 epochs max + early stopping):
  `.venv/bin/python -m ANN.train --max-events 200000 --max-val-events 40000 --epochs 200 --batch-size 512 --early-stopping-patience 5`
- ANN evaluate on test split:
  `.venv/bin/python -m ANN.evaluate --ckpt checkpoints/ann_best.pt`
- History per epoch (loss, accuracy, AUC for train+val):
  `checkpoints/cnn_history.json`, `checkpoints/ann_history.json`

## Dataset
- HDF5 files from https://zenodo.org/record/2603256 (train.h5, val.h5, test.h5).
- Format: PyTables (`tables` package required, not plain h5py). Each row in
  `/table/table` is a structured array with:
  - `values_block_0` (804,) float32 = 200 constituent four-momenta stored
    interleaved `[E, PX, PY, PZ]` (800 values) + truth top-quark 4-vec (4, zero for QCD).
  - `values_block_1` (2,) int64 = `[ttv, is_signal_new]`; label = index 1 (1 = top, 0 = QCD).
  - `index` int64 row id.
- Eta and Phi are NOT stored; computed from (PX, PY, PZ) in `CNN.jet_image.load_split_arrays`.
- Downloaded automatically into `data/` by `CNN.download` / `ANN.download`.
- Sizes: train.h5 ~1.04 GB (1.2M jets), val.h5 ~0.35 GB (400k), test.h5 ~0.35 GB (404k).

## Verified
- Smoke test passes: 3 epochs / 4000 jets -> val AUC 0.9274, test AUC 0.9247.
- CNN 200k (2026-09-16): early stop ep13, best val AUC 0.9739; full 404k test:
  AUC 0.9751, acc 0.9194, 1/eps_B @ eps_S=0.3 = 401.4.
- ANN (jet-relative feats, 200k): v1 raw 4-momenta -> 0.8923 test AUC;
  v2 [log pT, d_eta, d_phi, log E] -> 0.9555; +augment+1024-512-256 -> 0.9689.
- ANN >200k runs (record only): 600k -> 0.9690; 1.2M + bigger net -> 0.9734
  (test acc 0.9152, 1/eps_B = 442.8 — matches CNN at 200k).
- Streaming image build (CNN.build_jet_images_streaming): HDF5 read in 50k
  chunks, constituents never fully resident -> 800k images built in ~4 min,
  peak ~5.6 GB. All chunked/cached builds give bit-identical images.
- Augmentation (both models): random eta/phi flips (CNN images, ANN d_eta/
  d_phi sign flips) via `--augment`, train split only — 4x effective data.
- `--init-from` resumes weights from a checkpoint; `--lr-sched cosine|plateau`
  + `--min-lr` for LR decay; `--hidden` sets ANN layer widths.

## Layout
- `CNN/` — self-contained CNN package: `download.py` (Zenodo fetch),
  `jet_image.py` (raw arrays + chunked jet-image build), `dataset.py`
  (JetImageDataset + DatasetConfig + .npz image cache), `model.py`
  (ConvBlock + JetImageCNN + build_model), `train.py`, `evaluate.py`.
- `ANN/` — self-contained MLP package: `download.py`, `data.py` (raw
  800-feature loading, signed-log transform, train-split standardization,
  .npz cache), `model.py` (DenseBlock + JetFeatureANN + build_ann),
  `train.py`, `evaluate.py`.
- `tests/` — pytest suite: `test_model.py`, `test_data.py`,
  `test_train.py`, `test_ann.py`. Run `.venv/bin/python -m pytest tests/ -v`.
  Data-dependent tests skip themselves if `data/*.h5` are absent.
