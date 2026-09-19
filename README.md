# Jet Classification Using Machine Learning — CNN & ANN Baselines

PyTorch implementations of two baseline taggers for top-quark vs. QCD jet
classification on the
[Top Quark Tagging Reference Dataset](https://zenodo.org/record/2603256)
(Zenodo 2603256):

- **CNN** — jets are rendered as 2D pT-weighted images in (η, φ) and
  classified by a convolutional network, like an image classifier.
- **ANN** — a plain MLP on the flattened raw constituent four-momenta
  (200 × [E, px, py, pz] = 800 features). Deliberately the weakest baseline:
  it quantifies how much tagging power comes from architectural inductive
  bias rather than raw information content.

> The full research report (problem statement, dataset review, literature
> comparison, timeline, and references) lives in **[docs/THESIS.md](docs/THESIS.md)**.
> This README documents the model and the code only.

---

## Project structure

```
thesis-repo/
├── README.md
├── AGENTS.md            # notes for AI coding agents
├── requirements.txt     # pinned dependencies
├── .gitignore           # ignores data/, checkpoints/, .venv/
├── docs/
│   └── THESIS.md        # research report (problem, dataset, literature)
├── data/                # downloaded HDF5 files + cached arrays (gitignored)
├── checkpoints/         # saved models + history (gitignored)
├── CNN/                 # CNN baseline (jet-image approach) — self-contained
│   ├── __init__.py      # re-exports the public API
│   ├── download.py      # Zenodo download + caching
│   ├── jet_image.py     # raw array loading + jet-image construction
│   ├── dataset.py       # JetImageDataset + DatasetConfig
│   ├── model.py         # ConvBlock + JetImageCNN + build_model
│   ├── train.py         # training loop, acc/AUC tracking, checkpointing
│   ├── evaluate.py      # test-split evaluation
│   └── comparison_results.md  # CNN vs. published models (ParticleNet etc.)
├── ANN/                 # ANN (MLP) baseline — self-contained
│   ├── __init__.py      # re-exports the public API
│   ├── download.py      # Zenodo download + caching
│   ├── data.py          # raw 800-feature loading + standardization + cache
│   ├── model.py         # DenseBlock + JetFeatureANN + build_ann
│   ├── train.py         # training loop, acc/AUC tracking, checkpointing
│   ├── evaluate.py      # test-split evaluation
│   └── comparison_results.md  # ANN vs. published models + ANN-vs-CNN table
├── logs_*.txt           # training logs from the 200k-jet runs
├── eval_*_full_test.txt # full-test-set evaluation logs
└── tests/
    ├── __init__.py
    ├── test_model.py    # CNN shape / param / gradient tests
    ├── test_data.py     # jet-image + kinematics + dataset tests
    ├── test_train.py    # one training step smoke tests
    └── test_ann.py      # ANN model + signed-log feature tests
```

## Setup

A Python virtual environment is used because the system `pip` is blocked
by PEP 668 on this machine.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## How to run

The dataset is downloaded automatically from Zenodo into `data/` on
first use, so no manual download step is needed. Both models share the
same `data/` files and the same `data/cache/` for processed arrays.

### CNN (jet images)

Quick smoke test (caps each split to 4000 jets, 3 epochs, CPU-friendly):

```bash
.venv/bin/python -m CNN.train --max-events 4000 --epochs 3 --batch-size 256 --splits train val
```

200k-jet run (up to 200 epochs, early stopping after 5 flat epochs):

```bash
.venv/bin/python -m CNN.train --max-events 200000 --max-val-events 40000 \
    --epochs 200 --batch-size 512 --early-stopping-patience 5
```

Evaluate the best checkpoint on the held-out test split:

```bash
.venv/bin/python -m CNN.evaluate --ckpt checkpoints/cnn_best.pt
```

### ANN (raw constituent features)

Same flags; features are the flattened 200x4 constituent four-momenta
(800 dims), signed-log transformed and standardized with train-split stats:

```bash
.venv/bin/python -m ANN.train --max-events 200000 --max-val-events 40000 \
    --epochs 200 --batch-size 512 --early-stopping-patience 5

.venv/bin/python -m ANN.evaluate --ckpt checkpoints/ann_best.pt
```

Each run writes `checkpoints/{cnn,ann}_best.pt` and a per-epoch history
(`checkpoints/{cnn,ann}_history.json` with loss, accuracy, and AUC for
train and val) so the two models can be compared epoch by epoch.

## Tests

The test suite covers the model, the data layer, and a single training
step. Tests that need the downloaded HDF5 files skip themselves
automatically when the files are absent, so the suite runs on a fresh
checkout without downloading 1.4 GB.

```bash
.venv/bin/python -m pytest tests/ -v
```

What is covered:

| File | What it checks |
|---|---|
| `tests/test_model.py` | `ConvBlock` and `JetImageCNN` output shapes, parameter count (~913k), presence of BatchNorm/Dropout, gradient flow through all parameters, determinism in eval mode, finiteness of outputs. |
| `tests/test_data.py` | `_delta_phi` wrap-around behaviour, `build_jet_images` shape / standardization / robustness to all-zero jets, `load_split_arrays` shapes and on-shell energy constraint (E² ≥ p²), `JetImageDataset` length and item format. |
| `tests/test_train.py` | `run_epoch` runs in train and eval modes, returns a valid AUC in [0,1], and loss decreases over repeated steps on separable synthetic data. |
| `tests/test_ann.py` | `signed_log` transform (zero, sign, compression), `DenseBlock`/`JetFeatureANN` output shapes, parameter count, gradient flow, eval-mode determinism, `build_ann` device placement. |

## Current status

Both baselines run end-to-end. Latest results are from matched
**200k-jet training runs** (16.5% of the 1.2M training set, 40k val jets,
CPU only), each evaluated on the **full 404k-jet test set**:

| | CNN (jet images) | ANN (raw features) |
|---|---|---|
| Train jets | 200,000 | 200,000 |
| Params | 912,833 | 576,257 |
| Best val AUC | 0.9739 (epoch 8) | 0.8920 (epoch 54) |
| **Test AUC** | **0.9751** | **0.8923** |
| Test accuracy | 0.9194 | 0.8133 |
| 1/ε_B @ ε_S = 0.3 | 401.4 | 36.5 |
| Epochs run | 13 (early stop) | 59 (early stop) |
| Time/epoch (CPU) | ~10 min | ~15 s |

The ~0.08 AUC gap between the two models on identical data is entirely
due to the CNN's spatial image representation — the ANN sees the same
constituents as a flat, unstructured vector.

For reference, published models trained on the full 1.2M set: P-CNN
AUC **0.9803**, ResNeXt-50 AUC **0.9837**, ParticleNet AUC **0.9858**.
Our numbers are not directly comparable (200k vs. 1.2M training jets)
but the CNN is already within ~1% of P-CNN.

Full comparison tables with sources:
[CNN/comparison_results.md](CNN/comparison_results.md) and
[ANN/comparison_results.md](ANN/comparison_results.md).
Training logs: `logs_200k.txt` (CNN), `logs_ann_200k.txt` (ANN); eval
logs: `eval_200k_full_test.txt`, `eval_ann_200k_full_test.txt`.

## Next steps

1. Run full training (full 1.2M training set) on the CNN and record the
   test AUC alongside the published P-CNN / ResNeXt numbers. The chunked
   data pipeline now fits in ~2.5 GB RAM, but CPU-only training would
   take several hours.
2. Build the point-cloud / GNN model (ParticleNet-style EdgeConv) as a
   third package so all three can be compared on the same test split —
   this is the main CNN-vs-GNN comparison of the thesis.
3. Add a plotting script for ROC curves and the AUC comparison table.
