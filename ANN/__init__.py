"""ANN (MLP) baseline for top-quark jet tagging.

Fully-connected network on the raw constituent four-momenta
(200 constituents x [E, PX, PY, PZ] = 800 features). No image
construction — the flattened feature vector feeds dense layers.
"""

from .model import JetFeatureANN, build_ann
from .data import JetFeatureDataset, AnnDatasetConfig

__all__ = [
    "JetFeatureANN",
    "build_ann",
    "JetFeatureDataset",
    "AnnDatasetConfig",
]
