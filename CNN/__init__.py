"""CNN baseline for top-quark jet tagging on jet images.

Self-contained package: download, jet-image construction, Dataset,
model, training and evaluation all live inside CNN/.
"""

from .dataset import DatasetConfig, JetImageDataset
from .download import SPLIT_URLS, download_split, ensure_splits
from .jet_image import IMG_RANGE, IMG_SIZE, build_jet_images, load_split_arrays
from .model import ConvBlock, JetImageCNN, build_model

__all__ = [
    "DatasetConfig",
    "JetImageDataset",
    "SPLIT_URLS",
    "download_split",
    "ensure_splits",
    "IMG_SIZE",
    "IMG_RANGE",
    "build_jet_images",
    "load_split_arrays",
    "ConvBlock",
    "JetImageCNN",
    "build_model",
]
