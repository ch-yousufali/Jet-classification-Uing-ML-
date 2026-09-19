"""Raw array loading and jet-image construction.

The Zenodo HDF5 files use PyTables (blosc-compressed structured arrays).
Each row in `/table/table` stores:

  - values_block_0 : (804,) float32 = 200 constituent four-momenta stored
                     interleaved as [E, PX, PY, PZ] per constituent
                     (800 values) followed by the truth top-quark
                     four-momentum (4 values, zero for QCD).
  - values_block_1 : (2,) int64 = [ttv, is_signal_new]; label = index 1
                     (1 = top quark, 0 = QCD background).
  - index          : int64 row id.

Eta and Phi are not stored; they are derived from (PX, PY, PZ).

Constituents are turned into a 2D "jet image" (pT-weighted eta-phi
histogram) following the image-based approach in the README
(Approach 1 / DeepTop).
"""

from __future__ import annotations

import gc

import numpy as np
import tables  # PyTables: native reader for the Zenodo HDF5 format

# Image grid parameters. Anti-kT R=0.8 jets fit within |eta_rel|, |phi_rel| < 0.8.
IMG_SIZE = 40
IMG_RANGE = 0.8  # half-width of the eta-phi window in radians


def load_split_arrays(
    path: str, max_events: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one Zenodo HDF5 split and return per-constituent arrays.

    Returns (E, PX, PY, PZ, Eta, Phi, labels), each of shape (N, 200)
    except `labels` which is (N,). Constituents are zero-padded to 200.
    Eta and Phi are computed from (PX, PY, PZ).
    """
    with tables.open_file(path) as f:
        tbl = f.get_node("/table/table")
        n = tbl.nrows
        if max_events is not None:
            n = min(n, max_events)
        arr = tbl.read(0, n)  # structured array

    vb0 = arr["values_block_0"]  # (N, 804)
    vb1 = arr["values_block_1"]  # (N, 2) = [ttv, is_signal_new]
    labels = vb1[:, 1].astype(np.float32)  # is_signal_new

    # Interleaved [E, PX, PY, PZ] per constituent -> (N, 200, 4).
    feat = vb0[:, :800].reshape(n, 200, 4).astype(np.float32)
    # Free the raw structured array early to save ~1.3 GB for 400k jets.
    del arr, vb0, vb1
    gc.collect()

    E = feat[..., 0]
    PX = feat[..., 1]
    PY = feat[..., 2]
    PZ = feat[..., 3]

    # Derived kinematics. Guard pT=0 (zero-padded constituents).
    pt = np.sqrt(np.maximum(PX**2 + PY**2, 0.0))
    pt_safe = np.where(pt > 0, pt, 1.0)
    Eta = np.arcsinh(PZ / pt_safe)            # pseudorapidity
    Eta = np.where(pt > 0, Eta, 0.0).astype(np.float32)
    Phi = np.arctan2(PY, PX).astype(np.float32)  # azimuth [-pi, pi]

    return E, PX, PY, PZ, Eta, Phi, labels


def _delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Smallest signed angular difference on the azimuthal circle."""
    d = phi1 - phi2
    return (d + np.pi) % (2 * np.pi) - np.pi


def _fill_chunk_images(
    images: np.ndarray,
    sl: slice,
    pt: np.ndarray,
    Eta: np.ndarray,
    Phi: np.ndarray,
    img_size: int,
    img_range: float,
) -> None:
    """Histogram one chunk of constituents into `images[sl]` (in place).

    `pt`, `Eta`, `Phi` are (cs, 200) arrays for this chunk only.
    """
    cs = sl.stop - sl.start
    pixel_area = img_size * img_size
    bin_width = 2.0 * img_range / img_size

    # Jet axis: pT-weighted mean of constituent eta/phi.
    # Guard against all-zero rows (should not happen, but be safe).
    pt_sum = pt.sum(axis=1, keepdims=True)
    pt_sum[pt_sum <= 0] = 1.0
    jet_eta = (pt * Eta).sum(axis=1, keepdims=True) / pt_sum  # (cs,1)
    jet_phi = (pt * Phi).sum(axis=1, keepdims=True) / pt_sum

    eta_rel = Eta - jet_eta  # (cs, 200)
    phi_rel = _delta_phi(Phi, jet_phi)

    # Uniform-grid binning: floor((x + range) / width) is equivalent
    # to np.digitize on evenly-spaced bins (after clipping), but is a
    # single vectorized op instead of a per-element binary search —
    # ~10-50x faster on large arrays.
    ix = np.floor((eta_rel + img_range) / bin_width)
    iy = np.floor((phi_rel + img_range) / bin_width)
    np.clip(ix, 0, img_size - 1, out=ix)
    np.clip(iy, 0, img_size - 1, out=iy)

    # Flat pixel index within each event: ix * img_size + iy, plus a
    # per-event offset so bincount lands in the right image slice.
    flat_idx = (
        ix.astype(np.int64) * img_size + iy.astype(np.int64)
        + (np.arange(cs, dtype=np.int64) * pixel_area).reshape(cs, 1)
    ).ravel()

    chunk = np.bincount(flat_idx, weights=pt.ravel(), minlength=cs * pixel_area)
    images[sl] = chunk.astype(np.float32).reshape(cs, img_size, img_size)


def _postprocess_images(images: np.ndarray) -> np.ndarray:
    """Log-compress + standardize per-image (mean 0, std 1), in place."""
    np.log1p(images, out=images)
    mean = images.mean(axis=(1, 2), keepdims=True)
    std = images.std(axis=(1, 2), keepdims=True)
    std[std <= 1e-6] = 1.0
    images -= mean
    images /= std
    return images[:, None, :, :].astype(np.float32)


def build_jet_images(
    E: np.ndarray,
    PX: np.ndarray,
    PY: np.ndarray,
    PZ: np.ndarray,
    Eta: np.ndarray,
    Phi: np.ndarray,
    img_size: int = IMG_SIZE,
    img_range: float = IMG_RANGE,
    chunk_size: int = 50000,
) -> np.ndarray:
    """Convert constituent four-momenta into pT-weighted eta-phi images.

    Inputs are (N, 200) arrays (zero-padded constituents). The jet axis is
    taken as the pT-weighted centroid of the constituents, and constituents
    are histogrammed into an `img_size` x `img_size` grid spanning
    [-img_range, img_range] in (eta_rel, phi_rel). Pixel value = sum of pT
    of constituents falling in that bin.

    All per-constituent intermediates are computed inside a chunk loop of
    `chunk_size` jets, so peak extra memory stays ~chunk_size * 200 * 4 B
    * ~8 arrays (~320 MB at the default 50k) regardless of N.

    Returns a float32 array of shape (N, 1, img_size, img_size).
    """
    N = E.shape[0]
    images = np.empty((N, img_size, img_size), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sl = slice(start, end)
        pt = np.sqrt(np.maximum(PX[sl] * PX[sl] + PY[sl] * PY[sl], 0.0))
        _fill_chunk_images(images, sl, pt, Eta[sl], Phi[sl], img_size, img_range)
        del pt

    return _postprocess_images(images)


def _chunk_kinematics(feat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cs,200,4) [E,PX,PY,PZ] -> (pt, Eta, Phi) each (cs,200) float32."""
    PX, PY, PZ = feat[..., 1], feat[..., 2], feat[..., 3]
    pt = np.sqrt(np.maximum(PX * PX + PY * PY, 0.0))
    pt_safe = np.where(pt > 0, pt, 1.0)
    Eta = np.arcsinh(PZ / pt_safe)
    Eta = np.where(pt > 0, Eta, 0.0).astype(np.float32)
    Phi = np.arctan2(PY, PX).astype(np.float32)
    return pt.astype(np.float32), Eta, Phi


def build_jet_images_streaming(
    path: str,
    max_events: int | None = None,
    img_size: int = IMG_SIZE,
    img_range: float = IMG_RANGE,
    chunk_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Build jet images by streaming the HDF5 file in row chunks.

    Unlike load_split_arrays + build_jet_images, the constituent arrays
    are never fully resident: each chunk is read, converted to kinematics,
    histogrammed, and freed. Peak memory is the output image array
    (~N * img_size^2 * 4 B) plus ~chunk_size scratch — enabling builds at
    800k-1.2M jets on a 16 GB machine.

    Returns (images (N,1,S,S) float32, labels (N,) float32).
    """
    with tables.open_file(path) as f:
        tbl = f.get_node("/table/table")
        n = tbl.nrows
        if max_events is not None:
            n = min(n, max_events)
        images = np.empty((n, img_size, img_size), dtype=np.float32)
        labels = np.empty(n, dtype=np.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            arr = tbl.read(start, end)
            labels[start:end] = arr["values_block_1"][:, 1]
            feat = arr["values_block_0"][:, :800].reshape(-1, 200, 4)
            pt, Eta, Phi = _chunk_kinematics(feat)
            _fill_chunk_images(
                images, slice(start, end), pt, Eta, Phi, img_size, img_range
            )
            del arr, feat, pt, Eta, Phi
            gc.collect()

    return _postprocess_images(images), labels
