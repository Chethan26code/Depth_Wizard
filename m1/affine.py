"""Recover metres from a unitless field.

From orbit the camera is hundreds of kilometres away, so rays are nearly
parallel. True height H and the network's relative field ĥ then differ by
exactly two unknown numbers:

    H(x, y) = a * ĥ(x, y) + b

That is the entire scale-ambiguity of monocular satellite height (blueprint §2).
M1 produces ĥ. This helper estimates (a, b) so we can report RMSE in metres
before M2 exists. M2 will later replace this global fit with a spatially-varying
polynomial + DEM fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AffineFit:
    a: float
    b: float
    used_pixels: int

    def apply(self, relative: np.ndarray) -> np.ndarray:
        return self.a * relative + self.b


def fit_affine(
    relative: np.ndarray,
    height_m: np.ndarray,
    valid: np.ndarray | None = None,
) -> AffineFit:
    """Least-squares solve for H = a * ĥ + b on valid pixels.

    `relative` and `height_m` must be the same shape. Invalid pixels (NaN,
    nodata) are dropped. We do not use RANSAC here — that is an M2 upgrade.
    """
    h = relative.reshape(-1).astype(np.float64)
    y = height_m.reshape(-1).astype(np.float64)
    if valid is not None:
        m = valid.reshape(-1).astype(bool)
    else:
        m = np.isfinite(h) & np.isfinite(y)
    h, y = h[m], y[m]
    if h.size < 8:
        raise ValueError("Need at least 8 valid pixels to fit scale and shift.")

    # Design matrix [ĥ, 1] @ [a, b]^T = H
    A = np.stack([h, np.ones_like(h)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return AffineFit(a=float(coef[0]), b=float(coef[1]), used_pixels=int(h.size))


def apply_affine(relative: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * relative + b
