"""Numbers we will put on the ablation table (blueprint §7.3 / §10)."""

from __future__ import annotations

import numpy as np


def _flat(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None):
    p = pred.reshape(-1).astype(np.float64)
    t = target.reshape(-1).astype(np.float64)
    if valid is None:
        m = np.isfinite(p) & np.isfinite(t)
    else:
        m = valid.reshape(-1).astype(bool) & np.isfinite(p) & np.isfinite(t)
    return p[m], t[m]


def rmse(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> float:
    p, t = _flat(pred, target, valid)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> float:
    p, t = _flat(pred, target, valid)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - t)))


def delta_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    thresh: float = 1.25,
) -> float:
    """Fraction of pixels where max(pred/target, target/pred) < thresh.

    Classic monocular-depth metric. Shift both maps so they are positive first.
    """
    p, t = _flat(pred, target, valid)
    if p.size == 0:
        return float("nan")
    # Heights can be ~0 on bare ground; offset so the ratio is defined.
    offset = 1.0
    p = p + offset
    t = t + offset
    ratio = np.maximum(p / t, t / p)
    return float(np.mean(ratio < thresh))
