"""M1 Height Prior Network — DepthWizard.

Owns the unitless relative height field ĥ(x, y) and a per-pixel uncertainty map.
Metric scale (metres) is M2's job. Never leak unitless numbers past calibration.
"""

from m1.config import M1Config
from m1.affine import fit_affine, apply_affine
from m1.metrics import rmse, mae, delta_accuracy

__all__ = [
    "M1Config",
    "fit_affine",
    "apply_affine",
    "rmse",
    "mae",
    "delta_accuracy",
]
