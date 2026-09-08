"""M1 training losses. All of these are scale-aware in different ways.

SILog is *scale-invariant*: if you multiply ĥ by 2, the loss barely changes.
That is deliberate. Absolute metres are M2's problem. If we punished the
network for getting scale wrong we would be training it to do M2's job badly.

Gradient and normal losses keep building edges and flat roofs sharp.
Gaussian NLL trains the uncertainty head to be honest about where it is unsure.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silog_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    lam: float = 0.85,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-invariant logarithmic loss (Eigen et al. / DPT / Depth Anything).

    d_i = log(pred_i) - log(target_i)
    SILog = sqrt( mean(d^2) - λ * mean(d)^2 )

    The second term subtracts the global log-scale, which is exactly the
    unknown `a` in H = a·ĥ + b (in log space).
    """
    pred = pred.clamp_min(eps)
    target = target.clamp_min(eps)
    d = torch.log(pred) - torch.log(target)
    if valid is not None:
        d = d[valid]
        if d.numel() == 0:
            return pred.new_zeros(())
    return torch.sqrt((d ** 2).mean() - lam * (d.mean() ** 2) + eps)


def gradient_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 on image-space derivatives. Punishes smeared building edges."""
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    loss_x = (dx_p - dx_t).abs()
    loss_y = (dy_p - dy_t).abs()
    if valid is not None:
        loss_x = loss_x * valid[..., :, 1:]
        loss_y = loss_y * valid[..., 1:, :]
        denom_x = valid[..., :, 1:].sum().clamp_min(1.0)
        denom_y = valid[..., 1:, :].sum().clamp_min(1.0)
        return loss_x.sum() / denom_x + loss_y.sum() / denom_y
    return loss_x.mean() + loss_y.mean()


def gaussian_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_var: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aleatoric uncertainty: ½ (log σ² + (pred − target)² / σ²).

    After a few epochs, σ should be large on occlusions / water / canopy and
    small on clear rooftops. That map later becomes M2/M3 blending weights.
    """
    var = log_var.exp().clamp_min(1e-6)
    nll = 0.5 * (log_var + (pred - target) ** 2 / var)
    if valid is not None:
        nll = nll[valid]
        if nll.numel() == 0:
            return pred.new_zeros(())
    return nll.mean()


def total_m1_loss(
    pred: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    w_silog: float = 1.0,
    w_grad: float = 0.5,
    w_nll: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    silog = silog_loss(pred, target, valid)
    grad = gradient_matching_loss(pred, target, valid)
    nll = gaussian_nll(pred, target, log_var, valid)
    total = w_silog * silog + w_grad * grad + w_nll * nll
    parts = {
        "silog": float(silog.detach()),
        "grad": float(grad.detach()),
        "nll": float(nll.detach()),
        "total": float(total.detach()),
    }
    return total, parts
