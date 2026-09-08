"""Lesson 0 — no neural network.

Run this first. It proves the geometry claim in Figure 1 of the blueprint:
from orbit, recovering metres is just fitting two numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from m1.affine import fit_affine
from m1.metrics import mae, rmse
from m1.synthetic import make_city_tile


def main() -> None:
    out = Path("outputs/lesson0")
    out.mkdir(parents=True, exist_ok=True)

    tile = make_city_tile(gsd_m=1.0, seed=7)
    H = tile.height_m

    # Pretend a frozen depth network spat out a scrambled version of the truth:
    # unknown scale, unknown shift, plus a little noise. That *is* what
    # monocular depth models give you.
    rng = np.random.default_rng(0)
    secret_a, secret_b = 0.37, 12.0
    h_hat = (H - secret_b) / secret_a + rng.normal(0, 0.4, H.shape)

    naive = (h_hat - h_hat.min()) / (h_hat.max() - h_hat.min()) * H.max()
    fit = fit_affine(h_hat, H)
    recovered = fit.apply(h_hat)

    print("SECRET scale/shift the fake network used:  "
          f"a={secret_a:.3f}  b={secret_b:.3f}")
    print("WHAT we recovered by least squares:        "
          f"a={fit.a:.3f}  b={fit.b:.3f}")
    print()
    print(f"minmax stretch (what beginners do)  RMSE={rmse(naive, H):.2f} m  MAE={mae(naive, H):.2f} m")
    print(f"affine  H = a*h_hat + b             RMSE={rmse(recovered, H):.2f} m  MAE={mae(recovered, H):.2f} m")
    print()
    print("Takeaway: the shape was already right. Scale was the only missing piece.")
    print("M1's job is a sharper relative field. M2's job is a better (a, b) -- later, using a DEM.")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(tile.rgb)
    axes[0].set_title("RGB (synthetic nadir)")
    axes[1].imshow(H, cmap="terrain")
    axes[1].set_title("True height (m)")
    axes[2].imshow(h_hat, cmap="magma")
    axes[2].set_title("Fake network ĥ (unitless)")
    im = axes[3].imshow(recovered, cmap="terrain")
    axes[3].set_title("After affine (m)")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "scale_ambiguity.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out / 'scale_ambiguity.png'}")


if __name__ == "__main__":
    main()
