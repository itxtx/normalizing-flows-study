"""Invertibility check: a normalizing flow must satisfy f(f^{-1}(x)) = x.
Left: data and its round-trip reconstruction overlaid (indistinguishable).
Right: distribution of per-point reconstruction error for Real NVP on all four
datasets, against float32 machine precision. Mirrors tests/correctness/.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import _common as C

DATASETS = ["moons", "circles", "checkerboard", "spirals"]
DS_COLOR = {
    "moons": "#2563EB",
    "circles": "#059669",
    "checkerboard": "#D97706",
    "spirals": "#7C3AED",
}
EPS32 = np.finfo(np.float32).eps  # ~1.19e-7


@torch.no_grad()
def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 5.0), layout="constrained")

    # ---- left: round-trip overlay (moons) ----
    m = C.load_cache("moons", "realnvp")["model"]
    m.eval()
    data = C.get_dataset("moons", 3000)
    z, _ = m.inverse(data)
    rec, _ = m.forward(z)
    d, rc = data.numpy(), rec.numpy()
    axL.scatter(d[:, 0], d[:, 1], s=14, c=C.HEUR, label="data  x", linewidths=0)
    axL.scatter(
        rc[:, 0],
        rc[:, 1],
        s=3,
        c=C.AGENT,
        label=r"reconstruction  $f(f^{-1}(x))$",
        linewidths=0,
    )
    axL.set_title("Round trip is exact", fontsize=13, color=C.INK, loc="left")
    axL.set_xlim(-3, 3)
    axL.set_ylim(-3, 3)
    axL.set_xticks([])
    axL.set_yticks([])
    axL.legend(loc="upper right", frameon=False, fontsize=9, markerscale=1.5)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # ---- right: error distribution ----
    means = []
    for ds in DATASETS:
        mm = C.load_cache(ds, "realnvp")["model"]
        mm.eval()
        err = C.reconstruction_error(mm, C.get_dataset(ds, 3000))
        err = np.clip(err, 1e-12, None)
        axR.hist(
            np.log10(err),
            bins=45,
            histtype="step",
            linewidth=2.0,
            color=DS_COLOR[ds],
            label=f"{C.DATASETS[ds][0]} (mean {err.mean():.1e})",
        )
        means.append(err.mean())
    axR.axvline(np.log10(EPS32), color=C.ACCENT, linestyle="--", linewidth=1.4)
    axR.text(
        np.log10(EPS32),
        axR.get_ylim()[1] * 0.96,
        " float32 epsilon",
        color=C.ACCENT,
        fontsize=9,
        va="top",
        ha="left",
    )
    axR.set_xlabel(r"$\log_{10}\;|x - f(f^{-1}(x))|$  (sum over dims)")
    axR.set_ylabel("count")
    axR.set_title(
        "Reconstruction error sits at machine precision",
        fontsize=13,
        color=C.AGENT,
        loc="left",
        fontweight="bold",
    )
    axR.legend(loc="upper left", frameon=False, fontsize=8.5)
    C.style_axes(axR, grid_axis="y")

    C.titled(fig, "Real NVP is invertible by construction")
    C.caption(
        fig,
        "Coupling layers have an analytic inverse, so encoding then decoding "
        "returns the input to ~1e-7 (float32 limit) on every distribution.",
    )
    C.finish(fig, f"{C.ASSETS}/reconstruction_error.png")


if __name__ == "__main__":
    main()
