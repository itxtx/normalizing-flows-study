"""Training curves: Real NVP negative log-likelihood vs epoch on the four toy
distributions, with a secondary bits/dim axis. Reads cached curves.

(Real NVP is used for a like-for-like comparison: its NLL is an exact,
batch-norm-free likelihood, so the curves are directly comparable across
datasets. MAF/IAF use BatchNorm, whose log-det offsets make their raw loss
values non-comparable.)
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as C

DATASETS = ["moons", "circles", "checkerboard", "spirals"]
DS_COLOR = {
    "moons": "#2563EB",
    "circles": "#059669",
    "checkerboard": "#D97706",
    "spirals": "#7C3AED",
}
D = 2
BPD = 1.0 / (D * np.log(2))


def smooth(y, k=9):
    ker = np.ones(k) / k
    return np.convolve(np.asarray(y), ker, mode="valid")


def main():
    fig, ax = plt.subplots(figsize=(9.0, 5.4), layout="constrained")

    finals = []
    for ds in DATASETS:
        curve = C.load_cache(ds, "realnvp")["curve"]
        ys = smooth(curve, 9)
        ax.plot(
            np.arange(len(ys)),
            ys,
            color=DS_COLOR[ds],
            linewidth=2.2,
            label=C.DATASETS[ds][0],
        )
        finals.append((C.DATASETS[ds][0], float(np.mean(curve[-20:]))))

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Negative log-likelihood  (nats, lower = better)")
    ax.set_xlim(0, 700)
    ax.set_ylim(1.3, 3.0)  # clip an early transient spike on checkerboard
    C.style_axes(ax, grid_axis="both")
    ax.legend(loc="upper right", frameon=False, title="Target distribution")

    lo, hi = ax.get_ylim()
    ax2 = ax.twinx()
    ax2.set_ylim(lo * BPD, hi * BPD)
    ax2.set_ylabel("bits / dim")
    ax2.spines["top"].set_visible(False)

    C.titled(
        fig,
        "Real NVP converges on every distribution",
        "Negative log-likelihood falls from ~2.8 to 1.5–2.0 nats within a few hundred epochs",
    )
    sub = "    ".join(f"{n}: {v:.2f}" for n, v in finals)
    C.caption(fig, "Final NLL  →   " + sub)
    C.finish(fig, f"{C.ASSETS}/training_curves.png")


if __name__ == "__main__":
    main()
