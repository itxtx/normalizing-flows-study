"""Hero gallery: rows = toy 2D datasets; columns = target data + each flow
family's learned density. Reads cached models from plots/_cache/.

All three families now produce calibrated, bit-exact-invertible densities (see
docs/bugfix-plan.md), so they can be compared head to head.
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as C

DATASETS = ["moons", "circles", "checkerboard", "spirals"]
FLOWS = ["realnvp", "spline", "maf"]
LIM = 3.0


def main():
    ncol = 1 + len(FLOWS)
    nrow = len(DATASETS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.55 * nrow),
                             layout="constrained")

    for r, ds in enumerate(DATASETS):
        # column 0: target samples
        d = C.get_dataset(ds, n=4000, seed=1).numpy()
        ax = axes[r][0]
        ax.scatter(d[:, 0], d[:, 1], s=2, c=C.HEUR, alpha=0.6, linewidths=0)
        ax.set_ylabel(C.DATASETS[ds][0], fontsize=12.5, color=C.INK)
        if r == 0:
            ax.set_title("Target data", fontsize=12.5, color=C.INK)

        # columns 1..: learned densities
        for c, fl in enumerate(FLOWS, start=1):
            ax = axes[r][c]
            blob = C.load_cache(ds, fl)
            xx, yy, dens = C.log_density_grid(blob["model"], lim=LIM, res=200)
            ax.pcolormesh(xx, yy, dens, cmap=C.DENSITY_CMAP, vmin=0,
                          vmax=np.percentile(dens, 99.5), shading="auto",
                          rasterized=True)
            if r == 0:
                ax.set_title(C.FLOW_LABEL[fl], fontsize=12.5, color=C.AGENT,
                             fontweight="bold")
            ax.text(0.04, 0.04, f"nll {blob['final_nll']:.2f}", transform=ax.transAxes,
                    fontsize=8, color="#6B7280", va="bottom", ha="left")

        for ax in axes[r]:
            ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
            ax.set_xticks([]); ax.set_yticks([])

    C.titled(fig, "One library, every flow family: learned densities on four 2D distributions")
    C.caption(
        fig,
        "Gray = target samples. Heatmaps = each flow's exact learned density via "
        "change of variables (white = low, blue = high). Coupling (RealNVP), spline, "
        "and autoregressive (MAF) flows all recover the multi-modal structure.",
    )
    C.finish(fig, f"{C.ASSETS}/gallery_density.png")


if __name__ == "__main__":
    main()
