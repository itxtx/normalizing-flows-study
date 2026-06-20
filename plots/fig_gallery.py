"""Hero gallery: rows = toy 2D datasets; columns = target data, the learned
density, and samples drawn from the trained flow. Uses the RealNVP (coupling)
models cached in plots/_cache/ -- numerically the most stable for exact
change-of-variables density evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as C

DATASETS = ["moons", "circles", "checkerboard", "spirals"]
LIM = 3.0


def main():
    nrow = len(DATASETS)
    fig, axes = plt.subplots(nrow, 3, figsize=(8.6, 2.75 * nrow))

    for r, ds in enumerate(DATASETS):
        model = C.load_cache(ds, "realnvp")["model"]

        # column 0 -- target samples
        d = C.get_dataset(ds, n=4000, seed=1).numpy()
        ax = axes[r][0]
        ax.scatter(d[:, 0], d[:, 1], s=2, c=C.HEUR, alpha=0.6, linewidths=0)

        # column 1 -- learned density (exact, via change of variables)
        ax1 = axes[r][1]
        xx, yy, dens = C.log_density_grid(model, lim=LIM, res=220)
        ax1.pcolormesh(
            xx,
            yy,
            dens,
            cmap=C.DENSITY_CMAP,
            vmin=0,
            vmax=np.percentile(dens, 99.5),
            shading="auto",
            rasterized=True,
        )

        # column 2 -- samples from the flow (z ~ N(0, I) -> x)
        ax2 = axes[r][2]
        xs = C.model_samples(model, 30000).numpy()
        ax2.hist2d(
            xs[:, 0],
            xs[:, 1],
            bins=140,
            range=[[-LIM, LIM], [-LIM, LIM]],
            cmap=C.DENSITY_CMAP,
        )

        for ax in (axes[r][0], ax1, ax2):
            ax.set_xlim(-LIM, LIM)
            ax.set_ylim(-LIM, LIM)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(C.DATASETS[ds][0], fontsize=12.5, color=C.INK)

    axes[0][0].set_title("Target data", fontsize=12.5, color=C.INK)
    axes[0][1].set_title(
        "Learned density", fontsize=12.5, color=C.AGENT, fontweight="bold"
    )
    axes[0][2].set_title(
        "Samples from the flow", fontsize=12.5, color=C.AGENT, fontweight="bold"
    )

    fig.suptitle(
        "Real NVP learns to model — and generate — four 2D distributions",
        fontsize=15,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=0.997,
    )
    fig.text(
        0.012,
        0.011,
        "Left: target samples (gray).  Middle: the flow's exact learned "
        "density via change of variables.  Right: new samples drawn from "
        "the trained flow.",
        fontsize=9.5,
        color=C.INK,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1, 0.965))
    out = f"{C.ASSETS}/gallery_density.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
