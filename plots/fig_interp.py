"""Latent-space structure: how a Real NVP warps the Gaussian latent space onto
the data manifold. Left = latent space z (a regular grid + an interpolation
path). Right = the same grid and path pushed through the flow into data space,
over the learned density. Uses the cached moons RealNVP.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import _common as C

LIM = 3.0


@torch.no_grad()
def main():
    model = C.load_cache("moons", "realnvp")["model"]

    # a regular grid of lines in latent space
    g = np.linspace(-2.2, 2.2, 13)
    fine = np.linspace(-2.6, 2.6, 200)
    h_lines = [np.stack([fine, np.full_like(fine, c)], 1) for c in g]  # horizontal
    v_lines = [np.stack([np.full_like(fine, c), fine], 1) for c in g]  # vertical

    # a latent interpolation path (straight line in z)
    z0, z1 = np.array([-1.8, -1.4]), np.array([1.7, 1.5])
    tt = np.linspace(0, 1, 60)[:, None]
    path = z0[None] * (1 - tt) + z1[None] * tt
    marks = np.linspace(0, 1, 7)[:, None]
    path_marks = z0[None] * (1 - marks) + marks * z1[None]

    def push(arr):
        x, _ = model.forward(torch.from_numpy(arr.astype(np.float32)))
        return x.numpy()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 5.2), layout="constrained")

    # ---- latent space ----
    for ln in h_lines:
        axL.plot(ln[:, 0], ln[:, 1], color="#93C5FD", lw=0.8)
    for ln in v_lines:
        axL.plot(ln[:, 0], ln[:, 1], color="#93C5FD", lw=0.8)
    axL.plot(path[:, 0], path[:, 1], color=C.ACCENT, lw=2.4, zorder=5)
    axL.scatter(
        path_marks[:, 0],
        path_marks[:, 1],
        color=C.ACCENT,
        s=42,
        zorder=6,
        edgecolor="white",
        linewidth=1.0,
    )
    axL.set_title("Latent space  z ~ N(0, I)", fontsize=13, color=C.INK, loc="left")
    axL.set_xlim(-LIM, LIM)
    axL.set_ylim(-LIM, LIM)

    # ---- data space ----
    xx, yy, dens = C.log_density_grid(model, lim=LIM, res=220)
    axR.pcolormesh(
        xx,
        yy,
        dens,
        cmap=C.DENSITY_CMAP,
        vmin=0,
        vmax=np.percentile(dens, 99.5),
        shading="auto",
        rasterized=True,
    )
    for ln in h_lines:
        p = push(ln)
        axR.plot(p[:, 0], p[:, 1], color="#1D4ED8", lw=0.7, alpha=0.7)
    for ln in v_lines:
        p = push(ln)
        axR.plot(p[:, 0], p[:, 1], color="#1D4ED8", lw=0.7, alpha=0.7)
    pp = push(path)
    axR.plot(pp[:, 0], pp[:, 1], color=C.ACCENT, lw=2.4, zorder=5)
    ppm = push(path_marks)
    axR.scatter(
        ppm[:, 0],
        ppm[:, 1],
        color=C.ACCENT,
        s=42,
        zorder=6,
        edgecolor="white",
        linewidth=1.0,
    )
    axR.set_title(
        "Data space  x = f(z)   (over learned density)",
        fontsize=13,
        color=C.AGENT,
        loc="left",
        fontweight="bold",
    )
    axR.set_xlim(-LIM, LIM)
    axR.set_ylim(-LIM, LIM)

    for ax in (axL, axR):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    C.titled(fig, "A normalizing flow is a learned, invertible warp of space")
    C.caption(
        fig,
        "The flow bends the regular Gaussian grid (left) into the data "
        "manifold (right). A straight line in latent space (red) becomes a "
        "smooth path along the data distribution.",
    )
    C.finish(fig, f"{C.ASSETS}/latent_interpolation.png")


if __name__ == "__main__":
    main()
