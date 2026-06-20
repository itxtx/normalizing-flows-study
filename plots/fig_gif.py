"""Training-progress GIF: a Real NVP density tightening onto the two-moons
target over training epochs. Trains from scratch while snapshotting the
learned density, then writes assets/training_progress.gif.
"""

import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import _common as C

DATASET = "moons"
LIM = 3.0
RES = 95
SNAP_EVERY = 30
EPOCHS = 750
LR = 1.3e-3


def density(model):
    xx, yy, d = C.log_density_grid(model, lim=LIM, res=RES)
    return xx, yy, d


def main():
    torch.manual_seed(0)
    data = C.get_dataset(DATASET, 2000)
    base = C.base_dist()
    model = C.build_model("realnvp")
    opt = torch.optim.Adam(model.parameters(), LR)

    snaps = []  # (epoch, density, nll)
    for ep in range(EPOCHS + 1):
        if ep % SNAP_EVERY == 0:
            with torch.no_grad():
                z, ld = model.inverse(data)
                nll = float(-(base.log_prob(z) + ld).mean())
            snaps.append((ep, density(model)[2], nll))
        z, ld = model.inverse(data)
        loss = -(base.log_prob(z) + ld).mean()
        if not torch.isfinite(loss):
            break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    vmax = np.percentile(snaps[-1][1], 99.5)
    d_np = data.numpy()

    frames = []
    for ep, dens, nll in snaps:
        fig = plt.figure(figsize=(4.2, 4.2))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.84])
        xs = np.linspace(-LIM, LIM, RES)
        xx, yy = np.meshgrid(xs, xs)
        ax.pcolormesh(xx, yy, dens, cmap=C.DENSITY_CMAP, vmin=0, vmax=vmax, shading="auto")
        ax.scatter(d_np[:, 0], d_np[:, 1], s=1.0, c="#111827", alpha=0.18, linewidths=0)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Real NVP — epoch {ep:3d}   NLL {nll:.2f}",
                     fontsize=12.5, color=C.AGENT, fontweight="bold")
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
        plt.close(fig)

    # ease: hold first and last frames
    seq = [frames[0]] * 3 + frames + [frames[-1]] * 6
    out = f"{C.ASSETS}/training_progress.gif"
    imageio.mimsave(out, seq, duration=0.18, loop=0)
    print("wrote", out, "frames:", len(seq))


if __name__ == "__main__":
    main()
