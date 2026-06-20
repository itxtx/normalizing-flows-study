"""Training-progress GIF: a Real NVP density tightening onto the two-moons
target over training. Uses the same recipe as the notebook (which fits cleanly):

    data_dim = 2, n_samples = 20000, n_layers = 8, hidden_dim = 256,
    learning_rate = 1e-3, n_epochs = 1000, batch_size = 1024

Snapshots the learned density at a schedule of epochs and writes
assets/training_progress.gif.

Runtime: ~10-15 min on CPU; seconds on a GPU. Override the heavy knobs with
env vars for a quick preview, e.g.  GIF_EPOCHS=300 GIF_SAMPLES=8000 python fig_gif.py
"""

import os

import numpy as np
import torch
from torch.distributions import MultivariateNormal
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.models import RealNVP
import _common as C

# ---- notebook recipe (override via env for a faster preview) ----
DATA_DIM = 2
N_SAMPLES = int(os.environ.get("GIF_SAMPLES", 20000))
N_LAYERS = int(os.environ.get("GIF_LAYERS", 8))
HIDDEN_DIM = int(os.environ.get("GIF_HIDDEN", 256))
LR = float(os.environ.get("GIF_LR", 1e-3))
N_EPOCHS = int(os.environ.get("GIF_EPOCHS", 1000))
BATCH_SIZE = int(os.environ.get("GIF_BATCH", 1024))

LIM = 3.0
RES = 110


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def snapshot_epochs(n_epochs):
    """Dense early (fast change), sparser later -> a smooth ~40-frame GIF."""
    early = [0, 2, 5, 10, 15, 20, 30, 40, 60, 80]
    rest = list(range(100, n_epochs + 1, max(25, n_epochs // 36)))
    return sorted(set(e for e in early + rest if e <= n_epochs) | {n_epochs})


@torch.no_grad()
def density_grid(model, device, lim=LIM, res=RES):
    xs = torch.linspace(-lim, lim, res)
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1).to(device)
    base = MultivariateNormal(
        torch.zeros(2, device=device), torch.eye(2, device=device)
    )
    z, log_det = model.inverse(grid)
    dens = torch.exp(base.log_prob(z) + log_det).reshape(res, res)
    return xx.numpy(), yy.numpy(), dens.cpu().numpy()


def main():
    torch.manual_seed(0)
    device = get_device()
    print(
        f"device: {device}  |  {N_LAYERS} layers, hidden {HIDDEN_DIM}, "
        f"{N_SAMPLES} samples, {N_EPOCHS} epochs, batch {BATCH_SIZE}"
    )

    data = C.get_dataset("moons", N_SAMPLES).to(device)
    base = MultivariateNormal(
        torch.zeros(DATA_DIM, device=device), torch.eye(DATA_DIM, device=device)
    )
    model = RealNVP(DATA_DIM, N_LAYERS, HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    snaps_at = set(snapshot_epochs(N_EPOCHS))
    frames = []  # (epoch, nll, density)

    def take_snapshot(ep, nll):
        model.eval()
        frames.append((ep, nll, density_grid(model, device)[2]))
        model.train()

    take_snapshot(0, float("nan"))
    n = data.shape[0]
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        running = 0.0
        nb = 0
        for i in range(0, n, BATCH_SIZE):
            xb = data[perm[i : i + BATCH_SIZE]]
            z, log_det = model.inverse(xb)
            loss = -(base.log_prob(z) + log_det).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        epoch_nll = running / max(nb, 1)
        if epoch in snaps_at:
            take_snapshot(epoch, epoch_nll)
            print(f"epoch {epoch:4d}/{N_EPOCHS}  nll {epoch_nll:.3f}")

    # ---- render frames ----
    vmax = np.percentile(frames[-1][2], 99.5)
    d_np = data.detach().cpu().numpy()
    xs = np.linspace(-LIM, LIM, RES)
    xx, yy = np.meshgrid(xs, xs)

    imgs = []
    for ep, nll, dens in frames:
        fig = plt.figure(figsize=(4.4, 4.4))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.84])
        ax.pcolormesh(
            xx, yy, dens, cmap=C.DENSITY_CMAP, vmin=0, vmax=vmax, shading="auto"
        )
        ax.scatter(d_np[:, 0], d_np[:, 1], s=0.6, c="#111827", alpha=0.10, linewidths=0)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_xticks([])
        ax.set_yticks([])
        label = f"Real NVP — epoch {ep:4d}" + (
            "" if np.isnan(nll) else f"   NLL {nll:.2f}"
        )
        ax.set_title(label, fontsize=12.5, color=C.AGENT, fontweight="bold")
        canvas.draw()
        imgs.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    seq = [imgs[0]] * 4 + imgs + [imgs[-1]] * 8  # hold first/last
    out = f"{C.ASSETS}/training_progress.gif"
    imageio.mimsave(out, seq, duration=0.16, loop=0)
    print("wrote", out, "| frames:", len(seq))


if __name__ == "__main__":
    main()
