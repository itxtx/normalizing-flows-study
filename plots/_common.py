"""Shared style, datasets, model registry and training/caching helpers for the
README figures. Visual style mirrors the project's reference plotting style:
clean white background, blue = learned model, gray = data/baseline, red = accents.

Run figure scripts from the repo root after `pip install -e .`.
"""

import os
import time
import textwrap

import numpy as np
import torch
from torch.distributions import MultivariateNormal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.models import RealNVP, RealNVPSpline, NormalizingFlowModel
from src.flows import (
    MaskedAutoregressiveFlow,
    InverseAutoregressiveFlow,
    ContinuousFlow,
)

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

AGENT = "#2563EB"   # blue   -> learned model
HEUR = "#9CA3AF"    # gray   -> data / baseline
ACCENT = "#DC2626"  # red    -> reference lines / annotations
INK = "#374151"     # dark gray text
GRID = "#E5E7EB"

# qualitative palette for overlaying multiple flow families
PALETTE = {
    "RealNVP": "#2563EB",   # blue
    "Spline":  "#059669",   # green
    "MAF":     "#D97706",   # amber
    "IAF":     "#7C3AED",   # violet
    "CNF":     "#DC2626",   # red
}

# clean white -> brand-blue density colormap
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "brandblue", ["#FFFFFF", "#DBEAFE", "#93C5FD", "#3B82F6", "#1E3A8A"]
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "assets")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def style_axes(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Figure layout helpers (centered titles, wrapped captions, clean save)
#
# Create figures with layout="constrained" so the title and caption get proper
# spacing automatically, then save WITHOUT bbox_inches="tight" so the figure
# stays centered at its intended size (tight-bbox + manually placed text is what
# pushes plots off-center and lets long captions overflow the edges).
# --------------------------------------------------------------------------- #
def titled(fig, title, subtitle=None, size=14):
    """Centered, bold figure title with an optional second (subtitle) line."""
    head = title if not subtitle else f"{title}\n{textwrap.fill(subtitle, 80)}"
    fig.suptitle(head, fontsize=size, fontweight="bold")


def caption(fig, text, width=115):
    """Wrapped, centered footer caption that never runs off the figure."""
    fig.supxlabel(textwrap.fill(text, width), fontsize=9, color=INK)


def finish(fig, path):
    """Save centered at the figure's own size (no tight-bbox cropping)."""
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------- #
# Toy 2D datasets (each standardized to ~zero mean / unit std)
# --------------------------------------------------------------------------- #
def _standardize(x):
    x = np.asarray(x, dtype=np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    return x


def two_moons(n=4000, seed=0):
    from sklearn.datasets import make_moons
    x, _ = make_moons(n_samples=n, noise=0.07, random_state=seed)
    return _standardize(x)


def circles(n=4000, seed=0):
    from sklearn.datasets import make_circles
    x, _ = make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=seed)
    return _standardize(x * 2.0)


def checkerboard(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        c = rng.uniform(-2, 2, size=(n, 2))
        keep = (np.floor(c[:, 0]) + np.floor(c[:, 1])) % 2 == 0
        pts.extend(c[keep].tolist())
    return _standardize(np.array(pts[:n]))


def spirals(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    t = np.sqrt(rng.uniform(0, 1, n2)) * 3.0 * np.pi
    r = t
    a = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    b = np.stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)], 1)
    x = np.concatenate([a, b], 0)
    x += rng.normal(0, 0.25, x.shape)
    return _standardize(x)


DATASETS = {
    "moons": ("Two moons", two_moons),
    "circles": ("Circles", circles),
    "checkerboard": ("Checkerboard", checkerboard),
    "spirals": ("Two spirals", spirals),
}


def get_dataset(name, n=4000, seed=0):
    return torch.from_numpy(DATASETS[name][1](n=n, seed=seed))


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
def build_model(name, dim=2):
    name = name.lower()
    if name == "realnvp":
        return RealNVP(dim, 10, 128)
    if name == "spline":
        return RealNVPSpline(dim, 8, 64)
    if name == "maf":
        return NormalizingFlowModel([MaskedAutoregressiveFlow(dim, 64) for _ in range(6)])
    if name == "iaf":
        return NormalizingFlowModel([InverseAutoregressiveFlow(dim, 64) for _ in range(6)])
    if name == "cnf":
        return ContinuousFlow(dim, 64)
    raise ValueError(f"unknown model {name}")


FLOW_LABEL = {
    "realnvp": "RealNVP", "spline": "Spline", "maf": "MAF",
    "iaf": "IAF", "cnf": "CNF",
}

# sensible epoch budgets (full-batch) keeping each train < ~40s on CPU
EPOCHS = {"realnvp": 700, "spline": 250, "maf": 800, "iaf": 600, "cnf": 45}
# per-flow learning rate (RealNVP/Spline need a lower LR for stability)
LR = {"realnvp": 1e-3, "spline": 5e-4, "maf": 1e-3, "iaf": 1e-3, "cnf": 2e-2}
# CNF integrates an ODE per point, so train it on fewer points to stay fast
NDATA = {"realnvp": 2000, "spline": 2000, "maf": 2000, "iaf": 2000, "cnf": 600}


def base_dist(dim=2):
    return MultivariateNormal(torch.zeros(dim), torch.eye(dim))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, data, epochs, lr=1e-3, grad_clip=5.0, record=True):
    """Full-batch maximum-likelihood training. Returns the NLL curve (nats).
    Skips any step whose loss is non-finite (defensive against rare blow-ups)."""
    base = base_dist(data.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for ep in range(epochs):
        z, log_det = model.inverse(data)
        loss = -(base.log_prob(z) + log_det).mean()
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if record:
            curve.append(float(loss.item()))
    return curve


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def model_samples(model, n=4000, dim=2):
    base = base_dist(dim)
    z = base.sample((n,))
    x, _ = model.forward(z)
    return x


def recalibrate_bn(model, data, passes=50):
    """MADE-based flows (MAF/IAF) use BatchNorm; their eval-mode running stats
    can be miscalibrated after full-batch training, which blows up density
    evaluation. Recompute cumulative running stats from the data, then eval."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    if not bns:
        return model
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            model.inverse(data)
    model.eval()
    return model


@torch.no_grad()
def log_density_grid(model, lim=3.2, res=220):
    """Return (xx, yy, density) over a square grid via change of variables."""
    base = base_dist(2)
    xs = torch.linspace(-lim, lim, res)
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    z, log_det = model.inverse(grid)
    log_p = base.log_prob(z) + log_det
    dens = torch.exp(log_p).reshape(res, res)
    return xx.numpy(), yy.numpy(), dens.numpy()


@torch.no_grad()
def reconstruction_error(model, data):
    """x -> inverse(z) -> forward(x') ; return |x - x'| per point."""
    z, _ = model.inverse(data)
    x_rec, _ = model.forward(z)
    return (data - x_rec).abs().sum(1).numpy()


@torch.no_grad()
def samples_per_sec(model, n=4000, dim=2, reps=3):
    base = base_dist(dim)
    z = base.sample((n,))
    # warmup
    model.forward(z)
    t = time.time()
    for _ in range(reps):
        model.forward(z)
    dt = (time.time() - t) / reps
    return n / dt


# --------------------------------------------------------------------------- #
# Cache IO
# --------------------------------------------------------------------------- #
def cache_path(dataset, flow):
    return os.path.join(CACHE, f"{dataset}__{flow}.pt")


def save_cache(dataset, flow, model, curve, train_time):
    torch.save({
        "dataset": dataset,
        "flow": flow,
        "state_dict": model.state_dict(),
        "curve": curve,
        "params": count_params(model),
        "train_time": train_time,
        "sps": float(samples_per_sec(model)),
        "final_nll": float(np.mean(curve[-20:])) if curve else None,
    }, cache_path(dataset, flow))


def load_cache(dataset, flow):
    blob = torch.load(cache_path(dataset, flow), map_location="cpu")
    model = build_model(flow)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    blob["model"] = model
    return blob


def has_cache(dataset, flow):
    return os.path.exists(cache_path(dataset, flow))
