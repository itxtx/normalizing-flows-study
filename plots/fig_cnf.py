"""Continuous flow: integration trajectories and velocity field of a Real,
maximum-likelihood-trained CNF (loaded from plots/_cache/moons__cnf.pt).

Now that ODEFunc computes the true divergence, the CNF trains by maximum
likelihood; here we integrate its learned velocity field dz/dt = v(z, t) to
carry the Gaussian base onto the two-moons target.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import _common as C

LIM = 3.0
STEPS = 60


@torch.no_grad()
def main():
    model = C.load_cache("moons", "cnf")["model"]
    model.eval()
    ode = model.ode_func
    base = C.base_dist()

    # integrate base samples forward in time (Euler) through the learned field
    pts = base.sample((1400,))
    traj = [pts.clone()]
    for s in range(STEPS):
        t = torch.tensor(s / STEPS)
        pts = pts + ode._velocity(t, pts) * (1.0 / STEPS)
        traj.append(pts.clone())
    traj = torch.stack(traj).numpy()              # (STEPS+1, N, 2)

    # velocity field on a grid at the midpoint of integration
    gx = np.linspace(-LIM, LIM, 22)
    GX, GY = np.meshgrid(gx, gx)
    grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
    V = ode._velocity(torch.tensor(0.5), grid).numpy()
    U = V[:, 0].reshape(GX.shape); Vv = V[:, 1].reshape(GX.shape)

    snaps = [0, STEPS // 3, 2 * STEPS // 3, STEPS]
    titles = ["t = 0  (Gaussian)", "t = 0.33", "t = 0.67", "t = 1  (data)"]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.7), layout="constrained")
    for ax, s, ttl in zip(axes, snaps, titles):
        ax.quiver(GX, GY, U, Vv, color="#CBD5E1", angles="xy",
                  scale_units="xy", scale=4.0, width=0.004)
        P = traj[s]
        ax.scatter(P[:, 0], P[:, 1], s=4, c="#111827" if s == 0 else C.AGENT,
                   alpha=0.5, linewidths=0)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=12, color=C.INK if s == 0 else C.AGENT,
                     fontweight="normal" if s == 0 else "bold")

    for i in range(0, traj.shape[1], 70):
        axes[-1].plot(traj[:, i, 0], traj[:, i, 1], color=C.ACCENT, lw=0.6, alpha=0.5)

    C.titled(fig, "Continuous flow: the Gaussian base transported along a learned velocity field")
    C.caption(
        fig,
        "A maximum-likelihood-trained CNF. Gray arrows = the ODE velocity field "
        "v(z, t); integrating the Gaussian base (black) forward in time carries it "
        "onto the two-moons target (blue); red curves trace individual trajectories.",
    )
    C.finish(fig, f"{C.ASSETS}/cnf_trajectories.png")


if __name__ == "__main__":
    main()
