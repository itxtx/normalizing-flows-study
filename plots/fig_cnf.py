"""Continuous flow: integration trajectories and velocity field.

The repo's ODEFunc defines a velocity field dz/dt = net(z). Here that network
is trained with a flow-matching objective to transport the Gaussian base onto
the two-moons target, then integrated over time to show how the point cloud
moves along the learned field. (Flow matching is used because the ODEFunc's
exact log-determinant term is not implemented, so maximum-likelihood CNF
training collapses -- see the README notes.)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.flows import ODEFunc
import _common as C

LIM = 3.0


def main():
    torch.manual_seed(0)
    data = C.get_dataset("moons", 3000)
    ode = ODEFunc(2, 96)
    opt = torch.optim.Adam(ode.net.parameters(), 3e-3)

    # ---- flow-matching training of the velocity net ----
    n = data.shape[0]
    for step in range(4000):
        idx = torch.randint(0, n, (512,))
        x1 = data[idx]
        x0 = torch.randn(512, 2)
        t = torch.rand(512, 1)
        xt = (1 - t) * x0 + t * x1
        v_target = x1 - x0
        v_pred = ode.net(xt)
        loss = ((v_pred - v_target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- integrate base samples along the field (Euler) ----
    with torch.no_grad():
        steps = 80
        pts = torch.randn(1200, 2)
        traj = [pts.clone()]
        for _ in range(steps):
            pts = pts + ode.net(pts) * (1.0 / steps)
            traj.append(pts.clone())
        traj = torch.stack(traj).numpy()          # (steps+1, N, 2)

        # velocity field on a grid
        gx = np.linspace(-LIM, LIM, 22)
        GX, GY = np.meshgrid(gx, gx)
        grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
        V = ode.net(grid).numpy()
        U = V[:, 0].reshape(GX.shape); Vv = V[:, 1].reshape(GX.shape)

    snaps = [0, steps // 3, 2 * steps // 3, steps]
    titles = ["t = 0  (Gaussian)", "t = 0.33", "t = 0.67", "t = 1  (data)"]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.7), layout="constrained")
    for ax, s, ttl in zip(axes, snaps, titles):
        ax.quiver(GX, GY, U, Vv, color="#CBD5E1", angles="xy",
                  scale_units="xy", scale=4.0, width=0.004)
        P = traj[s]
        c = "#111827" if s == 0 else C.AGENT
        ax.scatter(P[:, 0], P[:, 1], s=4, c=c, alpha=0.5, linewidths=0)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=12,
                     color=C.INK if s == 0 else C.AGENT,
                     fontweight="normal" if s == 0 else "bold")

    # overlay a few trajectories on the last panel
    for i in range(0, 1200, 60):
        axes[-1].plot(traj[:, i, 0], traj[:, i, 1], color=C.ACCENT,
                      lw=0.6, alpha=0.5)

    C.titled(fig, "Continuous flow: a point cloud transported along a learned velocity field")
    C.caption(
        fig,
        "The ODEFunc network defines dz/dt = v(z) (gray arrows). Integrating "
        "the Gaussian base (black) forward in time carries it onto the "
        "two-moons target (blue); red curves trace individual trajectories.",
    )
    C.finish(fig, f"{C.ASSETS}/cnf_trajectories.png")


if __name__ == "__main__":
    main()
