"""Benchmark across flow families on two-moons (2D models, CPU): test NLL vs
parameter count, with marker size encoding sampling throughput. Now that every
family produces a calibrated likelihood, the NLL axis is meaningful.
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as C

FLOWS = ["realnvp", "spline", "maf", "iaf", "cnf"]


def main():
    rows = []
    for fl in FLOWS:
        b = C.load_cache("moons", fl)
        rows.append((C.FLOW_LABEL[fl], b["params"], b["sps"], b["final_nll"]))

    # marker area scaled by log10(samples/sec)
    sps_all = np.array([r[2] for r in rows])
    smin, smax = np.log10(sps_all.min()), np.log10(sps_all.max())

    # nudge labels so the overlapping MAF/IAF (same size) don't collide
    offsets = {"RealNVP": (14, 10), "Spline": (14, 10), "MAF": (16, 12),
               "IAF": (16, -26), "CNF": (14, 10)}

    fig, ax = plt.subplots(figsize=(9.2, 5.6), layout="constrained")
    for name, params, sps, nll in rows:
        frac = (np.log10(sps) - smin) / (smax - smin + 1e-9)
        size = 160 + frac * 900
        ax.scatter(params, nll, s=size, color=C.PALETTE[name], alpha=0.85,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(f"{name}  ·  nll {nll:.2f}  ·  {sps / 1e3:.0f}k smp/s",
                    (params, nll), textcoords="offset points",
                    xytext=offsets.get(name, (14, 10)), fontsize=9.5, color=C.INK)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters  (log scale)")
    ax.set_ylabel("Test NLL on two-moons  (nats, lower = better)")
    ax.set_xlim(3e3, 9e5)
    ax.set_ylim(1.3, 2.7)
    C.style_axes(ax, grid_axis="both")

    C.titled(
        fig,
        "Flow families on two-moons: quality vs size vs speed",
        "Coupling and spline flows reach the lowest NLL; autoregressive flows fit less sharply",
    )
    C.caption(
        fig,
        "2D models, CPU. NLL is the final training negative log-likelihood; "
        "marker area grows with samples/sec. RealNVP and Spline are bit-exact "
        "invertible; all families are now calibrated (see docs/bugfix-plan.md).",
    )
    C.finish(fig, f"{C.ASSETS}/benchmark.png")


if __name__ == "__main__":
    main()
