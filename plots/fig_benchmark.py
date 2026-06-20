"""Benchmark: parameter count vs sampling throughput across flow families
(2D models, CPU). Reads cached metrics. Marker color = family; an exact,
calibrated likelihood is only available for the coupling/spline families in the
current implementation, so this compares architecture and speed rather than NLL.
"""

import numpy as np
import matplotlib.pyplot as plt

import _common as C

FLOWS = ["realnvp", "spline", "maf", "iaf", "cnf"]


def main():
    rows = []
    for fl in FLOWS:
        b = C.load_cache("moons", fl)
        rows.append((C.FLOW_LABEL[fl], b["params"], b["sps"]))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for name, params, sps in rows:
        ax.scatter(params, sps, s=240, color=C.PALETTE[name], alpha=0.9,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(f"{name}\n{params/1e3:.0f}k params · {sps/1e3:.0f}k smp/s",
                    (params, sps), textcoords="offset points", xytext=(12, 8),
                    fontsize=9.5, color=C.INK)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Parameters  (log scale)")
    ax.set_ylabel("Sampling throughput  (samples / sec, log scale)")
    ax.set_xlim(3e3, 8e5)
    ax.set_ylim(4e3, 2e6)
    C.style_axes(ax, grid_axis="both")

    ax.set_title("Flow families: parameters vs sampling speed\n"
                 "Autoregressive flows sample fastest at these sizes; "
                 "the continuous flow is smallest but slowest (ODE solves)",
                 fontsize=14, fontweight="bold", loc="left")
    fig.text(0.012, 0.012,
             "2D models on CPU, measured over batches of 4,000 samples.  "
             "IAF samples in one pass; MAF's fast direction is density "
             "evaluation; the CNF integrates an ODE per sample.",
             fontsize=9, color=C.INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = f"{C.ASSETS}/benchmark.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
