"""Build the figures used in the results note.

    python experiments/make_figures.py

Reads results/convergence.csv and results/stress.json; writes
results/figures/*.pdf and *.png.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.0,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

ORDER = ["reference", "sine", "expbubble", "gaussbump", "harmonic", "layer", "mixed"]
STYLE = {
    "reference": ("k", "o"),
    "sine": ("#1f77b4", "s"),
    "expbubble": ("#2ca02c", "^"),
    "gaussbump": ("#9467bd", "D"),
    "harmonic": ("#ff7f0e", "v"),
    "layer": ("#d62728", "P"),
    "mixed": ("#8c564b", "X"),
}
LABEL = {
    "reference": "reference (single sine mode)",
    "sine": "sine series",
    "expbubble": "exp bubble",
    "gaussbump": "Gaussian bump",
    "harmonic": "harmonic, $g\\neq0$",
    "layer": "sharp layer",
    "mixed": "mixed",
}


def fig_convergence():
    rows = list(csv.DictReader(open(RES / "convergence.csv")))
    by = defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.45))
    handles = []
    for k in ORDER:
        v = sorted(by[k], key=lambda r: int(r["m"]))
        m = np.array([int(r["m"]) for r in v])
        e = np.maximum(np.array([float(r["max_abs_error"]) for r in v]), 1e-16)
        res = np.maximum(np.array([float(r["residual"]) for r in v]), 1e-18)
        c, mk = STYLE[k]
        (h,) = ax[0].semilogy(m, e, "-", color=c, marker=mk, label=LABEL[k])
        ax[1].semilogy(m, res, "-", color=c, marker=mk)
        handles.append(h)

    # panel (a): reference levels, labelled in the empty upper-right corner
    ax[0].axhline(2.85e-9, color="0.35", ls=":", lw=1.0)
    ax[0].axhline(2.85e-3, color="0.55", ls="--", lw=1.0)
    ax[0].text(645, 2.85e-9 * 1.9, r"$E_{\max}$ term saturated ($\tau/10^{6}$)",
               fontsize=6.5, color="0.25", ha="right", va="bottom")
    ax[0].text(645, 2.85e-3 * 1.9, "Stage-1 target $2.85\\times10^{-3}$",
               fontsize=6.5, color="0.45", ha="right", va="bottom")
    ax[0].set_xlabel("CGL nodes per direction, $m$")
    ax[0].set_ylabel(r"$\|u_h-u\|_\infty$")
    ax[0].set_title("(a) error against the exact solution")
    ax[0].set_xlim(20, 660)
    ax[0].set_ylim(1e-16, 5e0)

    ax[1].axhline(1e-13, color="0.35", ls=":", lw=1.0)
    ax[1].text(645, 1e-13 * 1.9, "collocation residual stays at roundoff",
               fontsize=6.5, color="0.25", ha="right", va="bottom")
    ax[1].set_xlabel("CGL nodes per direction, $m$")
    ax[1].set_ylabel("relative discrete residual")
    ax[1].set_title("(b) relative discrete residual")
    ax[1].set_xlim(20, 660)
    ax[1].set_ylim(1e-18, 1e-7)

    fig.legend(handles=handles, labels=[LABEL[k] for k in ORDER], loc="lower center",
               ncol=4, frameon=False, handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"convergence.{ext}")
    plt.close(fig)
    print("wrote", FIG / "convergence.pdf")


def fig_stress():
    d = json.loads((RES / "stress.json").read_text())
    fams = ["sine", "expbubble", "gaussbump", "layer", "harmonic", "mixed"]
    metrics = [
        ("mae_max", r"MAE$_{\max}$", "#4c72b0", "o"),
        ("max_abs_error_max", r"$E_{\max}$", "#55a868", "s"),
        ("relative_l2_max", r"$E_{rel,L2}$", "#c44e52", "^"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    y = np.arange(len(fams))[::-1]
    handles = []
    for j, (key, lab, col, mk) in enumerate(metrics):
        vals = [d["by_family"][f][key] for f in fams]
        (h,) = ax.semilogx(vals, y + (j - 1) * 0.2, mk, color=col, label=lab,
                           ms=4.5, linestyle="none")
        handles.append(h)
    ax.set_yticks(y)
    ax.set_yticklabels(fams)
    ax.set_xlim(1e-14, 3e-1)
    ax.set_ylim(-1.05, len(fams) - 0.45)
    ax.set_xlabel("worst value over 40 hidden-like cases")
    ax.axvline(5e-2, color="0.35", ls="--", lw=1.0)
    ax.axvline(2.39e-10, color="0.35", ls=":", lw=1.0)
    ax.text(5e-2, -0.62, "Stage-2 limit", fontsize=6.5, color="0.3",
            ha="center", va="center")
    ax.text(2.39e-10, -0.62, "MAE scoring floor", fontsize=6.5, color="0.3",
            ha="center", va="center")
    ax.set_title(f"hidden-like families, {d['seeds']} seeds $\\times$ 2 cases: "
                 f"worst metric per family", pad=14)
    ax.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              frameon=False, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"stress.{ext}")
    plt.close(fig)
    print("wrote", FIG / "stress.pdf")


def fig_runtime():
    d = json.loads((RES / "stress.json").read_text())
    rows = d["rows"]
    grids = sorted({r["n"] for r in rows})
    data = [[r["runtime_s"] for r in rows if r["n"] == g] for g in grids]
    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    bp = ax.boxplot(data, tick_labels=[str(g) for g in grids], widths=0.55,
                    patch_artist=True, medianprops=dict(color="black"))
    for b in bp["boxes"]:
        b.set(facecolor="#a6bddb", alpha=0.85)
    ax.set_xlabel("requested grid $N$")
    ax.set_ylabel("wall-clock per case (s)")
    ax.set_title("240 hidden-like cases")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"runtime.{ext}")
    plt.close(fig)
    print("wrote", FIG / "runtime.pdf")


if __name__ == "__main__":
    fig_convergence()
    if (RES / "stress.json").exists():
        fig_stress()
        fig_runtime()
