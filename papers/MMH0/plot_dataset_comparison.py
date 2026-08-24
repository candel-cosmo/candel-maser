#!/usr/bin/env python3
"""Input-spot-table comparison figure for the MMH0 paper.

  Angular-diameter distance posterior of each galaxy under the four input
  spot tables, together with the Pesce+2020 posterior.  The priors, sampler,
  and disc model are held fixed across the four inferences; only the input
  table changes.  -> figs/mcp_dataset_comparison.pdf

The two baselines (updated-MCP, updated-ours) carry the current MCP
measurements; original-all and updated-all are diagnostics that separate the
measurement update from the spot removal, and are labelled as such.

The Pesce+2020 curve is a two-piece Gaussian built from the median and the
asymmetric 1-sigma interval of their table 1, which is all they report.

Run from the repo root with venv_candel:
    python notebooks/paper_MMH0/plot_dataset_comparison.py
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from palette import BROWN, GOLD, RED, TEAL

ROOT = Path(__file__).resolve().parents[2]
P20_TABLE = ROOT / "data" / "Megamaser" / "pesce2020_table1.csv"
OUTDIR = "/Users/rstiskalek/Papers/MMH0/figs"

# Plotted in this order, so the legend reads baselines first.  Keys are the
# results/Megamaser/ directory names; labels are the names the paper uses in
# Section 2.1, with the two diagnostic tables marked.
DATASETS = [
    ("clipped", r"\texttt{updated-ours}", GOLD),
    ("fiducial", r"\texttt{updated-MCP}", RED),
    ("original_published", r"\texttt{original-all} (diagnostic)", TEAL),
    ("unpruned", r"\texttt{updated-all} (diagnostic)", BROWN),
]

GALAXIES = ["CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789"]
GAL_LABEL = {"CGCG074-064": "CGCG 074-064", "NGC5765b": "NGC 5765b",
             "NGC6264": "NGC 6264", "NGC6323": "NGC 6323",
             "UGC3789": "UGC 3789"}
# Panel slots: three on the top row, two on the bottom, leaving the
# bottom-left slot for the legend.
SLOTS = {"CGCG074-064": (0, 0), "NGC5765b": (0, 1), "NGC6264": (0, 2),
         "NGC6323": (1, 1), "UGC3789": (1, 2)}


def chain_path(dataset, galaxy):
    return (ROOT / "results" / "Megamaser" / dataset / galaxy
            / f"{galaxy}_blackjax_mcmc_rphi_initconfig.hdf5")


def load_DA(dataset, galaxy):
    """Angular-diameter distance samples, or None if the run is absent."""
    p = chain_path(dataset, galaxy)
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        return np.asarray(f["samples/D_A"], dtype=float).ravel()


def load_p20():
    """{galaxy: (median, upper, lower)} in Mpc from the Pesce+2020 table."""
    out = {}
    for line in open(P20_TABLE):
        if line.startswith("#") or line.startswith("galaxy"):
            continue
        f = line.strip().split(",")
        out[f[0].replace(" ", "")] = (float(f[1]), float(f[2]), float(f[3]))
    return out


def split_normal(x, med, up, lo):
    """Two-piece Gaussian with the given median and asymmetric 1-sigma
    half-widths, normalised to unit area."""
    sig = np.where(x >= med, up, lo)
    d = np.exp(-0.5 * ((x - med) / sig) ** 2)
    return d / np.trapezoid(d, x)


def fig_dataset_comparison(out):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    from scipy.stats import gaussian_kde

    p20 = load_p20()
    with plt.style.context(["science"]):
        fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.0))
        for ax in axs.ravel():
            ax.set_visible(False)

        for gal in GALAXIES:
            r, c = SLOTS[gal]
            ax = axs[r, c]
            ax.set_visible(True)

            curves, lo, hi = [], [], []
            for dataset, label, colour in DATASETS:
                s = load_DA(dataset, gal)
                if s is None:
                    print(f"missing: {dataset}/{gal}")
                    continue
                curves.append((s, label, colour))
                lo.append(np.percentile(s, 0.1))
                hi.append(np.percentile(s, 99.9))

            med, up, dn = p20[gal]
            lo.append(med - 4 * dn)
            hi.append(med + 4 * up)
            grid = np.linspace(max(0.0, min(lo)), max(hi), 512)

            for s, label, colour in curves:
                ax.plot(grid, gaussian_kde(s)(grid), color=colour, lw=1.4,
                        label=label)
            ax.plot(grid, split_normal(grid, med, up, dn), color="black",
                    ls="--", lw=1.6, label="P20-reported posterior")

            ax.text(0.97, 0.93, GAL_LABEL[gal], transform=ax.transAxes,
                    ha="right", va="top", fontsize=14)
            ax.set_xlabel(r"$D_\mathrm{A}$ [Mpc]")
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(bottom=0.0)
            ax.set_yticks([])
            if c == 0 or gal == "NGC6323":
                ax.set_ylabel("Posterior density")

        # Legend goes in the empty bottom-left slot.
        handles, labels = axs[0, 0].get_legend_handles_labels()
        axs[1, 0].set_visible(True)
        axs[1, 0].axis("off")
        axs[1, 0].legend(handles, labels, loc="center", frameon=False,
                         handlelength=1.8, fontsize=14)

        fig.tight_layout()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"   [INFO] Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUTDIR)
    args = ap.parse_args()
    fig_dataset_comparison(os.path.join(args.out_dir,
                                        "mcp_dataset_comparison.pdf"))
