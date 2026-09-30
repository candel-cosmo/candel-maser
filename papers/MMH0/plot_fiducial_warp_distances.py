#!/usr/bin/env python3
"""Compare fiducial linear- and quadratic-warp distance posteriors.

Reads the config-initialised, non-eccentric MCMC chains for each of the five
MCP H0 galaxies and writes one five-panel PDF.

Run from the repository root with::

    venv_candel/bin/python papers/MMH0/plot_fiducial_warp_distances.py
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import gaussian_kde

from candel_maser.maser_config import check_chain_dataset


from candel_maser.paths import CANDEL_ROOT, RESULTS_ROOT  # noqa: E402

ROOT = Path(CANDEL_ROOT)
DATASET = "fiducial"
GALAXIES = (
    "CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789")
GALAXY_LABELS = {
    "CGCG074-064": "CGCG 074-064",
    "NGC5765b": "NGC 5765b",
    "NGC6264": "NGC 6264",
    "NGC6323": "NGC 6323",
    "UGC3789": "UGC 3789",
}
MODELS = ("linear", "quadratic")
MODEL_LABELS = {"linear": "Linear warp", "quadratic": "Quadratic warp"}
COLORS = {"linear": "C0", "quadratic": "C3"}


def _chain_path(results_root, galaxy, model):
    suffix = "_qw" if model == "quadratic" else ""
    name = f"{galaxy}_blackjax_mcmc_rphi{suffix}_initconfig.hdf5"
    return results_root / galaxy / name


def _load_distance(path, model):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {model}-warp chain: {path}")
    with h5py.File(path, "r") as f:
        check_chain_dataset(dict(f.attrs), DATASET, str(path))
        if bool(f.attrs.get("use_ecc", False)):
            raise ValueError(f"Expected a non-eccentric chain: {path}")
        quadratic = bool(f.attrs.get("use_quadratic_warp", False))
        if quadratic != (model == "quadratic"):
            raise ValueError(f"Expected a {model}-warp chain: {path}")
        if "samples/D_A" not in f:
            raise KeyError(f"{path} has no samples/D_A dataset")
        samples = np.asarray(f["samples/D_A"], dtype=float).ravel()
    if samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError(f"Invalid D_A samples in {path}")
    return samples


def _interval(samples):
    q16, q50, q84 = np.percentile(samples, [16, 50, 84])
    return q16, q50, q84


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", default=Path(RESULTS_ROOT) / "results" / "Megamaser" / DATASET,
        help="Directory containing the fiducial per-galaxy results.")
    parser.add_argument(
        "--output",
        default=ROOT / "output" / "pdf" /
        "megamaser_distance_warps_fiducial.pdf")
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    output = Path(args.output)
    chains = {
        (model, galaxy): _load_distance(
            _chain_path(results_root, galaxy, model), model)
        for galaxy in GALAXIES for model in MODELS
    }

    print(f"{'Galaxy':14} {'Model':14} {'D_A [Mpc] (16, 50, 84%)':>31}")
    for galaxy in GALAXIES:
        for model in MODELS:
            q16, q50, q84 = _interval(chains[model, galaxy])
            print(f"{GALAXY_LABELS[galaxy]:14} {MODEL_LABELS[model]:14} "
                  f"{q50:7.2f}  -{q50 - q16:6.2f}  +{q84 - q50:6.2f}")

    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401  (registers the science style)

    with plt.style.context(["science"]):
        fig, axes = plt.subplots(1, len(GALAXIES), figsize=(11.0, 2.35))
        for i, (ax, galaxy) in enumerate(zip(axes, GALAXIES)):
            limits = [np.percentile(chains[model, galaxy], [0.1, 99.9])
                      for model in MODELS]
            lo = min(x[0] for x in limits)
            hi = max(x[1] for x in limits)
            pad = 0.04 * (hi - lo)
            grid = np.linspace(lo - pad, hi + pad, 400)

            for model in MODELS:
                samples = chains[model, galaxy]
                color = COLORS[model]
                label = MODEL_LABELS[model] if i == 0 else None
                ax.plot(grid, gaussian_kde(samples)(grid), color=color,
                        lw=1.3, label=label)
                ax.axvline(np.median(samples), color=color, lw=0.8, ls="--")

            ax.text(0.96, 0.95, GALAXY_LABELS[galaxy], transform=ax.transAxes,
                    ha="right", va="top", fontsize=8)
            ax.set_xlabel(r"$D_\mathrm{A}\ [\mathrm{Mpc}]$")
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(bottom=0.0)
            ax.set_yticks([])
        axes[0].set_ylabel("Posterior density")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=0.6)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
