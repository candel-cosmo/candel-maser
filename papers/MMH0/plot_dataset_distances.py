#!/usr/bin/env python3
"""Compare four linear-warp CANDEL datasets with Dom's archived P20 posteriors.

Reads the config-initialised, non-eccentric, non-quadratic MCMC chain for each
of the five MCP H0 galaxies and writes one five-panel PDF.

Run from the repository root with::

    venv_candel/bin/python papers/MMH0/plot_dataset_distances.py
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import gaussian_kde

from candel_maser.maser_config import check_chain_dataset


from candel_maser.paths import CANDEL_ROOT, DATA_ROOT, RESULTS_ROOT  # noqa: E402

ROOT = Path(CANDEL_ROOT)
from palette import BROWN, GOLD, RED, TEAL  # noqa: E402
GALAXIES = (
    "CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789")
GALAXY_LABELS = {
    "CGCG074-064": "CGCG 074-064",
    "NGC5765b": "NGC 5765b",
    "NGC6264": "NGC 6264",
    "NGC6323": "NGC 6323",
    "UGC3789": "UGC 3789",
}
DATASETS = ("fiducial", "original_published", "unpruned", "clipped")
DATASET_LABELS = {
    "original_published": "Original",
    "fiducial": "P20 table",
    "unpruned": "Revised",
    "clipped": "Revised-clipped",
}
COLORS = {
    "original_published": TEAL,
    "fiducial": RED,
    "unpruned": BROWN,
    "clipped": GOLD,
}
# Thinner lines for the two secondary tables.
LINEWIDTHS = {
    "original_published": 2.0,
    "fiducial": 2.0,
    "unpruned": 1.4,
    "clipped": 1.4,
}
DOM_ROOT = Path(DATA_ROOT) / "data" / "Megamaser" / "external" / "Dom_data"
DOM_LABEL = "P20 posterior"
DOM_COLOR = "#111111"


def _chain_path(results_root, dataset, galaxy):
    name = f"{galaxy}_blackjax_mcmc_rphi_initconfig.hdf5"
    return results_root / dataset / galaxy / name


def _load_distance(path, dataset):
    if not path.is_file():
        raise FileNotFoundError(f"Missing linear-warp chain: {path}")
    with h5py.File(path, "r") as f:
        check_chain_dataset(dict(f.attrs), dataset, str(path))
        if bool(f.attrs.get("use_ecc", False)):
            raise ValueError(f"Expected a non-eccentric chain: {path}")
        if bool(f.attrs.get("use_quadratic_warp", False)):
            raise ValueError(f"Expected a linear-warp chain: {path}")
        if "samples/D_A" not in f:
            raise KeyError(f"{path} has no samples/D_A dataset")
        samples = np.asarray(f["samples/D_A"], dtype=float).ravel()
    if samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError(f"Invalid D_A samples in {path}")
    return samples


def _load_dom_distance(galaxy):
    path = DOM_ROOT / f"D_archivedP20_{galaxy}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Dom posterior: {path}")
    samples = np.asarray(np.loadtxt(path), dtype=float).ravel()
    if samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError(f"Invalid D_A samples in {path}")
    return samples


def _interval(samples):
    q16, q50, q84 = np.percentile(samples, [16, 50, 84])
    return q16, q50, q84


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", default=Path(RESULTS_ROOT) / "results" / "Megamaser",
        help="Directory containing the four dataset result namespaces.")
    parser.add_argument(
        "--output",
        default=ROOT / "output" / "pdf" /
        "megamaser_distance_datasets_linear.pdf")
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    output = Path(args.output)
    chains = {
        (dataset, galaxy): _load_distance(
            _chain_path(results_root, dataset, galaxy), dataset)
        for galaxy in GALAXIES for dataset in DATASETS
    }
    dom_chains = {galaxy: _load_dom_distance(galaxy)
                  for galaxy in GALAXIES}

    print(f"{'Galaxy':14} {'Dataset':19} {'D_A [Mpc] (16, 50, 84%)':>31}")
    for galaxy in GALAXIES:
        for dataset in DATASETS:
            q16, q50, q84 = _interval(chains[dataset, galaxy])
            print(f"{GALAXY_LABELS[galaxy]:14} {dataset:19} "
                  f"{q50:7.2f}  -{q50 - q16:6.2f}  +{q84 - q50:6.2f}")
        q16, q50, q84 = _interval(dom_chains[galaxy])
        print(f"{GALAXY_LABELS[galaxy]:14} {DOM_LABEL:19} "
              f"{q50:7.2f}  -{q50 - q16:6.2f}  +{q84 - q50:6.2f}")

    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401  (registers the science style)

    # ponytail: 2x3 grid, not 3x3 -- 5 galaxies + legend fill it exactly.
    with plt.style.context(["science"]):
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6))
        legend_ax = axes[1, 0]
        legend_ax.axis("off")
        panels = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]
        for i, (ax, galaxy) in enumerate(zip(panels, GALAXIES)):
            limits = [np.percentile(chains[dataset, galaxy], [0.1, 99.9])
                      for dataset in DATASETS]
            limits.append(np.percentile(dom_chains[galaxy], [0.1, 99.9]))
            lo = min(x[0] for x in limits)
            hi = max(x[1] for x in limits)
            pad = 0.04 * (hi - lo)
            grid = np.linspace(lo - pad, hi + pad, 400)

            for dataset in DATASETS:
                samples = chains[dataset, galaxy]
                color = COLORS[dataset]
                label = DATASET_LABELS[dataset] if i == 0 else None
                ax.plot(grid, gaussian_kde(samples)(grid), color=color,
                        lw=LINEWIDTHS[dataset], label=label)

            samples = dom_chains[galaxy]
            ax.plot(grid, gaussian_kde(samples)(grid), color=DOM_COLOR,
                    lw=2.2, ls="--", label=DOM_LABEL if i == 0 else None,
                    zorder=5)

            ax.text(0.96, 0.95, GALAXY_LABELS[galaxy], transform=ax.transAxes,
                    ha="right", va="top", fontsize=14)
            ax.set_xlabel(r"$D_\mathrm{A}\ [\mathrm{Mpc}]$")
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(bottom=0.0)
            ax.set_yticks([])
        axes[0, 0].set_ylabel("Posterior density")
        axes[1, 1].set_ylabel("Posterior density")

        handles, labels = panels[0].get_legend_handles_labels()
        leg = legend_ax.legend(handles, labels, loc="center", frameon=False,
                               fontsize=15, handlelength=2.0,
                               labelspacing=1.0)
        for line in leg.get_lines():
            line.set_linewidth(1.5 * line.get_linewidth())
        fig.tight_layout(w_pad=0.8, h_pad=1.2)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=600, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
