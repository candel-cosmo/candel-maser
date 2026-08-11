#!/usr/bin/env python3
"""Compare four linear-warp CANDEL datasets with Dom's archived P20 posteriors.

Reads the config-initialised, non-eccentric, non-quadratic MCMC chain for each
of the five MCP H0 galaxies and writes one five-panel PDF.

Run from the repository root with::

    venv_candel/bin/python scripts/megamaser/plot_dataset_distances.py
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import gaussian_kde

from maser_config import check_chain_dataset


ROOT = Path(__file__).resolve().parents[2]
GALAXIES = (
    "CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789")
GALAXY_LABELS = {
    "CGCG074-064": "CGCG 074-064",
    "NGC5765b": "NGC 5765b",
    "NGC6264": "NGC 6264",
    "NGC6323": "NGC 6323",
    "UGC3789": "UGC 3789",
}
DATASETS = ("original_published", "fiducial", "unpruned", "clipped")
DATASET_LABELS = {
    "original_published": "Original published",
    "fiducial": "Fiducial",
    "unpruned": "Unpruned",
    "clipped": "Clipped",
}
COLORS = {
    "original_published": "#0077BB",
    "fiducial": "#EE7733",
    "unpruned": "#00A650",
    "clipped": "#EE3377",
}
DOM_ROOT = ROOT / "data" / "Megamaser" / "external" / "Dom_data"
DOM_LABEL = "P20 (Dom)"
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
        "--results-root", default=ROOT / "results" / "Megamaser",
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

    with plt.style.context(["science"]):
        fig, axes = plt.subplots(1, len(GALAXIES), figsize=(11.0, 2.35))
        for i, (ax, galaxy) in enumerate(zip(axes, GALAXIES)):
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
                        lw=1.3, ls=":" if dataset == "fiducial" else "-",
                        label=label)
                ax.axvline(np.median(samples), color=color, lw=0.8, ls="--")

            samples = dom_chains[galaxy]
            ax.plot(grid, gaussian_kde(samples)(grid), color=DOM_COLOR,
                    lw=1.8, ls="--", label=DOM_LABEL if i == 0 else None,
                    zorder=5)
            ax.axvline(np.median(samples), color=DOM_COLOR, lw=1.0,
                       ls="--", zorder=5)

            ax.text(0.96, 0.95, GALAXY_LABELS[galaxy], transform=ax.transAxes,
                    ha="right", va="top", fontsize=8)
            ax.set_xlabel(r"$D_\mathrm{A}\ [\mathrm{Mpc}]$")
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(bottom=0.0)
            ax.set_yticks([])
        axes[0].set_ylabel("Posterior density")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5,
                   frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=0.6)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
