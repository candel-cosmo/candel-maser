#!/usr/bin/env python3
"""Combined-H0 result figures for the MMH0 paper.

  forest     : H0 (posterior median, central 68% interval) for every combined
               run in the reconstruction x selection x warp grid, against the
               Planck, distance-ladder, and P20 references.
               -> figs/h0_variants.pdf

  corners    : full joint posteriors over the shared population parameters
               (H0, sigma_pec, the external bulk flow, and the selection
               thresholds), overlaying Manticore-Local and Carrick+2015 for
               the redshift- and distance-selection runs.
               -> figs/corner_h0_{redshift,distance}.pdf

The fiducial run is the recession-velocity (redshift) selection, the
Manticore-Local reconstruction, and a linear warp.

Run from the repo root with venv_candel:
    python papers/MMH0/plot_joint_H0.py all
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from palette import PALETTE, PLANCK_C, RED, SELECTION, SHOES_C, TEAL

from candel_maser.paths import RESULTS_ROOT  # noqa: E402

ROOT = Path(RESULTS_ROOT)
DATASET = "clipped"
RESULTS = ROOT / "results" / "Megamaser" / DATASET / "H0"
OUTDIR = "/Users/rstiskalek/Papers/MMH0/figs"

# The forest figure compares the two baseline input spot tables at fixed
# modelling variant.  The published catalogues are deliberately not shown:
# they are the historical reference and live in the H0-variants table only.
# The other figures in this module stay on DATASET, which is `clipped`
# (updated-ours) so that the population corners are shown on a baseline
# table rather than on the historical catalogues.
H0_ROOT = ROOT / "results" / "Megamaser"
DATASETS = ["fiducial", "clipped"]
DATASET_LABEL = {"fiducial": "P20",
                 "clipped": "Revised-clipped",
                 "original_published": "Original"}
# Colour already encodes the selection, so the input table is encoded by
# marker shape and fill.
DATASET_MARKER = {"fiducial": "o", "clipped": "s", "original_published": "o"}
DATASET_OPEN = {"fiducial": False, "clipped": False,
                "original_published": True}

# (value, sigma) reference H0 measurements, all symmetric.
PLANCK = (67.4, 0.5)        # Planck 2018
SHOES = (73.17, 0.86)       # SH0ES, Breuval+2024 (arXiv:2404.08038)
P20 = (73.9, 3.0)           # MCP, Pesce+2020

RECON = ["none", "Carrick2015", "ManticoreLocalCOLA"]
SEL = ["distance", "redshift"]
RECON_LABEL = {"none": "No peculiar vel.",
               "Carrick2015": "Linear (Carrick+2015)",
               "ManticoreLocalCOLA": r"\textbf{\texttt{Manticore-Local}}"}
HLABEL = r"$H_0\ [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$"
SEL_LABEL = {"none": "No selection", "distance": "Distance",
             "redshift": "Redshift"}
WARP_LABEL = {False: "Linear", True: "Quadratic"}
PAL = SELECTION


GALAXIES = ["CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789"]
GAL_LABEL = {"CGCG074-064": "CGCG 074-064", "NGC5765b": "NGC 5765b",
             "NGC6264": "NGC 6264", "NGC6323": "NGC 6323",
             "UGC3789": "UGC 3789"}


def _path(sel, recon, qw):
    stem = f"joint_H0_toy_all_{sel}_{recon}_r2" + ("_qw" if qw else "")
    return RESULTS / f"{stem}.hdf5"


def _single_path(gal):
    return RESULTS / gal / f"{gal}_blackjax_mcmc_rphi_initconfig.hdf5"


def _da_summary(s):
    q16, q50, q84 = np.percentile(s, [16, 50, 84])
    return q50, q50 - q16, q84 - q50


def _dataset_path(dataset, sel, recon):
    """Volume-prior, linear-warp combined run for one input spot table."""
    stem = f"joint_H0_toy_all_{sel}_{recon}_r2"
    return H0_ROOT / dataset / "H0" / f"{stem}.hdf5"


def load_grid():
    """Return {(recon, sel, dataset): H0 samples} for every grid file present.

    Missing files are skipped rather than raising, so the figure degrades to
    whichever input tables have been run.
    """
    out = {}
    missing = []
    for recon in RECON:
        for sel in SEL:
            for dataset in DATASETS:
                p = _dataset_path(dataset, sel, recon)
                if not p.exists():
                    missing.append(f"{dataset}/{sel}/{recon}")
                    continue
                with h5py.File(p, "r") as f:
                    out[(recon, sel, dataset)] = np.asarray(
                        f["samples/H0"], dtype=float).ravel()
    if missing:
        print(f"load_grid: {len(missing)} run(s) absent: "
              + ", ".join(missing))
    return out


def _ref_bands(ax):
    """Shade the Planck and SH0ES reference values as vertical bands."""
    for (mu, sig), c in ((PLANCK, PLANCK_C), (SHOES, SHOES_C)):
        ax.axvspan(mu - sig, mu + sig, color=c, alpha=0.18, lw=0)


def fig_forest(out):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401  (registers the "science" style)
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    grid = load_grid()

    # Row / selection / reconstruction-block vertical gaps.
    PAIR_STEP, SEL_GAP, RECON_GAP = 0.8, 1.6, 2.6
    y, med, lo, hi, col, dset = [], [], [], [], [], []
    sel_labels = []   # (text, y_centre) one per selection pair
    blocks = []       # (recon, y_centre) one per reconstruction block
    block_bounds = []  # (y_min, y_max) per block, for separator lines
    pos = 0.0
    for ri, recon in enumerate(RECON):
        if ri > 0:
            pos += RECON_GAP
        block_ys = []
        for si, sel in enumerate(SEL):
            if si > 0:
                pos += SEL_GAP
            pair_ys = []
            for dataset in DATASETS:
                if (recon, sel, dataset) not in grid:
                    continue
                q16, q50, q84 = np.percentile(grid[(recon, sel, dataset)],
                                              [16, 50, 84])
                y.append(pos)
                pair_ys.append(pos)
                block_ys.append(pos)
                med.append(q50)
                lo.append(q50 - q16)
                hi.append(q84 - q50)
                col.append(PAL[sel])
                dset.append(dataset)
                pos += PAIR_STEP
            pos -= PAIR_STEP  # SEL_GAP runs from the last row of the pair
            base = recon == "ManticoreLocalCOLA"
            stext = (r"\textbf{Redshift}" if (sel == "redshift" and base)
                     else SEL_LABEL[sel])
            sel_labels.append((stext, float(np.mean(pair_ys))))
        blocks.append((recon, float(np.mean(block_ys))))
        block_bounds.append((min(block_ys), max(block_ys)))
    y = np.asarray(y)

    # External reference values as their own rows below the maser variants.
    refs = [("Planck 2018", PLANCK, PLANCK_C),
            ("SH0ES\n(Breuval+2024)", SHOES, SHOES_C),
            ("MCP\n(Pesce+2020)", P20, "0.35")]
    ref_y0 = y.max() + 2.0
    # room for 2-line labels
    ref_y = [ref_y0 + 1.5 * i for i in range(len(refs))]

    preamble = (r"\usepackage{amsmath}\usepackage{amssymb}"
                r"\usepackage{lmodern}")
    with plt.style.context(["science", {"text.latex.preamble": preamble}]):
        fig, ax = plt.subplots(figsize=(3.5, 6.1))
        _ref_bands(ax)
        # Shade the baseline Manticore-Local reconstruction block.
        mlc_lo = 0.5 * (block_bounds[1][1] + block_bounds[2][0])
        mlc_hi = 0.5 * (y.max() + ref_y0)
        ax.axhspan(mlc_lo, mlc_hi, color="0.9", alpha=0.6, lw=0, zorder=-5)
        for i in range(len(y)):
            ax.errorbar(
                med[i], y[i], xerr=[[lo[i]], [hi[i]]],
                fmt=DATASET_MARKER[dset[i]], ms=4.5,
                color=col[i], capsize=2, lw=1.0,
                markerfacecolor=("white" if DATASET_OPEN[dset[i]]
                                 else col[i]),
                markeredgecolor=col[i])
        for (name, (mu, sig), c), yy in zip(refs, ref_y):
            ax.errorbar(mu, yy, xerr=sig, fmt="D", ms=4.5, color=c,
                        capsize=2, lw=1.0)
        ax.axhline(0.5 * (y.max() + ref_y0), color="0.7", lw=0.6)
        for i in range(len(block_bounds) - 1):
            ax.axhline(0.5 * (block_bounds[i][1] + block_bounds[i + 1][0]),
                       color="0.7", lw=0.6)

        ax.set_yticks([yc for _, yc in sel_labels] + ref_y)
        ax.set_yticklabels([t for t, _ in sel_labels]
                           + [n for n, _, _ in refs], fontsize=7)
        ax.set_ylim(ref_y[-1] + 0.8, -0.8)
        ax.yaxis.set_minor_locator(NullLocator())  # no minor ticks on y
        # reconstruction group labels (plus the reference block) outside spine
        for recon, yc in blocks:
            base = recon == "ManticoreLocalCOLA"
            ax.annotate(RECON_LABEL[recon], xy=(1.035, yc),
                        xycoords=("axes fraction", "data"),
                        rotation=-90, va="center", ha="center",
                        fontsize=8.5 if base else 7,
                        color="0.0" if base else "0.3",
                        annotation_clip=False)
        ax.annotate("Literature", xy=(1.035, float(np.mean(ref_y))),
                    xycoords=("axes fraction", "data"),
                    rotation=-90, va="center", ha="center", fontsize=7,
                    color="0.3", annotation_clip=False)
        # marker key centred above the panel: one entry per input spot table
        table_key = [
            Line2D([], [], marker=DATASET_MARKER[d], color="0.35",
                   markerfacecolor=("white" if DATASET_OPEN[d] else "0.35"),
                   markeredgecolor="0.35", linestyle="none", ms=4.5,
                   label=DATASET_LABEL[d])
            for d in DATASETS]
        ax.legend(handles=table_key, loc="lower center",
                  bbox_to_anchor=(0.5, 1.004), ncol=3, frameon=False,
                  fontsize=7, handletextpad=0.3, columnspacing=1.0)
        # header centred above the left-hand selection labels
        ax.text(-0.12, 1.008, r"\textit{Selection}" "\n" r"\textit{variants}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
        ax.set_xlabel(HLABEL)
        ax.set_xlim(59.5, 77.8)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def fig_distance(out, selection="redshift", recon="ManticoreLocalCOLA"):
    """Per-galaxy angular-diameter distance from the independent single-disc
    inference (x) against the H0-coupled joint inference (y), with the 1-1
    line.  Coupling the five discs through the shared H0 and the host
    redshifts tightens every per-galaxy distance, most strikingly for the
    weakly-constrained NGC 6264 and NGC 6323.  The joint D_A is the per-step
    comoving distance converted with the sampled H0,
    D_A = D_c / (1 + z(D_c, H0)), i.e. the deterministic stored in the
    chain."""
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401  (registers the "science" style)

    with h5py.File(_path(selection, recon, False), "r") as fj:
        joint = {g: np.asarray(fj[f"samples/{g}__D_A"], float).ravel()
                 for g in GALAXIES}
        h16, h50, h84 = np.percentile(
            np.asarray(fj["samples/H0"], float).ravel(), [16, 50, 84])
    single = {}
    for g in GALAXIES:
        with h5py.File(_single_path(g), "r") as fs:
            single[g] = np.asarray(fs["samples/D_A"], float).ravel()

    cols = PALETTE
    preamble = (r"\usepackage{amsmath}\usepackage{amssymb}"
                r"\usepackage{lmodern}")
    with plt.style.context(["science", {"text.latex.preamble": preamble}]):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        lo, hi = np.inf, -np.inf
        for i, g in enumerate(GALAXIES):
            xs, xl, xh = _da_summary(single[g])
            ys, yl, yh = _da_summary(joint[g])
            ax.errorbar(xs, ys, xerr=[[xl], [xh]], yerr=[[yl], [yh]],
                        fmt="o", ms=4, color=cols[i], capsize=2, lw=1.0,
                        label=GAL_LABEL[g])
            lo = min(lo, xs - xl, ys - yl)
            hi = max(hi, xs + xh, ys + yh)
        pad = 0.06 * (hi - lo)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, ls="--", color="0.5", lw=0.8, zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$D_\mathrm{A}$, independent disc $[\mathrm{Mpc}]$")
        ax.set_ylabel(r"$D_\mathrm{A}$, $H_0$-coupled $[\mathrm{Mpc}]$")
        ax.text(0.96, 0.05,
                rf"$H_0 = {h50:.1f}^{{+{h84 - h50:.1f}}}_{{-{h50 - h16:.1f}}}$"
                "\n" r"$\mathrm{km\,s^{-1}\,Mpc^{-1}}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left",
                  handletextpad=0.3, labelspacing=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def fig_corner(selection, out):
    """Overlay the Manticore-Local and Carrick+2015 joint posteriors for one
    selection, over the shared population parameters (H0, sigma_pec, the
    external bulk flow, and the selection thresholds).  The bulk flow is shown
    as its magnitude plus its direction in Galactic coordinates (ell, b)."""
    import scienceplots  # noqa: F401

    from candel.plotting.corner import plot_corner_getdist
    from candel.util import radec_cartesian_to_galactic

    sel_keys = {"redshift": ["cz_lim_selection", "cz_lim_selection_width"],
                "distance": ["D_lim", "D_width"]}[selection]
    raw_keys = ["H0", "sigma_pec"] + sel_keys
    keys = ["H0", "sigma_pec", "Vext_mag", "Vext_ell", "Vext_b"] + sel_keys

    recons = [("ManticoreLocalCOLA", r"\texttt{Manticore-Local}"),
              ("Carrick2015", "Linear (Carrick+2015)")]
    samples_list, labels = [], []
    for recon, lab in recons:
        with h5py.File(_path(selection, recon, False), "r") as f:
            grp = f["samples"]
            s = {k: np.asarray(grp[k], float).ravel() for k in raw_keys}
            # Vext is sampled in the ICRS-Cartesian frame: read the 3-vector
            # directly (legacy chains store it as mag + phi/cos_theta instead).
            if "Vext" in grp:
                V = np.asarray(grp["Vext"], float).reshape(-1, 3)
                vx, vy, vz = V[:, 0], V[:, 1], V[:, 2]
            else:
                st = np.sqrt(1.0 - np.asarray(grp["Vext_cos_theta"],
                                              float).ravel() ** 2)
                mag = np.asarray(grp["Vext_mag"], float).ravel()
                phi = np.asarray(grp["Vext_phi"], float).ravel()
                vx = mag * st * np.cos(phi)
                vy = mag * st * np.sin(phi)
                vz = mag * np.asarray(grp["Vext_cos_theta"], float).ravel()
        s["Vext_mag"] = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        # Convert to Galactic (ell, b) to report the bulk-flow direction.
        _, s["Vext_ell"], s["Vext_b"] = radec_cartesian_to_galactic(vx, vy, vz)
        samples_list.append({k: s[k] for k in keys})
        labels.append(lab)

    # Hard boundaries so the getdist KDE is boundary-corrected rather than
    # leaking past them: sigma_pec is non-negative, Vext_mag is bounded by
    # mag_range, and (ell, b) by ell_range/b_range.  The selection threshold
    # and its transition width follow the uniform priors of the joint H0
    # model.
    sel_range = {"cz_lim_selection": [500.0, 20000.0],
                 "cz_lim_selection_width": [50.0, 10000.0],
                 "D_lim": [15.0, 1000.0],
                 "D_width": [15.0, 500.0]}
    ranges = {"sigma_pec": [0, None]}
    ranges.update({k: sel_range[k] for k in sel_keys})

    # The distance-threshold priors extend far past the posterior (99th
    # percentile 182 and 121 Mpc), so clip the plotted axes without touching
    # the KDE boundaries above.
    param_limits = {"D_lim": [15.0, 250.0], "D_width": [15.0, 150.0]}
    param_limits = {k: v for k, v in param_limits.items() if k in sel_keys}

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_corner_getdist(
        samples_list, labels=labels, keys=keys, cols=[TEAL, RED],
        filled=True, show_fig=False, filename=out, fontsize=17,
        legend_fontsize=24, mag_range=[0.0, 300.0], ell_range=[0.0, 360.0],
        b_range=[-90.0, 90.0], ranges=ranges, param_limits=param_limits)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["forest", "corners", "distance", "all"],
                    nargs="?", default="all")
    args = ap.parse_args()
    if args.which in ("forest", "all"):
        fig_forest(os.path.join(OUTDIR, "h0_variants.pdf"))
    if args.which in ("distance", "all"):
        fig_distance(os.path.join(OUTDIR, "distance_single_vs_joint.pdf"))
    if args.which in ("corners", "all"):
        fig_corner("redshift", os.path.join(OUTDIR, "corner_h0_redshift.pdf"))
        fig_corner("distance", os.path.join(OUTDIR, "corner_h0_distance.pdf"))
