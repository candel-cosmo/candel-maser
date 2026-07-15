#!/usr/bin/env python3
"""Overlay a Reid fit_disk global chain against a CANDEL posterior.

Both samplers fit the *same* data with flat priors on the Reid globals, so on
the shared globals the posteriors should agree if the two samplers are
consistent.  CANDEL samples in its own convention (D_A, i0, Omega0, ...); this
maps each CANDEL draw into Reid's 20-global convention (the inverse of
``make_candel_globals``) including the data-derived warp-pivot shift, then
writes a side-by-side median/68% table (with the Reid chain's split-Rhat, since
the single Reid chain has no native between-chain diagnostic) and an overlay
corner plot over all shared globals.

Caveat: CANDEL puts a flat prior on D_A while Reid is flat in H0; over a
narrow, data-dominated posterior the induced Jacobian is negligible, but it
is not exactly identical priors.
"""
from __future__ import annotations

import argparse
import contextlib
import logging
from pathlib import Path

import corner
import h5py
import matplotlib
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D
from numpyro.diagnostics import split_gelman_rubin

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from run_reid_mcmc import (DEFAULT_CONFIG, PARAM_LABELS, ROOT,  # noqa: E402
                           compute_reid_r_ref, load_config_init,
                           load_galaxy_config, parse_data_rows, reid_H0)

# Reid-convention name -> LaTeX label, for the params both samplers constrain.
SHARED = [
    "H0", "D_Mpc", "Mbh_1e7Msun", "Vsys_km_s", "x0_mas", "y0_mas",
    "i0_deg", "di_dr_deg_mas", "PA_deg", "dPA_dr_deg_mas",
    "sigma_x_mas", "sigma_y_mas", "sigma_vsys_km_s", "sigma_vhv_km_s",
    "sigma_acc_km_s_yr",
]
MAX_PLOT = 20000
CORNER_BINS = 20  # shared 2D-panel bin count; 1D marginals scale per-dataset

# Fixed corner-plot axis windows; params not listed use the data range.
PLOT_RANGES = {"H0": (5.0, 220.0)}


def _robust_range(x, pct=0.5):
    """Percentile-based (not min/max) range: robust to the rare extreme
    outlier draws a wide-open Ho window can produce (e.g. a near-zero-H0
    excursion blowing up D_Mpc/Mbh_1e7Msun), which otherwise stretch the
    axis until the real posterior mass collapses into a sliver."""
    lo, hi = np.nanpercentile(x, [pct, 100.0 - pct])
    return float(lo), float(hi)


def _pad_degenerate(lo, hi):
    """Nudge a zero-width (lo == hi) range so downstream binning doesn't
    divide by zero (e.g. a frozen/constant global that has the same value
    in every overlaid dataset, so even the pooled range collapses)."""
    if lo != hi:
        return lo, hi
    pad = 0.5 if lo == 0.0 else 0.05 * abs(lo)
    return lo - pad, hi + pad


def _plot_ranges(both):
    rng = [_robust_range(both[:, j]) for j in range(both.shape[1])]
    for k, window in PLOT_RANGES.items():
        rng[SHARED.index(k)] = window
    return [_pad_degenerate(lo, hi) for lo, hi in rng]


def _auto_hist_bin_factor(x):
    """This dataset's own numpy 'auto' 1D bin count (Freedman-Diaconis /
    Sturges over its own finite values), as a factor relative to
    CORNER_BINS -- so each overlaid dataset's 1D marginal gets its own
    resolution via corner's hist_bin_factor instead of every dataset being
    forced onto one shared bin grid. The 2D panels keep CORNER_BINS
    (hist_bin_factor only scales the 1D histograms)."""
    x = x[np.isfinite(x)]
    n_bins = len(np.histogram_bin_edges(x, bins="auto")) - 1
    return n_bins / CORNER_BINS


def candel_to_reid(h5_path, galaxy, config, variant, data):
    """Map CANDEL posterior draws into Reid's global convention."""
    gcfg = load_galaxy_config(config, galaxy)
    v_sys_obs = float(gcfg["v_sys_obs"])
    icfg = dict(gcfg[variant])
    r_ref_i = float(gcfg.get("r_ang_ref_i", icfg.get("r_ang_ref", 0.0)))
    r_ref_pa = float(gcfg.get("r_ang_ref_Omega", icfg.get("r_ang_ref", 0.0)))

    init = load_config_init(config, galaxy, 0.0, variant=variant).values
    header, rows = parse_data_rows(data)
    reid_r_ref = compute_reid_r_ref(rows, header, init)
    dr_i = reid_r_ref - r_ref_i
    dr_pa = reid_r_ref - r_ref_pa

    with h5py.File(h5_path, "r") as f:
        s = {k: f["samples"][k][:].reshape(-1).astype(np.float64)
             for k in f["samples"].keys()}

    if "D_A" in s:
        D = s["D_A"]
    elif "D_c" in s:
        # Legacy comoving-only chain: convert to angular-diameter distance,
        # D_A = D_c/(1+z). z ~ vsys/c is the low-z form (<0.1% at MCP
        # redshifts) of CANDEL's own D_A = D_c/(1+z_cosmo), and matches the
        # z=vsys/c convention of the reid_H0 map applied below -- unlike the
        # old code, which used D_c directly as D_A (~(1+z) too large).
        D = s["D_c"] / (1.0 + (v_sys_obs + s["dv_sys"]) / 299792.458)
    else:
        raise KeyError(
            f"{h5_path} has no D_A or D_c distance samples; candel_to_reid "
            f"needs the CANDEL distance site")
    if "log_MBH" not in s:
        raise KeyError(
            f"{h5_path} has no log_MBH mass samples; candel_to_reid needs the "
            f"log_MBH parametrisation")
    vsys = v_sys_obs + s["dv_sys"]
    d2i = -s["d2i_dr2"] if "d2i_dr2" in s else 0.0
    d2pa = s["d2Omega_dr2"] if "d2Omega_dr2" in s else 0.0
    out = {
        # cosmological H0 via dampc's inverse (same mapping Reid samples in),
        # not the naive vsys/D_A -- so the H0 overlay is like-for-like.
        "H0": reid_H0(vsys, D),
        "D_Mpc": D,
        "Mbh_1e7Msun": 10.0 ** (s["log_MBH"] - 7.0),
        "Vsys_km_s": vsys,
        "x0_mas": s["x0"] / 1000.0,
        "y0_mas": s["y0"] / 1000.0,
        "i0_deg": (180.0 - s["i0"]) - s["di_dr"] * dr_i + d2i * dr_i**2,
        "di_dr_deg_mas": -s["di_dr"],
        "PA_deg": s["Omega0"] + s["dOmega_dr"] * dr_pa + d2pa * dr_pa**2,
        "dPA_dr_deg_mas": s["dOmega_dr"],
        "sigma_x_mas": s["sigma_x_floor"] / 1000.0,
        "sigma_y_mas": s["sigma_y_floor"] / 1000.0,
        "sigma_vsys_km_s": s["sigma_v_sys"],
        "sigma_vhv_km_s": s["sigma_v_hv"],
        "sigma_acc_km_s_yr": s["sigma_a_floor"],
    }
    return out


def load_reid(chain_csv, burn):
    data = np.atleast_1d(np.genfromtxt(chain_csv, delimiter=",", names=True))
    names = data.dtype.names
    n = len(data)
    trimmed = data[int(burn * n):]
    finite = np.all([np.isfinite(trimmed[name]) for name in names], axis=0)
    return {name: trimmed[name][finite] for name in names}


def med68(x):
    return np.nanmedian(x), np.nanpercentile(x, 16), np.nanpercentile(x, 84)


def reid_split_rhat(reid):
    """Split-Rhat of each shared global from the single Reid chain.

    Reid runs one walker, so there is no native between-chain diagnostic; split
    the post-burn-in chain into halves and apply the standard Gelman-Rubin
    statistic. Frozen (constant) columns have no within-chain variance and map
    to NaN.
    """
    out = {}
    for k in SHARED:
        x = np.asarray(reid[k], dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 4 or np.ptp(x) == 0.0:
            out[k] = np.nan
        else:
            out[k] = float(split_gelman_rubin(x[None, :]))
    return out


def write_table(candel, reid, rhat, path):
    hdr = (f"{'param':>18s} {'CANDEL (med +84/-16)':>30s} "
           f"{'Reid (med +84/-16)':>30s} {'Reid Rhat':>9s}")
    lines = [hdr, "-" * len(hdr)]
    for k in SHARED:
        cm, clo, chi = med68(candel[k])
        rm, rlo, rhi = med68(reid[k])
        lines.append(
            f"{k:>18s} "
            + f"{cm:12.5g} +{chi-cm:.3g}/-{cm-clo:.3g}".ljust(31)
            + f"{rm:12.5g} +{rhi-rm:.3g}/-{rm-rlo:.3g}".ljust(31)
            + f"{rhat[k]:9.3f}"
        )
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)


def subsample(x):
    if len(x) <= MAX_PLOT:
        return x
    return x[np.linspace(0, len(x) - 1, MAX_PLOT, dtype=int)]


@contextlib.contextmanager
def _quiet_corner_contour_warnings():
    """corner.hist2d logs 'Too few points to create valid contours' on the
    root logger once per panel with a pinned/zero-variance global (several
    Reid globals are frozen at 0); with 20+ globals x several overlaid
    datasets that can fire hundreds of times per plot. Benign -- silence
    it for just the plotting call."""
    logger = logging.getLogger()
    prev_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(prev_level)


def overlay_three(datasets, path):
    """Multi-dataset overlay corner over the SHARED globals.

    datasets: list of (label, color, {param: samples}[, weights]).
    Used by run_gibbs_comparison.py for the CANDEL vs fit_disk-variant
    overlays (formerly in compare_reflection_corner.py).
    """
    datasets = [(d[0], d[1], d[2], d[3] if len(d) > 3 else None)
                for d in datasets]
    cols = [np.column_stack([subsample(d[k]) for k in SHARED])
            for _, _, d, _ in datasets]
    # subsample() thins deterministically, so weights stay row-aligned
    wts = [None if w is None else subsample(np.asarray(w, dtype=np.float64))
           for _, _, _, w in datasets]
    both = np.vstack(cols)
    rng = _plot_ranges(both)
    labels = [PARAM_LABELS.get(k, k) for k in SHARED]
    # 1- and 2-sigma enclosed-mass levels: two contours per 2D panel.
    levels = (1.0 - np.exp(-0.5), 1.0 - np.exp(-2.0))

    fig = None
    with plt.style.context("science"):
        with _quiet_corner_contour_warnings():
            for (label, color, _, _), c, w in zip(datasets, cols, wts):
                hist_bin_factor = [_auto_hist_bin_factor(c[:, j])
                                   for j in range(c.shape[1])]
                fig = corner.corner(
                    c, fig=fig, bins=CORNER_BINS, labels=labels, range=rng,
                    color=color, weights=w, levels=levels,
                    plot_datapoints=False, hist_bin_factor=hist_bin_factor,
                    hist_kwargs={"density": True})
        handles = [Line2D([0], [0], color=color, lw=2.5)
                   for _, color, _, _ in datasets]
        legend_labels = [label for label, _, _, _ in datasets]
        fig.legend(handles, legend_labels, loc="upper right",
                   bbox_to_anchor=(0.92, 0.92), fontsize=16, frameon=True,
                   framealpha=0.9, borderpad=0.8, handlelength=1.5)
        for ax in fig.axes:
            ax.tick_params(labelsize=6)
        fig.savefig(path, dpi=450)
    plt.close(fig)


def overlay_distance_histogram(datasets, path, param="D_Mpc", legend_ncol=2):
    """Overlay one-dimensional histograms for the inferred distance."""
    datasets = [(d[0], d[1], d[2], d[3] if len(d) > 3 else None)
                for d in datasets]
    prepared = []
    for label, color, data, weights in datasets:
        x = np.asarray(data[param], dtype=np.float64)
        finite = np.isfinite(x)
        w = None
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            finite &= np.isfinite(w) & (w > 0.0)
        x = x[finite]
        if w is not None:
            w = w[finite]
        if len(x) == 0:
            continue
        prepared.append((label, color, x, w))

    if not prepared:
        raise ValueError(f"No finite {param} samples available for histogram")

    all_x = np.concatenate([x for _, _, x, _ in prepared])
    xmin, xmax = _pad_degenerate(*_robust_range(all_x))
    # bins="auto" is unsupported by np.histogram for weighted data (e.g.
    # --reweight-da2's D_A^2 weights on eta-off datasets); compute shared
    # edges once from the pooled unweighted samples instead, which also
    # gives every overlaid dataset the same bins for a fair comparison.
    bin_edges = np.histogram_bin_edges(all_x, bins="auto", range=(xmin, xmax))

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
        for label, color, x, w in prepared:
            ax.hist(x, bins=bin_edges, weights=w, density=True,
                    histtype="step", lw=1.1, color=color, label=label)
        ax.set_xlabel(PARAM_LABELS.get(param, param))
        ax.set_ylabel({"D_Mpc": r"$p(D)$", "H0": r"$p(H_0)$"}.get(param,
                                                                  r"$p(x)$"))
        ax.set_xlim(xmin, xmax)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=legend_ncol, fontsize=7, frameon=True, framealpha=0.9)
        fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def per_chain_distance_histogram(chain_arrs, path, param="D_Mpc"):
    """Overlay one 1D distance histogram per chain (not pooled) for a
    single Gibbs variant, so inter-chain agreement/mixing can be checked
    by eye instead of only via the pooled R-hat number."""
    n = len(chain_arrs)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, n))
    datasets = [(f"chain {i}", colors[i], arr)
                for i, arr in enumerate(chain_arrs)]
    overlay_distance_histogram(datasets, path, param=param,
                               legend_ncol=min(n, 6))


def overlay_corner(candel, reid, path):
    c = np.column_stack([subsample(candel[k]) for k in SHARED])
    r = np.column_stack([subsample(reid[k]) for k in SHARED])
    both = np.vstack([c, r])
    rng = _plot_ranges(both)
    labels = [PARAM_LABELS.get(k, k) for k in SHARED]

    with _quiet_corner_contour_warnings():
        fig = corner.corner(c, labels=labels, range=rng, color="C0",
                            plot_datapoints=False,
                            hist_kwargs={"density": True})
        corner.corner(r, fig=fig, range=rng, color="C3",
                      plot_datapoints=False, hist_kwargs={"density": True})
    fig.text(0.97, 0.97, "blue: CANDEL", color="C0", ha="right", fontsize=12)
    fig.text(0.97, 0.94, "red: Reid", color="C3", ha="right", fontsize=12)
    for ax in fig.axes:
        ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Overlay corner: {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--galaxy", required=True)
    p.add_argument("--variant", default="init", choices=["init", "init_qw"])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--reid-chain", type=Path, default=None)
    p.add_argument("--candel", type=Path, default=None)
    p.add_argument("--burn", type=float, default=0.0,
                   help="Fraction of the Reid chain to discard as burn-in.")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    g = args.galaxy
    suffix = "_qw" if args.variant == "init_qw" else ""
    data = args.data or ROOT / f"data/Megamaser/{g}_loader_reid.inp"
    reid_chain = args.reid_chain or (
        ROOT / f"results/Megamaser/reid_mcmc/{g}_try/global_chain.csv")
    candel = args.candel or (
        ROOT / f"results/Megamaser/{g}/{g}_blackjax_mcmc_rphi{suffix}_initreid.hdf5")  # noqa: E501
    out_dir = args.out_dir or reid_chain.parent

    for pth in (data, reid_chain, candel):
        if not Path(pth).exists():
            p.error(f"missing input: {pth}")
    out_dir.mkdir(parents=True, exist_ok=True)

    reid = load_reid(reid_chain, args.burn)
    cand = candel_to_reid(candel, g, args.config, args.variant, data)
    rhat = reid_split_rhat(reid)
    write_table(cand, reid, rhat, out_dir / f"compare_{g}{suffix}.txt")
    overlay_corner(cand, reid, out_dir / f"compare_{g}{suffix}_corner.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
