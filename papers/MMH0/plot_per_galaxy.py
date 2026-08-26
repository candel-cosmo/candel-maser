#!/usr/bin/env python3
"""Per-galaxy result figures for the MMH0 paper.

  distance_redshift : inferred comoving distance D_c (posterior median and 68%
                      interval) against the observed CMB-frame redshift
                      z_obs = v_sys_obs / c, one point per galaxy, with flat
                      LCDM D_C(z, H0) reference lines.  Uses v_sys_obs from
                      config_maser.toml -- the CMB-frame velocity the H0
                      inference actually fits (run_joint_H0) -- not a
                      re-applied CMB conversion.  On the `clipped`
                      (updated-ours) table, as the rest of the paper's
                      figures.  -> figs/distance_redshift.pdf

  corner            : a corner of the NGC 5765b global posterior on the
                      `clipped` (updated-ours) table, an example per-galaxy
                      disc fit for the appendix.  That table is used rather
                      than `fiducial` because it is the one carrying the
                      clump-2 error floors.  Uses the standard candel
                      plot_corner under the science style.
                      -> figs/corner_NGC5765b.pdf

Run from the repo root with venv_candel:
    python notebooks/paper_MMH0/plot_per_galaxy.py both
"""
import argparse
import os
import tomllib
from pathlib import Path

import h5py
import numpy as np
from palette import PALETTE

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "scripts" / "megamaser" / "config_maser.toml"
DATASET = "clipped"
RESULTS = ROOT / "results" / "Megamaser" / DATASET
OUTDIR = "/Users/rstiskalek/Papers/MMH0/figs"
C_KMS = 299792.458

GALAXIES = ["CGCG074-064", "NGC5765b", "UGC3789", "NGC6264", "NGC6323"]
GAL_LABEL = {
    "CGCG074-064": "CGCG 074-064", "NGC5765b": "NGC 5765b",
    "UGC3789": "UGC 3789", "NGC6264": "NGC 6264", "NGC6323": "NGC 6323",
}


def chain_path(galaxy, dataset=DATASET, variant=""):
    return (ROOT / "results" / "Megamaser" / dataset / galaxy
            / f"{galaxy}_blackjax_mcmc_rphi{variant}_initconfig.hdf5")


def fig_distance_redshift(out):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    from astropy.cosmology import FlatLambdaCDM

    cfg = tomllib.load(open(CONFIG, "rb"))["model"]["galaxies"]
    z, dmed, dlo, dhi = [], [], [], []
    for g in GALAXIES:
        with h5py.File(chain_path(g), "r") as f:
            s = f["samples"]
            if "D_c" in s:
                dc = np.asarray(s["D_c"], dtype=float).ravel()
            else:
                # uniform_D_A chains store D_A; convert to comoving for the
                # D_c-vs-z panel using the galaxy's observed redshift.
                da = np.asarray(s["D_A"], dtype=float).ravel()
                dc = da * (1.0 + float(cfg[g]["v_sys_obs"]) / C_KMS)
        q16, q50, q84 = np.percentile(dc, [16, 50, 84])
        z.append(float(cfg[g]["v_sys_obs"]) / C_KMS)
        dmed.append(q50)
        dlo.append(q50 - q16)
        dhi.append(q84 - q50)
    z = np.asarray(z)
    zerr = 250.0 / C_KMS  # 250 km/s redshift (peculiar-velocity) uncertainty

    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        zline = np.linspace(0.0, 1.12 * z.max(), 200)
        for H0, ls in ((73.2, "--"), (67.4, ":")):
            d = FlatLambdaCDM(H0=H0, Om0=0.315).comoving_distance(zline).value
            ax.plot(d, zline, ls=ls, color="0.5", lw=0.9,
                    label=rf"$H_0={H0:.1f}$")
        PAL = PALETTE
        for i, g in enumerate(GALAXIES):
            ax.errorbar(dmed[i], z[i], xerr=[[dlo[i]], [dhi[i]]], yerr=zerr,
                        fmt="o", ms=4, color=PAL[i % len(PAL)], capsize=2)
            below = g == "NGC6323"  # label collides with the H0 lines above
            ax.annotate(GAL_LABEL[g], (dmed[i], z[i]),
                        textcoords="offset points",
                        xytext=(0, -4 if below else 4),
                        ha="center", va="top" if below else "bottom",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
        ax.set_xlabel(
            r"$\mathrm{Comoving\ distance,}\ D_\mathrm{C}\ [\mathrm{Mpc}]$")
        ax.set_ylabel(r"$\mathrm{Observed\ redshift,}\ z_\mathrm{obs}$")
        ax.set_xlim(0.0, 1.15 * max(m + h for m, h in zip(dmed, dhi)))
        ax.set_ylim(0.0, 1.12 * z.max())
        ax.legend(loc="lower right")
        fig.tight_layout()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"wrote {out}")
    for g, zi, dm, dl, dh in zip(GALAXIES, z, dmed, dlo, dhi):
        print(f"  {GAL_LABEL[g]:14} z={zi:.5f}  D_c={dm:.1f} "
              f"-{dl:.1f}/+{dh:.1f} Mpc")


def fig_corner(out, galaxy="NGC5765b", dataset="clipped", variant=""):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    from candel.plotting.corner import plot_corner

    # The clump-2 floors exist only on the tables the MCP cut is not applied
    # to, so the corner comes from `clipped` (updated-ours), not `fiducial`.
    path = chain_path(galaxy, dataset, variant)
    with h5py.File(path, "r") as f:
        theta = [s for s in
                 str(dict(f.attrs).get("theta_sites", "")).split(",")
                 if s]
        # chains are stored (n_chains, n_samples); flatten to 1D.
        samples = {k: np.asarray(v, dtype=float).ravel()
                   for k, v in f["samples"].items()}
    # Keep eta (the sampled reparametrisation) alongside D_A and the physical
    # mass log_MBH: D_A-log_MBH is strongly correlated, D_A-eta is not.
    keys = list(theta)
    if "log_MBH" in samples and "log_MBH" not in keys:
        keys.insert(keys.index("eta") + 1, "log_MBH")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with plt.style.context(["science", {"axes.labelsize": 26,
                                        "axes.titlesize": 14}]):
        plot_corner({k: samples[k] for k in keys}, keys=keys,
                    show_fig=False, filename=out, smooth=None)
    print(f"wrote {out}  ({galaxy}, {dataset}, {len(keys)} params)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fig", choices=("distance_redshift", "corner",
                                    "corner_ngc4258_ecc", "both"),
                    default="both", nargs="?")
    ap.add_argument("--out-dir", default=OUTDIR)
    args = ap.parse_args(argv)

    if args.fig in ("distance_redshift", "both"):
        fig_distance_redshift(os.path.join(args.out_dir,
                                           "distance_redshift.pdf"))
    if args.fig in ("corner", "both"):
        fig_corner(os.path.join(args.out_dir, "corner_NGC5765b.pdf"))
    if args.fig == "corner_ngc4258_ecc":
        # NGC 4258 has one spot table, so `fiducial` is also the published
        # one; the eccentric variant adds e_x, e_y, and dperiapsis_dr.
        fig_corner(os.path.join(args.out_dir, "corner_NGC4258_ecc.pdf"),
                   galaxy="NGC4258", dataset="fiducial", variant="_ecc_qw")


if __name__ == "__main__":
    main()
