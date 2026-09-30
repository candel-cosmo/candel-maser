"""Per-chain distance-posterior grid: native Reid ``fit_disk`` vs. our sampler.

Appendix figure for the megamaser H0 paper.  It shows, for each galaxy, that
independent chains of the unmodified~\\citet{Reid2013} ``fit_disk`` sampler
converge to *different* angular-diameter distances (non-mixing on the
mass--distance ridge), and that adding our modifications makes every chain
agree on the same posterior.

Layout
------
Three figures, split so the panels are not squished.  In each figure a galaxy
occupies one COLUMN and its two sampler variants are stacked vertically:
    top    = ``reid_original``        (native joint Metropolis sampler)
    bottom = ``eta_gibbs_reflection`` (native code + our (H0, eta) coordinates,
                                       Metropolis-within-Gibbs latents, and the
                                       azimuthal reflection move)
    figure 1 : CGCG 074-064, NGC 5765b   (2x2)
    figure 2 : NGC 6264, NGC 6323        (2x2)
    figure 3 : UGC 3789                  (single column, 2x1)
The two panels in a column share their distance axis, so the change in
inter-chain agreement is read directly.  Each galaxy's twelve chains share ONE
starting point and differ only in their random seed, so any inter-chain spread
is the sampler alone (see ``docs/reid_fit_disk_modifications.md``).

Curves in every panel
---------------------
    thin coloured  : the twelve individual chains, one step histogram each
                     (``tab20`` colours), all density-normalised
    bold black     : the pooled histogram of all twelve chains stacked together
                     (the naive combined posterior)
    bold red       : our CANDEL posterior for the same galaxy (baseline linear
                     warp), i.e. the answer the modified sampler should recover

Data sources
------------
    Reid chains : ``results/Megamaser/reid_mcmc/gibbs_sweep_20260705_014726/
                   <GALAXY>_config/<variant>/chain_*/fort.7`` loaded with
                   ``run_gibbs_comparison.load_variant_chains`` (drops the
                   first 20% as burn-in); the sampled H0 is mapped to the
                   angular-diameter distance ``D_Mpc`` inside ``load_chain``.
    R-hat, Neff : the split-R-hat and effective sample size on the distance,
                   read off the ``D_Mpc`` row of ``<GALAXY>_<variant>_global_
                   summary.txt`` (numpyro; same diagnostic as the CANDEL
                   summaries), so the annotated numbers are never hand-entered.
    CANDEL      : ``results/Megamaser/<dataset>/<GALAXY>/<GALAXY>_blackjax_
                   mcmc_rphi_initreid.hdf5``, distance samples ``samples/D_A``
                   (Mpc), directly comparable to Reid's ``D_Mpc``.

Run
---
    python plot_reid_chain_nonmixing.py --outdir figs
    # writes the three figures listed above into <outdir>
    # optional: --sweep <dir>  --dataset <name>  --candel-root <dir>
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from ..paths import RESULTS_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
from .run_gibbs_comparison import load_variant_chains  # noqa: E402
from .run_reid_mcmc import (  # noqa: E402
    DEFAULT_CONFIG, add_dataset_arg, load_toml, resolve_dataset)

ROOT = Path(RESULTS_ROOT)
SWEEP = ROOT / "results/Megamaser/reid_mcmc/gibbs_sweep_20260705_014726"
MASER_RESULTS = ROOT / "results/Megamaser"

# galaxies in paper order (sweep sub-directory stem, display label)
GALAXIES = [("CGCG074-064", "CGCG 074-064"),
            ("NGC5765b", "NGC 5765b"),
            ("UGC3789", "UGC 3789"),
            ("NGC6264", "NGC 6264"),
            ("NGC6323", "NGC 6323")]
VARIANTS = [("reid_original", "native"),
            ("eta_gibbs_reflection", "Gibbs + reflection")]

# the three output figures: (column galaxy stems, output filename stem)
GROUPS = [(["CGCG074-064", "NGC5765b"], "reid_chain_nonmixing_cgcg_ngc5765b"),
          (["NGC6264", "NGC6323"], "reid_chain_nonmixing_ngc6264_ngc6323"),
          (["UGC3789"], "reid_chain_nonmixing_ugc3789")]


def parse_diag(summary_file):
    """(split-R-hat, N_eff) on the distance from a numpyro global summary,
    read off the ``D_Mpc`` row (fields: name mean std median 5% 95% n_eff
    r_hat).  numpyro computes these, so they match the CANDEL summaries."""
    for line in Path(summary_file).read_text().splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "D_Mpc":
            return float(p[7]), float(p[6])
    raise ValueError(f"no D_Mpc row in {summary_file}")


def candel_D_A(galaxy, candel_root):
    """CANDEL baseline linear-warp angular-diameter distance samples [Mpc]."""
    h5 = candel_root / galaxy / f"{galaxy}_blackjax_mcmc_rphi_initreid.hdf5"
    with h5py.File(h5, "r") as f:
        return f["samples"]["D_A"][:].reshape(-1).astype(np.float64)


def plot_group(subset, chains, cand, diag, out):
    """One figure: galaxies as columns, the two variants stacked as rows."""
    preamble = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{lmodern}"
    with plt.style.context(["science", {"text.latex.preamble": preamble}]):
        ncol, nrow = len(subset), len(VARIANTS)
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(3.5 * ncol, 2.1 * nrow + 0.3),
                                 sharex="col", squeeze=False)
        for c, (g, glabel) in enumerate(subset):
            # shared distance axis per galaxy: pool every chain plus CANDEL
            pool = np.concatenate([a["D_Mpc"] for v, _ in VARIANTS
                                   for a in chains[(g, v)]] + [cand[g]])
            lo, hi = np.percentile(pool, [0.5, 99.5])
            bins = np.linspace(lo, hi, 90)
            for r, (v, case) in enumerate(VARIANTS):
                ax = axes[r, c]
                arrs = chains[(g, v)]
                colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(arrs)))
                for a, col in zip(arrs, colors):
                    ax.hist(a["D_Mpc"], bins=bins, density=True,
                            histtype="step", lw=0.8, color=col, alpha=0.75,
                            zorder=3)
                pooled = np.concatenate([a["D_Mpc"] for a in arrs])
                ax.hist(pooled, bins=bins, density=True, histtype="step",
                        lw=1.7, color="k", zorder=5,
                        label=r"Pooled \texttt{fit\_disk} chains"
                        if (r, c) == (0, 0) else None)
                ax.hist(cand[g], bins=bins, density=True, histtype="step",
                        lw=1.7, color="crimson", zorder=6,
                        label="This work" if (r, c) == (0, 0) else None)
                ax.set_xlim(lo, hi)
                ax.set_ylim(bottom=0.0)
                r_hat, neff = diag[(g, v)]
                ax.set_title(rf"{glabel}, {case} ($\hat{{R}} = {r_hat:.2f}$, "
                             rf"$N_\mathrm{{eff}} = {neff:.0f}$)",
                             fontsize=7.5)
                if r == nrow - 1:
                    ax.set_xlabel(r"$D_\mathrm{A}\ [\mathrm{Mpc}]$")
                if c == 0:
                    ax.set_ylabel(r"$p(D_\mathrm{A})$")
        axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.18, hspace=0.45)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-chain distance grids.")
    ap.add_argument("--sweep", type=Path, default=SWEEP)
    add_dataset_arg(ap)
    ap.add_argument("--candel-root", type=Path, default=None,
                    help="CANDEL results root (default: "
                         "results/Megamaser/<dataset>).")
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args(argv)

    if args.candel_root is None:
        args.candel_root = MASER_RESULTS / resolve_dataset(
            load_toml(DEFAULT_CONFIG), args.dataset)

    label = dict(GALAXIES)
    chains, cand, diag = {}, {}, {}
    for stems, _ in GROUPS:
        for g in stems:
            gdir = args.sweep / f"{g}_config"
            cand[g] = candel_D_A(g, args.candel_root)
            for v, _ in VARIANTS:
                chains[(g, v)] = load_variant_chains(gdir / v)
                diag[(g, v)] = parse_diag(
                    gdir / f"{g}_{v}_global_summary.txt")

    for stems, stem in GROUPS:
        subset = [(g, label[g]) for g in stems]
        plot_group(subset, chains, cand, diag,
                   args.outdir / f"{stem}.pdf")


if __name__ == "__main__":
    main()
