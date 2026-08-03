"""Convergence table for the megamaser H0 appendix (Non-convergence of the
unmodified MCP sampler).

For each galaxy, the native Reid ``fit_disk`` sampler and the cumulative
modifications (``+eta``, ``+Gibbs``, ``+reflection``) are summarised on the
angular-diameter distance for TWO starting points: our configuration median
(``config``) and the~\\citetalias{Pesce2020} median (``pesce``).  The reported
quantities are

    Rhat, Neff   : split-R-hat and effective sample size on ``D_Mpc``, read
                   off the ``D_Mpc`` row of the numpyro global summary written
                   by ``run_gibbs_comparison`` (never recomputed here, so the
                   table matches the per-panel annotations of the figure).
    mean D        : pooled posterior mean distance over the twelve chains, Mpc.
    sigma_chain   : DIMENSIONLESS non-mixing statistic, the between-chain
                   scatter of the per-chain mean distances expressed in units
                   of the typical within-chain posterior width,
                   ``std(per-chain means) / rms(per-chain stds)``.  Normalising
                   by the within-chain spread turns the raw Mpc scatter into a
                   "statistical sigma": <<1 when chains agree within their own
                   width, ~1 when they sit a full posterior width apart.  This
                   makes galaxies with very different posterior widths (e.g.
                   the tight UGC~3789 vs the broad NGC~6323) directly
                   comparable.

Data source: the sweep produced by ``submit_gibbs_comparison.sh`` (twelve
chains of 1e6 warm-up + 1e6 sampling steps per variant), laid out as
``<sweep>/<GALAXY>_<init>/<variant>/chain_*/fort.7`` with the numpyro summary
``<GALAXY>_<init>/<GALAXY>_<variant>_global_summary[_pesceinit].txt``.

Run
---
    python make_reid_neff_table.py                 # plain-text table, all cols
    python make_reid_neff_table.py --latex         # emit the LaTeX table body
    # optional: --sweep <dir>  --galaxies NGC6323 CGCG074-064
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_gibbs_comparison import load_variant_chains  # noqa: E402

ROOT = HERE.parents[2]
SWEEP = ROOT / "results/Megamaser/reid_mcmc/gibbs_sweep_20260705_014726"

# galaxies in paper order (sweep stem, display label)
GALAXIES = [("CGCG074-064", "CGCG~074-064"),
            ("NGC5765b", "NGC~5765b"),
            ("UGC3789", "UGC~3789"),
            ("NGC6264", "NGC~6264"),
            ("NGC6323", "NGC~6323")]
# (variant sub-dir, column label); paper order along the modification ladder
VARIANTS = [("reid_original", "Native"),
            ("eta_reparam", r"$+\,\eta$"),
            ("eta_gibbs", r"$+\,$Gibbs"),
            ("eta_gibbs_reflection", r"$+\,$reflection")]
INITS = [("config", ""), ("pesce", "_pesceinit")]


def parse_diag(summary_file):
    """(split-R-hat, N_eff) on the distance from the ``D_Mpc`` row of a numpyro
    global summary (fields: name mean std median 5% 95% n_eff r_hat)."""
    for line in Path(summary_file).read_text().splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "D_Mpc":
            return float(p[7]), float(p[6])
    raise ValueError(f"no D_Mpc row in {summary_file}")


def distance_stats(variant_dir):
    """Pooled median distance with its central 68% (16--84 percentile)
    interval, and the dimensionless non-mixing statistic, from the twelve
    ``fort.7`` chains (first 20% dropped).  Returns
    ``(med, lo_err, hi_err, sigma_chain)`` where the errors are the
    median-to-P16 and P84-to-median offsets and

        sigma_chain = std(per-chain means, ddof=1) / rms(per-chain stds)

    is the between-chain scatter of the per-chain mean distances in units of
    the typical within-chain posterior width (Harry's "statistical sigma").
    The denominator is ``sqrt(mean of per-chain variances)`` = the
    Gelman--Rubin within-chain standard deviation, so the whole ratio is
    ``sqrt(Rhat^2 - 1)`` in the large-sample limit -- the same non-mixing that
    R-hat encodes, but read as a fraction of the posterior width.  The median
    matches the summary statistic of the paper's reported distances, so the
    modified sampler's median lines up with the ``This work`` column even for
    the skewed broad distance posteriors."""
    arrs = load_variant_chains(variant_dir)
    per_chain_mean = np.array([a["D_Mpc"].mean() for a in arrs])
    sig_between = per_chain_mean.std(ddof=1)
    sig_intra = np.sqrt(np.mean([a["D_Mpc"].var(ddof=1) for a in arrs]))
    pooled = np.concatenate([a["D_Mpc"] for a in arrs])
    p16, med, p84 = np.percentile(pooled, [16, 50, 84])
    return med, med - p16, p84 - med, sig_between / sig_intra


def collect(sweep, gals):
    """rows[(galaxy, init, variant)] = (rhat, neff, mean, lo, hi, schain)."""
    rows = {}
    for g, _ in gals:
        for init, suf in INITS:
            gdir = sweep / f"{g}_{init}"
            for v, _ in VARIANTS:
                summ = gdir / f"{g}_{v}_global_summary{suf}.txt"
                rhat, neff = parse_diag(summ)
                mean, lo, hi, schain = distance_stats(gdir / v)
                rows[(g, init, v)] = (rhat, neff, mean, lo, hi, schain)
    return rows


def print_text(rows, gals):
    hdr = (f"{'galaxy':14s} {'init':7s} {'variant':16s} "
           f"{'Rhat':>6s} {'Neff':>7s} {'medD':>18s} {'schain':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for g, glabel in gals:
        print(f"  reported {'':30s} {REPORTED[g]}")
        for init, _ in INITS:
            for v, vlabel in VARIANTS:
                rhat, neff, med, lo, hi, schain = rows[(g, init, v)]
                d = f"{med:.1f} +{hi:.1f}/-{lo:.1f}"
                print(f"{g:14s} {init:7s} {v:16s} "
                      f"{rhat:6.2f} {neff:7.0f} {d:>18s} {schain:8.2f}")
        print()


def _cell(rhat, neff, med, lo, hi, schain):
    return (f"${rhat:.2f}$ & ${neff:.0f}$ & "
            f"${med:.1f}^{{+{hi:.1f}}}_{{-{lo:.1f}}}$ & ${schain:.2f}$")


# native and the fully modified (our (H0, eta) + Gibbs + reflection) endpoints
SHOW = [("reid_original", "Native"), ("eta_gibbs_reflection", "Modified")]
INIT_LABELS = {"config": r"\textit{Our configuration starting point}",
               "pesce": r"\textit{\citetalias{Pesce2020} starting point}"}

# Reported linear-warp distance [Mpc] (median + central 68%), copied verbatim
# from the paper's tab:distance_pesce_comparison so the appendix shows the same
# reported value as the main results table.  This is the target the modified
# sampler should recover (the bold red curve of fig:reid_nonmixing).
REPORTED = {
    "CGCG074-064": r"$86.1^{+8.2}_{-7.2}$",
    "NGC5765b": r"$120.6^{+8.4}_{-7.6}$",
    "UGC3789": r"$54.6^{+5.6}_{-4.7}$",
    "NGC6264": r"$162^{+33}_{-25}$",
    "NGC6323": r"$148^{+55}_{-36}$",
}


def print_latex(rows, gals):
    """Full ``tabular`` for the appendix convergence table: native vs modified
    \\texttt{fit\\_disk}, two starting-point blocks, four quantities each.
    Numbers are traced to the sweep, never hand-set."""
    print(r"    \begin{tabular*}{\textwidth}"
          r"{@{\extracolsep{\fill}}lccccccccc@{}}")
    print(r"    \toprule")
    print(r"    & \multicolumn{4}{c}{Original \texttt{fit\_disk}} "
          r"& \multicolumn{4}{c}{Modified \texttt{fit\_disk}} "
          r"& \multicolumn{1}{c}{This work} \\")
    print(r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-10}")
    dmpc = r"$\DA$ [$\mathrm{Mpc}$]"
    schain = r"$\sigma_\mathrm{chain}/\sigma_\mathrm{intra}$"
    print(rf"    Galaxy & $\hat{{R}}$ & $N_\mathrm{{eff}}$ & {dmpc} & {schain}"
          rf" & $\hat{{R}}$ & $N_\mathrm{{eff}}$ & {dmpc} & {schain}"
          rf" & {dmpc} \\")
    for init, _ in INITS:
        print(r"    \midrule")
        print(f"    \\multicolumn{{10}}{{l}}{{{INIT_LABELS[init]}}} \\\\")
        for g, glabel in gals:
            cells = " & ".join(_cell(*rows[(g, init, v)]) for v, _ in SHOW)
            print(f"    {glabel} & {cells} & {REPORTED[g]} \\\\")
    print(r"    \bottomrule")
    print(r"    \end{tabular*}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, default=SWEEP)
    ap.add_argument("--galaxies", nargs="+", default=None)
    ap.add_argument("--latex", action="store_true",
                    help="emit the LaTeX table body instead of the text table")
    args = ap.parse_args(argv)

    gals = ([(g, lab) for g, lab in GALAXIES if g in set(args.galaxies)]
            if args.galaxies else GALAXIES)
    rows = collect(args.sweep, gals)
    (print_latex if args.latex else print_text)(rows, gals)


if __name__ == "__main__":
    main()
