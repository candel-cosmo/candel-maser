#!/usr/bin/env python3
"""Generate MMH0 per-galaxy result tables from megamaser chains."""
import argparse
import math
import tomllib
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

ROOT = Path(__file__).resolve().parents[2]
DATASET = "original_published"
DEFAULT_RESULTS = ROOT / "results" / "Megamaser" / DATASET
PESCE_PARAMS = ROOT / "scripts" / "megamaser" / "check_reid" / (
    "pesce_disk_params.toml")

GALAXIES = ["CGCG074-064", "NGC5765b", "UGC3789", "NGC6264", "NGC6323"]
GAL_LABELS = {
    "CGCG074-064": r"CGCG~074-064",
    "NGC5765b": r"NGC~5765b",
    "UGC3789": r"UGC~3789",
    "NGC6264": r"NGC~6264",
    "NGC6323": r"NGC~6323",
}
PESCE_DA_ERR = {
    "CGCG074-064": (5.8, 4.7),
    "NGC5765b": (5.4, 5.1),
    "UGC3789": (3.9, 3.8),
    "NGC6264": (8.9, 8.0),
    "NGC6323": (7.4, 7.1),
}

BASE_ROWS = [
    ("D_A", r"$\DA$ (Mpc)", "D_A (Mpc)"),
    ("log_MBH", r"$\log_{10}(\MBH/\Msun)$", "log10(M_BH/Msun)"),
    ("i0", r"$i_0$ (deg)", "i0 (deg)"),
    ("di_dr", r"$\mathrm{d}i/\mathrm{d}r$ (deg mas$^{-1}$)",
     "di/dr (deg/mas)"),
    ("Omega0", r"$\Omega_0$ (deg)", "Omega0 (deg)"),
    ("dOmega_dr",
     r"$\mathrm{d}\Omega/\mathrm{d}r$ (deg mas$^{-1}$)",
     "dOmega/dr (deg/mas)"),
    ("x0", r"$x_0$ ($\mu$as)", "x0 (uas)"),
    ("y0", r"$y_0$ ($\mu$as)", "y0 (uas)"),
    ("dv_sys", r"$\Delta v_\mathrm{sys}$ ($\kmsec$)",
     "Delta v_sys (km/s)"),
    ("sigma_x_floor", r"$\sigma_x$ ($\mu$as)", "sigma_x (uas)"),
    ("sigma_y_floor", r"$\sigma_y$ ($\mu$as)", "sigma_y (uas)"),
    ("sigma_v_sys", r"$\sigma_{v,\mathrm{sys}}$ ($\kmsec$)",
     "sigma_v_sys (km/s)"),
    ("sigma_v_hv", r"$\sigma_{v,\mathrm{hv}}$ ($\kmsec$)",
     "sigma_v_hv (km/s)"),
    ("sigma_a_floor", r"$\sigma_a$ ($\kmsecyr$)",
     "sigma_a (km/s/yr)"),
    # NGC 5765b only; blank elsewhere (see table_rows).
    ("sigma_x_floor_clump2", r"$\sigma_x^{(2)}$ ($\mu$as)",
     "sigma_x clump2 (uas)"),
    ("sigma_y_floor_clump2", r"$\sigma_y^{(2)}$ ($\mu$as)",
     "sigma_y clump2 (uas)"),
    ("sigma_v_floor_clump2", r"$\sigma_{v,\mathrm{sys}}^{(2)}$ ($\kmsec$)",
     "sigma_v_sys clump2 (km/s)"),
    ("sigma_a_floor_clump2", r"$\sigma_a^{(2)}$ ($\kmsecyr$)",
     "sigma_a clump2 (km/s/yr)"),
]
QW_ROWS = BASE_ROWS[:6] + [
    ("d2i_dr2", r"$\mathrm{d}^2i/\mathrm{d}r^2$ (deg mas$^{-2}$)",
     "d2i/dr2 (deg/mas^2)"),
    ("d2Omega_dr2",
     r"$\mathrm{d}^2\Omega/\mathrm{d}r^2$ (deg mas$^{-2}$)",
     "d2Omega/dr2 (deg/mas^2)"),
] + BASE_ROWS[6:]


# NGC 4258 has a single spot table, so `fiducial` and `original_published`
# are the same data; the clipped runs drop the spots our own iterative
# rejection removes under the quadratic warp.
NGC4258_RUNS = [
    ("Quadratic", "original_published", "_qw"),
    ("Quadratic + clipping", "clipped", "_qw"),
    ("Quadratic + eccentricity", "fiducial", "_ecc_qw"),
    ("Quadratic + eccentricity + clipping", "clipped", "_ecc_qw"),
]
# Labels here use the paper's unit macros, unlike the five-galaxy tables
# above, so the emitted rows can be pasted into main.tex unedited.
NGC4258_ROWS = [
    ("D_A", r"$\DA$ (\Mpc)", "D_A (Mpc)"),
    ("log_MBH", r"$\log(\MBH/\Msun)$", "log10(M_BH/Msun)"),
    ("i0", r"$i_0$ (\degunit)", "i0 (deg)"),
    ("di_dr", r"$\mathrm{d}i/\mathrm{d}r|_{r_\mathrm{ref}}$ (\degmas)",
     "di/dr (deg/mas)"),
    ("d2i_dr2",
     r"$\mathrm{d}^2i/\mathrm{d}r^2|_{r_\mathrm{ref}}$ (\degmassq)",
     "d2i/dr2 (deg/mas^2)"),
    ("Omega0", r"$\Omega_0$ (\degunit)", "Omega0 (deg)"),
    ("dOmega_dr",
     r"$\mathrm{d}\Omega/\mathrm{d}r|_{r_\mathrm{ref}}$ (\degmas)",
     "dOmega/dr (deg/mas)"),
    ("d2Omega_dr2",
     r"$\mathrm{d}^2\Omega/\mathrm{d}r^2|_{r_\mathrm{ref}}$ (\degmassq)",
     "d2Omega/dr2 (deg/mas^2)"),
    ("e", r"$e$", "e"),
    ("omega0", r"$\omega_0$ (\degunit)", "omega0 (deg)"),
    ("dperiapsis_dr",
     r"$\mathrm{d}\omega/\mathrm{d}r|_{r_\mathrm{ref}^{\omega}}$ (\degmas)",
     "domega/dr (deg/mas)"),
    ("x0", r"$x_0$ (\muas)", "x0 (uas)"),
    ("y0", r"$y_0$ (\muas)", "y0 (uas)"),
    ("dv_sys", r"$\Delta V_\mathrm{sys}$ ($\kmsec$)", "Delta V_sys (km/s)"),
    ("sigma_x_floor", r"$\sigma_x$ (\muas)", "sigma_x (uas)"),
    ("sigma_y_floor", r"$\sigma_y$ (\muas)", "sigma_y (uas)"),
    ("sigma_v_sys", r"$\sigma_{v,\mathrm{sys}}$ ($\kmsec$)",
     "sigma_v_sys (km/s)"),
    ("sigma_v_hv", r"$\sigma_{v,\mathrm{hv}}$ ($\kmsec$)",
     "sigma_v_hv (km/s)"),
    ("sigma_a_floor", r"$\sigma_a$ ($\kmsecyr$)", "sigma_a (km/s/yr)"),
]


def distance_to_redshift(dc, om=0.315, h=0.73):
    cosmo = FlatLambdaCDM(H0=100, Om0=om)
    z_grid = np.logspace(-8, np.log10(0.5), 1000)
    r_grid = cosmo.comoving_distance(z_grid).value
    return np.interp(np.asarray(dc, dtype=float) * h, r_grid, z_grid)


def da_from_dc(dc):
    z = distance_to_redshift(dc)
    return np.asarray(dc, dtype=float) / (1.0 + z)


def chain_path(results_root, galaxy, qw, init):
    variant = "_qw" if qw else ""
    name = f"{galaxy}_blackjax_mcmc_rphi{variant}_init{init}.hdf5"
    return results_root / galaxy / name


def load_chain(path):
    with h5py.File(path, "r") as f:
        out = {k: np.asarray(v, dtype=float).ravel()
               for k, v in f["samples"].items()}
    # uniform_D_A chains store D_A directly; legacy uniform-D_c chains
    # derive it.
    if "D_A" not in out:
        out["D_A"] = da_from_dc(out["D_c"])
    return out


def interval(x):
    q16, med, q84 = np.percentile(np.asarray(x, dtype=float), [16, 50, 84])
    return med, q84 - med, med - q16


def decimals_for_uncertainty(err, nsig=2):
    """Decimal places so `err` carries `nsig` significant figures.

    With nsig=1 an uncertainty whose leading digit is 1 keeps two figures,
    the usual rule: rounding 0.013 to 0.01 would throw away 30 per cent.
    """
    err = abs(float(err))
    if not math.isfinite(err) or err <= 0:
        return 2
    exponent = math.floor(math.log10(err))
    if nsig == 1 and int(err / 10 ** exponent) == 1:
        nsig = 2
    return max(0, nsig - 1 - exponent)


def fmt_interval(x, latex, nsig=2):
    med, plus, minus = interval(x)
    ndp = max(
        decimals_for_uncertainty(plus, nsig),
        decimals_for_uncertainty(minus, nsig),
    )
    fmt = f"{{:.{ndp}f}}"
    s = f"{fmt.format(med)}^{{+{fmt.format(plus)}}}_{{-{fmt.format(minus)}}}"
    return f"${s}$" if latex else (
        f"{fmt.format(med)} (+{fmt.format(plus)}/-{fmt.format(minus)})")


def table_rows(chains, rows, latex):
    labels = [GAL_LABELS[g] if latex else g for g in GALAXIES]
    out = []
    if latex:
        out += [
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            "Parameter & " + " & ".join(labels) + r" \\",
            r"\midrule",
        ]
    else:
        out += [
            "| Parameter | " + " | ".join(labels) + " |",
            "|---|" + "|".join(["---"] * len(labels)) + "|",
        ]

    for key, tex_label, md_label in rows:
        # The clump-2 floors are sampled for NGC 5765b only, and only on the
        # tables that retain the second systemic clump. Skip a row no galaxy
        # has; dash the galaxies that lack one the others have.
        if not any(key in chains[g] for g in GALAXIES):
            continue
        label = tex_label if latex else md_label
        vals = [fmt_interval(chains[g][key], latex) if key in chains[g]
                else ("---" if latex else "-") for g in GALAXIES]
        if latex:
            out.append(label + " & " + " & ".join(vals) + r" \\")
        else:
            out.append("| " + label + " | " + " | ".join(vals) + " |")

    if latex:
        out += [r"\bottomrule", r"\end{tabular}"]
    return out


def parameter_table(chains, qw, latex):
    title = (
        "Quadratic-warp disc parameters" if qw
        else "Fiducial disc parameters"
    )
    rows = QW_ROWS if qw else BASE_ROWS
    body = table_rows(chains, rows, latex)
    if not latex:
        return "\n".join([f"## {title}", "", *body])

    label = "tab:disc_params_qw" if qw else "tab:disc_params"
    caption = ("Posterior intervals for the quadratic-warp per-galaxy disc "
               "model." if qw else
               "Posterior intervals for the fiducial per-galaxy disc model.")
    return "\n".join([
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{" + caption + " Entries give the median and central "
        r"$68\%$ posterior interval for the config-initialised chains.}",
        r"\label{" + label + "}",
        *body,
        r"\end{table*}",
    ])


def ngc4258_table(results_root, latex):
    """Variant table for NGC 4258, one column per disc-model variant."""
    cols = []
    for label, dataset, variant in NGC4258_RUNS:
        path = (Path(results_root).parent / dataset / "NGC4258"
                / f"NGC4258_blackjax_mcmc_rphi{variant}_initconfig.hdf5")
        if not path.exists():
            print(f"# missing, column left blank: {path}")
            cols.append((label, None, None))
            continue
        with h5py.File(path, "r") as f:
            s = {k: np.asarray(v, dtype=float).ravel()
                 for k, v in f["samples"].items()}
            n_spots = int(f.attrs["n_spots"])
        if "e_x" in s:
            s["e"] = np.hypot(s["e_x"], s["e_y"])
            s["omega0"] = np.degrees(np.arctan2(s["e_y"], s["e_x"])) % 360.0
        cols.append((label, s, n_spots))

    dash = "---" if latex else "-"
    sep = " & " if latex else " | "
    out = []
    if latex:
        out += [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
                "Parameter" + sep
                + sep.join(c[0] for c in cols) + r" \\", r"\midrule"]
    else:
        out += ["| Parameter | " + " | ".join(c[0] for c in cols) + " |",
                "|---|" + "|".join(["---"] * len(cols)) + "|"]

    # One significant figure on the uncertainty, and a decimal count shared
    # by every column of a row, so the columns line up and can be compared.
    def fmt_row(key):
        present = [c[1][key] for c in cols if c[1] is not None and key in c[1]]
        ndp = max(max(decimals_for_uncertainty(e, nsig=1)
                      for e in interval(x)[1:]) for x in present)
        fmt = f"{{:.{ndp}f}}"
        out = []
        for c in cols:
            if c[1] is None or key not in c[1]:
                out.append(dash)
                continue
            med, plus, minus = interval(c[1][key])
            s = (f"{fmt.format(med)}^{{+{fmt.format(plus)}}}"
                 f"_{{-{fmt.format(minus)}}}")
            out.append(f"${s}$" if latex else
                       f"{fmt.format(med)} (+{fmt.format(plus)}/"
                       f"-{fmt.format(minus)})")
        return out

    rows = [("Spots", [dash if c[2] is None else str(c[2]) for c in cols])]
    for key, tex_label, md_label in NGC4258_ROWS:
        if not any(c[1] is not None and key in c[1] for c in cols):
            continue
        rows.append((tex_label if latex else md_label, fmt_row(key)))

    for label, vals in rows:
        if latex:
            out.append(label + sep + sep.join(vals) + r" \\")
        else:
            out.append("| " + label + " | " + " | ".join(vals) + " |")

    if latex:
        out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def load_pesce_distances(path):
    with open(path, "rb") as f:
        return tomllib.load(f)["galaxies"]


def comparison_table(chains, pesce, latex):
    headers = ["Galaxy", "This work D_A (Mpc)", "Pesce D_A (Mpc)",
               "Delta D_A (Mpc)", "Delta/sigma_comb"]
    if latex:
        out = [
            r"\begin{table*}",
            r"\centering",
            r"\caption{Comparison between the fiducial config-initialised "
            r"angular-diameter distances and the reported "
            r"\citetalias{Pesce2020} values. The offset is quoted in units "
            r"of the quadrature sum of the central $68\%$ posterior width "
            r"and the symmetrised reported uncertainty.}",
            r"\label{tab:distance_pesce_comparison}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Galaxy & This work $\DA$ (Mpc) & \citetalias{Pesce2020} "
            r"$\DA$ (Mpc) & $\Delta\DA$ (Mpc) & "
            r"$\Delta/\sigma_\mathrm{comb}$ \\",
            r"\midrule",
        ]
    else:
        out = ["## Distance comparison", ""]
        out.append("| " + " | ".join(headers) + " |")
        out.append("|---|" + "|".join(["---"] * (len(headers) - 1)) + "|")

    for gal in GALAXIES:
        med, plus, minus = interval(chains[gal]["D_A"])
        p20 = float(pesce[gal]["D_Mpc"])
        p20_plus, p20_minus = PESCE_DA_ERR[gal]
        sig_this = 0.5 * (plus + minus)
        sig_p20 = 0.5 * (p20_plus + p20_minus)
        delta = med - p20
        nsig = delta / math.hypot(sig_this, sig_p20)
        if latex:
            ndp = decimals_for_uncertainty(math.hypot(sig_this, sig_p20))
            row = [
                GAL_LABELS[gal],
                fmt_interval(chains[gal]["D_A"], True),
                f"${p20:.1f}^{{+{p20_plus:.1f}}}_{{-{p20_minus:.1f}}}$",
                f"${delta:+.{ndp}f}$",
                f"${nsig:+.2f}$",
            ]
            out.append(" & ".join(row) + r" \\")
        else:
            ndp = decimals_for_uncertainty(math.hypot(sig_this, sig_p20))
            row = [
                gal,
                fmt_interval(chains[gal]["D_A"], False),
                f"{p20:.1f} (+{p20_plus:.1f}/-{p20_minus:.1f})",
                f"{delta:+.{ndp}f}",
                f"{nsig:+.2f}",
            ]
            out.append("| " + " | ".join(row) + " |")

    if latex:
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out)


def build_tables(args):
    results_root = Path(args.results_root)
    fid = qw = pesce = None
    if args.table in ("all", "fiducial", "comparison"):
        fid = {
            gal: load_chain(chain_path(results_root, gal, False, args.init))
            for gal in GALAXIES
        }
    if args.table in ("all", "quadratic"):
        qw = {
            gal: load_chain(chain_path(results_root, gal, True, args.init))
            for gal in GALAXIES
        }
    if args.table in ("all", "comparison"):
        pesce = load_pesce_distances(Path(args.pesce_params))
    latex = args.format == "latex"

    pieces = []
    if args.table in ("all", "fiducial"):
        pieces.append(parameter_table(fid, False, latex))
    if args.table in ("all", "quadratic"):
        pieces.append(parameter_table(qw, True, latex))
    if args.table in ("all", "comparison"):
        pieces.append(comparison_table(fid, pesce, latex))
    if args.table == "ngc4258":
        pieces.append(ngc4258_table(results_root, latex))
    return "\n\n".join(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--pesce-params", default=str(PESCE_PARAMS))
    parser.add_argument("--init", default="config",
                        help="chain init suffix, e.g. config or reid")
    parser.add_argument("--table",
                        choices=("all", "fiducial", "quadratic",
                                 "comparison", "ngc4258"),
                        default="all")
    parser.add_argument("--format", choices=("markdown", "latex"),
                        default="markdown")
    parser.add_argument("--output",
                        help="write to this file instead of stdout")
    args = parser.parse_args()

    text = build_tables(args)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
