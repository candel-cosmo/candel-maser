#!/usr/bin/env python3
"""Run fit_disk_Reid_reflection sampler variants (reid_original = Reid's
true original formulation, always eta-off; eta_reparam, eta_gibbs,
eta_gibbs_reflection = Gibbs+reflection; all four or a --variants subset),
each as several parallel chains via run_gibbs_chains.sh, then report
genuine multi-chain Gelman-Rubin R-hat and make the CANDEL vs Reid
overlay plots.

The Reid data file (data/Megamaser/<galaxy>_loader_reid.inp) is generated
automatically if missing.

Modes:
  default        run chains AND the R-hat/summaries/plots post-processing
  --chains-only  run chains only (one cluster job per variant can share an
                 --out-dir; see submit_gibbs_comparison.sh)
  --skip-run     post-process existing chains only (the --collect step)

Example:
    venv_candel/bin/python run_gibbs_comparison.py --out-dir /path/to/output
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_reid_data  # noqa: E402
from compare_reid_candel import (DEFAULT_CONFIG, SHARED,  # noqa: E402
                                 candel_to_reid, overlay_distance_histogram,
                                 overlay_three, per_chain_distance_histogram)
from run_reid_mcmc import (GLOBAL_NAMES, compute_reid_r_ref,  # noqa: E402
                           load_chain, load_galaxy_config,
                           numpyro_summary_text, parse_data_rows,
                           set_control_numbers)

ROOT = HERE.parents[2]
RUN_SCRIPT = HERE / "run_gibbs_chains.sh"

# (label, use_gibbs, p_reflect, use_gcov, eta, plot color).  eta None
# follows --eta/--no-eta; "F" pins the variant to Reid's true original
# flat-(H0, M) formulation regardless of the flag.
VARIANTS = [
    ("reid_original", "F", "0.0", "F", "F", "C5"),
    ("eta_reparam", "F", "0.0", "F", None, "C3"),
    ("eta_gibbs", "T", "0.0", "F", None, "C1"),
    ("eta_gibbs_reflection", "T", "0.25", "F", None, "C2"),
]

PARAMS = ["H0", "D_Mpc", "Mbh_1e7Msun", "sigma_vsys_km_s", "sigma_vhv_km_s"]
SUMMARY_NAMES = GLOBAL_NAMES + ["D_Mpc"]
MAX_RHAT_SAMPLES = 100_000


def variant_summary(arrs):
    """numpyro multi-chain summary (n_eff, split-R-hat) of the global
    chains."""
    minlen = min(len(a) for a in arrs)
    samples = {k: np.stack([a[k][-minlen:].astype(np.float64) for a in arrs],
                           axis=0)
               for k in SUMMARY_NAMES}
    return numpyro_summary_text(samples)


def gelman_rubin(chains):
    """Standard Gelman & Rubin (1992) / BDA3 R-hat.

    chains: (M chains, N draws)."""
    M, N = chains.shape
    theta_bar_m = chains.mean(axis=1)
    theta_bar = theta_bar_m.mean()
    W = chains.var(axis=1, ddof=1).mean()
    B = N / (M - 1) * np.sum((theta_bar_m - theta_bar) ** 2)
    var_plus = (N - 1) / N * W + B / N
    return float(np.sqrt(var_plus / W))


def _banner(title):
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def run_identity(g, args):
    """One-line run identity (galaxy, init source, eta/floor/reweight
    toggles) stamped on every printed section and saved artifact, so a
    report or log excerpt is self-describing without needing the
    filename's tag suffix."""
    bits = [f"galaxy={g}", f"init={args.init}",
            f"eta={'on' if args.eta else 'off'}"]
    if args.H0_range:
        bits.append(f"H0={args.H0_range[0]:g}-{args.H0_range[1]:g}")
    if args.reweight_da2:
        bits.append("reweight_da2=on")
    if args.match_priors != "T":
        bits.append(f"match_priors={args.match_priors}")
    return " ".join(bits)


def thin_chains(chains, max_total=MAX_RHAT_SAMPLES):
    """Downsample (M chains, N draws) to at most max_total total draws by
    taking every nth draw from each chain."""
    M, N = chains.shape
    per_chain_cap = max(1, max_total // M)
    step = max(1, int(np.ceil(N / per_chain_cap)))
    return chains[:, ::step]


def run_variant(control_template, data_file, out_dir, gibbs, reflect, gcov,
                chains, warmup, samples, seed, n_inner, eta, write_thin,
                match_priors, galaxy):
    cmd = ["bash", str(RUN_SCRIPT), str(control_template), str(data_file),
           str(out_dir), "--chains", str(chains), "--gibbs", gibbs,
           "--reflect", reflect, "--global-cov", gcov,
           "--eta", "T" if eta else "F",
           # error-floor treatment, forwarded verbatim (T=CANDEL Gaussian
           # sampled / F=template verbatim / orig=per-galaxy published fixed
           # floors, so --galaxy selects the right paper's values)
           "--match-priors", match_priors, "--galaxy", galaxy,
           "--n-inner", str(n_inner), "--write-thin", str(write_thin),
           "--warmup", str(warmup),
           "--samples", str(samples), "--seed", str(seed)]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_variant_chains(variant_dir, burn_frac=0.2):
    chain_dirs = sorted(Path(variant_dir).glob("chain_*"))
    if not chain_dirs:
        raise FileNotFoundError(f"No chain_* directories under {variant_dir}")
    arrs = []
    for cd in chain_dirs:
        arr = load_chain(cd / "fort.7")
        b = int(burn_frac * len(arr))
        finite = np.isfinite(arr["lnP"][b:])
        arrs.append(arr[b:][finite])
    return arrs


def rhat_row(variant_arrs):
    out = {}
    for k in PARAMS:
        chains = [a[k].astype(np.float64) for a in variant_arrs]
        minlen = min(len(c) for c in chains)
        chains = np.array([c[-minlen:] for c in chains])
        chains = thin_chains(chains)
        out[k] = gelman_rubin(chains) if len(chains) > 1 else float("nan")
    return out


def ks_block(reid_pooled, cand):
    """Two-sample KS statistic + p-value of each shared global against the
    CANDEL posterior (unweighted, pooled over chains; frozen columns ->
    nan)."""
    lines = ["", "KS 2-sample vs CANDEL (unweighted):",
             f"{'param':>18s} {'KS':>10s} {'p_value':>12s}"]
    for k in SHARED:
        x = np.asarray(reid_pooled[k], dtype=np.float64)
        y = np.asarray(cand[k], dtype=np.float64)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if len(x) < 2 or len(y) < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
            lines.append(f"{k:>18s} {'nan':>10s} {'nan':>12s}")
            continue
        res = ks_2samp(x, y)
        lines.append(f"{k:>18s} {res.statistic:10.4f} {res.pvalue:12.3e}")
    return "\n".join(lines)


def pesce_reid_globals(galaxy, config, data):
    """Published Pesce/Reid disk point in Reid GLOBAL_NAMES convention, with
    the warp angles evaluated at the data-derived reid_r_ref.

    Reuses the tested Pesce->CANDEL conversion (velocity frame, D_A, the
    zero-radius warp intercepts) in pesce_globals, then applies the same
    CANDEL->Reid mapping as candel_to_reid (inclination flip, i-warp sign
    flip; PA carries over) and shifts the r=0 intercepts to reid_r_ref."""
    import math

    import pesce_globals  # local: pulls jax/astropy, only for --init pesce
    master = {"model": {"galaxies": {galaxy: load_galaxy_config(config,
                                                                galaxy)}}}
    point, missing = pesce_globals.paper_point(galaxy, master)
    if point is None:
        sys.exit(f"[ERROR] no published Pesce point for {galaxy} "
                 f"(missing {missing})")
    ex, ey = float(point.get("e_x", 0.0)), float(point.get("e_y", 0.0))
    r0 = {
        "H0": point["v_native"] / point["D_A"],
        "Mbh_1e7Msun": point["M_BH_1e7"],
        "Vsys_km_s": point["v_native"],
        "x0_mas": point["x0_mas"],
        "y0_mas": point["y0_mas"],
        "i0_deg": 180.0 - point["i0_r0_deg"],
        "di_dr_deg_mas": -point.get("di_dr_r0", 0.0),
        "d2i_dr2_deg_mas2": -point.get("d2i_dr2", 0.0),
        "PA_deg": point["Omega0_r0_deg"],
        "dPA_dr_deg_mas": point.get("dOmega_dr_r0", 0.0),
        "d2PA_dr2_deg_mas2": point.get("d2Omega_dr2", 0.0),
        "ecc": math.hypot(ex, ey),
        "peri_az_deg": (math.degrees(math.atan2(ey, ex)) % 360.0
                        if (ex or ey) else 0.0),
        "dperi_dr_deg_mas": point.get("dperiapsis_dr", 0.0),
        "Vcor_km_s": 0.0,
        "sigma_x_mas": point["sigma_x_mas"],
        "sigma_y_mas": point["sigma_y_mas"],
        "sigma_vsys_km_s": point["sigma_v_sys"],
        "sigma_vhv_km_s": point["sigma_v_hv"],
        "sigma_acc_km_s_yr": point["sigma_a"],
    }
    header, rows = parse_data_rows(data)
    r_ref = compute_reid_r_ref(rows, header, r0)
    out = dict(r0)
    out["i0_deg"] = (r0["i0_deg"] + r0["di_dr_deg_mas"] * r_ref
                     + r0["d2i_dr2_deg_mas2"] * r_ref * r_ref)
    out["PA_deg"] = (r0["PA_deg"] + r0["dPA_dr_deg_mas"] * r_ref
                     + r0["d2PA_dr2_deg_mas2"] * r_ref * r_ref)
    return out, r_ref


def write_pesce_control(template, out_path, glob):
    """Copy the control template but overwrite the 20 global value columns
    (lines 6-25) with the Pesce point; priors/steps/ranges are untouched, so
    run_gibbs_chains.sh sees an otherwise-identical control file."""
    lines = Path(template).read_text().splitlines()
    for i, name in enumerate(GLOBAL_NAMES, start=5):
        parts = lines[i].split("!", 1)[0].split()
        prior, post = float(parts[1]), float(parts[2])
        lines[i] = set_control_numbers(lines[i], [glob[name], prior, post])
    Path(out_path).write_text("\n".join(lines) + "\n")


def read_control_values(path):
    """Value column (lines 6-25) of a fit_disk_control.inp control file,
    keyed by GLOBAL_NAMES -- the actual chain starting point, whichever
    --init source produced it."""
    lines = Path(path).read_text().splitlines()
    return {name: float(lines[i].split("!", 1)[0].split()[0])
            for i, name in enumerate(GLOBAL_NAMES, start=5)}


def write_h0_range_control(template, out_path, lo, hi):
    """Copy the control template but overwrite line 3's Ho_low/Ho_high
    fields (the M-H strand seeding + prior window) with (lo, hi); itermax
    and num_walkers on the same line are untouched. Reid's Fortran further
    expands this by +-10 km/s for the actual hard prior bound
    (Ho_low_lim/Ho_high_lim in fit_disk_v24d_unblinded.f), so the true
    enforced range is [lo-10, hi+10], not exactly [lo, hi]."""
    lines = Path(template).read_text().splitlines()
    itermax, num_walkers = (int(x) for x in
                            lines[2].split("!", 1)[0].split()[:2])
    lines[2] = set_control_numbers(lines[2], [itermax, num_walkers, lo, hi])
    Path(out_path).write_text("\n".join(lines) + "\n")


def _parse_h0_range(spec):
    try:
        lo_s, hi_s = spec.split(",")
        lo, hi = float(lo_s), float(hi_s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--H0-range must be 'LOW,HIGH' (got {spec!r})")
    if not lo < hi:
        raise argparse.ArgumentTypeError(
            f"--H0-range LOW must be < HIGH (got {spec!r})")
    return (lo, hi)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--galaxy", default="NGC6323")
    p.add_argument("--variant", default="init", choices=["init", "init_qw"])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--control-template", type=Path, default=None,
                   help="fit_disk_control.inp with this galaxy's priors "
                        "(default: reid_control_<galaxy>.inp next to this "
                        "script)")
    p.add_argument("--init", choices=["config", "pesce"], default="config",
                   help="chain start point: 'config' uses the control "
                        "template's value column (CANDEL config globals in "
                        "Reid convention); 'pesce' overwrites it with the "
                        "published Pesce/Reid disk point "
                        "(pesce_disk_params.toml) mapped to Reid convention "
                        "at reid_r_ref (priors/steps unchanged)")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--candel", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--num-warmup", type=int, default=1_000_000)
    p.add_argument("--num-samples", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=47351937)
    p.add_argument("--n-inner", type=int, default=20,
                   help="per-spot latent sweeps per Gibbs iteration "
                        "(applies to the Gibbs variants only)")
    p.add_argument("--write-thin", type=int, default=10,
                   help="write only every Nth stored sample to fort.7/"
                        "fort.74 (default: 10); the "
                        "in-Fortran Ho-M bootstrap still sees every "
                        "stored sample, so this only shrinks the chain "
                        "files and the collect step's parse time")
    p.add_argument("--eta", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="reparameterize the BH mass as eta = "
                        "log10(Mbh/D_A) (CANDEL's mass coordinate) and "
                        "target a measure flat in (D_A, eta); applies to "
                        "all selected variants; fort.7 keeps the Mbh "
                        "column so post-processing is unchanged "
                        "(--no-eta for Reid's original flat-(H0,M) "
                        "sampling; the reid_original variant always "
                        "runs with eta off)")
    p.add_argument("--match-priors", choices=("T", "F", "orig"), default="T",
                   help="error-floor treatment forwarded to "
                        "run_gibbs_chains.sh, applied to all variants: "
                        "T=CANDEL Gaussian floor priors (SAMPLED, "
                        "Pesce-style, config_maser.toml [model.priors], "
                        "untruncated + 0.001 clamp); F=template floor lines "
                        "verbatim; orig=original Reid/Kuo published floors "
                        "FIXED (prior_unc=0, NOT sampled; reproduces Kuo 2015 "
                        "/ Reid 2013) (default: T)")
    p.add_argument("--H0-range", type=_parse_h0_range, default=None,
                   metavar="LOW,HIGH",
                   help="restrict the Ho sampling window: overwrites the "
                        "control file's 'Initial Ho strand range' line "
                        "(itermax/num_walkers on that line are preserved) "
                        "with LOW,HIGH. Reid's Fortran further expands this "
                        "by +-10 km/s for the actual hard prior bound "
                        "(Ho_low_lim=LOW-10, Ho_high_lim=HIGH+10 in "
                        "fit_disk_v24d_unblinded.f), so the true enforced "
                        "range is [LOW-10, HIGH+10], not exactly "
                        "[LOW, HIGH] (default: unset, control template's "
                        "own window)")
    p.add_argument("--reweight-da2", action="store_true",
                   help="importance-reweight the Reid chains by D_A^2 in the "
                        "corner/distance overlays, undoing the D_A^-2 prior "
                        "induced by "
                        "Reid's flat-H0 sampling (approximates CANDEL's flat "
                        "distance prior); R-hat/summaries stay unweighted")
    p.add_argument("--variants", default="all",
                   help="comma-separated subset of "
                        + ",".join(v[0] for v in VARIANTS)
                        + " (default: all)")
    p.add_argument("--skip-run", action="store_true",
                   help="Reuse existing chain outputs in --out-dir; only "
                        "recompute R-hat and the plots")
    p.add_argument("--chains-only", action="store_true",
                   help="Run the variant chains, print this variant's R-hat "
                        "and numpyro summary, and skip the combined table/"
                        "plots (those come from the --collect step); lets "
                        "one cluster job per variant share an --out-dir")
    args = p.parse_args(argv)

    if args.variants == "all":
        variants = VARIANTS
    else:
        want = args.variants.split(",")
        known = {v[0] for v in VARIANTS}
        unknown = sorted(set(want) - known)
        if unknown:
            sys.exit(f"[ERROR] unknown variants {unknown}; "
                     f"choose from {sorted(known)}")
        variants = [v for v in VARIANTS if v[0] in want]

    g = args.galaxy
    # combined-artifact filename tag: run configurations that change the
    # sampled/plotted data (but not the variant name) must not collide when
    # an --out-dir is reused (e.g. --collect on an existing directory).
    tag = (("_etaoff" if not args.eta else "")
           + ("_da2rw" if args.reweight_da2 else "")
           + ("_pesceinit" if args.init == "pesce" else "")
           + ("_flatfloor" if args.match_priors == "F" else "")
           + ("_origfloor" if args.match_priors == "orig" else "")
           + (f"_h0{args.H0_range[0]:g}_{args.H0_range[1]:g}"
              if args.H0_range else ""))
    control_template = args.control_template or HERE / f"reid_control_{g}.inp"
    data = args.data or ROOT / f"data/Megamaser/{g}_loader_reid.inp"
    suffix = "_qw" if args.variant == "init_qw" else ""
    candel = args.candel or (
        ROOT / "results/Megamaser"
        / f"{g}/{g}_blackjax_mcmc_rphi{suffix}_initreid.hdf5")

    if not Path(data).exists():
        # generated artifact (data/ is untracked); rebuild it in place.
        # Write-to-temp + atomic replace: concurrent per-variant cluster
        # jobs may all regenerate it, and the content is deterministic.
        print(f"[INFO] generating Reid data file: {data}")
        Path(data).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{data}.tmp{os.getpid()}"
        prepare_reid_data.main([g, "--out", tmp])
        os.replace(tmp, data)

    checks = [(control_template, "control template"), (data, "data file")]
    if not args.chains_only:
        checks.append((candel, "CANDEL posterior"))
    for pth, label in checks:
        if not Path(pth).exists():
            sys.exit(f"[ERROR] missing {label}: {pth}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    identity = run_identity(g, args)

    if args.init == "pesce":
        # reid_control_<g>_pesce.inp lives in --out-dir (not next to this
        # script) so config and pesce never share a control file; the
        # collect step (--skip-run, a separate job) reuses the one the
        # launch step already wrote instead of regenerating it.
        pesce_ctrl = args.out_dir / f"reid_control_{g}_pesce.inp"
        if not args.skip_run:
            # Rewrite the control template's value column with the Pesce
            # point. Write-to-temp + atomic replace: concurrent per-variant
            # cluster jobs sharing an --out-dir all regenerate the same
            # deterministic file.
            glob, r_ref = pesce_reid_globals(g, args.config, data)
            tmp = f"{pesce_ctrl}.tmp{os.getpid()}"
            write_pesce_control(control_template, tmp, glob)
            os.replace(tmp, pesce_ctrl)
            print(f"[INFO] Pesce init: wrote {pesce_ctrl} (reid_r_ref="
                  f"{r_ref:.6f} mas, H0={glob['H0']:.3f}, "
                  f"D={glob['Vsys_km_s'] / glob['H0']:.2f} Mpc)")
        control_template = pesce_ctrl

    if args.H0_range is not None:
        # Same write-to-temp + atomic-replace + reuse-on-skip-run pattern as
        # the Pesce control above, layered on top of whichever control_
        # template (config or Pesce) is currently in effect.
        lo, hi = args.H0_range
        h0_ctrl = args.out_dir / f"reid_control_{g}_h0_{lo:g}_{hi:g}.inp"
        if not args.skip_run:
            tmp = f"{h0_ctrl}.tmp{os.getpid()}"
            write_h0_range_control(control_template, tmp, lo, hi)
            os.replace(tmp, h0_ctrl)
            print(f"[INFO] Ho range override: wrote {h0_ctrl} (Ho strand "
                  f"range {lo:g}-{hi:g}; Reid's Fortran hard prior bound "
                  f"is [{lo - 10:g}, {hi + 10:g}])")
        control_template = h0_ctrl

    # Best-effort: never let a missing/unreadable control file (e.g. a
    # --skip-run collect job pointed at an --out-dir whose launch step
    # hasn't written reid_control_<g>_pesce.inp yet) break the rest of the
    # collect step's R-hat/summaries/plots.
    try:
        start_glob = read_control_values(control_template)
    except (OSError, KeyError, ValueError, IndexError) as exc:
        print(f"[WARN] could not read starting point from "
              f"{control_template}: {exc}")
    else:
        _banner(f"Starting point -- {identity}")
        print("\n".join(f"{k:>18s} {v:14.6g}" for k, v in start_glob.items()))

    if not args.skip_run:
        for name, gibbs, reflect, gcov, v_eta, _ in variants:
            eta = args.eta if v_eta is None else v_eta == "T"
            run_variant(control_template, data, args.out_dir / name,
                        gibbs, reflect, gcov, args.chains, args.num_warmup,
                        args.num_samples, args.seed, args.n_inner, eta,
                        args.write_thin, args.match_priors, g)

    variant_arrs = {name: load_variant_chains(args.out_dir / name)
                    for name, *_ in variants}

    def pooled(name):
        return {k: np.concatenate([a[k].astype(np.float64)
                                   for a in variant_arrs[name]])
                for k in SHARED}

    # CANDEL posterior in Reid globals: the reference for the per-variant KS
    # test and the overlays (not available in --chains-only mode).
    cand = None
    if not args.chains_only:
        cand = candel_to_reid(candel, g, args.config, args.variant, data)

    _banner(f"R-hat -- {identity}")
    header = f"{'variant':14s}" + "".join(f"{k:>16s}" for k in PARAMS)
    lines = [identity,
             f"Gelman-Rubin R-hat ({args.chains} chains/variant, "
             f"post 20% burn, thinned to <={MAX_RHAT_SAMPLES:,} samples):",
             header]
    for name, *_ in variants:
        rhat = rhat_row(variant_arrs[name])
        lines.append(f"{name:14s}"
                     + "".join(f"{rhat[k]:16.3f}" for k in PARAMS))

    # Distance: median with asymmetric 1sigma (16-84%) errors per variant,
    # in the published D = med +hi -lo Mpc style, pooled over all chains'
    # post-burn samples.
    lines.append("")
    lines.append("Distance D_Mpc (median, asymmetric 1sigma [16-84%]):")
    for name, *_ in variants:
        d = np.concatenate([a["D_Mpc"].astype(np.float64)
                            for a in variant_arrs[name]])
        p16, p50, p84 = np.percentile(d, [16.0, 50.0, 84.0])
        lines.append(f"  {name:22s} {p50:7.1f} "
                     f"+{p84 - p50:.1f} -{p50 - p16:.1f} Mpc")

    report = "\n".join(lines)
    print(report)

    if not args.chains_only:
        # combined artifact: only the collect step may write it, else
        # concurrent per-variant jobs would clobber each other
        rhat_txt = args.out_dir / f"{g}_gibbs_comparison_rhat{tag}.txt"
        rhat_txt.write_text(report + "\n")
        print(f"\nWrote {rhat_txt}")

    for name, *_ in variants:
        _banner(f"{name} -- {len(variant_arrs[name])} chains, post 20% burn "
                f"-- {identity}")
        block = (f"{identity}\n"
                 f"=== numpyro global summary: {name} "
                 f"({len(variant_arrs[name])} chains, post 20% burn) ===\n"
                 + variant_summary(variant_arrs[name]))
        if cand is not None:
            block += "\n" + ks_block(pooled(name), cand)
        print(block)
        summ_path = args.out_dir / f"{g}_{name}_global_summary{tag}.txt"
        summ_path.write_text(block + "\n")
        print(f"Wrote {summ_path}")

        chain_hist_path = args.out_dir / f"{g}_{name}_chains_distance{tag}.png"
        per_chain_distance_histogram(variant_arrs[name], chain_hist_path)
        print(f"Wrote {chain_hist_path}")

    if args.chains_only:
        print(f"\nChains + per-variant summaries done for: "
              f"{', '.join(v[0] for v in variants)}. Combined R-hat table "
              f"and plots come from the --collect step (--skip-run).")
        return

    datasets = [("CANDEL", "C0", cand)]
    for name, _, _, _, v_eta, color in variants:
        d = pooled(name)
        eta = args.eta if v_eta is None else v_eta == "T"
        # eta runs already target the flat-(D_A, eta) measure in-sampler
        w = d["D_Mpc"] ** 2 if args.reweight_da2 and not eta else None
        datasets.append((f"Reid ({name})", color, d, w))

    _banner("Plots")
    out_png = args.out_dir / f"{g}_gibbs_comparison_corner{tag}.png"
    overlay_three(datasets, out_png)
    print(f"Wrote {out_png}")

    out_hist = args.out_dir / f"{g}_gibbs_comparison_distance{tag}.png"
    overlay_distance_histogram(datasets, out_hist)
    print(f"Wrote {out_hist}")

    out_h0 = args.out_dir / f"{g}_gibbs_comparison_H0{tag}.png"
    overlay_distance_histogram(datasets, out_h0, param="H0")
    print(f"Wrote {out_h0}")


if __name__ == "__main__":
    raise SystemExit(main())
