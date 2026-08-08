#!/usr/bin/env python3
"""Analyse the megamaser MCMC sweep (warp off/on x init config/reid).

For each galaxy it loads whichever of the four variant chains exist under
``[io].root_output/<galaxy>/`` and answers three questions:

  (a) effect of the quadratic disk  -> warp on vs off: evidence (ln Z),
      the d2i_dr2 / d2Omega_dr2 posteriors (consistent with zero?), and the
      D_c shift.
  (b) effect of the starting point  -> init config vs reid at fixed disk model:
      do the posteriors agree (convergence / init-insensitivity)?
  (c) goodness of fit, Pesce vs me  -> the per-spot-marginalised logZ_2d
      of the MCMC point minus the Pesce/Reid point (>0 = the sampled fit is
      preferred).

Posteriors come from the HDF5; the goodness-of-fit table and ln Z are
scraped from the run log (they are printed, not saved structured), matched
to each chain via its ``saved samples to <hdf5>`` line.

A per-chain convergence verdict is printed from the chain's
``<chain>_summary.txt`` (single-chain split-R-hat and ESS), covering both the
globals and the per-spot ``r_ang``/``phi`` latents; a chain that passes R-hat
and ESS but has >1% divergent samples is flagged "suspect".

Comparisons (a) and (b) report both the distance (median +/- 1sigma, with the
shift quoted as a multiple of the quadrature-combined 1sigma) and the
per-spot-marginalised logP at the MCMC median. It also writes a getdist corner
overlay of the config-init vs reid-init posteriors per disk model,
``<galaxy>_init_compare[_qw].png`` under the galaxy directory (skip with
``--no-plots``).
"""
import re
import sys
import tomllib
from glob import glob
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "scripts" / "megamaser" / "config_maser.toml"
# Spot-table dataset whose results are analysed; the runners namespace
# [io].root_output by this name.
DATASET = "original_published"
WARP_COEFS = ("d2i_dr2", "d2Omega_dr2")
ROW_LABELS = ("Pesce/Reid", "config init", "MCMC median")
ORDER = [("nowarp", "config"), ("nowarp", "reid"),
         ("qw", "config"), ("qw", "reid")]
RHAT_WARN = 1.02


def root_output():
    with open(CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    rel = cfg.get("io", {}).get("root_output", "results/Maser")
    return ROOT / rel / DATASET


def pct(x):
    return np.percentile(x, [2.5, 16, 50, 84, 97.5])


def sigma(p):
    """1sigma half-width from the [2.5,16,50,84,97.5] percentile vector."""
    return 0.5 * (p[3] - p[1])


def shift_sigma(p_off, p_on):
    """Median shift in units of the quadrature-combined 1sigma.

    The two posteriors share the same data (nested models / different init),
    so they are correlated and this independent-quadrature combination is a
    conservative significance (it over-states the error on the difference).
    """
    s = np.hypot(sigma(p_off), sigma(p_on))
    return (p_on[2] - p_off[2]) / s if s > 0 else np.nan


def _pm(p):
    return f"{p[2]:.2f} +/- {sigma(p):.2f}"


def _mcmc_logZ(c):
    return c["rows"].get("MCMC median", {}).get("logZ_2d")


def load_chain(path):
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        # uniform_D_A chains sample D_A directly; fall back for legacy D_c
        # chains. Kept under the legacy "D_c" key (~2-3% from true D_c; the
        # sweep compares distance shifts, for which the difference is moot).
        s = f["samples"]
        dc = (s["D_A"] if "D_A" in s else s["D_c"])[...].astype(float)
        warp = {k: f["samples"][k][...].astype(float)
                for k in WARP_COEFS if k in f["samples"]}
        div = int(np.asarray(f["info"]["theta_is_divergent"]).sum()) \
            if "theta_is_divergent" in f["info"] else None
        theta = [t for t in str(a.get("theta_sites", "")).split(",") if t]
    return {
        "path": str(path),
        "theta_sites": theta,
        "warp_on": bool(a.get("use_quadratic_warp", False)),
        "init": str(a.get("init_strategy", "?")),
        "num_samples": int(a.get("num_samples", 0)),
        "num_draws": int(dc.size),
        "D_c": pct(dc),
        "warp": {k: pct(v) for k, v in warp.items()},
        "divergences": div,
    }


def find_log(galaxy_dir, h5path):
    """Latest MCMC/evidence logs matching this exact HDF5."""
    name = Path(h5path).name
    mcmc_hits = []
    evidence_hits = []
    log_files = glob(str(galaxy_dir / "logs" / "*.log"))
    log_files += glob(str(galaxy_dir / "logs" / "*.out"))
    for lg in log_files:
        txt = Path(lg).read_text(errors="ignore")
        if name not in txt:
            continue
        hit = (Path(lg).stat().st_mtime, txt)
        if "saved samples to" in txt:
            mcmc_hits.append(hit)
        if "chain:" in txt:
            evidence_hits.append(hit)
    parts = []
    if mcmc_hits:
        parts.append(max(mcmc_hits, key=lambda t: t[0])[1])
    if evidence_hits:
        parts.append(max(evidence_hits, key=lambda t: t[0])[1])
    if not parts:
        return None
    return "\n".join(parts)


_ROW_RE = re.compile(
    r"^\s*("
    + "|".join(re.escape(label) for label in ROW_LABELS)
    + r")\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)",
    re.M)
_HARM_RE = re.compile(r"ln Z \(harmonic\)\s*=\s*(-?\d+\.\d+)")


def parse_log(txt):
    if txt is None:
        return {}, None
    rows = {}
    for m in _ROW_RE.finditer(txt):
        label, logP, _dlogP, logZ, D_A, log_MBH = m.groups()
        rows[label] = {"logP_2d": float(logP), "logZ_2d": float(logZ),
                       "D_A": float(D_A), "log_MBH": float(log_MBH)}
    h = _HARM_RE.search(txt)
    return rows, (float(h.group(1)) if h else None)


_GLOBAL_ROW = re.compile(r"^\s*(\w+)((?:\s+-?\d+\.\d+){7})\s*$", re.M)
_NEFF_TRIPLE = re.compile(
    r"n_eff min/median/max\s*=\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)")
_RHAT_MAX = re.compile(r"r_hat max\s*=\s*([\d.]+)")


def _section(txt, header, stops):
    """The slice of txt from `header` up to the first following `stop`."""
    i = txt.find(header)
    if i < 0:
        return ""
    j = len(txt)
    for s in stops:
        k = txt.find(s, i + len(header))
        if k >= 0:
            j = min(j, k)
    return txt[i:j]


def parse_summary(h5path):
    """Convergence stats from the chain's <chain>_summary.txt, if present.

    Globals carry per-parameter n_eff/r_hat (last two columns of the Global
    Summary table); the r_ang/phi latent blocks carry n_eff min/median/max and
    r_hat max. Chains are single-chain, so r_hat is the split-R-hat.
    """
    sp = Path(h5path).with_suffix("").as_posix() + "_summary.txt"
    if not Path(sp).exists():
        return None
    txt = Path(sp).read_text(errors="ignore")
    out = {}
    g = _section(txt, "Global Summary", ["r_ang Summary", "phi Summary"])
    neff, rhat = [], []
    global_params = {}
    for m in _GLOBAL_ROW.finditer(g):
        name = m.group(1)
        nums = [float(x) for x in m.group(2).split()]
        neff.append(nums[5])
        rhat.append(nums[6])
        global_params[name] = {"min_neff": nums[5], "max_rhat": nums[6]}
    if neff:
        out["globals"] = {"min_neff": min(neff), "max_rhat": max(rhat)}
        out["global_params"] = global_params
    for name, hdr in (("r_ang", "r_ang Summary"), ("phi", "phi Summary")):
        s = _section(txt, hdr, ["phi Summary", "MCMC/Pesce", "Evidence"])
        nm, rm = _NEFF_TRIPLE.search(s), _RHAT_MAX.search(s)
        if nm and rm:
            out[name] = {"min_neff": float(nm.group(1)),
                         "max_rhat": float(rm.group(1))}
    return out or None


def _conv_metrics(c):
    conv = c.get("conv")
    if not conv:
        return {"status": "unknown", "max_rhat": None, "min_neff": None,
                "div_frac": None}
    vals = [conv[blk] for blk in ("globals", "r_ang", "phi") if blk in conv]
    if not vals:
        return {"status": "unknown", "max_rhat": None, "min_neff": None,
                "div_frac": None}
    max_rhat = max(v["max_rhat"] for v in vals)
    min_neff = min(v["min_neff"] for v in vals)
    ns = c.get("num_draws") or c.get("num_samples") or 0
    div_frac = (c.get("divergences") or 0) / ns if ns else None
    if max_rhat > 1.01 or min_neff < 400.0:
        status = "BAD"
    elif div_frac is not None and div_frac >= 0.01:
        status = "suspect"
    else:
        status = "OK"
    return {"status": status, "max_rhat": max_rhat, "min_neff": min_neff,
            "div_frac": div_frac}


def _convergence_line(c):
    """Per-chain verdict over globals + r_ang/phi latents."""
    conv = c.get("conv")
    if not conv:
        return "no summary.txt (convergence unknown)"
    parts, rhats, neffs, missing = [], [], [], []
    for blk in ("globals", "r_ang", "phi"):
        v = conv.get(blk)
        if v is None:
            missing.append(blk)
            continue
        rhats.append(v["max_rhat"])
        neffs.append(v["min_neff"])
        parts.append(f"{blk}: rhat_max={v['max_rhat']:.2f} "
                     f"ESS_min={v['min_neff']:.0f}")
    if not rhats:
        return "no convergence stats parsed"
    metrics = _conv_metrics(c)
    rhat_ok = metrics["max_rhat"] <= 1.01
    ess_ok = metrics["min_neff"] >= 400.0
    dfrac = metrics["div_frac"] or 0.0
    if not rhat_ok or not ess_ok:
        verdict = "NOT CONVERGED"
        if not rhat_ok:
            verdict += f" (R-hat up to {max(rhats):.2f})"
        if not ess_ok:
            verdict += f" (ESS down to {min(neffs):.0f})"
    elif dfrac >= 0.01:
        verdict = f"converged on R-hat/ESS but {dfrac:.1%} divergent (suspect)"
    else:
        verdict = "CONVERGED"
    tag = f" [{','.join(missing)} missing]" if missing else ""
    return " | ".join(parts) + tag + "  -> " + verdict


def _ci(p):
    """median [16, 84] from a percentile vector [2.5,16,50,84,97.5]."""
    return f"{p[2]:7.2f} [{p[1]:.2f}, {p[3]:.2f}]"


def _fnum(x, w, sign=False):
    """Right-justified float in width w, or blanks if None."""
    if x is None:
        return " " * w
    return (f"{x:+.2f}" if sign else f"{x:.2f}").rjust(w)


def _nonzero(p):
    """significance that a warp coefficient differs from zero."""
    if p[1] <= 0.0 <= p[3]:
        return "0 within 1sigma"
    if p[0] <= 0.0 <= p[4]:
        return "0 within 2sigma"
    return "EXCLUDES 0 (>2sigma)"


def variant_key(c):
    return ("qw" if c["warp_on"] else "nowarp", c["init"])


def _plot_keys(c):
    """Same global set as the single-chain corner: theta_sites (+ log_MBH)."""
    keys = list(c["theta_sites"])
    if "log_MBH" not in keys:
        idx = keys.index("eta") + 1 if "eta" in keys else len(keys)
        keys.insert(idx, "log_MBH")
    return keys


def plot_init_comparison(galaxy, gdir, chains):
    """Overlay the config-init and reid-init posteriors per disk model."""
    import matplotlib
    matplotlib.use("Agg")
    from candel.plotting.corner import plot_corner_from_hdf5

    made = []
    for warp, tag in (("nowarp", ""), ("qw", "_qw")):
        a, b = chains.get((warp, "config")), chains.get((warp, "reid"))
        if a is None or b is None:
            continue
        out = gdir / f"{galaxy}_init_compare{tag}.png"
        try:
            plot_corner_from_hdf5(
                [a["path"], b["path"]], keys=_plot_keys(a),
                labels=["config init", "reid init"],
                show_fig=False, filename=str(out))
            made.append(str(out))
        except Exception as exc:
            print(f"  [plot] skipped {warp} init overlay: {exc}")
    return made


def analyse_galaxy(galaxy, gdir, make_plots=True):
    pattern = f"{galaxy}_blackjax_mcmc_rphi*_init*.hdf5"
    files = sorted(glob(str(gdir / pattern)))
    if not files:
        return
    chains = {}
    for fp in files:
        c = load_chain(fp)
        c["rows"], c["lnZ_harmonic"] = parse_log(find_log(gdir, fp))
        c["conv"] = parse_summary(fp)
        chains[variant_key(c)] = c

    rule = "=" * 78
    print(f"\n{rule}\n{galaxy}: {len(chains)}/4 variants present\n{rule}")
    hdr = (f"{'variant':14} {'D_c [68% CI] (Mpc)':26} {'div':>5}  "
           f"{'logZ_2d(MCMC)':>13} {'logZ_2d(Pesce)':>14} "
           f"{'fit-Pesce':>9} {'lnZ_harm':>9}")
    print(hdr)
    print("-" * len(hdr))
    for key in ORDER:
        c = chains.get(key)
        if c is None:
            continue
        mcmc = c["rows"].get("MCMC median", {})
        pesce = c["rows"].get("Pesce/Reid", {})
        lz_m = mcmc.get("logZ_2d")
        lz_p = pesce.get("logZ_2d")
        dfit = (
            lz_m - lz_p
            if lz_m is not None and lz_p is not None else None)
        div = "" if c["divergences"] is None else str(c["divergences"])
        print(f"{key[0]+'/'+key[1]:14} {_ci(c['D_c']):26} {div:>5}  "
              f"{_fnum(lz_m, 13)} {_fnum(lz_p, 14)} "
              f"{_fnum(dfit, 9, sign=True)} "
              f"{_fnum(c['lnZ_harmonic'], 9)}")

    # Convergence.
    print("\nConvergence (single-chain split-R-hat & ESS; "
          "latents are the per-spot r_ang/phi)")
    for key in ORDER:
        c = chains.get(key)
        if c is None:
            continue
        print(f"  {key[0]+'/'+key[1]:14}: {_convergence_line(c)}")

    # (a) quadratic disk.
    print("\n(a) Effect of the quadratic disk")
    any_qw = False
    for init in ("config", "reid"):
        off, on = chains.get(("nowarp", init)), chains.get(("qw", init))
        if on is None:
            continue
        any_qw = True
        coefs = "; ".join(
            f"{k}: {_nonzero(p)}" for k, p in on["warp"].items())
        print(f"  init={init}: warp coefficients -> {coefs or '(none saved)'}")
        if off is not None:
            print(f"           D_c (Mpc): {_pm(off['D_c'])} -> "
                  f"{_pm(on['D_c'])}"
                  f"  | shift {on['D_c'][2] - off['D_c'][2]:+.2f} "
                  f"({shift_sigma(off['D_c'], on['D_c']):+.1f} sigma)")
            lz_off, lz_on = _mcmc_logZ(off), _mcmc_logZ(on)
            if lz_off is not None and lz_on is not None:
                favours = (
                    "favours warp" if lz_on > lz_off
                    else "favours circular")
                print(f"           marg. logP (MCMC median): "
                      f"{lz_off:.2f} -> {lz_on:.2f}  | Delta "
                      f"{lz_on - lz_off:+.2f} "
                      f"({favours})")
            if (on["lnZ_harmonic"] is not None
                    and off["lnZ_harmonic"] is not None):
                dz = on["lnZ_harmonic"] - off["lnZ_harmonic"]
                print(f"           ln Z(qw) - ln Z(nowarp) = {dz:+.2f} "
                      f"({'favours warp' if dz > 0 else 'favours circular'})")
            else:
                print("           ln Z unavailable "
                      "(run submit_sweep.sh --evidence on a GPU)")
    if not any_qw:
        print("  no quadratic-warp chain present yet.")

    # (b) starting point.
    print("\n(b) Effect of the starting point (config vs reid)")
    any_init = False
    for warp in ("nowarp", "qw"):
        a, b = chains.get((warp, "config")), chains.get((warp, "reid"))
        if a is None or b is None:
            continue
        any_init = True
        da, db = a["D_c"], b["D_c"]
        overlap = not (da[3] < db[1] or db[3] < da[1])
        verdict = (
            "consistent (1sigma overlap)" if overlap
            else "DISAGREE")
        print(f"  {warp}: D_c config={_ci(da)}  reid={_ci(db)}  "
              f"| shift {db[2] - da[2]:+.2f} "
              f"({shift_sigma(da, db):+.1f} sigma) "
              f"-> {verdict}")
        lz_a, lz_b = _mcmc_logZ(a), _mcmc_logZ(b)
        if lz_a is not None and lz_b is not None:
            print(f"         marg. logP (MCMC median): config={lz_a:.2f}  "
                  f"reid={lz_b:.2f}  | Delta {lz_b - lz_a:+.2f}")
    if not any_init:
        print("  need both config and reid for a fixed disk model.")

    # (c) goodness of fit vs Pesce.
    print("\n(c) Goodness of fit: my result vs Pesce/Reid "
          "(per-spot-marginalised logZ_2d)")
    shown = False
    for key in ORDER:
        c = chains.get(key)
        if (c is None or "MCMC median" not in c["rows"]
                or "Pesce/Reid" not in c["rows"]):
            continue
        shown = True
        mcmc, pesce = c["rows"]["MCMC median"], c["rows"]["Pesce/Reid"]
        d = mcmc["logZ_2d"] - pesce["logZ_2d"]
        print(f"  {key[0]+'/'+key[1]:14}: "
              f"logZ_2d  mine={mcmc['logZ_2d']:.2f}  "
              f"Pesce={pesce['logZ_2d']:.2f}  ->  {d:+.2f} "
              f"({'mine preferred' if d > 0 else 'Pesce preferred'}); "
              f"D_A mine={mcmc['D_A']:.1f}  Pesce={pesce['D_A']:.1f} Mpc")
    if not shown:
        print("  no comparison table found in logs "
              "(was the run --compare-reid?).")

    if make_plots:
        made = plot_init_comparison(galaxy, gdir, chains)
        for p in made:
            print(f"\n  init-comparison corner saved: {p}")
    return chains


def _div_frac(c):
    d = _conv_metrics(c)["div_frac"]
    return None if d is None else 100.0 * d


def _status(c):
    return _conv_metrics(c)["status"]


def _max_rhat(c):
    return _conv_metrics(c)["max_rhat"]


def _min_ess(c):
    return _conv_metrics(c)["min_neff"]


def _print_final_tables(results):
    rule = "=" * 100
    print(f"\n{rule}\nSUMMARY — chain distances and convergence\n{rule}")
    hdr = (f"{'Galaxy':<14} {'variant':<14} {'D_c [68% CI]':<25} "
           f"{'div%':>6} {'max Rhat':>8} {'min ESS':>8} {'status':>8}")
    print(hdr)
    print("-" * len(hdr))
    for galaxy, chains in results:
        for key in ORDER:
            c = chains.get(key)
            if c is None:
                continue
            div = _div_frac(c)
            max_rhat = _max_rhat(c)
            min_ess = _min_ess(c)
            print(f"{galaxy:<14} {key[0]+'/'+key[1]:<14} "
                  f"{_ci(c['D_c']):<25} "
                  f"{_fnum(div, 6)} {_fnum(max_rhat, 8)} "
                  f"{_fnum(min_ess, 8)} {_status(c):>8}")

    print(f"\n{rule}\nSUMMARY — quadratic-warp distance shifts\n{rule}")
    hdr = (f"{'Galaxy':<14} {'init':<7} {'nowarp D_c':<29} "
           f"{'QW D_c':<29} {'shift':>8} {'sigma':>7} "
           f"{'Delta logZ_2d':>13}")
    print(hdr)
    print("-" * len(hdr))
    for galaxy, chains in results:
        for init in ("config", "reid"):
            off, on = chains.get(("nowarp", init)), chains.get(("qw", init))
            if off is None or on is None:
                continue
            lz_off, lz_on = _mcmc_logZ(off), _mcmc_logZ(on)
            dlz = (
                lz_on - lz_off
                if lz_off is not None and lz_on is not None else None)
            dlz_txt = f"{dlz:+.2f}" if dlz is not None else "--"
            print(f"{galaxy:<14} {init:<7} {_ci(off['D_c']):<29} "
                  f"{_ci(on['D_c']):<29} "
                  f"{on['D_c'][2] - off['D_c'][2]:+8.2f} "
                  f"{shift_sigma(off['D_c'], on['D_c']):+7.1f} "
                  f"{dlz_txt:>13}")

    rows = []
    for galaxy, chains in results:
        for key in ORDER:
            c = chains.get(key)
            if c is None or not c.get("conv"):
                continue
            variant = key[0] + "/" + key[1]
            for name, stats in c["conv"].get("global_params", {}).items():
                if stats["max_rhat"] > RHAT_WARN:
                    rows.append((galaxy, variant, "global", name,
                                 stats["max_rhat"], stats["min_neff"]))
            for block in ("r_ang", "phi"):
                stats = c["conv"].get(block)
                if stats and stats["max_rhat"] > RHAT_WARN:
                    rows.append((galaxy, variant, block, "block max",
                                 stats["max_rhat"], stats["min_neff"]))

    print(f"\n{rule}\nSUMMARY — R-hat > {RHAT_WARN:.2f}\n{rule}")
    if not rows:
        print("No parameters or latent blocks exceed the threshold.")
        return
    hdr = (f"{'Galaxy':<14} {'variant':<14} {'block':<8} "
           f"{'item':<18} {'R-hat':>8} {'ESS':>8}")
    print(hdr)
    print("-" * len(hdr))
    for galaxy, variant, block, name, rhat, ess in rows:
        print(f"{galaxy:<14} {variant:<14} {block:<8} {name:<18} "
              f"{rhat:8.3f} {ess:8.0f}")


def main(argv):
    make_plots = "--no-plots" not in argv
    argv = [a for a in argv if a != "--no-plots"]
    out = root_output()
    galaxies = argv or sorted(
        p.name for p in out.iterdir()
        if p.is_dir()
        and glob(str(p / f"{p.name}_blackjax_mcmc_rphi*_init*.hdf5")))
    if not galaxies:
        print(f"No sweep outputs under {out}")
        return
    results = []
    for g in galaxies:
        chains = analyse_galaxy(g, out / g, make_plots=make_plots)
        if chains:
            results.append((g, chains))
    if results:
        _print_final_tables(results)


if __name__ == "__main__":
    main(sys.argv[1:])
