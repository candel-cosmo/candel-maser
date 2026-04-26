"""
Mode 1 phi-marginal convergence test.

For each galaxy, build the unified phi grid at several (n_phi_hv_high,
n_phi_hv_low, n_phi_sys) settings and compare to a full-2π brute-force
reference evaluated on a single uniform grid per spot type. This tells
us (a) whether the config sub-range grids have enough density, and
(b) whether the sub-range restrictions miss any likelihood mass outside
the configured ranges (full-2π catches leakage).

Additionally, at every (galaxy, grid, r_ang-scale) combination, compare
the AD gradient of the production sum-log-L against the same quantity
on the full-2π reference, split into:
  * summed gradient w.r.t. the globals (GRAD_PARAMS_BASE, galaxy-
    extended when applicable); and
  * per-spot gradient w.r.t. r_ang (length-N_spots vector since the
    cross-spot coupling is zero, one reverse-mode pass).

Usage:
    python scripts/megamaser/convergence_phi_marginal.py [--galaxies ...]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import tomli

from candel.model.maser_convergence import (
    bruteforce_ll_mode1, build_model, ensure_grad_sample, extend_grad_params,
    grad_diff_report, grad_mode1_production, grad_mode1_reference,
    resolve_grid_for_galaxy, vector_diff_report,
)


def _phys_from_init(model, galaxies_cfg, galaxy):
    """Build (phys_args, phys_kw, diag) from the config init block by
    delegating to MaserDiskModel.phys_from_sample."""
    init = galaxies_cfg[galaxy]["init"]
    sample = {k: np.asarray(v) for k, v in init.items()}
    return model.phys_from_sample(sample)

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
ALL_GALAXIES = ["CGCG074-064", "NGC5765b", "NGC6264", "NGC6323",
                "UGC3789", "NGC4258"]

# Mode 1 is the production case for NGC4258, which needs a denser phi
# grid than the Mode 2 default. Anchor the phi-marginal test at this
# setting (rather than the config default) so the sweep brackets the
# NGC4258 regime.
PHI_MARGINAL_ANCHOR = (2001, 501, 3001)

# Extra test grids (n_phi_hv_high, n_phi_hv_low, n_phi_sys) around the
# anchor. Always include the config default and the anchor.
EXTRA_GRIDS = [
    (3001,  501,   4501),
    (5001,  1001,  5001),
    (5001,  2501,  5001),
    (10001, 2001,  10001),
]


def build_test_grids(galaxy_default):
    """Build the convergence test list: per-galaxy production default +
    anchor + EXTRA, deduplicated and sorted ascending (tightest last)."""
    d = (galaxy_default["n_hv_high"], galaxy_default["n_hv_low"],
         galaxy_default["n_sys"])
    grids = set(EXTRA_GRIDS) | {d, PHI_MARGINAL_ANCHOR}
    return sorted(grids, key=lambda g: (g[0], g[2], g[1]))

# r_ang scales: sample r_ang at the data-driven estimate × scale.
R_SCALES = [0.5, 1.0, 2.0]

# Multiplier used by the fiducial driver diagnostic: bump n_hv_high or
# n_hv_low (one at a time) by this factor to see which dimension is
# limiting accuracy at the production grid.
DRIVER_FACTOR = 2


def per_type_logl(ll_per_spot, model):
    return dict(
        sys=float(jnp.sum(ll_per_spot[model._idx_sys])),
        red=float(jnp.sum(ll_per_spot[model._idx_red])),
        blue=float(jnp.sum(ll_per_spot[model._idx_blue])),
        total=float(jnp.sum(ll_per_spot)))


def per_category_r_grad_diff(grad_test_r, grad_ref_r, model):
    """Split the per-spot r_ang gradient diff into sys/red/blue."""
    out = {}
    for key, idx in (("sys", model._idx_sys),
                     ("red", model._idx_red),
                     ("blue", model._idx_blue)):
        idx_np = np.asarray(idx)
        if idx_np.size == 0:
            out[key] = dict(max_rel=0.0, max_abs=0.0)
            continue
        out[key] = vector_diff_report(
            np.asarray(grad_test_r)[idx_np],
            np.asarray(grad_ref_r)[idx_np])
    return out


def worst_global_param(per_param):
    """Return (param_name, rel_diff) with the largest rel diff, or
    ('-', 0.0) if empty."""
    if not per_param:
        return ("-", 0.0)
    k = max(per_param, key=lambda kk: per_param[kk]["rel_diff"])
    return (k, per_param[k]["rel_diff"])


def eval_grid_logl(model, master_cfg, galaxy, n_high, n_low, n_sys,
                    r_ang, phys_args, phys_kw, ref_cfg):
    """Build a model at (n_high, n_low, n_sys), evaluate per-type logL
    at fixed r_ang, return the per_type_logl dict and the test model."""
    m_t = build_model(
        galaxy, master_cfg, mode="mode1",
        n_phi_hv_high=n_high, n_phi_hv_low=n_low,
        n_phi_sys=n_sys)
    spot_groups = []
    if m_t._n_sys > 0:
        spot_groups.append(
            ("sys", m_t._idx_sys, r_ang[m_t._idx_sys], None))
    if m_t._n_red > 0:
        spot_groups.append(
            ("red", m_t._idx_red, r_ang[m_t._idx_red], None))
    if m_t._n_blue > 0:
        spot_groups.append(
            ("blue", m_t._idx_blue, r_ang[m_t._idx_blue], None))
    ll = m_t._eval_phi_marginal(
        spot_groups, phys_args, phys_kw,
        spot_batch=int(ref_cfg["spot_batch"]))
    return per_type_logl(ll, m_t), m_t


def estimate_r_ang(model, D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv):
    r_est, _, _, _ = model._closed_form_seeds(
        D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
    return np.asarray(r_est)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxies", nargs="+", default=ALL_GALAXIES)
    parser.add_argument(
        "--no-grad", action="store_true",
        help="Skip the AD-gradient check (log-L convergence only).")
    parser.add_argument(
        "--grad-rtol-globals", type=float, default=1e-3,
        help="Pass tolerance on max rel. diff of summed global gradients "
             "(default: 1e-3).")
    parser.add_argument(
        "--grad-rtol-r", type=float, default=1e-3,
        help="Pass tolerance on max rel. diff of per-spot r_ang "
             "gradients (default: 1e-3).")
    parser.add_argument(
        "--driver-factor", type=int, default=DRIVER_FACTOR,
        help="Multiplier used in the fiducial driver diagnostic — "
             "n_hv_high and n_hv_low are each bumped by this factor "
             "(one at a time) to see which limits the fiducial grid "
             "(default: 2).")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_enable_x64", True)
    print(f"JAX platform: {jax.default_backend()}, precision: float64",
          flush=True)

    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)
    galaxies_cfg = master_cfg["model"]["galaxies"]
    ref_cfg = master_cfg["convergence"]["mode1_reference"]
    grad_ref_cfg = master_cfg["convergence"].get(
        "mode1_gradient_reference", ref_cfg)

    do_grad = not args.no_grad

    print("=" * 80)
    print("Mode 1 phi-marginal convergence")
    print(f"Reference (ll): full-2π uniform grid, "
          f"n_phi = {ref_cfg['n_phi']}")
    if do_grad:
        print(f"Reference (grad): full-2π uniform grid, "
              f"n_phi = {grad_ref_cfg['n_phi']}, "
              f"spot_batch = {grad_ref_cfg['spot_batch']}, "
              f"jax.checkpoint per spot-batch")
    print(f"Anchor: {PHI_MARGINAL_ANCHOR}, EXTRA_GRIDS={EXTRA_GRIDS}")
    print(f"r_ang scales: {R_SCALES}")
    print("=" * 80)

    summary = []
    galaxy_grids_map = {}
    for galaxy in args.galaxies:
        print(f"\n{'─' * 70}")
        print(f"Galaxy: {galaxy}")
        print(f"{'─' * 70}")

        # Per-galaxy production grid (per-galaxy block → _mode1 fallback
        # → generic [model]). NGC4258 carries 20001/4001/20001, the
        # other galaxies fall to _mode1 (2001/501/3001).
        galaxy_default = resolve_grid_for_galaxy(
            master_cfg, galaxy, "mode1")
        galaxy_grids = build_test_grids(galaxy_default)
        galaxy_grids_map[galaxy] = galaxy_grids
        print(f"  production φ grid: "
              f"({galaxy_default['n_hv_high']}, "
              f"{galaxy_default['n_hv_low']}, "
              f"{galaxy_default['n_sys']})")
        print(f"  test grids: {galaxy_grids}")

        # Build model once with Mode 1 forced on. Defaults used; we
        # override per-iteration via rebuild for the test grids.
        model = build_model(galaxy, master_cfg, mode="mode1")
        phys_args, phys_kw, diag = _phys_from_init(
            model, galaxies_cfg, galaxy)
        D_A, M_BH, v_sys = diag["D_A"], diag["M_BH"], diag["v_sys"]
        i0 = phys_args[8]
        sigma_a_floor2 = phys_args[16]
        var_v_hv = phys_args[15]
        r_est = estimate_r_ang(model, D_A, M_BH, v_sys,
                                sigma_a_floor2, i0, var_v_hv)
        print(f"  D_A={D_A:.2f} Mpc, n_spots={model.n_spots}")

        _r_ang_lo, _r_ang_hi = model.r_ang_range(D_A)

        if do_grad:
            sample = ensure_grad_sample(model, galaxies_cfg[galaxy]["init"])
            grad_param_keys = extend_grad_params(model, sample)
        else:
            sample = None
            grad_param_keys = None

        for scale in R_SCALES:
            r_ang = jnp.clip(
                jnp.asarray(r_est * scale), _r_ang_lo, _r_ang_hi)
            med_r = float(jnp.median(r_ang))
            print(f"\n  r_ang scale = {scale:.2f}x  "
                  f"(median r = {med_r:.4f} mas)")

            # Reference: single full-2π uniform grid per type (log-L).
            ref = bruteforce_ll_mode1(
                model, phys_args, phys_kw, r_ang, ref_cfg)
            print(f"    full-2π ref (ll): sys={ref['sys']:.4f}, "
                  f"red={ref['red']:.4f}, blue={ref['blue']:.4f}, "
                  f"total={ref['total']:.4f}")

            # Reference gradient (computed once per scale and reused).
            if do_grad:
                print(f"    computing full-2π ref gradient...", flush=True)
                grad_ref_glob, grad_ref_r = grad_mode1_reference(
                    model, sample, r_ang, grad_ref_cfg)
            else:
                grad_ref_glob = None
                grad_ref_r = None

            # First pass: evaluate per-type logL (and gradients if
            # enabled) at every test grid. Print three narrow tables
            # afterwards so each metric stays readable.
            grid_results = {}
            grid_grad = {}  # (n_high,n_low,n_sys) -> (rep_g, rep_r_per)
            for n_high, n_low, n_sys in galaxy_grids:
                per_t, m_t = eval_grid_logl(
                    model, master_cfg, galaxy,
                    n_high, n_low, n_sys,
                    r_ang, phys_args, phys_kw, ref_cfg)
                grid_results[(n_high, n_low, n_sys)] = per_t

                d_tot = per_t["total"] - ref["total"]
                d_sys = per_t["sys"] - ref["sys"]
                d_red = per_t["red"] - ref["red"]
                d_blue = per_t["blue"] - ref["blue"]

                row = dict(
                    galaxy=galaxy, scale=scale,
                    n_high=n_high, n_low=n_low, n_sys=n_sys,
                    n_spots=model.n_spots,
                    d_total=d_tot, d_sys=d_sys, d_red=d_red,
                    d_blue=d_blue)

                if do_grad:
                    grad_test_glob, grad_test_r = grad_mode1_production(
                        m_t, sample, r_ang)
                    rep_g = grad_diff_report(
                        grad_test_glob, grad_ref_glob, grad_param_keys)
                    rep_r_per = per_category_r_grad_diff(
                        grad_test_r, grad_ref_r, model)
                    grid_grad[(n_high, n_low, n_sys)] = (rep_g, rep_r_per)
                    row["rel_dgrad_globals"] = rep_g["max_rel"]
                    row["abs_dgrad_globals"] = rep_g["max_abs"]
                    for k in ("sys", "red", "blue"):
                        row[f"rel_dgrad_r_{k}"] = rep_r_per[k]["max_rel"]
                        row[f"abs_dgrad_r_{k}"] = rep_r_per[k]["max_abs"]
                    row["rel_dgrad_r"] = max(
                        rep_r_per[k]["max_rel"]
                        for k in ("sys", "red", "blue"))

                summary.append(row)

            # Table 1 — Δ logL per category.
            print(f"\n    Δ logL per category (production − full-2π ref):")
            hdr1 = (f"      {'n_hv_high':>10} {'n_hv_low':>9} "
                    f"{'n_sys':>7}  {'Δ total':>11}  "
                    f"{'Δ sys':>9}  {'Δ red':>9}  {'Δ blue':>9}")
            print(hdr1)
            for n_high, n_low, n_sys in galaxy_grids:
                per_t = grid_results[(n_high, n_low, n_sys)]
                d_tot = per_t["total"] - ref["total"]
                d_sys = per_t["sys"] - ref["sys"]
                d_red = per_t["red"] - ref["red"]
                d_blue = per_t["blue"] - ref["blue"]
                print(f"      {n_high:>10d} {n_low:>9d} {n_sys:>7d}  "
                      f"{d_tot:+11.4f}  {d_sys:+9.4f}  "
                      f"{d_red:+9.4f}  {d_blue:+9.4f}")

            if do_grad:
                # Table 2 — rel. diff of summed gradient w.r.t. globals.
                print(f"\n    rel. diff of ∇globals (max over "
                      f"{len(grad_param_keys)} params, plus worst param):")
                hdr2 = (f"      {'n_hv_high':>10} {'n_hv_low':>9} "
                        f"{'n_sys':>7}  {'max rel':>10}  "
                        f"{'max abs':>10}  worst param (rel)")
                print(hdr2)
                for n_high, n_low, n_sys in galaxy_grids:
                    rep_g, _ = grid_grad[(n_high, n_low, n_sys)]
                    wname, wrel = worst_global_param(rep_g["per_param"])
                    print(f"      {n_high:>10d} {n_low:>9d} "
                          f"{n_sys:>7d}  {rep_g['max_rel']:10.3e}  "
                          f"{rep_g['max_abs']:10.3e}  "
                          f"{wname} ({wrel:.2e})")

                # Table 3 — rel. diff of per-spot ∇r_ang per category.
                print(f"\n    rel. diff of ∇r_ang per spot category "
                      f"(max over spots in category):")
                hdr3 = (f"      {'n_hv_high':>10} {'n_hv_low':>9} "
                        f"{'n_sys':>7}  {'rel∇r-sys':>11}  "
                        f"{'rel∇r-red':>11}  {'rel∇r-blue':>11}")
                print(hdr3)
                for n_high, n_low, n_sys in galaxy_grids:
                    _, rep_r_per = grid_grad[(n_high, n_low, n_sys)]
                    print(f"      {n_high:>10d} {n_low:>9d} "
                          f"{n_sys:>7d}  "
                          f"{rep_r_per['sys']['max_rel']:11.3e}  "
                          f"{rep_r_per['red']['max_rel']:11.3e}  "
                          f"{rep_r_per['blue']['max_rel']:11.3e}")

            # Check for self-convergence between last two grids.
            g1 = galaxy_grids[-2]
            g2 = galaxy_grids[-1]
            d_conv = abs(grid_results[g2]["total"]
                         - grid_results[g1]["total"])
            print(f"\n    |Δ(last two grids)| = {d_conv:.4f} nats")

            # ----------------------------------------------------------
            # Driver diagnostic at the fiducial: bump n_hv_high or
            # n_hv_low one at a time and report which gives the larger
            # |Δ_total| reduction. Same r_ang, same reference.
            # ----------------------------------------------------------
            n_h_d = galaxy_default["n_hv_high"]
            n_l_d = galaxy_default["n_hv_low"]
            n_s_d = galaxy_default["n_sys"]
            fid_key = (n_h_d, n_l_d, n_s_d)
            fid = grid_results[fid_key]
            F = int(args.driver_factor)
            drv_grids = [
                ("bump n_hv_high", (n_h_d * F, n_l_d, n_s_d)),
                ("bump n_hv_low ", (n_h_d, n_l_d * F, n_s_d)),
            ]
            print(f"\n    Driver diagnostic at fiducial "
                  f"({n_h_d}, {n_l_d}, {n_s_d}) — factor ×{F}:")
            print(f"      {'label':<16} {'grid':<24} "
                  f"{'Δ total':>11}  {'Δ red':>9}  {'Δ blue':>9}  "
                  f"{'|Δtot| reduction':>17}")
            print(f"      {'fiducial':<16} "
                  f"{'(' + str(n_h_d) + ', ' + str(n_l_d) + ', ' + str(n_s_d) + ')':<24} "
                  f"{fid['total']-ref['total']:+11.4f}  "
                  f"{fid['red']-ref['red']:+9.4f}  "
                  f"{fid['blue']-ref['blue']:+9.4f}  "
                  f"{'-':>17}")
            fid_abs = abs(fid["total"] - ref["total"])
            reductions = {}
            for label, grid in drv_grids:
                if grid in grid_results:
                    per_t = grid_results[grid]
                else:
                    per_t, _ = eval_grid_logl(
                        model, master_cfg, galaxy,
                        grid[0], grid[1], grid[2],
                        r_ang, phys_args, phys_kw, ref_cfg)
                d_tot = per_t["total"] - ref["total"]
                d_red = per_t["red"] - ref["red"]
                d_blue = per_t["blue"] - ref["blue"]
                impr = fid_abs - abs(d_tot)
                reductions[label] = impr
                gtag = f"({grid[0]}, {grid[1]}, {grid[2]})"
                print(f"      {label:<16} {gtag:<24} "
                      f"{d_tot:+11.4f}  {d_red:+9.4f}  "
                      f"{d_blue:+9.4f}  {impr:+17.4f}")
            # Pick the bigger absolute reduction (works whether
            # bumping reduces or barely changes |Δ|).
            best = max(reductions, key=lambda k: reductions[k])
            other = next(k for k in reductions if k != best)
            ratio = (reductions[best]
                     / max(abs(reductions[other]), 1e-30))
            driver = "n_hv_high" if "high" in best else "n_hv_low"
            print(f"      → DRIVER: {driver}  "
                  f"(reduction ratio {best.strip()}/{other.strip()} = "
                  f"{ratio:+.2f})")

    # --------------------------------------------------------------
    # Summary tables — one per metric to match the per-(galaxy,scale)
    # layout above and keep each table readable.
    # --------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("SUMMARY — Δ logL per category (production − full-2π reference)")
    print(f"{'=' * 100}")
    h1 = (f"{'Galaxy':<14} {'scale':>5} "
          f"{'n_hv_high':>10} {'n_hv_low':>9} {'n_sys':>7}  "
          f"{'Δ total':>10}  {'Δ sys':>9}  "
          f"{'Δ red':>9}  {'Δ blue':>9}")
    print(h1)
    print("-" * len(h1))
    for r in summary:
        print(f"{r['galaxy']:<14} {r['scale']:>5.2f} "
              f"{r['n_high']:>10d} {r['n_low']:>9d} "
              f"{r['n_sys']:>7d}  {r['d_total']:+10.4f}  "
              f"{r['d_sys']:+9.4f}  {r['d_red']:+9.4f}  "
              f"{r['d_blue']:+9.4f}")

    if do_grad:
        print(f"\n{'=' * 100}")
        print("SUMMARY — rel. diff of ∇globals "
              "(max over global params)")
        print(f"{'=' * 100}")
        h2 = (f"{'Galaxy':<14} {'scale':>5} "
              f"{'n_hv_high':>10} {'n_hv_low':>9} {'n_sys':>7}  "
              f"{'max rel':>10}  {'max abs':>10}")
        print(h2)
        print("-" * len(h2))
        for r in summary:
            print(f"{r['galaxy']:<14} {r['scale']:>5.2f} "
                  f"{r['n_high']:>10d} {r['n_low']:>9d} "
                  f"{r['n_sys']:>7d}  "
                  f"{r.get('rel_dgrad_globals', 0.0):10.3e}  "
                  f"{r.get('abs_dgrad_globals', 0.0):10.3e}")

        print(f"\n{'=' * 100}")
        print("SUMMARY — rel. diff of ∇r_ang per spot category")
        print(f"{'=' * 100}")
        h3 = (f"{'Galaxy':<14} {'scale':>5} "
              f"{'n_hv_high':>10} {'n_hv_low':>9} {'n_sys':>7}  "
              f"{'rel∇r-sys':>11}  {'rel∇r-red':>11}  "
              f"{'rel∇r-blue':>11}")
        print(h3)
        print("-" * len(h3))
        for r in summary:
            print(f"{r['galaxy']:<14} {r['scale']:>5.2f} "
                  f"{r['n_high']:>10d} {r['n_low']:>9d} "
                  f"{r['n_sys']:>7d}  "
                  f"{r.get('rel_dgrad_r_sys', 0.0):11.3e}  "
                  f"{r.get('rel_dgrad_r_red', 0.0):11.3e}  "
                  f"{r.get('rel_dgrad_r_blue', 0.0):11.3e}")

    tol_tag = "|Δ total| ≤ 0.5 nats"
    if do_grad:
        tol_tag += (f" AND rel∇glob ≤ {args.grad_rtol_globals:.0e}"
                    f" AND rel∇r ≤ {args.grad_rtol_r:.0e}")
    print(f"\nSmallest grid per galaxy with {tol_tag} "
          "(worst across r_ang scales):")
    per_galaxy = {}
    for r in summary:
        per_galaxy.setdefault(r["galaxy"], []).append(r)
    for g, rows in per_galaxy.items():
        # Group rows by (n_high, n_low, n_sys), keep worst |Δ| over scales.
        grids = {}
        for row in rows:
            key = (row["n_high"], row["n_low"], row["n_sys"])
            grids.setdefault(key, []).append(row)
        worst = {}
        for k, rs in grids.items():
            worst[k] = dict(
                d_total=max(abs(r["d_total"]) for r in rs),
                rel_glob=(max(r.get("rel_dgrad_globals", 0.0) for r in rs)
                          if do_grad else 0.0),
                rel_r=(max(r.get("rel_dgrad_r", 0.0) for r in rs)
                       if do_grad else 0.0))
        gg = galaxy_grids_map[g]
        best = None
        for gk in gg:
            w = worst.get(gk)
            if w is None:
                continue
            if w["d_total"] > 0.5:
                continue
            if do_grad and w["rel_glob"] > args.grad_rtol_globals:
                continue
            if do_grad and w["rel_r"] > args.grad_rtol_r:
                continue
            best = (gk, w)
            break
        if best is None:
            best = (gg[-1], worst[gg[-1]])
            tag = "(none within tolerance)"
        else:
            tag = ""
        (n_h, n_l, n_s), w = best
        extra = f"worst |Δ|={w['d_total']:.3f} nats"
        if do_grad:
            extra += (f", worst rel∇glob={w['rel_glob']:.2e}"
                      f", worst rel∇r={w['rel_r']:.2e}")
        print(f"  {g:<14} -> n_hv_high={n_h:d}, n_hv_low={n_l:d}, "
              f"n_sys={n_s:d}  ({extra}) {tag}")


if __name__ == "__main__":
    main()
