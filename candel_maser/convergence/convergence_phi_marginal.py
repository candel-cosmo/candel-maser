"""
Mode 1 phi-marginal convergence test.

For each galaxy, build the unified phi grid at several (n_phi_hv_high,
n_phi_hv_low, n_phi_sys) settings and compare to a full-2π brute-force
reference evaluated on a single uniform grid per spot type. This tells
us (a) whether the config sub-range grids have enough density, and
(b) whether the sub-range restrictions miss any likelihood mass outside
the configured ranges (full-2π catches leakage).

Usage:
    python scripts/megamaser/convergence_phi_marginal.py [--galaxies ...]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import tomli

from candel.model.maser_convergence import (
    bruteforce_ll_mode1, build_model, get_default_grid,
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
    (1001,  301,   1501),
    (2001,  501,   2001),
    (3001,  501,   4501),
    (5001,  1001,  5001),
    (10001, 2001,  10001),
]


def build_test_grids(default):
    """Build the convergence test list: config default + anchor + EXTRA,
    deduplicated and sorted ascending (tightest last)."""
    d = (default["n_hv_high"], default["n_hv_low"], default["n_sys"])
    grids = set(EXTRA_GRIDS) | {d, PHI_MARGINAL_ANCHOR}
    return sorted(grids, key=lambda g: (g[0], g[2], g[1]))

# r_ang scales: sample r_ang at the data-driven estimate × scale.
R_SCALES = [0.5, 1.0, 2.0]


def per_type_logl(ll_per_spot, model):
    return dict(
        sys=float(jnp.sum(ll_per_spot[model._idx_sys])),
        red=float(jnp.sum(ll_per_spot[model._idx_red])),
        blue=float(jnp.sum(ll_per_spot[model._idx_blue])),
        total=float(jnp.sum(ll_per_spot)))


def estimate_r_ang(model, D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv):
    r_est, _, _, _ = model._estimate_adaptive_r(
        D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
    return np.asarray(r_est)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxies", nargs="+", default=ALL_GALAXIES)
    args = parser.parse_args()

    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_enable_x64", True)
    print(f"JAX platform: {jax.default_backend()}, precision: float64",
          flush=True)

    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)
    galaxies_cfg = master_cfg["model"]["galaxies"]
    ref_cfg = master_cfg["convergence"]["mode1_reference"]

    default = get_default_grid(master_cfg)
    test_grids = build_test_grids(default)

    print("=" * 80)
    print("Mode 1 phi-marginal convergence")
    print(f"Reference: full-2π uniform grid, n_phi = {ref_cfg['n_phi']}")
    print(f"Config default (n_hv_high, n_hv_low, n_sys) = "
          f"({default['n_hv_high']}, {default['n_hv_low']}, "
          f"{default['n_sys']})")
    print(f"Test grids: {test_grids}")
    print(f"r_ang scales: {R_SCALES}")
    print("=" * 80)

    summary = []
    galaxy_grids_map = {}
    for galaxy in args.galaxies:
        print(f"\n{'─' * 70}")
        print(f"Galaxy: {galaxy}")
        print(f"{'─' * 70}")

        # Include galaxy-specific phi overrides from the config (e.g. NGC4258
        # uses 20001/4001/20001, well above the global EXTRA_GRIDS ceiling).
        gblk = galaxies_cfg[galaxy]
        gkeys = ("n_phi_hv_high", "n_phi_hv_low", "n_phi_sys")
        if all(k in gblk for k in gkeys):
            galaxy_cfg_grid = tuple(int(gblk[k]) for k in gkeys)
            galaxy_grids = sorted(
                set(test_grids) | {galaxy_cfg_grid},
                key=lambda g: (g[0], g[2], g[1]))
        else:
            galaxy_grids = test_grids
        galaxy_grids_map[galaxy] = galaxy_grids

        # Build model once with Mode 1 forced on. Defaults used; we
        # override per-iteration via rebuild for the test grids.
        model = build_model(galaxy, master_cfg, mode="mode1")
        phys_args, phys_kw, diag = _phys_from_init(
            model, galaxies_cfg, galaxy)
        D_A, M_BH, v_sys = diag["D_A"], diag["M_BH"], diag["v_sys"]
        # Pull the scalars out of phys_args we need for r estimation.
        i0 = phys_args[8]
        sigma_a_floor2 = phys_args[16]
        var_v_hv = phys_args[15]
        r_est = estimate_r_ang(model, D_A, M_BH, v_sys,
                                sigma_a_floor2, i0, var_v_hv)
        print(f"  D_A={D_A:.2f} Mpc, n_spots={model.n_spots}")

        for scale in R_SCALES:
            r_ang = jnp.clip(
                jnp.asarray(r_est * scale),
                model._r_ang_lo, model._r_ang_hi)
            med_r = float(jnp.median(r_ang))
            print(f"\n  r_ang scale = {scale:.2f}x  "
                  f"(median r = {med_r:.4f} mas)")

            # Reference: single full-2π uniform grid per type.
            ref = bruteforce_ll_mode1(
                model, phys_args, phys_kw, r_ang, ref_cfg)
            print(f"    full-2π ref: sys={ref['sys']:.4f}, "
                  f"red={ref['red']:.4f}, blue={ref['blue']:.4f}, "
                  f"total={ref['total']:.4f}")

            print(f"    {'n_hv_high':>10} {'n_hv_low':>9} {'n_sys':>7}  "
                  f"{'Δ total':>11}  {'Δ sys':>9}  {'Δ red':>9}  "
                  f"{'Δ blue':>9}")
            grid_results = {}
            for n_high, n_low, n_sys in galaxy_grids:
                m_t = build_model(
                    galaxy, master_cfg, mode="mode1",
                    n_phi_hv_high=n_high, n_phi_hv_low=n_low,
                    n_phi_sys=n_sys)
                # spot_groups: sys, red, blue (Mode 1 — log_w_r=None).
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
                per_t = per_type_logl(ll, m_t)
                grid_results[(n_high, n_low, n_sys)] = per_t

                d_tot = per_t["total"] - ref["total"]
                d_sys = per_t["sys"] - ref["sys"]
                d_red = per_t["red"] - ref["red"]
                d_blue = per_t["blue"] - ref["blue"]
                print(
                    f"    {n_high:>10d} {n_low:>9d} {n_sys:>7d}  "
                    f"{d_tot:+11.4f}  {d_sys:+9.4f}  {d_red:+9.4f}  "
                    f"{d_blue:+9.4f}")

                summary.append(dict(
                    galaxy=galaxy, scale=scale,
                    n_high=n_high, n_low=n_low, n_sys=n_sys,
                    n_spots=model.n_spots,
                    d_total=d_tot, d_sys=d_sys, d_red=d_red,
                    d_blue=d_blue))

            # Check for self-convergence between last two grids.
            g1 = galaxy_grids[-2]
            g2 = galaxy_grids[-1]
            d_conv = abs(grid_results[g2]["total"]
                         - grid_results[g1]["total"])
            print(f"    |Δ(last two grids)| = {d_conv:.4f} nats")

    # --------------------------------------------------------------
    # Summary table
    # --------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("SUMMARY — Δ total logL (cfg − full-2π ref), nats")
    print(f"{'=' * 100}")
    header = (f"{'Galaxy':<14} {'scale':>5} "
              f"{'n_hv_high':>10} {'n_hv_low':>9} {'n_sys':>7}  "
              f"{'Δ total':>10}  {'Δ sys':>9}  {'Δ red':>9}  {'Δ blue':>9}")
    print(header)
    print("-" * len(header))
    for r in summary:
        print(f"{r['galaxy']:<14} {r['scale']:>5.2f} "
              f"{r['n_high']:>10d} {r['n_low']:>9d} {r['n_sys']:>7d}  "
              f"{r['d_total']:+10.4f}  {r['d_sys']:+9.4f}  "
              f"{r['d_red']:+9.4f}  {r['d_blue']:+9.4f}")

    # Per-galaxy tightest grid that stays within 0.5 nats of ref.
    print(f"\nSmallest grid per galaxy with |Δ total| ≤ 0.5 nats "
          "(worst across r_ang scales):")
    per_galaxy = {}
    for r in summary:
        per_galaxy.setdefault(r["galaxy"], []).append(r)
    for g, rows in per_galaxy.items():
        # Group rows by (n_high, n_low, n_sys), pick worst |Δ| over scales.
        grids = {}
        for row in rows:
            key = (row["n_high"], row["n_low"], row["n_sys"])
            grids.setdefault(key, []).append(abs(row["d_total"]))
        worst = {k: max(v) for k, v in grids.items()}
        # galaxy_grids_map[g] is ordered ascending; find smallest passing.
        gg = galaxy_grids_map[g]
        best = None
        for gk in gg:
            if worst.get(gk, float("inf")) <= 0.5:
                best = (gk, worst[gk])
                break
        if best is None:
            best = (gg[-1], worst[gg[-1]])
            tag = "(none within 0.5)"
        else:
            tag = ""
        (n_h, n_l, n_s), d = best
        print(f"  {g:<14} -> n_hv_high={n_h:d}, n_hv_low={n_l:d}, "
              f"n_sys={n_s:d}  (worst |Δ|={d:.3f} nats) {tag}")


if __name__ == "__main__":
    main()
