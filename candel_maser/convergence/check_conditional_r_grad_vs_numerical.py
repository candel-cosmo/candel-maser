"""Conditional-r AD gradient vs numerical finite differences.

Validates that gradients w.r.t. global disk parameters propagate correctly
through the conditional-r (r+phi marginalised) likelihood, with r-grids frozen by  # noqa: E501
stop_gradient.  Tests ONE spot at a time to keep memory bounded.

Two independent checks per test spot:
  (A) Separated: grids built once outside jax.grad; AD grad = d/dtheta of
      the integrand evaluated at fixed grid nodes.  Compared to central FD
      at the same fixed nodes.  Because both AD and FD differentiate the
      SAME trapezoidal-sum function (identical grid positions), agreement
      should be near machine precision -- integration resolution does NOT
      limit the comparison.
  (B) Full pipeline: jax.grad through the entire _build_conditional_r_grids +
      _eval_phi_marginal chain.  Must match (A) if stop_gradient on
      r_union and log_w_union correctly severs the grid-position gradient.

After passing, prints analytical and empirical memory estimates for
production-scale gradient computation (one spot and all spots).

Usage:
    python check_conditional_r_grad_vs_numerical.py [--galaxy NGC5765b] [options]  # noqa: E501
"""
import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tomli  # noqa: E402
from jax.scipy.special import logsumexp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from convergence_utils import (build_model, ensure_grad_sample,  # noqa: E402
                               extend_grad_params, jax_phys_from_sample,
                               resolve_grid_for_galaxy)

from candel.pvdata.megamaser_data import load_megamaser_spots  # noqa: E402

CONFIG_PATH = "scripts/megamaser/config_maser.toml"

_N_CHECKB_SPOTS = 10


def subsample_data(data, n_total=_N_CHECKB_SPOTS, seed=0):
    """Return a copy of ``data`` with at most ``n_total`` spots.

    Picks ~equal numbers from each type (blue/sys/red) via the
    ``is_highvel`` and ``is_blue`` flags set by the loader.
    """
    n = data["n_spots"]
    if n <= n_total:
        return data

    is_hv = np.asarray(data["is_highvel"])
    is_blue = np.asarray(data["is_blue"])
    idx_sys = np.where(~is_hv)[0]
    idx_blue = np.where(is_hv & is_blue)[0]
    idx_red = np.where(is_hv & ~is_blue)[0]

    rng = np.random.default_rng(seed)
    per_type = max(1, n_total // 3)
    keep = np.concatenate([
        rng.choice(g, size=min(per_type, len(g)), replace=False)
        for g in (idx_sys, idx_blue, idx_red) if len(g) > 0])
    keep = np.sort(keep)[:n_total]

    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.shape == (n,):
            out[k] = v[keep]
        else:
            out[k] = v
    out["n_spots"] = len(keep)
    return out


def build_model_from_data(galaxy, master_cfg, data, **overrides):
    """Like ``build_model`` but accepts pre-loaded (subsampled) data."""
    import tempfile

    import tomli_w

    from candel.model.model_H0_maser import MaserDiskModel

    cfg = {k: (v.copy() if isinstance(v, dict) else v)
           for k, v in master_cfg.items()}
    cfg["model"] = dict(master_cfg["model"])
    cfg["model"]["galaxies"] = {
        g: dict(blk) for g, blk in master_cfg["model"]["galaxies"].items()}
    gblk = cfg["model"]["galaxies"][galaxy]
    for key in ("n_phi_hv_high", "n_phi_hv_low", "n_phi_sys",
                "phi_hv_inner_deg", "phi_hv_outer_deg",
                "phi_sys_ranges_deg",
                "n_r_local", "n_r_global", "K_sigma",
                "refine_r_center", "n_refine_steps",
                "conditional_spot_batch"):
        gblk.pop(key, None)
    for k, v in overrides.items():
        cfg["model"][k] = v
    for key in ("D_lo", "D_hi"):
        if key in master_cfg["model"]["galaxies"][galaxy]:
            data[key] = float(master_cfg["model"]["galaxies"][galaxy][key])
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(cfg, tmp)
    tmp.close()
    model = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)
    return model


# -----------------------------------------------------------------------
# Grid building (outside JAX tracing)
# -----------------------------------------------------------------------

def build_conditional_r_grids(model, init_block):
    """Build conditional-r grids at the init parameters (numpy path)."""
    sample_np = {k: np.asarray(v) for k, v in init_block.items()}
    phys_args, phys_kw, diag = model.phys_from_sample(sample_np)
    groups = model._build_conditional_r_grids(
        phys_args[2], phys_args[3], phys_args[4],
        phys_args[16], phys_args[8], phys_args[15],
        phys_args=phys_args, phys_kw=phys_kw)
    return groups, diag


def find_spot_in_groups(groups, global_idx):
    """Return (type_key, group_pos, single_r, single_lw)."""
    for type_key, idx, r_union, log_w_union in groups:
        idx_np = np.asarray(idx)
        pos = np.where(idx_np == global_idx)[0]
        if len(pos) == 0:
            continue
        pos = int(pos[0])
        return type_key, pos, r_union[pos:pos + 1], log_w_union[pos:pos + 1]
    raise ValueError(f"spot {global_idx} not found in any group")


# -----------------------------------------------------------------------
# Differentiable single-spot log-likelihood
# -----------------------------------------------------------------------

def make_ll_separated(model, global_idx, group_type, single_r, single_lw,
                      has_any_accel):
    """Closure: f(sample) -> scalar ll for one spot at fixed (frozen) grid."""
    single_idx = jnp.array([global_idx])
    pc = model._phi_concat[group_type]
    sin_phi = pc["sin_phi"]
    cos_phi = pc["cos_phi"]
    log_w_phi = pc["log_w_phi"]

    def f(sample):
        pa, pk = jax_phys_from_sample(model, sample)
        r_pre = model._r_precompute(
            single_r, single_idx, *pa, **pk,
            has_any_accel=has_any_accel)
        nhc = model._phi_eval(r_pre, sin_phi, cos_phi)
        lnorm = r_pre["lnorm"] + r_pre["lnorm_a"]
        w2d = single_lw[:, :, None] + log_w_phi[None, None, :]
        return lnorm[0] + logsumexp(nhc[0] + w2d[0])

    return f


def make_ll_full_pipeline(model, global_idx):
    """Closure: f(sample) -> scalar ll via the full production pipeline.

    Builds conditional-r grids INSIDE jax.grad so the stop_gradient mechanism
    is exercised.  Evaluates all spots via scan (spot_batch=1) and
    returns only the target spot's ll.
    """
    def f(sample):
        pa, pk = jax_phys_from_sample(model, sample)
        D_A, M_BH, v_sys = pa[2], pa[3], pa[4]
        i0, var_v_hv, sa2 = pa[8], pa[15], pa[16]
        groups = model._build_conditional_r_grids(
            D_A, M_BH, v_sys, sa2, i0, var_v_hv,
            phys_args=pa, phys_kw=pk)
        ll = model._eval_phi_marginal(groups, pa, pk, spot_batch=1)
        return ll[global_idx]

    return f


# -----------------------------------------------------------------------
# Finite-difference helpers
# -----------------------------------------------------------------------

def central_fd(f, sample, param, h):
    s_plus = {**sample, param: sample[param] + h}
    s_minus = {**sample, param: sample[param] - h}
    return (f(s_plus) - f(s_minus)) / (2.0 * h)


def fd_step(x, scale=1e-5):
    """Optimal-ish step for central FD in float64."""
    return scale * max(abs(float(x)), 1.0)


# -----------------------------------------------------------------------
# Test spot selection
# -----------------------------------------------------------------------

def all_spots(model):
    """Return (global_idx, type_label) for every spot."""
    spots = []
    for type_key, idx in [("red", model._idx_red),
                          ("blue", model._idx_blue),
                          ("sys", model._idx_sys_cons),
                          ("sys", model._idx_sys_uncons)]:
        for i in np.asarray(idx).ravel():
            spots.append((int(i), type_key))
    spots.sort()
    return spots


# -----------------------------------------------------------------------
# Memory estimation
# -----------------------------------------------------------------------

def analytical_memory(n_r, n_phi, n_spots, dtype_bytes=8):
    """Analytical per-spot and total memory estimate (MB)."""
    nhc = n_r * n_phi * dtype_bytes
    rpre = 20 * n_r * dtype_bytes
    fwd = nhc + rpre + nhc
    bwd = 3 * fwd
    one = (fwd + bwd) / 1e6
    return dict(one_spot_MB=one, all_spots_MB=n_spots * one,
                n_r=n_r, n_phi=n_phi)


def gpu_peak_mb():
    """Peak GPU memory used (MB), or None if unavailable."""
    try:
        stats = jax.local_devices()[0].memory_stats()
        return stats.get("peak_bytes_in_use", 0) / 1e6
    except Exception:
        return None


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Conditional-r AD gradient vs numerical FD validation.")
    p.add_argument("--galaxy", default="NGC5765b")
    p.add_argument("--small-grids", action="store_true",
                   help="Use reduced grids (faster, for quick checks)")
    p.add_argument("--rtol-fd", type=float, default=1e-5,
                   help="Relative tolerance for AD vs FD [check A]")
    p.add_argument("--rtol-pipeline", type=float, default=1e-7,
                   help="Relative tolerance for full vs separated [check B]")
    p.add_argument("--atol", type=float, default=1e-10,
                   help="Absolute tolerance: skip relative test when both "
                        "|AD| and |FD| are below this")
    p.add_argument("--skip-full-pipeline", action="store_true",
                   help="Skip check B (slower, traces through grid builder)")
    args = p.parse_args()

    os.system("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
              " 2>/dev/null || true")
    print(f"\nJAX backend: {jax.default_backend()}", flush=True)
    print(f"Galaxy: {args.galaxy}", flush=True)
    print(f"Float64: {jax.config.jax_enable_x64}", flush=True)

    with open(CONFIG_PATH, "rb") as fh:
        master_cfg = tomli.load(fh)
    galaxies_cfg = master_cfg["model"]["galaxies"]

    # ── Build model ──
    if args.small_grids:
        kw = dict(n_phi_hv_high=301, n_phi_hv_low=101,
                  n_phi_sys=301, n_r_local=101, n_r_global=51)
    else:
        grid = resolve_grid_for_galaxy(
            master_cfg, args.galaxy, "conditional_r")
        kw = dict(n_phi_hv_high=grid["n_hv_high"],
                  n_phi_hv_low=grid["n_hv_low"],
                  n_phi_sys=grid["n_sys"],
                  n_r_local=grid["n_r_local"],
                  n_r_global=grid["n_r_global"])
    print(f"Grid overrides: {kw}", flush=True)
    model = build_model(args.galaxy, master_cfg, **kw)

    init_block = galaxies_cfg[args.galaxy]["init"]
    sample = ensure_grad_sample(model, init_block)
    param_keys = extend_grad_params(model, sample)
    print(f"\nn_spots={model.n_spots}", flush=True)
    print(f"Grad params ({len(param_keys)}): {param_keys}", flush=True)

    # ── Build grids once (outside JAX tracing) ──
    print("\nBuilding conditional-r grids...", flush=True)
    t0 = time.time()
    groups, diag = build_conditional_r_grids(model, init_block)
    print(f"  {time.time() - t0:.1f}s, D_A={diag['D_A']:.2f} Mpc", flush=True)
    for tk, idx, r_u, _ in groups:
        n_r = int(r_u.shape[1])
        n_phi = int(model._phi_concat[tk]["sin_phi"].shape[0])
        print(f"  {tk}: {int(idx.shape[0])} spots, "
              f"r_union=({r_u.shape[0]},{n_r}), n_phi={n_phi}", flush=True)

    # ── Select test spots ──
    test_spots = all_spots(model)
    print(f"\nTest spots: {len(test_spots)} / {model.n_spots}", flush=True)

    # ── Memory estimates (analytical) ──
    print("\n── Analytical memory estimates ──", flush=True)
    for tk, idx, r_u, _ in groups:
        n_r = int(r_u.shape[1])
        n_phi = int(model._phi_concat[tk]["sin_phi"].shape[0])
        mem = analytical_memory(n_r, n_phi, model.n_spots)
        print(f"  {tk}: {mem['one_spot_MB']:.1f} MB/spot, "
              f"{mem['all_spots_MB']:.0f} MB all spots "
              f"(n_r={n_r}, n_phi={n_phi})", flush=True)

    # ── Per-spot gradient tests ──
    all_passed = True
    for spot_idx, spot_type in test_spots:
        print(f"\n{'=' * 70}", flush=True)
        print(f"Spot {spot_idx} (type={spot_type})", flush=True)
        print(f"{'=' * 70}", flush=True)

        type_key, gpos, single_r, single_lw = find_spot_in_groups(
            groups, spot_idx)
        has_accel = model._group_has_any_accel(type_key)
        n_r = int(single_r.shape[1])
        n_phi = int(model._phi_concat[type_key]["sin_phi"].shape[0])
        print(f"  group={type_key}, gpos={gpos}, n_r={n_r}, n_phi={n_phi}",
              flush=True)

        # ── Check A: separated AD vs FD ──
        f_sep = make_ll_separated(
            model, spot_idx, type_key, single_r, single_lw, has_accel)

        print("\n  [A] AD gradient vs numerical FD (separated grids)",
              flush=True)
        t0 = time.time()
        ll_val = float(f_sep(sample))
        ad = jax.grad(f_sep)(sample)
        ad = {k: float(v) for k, v in ad.items()}
        dt_ad = time.time() - t0

        t0 = time.time()
        fd = {}
        for pk in param_keys:
            if pk not in sample:
                continue
            h = fd_step(sample[pk])
            fd[pk] = float(central_fd(f_sep, sample, pk, h))
        dt_fd = time.time() - t0
        print(f"      ll = {ll_val:.6f}", flush=True)
        print(f"      AD: {dt_ad:.1f}s   FD: {dt_fd:.1f}s", flush=True)

        max_rel_a = 0.0
        worst_a = ""
        n_fail_a = 0
        hdr = (f"      {'Param':<20s} {'AD':>14s} {'FD':>14s} "
               f"{'|diff|':>10s} {'rel':>10s}")
        print(hdr, flush=True)
        print(f"      {'-' * 70}", flush=True)
        for pk in param_keys:
            if pk not in ad:
                continue
            a_val = ad[pk]
            f_val = fd[pk]
            d = abs(a_val - f_val)
            sc = max(abs(a_val), abs(f_val), 1e-30)
            rel = d / sc
            below_atol = d < args.atol
            if below_atol:
                flag = "  (atol)"
            elif rel > args.rtol_fd:
                flag = "  *** FAIL"
                n_fail_a += 1
            else:
                flag = ""
            if not below_atol and rel > max_rel_a:
                max_rel_a = rel
                worst_a = pk
            print(f"      {pk:<20s} {a_val:>14.6e} {f_val:>14.6e} "
                  f"{d:>10.2e} {rel:>10.2e}{flag}", flush=True)

        passed_a = n_fail_a == 0
        print(f"      max_rel = {max_rel_a:.2e} ({worst_a}) -> "
              f"{'PASS' if passed_a else 'FAIL'} "
              f"(tol = {args.rtol_fd:.0e})", flush=True)
        if not passed_a:
            all_passed = False

    # ── Check B: full pipeline vs separated on subsampled model ──
    if not args.skip_full_pipeline:
        print(f"\n{'=' * 70}", flush=True)
        print(f"Check B: stop_gradient validation "
              f"(subsampled to {_N_CHECKB_SPOTS} spots)", flush=True)
        print(f"{'=' * 70}", flush=True)

        data_sub = subsample_data(
            load_megamaser_spots(
                master_cfg["io"]["maser_data"]["root"],
                galaxy=args.galaxy,
                v_sys_obs=galaxies_cfg[args.galaxy]["v_sys_obs"]),
            n_total=_N_CHECKB_SPOTS)
        model_B = build_model_from_data(
            args.galaxy, master_cfg, data_sub, **kw)
        sample_B = ensure_grad_sample(model_B, init_block)
        print(f"  model_B: {model_B.n_spots} spots", flush=True)

        groups_pipeline, _ = build_conditional_r_grids(model_B, init_block)
        spots_B = all_spots(model_B)

        for spot_idx, spot_type in spots_B:
            print(f"\n  Spot {spot_idx} (type={spot_type})", flush=True)
            type_key, gpos, single_r, single_lw = find_spot_in_groups(
                groups_pipeline, spot_idx)
            has_accel = model_B._group_has_any_accel(type_key)

            f_sep = make_ll_separated(
                model_B, spot_idx, type_key, single_r, single_lw, has_accel)
            ad_sep = jax.grad(f_sep)(sample_B)
            ad_sep = {k: float(v) for k, v in ad_sep.items()}

            f_full = make_ll_full_pipeline(model_B, spot_idx)
            t0 = time.time()
            f_full(sample_B)
            ad_full = jax.grad(f_full)(sample_B)
            ad_full = {k: float(v) for k, v in ad_full.items()}
            dt_full = time.time() - t0
            print(f"      {dt_full:.1f}s", flush=True)

            max_rel_b = 0.0
            worst_b = "(none)"
            n_fail_b = 0
            for pk in param_keys:
                if pk not in ad_sep or pk not in ad_full:
                    continue
                d = abs(ad_sep[pk] - ad_full[pk])
                sc = max(abs(ad_sep[pk]), abs(ad_full[pk]), 1e-30)
                rel = d / sc
                if d < args.atol:
                    continue
                if rel > max_rel_b:
                    max_rel_b = rel
                    worst_b = pk
                if rel > args.rtol_pipeline:
                    n_fail_b += 1

            passed_b = n_fail_b == 0
            print(f"      max_rel = {max_rel_b:.2e} ({worst_b}) -> "
                  f"{'PASS' if passed_b else 'FAIL'} "
                  f"(tol = {args.rtol_pipeline:.0e})", flush=True)
            if not passed_b:
                all_passed = False
                for pk in param_keys:
                    if pk not in ad_sep or pk not in ad_full:
                        continue
                    a_s = ad_sep[pk]
                    a_f = ad_full[pk]
                    d = abs(a_s - a_f)
                    sc = max(abs(a_s), abs(a_f), 1e-30)
                    print(f"        {pk:<20s} sep={a_s:>14.6e} "
                          f"full={a_f:>14.6e} rel={d / sc:.2e}", flush=True)

    # ── Empirical GPU memory ──
    peak = gpu_peak_mb()
    if peak is not None:
        print(f"\nGPU peak memory used: {peak:.0f} MB", flush=True)

    # ── Production-grid memory estimate ──
    print("\n── Production-grid memory estimate ──", flush=True)
    prod_grid = resolve_grid_for_galaxy(
        master_cfg, args.galaxy, "conditional_r")
    n_r_prod = prod_grid["n_r_local"] + prod_grid["n_r_global"]
    for tk in ("red", "blue", "sys"):
        pc = model._phi_concat.get(tk)
        if pc is None:
            continue
        n_phi_prod = int(pc["sin_phi"].shape[0])
        mem = analytical_memory(n_r_prod, n_phi_prod, model.n_spots)
        print(f"  {tk}: {mem['one_spot_MB']:.1f} MB/spot, "
              f"{mem['all_spots_MB']:.0f} MB all spots "
              f"(n_r={n_r_prod}, n_phi={n_phi_prod})", flush=True)
    print("  NOTE: scan backward recomputes per-step activations, so peak "
          "memory is ~one_spot_MB, not all_spots_MB.", flush=True)

    # ── Summary ──
    print(f"\n{'=' * 70}", flush=True)
    status = "PASS" if all_passed else "FAIL"
    print(f"OVERALL: {status}", flush=True)
    print(f"{'=' * 70}", flush=True)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
