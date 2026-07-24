# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Single-galaxy megamaser marginal-objective diagnostic via ``harmonic``.

The saved MCMC chain samples explicit per-spot ``(r_ang, phi)`` latents, while
this utility re-scores its global draws with DE's finite-support 2D-marginal
objective (``run_de_map._logp_2d_terms``).  These are not exactly the same
posterior measure: the explicit radius target has unbounded positive support,
whereas the marginal objective uses the configured finite physical-radius
support.  The learned harmonic-mean result is therefore a surrogate diagnostic
unless omitted radial tails are shown to be negligible; it is not a rigorous
absolute evidence or Bayes factor.

Stage 1 is a grid gate: at ``--n-check`` random posterior points, score the
marginal log-likelihood on the configured production marginal grid (at the
chain's recorded precision) and on a super-high-res float64 reference; abort
if any differ by more than
``--tol`` nats.  A failure means the production grid is not sufficiently
stable for this marginal-objective diagnostic.
The whole process runs with ``jax_enable_x64`` so the reference is genuinely
float64; the production replica uses the precision recorded by the chain.

The finite integrations also omit latent-prior normalisation constants.
Consequently, even when radial tails are negligible, the reported ``ln Z`` is
an objective-dependent score rather than a normalised model evidence.
"""
import argparse
import copy
import os
import sys
import tempfile

import numpy as np
import tomli
import tomli_w
from h5py import File as H5File

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_CONFIG = os.path.join(_HERE, "../../local_config.toml")
try:
    with open(_LOCAL_CONFIG, "rb") as f:
        _lcfg = tomli.load(f)
except OSError:
    _lcfg = {}
ld = os.environ.get("LD_LIBRARY_PATH", "")
needed = [p for p in _lcfg.get("gpu_ld_library_path", []) if p not in ld]
if needed:
    os.environ["LD_LIBRARY_PATH"] = ":".join(needed) + (f":{ld}" if ld else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# The reference grid must be genuinely float64; that is a global JAX flag, so
# it is set before any array is created. The production replica is explicitly
# cast back to the precision recorded by the chain.
from jax import config as _jax_config  # noqa: E402

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

_CONFIG_PATH = os.path.join(_HERE, "config_maser.toml")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "convergence"))

from convergence_utils import cast_model_floats  # noqa: E402
from run_de_map import (_clean_init, _evaluate_one_at_a_time,  # noqa: E402
                        _logp_2d_terms, _make_logp, _plan_de_batch)

from candel.inference.evidence import harmonic_evidence  # noqa: E402
from candel.inference.evidence import laplace_evidence  # noqa: E402
from candel.model.maser_blackjax import MaserBlackJaxTarget  # noqa: E402
from candel.model.model_H0_maser import MaserDiskModel  # noqa: E402
from candel.pvdata.megamaser_data import load_megamaser_spots  # noqa: E402
from candel.util import data_path, fprint, fsection  # noqa: E402

# Quadrature grids scaled to build the float64 reference.
_GRID_KEYS = (
    "n_r_local", "n_r_global", "n_phi_hv_high", "n_phi_hv_low", "n_phi_sys",
    "n_phi_partition_sys", "n_phi_partition_hv",
)


def _attr_str(attrs, key):
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _attr_bool(attrs, key):
    value = attrs.get(key, False)
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _apply_chain_attrs_to_config(cfg, galaxy, attrs):
    """Apply chain-recorded model choices before rebuilding the target."""
    gblk = cfg["model"]["galaxies"][galaxy]
    if "mass_parameterization" in attrs:
        gblk["mass_parameterization"] = _attr_str(
            attrs, "mass_parameterization")
    if "use_quadratic_warp" in attrs:
        gblk["use_quadratic_warp"] = _attr_bool(attrs, "use_quadratic_warp")
    if "use_ecc" in attrs:
        gblk["use_ecc"] = _attr_bool(attrs, "use_ecc")
    if _attr_bool(attrs, "uniform_da_prior"):
        cfg["model"]["D_c_prior"] = "uniform_D_A"
    return gblk


def _build_target(galaxy, master_cfg, data_root, attrs, *, grid_scale, dtype,
                  spot_batch=None):
    """Build a MaserBlackJaxTarget for ``galaxy`` matching the saved chain.

    The model variant (mass parameterisation, distance site, ecc, quadratic
    warp) is pinned to what the chain recorded so ``target.names`` matches the
    saved sampled sites.  With ``grid_scale > 1`` every quadrature grid is
    enlarged for the reference.
    """
    cfg = {
        "inference": master_cfg.get("inference", {}),
        "model": copy.deepcopy(master_cfg["model"]),
        "io": master_cfg["io"],
    }
    gblk = _apply_chain_attrs_to_config(cfg, galaxy, attrs)
    if grid_scale != 1:
        for d in (cfg["model"], gblk):
            for k in _GRID_KEYS:
                if k in d:
                    d[k] = int(round(int(d[k]) * grid_scale))

    data = load_megamaser_spots(data_root, galaxy, v_sys_obs=gblk["v_sys_obs"])
    if "D_lo" in gblk and "D_hi" in gblk:
        data["D_lo"] = float(gblk["D_lo"])
        data["D_hi"] = float(gblk["D_hi"])
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(cfg, tmp)
    tmp.close()
    try:
        model = MaserDiskModel(tmp.name, data)
    finally:
        os.unlink(tmp.name)
    cast_model_floats(model, dtype)
    h = float(model.config["model"].get("H0_ref", 73.0)) / 100.0
    init = _clean_init(model, gblk.get("init", {}))
    return MaserBlackJaxTarget(model, h, init, spot_batch=spot_batch)


def _load_chain(chain_path):
    """Return ``(attrs, samples_dict)`` from a run_maser HDF5 chain."""
    with H5File(chain_path, "r") as f:
        attrs = dict(f.attrs)
        samples = {k: np.asarray(f["samples"][k]) for k in f["samples"].keys()}
    return attrs, samples


def _production_dtype(attrs, gblk, samples):
    """Resolve the precision used by the saved production chain."""
    if "precision" in attrs:
        precision = _attr_str(attrs, "precision").lower()
        if precision not in ("float32", "float64"):
            raise ValueError(
                f"chain has unsupported precision metadata {precision!r}")
        return jnp.float64 if precision == "float64" else jnp.float32
    if gblk.get("force_f64", False) or any(
            np.issubdtype(np.asarray(value).dtype, np.floating)
            and np.asarray(value).dtype.itemsize > 4
            for value in samples.values()):
        return jnp.float64
    return jnp.float32


def _stack_globals(samples, names, dtype=np.float32):
    """Stack the sampled global sites into ``(N, ndim)`` in names order."""
    cols = [
        np.asarray(samples[n], dtype=dtype).reshape(-1) for n in names]
    return np.column_stack(cols)


def _thin_rows(X, max_samples):
    """Keep at most ``max_samples`` rows by taking every Nth sample."""
    if max_samples is None or max_samples <= 0 or X.shape[0] <= max_samples:
        return X
    step = int(np.ceil(X.shape[0] / max_samples))
    return X[::step]


def _append_evidence_summary(chain_path, galaxy, lnZ, err, lnZ_lap, err_lap):
    path = os.path.splitext(chain_path)[0] + "_summary.txt"
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("Megamaser MCMC summary\n")
        f.write("\nMarginal-objective diagnostic: " + galaxy + "\n")
        f.write(f"  ln Z diagnostic (harmonic) = {lnZ:.4f}   "
                f"err(ln 1/Z) = {err}\n")
        f.write(f"  ln Z diagnostic (Laplace)  = {lnZ_lap:.4f} "
                f"+/- {err_lap:.4f}  "
                "(coarse cross-check)\n")
    fprint(f"appended evidence summary to {path}")


def _sampled_global_names(target, samples):
    """Return the sampled global sites required by the target.

    HDF5 chains may contain derived/deterministic datasets such as ``D_A``
    from a ``D_c`` chain or ``log_MBH`` from an ``eta`` chain.  Evidence must
    not add those extra transformed columns; it uses exactly the target's
    sampled theta sites.
    """
    names = tuple(target.names)
    missing = [n for n in names if n not in samples]
    if missing:
        raise KeyError(
            f"chain is missing sampled sites {missing}; cannot score the "
            "marginal posterior. Evidence is scored only in sampled site "
            "coordinates; check the chain metadata (uniform_da_prior, "
            "mass_parameterization, fix_floors_pesce).")
    return names


def _marginal_ll(target, sample, dtype):
    """Data-only 2D marginal log-likelihood at one global point."""
    theta = target.complete_params(
        {n: jnp.asarray(sample[n], dtype=dtype) for n in target.names})
    _, ll, _, _ = _logp_2d_terms(target, theta)
    return float(jax.device_get(jax.block_until_ready(ll)))


def _grid_gate(target_prod, target_ref, X, names, n_check, tol, seed,
               prod_dtype):
    """Stage 1: production-precision vs float64 reference at random points."""
    precision = np.dtype(prod_dtype).name
    fsection(
        f"Stage 1 — grid gate (production {precision} vs float64 reference)")
    rng = np.random.default_rng(seed)
    n_check = min(n_check, X.shape[0])
    idx = rng.choice(X.shape[0], size=n_check, replace=False)
    print(f"  {'sample':>8}  {'prod lnL':>14}  {'ref lnL (f64)':>14}  "
          f"{'|Δ| nats':>10}", flush=True)
    dmax = 0.0
    for i in idx:
        sample = {n: float(X[i, j]) for j, n in enumerate(names)}
        ll_p = _marginal_ll(target_prod, sample, prod_dtype)
        ll_r = _marginal_ll(target_ref, sample, jnp.float64)
        d = abs(ll_p - ll_r)
        dmax = max(dmax, d)
        print(f"  {int(i):>8d}  {ll_p:>14.4f}  {ll_r:>14.4f}  {d:>10.4f}",
              flush=True)
    if not np.isfinite(dmax) or dmax > tol:
        raise SystemExit(
            f"GRID GATE FAILED: max |Δ lnL| = {dmax:.4f} nats > tol "
            f"{tol:.4f}. "
            "The configured marginal-objective grid is not reference-stable; "
            "raise its phi/r resolution and rerun this diagnostic (or relax "
            "--tol if the discrepancy is acceptable).")
    fprint(f"PASS: max |Δ lnL| = {dmax:.4f} nats <= tol {tol:.4f}")


def _resolve_batching(target_prod, target_ref, n_samples, args, gblk,
                      prod_dtype):
    """Pick the production spot batch for the evidence calculation.

    The f64 4x reference is the memory hog, so it is always scored ONE SPOT at
    a time (``target_ref.spot_batch = 1``). The production spot batch
    mirrors run_de_map: auto-size it from the device VRAM budget on GPU, while
    scoring exactly one global sample per GPU wave. ``--spot-batch`` can
    override the production value. Sets ``target.spot_batch`` on both targets.
    """
    cfg_sb = gblk.get("conditional_spot_batch", None)
    cfg_sb = int(cfg_sb) if cfg_sb is not None else None

    plan_sb_p, plan_available, info_p = _plan_de_batch(
        target_prod.model, n_samples, gpu_mem_gb=args.gpu_mem)

    if plan_available:  # GPU budget available -> auto-size
        fprint(
            f"memory plan (production {np.dtype(prod_dtype).name}): " + info_p)
        if "UNRECOGNISED" in info_p:
            fprint("  WARNING: GPU not recognised; conservative budget. Pass "
                   "--gpu-mem GB or add it to _GPU_VRAM_GB in run_de_map.py.")
        sb_p = plan_sb_p if args.spot_batch is None else args.spot_batch
    else:  # no device budget (CPU) -> configured defaults
        sb_p = cfg_sb if args.spot_batch is None else args.spot_batch
        fprint("no GPU memory budget; using config defaults")

    target_prod.spot_batch = sb_p
    target_ref.spot_batch = 1
    fprint(f"batching: prod spot_batch={sb_p} (None=all spots), "
           f"ref(f64,{args.ref_grid_scale}x) spot_batch=1 (forced), "
           "one global sample per GPU wave"
           f"{' [--spot-batch]' if args.spot_batch is not None else ''}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Single-galaxy megamaser marginal-objective diagnostic "
                    "via harmonic.")
    parser.add_argument("galaxy", type=str)
    parser.add_argument("--chain", required=True,
                        help="run_maser HDF5 chain with global samples.")
    parser.add_argument("--data-root", default=data_path("data", "Megamaser"))
    parser.add_argument("--spot-batch", type=int, default=None,
                        help="Spots per quadrature batch (memory control). "
                             "Unset: auto-size from VRAM on GPU (per target, "
                             "so the f64 reference gets its own batch), else "
                             "the galaxy's conditional_spot_batch from the "
                             "config.")
    parser.add_argument("--gpu-mem", type=int, default=None,
                        help="VRAM (GB) for the auto-batcher; only selects "
                             "the V100 16/32GB variant (see run_de_map).")
    parser.add_argument("--ref-grid-scale", type=int, default=4,
                        help="Quadrature-grid multiplier for the float64 "
                             "reference used by the Stage-1 gate "
                             "(default: 4).")
    parser.add_argument("--tol", type=float, default=0.1,
                        help="Max |Δ lnL| (nats) at a check point before the "
                             "grid gate fails (default: 0.1).")
    parser.add_argument("--n-check", type=int, default=3,
                        help="Random posterior points for the grid gate.")
    parser.add_argument("--num-chains-harmonic", type=int, default=4,
                        help="Reshape the pooled samples into this many "
                             "pseudo-chains for harmonic's train/infer split.")
    parser.add_argument("--max-samples", type=int, default=10_000,
                        help="Thin to at most this many posterior samples by "
                             "taking every Nth draw before Stage 1/2 "
                             "(default: 10000; <=0 disables thinning).")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    with open(_CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)
    if args.galaxy not in master_cfg["model"]["galaxies"]:
        raise SystemExit(f"Unknown galaxy {args.galaxy!r}.")

    attrs, samples = _load_chain(args.chain)
    if _attr_bool(attrs, "fix_floors_pesce"):
        raise SystemExit(
            "the diagnostic for --fix-floors-pesce chains is not supported "
            "yet "
            "(the fixed floors are dropped from the sampled sites).")

    fsection(f"Single-galaxy marginal-objective diagnostic: {args.galaxy}")
    fprint(f"chain: {args.chain}")
    gblk = master_cfg["model"]["galaxies"][args.galaxy]
    prod_dtype = _production_dtype(attrs, gblk, samples)
    prod_precision = np.dtype(prod_dtype).name
    fprint(f"JAX backend: {jax.default_backend()}; reference precision: "
           f"float64; production precision: {prod_precision}")
    fprint("WARNING: this re-scores an explicit-latent MCMC chain with a "
           "finite-radius marginal objective. It is not a rigorous absolute "
           "evidence or Bayes factor; radial-tail and latent-normalisation "
           "constants are not established.")

    target_prod = _build_target(
        args.galaxy, master_cfg, args.data_root, attrs,
        grid_scale=1, dtype=prod_dtype, spot_batch=args.spot_batch)
    target_ref = _build_target(
        args.galaxy, master_cfg, args.data_root, attrs,
        grid_scale=args.ref_grid_scale, dtype=jnp.float64,
        spot_batch=args.spot_batch)

    names = _sampled_global_names(target_prod, samples)
    X_full = _stack_globals(samples, names, dtype=np.dtype(prod_dtype))
    if args.max_samples > 0:
        fprint(f"posterior sample cap: {args.max_samples} "
               "(take every Nth draw)")
    else:
        fprint("posterior sample cap: disabled")
    X = _thin_rows(X_full, args.max_samples)
    if X.shape[0] < X_full.shape[0]:
        fprint(f"thinned posterior samples: {X_full.shape[0]} -> "
               f"{X.shape[0]} (max_samples={args.max_samples})")
    fprint(f"{X.shape[0]} posterior samples, {X.shape[1]} global parameters")
    fprint("globals: " + ", ".join(names))

    _resolve_batching(
        target_prod, target_ref, X.shape[0], args, gblk, prod_dtype)

    _grid_gate(target_prod, target_ref, X, names,
               args.n_check, args.tol, args.seed, prod_dtype)

    fsection("Stage 2 — marginal log-posterior over globals (production grid)")
    logp = jax.jit(jax.vmap(_make_logp(target_prod, names)))
    lnpost = np.asarray(_evaluate_one_at_a_time(
        logp, jnp.asarray(X), desc="marginal logP"))
    ok = np.isfinite(lnpost)
    if int(ok.sum()) < X.shape[0]:
        fprint(f"dropped {int((~ok).sum())} non-finite log-posterior samples")
    X, lnpost = X[ok], lnpost[ok]
    fprint(f"logP range: [{lnpost.min():.2f}, {lnpost.max():.2f}], "
           f"{X.shape[0]} finite samples")

    C = int(args.num_chains_harmonic)
    if C < 2:
        raise SystemExit("--num-chains-harmonic must be >= 2 for harmonic.")
    M = X.shape[0] // C
    if M < 2:
        raise SystemExit(
            f"too few samples ({X.shape[0]}) for {C} harmonic chains.")
    samples_arr = X[:C * M].reshape(C, M, X.shape[1])
    log_density = lnpost[:C * M].reshape(C, M)

    fsection("Stage 3 — harmonic marginal-objective diagnostic")
    lnZ, err = harmonic_evidence(
        samples_arr, log_density, temperature=args.temperature,
        epochs_num=args.epochs, return_flow_samples=False, verbose=True)
    lnZ_lap, err_lap = laplace_evidence(samples_arr, log_density)

    fsection(f"Marginal-objective diagnostic: {args.galaxy}")
    fprint(f"ln Z diagnostic (harmonic) = {lnZ:.4f}   "
           f"err(ln 1/Z) = {err}")
    fprint(f"ln Z diagnostic (Laplace)  = {lnZ_lap:.4f} "
           f"+/- {err_lap:.4f}  "
           "(coarse cross-check)")
    _append_evidence_summary(
        args.chain, args.galaxy, lnZ, err, lnZ_lap, err_lap)
    return float(lnZ)


if __name__ == "__main__":
    main()
