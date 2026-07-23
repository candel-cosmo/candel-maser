# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Run 2D-marginal MAP optimisation for one megamaser disk.

For each global proposal the per-spot latents ``(r_ang, phi)`` are
marginalised jointly on the conditional r-grid
(``_build_conditional_r_grids`` + ``_sum_phi_marginal``) — phi AND r
integrated together, NOT phi at a profiled ``r_ang`` (which overfits ``D_A``
upward).  The objective is multimodal, so the global search uses L-SHADE with
success-history mutation/crossover adaptation and linear population reduction.
Every proposal is scored with the exact all-spot objective.  The phi/r grid is
taken from the per-galaxy ``config_maser.toml`` settings (same grid the MCMC
and convergence checks use).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import warnings

import tomli

_LOCAL_CONFIG = os.path.join(
    os.path.dirname(__file__), "../../local_config.toml")
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

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_maser.toml")
with open(_CONFIG_PATH, "rb") as f:
    _MASTER_CFG = tomli.load(f)


def _required_inference(cfg, key):
    if key not in cfg:
        raise KeyError(f"Missing [inference].{key} in {_CONFIG_PATH}")
    return cfg[key]


def _f64_reason_from_argv(argv):
    if "--f64" in argv:
        return "--f64"
    galaxies = _MASTER_CFG["model"]["galaxies"]
    target = next((g for g in galaxies if g in argv), None)
    if target is not None and galaxies[target].get("force_f64", False):
        return f"forced for {target}"
    return None


_F64_REASON = _f64_reason_from_argv(sys.argv[1:])
_ENABLE_F64 = _F64_REASON is not None
_F64_ENABLED_HERE = False

if _ENABLE_F64:
    from jax import config as _jax_config  # noqa: E402
    _F64_ENABLED_HERE = not _jax_config.jax_enable_x64
    _jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402

# Persistent XLA compilation cache: resubmits/restarts reuse the compiled
# DE executable (keyed on HLO + jaxlib + GPU arch; results unaffected).
if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
    jax.config.update("jax_compilation_cache_dir",
                      os.path.expanduser("~/.cache/candel_jax"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tomli_w  # noqa: E402
from scipy.stats.qmc import Sobol  # noqa: E402
from tqdm import trange  # noqa: E402

from candel.inference.optimise import _prior_bounds  # noqa: E402
from candel.inference.optimise import _select_distinct  # noqa: E402
from candel.model.maser_blackjax import MaserBlackJaxTarget  # noqa: E402
from candel.model.maser_blackjax import init_from_prior_median  # noqa: E402
from candel.model.maser_physics import C_v  # noqa: E402
from candel.model.model_H0_maser import MaserDiskModel  # noqa: E402
from candel.pvdata.megamaser_data import load_megamaser_spots  # noqa: E402
from candel.util import (data_path, fprint, fsection, get_nested,  # noqa: E402
                         results_path)

if _F64_ENABLED_HERE:
    print(f"float64 enabled ({_F64_REASON})", flush=True)


def _h_ref(model):
    return float(get_nested(model.config, "model/H0_ref", 73.0)) / 100.0


def _D_A_from_D_c(model, D_c):
    h = _h_ref(model)
    z_cosmo = model.distance2redshift(jnp.atleast_1d(D_c), h=h).squeeze()
    return D_c / (1.0 + z_cosmo)


def _distance_bounds(gcfg):
    if "D_de_lo" in gcfg and "D_de_hi" in gcfg:
        return float(gcfg["D_de_lo"]), float(gcfg["D_de_hi"]), "DE"
    if "D_lo" in gcfg and "D_hi" in gcfg:
        return float(gcfg["D_lo"]), float(gcfg["D_hi"]), "default"
    return None


def _clean_init(model, init_cfg):
    init_params = {key: jnp.asarray(value) for key, value in init_cfg.items()}
    init_params.pop("M_BH", None)
    if model._D_A_uniform:
        if "D_A" not in init_params:
            if "D_c" not in init_params:
                raise KeyError(
                    "D_A-uniform DE init requires D_A or D_c.")
            init_params["D_A"] = _D_A_from_D_c(model, init_params["D_c"])
    elif "D_c" not in init_params:
        raise KeyError("DE init requires D_c.")
    D_A = (init_params["D_A"] if model._D_A_uniform
           else _D_A_from_D_c(model, init_params["D_c"]))
    mass_param = getattr(model, "mass_parameterization", "eta")
    if mass_param == "eta":
        if "eta" not in init_params:
            if "log_MBH" not in init_params:
                raise KeyError(
                    "eta mass parameterization requires initial 'eta' or "
                    "'log_MBH'.")
            init_params["eta"] = init_params["log_MBH"] - jnp.log10(D_A)
    else:
        if "log_MBH" not in init_params:
            if "eta" not in init_params:
                raise KeyError("DE init requires log_MBH.")
            init_params["log_MBH"] = init_params["eta"] + jnp.log10(D_A)
        init_params.pop("eta", None)
    if model._D_A_uniform:
        init_params.pop("D_c", None)
    if not model.use_ecc:
        for key in ("e_x", "e_y", "ecc", "periapsis", "periapsis_rad",
                    "dperiapsis_dr"):
            init_params.pop(key, None)
    elif model.ecc_cartesian:
        for key in ("ecc", "periapsis", "periapsis_rad"):
            init_params.pop(key, None)
        init_params.setdefault("e_x", jnp.asarray(0.0))
        init_params.setdefault("e_y", jnp.asarray(0.0))
        init_params.setdefault("dperiapsis_dr", jnp.asarray(0.0))
    if not model.use_quadratic_warp:
        for key in ("d2i_dr2", "d2Omega_dr2"):
            init_params.pop(key, None)
    else:
        init_params.setdefault("d2i_dr2", jnp.asarray(0.0))
        init_params.setdefault("d2Omega_dr2", jnp.asarray(0.0))
    return init_params


def _principal_angle_deg(x, y):
    """Sky position angle (deg) of the dominant axis of the spot cloud.

    ``predict_position`` places an edge-on HV spot at offset
    ``R·(sin Omega, cos Omega)`` from the centre, so the line-of-nodes PA is
    ``Omega = atan2(dX, dY)``.  The eigenvector sign (±180°) is left for the
    random Omega flip in the seed population.
    """
    pts = np.column_stack([x - x.mean(), y - y.mean()])
    vecs = np.linalg.eigh(pts.T @ pts)[1]
    vx, vy = vecs[:, -1]
    return float(np.rad2deg(np.arctan2(vx, vy)) % 360.0)


def _data_driven_seed(model, target, base_init, h0_ref, n_seed, seed,
                      sobol_n_sigma=5):
    """Random DE seed points sliding along the distance-mass degeneracy.

    The masers fix ``eta = log10(M_BH/D_A)`` (distance-free, from the angular
    Keplerian envelope) but barely constrain distance; the maser likelihood is
    near-flat along the ``M_BH ∝ D_A`` ridge.  So instead of one seed, draw
    ``n_seed`` points with random distance spanning the prior box at *fixed*
    ``eta`` — in the eta parameterisation that is literally sliding along the
    ridge, with ``log M_BH`` tracking ``D_A`` automatically.  Geometry
    (centre/PA/inclination/``dv_sys``) is jittered around its data seeds, with
    the ±180° PA ambiguity flipped on a random half.  Every other dimension
    (error floors, ecc/warp) is drawn Sobol-random within the DE box
    (``sobol_n_sigma``) so the ridge seeds are not identical there.  Returns
    ``(seed_points (n_seed, D) in target.names order, info)`` or
    ``(None, reason)``.
    """
    x = np.asarray(model._all_x)
    y = np.asarray(model._all_y)
    v = np.asarray(model._all_v)
    is_hv = np.asarray(model.is_highvel).astype(bool)
    n_hv = int(is_hv.sum())
    if n_hv < 2:
        return None, f"only {n_hv} HV spot(s); need >=2 for an eta seed"
    n_seed = int(n_seed)
    if n_seed < 1:
        return None, f"n_seed={n_seed} < 1"

    cz = float(model.v_sys_obs)

    is_sys = ~is_hv
    sel = is_sys if is_sys.any() else np.ones_like(is_sys)
    x0 = float(np.clip(x[sel].mean(), -750.0, 750.0))
    y0 = float(np.clip(y[sel].mean(), -750.0, 750.0))
    Omega0 = _principal_angle_deg(x, y)

    # Systemic masers sit at phi≈0,π (v_z≈0), so their velocity centroid is a
    # data-driven v_sys — without it the dv_sys offset (up to ~260 km/s) biases
    # eta by ~0.3 dex.  Angular Keplerian: edge-on HV spot has sky offset
    # theta=R·sin(phi) [mas] and dv=v_kep·sin(phi), so theta·dv² =
    # C_v²·(M_BH[1e7]/D_A)·sin³(phi).  HV masers concentrate near the tangent
    # (sin phi≈1), so the median recovers M_BH/D_A within ~10% (validated vs
    # Pesce) and is robust to the noisy upper tail.
    v_sys = float(np.median(v[sel]))
    theta = np.hypot(x[is_hv] - x0, y[is_hv] - y0) / 1e3
    dv = v[is_hv] - v_sys
    g = (theta > 0) & np.isfinite(dv)
    s = float(np.median(theta[g] * dv[g] ** 2))
    eta_seed = np.log10(s) - 2.0 * np.log10(C_v) + 7.0  # M_BH in Msun, +log1e7
    dv_sys_seed = float(np.clip(v_sys - cz, -900.0, 900.0))

    rng = np.random.default_rng(seed)
    D_lo, D_hi = _prior_bounds(model.priors["D"])
    distance_name = "D_A" if model._D_A_uniform else "D_c"
    D = rng.uniform(D_lo, D_hi, n_seed)                # slide along the ridge
    eta = eta_seed + rng.normal(0.0, 0.02, n_seed)     # tight: on the ridge
    i0 = np.clip(90.0 + rng.normal(0.0, 3.0, n_seed), 65.0, 115.0)
    flip = np.where(rng.random(n_seed) < 0.5, 180.0, 0.0)
    Omega = (Omega0 + flip + rng.normal(0.0, 5.0, n_seed)) % 360.0
    x0s = np.clip(x0 + rng.normal(0.0, 20.0, n_seed), -750.0, 750.0)
    y0s = np.clip(y0 + rng.normal(0.0, 20.0, n_seed), -750.0, 750.0)
    # dv_sys seed is the CMB↔LSR/bary frame offset; kept wide so the DE still
    # explores it rather than trusting the systemic centroid.
    dvs = dv_sys_seed + rng.normal(0.0, 100.0, n_seed)

    names = target.names
    seeds = np.tile(_theta_to_flat(base_init, names), (n_seed, 1))
    # Nuisance dims (error floors, ecc/warp) get Sobol-random draws within the
    # DE box; the ridge/geometry columns below overwrite their share.
    priors = {site: prior for site, _, prior in target.sites}
    b = np.array([_prior_bounds(priors[n], sobol_n_sigma=sobol_n_sigma)
                  for n in names], dtype=float)
    fin = np.all(np.isfinite(b), axis=1)
    if fin.any():
        with warnings.catch_warnings():       # n_seed need not be a power of 2
            warnings.simplefilter("ignore")
            sob = Sobol(d=int(fin.sum()), scramble=True, seed=seed).random(
                n_seed)
        seeds[:, fin] = b[fin, 0] + sob * (b[fin, 1] - b[fin, 0])

    def set_col(name, vals):
        if name in names:
            seeds[:, names.index(name)] = vals

    set_col(distance_name, D)
    set_col("x0", x0s)
    set_col("y0", y0s)
    set_col("i0", i0)
    set_col("Omega0", Omega)
    set_col("dv_sys", dvs)
    for key in ("di_dr", "dOmega_dr"):
        set_col(key, np.zeros(n_seed))
    if target.mass_parameterization == "eta":
        set_col("eta", eta)
    else:
        if model._D_A_uniform:
            D_A = D
        else:
            D_A = np.asarray(_D_A_from_D_c(model, jnp.asarray(D)))
        set_col("log_MBH", eta + np.log10(D_A))

    info = (f"{n_seed} seed candidates along the {distance_name} degeneracy "
            f"ridge: {distance_name}~U({D_lo:.0f},{D_hi:.0f}) Mpc, other "
            f"globals fixed at eta_seed={eta_seed:.3f} (BH-mass coordinate), "
            f"dv_sys={dv_sys_seed:.0f} km/s (systemic velocity), disc centre "
            f"x0={x0:.1f}, y0={y0:.1f} uas, PA Omega0={Omega0:.1f} deg; "
            f"{n_hv} high-velocity spots")
    return seeds, info


def _make_init(model, init_cfg, strategy, num_samples, rng_key):
    strategy = str(strategy).lower()
    if strategy == "median":
        h = _h_ref(model)
        return _clean_init(
            model, init_from_prior_median(
                model, rng_key, num_samples, h=h))
    if strategy == "config":
        return _clean_init(model, init_cfg)
    raise ValueError("DE init_strategy must be 'median' or 'config'.")


def _layout(target, sobol_n_sigma, fixed=()):
    names, lo, hi = [], [], []
    for site, _, prior in target.sites:
        if site in fixed:
            continue
        lower, upper = _prior_bounds(prior, sobol_n_sigma=sobol_n_sigma)
        if lower is None:
            continue
        names.append(site)
        lo.append(float(lower))
        hi.append(float(upper))
    sizes = [1] * len(names)
    return tuple(names), sizes, np.asarray(lo), np.asarray(hi)


def _flat_to_theta(x, names):
    theta = {}
    for i, name in enumerate(names):
        theta[name] = x[i]
    return theta


def _theta_to_flat(theta, names):
    return np.asarray([float(np.asarray(theta[name])) for name in names])


def _normalise_theta_point(theta, names, lo, hi):
    """Map a named physical point into the DE unit box coordinates."""
    lo = np.asarray(lo)
    scale = np.asarray(hi) - lo
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("DE bounds must have finite positive widths.")
    return (_theta_to_flat(theta, names) - lo) / scale


_DE_ALGORITHM = "lshade"
_DE_SEED_POLICY = "data_sobol_only"
_DE_POPULATION_SCHEDULE = "nfe_linear"
_DE_OBJECTIVE_POLICY = "scan_marginal_reuse_v2"
_DE_PEAK_PARTITION_POLICY = "peak_partition_v6"
_CANDIDATES_PER_GPU_WAVE = 1
_PEAK_PARTITION_CANDIDATES_PER_GPU_WAVE = 8
_F32_ALL_SPOT_GALAXIES = frozenset((
    "CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789"))


def _de_candidates_per_wave(model, peak_override=None):
    if model.phi_integration == "peak-partition":
        return (_PEAK_PARTITION_CANDIDATES_PER_GPU_WAVE
                if peak_override is None else int(peak_override))
    if peak_override is not None:
        raise ValueError(
            "peak candidate-wave override requires peak-partition")
    return _CANDIDATES_PER_GPU_WAVE


def _initial_de_seed_points(data_seeds):
    """Return only data-derived seeds; reference solutions are never seeded."""
    if data_seeds is None:
        return None
    return np.asarray(data_seeds, dtype=float).copy()


def _de_spot_batch_policy(galaxy, use_f64, requested, configured, planned):
    """Choose exact spot batching without overriding explicit controls."""
    if requested is not None:
        return int(requested), "explicit --spot-batch"
    if configured is not None:
        return int(configured), "per-galaxy config"
    if galaxy in _F32_ALL_SPOT_GALAXIES and not use_f64:
        return None, "measured float32 all-spots default"
    return planned, "automatic VRAM plan"


def _theta_to_output(theta, r_ang):
    out = dict(theta)
    out["r_ang"] = r_ang
    return out


def _global_logprior(target, theta, dtype):
    lp = jnp.asarray(0.0, dtype=dtype)
    for site, _, prior in target.sites:
        if site != "eta":
            lp = lp + prior.log_prob(theta[site])
    if target.mass_parameterization == "eta":
        lp = lp + target.model.priors["log_MBH"].log_prob(
            theta["log_MBH"])
    return lp


def _logp_2d_terms(target, theta):
    """Global log-prior and joint per-spot 2D ``(r_ang, phi)`` marginal.

    ``theta`` must already be completed (``target.complete_params``).  The data
    term integrates phi AND r jointly per spot on the conditional r-grid; there
    is no profiled ``r_ang`` and no radius barrier (r is integrated within its
    support).  Returns ``(log_prior, data_loglik, phys_args, phys_kw)``.
    """
    model = target.model
    phys_args, phys_kw = model.phys_from_params_jax(theta, target.h)
    # Peak-partition caches evaluated values (plus overflow masks), so its
    # global scan can be reused for circular and eccentric models. The fixed
    # grid retains its exact circular-only algebraic reuse.
    reuse_scan = (model.phi_integration == "peak-partition"
                  or (model.phi_integration == "fixed-grid"
                      and not model.use_ecc))
    built = model._build_conditional_r_grids(
        phys_args[2], phys_args[3], phys_args[4], phys_args[16],
        phys_args[8], phys_args[15], phys_args, phys_kw,
        return_scan_cache=reuse_scan)
    if reuse_scan:
        groups, scan_cache = built
    else:
        groups, scan_cache = built, None
    ll = model._sum_phi_marginal(
        groups, phys_args, phys_kw, spot_batch=target.spot_batch,
        remat=False, scan_cache=scan_cache)
    lp = _global_logprior(target, theta, ll.dtype)
    return lp, ll, phys_args, phys_kw


def _make_logp(target, names, fixed=None):
    fixed = dict(fixed) if fixed else {}

    def logp_constrained(x):
        params = _flat_to_theta(x, names)
        params.update(fixed)
        theta = target.complete_params(params)
        lp, ll, _, _ = _logp_2d_terms(target, theta)
        return lp + ll

    return logp_constrained


def _evaluate_one_at_a_time(fn, x, desc=None):
    """Evaluate rows sequentially through a fixed batch-one executable.

    The evidence runner imports this helper for posterior draws.  Production
    DE uses ``_make_batched_fitness`` and its fixed device-local blocks.
    """
    n = x.shape[0]
    parts = []
    for i in trange(
            n, total=n,
            desc=desc,
            disable=desc is None):
        y = fn(x[i:i + 1])
        parts.append(y)
        jax.block_until_ready(y)
    return jnp.concatenate(parts, axis=0)


_DEVICE_LOCAL_BLOCK_SIZE = 8
_DEVICE_WEIGHT_EMA = 0.3
_DEVICE_REBALANCE_MIN_GAIN = 0.02


def _device_block_capacity(count, block_size=_DEVICE_LOCAL_BLOCK_SIZE):
    """Padded device-local work for ``count`` real candidates.

    Every visible device receives at least one block during the first call so
    it is ready for later work. Later calls retain exactly the same executable
    shape; only the number of invocations changes.
    """
    if block_size < 1:
        raise ValueError("Device block size must be positive.")
    return max(block_size, ((int(count) + block_size - 1) // block_size)
               * block_size)


def _project_device_weights(weights):
    """Normalise weights while giving every device at least half a share."""
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or not len(weights)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0) or not np.sum(weights) > 0.0):
        raise ValueError("Device weights must be finite and non-negative.")
    weights = weights / np.sum(weights)
    floor = 0.5 / len(weights)
    excess = np.maximum(weights - floor, 0.0)
    if not np.sum(excess) > 0.0:
        return np.full(len(weights), 1.0 / len(weights))
    return floor + 0.5 * excess / np.sum(excess)


def _weighted_round_robin_assignment(n, weights):
    """Deterministic smooth weighted round-robin device indices."""
    weights = _project_device_weights(weights)
    assigned = np.zeros(len(weights), dtype=int)
    out = np.empty(n, dtype=int)
    for i in range(n):
        deficit = (i + 1) * weights - assigned
        device = int(np.argmax(deficit))
        out[i] = device
        assigned[device] += 1
    return out


def _restore_device_outputs(device_values, indices, n):
    """Restore device-local weighted shards to original candidate order."""
    dtype = np.asarray(device_values[0]).dtype
    out = np.empty(n, dtype=dtype)
    for values, idx in zip(device_values, indices):
        out[idx] = np.asarray(values)[:len(idx)]
    return out


def _updated_device_profile_weights(current, observed, profile_samples):
    """Use the first timing directly, then smooth later measurements."""
    observed = _project_device_weights(observed)
    if profile_samples == 0:
        return observed
    return _project_device_weights(
        ((1.0 - _DEVICE_WEIGHT_EMA) * np.asarray(current)
         + _DEVICE_WEIGHT_EMA * observed))


def _device_assignment_counts(n, weights):
    assignment = _weighted_round_robin_assignment(n, weights)
    return assignment, np.bincount(
        assignment, minlength=len(np.asarray(weights)))


def _predicted_device_makespan(n, weights, throughput,
                               block_size=_DEVICE_LOCAL_BLOCK_SIZE):
    """Block-aware predicted wall time for one population evaluation."""
    throughput = np.asarray(throughput, dtype=float)
    if (throughput.ndim != 1 or not len(throughput)
            or np.any(~np.isfinite(throughput))
            or np.any(throughput <= 0.0)):
        return np.inf
    _, counts = _device_assignment_counts(n, weights)
    capacities = np.asarray([
        _device_block_capacity(count, block_size) for count in counts])
    return float(np.max(capacities / throughput))


def _predicted_rebalance_gain(n, current_weights, proposed_weights,
                              throughput,
                              block_size=_DEVICE_LOCAL_BLOCK_SIZE):
    """Fractional makespan reduction from a proposed device assignment."""
    current = _predicted_device_makespan(
        n, current_weights, throughput, block_size)
    proposed = _predicted_device_makespan(
        n, proposed_weights, throughput, block_size)
    if not np.isfinite(current) or not current > 0.0:
        return 0.0
    return max(0.0, (current - proposed) / current)


def _use_shared_pmap(n_dev, devices):
    """Compile homogeneous multi-device work once with pmap."""
    devices = tuple(devices[:n_dev])
    return (n_dev > 1 and len(devices) == n_dev
            and len({device.device_kind for device in devices}) == 1)


def _make_batched_fitness(fitness_one, n_dev, devices,
                          candidates_per_wave=_CANDIDATES_PER_GPU_WAVE):
    """Return ``batch_eval(x_normed (M,D)[, desc]) -> fitness (M,)``.

    A fixed-size executable evaluates ``candidates_per_wave`` candidates
    concurrently, eliminating population- and rebalance-dependent input
    shapes. Homogeneous multi-GPU jobs compile one shared ``pmap`` executable;
    heterogeneous devices retain concurrent device-local JITs and weighted
    round-robin assignment. Output is restored to input order on the host.
    Finite padding is excluded from the exact archive and algorithmic NFE
    count.
    """
    candidates_per_wave = int(candidates_per_wave)
    if (candidates_per_wave < 1
            or _DEVICE_LOCAL_BLOCK_SIZE % candidates_per_wave != 0):
        raise ValueError(
            "candidates_per_wave must be a positive divisor of the "
            f"device block size ({_DEVICE_LOCAL_BLOCK_SIZE}).")
    vmapped = jax.vmap(fitness_one)

    def per_device(block):
        if candidates_per_wave == _DEVICE_LOCAL_BLOCK_SIZE:
            return vmapped(block)
        waves = block.reshape(
            -1, candidates_per_wave, block.shape[-1])
        return jax.lax.map(vmapped, waves).reshape(-1)

    n_dev = max(1, int(n_dev))
    devices = tuple(devices[:n_dev])
    if len(devices) < n_dev:
        devices = tuple(jax.local_devices()[:n_dev])
    if len(devices) != n_dev:
        raise ValueError(
            f"Requested {n_dev} evaluator devices, found {len(devices)}.")
    shared_pmap = _use_shared_pmap(n_dev, devices)
    shared_runner = (jax.pmap(per_device, devices=devices)
                     if shared_pmap else None)
    runners = (() if shared_pmap else
               tuple(jax.jit(per_device, device=device)
                     for device in devices))
    state = {
        "assignment_weights": np.full(n_dev, 1.0 / n_dev),
        "profile_weights": np.full(n_dev, 1.0 / n_dev),
        "profile_samples": 0,
        "warmed": False,
        "rebalance_attempts": 0,
        "rebalances": 0,
        "candidates_per_wave": candidates_per_wave,
        "last_rebalance_gain": 0.0,
        "total_candidates": np.zeros(n_dev, dtype=np.int64),
        "total_real_candidates": np.zeros(n_dev, dtype=np.int64),
        "total_seconds": np.zeros(n_dev),
        "last_candidates": np.zeros(n_dev, dtype=int),
        "last_real_candidates": np.zeros(n_dev, dtype=int),
        "last_seconds": np.zeros(n_dev),
    }

    def batch_eval(x, desc=None):
        x = np.asarray(x)
        n = x.shape[0]
        if n == 0:
            return np.empty(0, dtype=float)
        assignment, counts = _device_assignment_counts(
            n, state["assignment_weights"])
        indices = [np.flatnonzero(assignment == i) for i in range(n_dev)]
        shards = []
        capacities = np.empty(n_dev, dtype=int)
        for i, idx in enumerate(indices):
            capacity = _device_block_capacity(len(idx))
            capacities[i] = capacity
            shard = np.broadcast_to(
                x[:1], (capacity,) + x.shape[1:]).copy()
            shard[:len(idx)] = x[idx]
            shards.append(shard)

        if shared_pmap:
            shared_capacity = int(np.max(capacities))
            for i, shard in enumerate(shards):
                if shard.shape[0] < shared_capacity:
                    padded = np.broadcast_to(
                        x[:1], (shared_capacity,) + x.shape[1:]).copy()
                    padded[:shard.shape[0]] = shard
                    shards[i] = padded
            capacities[:] = shared_capacity
            start = time.perf_counter()
            parts = []
            for start_idx in range(0, shared_capacity,
                                   _DEVICE_LOCAL_BLOCK_SIZE):
                blocks = np.stack([
                    shard[start_idx:start_idx + _DEVICE_LOCAL_BLOCK_SIZE]
                    for shard in shards])
                parts.append(shared_runner(blocks))
            jax.block_until_ready(parts[-1])
            values = np.concatenate(
                [np.asarray(part) for part in parts], axis=1)
            duration = time.perf_counter() - start
            results = [(values[i], duration) for i in range(n_dev)]
        else:
            def run_device(i):
                start = time.perf_counter()
                parts = []
                for start_idx in range(0, capacities[i],
                                       _DEVICE_LOCAL_BLOCK_SIZE):
                    block = shards[i][
                        start_idx:start_idx + _DEVICE_LOCAL_BLOCK_SIZE]
                    parts.append(
                        runners[i](jax.device_put(block, devices[i])))
                jax.block_until_ready(parts[-1])
                values = np.concatenate([np.asarray(part) for part in parts])
                return values, time.perf_counter() - start

            if n_dev == 1:
                results = [run_device(0)]
            else:
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=n_dev) as executor:
                    results = list(executor.map(run_device, range(n_dev)))

        durations = np.empty(n_dev)
        was_warmed = state["warmed"]
        for i, (values, duration) in enumerate(results):
            if was_warmed:
                state["total_candidates"][i] += capacities[i]
                state["total_real_candidates"][i] += counts[i]
                state["total_seconds"][i] += duration
            state["last_candidates"][i] = capacities[i]
            state["last_real_candidates"][i] = counts[i]
            state["last_seconds"][i] = duration
            durations[i] = duration
        state["warmed"] = True

        out = _restore_device_outputs(
            [result[0] for result in results], indices, n)

        # Exclude compilation and calls too small to exercise every device
        # from the throughput model.  A proposed change is adopted only when
        # it survives fixed-block padding and clears the material-gain gate.
        if (not shared_pmap and was_warmed and np.all(counts > 0)
                and np.all(durations > 0.0)):
            throughput = capacities / durations
            observed = _project_device_weights(throughput)
            proposed = _updated_device_profile_weights(
                state["profile_weights"], observed,
                state["profile_samples"])
            gain = _predicted_rebalance_gain(
                n, state["assignment_weights"], proposed, throughput)
            state["profile_weights"] = proposed
            state["profile_samples"] += 1
            state["rebalance_attempts"] += 1
            state["last_rebalance_gain"] = gain
            if gain >= _DEVICE_REBALANCE_MIN_GAIN:
                state["assignment_weights"] = proposed
                state["rebalances"] += 1
        return out

    def device_profile():
        return {
            "assignment_weights": state["assignment_weights"].copy(),
            "profile_weights": state["profile_weights"].copy(),
            "total_candidates": state["total_candidates"].copy(),
            "total_real_candidates": state[
                "total_real_candidates"].copy(),
            "total_seconds": state["total_seconds"].copy(),
            "last_candidates": state["last_candidates"].copy(),
            "last_real_candidates": state[
                "last_real_candidates"].copy(),
            "last_seconds": state["last_seconds"].copy(),
            "profile_samples": state["profile_samples"],
            "block_size": _DEVICE_LOCAL_BLOCK_SIZE,
            "candidates_per_wave": state["candidates_per_wave"],
            "execution_mode": ("shared pmap" if shared_pmap
                               else "device-local jit"),
            "rebalance_attempts": state["rebalance_attempts"],
            "rebalances": state["rebalances"],
            "last_rebalance_gain": state["last_rebalance_gain"],
        }

    batch_eval.device_profile = device_profile

    return batch_eval


_ARCHIVE_FINGERPRINT_VERSION = "blake2b64-v1"
_ARCHIVE_FINGERPRINT_BATCH = 10_000
# Buffered exact-value rows are flushed to SQLite in one commit at each
# checkpoint. The cap bounds RAM during the one-off Sobol screen.
_ARCHIVE_WRITE_BUFFER_ROWS = 65_536


class _ExactArchive:
    """Persistent exact-value cache for one optimiser checkpoint."""

    def __init__(self, path, dimension, resume=False, objective_policy=None):
        archive_exists = path != ":memory:" and os.path.exists(path)
        if archive_exists and not resume:
            os.unlink(path)
            archive_exists = False
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS evaluations "
            "(point BLOB PRIMARY KEY, fitness REAL NOT NULL) WITHOUT ROWID")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS evaluation_fingerprints "
            "(fingerprint INTEGER PRIMARY KEY)")
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='dimension'").fetchone()
        if row is not None and int(row[0]) != dimension:
            raise ValueError("Exact-evaluation archive dimension mismatch.")
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('dimension', ?)",
            (str(dimension),))
        if objective_policy is not None:
            row = self.connection.execute(
                "SELECT value FROM metadata "
                "WHERE key='objective_policy'").fetchone()
            if archive_exists and (row is None or row[0] != objective_policy):
                self.connection.close()
                raise ValueError(
                    "Exact-evaluation archive objective policy is "
                    f"{row[0] if row else 'legacy'!r}, requested "
                    f"{objective_policy!r}; start a fresh run.")
            self.connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES "
                "('objective_policy', ?)", (objective_policy,))
        self.connection.commit()

        backfill_start = time.perf_counter()
        row = self.connection.execute(
            "SELECT value FROM metadata "
            "WHERE key='fingerprint_version'").fetchone()
        if row is None or row[0] != _ARCHIVE_FINGERPRINT_VERSION:
            self.connection.execute("DELETE FROM evaluation_fingerprints")
            last_point = None
            while True:
                if last_point is None:
                    rows = self.connection.execute(
                        "SELECT point FROM evaluations ORDER BY point "
                        "LIMIT ?", (_ARCHIVE_FINGERPRINT_BATCH,)).fetchall()
                else:
                    rows = self.connection.execute(
                        "SELECT point FROM evaluations WHERE point > ? "
                        "ORDER BY point LIMIT ?",
                        (last_point,
                         _ARCHIVE_FINGERPRINT_BATCH)).fetchall()
                if not rows:
                    break
                self.connection.executemany(
                    "INSERT OR IGNORE INTO evaluation_fingerprints VALUES "
                    "(?)",
                    [(self._fingerprint(point),) for point, in rows])
                self.connection.commit()
                last_point = rows[-1][0]
            self.connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES "
                "('fingerprint_version', ?)",
                (_ARCHIVE_FINGERPRINT_VERSION,))
            self.connection.commit()
        self.index_backfill_seconds = time.perf_counter() - backfill_start

        t0 = time.perf_counter()
        # Only deterministic 64-bit fingerprints are loaded from NFS.  A
        # collision causes a redundant exact BLOB lookup, never a false cache
        # hit, because the evaluations table remains authoritative.
        self._known_fingerprints = {
            int(row[0]) for row in self.connection.execute(
                "SELECT fingerprint FROM evaluation_fingerprints")}
        self.index_load_seconds = time.perf_counter() - t0
        self.hits = 0
        self.evaluations = 0
        # Buffered (key -> fitness) rows and their fingerprints, committed on
        # flush(). Lookups consult this buffer before touching SQL.
        self._pending = {}
        self._pending_fingerprints = []
        self.reset_timing()

    @staticmethod
    def _key(point):
        return np.ascontiguousarray(point, dtype=np.float64).tobytes()

    @staticmethod
    def _fingerprint(key):
        digest = hashlib.blake2b(
            key, digest_size=8, person=b"CANDEL-DE-v1").digest()
        return int.from_bytes(digest, "little", signed=True)

    def __call__(self, batch_eval, points, desc=None):
        lookup_start = time.perf_counter()
        x = np.asarray(points)
        out = np.empty(x.shape[0], dtype=float)
        pending = {}
        for i, point in enumerate(x):
            key = self._key(point)
            if key in pending:
                pending[key].append(i)
                self.hits += 1
                continue
            pending[key] = [i]

        # A fingerprint hit may reference a buffered, not-yet-written row, so
        # the RAM buffer is consulted before issuing any SQL SELECT.
        possible_hits = [
            key for key in pending
            if self._fingerprint(key) in self._known_fingerprints]
        cached = {}
        sql_hits = []
        for key in possible_hits:
            if key in self._pending:
                cached[key] = self._pending[key]
            else:
                sql_hits.append(key)
        for start in range(0, len(sql_hits), 512):
            keys = sql_hits[start:start + 512]
            placeholders = ",".join("?" for _ in keys)
            rows = self.connection.execute(
                "SELECT point, fitness FROM evaluations WHERE point IN ("
                + placeholders + ")", keys)
            cached.update(rows)
        missing_keys = []
        missing = []
        for key, indices in pending.items():
            if key in cached:
                out[indices] = cached[key]
                self.hits += 1
            else:
                missing_keys.append(key)
                missing.append(indices[0])
        self.lookup_seconds += time.perf_counter() - lookup_start
        self.lookup_keys += len(pending)
        self.lookup_queries += ((len(sql_hits) + 511) // 512)

        if missing:
            evaluation_start = time.perf_counter()
            values = np.asarray(batch_eval(
                jnp.asarray(x[missing]), desc=desc), dtype=float)
            self.evaluation_seconds += (
                time.perf_counter() - evaluation_start)
            values = np.where(np.isnan(values), np.inf, values)
            for key, value in zip(missing_keys, values):
                indices = pending[key]
                out[indices] = value
            write_start = time.perf_counter()
            new_fingerprints = [
                self._fingerprint(key) for key in missing_keys]
            for key, value in zip(missing_keys, values):
                self._pending[key] = float(value)
            self._pending_fingerprints.extend(new_fingerprints)
            self._known_fingerprints.update(new_fingerprints)
            self.write_seconds += time.perf_counter() - write_start
            self.evaluations += len(missing)
            if len(self._pending) > _ARCHIVE_WRITE_BUFFER_ROWS:
                self.flush()
        return out

    def flush(self):
        """Commit buffered rows in one transaction and clear the buffer."""
        if not self._pending:
            return
        write_start = time.perf_counter()
        self.connection.executemany(
            "INSERT INTO evaluations VALUES (?, ?)",
            list(self._pending.items()))
        self.connection.executemany(
            "INSERT OR IGNORE INTO evaluation_fingerprints VALUES (?)",
            [(fingerprint,) for fingerprint in self._pending_fingerprints])
        self.connection.commit()
        self._pending.clear()
        self._pending_fingerprints.clear()
        self.write_seconds += time.perf_counter() - write_start

    def reset_timing(self):
        """Reset per-optimisation-phase timing without changing counters."""
        self.lookup_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.write_seconds = 0.0
        self.lookup_keys = 0
        self.lookup_queries = 0

    def count(self):
        self.flush()
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM evaluations").fetchone()[0])

    def close(self):
        self.flush()
        self.connection.close()


def _lshade_draw_indices(order, n_pbest, n, n_union, rng):
    """Vectorised pbest/r1/r2 index draws for a whole L-SHADE population.

    ``pbest[i]`` is uniform over ``order[:n_pbest]`` excluding ``i``; ``r1[i]``
    uniform over ``[0, n)`` excluding ``{i, pbest[i]}``; ``r2[i]`` uniform over
    ``[0, n_union)`` excluding ``{i, pbest[i], r1[i]}``.  Each forbidden set is
    per-member; the constraints are enforced by masked rejection (redraw only
    the offending members), which terminates fast in expectation.
    """
    idx = np.arange(n)
    pool = order[:n_pbest]
    pbest = pool[rng.integers(n_pbest, size=n)]
    bad = pbest == idx
    while np.any(bad):
        sub = np.flatnonzero(bad)
        pbest[sub] = pool[rng.integers(n_pbest, size=sub.size)]
        bad = pbest == idx
    r1 = rng.integers(n, size=n)
    bad = (r1 == idx) | (r1 == pbest)
    while np.any(bad):
        sub = np.flatnonzero(bad)
        r1[sub] = rng.integers(n, size=sub.size)
        bad = (r1 == idx) | (r1 == pbest)
    r2 = rng.integers(n_union, size=n)
    bad = (r2 == idx) | (r2 == pbest) | (r2 == r1)
    while np.any(bad):
        sub = np.flatnonzero(bad)
        r2[sub] = rng.integers(n_union, size=sub.size)
        bad = (r2 == idx) | (r2 == pbest) | (r2 == r1)
    return pbest, r1, r2


def _lshade_trials(population, fitness, mutation_archive, m_f, m_cr, rng,
                   pbest_fraction=0.11):
    """Generate one current-to-pbest/1/bin L-SHADE trial population."""
    pop = np.asarray(population)
    n, dimension = pop.shape
    if n < 4:
        raise ValueError("L-SHADE requires at least four population members.")
    archive = np.asarray(mutation_archive).reshape(-1, dimension)
    union = np.vstack([pop, archive]) if archive.size else pop
    order = np.argsort(fitness)
    n_pbest = max(2, min(n, int(np.ceil(pbest_fraction * n))))
    slot = rng.integers(len(m_f), size=n)

    # F: Cauchy proposal per member, redrawing only non-positive draws
    # (masked rejection), then clipped at 1.
    f = np.empty(n)
    bad = np.ones(n, dtype=bool)
    while np.any(bad):
        sub = np.flatnonzero(bad)
        proposal = (m_f[slot[sub]]
                    + 0.1 * np.tan(np.pi * (rng.random(sub.size) - 0.5)))
        ok = proposal > 0.0
        f[sub[ok]] = proposal[ok]
        bad[sub[ok]] = False
    np.minimum(f, 1.0, out=f)

    # CR: Gaussian around the memory value, zeroed where the slot is inactive.
    cr = np.clip(rng.normal(m_cr[slot], 0.1), 0.0, 1.0)
    cr[m_cr[slot] < 0.0] = 0.0

    pbest, r1, r2 = _lshade_draw_indices(order, n_pbest, n, len(union), rng)

    # Truncate the mutant to the population dtype before reflection, matching
    # the historical row-wise ``np.empty_like`` assignment.
    mutants = np.empty_like(pop)
    mutants[:] = (pop + f[:, None] * (pop[pbest] - pop)
                  + f[:, None] * (pop[r1] - union[r2]))

    # Triangle-wave fold of out-of-bounds values back into [0, 1], in the
    # population dtype (matches candel.inference.optimise._reflect_bounds).
    mutants = np.abs(mutants)
    cycle = np.floor(mutants).astype(np.int32)
    frac = mutants - np.floor(mutants)
    mutants = np.where(cycle % 2 == 0, frac, 1.0 - frac)
    cross = rng.random((n, dimension)) < cr[:, None]
    cross[np.arange(n), rng.integers(dimension, size=n)] = True
    return np.where(cross, mutants, pop), f, cr


def _update_lshade_memory(m_f, m_cr, memory_index, successful_f,
                          successful_cr, improvements):
    if not len(successful_f):
        return memory_index
    weights = improvements / np.sum(improvements)
    m_f[memory_index] = np.sum(weights * successful_f**2) / np.sum(
        weights * successful_f)
    if m_cr[memory_index] < 0.0 or np.max(successful_cr) == 0.0:
        m_cr[memory_index] = -1.0
    else:
        m_cr[memory_index] = np.sum(
            weights * successful_cr**2) / np.sum(
                weights * successful_cr)
    return (memory_index + 1) % len(m_f)


def _append_mutation_archive(archive, parents, capacity, rng):
    if not len(parents):
        return archive
    out = (np.vstack([archive, parents]) if len(archive)
           else np.asarray(parents))
    if len(out) > capacity:
        out = out[rng.choice(len(out), capacity, replace=False)]
    return out


def _linear_population_size(initial_size, minimum_size, evaluations,
                            reduction_evaluations):
    """L-SHADE population target at an algorithmic evaluation count.

    ``evaluations`` counts fitness evaluations of members of the DE population,
    including its initial population.  The separate Sobol pre-screen and
    Pesce/Reid reference score are deliberately outside this count.  Once the
    reduction horizon is exhausted, the population remains at ``minimum_size``
    until patience or the generation ceiling stops the run.
    """
    fraction = np.clip(evaluations / reduction_evaluations, 0.0, 1.0)
    target = int(round(
        initial_size + fraction * (minimum_size - initial_size)))
    return max(minimum_size, min(initial_size, target))


def _resolve_n_devices(requested):
    """``(n_dev, gpu_devices)`` for DE population sharding.

    Uses every visible local GPU by default; ``requested`` (``--n-devices``)
    caps it (1 forces the serial path).  Off-GPU -> ``(1, [])``.
    """
    gpus = [d for d in jax.local_devices() if d.platform == "gpu"]
    n = len(gpus) if gpus else 1
    if requested is not None:
        if requested > n:
            fprint(f"--n-devices {requested} > {n} available; using {n}")
        else:
            n = max(1, requested)
    return n, gpus


# Physical VRAM (GB) for cluster GPUs whose JAX ``device_kind`` does NOT
# encode the size.  Datacenter cards put the size in the name and
# are parsed in ``_vram_gb_from_kind``; this table covers the consumer / older
# cards (glamdring gpulong/cmbgpu/optgpu; arc Titan/RTX8000/P100/L40S).  Values
# follow the glamdring ``--gputype`` labels.  Keys are lower-cased substrings.
#
# H100 (always 80) and V100 (16 default, 32 via --gpu-mem) are handled
# explicitly in ``_vram_gb_from_kind``; A100 (40/80) comes from the name.
_GPU_VRAM_GB = {
    # glamdring
    "2080": 12,     # RTX 2080 Ti      (gpulong, rtx2080with12gb)
    "3070": 8,      # RTX 3070         (gpulong, rtx3070with8gb)
    "3090": 24,     # RTX 3090         (cmbgpu,  rtx3090with24gb)
    "a6000": 48,    # RTX A6000        (optgpu / arc)
    # arc (single-variant cards whose device_kind omits the size)
    "titan": 24,    # Titan RTX        (Turing)
    "8000": 48,     # Quadro RTX 8000  (Turing)
    "p100": 16,     # Tesla P100       (Pascal)
    "l40": 48,      # L40 / L40S       (Lovelace)
}
# JAX will not allocate past this fraction of the card (XLA allocator cap).
_XLA_MEM_FRACTION = float(
    os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.75))
# Conservative budget for a real GPU we can neither name nor read via
# memory_stats (= the smallest card in the fleet, the RTX 3070).
_DEFAULT_VRAM_GB = 8.0


def _vram_gb_from_kind(kind, gpu_mem_gb=None):
    """``(VRAM GB, how it was determined)`` for a JAX ``device_kind``.

    H100 is always 80GB (the 94GB NVL is treated as 80).  V100 is 16GB by
    default and 32GB only when ``--gpu-mem 32`` selects the larger variant (arc
    has both and the name is not trusted).  Other datacenter names encode the
    size (``A100-SXM4-80GB``) -> read it off; else fall back to the
    ``_GPU_VRAM_GB`` table of known cluster cards.  Returns ``(None, why)``
    when the name is missing or unrecognised.
    """
    if not kind:
        return None, "no device name"
    low = kind.lower()
    if "h100" in low:
        return 80.0, "pinned (incl. the 94GB NVL)"
    if "v100" in low:
        if gpu_mem_gb == 32:
            return 32.0, "selected by --gpu-mem 32"
        return 16.0, "default; pass --gpu-mem 32 for the 32GB card"
    m = re.search(r"(\d+)\s?GB", kind)
    if m:
        return float(m.group(1)), "size read from the device name"
    for key, gb in _GPU_VRAM_GB.items():
        if key in low:
            return float(gb), f"_GPU_VRAM_GB['{key}'] table lookup"
    return None, "unrecognised name"


def _device_free_bytes():
    """Free device memory (bytes) from JAX memory_stats, or None (e.g. CPU)."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats or "bytes_limit" not in stats:
        return None
    return float(stats["bytes_limit"]) - float(stats.get("bytes_in_use", 0.0))


def _device_peak_gb():
    """Peak bytes-in-use on device 0 (GB), or None (e.g. CPU / no stats)."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    peak = stats.get("peak_bytes_in_use") if stats else None
    return None if peak is None else float(peak) / 1e9


def _device_budget_bytes(gpu_mem_gb=None):
    """Usable device bytes for JAX + a source label, or ``(None, reason)``.

    Resolution order: GPU name -> physical VRAM (scaled by the XLA allocator
    fraction); live ``memory_stats`` limit; a conservative ``_DEFAULT_VRAM_GB``
    for a real but unrecognised GPU (with a warning -- add it to
    ``_GPU_VRAM_GB`` or pass ``--gpu-mem``); finally ``(None, ...)`` off-GPU so
    the caller keeps its configured defaults.  ``gpu_mem_gb`` only selects the
    V100 16/32GB variant.
    """
    try:
        dev = jax.local_devices()[0]
        kind, platform = dev.device_kind, dev.platform
    except Exception:
        kind, platform = None, None
    total_gb, why = _vram_gb_from_kind(kind, gpu_mem_gb)
    if total_gb is not None:
        return (total_gb * 1e9 * _XLA_MEM_FRACTION,
                f"{kind} -> {total_gb:g}GB: {why}")
    free = _device_free_bytes()
    if free is not None:
        return free, f"memory_stats ({kind}: {why})"
    if platform == "gpu":
        return _DEFAULT_VRAM_GB * 1e9 * _XLA_MEM_FRACTION, (
            f"DEFAULT {_DEFAULT_VRAM_GB:g}GB -- UNRECOGNISED GPU '{kind}'; "
            "add "
            f"it to _GPU_VRAM_GB or pass --gpu-mem")
    return None, f"no device (CPU?); '{kind}'"


def _plan_de_batch(model, pop_size, mem_frac=0.7, k_live=8, gpu_mem_gb=None):
    """Estimate DE memory and pick ``(spot_batch, budget_known, info)``.

    The 2D-marginal eval holds ~``k_live`` live arrays of shape
    ``(batch, n_r, n_phi)`` with ``n_r = n_r_local + n_r_global`` and ``n_phi``
    the widest spot class. The budget is the device VRAM (by GPU name, else
    ``memory_stats``). Returns ``(None, False, info)`` when it is
    unknown, so the caller keeps its configured defaults (e.g. CPU).
    ``gpu_mem_gb`` only selects the V100 16/32GB variant.

    Fixed-grid production evaluates one DE candidate per GPU wave. Peak
    partition bypasses this planner and evaluates its smaller eight-candidate
    block concurrently.
    """
    n_r = model._n_r_local + model._n_r_global
    n_phi = max(int(pc["sin_phi"].shape[0])
                for pc in model._phi_concat.values())
    max_group = max(model._n_sys, model._n_red, model._n_blue)
    dtype_bytes = 8 if jax.config.jax_enable_x64 else 4
    dt = "f64" if dtype_bytes == 8 else "f32"
    cell = k_live * n_r * n_phi * dtype_bytes
    free, src = _device_budget_bytes(gpu_mem_gb)
    grid = (f"one spot's (r,phi) grid = {cell / 1e6:.1f} MB "
            f"(n_r={n_r} radial x n_phi={n_phi} azimuth x {k_live} live "
            f"buffers, {dt})")
    if free is None:
        return None, False, (
            f"no GPU VRAM budget ({src}); keeping all spots per candidate; "
            f"{grid}")
    budget = mem_frac * free
    per_candidate_all = cell * max_group
    spot_batch = (None if budget >= per_candidate_all
                  else max(1, int(budget // cell)))
    spots = "all" if spot_batch is None else str(spot_batch)
    return spot_batch, True, (
        f"usable VRAM {free / 1e9:.1f} GB/GPU ({src}), DE plans to "
        f"{mem_frac:.0%} of that; {grid}; auto plan: "
        f"spot_batch={spots} (spots scored per candidate), "
        "one candidate per GPU wave, "
        f"of pop={pop_size}; biggest spot group={max_group}")


def _variant_suffix(model):
    parts = []
    if model.use_ecc:
        parts.append("ecc")
    if model.use_quadratic_warp:
        parts.append("qw")
    return "_" + "_".join(parts) if parts else ""


def _phi_integration_suffix(model):
    return ("_peakpartition"
            if model.phi_integration == "peak-partition" else "")


def _objective_policy(model):
    kernel = ":ecc_hybrid_qf1" if getattr(model, "use_ecc", False) else ""
    radial = (
        f"r{model._n_r_local}+{model._n_r_global}:"
        f"K{model._K_sigma:g}:"
        f"full{int(model._global_r_full_support)}:"
        f"asym{int(model._asymmetric_r_local)}:"
        f"width{model._scan_width_drop:g}")
    if model.phi_integration == "peak-partition":
        refine_scope = (
            ":rrhv" if (model._peak_r_refine_steps
                         and model._peak_r_refine_hv_only) else "")
        return (
            f"{_DE_PEAK_PARTITION_POLICY}{kernel}:"
            f"sys{model._n_phi_partition_sys}:"
            f"hv{model._n_phi_partition_hv}:"
            f"roots{model._phi_partition_root_capacity}:"
            f"rr{model._peak_r_refine_steps}x"
            f"{model._peak_r_refine_order}{refine_scope}:"
            f"rw{model._peak_r_width_steps}:"
            f"{radial}")
    return f"{_DE_OBJECTIVE_POLICY}{kernel}:{radial}"


def _init_block(gal_cfg, model):
    """Variant-specific [init...] block (init_ecc / init_qw / init_ecc_qw)
    selected by use_ecc/use_quadratic_warp, falling back to [init]."""
    suffix = _variant_suffix(model)
    if suffix:
        name = "init" + suffix
        if name in gal_cfg:
            fprint(f"init block: [{name}]")
            return gal_cfg[name]
        fprint(f"init block: [{name}] absent, falling back to [init]")
    return gal_cfg.get("init", {})


_DE_HISTORY_KEYS = ("history_generation", "history_logp", "history_D_A")


def _save_de_progress_plot(checkpoint_path, generation, logp, D_A):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    plot_path = os.path.splitext(checkpoint_path)[0] + "_progress.png"
    tmp = plot_path + ".tmp.png"
    figure = Figure(figsize=(9, 7))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    recent = slice(-100, None)
    for axis, x, y, ylabel in (
            (axes[0, 0], generation, logp, "Best logP"),
            (axes[0, 1], generation, D_A, r"Best $D_A$ [Mpc]"),
            (axes[1, 0], generation[recent], logp[recent], "Best logP"),
            (axes[1, 1], generation[recent], D_A[recent],
             r"Best $D_A$ [Mpc]")):
        axis.plot(x, y)
        axis.set(xlabel="Generation", ylabel=ylabel)
    figure.tight_layout()
    figure.savefig(tmp, dpi=300)
    os.replace(tmp, plot_path)
    return plot_path


def _load_de_history(checkpoint, generation, logp, D_A):
    if checkpoint is None:
        return [generation], [logp], [D_A]
    present = [key in checkpoint.files for key in _DE_HISTORY_KEYS]
    if not any(present):
        return [generation], [logp], [D_A]
    if not all(present):
        raise ValueError("Checkpoint DE progress history is incomplete.")
    history = [np.asarray(checkpoint[key]) for key in _DE_HISTORY_KEYS]
    if not history[0].size or len({values.size for values in history}) != 1:
        raise ValueError("Checkpoint DE progress history has invalid lengths.")
    if int(history[0][-1]) != int(generation):
        raise ValueError(
            "Checkpoint DE progress history does not end at its generation.")
    return tuple(values.tolist() for values in history)


def _save_de_checkpoint(path, population, fitness, best_solution,
                        best_fitness, generation, key,
                        gens_without_improvement, best_logp_so_far,
                        lo, hi, names, sizes, extra=None):
    tmp = path + ".tmp.npz"
    data = dict(
        population=np.asarray(population),
        fitness=np.asarray(fitness),
        best_solution=np.asarray(best_solution),
        best_fitness=np.asarray(best_fitness),
        generation_counter=np.asarray(generation),
        key=np.asarray(key),
        gens_without_improvement=np.array(gens_without_improvement),
        best_logp_so_far=np.array(best_logp_so_far),
        lo=lo, hi=hi,
        names=np.array(names, dtype=str),
        sizes=np.array(sizes),
    )
    if extra:
        data.update(extra)
    np.savez(tmp, **data)
    os.replace(tmp, path)
    if extra and all(key in extra for key in _DE_HISTORY_KEYS):
        try:
            plot_path = _save_de_progress_plot(
                path, *(extra[key] for key in _DE_HISTORY_KEYS))
        except Exception as error:
            fprint(f"WARNING: checkpoint saved but progress plot failed: "
                   f"{error}")
        else:
            fprint(f"  progress plot: {plot_path}")


def _load_de_checkpoint(path, lo, hi, names, sizes):
    d = np.load(path)
    if not np.allclose(d["lo"], lo) or not np.allclose(d["hi"], hi):
        raise ValueError("Checkpoint bounds do not match current model.")
    if list(d["names"]) != list(names) or list(d["sizes"]) != list(sizes):
        raise ValueError("Checkpoint parameter layout does not match.")
    expected_dtype = np.dtype(
        np.float64 if jax.config.jax_enable_x64 else np.float32)
    state_dtypes = {
        key: np.dtype(d[key].dtype) for key in ("population", "fitness")}
    if any(dtype != expected_dtype for dtype in state_dtypes.values()):
        saved = ", ".join(
            f"{key}={dtype.name}" for key, dtype in state_dtypes.items())
        raise ValueError(
            f"Checkpoint DE-state precision ({saved}) does not match the "
            f"current {expected_dtype.name} run; resume with the matching "
            "--f64 setting or start a fresh run.")
    return d


def _validate_de_checkpoint_policy(
        checkpoint, path, objective_policy=_DE_OBJECTIVE_POLICY):
    """Reject incompatible algorithm, objective, seed, or schedule state."""
    saved_algorithm = (
        str(np.asarray(checkpoint["algorithm"]).item())
        if "algorithm" in checkpoint.files else "classic")
    if saved_algorithm != _DE_ALGORITHM:
        raise ValueError(
            f"Checkpoint algorithm is {saved_algorithm!r}, requested "
            f"{_DE_ALGORITHM!r}.")
    saved_seed_policy = (
        str(np.asarray(checkpoint["seed_policy"]).item())
        if "seed_policy" in checkpoint.files else None)
    if saved_seed_policy is None:
        if not os.path.basename(path).endswith("_nopesce.npz"):
            raise ValueError(
                "Legacy L-SHADE checkpoint has no seed-policy marker; only "
                "a *_nopesce.npz checkpoint can be resumed.")
        fprint("Legacy *_nopesce checkpoint establishes the unseeded "
               "Pesce/Reid policy.")
    elif saved_seed_policy != _DE_SEED_POLICY:
        raise ValueError(
            f"Checkpoint seed policy is {saved_seed_policy!r}, requested "
            f"{_DE_SEED_POLICY!r}.")
    saved_schedule = (
        str(np.asarray(checkpoint["population_schedule"]).item())
        if "population_schedule" in checkpoint.files else None)
    if saved_schedule != _DE_POPULATION_SCHEDULE:
        raise ValueError(
            "Checkpoint population schedule is "
            f"{saved_schedule or 'legacy generation-linear'!r}, requested "
            f"{_DE_POPULATION_SCHEDULE!r}; start a fresh run.")
    saved_objective = (
        str(np.asarray(checkpoint["objective_policy"]).item())
        if "objective_policy" in checkpoint.files else None)
    if saved_objective != objective_policy:
        raise ValueError(
            f"Checkpoint objective policy is {saved_objective or 'legacy'!r}, "
            f"requested {objective_policy!r}; start a fresh run.")


def _screen_eval(batch_eval, x, desc, chunk=512):
    """Evaluate ``x`` in slices with progress/ETA prints.

    Slicing only changes call granularity: the same points reach the same
    per-candidate executable, so values, archive contents and NFE counts
    are identical to a single call.
    """
    n = x.shape[0]
    if n <= chunk:
        return np.asarray(batch_eval(x, desc=desc))
    t0 = time.time()
    parts = []
    for i in range(0, n, chunk):
        parts.append(np.asarray(batch_eval(x[i:i + chunk], desc=desc)))
        done = min(i + chunk, n)
        rate = done / (time.time() - t0)
        fprint(f"{desc}: {done}/{n} ({rate:.2f} cand/s, "
               f"ETA {(n - done) / rate / 60.0:.1f} min)")
    return np.concatenate(parts)


def _make_de_initial_population(batch_eval, lo, hi, pop_size, seed,
                                N_sobol, min_dist_frac,
                                seed_points=None):
    scale = hi - lo
    D = lo.size

    sampler = Sobol(d=D, scramble=True, seed=seed)
    sobol_01 = sampler.random(N_sobol)
    sobol_points = lo + sobol_01 * scale
    sobol_normed = jnp.asarray((sobol_points - lo) / scale)

    t0 = time.time()
    logp_all = -_screen_eval(batch_eval, sobol_normed, "Sobol candidates")
    valid = np.isfinite(logp_all)
    logp_all = np.where(valid, logp_all, -np.inf)
    best_sobol = logp_all[valid].max() if np.any(valid) else -np.inf
    fprint(f"Sobol candidates done in {time.time() - t0:.1f}s "
           f"({valid.sum()}/{N_sobol} valid, "
           f"best logP={best_sobol:.1f})")

    seeds = np.empty((0, D))
    if seed_points is not None:
        seeds = np.atleast_2d(np.asarray(seed_points, dtype=float))
        ok = (np.all(np.isfinite(seeds), axis=1)
              & np.all((seeds >= lo) & (seeds <= hi), axis=1))
        if np.any(~ok):
            fprint(f"Skipped {np.sum(~ok)} DE seed point(s) outside bounds.")
        seeds = seeds[ok][:pop_size]

    n_sobol = pop_size - seeds.shape[0]
    if n_sobol:
        selected = _select_distinct(
            sobol_points, logp_all, n_sobol, min_dist_frac)
        population = np.asarray((sobol_points[selected] - lo) / scale)
    else:
        population = np.empty((0, D))
    if seeds.shape[0]:
        population = np.vstack(((seeds - lo) / scale, population))
    fitness = _screen_eval(
        batch_eval, jnp.asarray(population), "Initial-population candidates")
    jax.block_until_ready(fitness)
    if population.shape[0] != pop_size:
        raise RuntimeError(
            f"DE initial population has {population.shape[0]} members, "
            f"expected {pop_size}.")
    if seeds.shape[0]:
        fprint(f"Initial DE population: {population.shape[0]} members "
               f"({seeds.shape[0]} seeded, {n_sobol} Sobol)")
    else:
        fprint(f"Initial DE population: {population.shape[0]} Sobol members")
    return jnp.asarray(population), jnp.asarray(fitness)


# Per-observable noise floors, in the order candel_theta_from_point emits them.
_PESCE_FLOOR_UNITS = (("sigma_x_floor", "uas"), ("sigma_y_floor", "uas"),
                      ("sigma_v_sys", "km/s"), ("sigma_v_hv", "km/s"),
                      ("sigma_a_floor", "km/s/yr"))
_PESCE_FLOOR_NAMES = tuple(name for name, _ in _PESCE_FLOOR_UNITS)
_FLOOR_UNIT = dict(_PESCE_FLOOR_UNITS)


def _run_de(target, opt_cfg, seed, n_dev=1, devices=(), checkpoint_path=None,
            resume_path=None, checkpoint_interval=900.0,
            seed_points=None, fixed_params=None, reference_params=None,
            reference_status=(), objective_policy=_DE_OBJECTIVE_POLICY,
            peak_candidates_per_wave=None):
    log2_N = int(opt_cfg.get("log2_N", 16))
    pop_size = int(opt_cfg.get("pop_size", 1000))
    max_generations = int(opt_cfg.get("max_generations", 5000))
    patience = int(opt_cfg.get("patience", 100))
    sobol_n_sigma = opt_cfg.get("sobol_n_sigma", 5)
    min_dist_frac = float(opt_cfg.get("min_dist_frac", 0.005))
    log_every = int(opt_cfg.get("log_every", 1))
    min_pop_size = int(opt_cfg.get("min_pop_size", 4))
    if not 4 <= min_pop_size <= pop_size:
        raise ValueError("min_pop_size must be between 4 and pop_size.")
    try:
        reduction_evaluations = int(
            opt_cfg["population_reduction_evaluations"])
    except KeyError as exc:
        raise KeyError(
            "Missing [optimise].population_reduction_evaluations") from exc
    if reduction_evaluations < pop_size:
        raise ValueError(
            "population_reduction_evaluations must be at least pop_size.")

    fixed = dict(fixed_params) if fixed_params else {}
    names, sizes, lo, hi = _layout(target, sobol_n_sigma, fixed=fixed)
    scale = hi - lo
    D = len(names)
    distance_name = "D_A" if "D_A" in names else "D_c"
    distance_idx = names.index(distance_name)
    N_sobol = 2 ** log2_N
    ckpt = None
    if resume_path is not None:
        ckpt = _load_de_checkpoint(resume_path, lo, hi, names, sizes)
        _validate_de_checkpoint_policy(
            ckpt, resume_path, objective_policy=objective_policy)
    # seed_points arrive in full target.names order; drop the fixed columns.
    if fixed and seed_points is not None:
        free_idx = [target.names.index(n) for n in names]
        seed_points = np.asarray(seed_points)[:, free_idx]

    candidates_per_wave = _de_candidates_per_wave(
        target.model, peak_candidates_per_wave)
    fsection("L-SHADE MAP optimizer")
    fprint(f"{D}D, pop={pop_size}, max_generations={max_generations}, "
           f"patience={patience}, "
           f"{candidates_per_wave} candidate"
           f"{'s' if candidates_per_wave != 1 else ''} per GPU wave")
    fprint(f"current-to-pbest/1, success-history F/CR, "
           f"linear pop {pop_size}->{min_pop_size} over "
           f"the first {reduction_evaluations:,} DE candidate evaluations")
    fprint("population-reduction horizon only; candidate evaluations are not "
           "capped and do not terminate the optimiser")
    fprint("seed policy: data-derived ridge + Sobol only; "
           "Pesce/Reid is scored as a reference and never inserted")
    fprint("initial DE population: seed points + scrambled Sobol candidates"
           if seed_points is not None
           else "initial DE population: scrambled Sobol candidates only")
    for name, lower, upper in zip(names, lo, hi):
        fprint(f"  {name:20s}: [{lower:.4g}, {upper:.4g}]")
    if fixed:
        fprint("  fixed at Pesce/Reid values (not searched):")
        for k, v in fixed.items():
            fprint(f"    {k:16s} = {float(np.asarray(v)):8.4g} "
                   f"{_FLOOR_UNIT.get(k, '')}")

    logp = _make_logp(target, names, fixed=fixed)

    def fitness_one(x_normed):
        x = jnp.asarray(lo) + x_normed * jnp.asarray(scale)
        return -logp(x)

    batch_eval = _make_batched_fitness(
        fitness_one, n_dev, devices,
        candidates_per_wave=candidates_per_wave)
    archive_path = (checkpoint_path + ".sqlite"
                    if checkpoint_path is not None else ":memory:")
    exact_archive = _ExactArchive(
        archive_path, D, resume=resume_path is not None,
        objective_policy=objective_policy)

    # Reproduce the archive's historical device-transfer downcast: the SQLite
    # cache holds float64, but x64-disabled runs consumed it as float32.
    host_dtype = np.float64 if jax.config.jax_enable_x64 else np.float32

    def exact_eval(points, desc=None):
        return np.asarray(
            exact_archive(batch_eval, points, desc=desc), dtype=host_dtype)

    t0 = time.time()
    reference_logp = None
    if reference_params is None:
        # A new process always needs to compile, independently of archive
        # resume state. Homogeneous GPUs share one pmap compilation; mixed
        # devices retain one device-local compilation each.
        warmup = batch_eval(jnp.full((1, D), 0.5))
        jax.block_until_ready(warmup)
    else:
        reference_point = _normalise_theta_point(
            reference_params, names, lo, hi)[None]
        evaluations_before = exact_archive.evaluations
        reference_fitness = exact_eval(reference_point)
        jax.block_until_ready(reference_fitness)
        if exact_archive.evaluations == evaluations_before:
            # The SQLite sidecar supplied the value, but this process still
            # needs the same executable for the population that follows.
            warmup = batch_eval(jnp.asarray(reference_point))
            jax.block_until_ready(warmup)
        reference_logp = -float(np.asarray(reference_fitness)[0])
    execution_mode = batch_eval.device_profile()["execution_mode"]
    fprint(f"JIT compiled in {time.time() - t0:.1f}s "
           f"(n_dev={n_dev}, {execution_mode}, "
           f"fixed {_DEVICE_LOCAL_BLOCK_SIZE}-candidate "
           f"device block; {candidates_per_wave} candidate"
           f"{'s' if candidates_per_wave != 1 else ''} per wave)")
    peak = _device_peak_gb()
    if peak is not None:
        fprint(f"device-0 peak after warmup: {peak:.1f} GB "
               f"({_DEVICE_LOCAL_BLOCK_SIZE}-candidate fixed block across "
               f"{n_dev} device(s))")
    if reference_logp is not None:
        fsection("Pesce/Reid reference probability (never seeded)")
        if reference_status:
            fprint("defaulted " + ", ".join(reference_status))
        fprint("A single point has zero probability mass in a continuous "
               "posterior; the comparable quantity is its posterior "
               "density.")
        fprint("Pesce/Reid fixed globals: exact all-spot marginal "
               "unnormalised log density "
               f"logP = {reference_logp:.6f}")
        fprint("Reference score reused the exact DE objective executable and "
               "was not inserted into the initial population.")

    if resume_path is not None:
        key = np.asarray(ckpt["key"])
        gen_start = int(ckpt["generation_counter"])
        gens_without_improvement = int(ckpt["gens_without_improvement"])
        best_logp_so_far = float(ckpt["best_logp_so_far"])
        population = np.asarray(ckpt["population"])
        fitness = np.asarray(ckpt["fitness"])
        best_solution = np.asarray(ckpt["best_solution"])
        best_fitness = np.asarray(ckpt["best_fitness"])
        initial_pop_size = int(
            ckpt["initial_pop_size"] if "initial_pop_size" in ckpt.files
            else len(population))
        if initial_pop_size != pop_size:
            raise ValueError(
                f"Checkpoint initial population is {initial_pop_size}, "
                f"requested {pop_size}.")
        required_schedule = {
            "de_evaluations", "population_reduction_evaluations",
            "min_pop_size",
        }
        missing_schedule = required_schedule.difference(ckpt.files)
        if missing_schedule:
            raise ValueError("L-SHADE checkpoint is missing: "
                             + ", ".join(sorted(missing_schedule)))
        de_evaluations = int(ckpt["de_evaluations"])
        saved_reduction_evaluations = int(
            ckpt["population_reduction_evaluations"])
        saved_min_pop_size = int(ckpt["min_pop_size"])
        if saved_reduction_evaluations != reduction_evaluations:
            raise ValueError(
                "Checkpoint population-reduction horizon is "
                f"{saved_reduction_evaluations}, requested "
                f"{reduction_evaluations}.")
        if saved_min_pop_size != min_pop_size:
            raise ValueError(
                f"Checkpoint minimum population is {saved_min_pop_size}, "
                f"requested {min_pop_size}.")
        fprint(f"Resumed from {resume_path} at generation {gen_start}")
    else:
        population, fitness = _make_de_initial_population(
            exact_eval, lo, hi, pop_size, seed, N_sobol,
            min_dist_frac, seed_points=seed_points)
        population = np.asarray(population)
        fitness = np.asarray(fitness)
        key = jax.random.PRNGKey(seed)
        initial_pop_size = pop_size
        best_idx = int(np.argmin(np.asarray(fitness)))
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        gen_start = 0
        best_logp_so_far = -float(best_fitness)
        gens_without_improvement = 0
        # Algorithmic NFE starts with the selected population.  The Sobol
        # pre-screen is an initialisation aid, not part of L-SHADE's population
        # reduction clock.
        de_evaluations = initial_pop_size

    rng = np.random.default_rng(seed)
    mutation_archive = np.empty((0, D))
    m_f = np.full(6, 0.5)
    m_cr = np.full(6, 0.5)
    memory_index = 0
    if resume_path is not None:
        required = {"rng_state", "mutation_archive", "m_f", "m_cr",
                    "memory_index"}
        missing = required.difference(ckpt.files)
        if missing:
            raise ValueError(
                "L-SHADE checkpoint is missing: " + ", ".join(missing))
        rng.bit_generator.state = json.loads(str(
            np.asarray(ckpt["rng_state"]).item()))
        mutation_archive = np.asarray(ckpt["mutation_archive"])
        m_f = np.asarray(ckpt["m_f"])
        m_cr = np.asarray(ckpt["m_cr"])
        memory_index = int(ckpt["memory_index"])

    def best_D_A():
        distance = float(
            lo[distance_idx]
            + float(best_solution[distance_idx]) * scale[distance_idx])
        if distance_name == "D_c":
            distance = float(_D_A_from_D_c(target.model, distance))
        return distance

    history_generation, history_logp, history_D_A = _load_de_history(
        ckpt, gen_start, -float(best_fitness), best_D_A())

    def checkpoint_extra():
        return {
            "algorithm": np.asarray(_DE_ALGORITHM),
            "seed_policy": np.asarray(_DE_SEED_POLICY),
            "population_schedule": np.asarray(_DE_POPULATION_SCHEDULE),
            "objective_policy": np.asarray(objective_policy),
            "initial_pop_size": np.asarray(initial_pop_size),
            "min_pop_size": np.asarray(min_pop_size),
            "de_evaluations": np.asarray(de_evaluations),
            "population_reduction_evaluations": np.asarray(
                reduction_evaluations),
            "rng_state": np.asarray(json.dumps(rng.bit_generator.state)),
            "mutation_archive": np.asarray(mutation_archive),
            "m_f": np.asarray(m_f),
            "m_cr": np.asarray(m_cr),
            "memory_index": np.asarray(memory_index),
            "history_generation": np.asarray(history_generation),
            "history_logp": np.asarray(history_logp),
            "history_D_A": np.asarray(history_D_A),
        }

    if resume_path is not None:
        fprint("Exact-archive fingerprint index: "
               f"{len(exact_archive._known_fingerprints):,} entries loaded "
               f"in {exact_archive.index_load_seconds:.2f}s; "
               f"legacy backfill={exact_archive.index_backfill_seconds:.2f}s")
    exact_archive.reset_timing()
    phase_timing = {
        "trials": 0.0,
        "update": 0.0,
        "checkpoint": 0.0,
    }
    timed_generations = 0

    def report_timing(label):
        host_seconds = (
            phase_timing["trials"] + phase_timing["update"]
            + phase_timing["checkpoint"] + exact_archive.lookup_seconds
            + exact_archive.write_seconds)
        per_generation = host_seconds / max(1, timed_generations)
        fprint(
            f"  {label}: {timed_generations} generations; "
            f"exact-eval={exact_archive.evaluation_seconds:.2f}s, "
            f"host={host_seconds:.2f}s ({per_generation:.3f}s/gen): "
            f"trials={phase_timing['trials']:.2f}s, "
            f"lookup={exact_archive.lookup_seconds:.2f}s "
            f"({exact_archive.lookup_queries} SQL queries/"
            f"{exact_archive.lookup_keys} keys), "
            f"archive-write={exact_archive.write_seconds:.2f}s, "
            f"update={phase_timing['update']:.2f}s, "
            f"checkpoint={phase_timing['checkpoint']:.2f}s")
        if hasattr(batch_eval, "device_profile"):
            profile = batch_eval.device_profile()
            throughput = np.divide(
                profile["total_candidates"], profile["total_seconds"],
                out=np.zeros_like(profile["total_seconds"]),
                where=profile["total_seconds"] > 0.0)

            def fmt(values):
                return "[" + ", ".join(
                    f"{value:.3f}" for value in values) + "]"

            fprint(
                "  device balance: assigned="
                f"{fmt(profile['assignment_weights'])}, profiled="
                f"{fmt(profile['profile_weights'])}, throughput="
                f"{fmt(throughput)} candidate/s, last="
                f"{fmt(profile['last_seconds'])}s "
                f"({profile['profile_samples']} timing samples; "
                f"fixed block={profile['block_size']}; rebalances="
                f"{profile['rebalances']}/"
                f"{profile['rebalance_attempts']}, last predicted gain="
                f"{profile['last_rebalance_gain']:.1%})")

    fsection(f"DE generations (L-SHADE, pop={len(population)}, "
             f"max_generations={max_generations}, "
             f"start_generation={gen_start})")
    last_ckpt = time.time()
    final_gen = gen_start
    de_progress = trange(
        max_generations - gen_start, desc="DE generations", unit="gen")
    for step in de_progress:
        gen = gen_start + step
        final_gen = gen + 1
        current_size = len(population)
        phase_start = time.perf_counter()
        trials, trial_f, trial_cr = _lshade_trials(
            population, fitness, mutation_archive, m_f, m_cr, rng)
        phase_timing["trials"] += time.perf_counter() - phase_start

        trial_fitness = exact_eval(trials)
        phase_start = time.perf_counter()
        de_evaluations += current_size
        improved = trial_fitness < fitness
        if np.any(improved):
            old_fitness = fitness
            mutation_archive = _append_mutation_archive(
                mutation_archive, population[improved],
                current_size, rng)
            memory_index = _update_lshade_memory(
                m_f, m_cr, memory_index, trial_f[improved],
                trial_cr[improved],
                old_fitness[improved] - trial_fitness[improved])
        population = np.where(improved[:, None], trials, population)
        fitness = np.where(improved, trial_fitness, fitness)

        target_size = _linear_population_size(
            initial_pop_size, min_pop_size, de_evaluations,
            reduction_evaluations)
        if target_size < len(population):
            keep = np.argsort(fitness)[:target_size]
            population = population[keep]
            fitness = fitness[keep]
            if len(mutation_archive) > target_size:
                mutation_archive = mutation_archive[rng.choice(
                    len(mutation_archive), target_size, replace=False)]

        gen_best_idx = int(np.argmin(fitness))
        gen_best_fitness = fitness[gen_best_idx]
        if float(gen_best_fitness) < float(best_fitness):
            best_fitness = gen_best_fitness
            best_solution = population[gen_best_idx]

        current_best = -float(best_fitness)
        history_generation.append(gen + 1)
        history_logp.append(current_best)
        history_D_A.append(best_D_A())
        if current_best > best_logp_so_far + 0.1:
            best_logp_so_far = current_best
            gens_without_improvement = 0
        else:
            gens_without_improvement += 1

        if log_every > 0 and (gen + 1) % log_every == 0:
            best_d_norm = float(best_solution[distance_idx])
            best_d = float(lo[distance_idx]
                           + best_d_norm * scale[distance_idx])
            de_progress.set_postfix_str(
                f"logP={current_best:.2f}, pop={len(population)}, "
                f"{distance_name}={best_d:.2f}, "
                f"candidate_evals={de_evaluations}, "
                f"stale={gens_without_improvement}/{patience}")
        phase_timing["update"] += time.perf_counter() - phase_start
        timed_generations += 1

        if (checkpoint_path is not None
                and time.time() - last_ckpt >= checkpoint_interval):
            # Commit buffered exact values so the archive is never staler than
            # the checkpoint it serves on resume.
            exact_archive.flush()
            phase_start = time.perf_counter()
            _save_de_checkpoint(
                checkpoint_path, population, fitness, best_solution,
                best_fitness, gen + 1, key, gens_without_improvement,
                best_logp_so_far, lo, hi, names, sizes,
                extra=checkpoint_extra())
            phase_timing["checkpoint"] += time.perf_counter() - phase_start
            last_ckpt = time.time()
            fprint(f"  checkpoint: generation {gen + 1}")
            report_timing("cumulative timing")

        if gens_without_improvement >= patience:
            fprint(f"  converged at generation {final_gen}")
            break

    if checkpoint_path is not None:
        exact_archive.flush()
        phase_start = time.perf_counter()
        _save_de_checkpoint(
            checkpoint_path, population, fitness, best_solution,
            best_fitness, final_gen, key, gens_without_improvement,
            best_logp_so_far, lo, hi, names, sizes,
            extra=checkpoint_extra())
        phase_timing["checkpoint"] += time.perf_counter() - phase_start
    report_timing("final timing")
    fprint(f"Exact archive: {exact_archive.count()} unique evaluations, "
           f"{exact_archive.hits} cache hits")
    exact_archive.close()

    x_best = np.asarray(lo + jnp.asarray(best_solution) * scale)
    params_best = _flat_to_theta(jnp.asarray(x_best), names)
    params_best.update(fixed)
    theta = target.complete_params(params_best)
    phys_args, phys_kw = target.model.phys_from_params_jax(theta, target.h)
    # This final diagnostic repeats the complete conditional-radius search.
    # Keeping it eager can dispatch thousands of tiny GPU operations after an
    # otherwise finished DE run (especially for NGC4258).  Stage it as one
    # executable, just like the likelihood used above.
    r_ang = jax.jit(target.model.conditional_r_ang_map)(phys_args, phys_kw)
    best_logp = -float(best_fitness)
    output = _theta_to_output(
        {k: np.asarray(jax.device_get(v)) for k, v in theta.items()},
        np.asarray(jax.device_get(r_ang)))
    return output, best_logp, {
        "generations": final_gen,
        "reference_logp": reference_logp,
    }


def _run_fixed_globals(target, init_params, data_only=False):
    params = {
        name: jnp.asarray(init_params[name]) for name in target.names}

    def score(sampled):
        completed = target.complete_params(sampled)
        lp, ll, _, _ = _logp_2d_terms(target, completed)
        return ll if data_only else lp + ll

    # The fixed-global/Pesce score uses the identical objective as DE, but it
    # used to execute as a sequence of eager JAX operations.  Large exact spot
    # batches then kept group intermediates alive simultaneously and could OOM
    # even though the compiled DE evaluator fit.  JIT compilation lets XLA
    # schedule and reuse those buffers just as it does for DE.
    logp = jax.jit(score)(params)
    logp = jax.block_until_ready(logp)
    theta = target.complete_params(params)
    phys_args, phys_kw = target.model.phys_from_params_jax(theta, target.h)
    r_ang = jax.jit(target.model.conditional_r_ang_map)(phys_args, phys_kw)
    return _theta_to_output(
        {k: np.asarray(jax.device_get(v)) for k, v in theta.items()},
        np.asarray(jax.device_get(r_ang))), float(jax.device_get(logp)), 0


def _pesce_init(target, galaxy, master):
    helper_dir = os.path.join(os.path.dirname(__file__), "check_reid")
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    from pesce_globals import candel_theta_from_point  # noqa: E402
    from pesce_globals import paper_point

    try:
        point, status = paper_point(galaxy, master)
    except KeyError as exc:
        raise KeyError(f"No Pesce/Reid fixed globals for {galaxy}") from exc
    if point is None:
        raise KeyError(
            f"Cannot build Pesce/Reid fixed globals for {galaxy}: {status}")
    return candel_theta_from_point(point, galaxy, master, target), status


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run 2D-marginal MAP optimisation for one megamaser disk.")
    parser.add_argument("galaxy", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fix-globals", action="store_true",
                        help="Skip the DE search: hold the disc globals at "
                             "the config [init] point, compute the per-spot "
                             "conditional r_ang MAP, and score "
                             "logP = log-prior + sum_i 2D (r,phi) marginal. "
                             "Prints the resulting [init] block "
                             "(globals + r_ang).")
    parser.add_argument("--fix-globals-pesce", action="store_true",
                        help="Like --fix-globals but hold the globals at the "
                             "published Pesce/Reid values (with their exact "
                             "paper error floors) and score the data-only "
                             "sum_i 2D (r,phi) marginal (no global priors).")
    parser.add_argument("--fix-floors-pesce", action="store_true",
                        help="Run the full DE but hold the five error floors "
                             "(sigma_x_floor, sigma_y_floor, sigma_v_sys, "
                             "sigma_v_hv, sigma_a_floor) fixed at the "
                             "published Pesce/Reid values; all other globals "
                             "searched.")
    parser.add_argument("--checkpoint-interval-minutes", type=float,
                        default=15.0)
    parser.add_argument("--f64", action="store_true", default=_ENABLE_F64)
    parser.add_argument("--no-ecc", action="store_true")
    parser.add_argument("--add-ecc", action="store_true")
    parser.add_argument("--no-quadratic-warp", action="store_true")
    parser.add_argument("--add-quadratic-warp", action="store_true")
    parser.add_argument("--mass-parameterization",
                        choices=("eta", "log_mbh"), default=None,
                        help="Global mass coordinate for the optimiser. "
                             "Default: config value, eta in config_maser.")
    parser.add_argument("--init-strategy",
                        choices=("median", "config"), default=None,
                        help="Fixed-global source for --fix-globals, and "
                             "target initial check otherwise. DE never uses "
                             "the Pesce/Reid point for initialisation; "
                             "--fix-globals-pesce scores that reference "
                             "without running DE.")
    parser.add_argument("--spot-batch", type=int, default=None)
    parser.add_argument(
        "--phi-integration", choices=("fixed-grid", "peak-partition"),
        default=None,
        help="Phi integration used by the DE 2D marginal. Default: config "
             "value (fixed-grid in config_maser.toml). peak-partition uses "
             "fixed-size numerical peak searches in both systemic "
             "half-planes and one half-plane per HV group.")
    parser.add_argument(
        "--peak-candidates-per-wave", type=int, choices=(1, 2, 4, 8),
        default=None,
        help="Concurrent candidates per GPU in peak-partition mode. "
             "Default: 8. Use 2 or 4 for hardware calibration if 8 lowers "
             "throughput; this does not change the objective.")
    parser.add_argument("--gpu-mem", type=float, default=None,
                        help="GPU VRAM hint in GB for the auto batch planner. "
                             "Only disambiguates the V100 16/32GB variant "
                             "(32 selects the 32GB card); other GPUs are "
                             "auto-detected by name.")
    parser.add_argument("--log2-N", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--min-pop-size", type=int, default=None)
    parser.add_argument("--population-reduction-evaluations", type=int,
                        default=None,
                        help="DE-population fitness-evaluation horizon for "
                             "linear population reduction; independent of "
                             "the generation ceiling and patience. This is "
                             "not an NFE stopping criterion.")
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--n-devices", type=int, default=None,
                        help="Same-node GPUs for adaptive weighted round-"
                             "robin DE evaluation. Default: all visible "
                             "local GPUs; 1 forces the single-device path. "
                             "ARC: request N with submit.sh --gpu-count N "
                             "(-> --gres=gpu:N).")
    args = parser.parse_args(argv)

    if args.fix_floors_pesce and (args.fix_globals or args.fix_globals_pesce):
        raise SystemExit(
            "--fix-floors-pesce only applies to the DE; it cannot combine "
            "with --fix-globals/--fix-globals-pesce (those skip the DE).")
    master_cfg = _MASTER_CFG
    galaxies = master_cfg["model"]["galaxies"]
    if args.galaxy not in galaxies:
        raise SystemExit(
            f"Unknown galaxy {args.galaxy!r}. Available: {list(galaxies)}")
    if galaxies[args.galaxy].get("force_f64", False):
        args.f64 = True
        _late_f64_reason = f"forced for {args.galaxy}"
    else:
        _late_f64_reason = "--f64"

    if args.f64 and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
        print(f"float64 enabled ({_late_f64_reason})", flush=True)
    gcfg = galaxies[args.galaxy]
    inf_cfg = master_cfg["inference"]
    seed = args.seed if args.seed is not None else _required_inference(
        inf_cfg, "seed")

    _devs = jax.devices()
    _dev_names = ", ".join(d.device_kind for d in _devs)
    _precision = "float64" if jax.config.jax_enable_x64 else "float32"
    fprint(f"JAX platform: {jax.default_backend()}, devices: {_devs} "
           f"({_dev_names}), precision: {_precision}")

    n_dev, gpu_devices = _resolve_n_devices(args.n_devices)
    if n_dev > 1:
        if _use_shared_pmap(n_dev, gpu_devices):
            fprint(f"DE population assigned by round-robin over {n_dev} "
                   "homogeneous same-node GPU(s) with one shared pmap "
                   "executable.")
        else:
            fprint(f"DE population assigned by adaptive weighted round-robin "
                   f"over {n_dev} same-node GPU(s).")

    fsection(f"Loading {args.galaxy} data")
    data = load_megamaser_spots(
        data_path("data", "Megamaser"), args.galaxy,
        v_sys_obs=gcfg["v_sys_obs"])
    distance_bounds = _distance_bounds(gcfg)
    if distance_bounds is not None:
        data["D_lo"], data["D_hi"], source = distance_bounds
        fprint(f"D prior bounds ({source}): "
               f"[{data['D_lo']:.1f}, {data['D_hi']:.1f}] Mpc")

    opt_cfg = dict(master_cfg.get("optimise", {}))
    for arg_name, cfg_name in (
            ("log2_N", "log2_N"),
            ("pop_size", "pop_size"),
            ("min_pop_size", "min_pop_size"),
            ("population_reduction_evaluations",
             "population_reduction_evaluations"),
            ("max_generations", "max_generations"),
            ("patience", "patience"),
            ("log_every", "log_every")):
        value = getattr(args, arg_name)
        if value is not None:
            opt_cfg[cfg_name] = int(value)
    config = {
        "inference": master_cfg["inference"],
        "model": dict(master_cfg["model"]),
        "io": master_cfg["io"],
        "optimise": opt_cfg,
    }
    config["model"]["galaxies"] = {
        g: dict(blk) for g, blk in master_cfg["model"]["galaxies"].items()}
    gal_blk = config["model"]["galaxies"][args.galaxy]
    if args.phi_integration is not None:
        gal_blk["phi_integration"] = args.phi_integration
    selected_phi_integration = gal_blk.get(
        "phi_integration", config["model"].get(
            "phi_integration", "fixed-grid"))
    if (args.peak_candidates_per_wave is not None
            and selected_phi_integration != "peak-partition"):
        raise SystemExit(
            "--peak-candidates-per-wave requires "
            "--phi-integration peak-partition.")
    if (selected_phi_integration == "peak-partition"
            and jax.default_backend() != "gpu"):
        # CPU PjRt reproducibly exited while deserialising the former
        # partition executable. Keep the conservative CPU guard; GPU jobs
        # retain the persistent cache so restarts can reuse their executable.
        jax.config.update("jax_enable_compilation_cache", False)

    def _grid_val(key):
        return gal_blk.get(key, config["model"].get(key))
    fprint("DE grid (from config): " + ", ".join(
        f"{k}={_grid_val(k)}" for k in
        ("n_phi_hv_high", "n_phi_hv_low", "n_phi_sys",
         "n_r_local", "n_r_global", "n_refine_steps")))
    if args.no_ecc:
        config["model"]["galaxies"][args.galaxy]["use_ecc"] = False
    if args.add_ecc:
        config["model"]["galaxies"][args.galaxy]["use_ecc"] = True
    if args.no_quadratic_warp:
        config["model"]["galaxies"][args.galaxy]["use_quadratic_warp"] = False
    if args.add_quadratic_warp:
        config["model"]["galaxies"][args.galaxy]["use_quadratic_warp"] = True
    if args.mass_parameterization is not None:
        config["model"]["galaxies"][args.galaxy][
            "mass_parameterization"] = args.mass_parameterization
    cfg_spot_batch = gal_blk.get("conditional_spot_batch", None)
    cfg_spot_batch = (None if cfg_spot_batch is None
                      else int(cfg_spot_batch))
    if args.spot_batch is not None:
        config["model"]["galaxies"][args.galaxy]["conditional_spot_batch"] = (
            int(args.spot_batch))
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp)
    tmp.close()
    try:
        model = MaserDiskModel(tmp.name, data)
    finally:
        os.unlink(tmp.name)

    init_strategy = str(args.init_strategy or _required_inference(
        inf_cfg, "init_strategy")).lower()
    if init_strategy == "reid":
        raise SystemExit(
            "DE initialisation never uses the Pesce/Reid point. Choose "
            "--init-strategy median or config; use --fix-globals-pesce only "
            "to score the Pesce/Reid reference.")
    init_params = _make_init(
        model, _init_block(config["model"]["galaxies"][args.galaxy], model),
        init_strategy,
        int(_required_inference(inf_cfg, "init_num_samples")),
        jax.random.PRNGKey(seed))
    h = float(get_nested(model.config, "model/H0_ref", 73.0)) / 100.0
    # The five float32 MCP galaxies have measured live peaks far below the
    # smallest production GPU and therefore default to true all-spot exact
    # evaluation.  Explicit CLI/per-galaxy controls still win, and other/f64
    # targets retain the conservative VRAM planner (notably NGC4258).
    if model.phi_integration == "peak-partition":
        plan_sb, plan_available = None, False
        plan_info = "peak-partition memory planner is not GPU-calibrated"
    else:
        plan_sb, plan_available, plan_info = _plan_de_batch(
            model, int(opt_cfg.get("pop_size", 1000)),
            gpu_mem_gb=args.gpu_mem)
    target_spot_batch, spot_batch_source = _de_spot_batch_policy(
        args.galaxy, bool(jax.config.jax_enable_x64), args.spot_batch,
        cfg_spot_batch, plan_sb)
    target = MaserBlackJaxTarget(
        model, h, init_params, spot_batch=target_spot_batch)
    pesce_logp = None
    pesce_params = None
    pesce_status = ()
    fixed_floors = None
    fixed_globals = args.fix_globals or args.fix_globals_pesce
    if args.fix_globals_pesce:
        try:
            init_params, pesce_status = _pesce_init(target, args.galaxy,
                                                    master_cfg)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
        if pesce_status:
            fprint("Pesce fixed globals: defaulted "
                   + ", ".join(pesce_status))
        fprint("Pesce fixed globals: exact published floors; data-only score")
    elif not args.fix_globals:
        try:
            pesce_params, pesce_status = _pesce_init(
                target, args.galaxy, master_cfg)
            if args.fix_floors_pesce:
                fixed_floors = {n: pesce_params[n] for n in _PESCE_FLOOR_NAMES
                                if n in target.names}
        except KeyError as exc:
            fprint(f"Pesce/Reid baseline unavailable: {exc}")
            if args.fix_floors_pesce:
                raise SystemExit(
                    f"--fix-floors-pesce needs Pesce floors: {exc}") from exc
    r_refinement = (
        "three-point global-scan interpolation"
        if model.phi_integration == "peak-partition"
        else f"Brent steps={model._n_refine_steps}")
    fprint("inner solve: joint 2D (r_ang, phi) marginal per spot, "
           f"n_r_global={model._n_r_global}, "
           f"r-centre={r_refinement}")
    fprint(f"phi integration: {model.phi_integration}")
    if model.phi_integration == "peak-partition":
        cache_state = ("enabled on GPU" if jax.default_backend() == "gpu"
                       else "disabled on CPU")
        fprint("persistent JAX compilation cache: " + cache_state)

    fsection(
        f"{'Fixed-global latent MAP' if fixed_globals else 'DE MAP'} "
        f"({args.galaxy}, {data['n_spots']} spots)")
    t0 = time.time()
    if fixed_globals:
        if args.resume:
            fprint("--resume ignored with fixed globals")
        init_params, best_logp, run_info = _run_fixed_globals(
            target, init_params, data_only=args.fix_globals_pesce)
        run_summary = (
            "Pesce globals fixed; data likelihood"
            if args.fix_globals_pesce else "globals fixed")
    else:
        ckpt_dir = results_path(
            master_cfg["io"].get("root_output", "results/Megamaser"),
            "de_checkpoints", args.galaxy)
        os.makedirs(ckpt_dir, exist_ok=True)
        floor_suffix = "_pescefloors" if args.fix_floors_pesce else ""
        ckpt_path = os.path.join(
            ckpt_dir,
            f"de_ckpt_rmap{_variant_suffix(model)}"
            f"{_phi_integration_suffix(model)}{floor_suffix}"
            "_lshade_nopesce.npz")
        resume_path = (
            ckpt_path if args.resume and os.path.isfile(ckpt_path)
            else None)
        if args.resume and resume_path is None:
            fprint(
                f"--resume: no checkpoint found at {ckpt_path}, "
                "starting fresh")
        # Pesce/Reid is passed only to the shared objective for an independent
        # reference score. The population is structurally limited to
        # data-derived and Sobol points.
        pop_size = int(opt_cfg.get("pop_size", 1000))
        data_seeds, seed_info = _data_driven_seed(
            model, target, init_params, _h_ref(model) * 100.0,
            max(1, pop_size // 2), seed + 1,
            sobol_n_sigma=opt_cfg.get("sobol_n_sigma", 5))
        if data_seeds is None:
            fprint(f"data seed unavailable ({seed_info}); Sobol only")
        else:
            fprint(f"data seed: {seed_info}")
        seed_points = _initial_de_seed_points(data_seeds)
        fprint("Pesce/Reid reference is not part of the DE initial "
               "population.")
        sb_flag = " [--spot-batch]" if args.spot_batch is not None else ""
        candidates_per_wave = _de_candidates_per_wave(
            model, args.peak_candidates_per_wave)
        fprint(f"DE batching: {candidates_per_wave} candidate"
               f"{'s' if candidates_per_wave != 1 else ''} per GPU wave, "
               f"spot_batch={target.spot_batch}{sb_flag} "
               f"({spot_batch_source}; None = all spots at once)")
        if model.phi_integration == "peak-partition":
            fprint("DE memory plan: peak-partition uses the selected "
                   "candidate wave and explicit/per-galaxy spot batching")
        n_waves = _DEVICE_LOCAL_BLOCK_SIZE // candidates_per_wave
        fprint(f"DE device executable: fixed {_DEVICE_LOCAL_BLOCK_SIZE}-"
               f"candidate block in {n_waves} "
               f"wave{'s' if n_waves != 1 else ''} "
               "(padding is excluded from NFE/archive)")
        if _use_shared_pmap(n_dev, gpu_devices):
            fprint("DE multi-GPU compilation: one shared pmap executable "
                   "for homogeneous devices")
        fprint("DE reduction compilation: phi integrands materialised before "
               "log-sum reductions")
        # Echo the conservative planner for diagnostics even when the measured
        # float32 all-spots policy intentionally supersedes it.
        if plan_available:
            fprint("DE memory plan: " + plan_info)
            if "UNRECOGNISED" in plan_info:
                fprint(
                    "  WARNING: GPU not recognised; using a conservative "
                    f"{_DEFAULT_VRAM_GB:g}GB budget. Add it to _GPU_VRAM_GB "
                    "in run_de_map.py, or pass --gpu-mem GB, for the size.")
            n_r = model._n_r_local + model._n_r_global
            n_phi = max(int(pc["sin_phi"].shape[0])
                        for pc in model._phi_concat.values())
            max_group = max(model._n_sys, model._n_red, model._n_blue)
            dtype_bytes = 8 if jax.config.jax_enable_x64 else 4
            cell = 8 * n_r * n_phi * dtype_bytes
            sb = max_group if target.spot_batch is None else target.spot_batch
            pred_gb = sb * cell / 1e9
            fprint(f"DE memory estimate: ~{pred_gb:.1f} GB/GPU peak "
                   f"= {sb} spots x "
                   f"{cell / 1e9:.2f} GB/spot "
                   f"({'f64' if dtype_bytes == 8 else 'f32'} grid "
                   f"n_r={n_r} x n_phi={n_phi}); "
                   f"{n_dev} GPUs share the DE population -> ~{n_dev}x "
                   f"throughput.")
        init_params, best_logp, run_info = _run_de(
            target, opt_cfg, seed, n_dev=n_dev, devices=gpu_devices,
            checkpoint_path=ckpt_path, resume_path=resume_path,
            checkpoint_interval=args.checkpoint_interval_minutes * 60.0,
            seed_points=seed_points, fixed_params=fixed_floors,
            reference_params=pesce_params,
            reference_status=pesce_status,
            objective_policy=_objective_policy(model),
            peak_candidates_per_wave=args.peak_candidates_per_wave)
        pesce_logp = run_info["reference_logp"]
        run_summary = f"generations = {run_info['generations']}"
    dt = time.time() - t0

    fsection(f"MAP results ({args.galaxy}, {dt:.0f}s)")
    label = "logL" if args.fix_globals_pesce else "logP"
    fprint(f"best {label} = {best_logp:.2f}; {run_summary}")
    if pesce_logp is not None and not fixed_globals:
        delta = best_logp - pesce_logp
        fprint(f"DE - Pesce/Reid baseline = {delta:.2f}")
        log10_ratio = -delta / np.log(10.0)
        if -300.0 <= log10_ratio <= 300.0:
            ratio = 10.0 ** log10_ratio
            fprint("Pesce/Reid density / DE-best density = "
                   f"{ratio:.4g} (log10 ratio = {log10_ratio:.3f})")
        else:
            fprint("Pesce/Reid density / DE-best density: "
                   f"log10 ratio = {log10_ratio:.3f}")
        if delta < -1e-4:
            fprint("WARNING: DE best is below the Pesce/Reid baseline.")
    for key, value in sorted(init_params.items()):
        value = np.asarray(value)
        if value.ndim == 0:
            fprint(f"  {key:20s} = {float(value):12.4f}")
        else:
            fprint(f"  {key:20s} = [{value.size} values]")

    lines = [f"\n[model.galaxies.{args.galaxy}.init{_variant_suffix(model)}]"]
    for key, value in sorted(init_params.items()):
        value = np.asarray(value)
        if value.ndim == 0:
            lines.append(f"{key} = {round(float(value), 4)}")
        else:
            vals = ", ".join(str(round(float(x), 4)) for x in value)
            lines.append(f"{key} = [{vals}]")
    fprint("MAP init (copy into config_maser.toml manually if desired):")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
