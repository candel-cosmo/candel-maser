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
upward).  The objective is multimodal, so the global search is differential
evolution; the optional hybrid uses spot-minibatch Adam only to polish diverse
elites, with exact all-spot acceptance.  The phi/r grid is taken from the
per-galaxy ``config_maser.toml`` settings (same grid the MCMC and convergence
checks use).
"""
import argparse
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
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import tomli_w  # noqa: E402
from scipy.stats.qmc import Sobol  # noqa: E402
from tqdm import trange  # noqa: E402

from candel.inference.optimise import _prior_bounds  # noqa: E402
from candel.inference.optimise import _reflect_bounds  # noqa: E402,E501
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
    groups = model._build_conditional_r_grids(
        phys_args[2], phys_args[3], phys_args[4], phys_args[16],
        phys_args[8], phys_args[15], phys_args, phys_kw)
    ll = model._sum_phi_marginal(
        groups, phys_args, phys_kw, spot_batch=target.spot_batch,
        remat=False)  # DE is gradient-free: skip rematerialisation overhead
    lp = _global_logprior(target, theta, ll.dtype)
    return lp, ll, phys_args, phys_kw


def _logp_2d_minibatch(target, theta, batch_positions):
    """Stratified spot-minibatch estimate with the production grids."""
    model = target.model
    phys_args, phys_kw = model.phys_from_params_jax(theta, target.h)
    groups = model._build_conditional_r_grids(
        phys_args[2], phys_args[3], phys_args[4], phys_args[16],
        phys_args[8], phys_args[15], phys_args, phys_kw)
    ll = jnp.asarray(0.0, dtype=jnp.asarray(phys_args[2]).dtype)
    for group, pos in zip(groups, batch_positions):
        type_key, idx, r_ang, log_w_r = group
        sub = (type_key, idx[pos], r_ang[pos], log_w_r[pos])
        group_ll = model._sum_phi_marginal(
            [sub], phys_args, phys_kw, spot_batch=target.spot_batch,
            remat=True)
        ll = ll + (idx.size / pos.size) * group_ll
    return _global_logprior(target, theta, ll.dtype) + ll


def _make_logp(target, names, fixed=None):
    fixed = dict(fixed) if fixed else {}

    def logp_constrained(x):
        params = _flat_to_theta(x, names)
        params.update(fixed)
        theta = target.complete_params(params)
        lp, ll, _, _ = _logp_2d_terms(target, theta)
        return lp + ll

    return logp_constrained


def _eval_chunks(fn, x, chunk, desc=None):
    n = x.shape[0]
    n_pad = (-n) % chunk
    if n_pad:
        pad = jnp.broadcast_to(x[:1], (n_pad,) + x.shape[1:])
        x = jnp.concatenate([x, pad], axis=0)
    parts = []
    for i in trange(
            0, x.shape[0], chunk,
            total=x.shape[0] // chunk,
            desc=desc,
            disable=desc is None):
        y = fn(x[i:i + chunk])
        parts.append(y)
        jax.block_until_ready(y)
    return jnp.concatenate(parts, axis=0)[:n]


def _make_batched_fitness(fitness_one, n_dev, eval_chunk, devices):
    """Return ``batch_eval(x_normed (M,D)[, desc]) -> fitness (M,)``.

    ``n_dev <= 1`` keeps the serial ``jit(vmap)`` path, host-chunked by
    ``eval_chunk`` (unchanged behaviour).  ``n_dev > 1`` shards the candidate
    axis over ``devices`` with ``pmap`` and runs ``lax.map`` over
    ``eval_chunk``-sized sub-batches inside each device, so one dispatch covers
    the whole array with a single host sync (bandwidth-friendly).  Padding to a
    multiple of the device/chunk group is internal; the pad is finite (a copy
    of row 0) and sliced off.
    """
    vmapped = jax.vmap(fitness_one)
    if n_dev <= 1:
        jitted = jax.jit(vmapped)

        def batch_eval(x, desc=None):
            return _eval_chunks(jitted, x, eval_chunk, desc=desc)

        return batch_eval

    def per_device(shard):                       # (per_dev, D), per_dev%ec==0
        n_sub = shard.shape[0] // eval_chunk
        sub = shard.reshape(n_sub, eval_chunk, shard.shape[-1])
        return jax.lax.map(vmapped, sub).reshape(-1)

    pmapped = jax.pmap(per_device, devices=list(devices)[:n_dev])
    group = n_dev * eval_chunk

    def batch_eval(x, desc=None):
        n = x.shape[0]
        n_pad = (-n) % group
        if n_pad:
            pad = jnp.broadcast_to(x[:1], (n_pad,) + x.shape[1:])
            x = jnp.concatenate([x, pad], axis=0)
        per_dev = x.shape[0] // n_dev
        out = pmapped(
            x.reshape(n_dev, per_dev, x.shape[-1]))   # (n_dev,per_dev)
        return jax.block_until_ready(out).reshape(-1)[:n]

    return batch_eval


class _ExactArchive:
    """Persistent exact-value cache for one optimiser checkpoint."""

    def __init__(self, path, dimension, resume=False):
        if path != ":memory:" and not resume and os.path.exists(path):
            os.unlink(path)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS evaluations "
            "(point BLOB PRIMARY KEY, fitness REAL NOT NULL) WITHOUT ROWID")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='dimension'").fetchone()
        if row is not None and int(row[0]) != dimension:
            raise ValueError("Exact-evaluation archive dimension mismatch.")
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('dimension', ?)",
            (str(dimension),))
        self.connection.commit()
        self.hits = 0
        self.evaluations = 0

    @staticmethod
    def _key(point):
        return np.ascontiguousarray(point, dtype=np.float64).tobytes()

    def __call__(self, batch_eval, points, desc=None):
        x = np.asarray(points)
        out = np.empty(x.shape[0], dtype=float)
        pending = {}
        for i, point in enumerate(x):
            key = self._key(point)
            if key in pending:
                pending[key].append(i)
                self.hits += 1
                continue
            row = self.connection.execute(
                "SELECT fitness FROM evaluations WHERE point=?", (key,)
            ).fetchone()
            if row is None:
                pending[key] = [i]
            else:
                out[i] = row[0]
                self.hits += 1
        if pending:
            keys = list(pending)
            missing = [indices[0] for indices in pending.values()]
            values = np.asarray(batch_eval(
                jnp.asarray(x[missing]), desc=desc), dtype=float)
            values = np.where(np.isnan(values), np.inf, values)
            for indices, value in zip(pending.values(), values):
                out[indices] = value
            self.connection.executemany(
                "INSERT INTO evaluations VALUES (?, ?)",
                [(key, float(value)) for key, value in zip(keys, values)])
            self.connection.commit()
            self.evaluations += len(missing)
        return jnp.asarray(out)

    def count(self):
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM evaluations").fetchone()[0])

    def close(self):
        self.connection.close()


def _draw_reshuffled_batch(order, cursor, batch_size, rng):
    """Fixed-size no-replacement stream, reshuffled only at epoch edges."""
    n = len(order)
    if cursor + batch_size <= n:
        return order[cursor:cursor + batch_size], order, cursor + batch_size
    tail = order[cursor:]
    new_order = rng.permutation(n)
    if tail.size:
        keep = ~np.isin(new_order, tail)
        new_order = np.concatenate([new_order[keep], new_order[~keep]])
    need = batch_size - tail.size
    return np.concatenate([tail, new_order[:need]]), new_order, need


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
    memory_slots = rng.integers(len(m_f), size=n)
    f = np.empty(n)
    cr = np.empty(n)
    mutants = np.empty_like(pop)

    def draw_index(limit, forbidden):
        while True:
            value = int(rng.integers(limit))
            if value not in forbidden:
                return value

    for i, slot in enumerate(memory_slots):
        value = -1.0
        while value <= 0.0:
            value = m_f[slot] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
        f[i] = min(value, 1.0)
        cr[i] = (0.0 if m_cr[slot] < 0.0 else
                 np.clip(rng.normal(m_cr[slot], 0.1), 0.0, 1.0))
        pbest_pool = order[:n_pbest]
        pbest_pool = pbest_pool[pbest_pool != i]
        pbest = int(rng.choice(pbest_pool))
        r1 = draw_index(n, {i, pbest})
        r2 = draw_index(len(union), {i, pbest, r1})
        mutants[i] = (pop[i] + f[i] * (pop[pbest] - pop[i])
                      + f[i] * (pop[r1] - union[r2]))

    mutants = np.asarray(_reflect_bounds(jnp.asarray(mutants)))
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


def _linear_population_size(initial_size, minimum_size, generation,
                            max_generations):
    fraction = min(1.0, generation / max(1, max_generations))
    return int(round(initial_size + fraction * (minimum_size - initial_size)))


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


def _batch_from_budget(budget, cell, max_group, pop_size):
    """Spots-first ``(spot_batch, eval_chunk)`` for a memory budget.

    ``cell`` is the peak bytes of one (spot x DE-member) 2D-marginal eval and
    ``max_group`` the largest spot class.  First try all spots in one shot and
    spend what is left on the DE-population chunk; only if a single member with
    all spots overflows do we shrink the spot batch (``eval_chunk`` then 1).
    ``spot_batch is None`` means "all spots".
    """
    per_member_all = cell * max_group
    if budget >= per_member_all:
        eval_chunk = max(1, min(int(pop_size), int(budget // per_member_all)))
        return None, eval_chunk
    return max(1, int(budget // cell)), 1


def _plan_de_batch(model, pop_size, mem_frac=0.7, k_live=8, gpu_mem_gb=None):
    """Estimate DE memory and pick ``(spot_batch, eval_chunk, info)``.

    The 2D-marginal eval holds ~``k_live`` live arrays of shape
    ``(batch, n_r, n_phi)`` with ``n_r = n_r_local + n_r_global`` and ``n_phi``
    the widest spot class (systemic).  The budget is the device VRAM (by GPU
    name, else ``memory_stats``).  Returns ``(None, None, info)`` when it is
    unknown, so the caller keeps its configured defaults (e.g. CPU).
    ``gpu_mem_gb`` only selects the V100 16/32GB variant.
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
        return None, None, (
            f"no GPU VRAM budget ({src}); keeping config eval_chunk, all "
            f"spots per candidate; {grid}")
    budget = mem_frac * free
    spot_batch, eval_chunk = _batch_from_budget(
        budget, cell, max_group, pop_size)
    spots = "all" if spot_batch is None else str(spot_batch)
    return spot_batch, eval_chunk, (
        f"usable VRAM {free / 1e9:.1f} GB/GPU ({src}), DE plans to "
        f"{mem_frac:.0%} of that; {grid}; auto plan: "
        f"spot_batch={spots} (spots scored per candidate), "
        f"eval_chunk={eval_chunk} (DE candidates scored per GPU pass), "
        f"of pop={pop_size}; biggest spot group={max_group}")


def _variant_suffix(model):
    parts = []
    if model.use_ecc:
        parts.append("ecc")
    if model.use_quadratic_warp:
        parts.append("qw")
    return "_" + "_".join(parts) if parts else ""


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


def _load_de_checkpoint(path, lo, hi, names, sizes):
    d = np.load(path)
    if not np.allclose(d["lo"], lo) or not np.allclose(d["hi"], hi):
        raise ValueError("Checkpoint bounds do not match current model.")
    if list(d["names"]) != list(names) or list(d["sizes"]) != list(sizes):
        raise ValueError("Checkpoint parameter layout does not match.")
    return d


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
    logp_all = -np.asarray(batch_eval(sobol_normed, desc="Sobol"))
    valid = np.isfinite(logp_all)
    logp_all = np.where(valid, logp_all, -np.inf)
    best_sobol = logp_all[valid].max() if np.any(valid) else -np.inf
    fprint(f"Sobol done in {time.time() - t0:.1f}s "
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
    fitness = np.asarray(batch_eval(jnp.asarray(population)))
    jax.block_until_ready(fitness)
    if population.shape[0] != pop_size:
        raise RuntimeError(
            f"DE initial population has {population.shape[0]} members, "
            f"expected {pop_size}.")
    if seeds.shape[0]:
        fprint(f"Initial population: {population.shape[0]} members "
               f"({seeds.shape[0]} seeded, {n_sobol} Sobol)")
    else:
        fprint(f"Initial population: {population.shape[0]} Sobol members")
    return jnp.asarray(population), jnp.asarray(fitness)


def _make_minibatch_value_grad(target, names, fixed, lo, scale):
    def loss(x_normed, batch_positions):
        x = jnp.asarray(lo) + x_normed * jnp.asarray(scale)
        params = _flat_to_theta(x, names)
        params.update(fixed)
        theta = target.complete_params(params)
        return -_logp_2d_minibatch(target, theta, batch_positions)

    return jax.jit(jax.value_and_grad(loss))


def _polish_elites(population, fitness, batch_eval, exact_archive,
                   value_grad, group_sizes, rng, n_elites, n_steps,
                   learning_rate, batch_fraction, min_dist_frac):
    """Minibatch-Adam selected diverse elites; exact-score their endpoints."""
    if n_steps <= 0 or n_elites <= 0:
        return population, fitness, np.empty((0, population.shape[1]))
    n_elites = min(int(n_elites), len(population))
    elite_idx = _select_distinct(
        np.asarray(population), -np.asarray(fitness), n_elites,
        min_dist_frac)
    starts = np.asarray(population)[elite_idx].copy()
    points = [jnp.asarray(point) for point in starts]
    optimiser = optax.adam(optax.cosine_decay_schedule(
        learning_rate, max(1, n_steps)))
    states = [optimiser.init(point) for point in points]
    batch_sizes = [max(1, min(n, int(np.ceil(batch_fraction * n))))
                   for n in group_sizes]
    orders = [rng.permutation(n) for n in group_sizes]
    cursors = [0] * len(group_sizes)

    for _ in range(n_steps):
        batches = []
        for i, (n, batch_size) in enumerate(zip(group_sizes, batch_sizes)):
            batch, orders[i], cursors[i] = _draw_reshuffled_batch(
                orders[i], cursors[i], batch_size, rng)
            batches.append(jnp.asarray(batch))
        batches = tuple(batches)
        for i, point in enumerate(points):
            _, grad = value_grad(point, batches)
            grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            updates, states[i] = optimiser.update(
                grad, states[i], params=point)
            points[i] = _reflect_bounds(
                optax.apply_updates(point, updates))

    endpoints = jnp.stack(points)
    endpoint_fitness = np.asarray(exact_archive(
        batch_eval, endpoints, desc="Adam exact"))
    population = np.asarray(population).copy()
    fitness = np.asarray(fitness).copy()
    accepted = endpoint_fitness < fitness[elite_idx]
    population[elite_idx[accepted]] = np.asarray(endpoints)[accepted]
    fitness[elite_idx[accepted]] = endpoint_fitness[accepted]
    fprint(f"Adam elites: {n_elites} x {n_steps} steps, "
           f"accepted {int(np.sum(accepted))}/{n_elites}, "
           f"best exact logP={-float(np.min(endpoint_fitness)):.2f}")
    return (jnp.asarray(population), jnp.asarray(fitness),
            starts[accepted])


# Per-observable noise floors, in the order candel_theta_from_point emits them.
_PESCE_FLOOR_UNITS = (("sigma_x_floor", "uas"), ("sigma_y_floor", "uas"),
                      ("sigma_v_sys", "km/s"), ("sigma_v_hv", "km/s"),
                      ("sigma_a_floor", "km/s/yr"))
_PESCE_FLOOR_NAMES = tuple(name for name, _ in _PESCE_FLOOR_UNITS)
_FLOOR_UNIT = dict(_PESCE_FLOOR_UNITS)


def _run_de(target, opt_cfg, seed, n_dev=1, devices=(), checkpoint_path=None,
            resume_path=None, checkpoint_interval=900.0,
            seed_points=None, fixed_params=None):
    algorithm = str(opt_cfg.get("algorithm", "classic")).lower()
    if algorithm not in ("classic", "lshade", "hybrid"):
        raise ValueError(
            "[optimise].algorithm must be classic, lshade, or hybrid.")
    adaptive = algorithm != "classic"
    use_adam = algorithm == "hybrid"
    log2_N = int(opt_cfg.get("log2_N", 16))
    pop_size = int(opt_cfg.get("pop_size", 1000))
    max_generations = int(opt_cfg.get("max_generations", 5000))
    patience = int(opt_cfg.get("patience", 100))
    eval_chunk = int(opt_cfg.get("eval_chunk", 5))
    sobol_n_sigma = opt_cfg.get("sobol_n_sigma", 5)
    min_dist_frac = float(opt_cfg.get("min_dist_frac", 0.005))
    log_every = int(opt_cfg.get("log_every", 1))
    mutation = float(opt_cfg.get("mutation", 0.8))
    crossover = float(opt_cfg.get("crossover", 0.7))
    min_pop_size = int(opt_cfg.get("min_pop_size", 4))
    adam_interval = int(opt_cfg.get("adam_interval", 10))
    adam_elites = int(opt_cfg.get("adam_elites", 4))
    adam_steps = int(opt_cfg.get("adam_steps", 20))
    adam_learning_rate = float(opt_cfg.get("adam_learning_rate", 0.005))
    adam_batch_fraction = float(opt_cfg.get("adam_batch_fraction", 0.5))
    if adaptive and not 4 <= min_pop_size <= pop_size:
        raise ValueError("min_pop_size must be between 4 and pop_size.")
    if use_adam and (adam_interval < 1 or not 0 < adam_batch_fraction <= 1):
        raise ValueError(
            "Hybrid Adam needs adam_interval >= 1 and "
            "0 < adam_batch_fraction <= 1.")

    fixed = dict(fixed_params) if fixed_params else {}
    names, sizes, lo, hi = _layout(target, sobol_n_sigma, fixed=fixed)
    scale = hi - lo
    D = len(names)
    distance_name = "D_A" if "D_A" in names else "D_c"
    distance_idx = names.index(distance_name)
    N_sobol = 2 ** log2_N
    # seed_points arrive in full target.names order; drop the fixed columns.
    if fixed and seed_points is not None:
        free_idx = [target.names.index(n) for n in names]
        seed_points = np.asarray(seed_points)[:, free_idx]

    fsection("DE MAP optimizer")
    fprint(f"algorithm={algorithm}; {D}D, pop={pop_size}, "
           f"max_gen={max_generations}, "
           f"patience={patience}, eval_chunk={eval_chunk}")
    if adaptive:
        fprint(f"L-SHADE: current-to-pbest/1, success-history F/CR, "
               f"linear pop {pop_size}->{min_pop_size}")
    if use_adam:
        fprint(f"elite Adam: every {adam_interval} generation(s), "
               f"{adam_elites} elites x {adam_steps} steps, "
               f"lr={adam_learning_rate:g}, "
               f"spot fraction={adam_batch_fraction:g}")
    fprint("initial population: seed points + scrambled Sobol screen"
           if seed_points is not None
           else "initial population: scrambled Sobol screen only")
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

    batch_eval = _make_batched_fitness(fitness_one, n_dev, eval_chunk, devices)
    archive_path = (checkpoint_path + ".sqlite"
                    if checkpoint_path is not None else ":memory:")
    exact_archive = _ExactArchive(
        archive_path, D, resume=resume_path is not None)

    def exact_eval(points, desc=None):
        return exact_archive(batch_eval, points, desc=desc)

    t0 = time.time()
    # Warm up the executable directly.  A resumed archive may already contain
    # the midpoint, but a new process still needs to compile the evaluator.
    _ = batch_eval(jnp.full((eval_chunk, D), 0.5))
    jax.block_until_ready(_)
    fprint(f"JIT compiled in {time.time() - t0:.1f}s "
           f"(n_dev={n_dev}, eval_chunk={eval_chunk})")
    peak = _device_peak_gb()
    if peak is not None:
        fprint(f"device-0 peak after warmup: {peak:.1f} GB "
               f"(one {n_dev * eval_chunk}-candidate wave)")

    if resume_path is not None:
        ckpt = _load_de_checkpoint(resume_path, lo, hi, names, sizes)
        saved_algorithm = (str(np.asarray(ckpt["algorithm"]).item())
                           if "algorithm" in ckpt.files else "classic")
        if saved_algorithm != algorithm:
            raise ValueError(
                f"Checkpoint algorithm is {saved_algorithm!r}, requested "
                f"{algorithm!r}.")
        key = jnp.asarray(ckpt["key"])
        gen_start = int(ckpt["generation_counter"])
        gens_without_improvement = int(ckpt["gens_without_improvement"])
        best_logp_so_far = float(ckpt["best_logp_so_far"])
        population = jnp.asarray(ckpt["population"])
        fitness = jnp.asarray(ckpt["fitness"])
        best_solution = jnp.asarray(ckpt["best_solution"])
        best_fitness = jnp.asarray(ckpt["best_fitness"])
        initial_pop_size = int(
            ckpt["initial_pop_size"] if "initial_pop_size" in ckpt.files
            else len(population))
        fprint(f"Resumed from {resume_path} at generation {gen_start}")
    else:
        population, fitness = _make_de_initial_population(
            exact_eval, lo, hi, pop_size, seed, N_sobol,
            min_dist_frac, seed_points=seed_points)
        key = jax.random.PRNGKey(seed)
        initial_pop_size = pop_size
        best_idx = int(np.argmin(np.asarray(fitness)))
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        gen_start = 0
        best_logp_so_far = -float(best_fitness)
        gens_without_improvement = 0

    rng = np.random.default_rng(seed)
    mutation_archive = np.empty((0, D))
    m_f = np.full(6, 0.5)
    m_cr = np.full(6, 0.5)
    memory_index = 0
    if adaptive and resume_path is not None:
        required = {"rng_state", "mutation_archive", "m_f", "m_cr",
                    "memory_index"}
        missing = required.difference(ckpt.files)
        if missing:
            raise ValueError(
                "Adaptive checkpoint is missing: " + ", ".join(missing))
        rng.bit_generator.state = json.loads(str(
            np.asarray(ckpt["rng_state"]).item()))
        mutation_archive = np.asarray(ckpt["mutation_archive"])
        m_f = np.asarray(ckpt["m_f"])
        m_cr = np.asarray(ckpt["m_cr"])
        memory_index = int(ckpt["memory_index"])

    value_grad = None
    group_sizes = None
    if use_adam:
        value_grad = _make_minibatch_value_grad(
            target, names, fixed, lo, scale)
        group_sizes = [n for n in (
            target.model._n_sys, target.model._n_red,
            target.model._n_blue) if n]

    def checkpoint_extra():
        out = {
            "algorithm": np.asarray(algorithm),
            "initial_pop_size": np.asarray(initial_pop_size),
        }
        if adaptive:
            out.update(
                rng_state=np.asarray(json.dumps(rng.bit_generator.state)),
                mutation_archive=np.asarray(mutation_archive),
                m_f=np.asarray(m_f), m_cr=np.asarray(m_cr),
                memory_index=np.asarray(memory_index))
        return out

    fsection(f"DE ({algorithm}, pop={len(population)}, "
             f"max_gen={max_generations}, "
             f"start={gen_start})")
    last_ckpt = time.time()
    final_gen = gen_start
    de_progress = trange(max_generations - gen_start, desc="DE")
    for step in de_progress:
        gen = gen_start + step
        final_gen = gen + 1
        if use_adam and gen % adam_interval == 0:
            population, fitness, replaced = _polish_elites(
                population, fitness, batch_eval, exact_archive,
                value_grad, group_sizes, rng, adam_elites, adam_steps,
                adam_learning_rate, adam_batch_fraction, min_dist_frac)
            mutation_archive = _append_mutation_archive(
                mutation_archive, replaced, len(population), rng)

        current_size = len(population)
        if adaptive:
            trials, trial_f, trial_cr = _lshade_trials(
                population, fitness, mutation_archive, m_f, m_cr, rng)
            trials = jnp.asarray(trials)
        else:
            key, k1, k2, k3, k_cross, k_force = jax.random.split(key, 6)
            idx1 = jax.random.permutation(k1, current_size)
            idx2 = jax.random.permutation(k2, current_size)
            idx3 = jax.random.permutation(k3, current_size)
            mutant = population[idx1] + mutation * (
                population[idx2] - population[idx3])
            mutant = _reflect_bounds(mutant)
            cross = (jax.random.uniform(
                k_cross, (current_size, D)) < crossover)
            forced = jax.random.randint(k_force, (current_size,), 0, D)
            cross = cross.at[jnp.arange(current_size), forced].set(True)
            trials = jnp.where(cross, mutant, population)

        trial_fitness = exact_eval(trials)
        jax.block_until_ready(trial_fitness)
        improved = np.asarray(trial_fitness) < np.asarray(fitness)
        if adaptive and np.any(improved):
            old_fitness = np.asarray(fitness)
            mutation_archive = _append_mutation_archive(
                mutation_archive, np.asarray(population)[improved],
                current_size, rng)
            memory_index = _update_lshade_memory(
                m_f, m_cr, memory_index, trial_f[improved],
                trial_cr[improved],
                old_fitness[improved] - np.asarray(trial_fitness)[improved])
        population = jnp.where(improved[:, None], trials, population)
        fitness = jnp.where(improved, trial_fitness, fitness)

        if adaptive:
            target_size = _linear_population_size(
                initial_pop_size, min_pop_size, gen + 1, max_generations)
            if target_size < len(population):
                keep = np.argsort(np.asarray(fitness))[:target_size]
                population = population[keep]
                fitness = fitness[keep]
                if len(mutation_archive) > target_size:
                    mutation_archive = mutation_archive[rng.choice(
                        len(mutation_archive), target_size, replace=False)]

        gen_best_idx = int(np.argmin(np.asarray(fitness)))
        gen_best_fitness = fitness[gen_best_idx]
        if float(gen_best_fitness) < float(best_fitness):
            best_fitness = gen_best_fitness
            best_solution = population[gen_best_idx]

        current_best = -float(best_fitness)
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
                f"stale={gens_without_improvement}/{patience}")

        if (checkpoint_path is not None
                and time.time() - last_ckpt >= checkpoint_interval):
            _save_de_checkpoint(
                checkpoint_path, population, fitness, best_solution,
                best_fitness, gen + 1, key, gens_without_improvement,
                best_logp_so_far, lo, hi, names, sizes,
                extra=checkpoint_extra())
            last_ckpt = time.time()
            fprint(f"  checkpoint: gen {gen + 1}")

        if gens_without_improvement >= patience:
            fprint(f"  converged at gen {final_gen}")
            break

    if checkpoint_path is not None:
        _save_de_checkpoint(
            checkpoint_path, population, fitness, best_solution,
            best_fitness, final_gen, key, gens_without_improvement,
            best_logp_so_far, lo, hi, names, sizes,
            extra=checkpoint_extra())
    fprint(f"Exact archive: {exact_archive.count()} unique evaluations, "
           f"{exact_archive.hits} cache hits")
    exact_archive.close()

    x_best = np.asarray(lo + best_solution * scale)
    params_best = _flat_to_theta(jnp.asarray(x_best), names)
    params_best.update(fixed)
    theta = target.complete_params(params_best)
    phys_args, phys_kw = target.model.phys_from_params_jax(theta, target.h)
    r_ang = target.model.conditional_r_ang_map(phys_args, phys_kw)
    best_logp = -float(best_fitness)
    return _theta_to_output(
        {k: np.asarray(jax.device_get(v)) for k, v in theta.items()},
        np.asarray(jax.device_get(r_ang))), best_logp, final_gen


def _run_fixed_globals(target, init_params, data_only=False):
    theta = target.complete_params({
        name: jnp.asarray(init_params[name]) for name in target.names})
    lp, ll, phys_args, phys_kw = _logp_2d_terms(target, theta)
    logp = ll if data_only else lp + ll
    r_ang = target.model.conditional_r_ang_map(phys_args, phys_kw)
    logp = jax.block_until_ready(logp)
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
                        choices=("median", "config", "reid"), default=None,
                        help="Fixed-global source for --fix-globals, and "
                             "target initial check otherwise. Default: "
                             "config inference/init_strategy. 'reid' uses "
                             "reported Pesce/Reid globals; for NGC4258 it "
                             "reads reid_ngc4258_best.toml.")
    parser.add_argument("--spot-batch", type=int, default=None)
    parser.add_argument("--gpu-mem", type=float, default=None,
                        help="GPU VRAM hint in GB for the auto batch planner. "
                             "Only disambiguates the V100 16/32GB variant "
                             "(32 selects the 32GB card); other GPUs are "
                             "auto-detected by name.")
    parser.add_argument("--log2-N", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--de-algorithm",
                        choices=("classic", "lshade", "hybrid"),
                        default=None)
    parser.add_argument("--min-pop-size", type=int, default=None)
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval-chunk", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--adam-interval", type=int, default=None)
    parser.add_argument("--adam-elites", type=int, default=None)
    parser.add_argument("--adam-steps", type=int, default=None)
    parser.add_argument("--adam-learning-rate", type=float, default=None)
    parser.add_argument("--adam-batch-fraction", type=float, default=None)
    parser.add_argument("--n-devices", type=int, default=None,
                        help="GPUs to shard the DE population across (pmap). "
                             "Default: all visible local GPUs; 1 forces the "
                             "single-device path. ARC: request N with "
                             "submit.sh --gpu-count N (-> --gres=gpu:N).")
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
        fprint(f"DE population sharded over {n_dev} GPU(s) via pmap.")

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
            ("max_generations", "max_generations"),
            ("patience", "patience"),
            ("eval_chunk", "eval_chunk"),
            ("log_every", "log_every"),
            ("adam_interval", "adam_interval"),
            ("adam_elites", "adam_elites"),
            ("adam_steps", "adam_steps")):
        value = getattr(args, arg_name)
        if value is not None:
            opt_cfg[cfg_name] = int(value)
    if args.de_algorithm is not None:
        opt_cfg["algorithm"] = args.de_algorithm
    for arg_name in ("adam_learning_rate", "adam_batch_fraction"):
        value = getattr(args, arg_name)
        if value is not None:
            opt_cfg[arg_name] = float(value)
    config = {
        "inference": master_cfg["inference"],
        "model": dict(master_cfg["model"]),
        "io": master_cfg["io"],
        "optimise": opt_cfg,
    }
    config["model"]["galaxies"] = {
        g: dict(blk) for g, blk in master_cfg["model"]["galaxies"].items()}
    gal_blk = config["model"]["galaxies"][args.galaxy]

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
    init_source = "config" if init_strategy == "reid" else init_strategy
    init_params = _make_init(
        model, _init_block(config["model"]["galaxies"][args.galaxy], model),
        init_source,
        int(_required_inference(inf_cfg, "init_num_samples")),
        jax.random.PRNGKey(seed))
    h = float(get_nested(model.config, "model/H0_ref", 73.0)) / 100.0
    target_spot_batch = (args.spot_batch if args.spot_batch is not None
                         else cfg_spot_batch)
    target = MaserBlackJaxTarget(
        model, h, init_params, spot_batch=target_spot_batch)
    # Size the memory plan up front and apply it to the target BEFORE any
    # evaluation. The Pesce/Reid baseline and _pesce_init below score the full
    # (spots x n_r x n_phi) grid, which OOMs large galaxies (e.g. NGC4258: 187
    # systemic spots x n_phi=60001 x f64 = 32 GiB in one buffer) unless spots
    # are batched. Applying here -- not only in the DE section -- keeps every
    # path (baseline, fixed-globals, DE) within the VRAM budget; the banner is
    # printed later from these same values.
    plan_sb, plan_ec, plan_info = _plan_de_batch(
        model, int(opt_cfg.get("pop_size", 1000)), gpu_mem_gb=args.gpu_mem)
    if plan_ec is not None:                      # GPU with a known VRAM budget
        if args.spot_batch is None and cfg_spot_batch is None:
            target.spot_batch = plan_sb
        if args.eval_chunk is None:
            opt_cfg["eval_chunk"] = plan_ec
    if init_strategy == "reid":
        try:
            init_params, reid_status = _pesce_init(target, args.galaxy,
                                                   master_cfg)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
        if reid_status:
            fprint("Reid/Pesce init defaulted " + ", ".join(reid_status))
    pesce_logp = None
    pesce_params = None
    fixed_floors = None
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
    else:
        try:
            pesce_params, pesce_status = _pesce_init(
                target, args.galaxy, master_cfg)
            _, pesce_logp, _ = _run_fixed_globals(target, pesce_params)
            fsection("Pesce/Reid baseline")
            if pesce_status:
                fprint("defaulted " + ", ".join(pesce_status))
            fprint(f"Pesce/Reid fixed globals + conditional r_ang MAP: "
                   f"logP = {pesce_logp:.2f}")
            if args.fix_floors_pesce:
                fixed_floors = {n: pesce_params[n] for n in _PESCE_FLOOR_NAMES
                                if n in target.names}
        except KeyError as exc:
            fprint(f"Pesce/Reid baseline unavailable: {exc}")
            if args.fix_floors_pesce:
                raise SystemExit(
                    f"--fix-floors-pesce needs Pesce floors: {exc}") from exc
    fprint("inner solve: joint 2D (r_ang, phi) marginal per spot, "
           f"n_r_global={model._n_r_global}, "
           f"n_refine_steps={model._n_refine_steps}")

    fixed_globals = args.fix_globals or args.fix_globals_pesce
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
        algorithm = str(opt_cfg.get("algorithm", "classic")).lower()
        ckpt_dir = results_path(
            master_cfg["io"].get("root_output", "results/Megamaser"),
            "de_checkpoints", args.galaxy)
        os.makedirs(ckpt_dir, exist_ok=True)
        floor_suffix = "_pescefloors" if args.fix_floors_pesce else ""
        algorithm_suffix = ("" if algorithm == "classic"
                            else f"_{algorithm}")
        ckpt_path = os.path.join(
            ckpt_dir,
            f"de_ckpt_rmap{_variant_suffix(model)}{floor_suffix}"
            f"{algorithm_suffix}.npz")
        resume_path = (
            ckpt_path if args.resume and os.path.isfile(ckpt_path)
            else None)
        if args.resume and resume_path is None:
            fprint(
                f"--resume: no checkpoint found at {ckpt_path}, "
                "starting fresh")
        # Classic DE retains its independent Pesce baseline. Adaptive modes
        # deliberately include Pesce in the data-ridge + Sobol seed mixture.
        pop_size = int(opt_cfg.get("pop_size", 1000))
        data_seeds, seed_info = _data_driven_seed(
            model, target, init_params, _h_ref(model) * 100.0,
            max(1, pop_size // 2), seed + 1,
            sobol_n_sigma=opt_cfg.get("sobol_n_sigma", 5))
        if data_seeds is None:
            fprint(f"data seed unavailable ({seed_info}); Sobol only")
            seed_points = None
        else:
            fprint(f"data seed: {seed_info}")
            seed_points = data_seeds
        if algorithm != "classic" and pesce_params is not None:
            reference_seed = _theta_to_flat(pesce_params, target.names)[None]
            seed_points = (reference_seed if seed_points is None else
                           np.vstack([reference_seed, seed_points]))
            fprint("adaptive seed: included exact Pesce/Reid point")
        # Memory plan was computed and applied to the target up front (before
        # the Pesce baseline); echo it here. On CPU plan_ec is None -- nothing
        # to report and the target keeps its configured defaults.
        if plan_ec is not None:
            fprint("DE memory plan: " + plan_info)
            if "UNRECOGNISED" in plan_info:
                fprint(
                    "  WARNING: GPU not recognised; using a conservative "
                    f"{_DEFAULT_VRAM_GB:g}GB budget. Add it to _GPU_VRAM_GB "
                    "in run_de_map.py, or pass --gpu-mem GB, for the size.")
            sb_flag = " [--spot-batch]" if args.spot_batch is not None else ""
            ec_flag = " [--eval-chunk]" if args.eval_chunk is not None else ""
            fprint(f"DE batching: eval_chunk={opt_cfg.get('eval_chunk')}"
                   f"{ec_flag} (DE candidates scored per GPU pass), "
                   f"spot_batch={target.spot_batch}{sb_flag} "
                   f"(spots per pass; None = all at once)")
            n_r = model._n_r_local + model._n_r_global
            n_phi = max(int(pc["sin_phi"].shape[0])
                        for pc in model._phi_concat.values())
            max_group = max(model._n_sys, model._n_red, model._n_blue)
            dtype_bytes = 8 if jax.config.jax_enable_x64 else 4
            cell = 8 * n_r * n_phi * dtype_bytes
            ec = int(opt_cfg.get("eval_chunk"))
            sb = max_group if target.spot_batch is None else target.spot_batch
            pred_gb = ec * sb * cell / 1e9
            fprint(f"DE memory estimate: ~{pred_gb:.1f} GB/GPU peak "
                   f"= eval_chunk {ec} x {sb} spots x "
                   f"{cell / 1e9:.2f} GB/spot "
                   f"({'f64' if dtype_bytes == 8 else 'f32'} grid "
                   f"n_r={n_r} x n_phi={n_phi}); "
                   f"{n_dev} GPUs share the DE population -> ~{n_dev}x "
                   f"throughput.")
        init_params, best_logp, run_info = _run_de(
            target, opt_cfg, seed, n_dev=n_dev, devices=gpu_devices,
            checkpoint_path=ckpt_path, resume_path=resume_path,
            checkpoint_interval=args.checkpoint_interval_minutes * 60.0,
            seed_points=seed_points, fixed_params=fixed_floors)
        run_summary = f"generations = {run_info}"
    dt = time.time() - t0

    fsection(f"MAP results ({args.galaxy}, {dt:.0f}s)")
    label = "logL" if args.fix_globals_pesce else "logP"
    fprint(f"best {label} = {best_logp:.2f}; {run_summary}")
    if pesce_logp is not None and not fixed_globals:
        delta = best_logp - pesce_logp
        fprint(f"DE - Pesce/Reid baseline = {delta:.2f}")
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
