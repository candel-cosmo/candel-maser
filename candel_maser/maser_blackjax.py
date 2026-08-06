# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""BlackJAX explicit-latent (r, phi) sampler for megamasers.

The sampler targets the megamaser disk likelihood:

* global disk parameters are sampled by BlackJAX NUTS in unconstrained
  Euclidean coordinates;
* the production distance/mass block samples direct ``D_A`` and either
  ``log_MBH`` or ``eta = log_MBH - log10(D_A)``;
* ``r_ang`` is represented by non-centred log-radius residuals,
  ``z_r = log(r_ang / r_hat(theta))``, and updated jointly with ``phi`` by a
  vectorised per-spot adaptive random-walk Metropolis (correlated ``(z_r,
  phi)`` block, plus a reflection move that hops the two ``phi`` modes of
  high-velocity spots);
* an optional unit-Jacobian translation carries systemic ``phi`` with the
  astrometric centre during the global update, without changing the physical
  coordinates used by the latent sweep or saved samples;
* warmup adapts the global NUTS step size and inverse mass matrix using the
  BlackJAX window-adaptation primitives, and adapts the latent block by an
  adaptive-Metropolis scheme (per-spot empirical covariance for the proposal
  shape, per-spot Robbins-Monro scalar for its scale);
* warmup and sampling run under ``jax.lax.scan`` in chunks.

The module does not import BlackJAX at import time.  This keeps the rest of the
package usable while BlackJAX is an optional dependency.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Delta, Uniform
from numpyro.distributions.transforms import biject_to

from ..util import get_nested
from . import maser_physics

# phys_args positional indices.
(_I_X0, _I_Y0, _I_DA, _I_MBH, _I_VSYS, _I_RREF_I, _I_RREF_OMEGA,
 _I_I0, _I_DI, _I_OMEGA0, _I_DOMEGA, _I_SXF2, _I_SYF2, _I_VARVHV,
 _I_SAF2) = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16)


def _require_blackjax():
    """Import BlackJAX and its window-adaptation module with a clear error."""
    try:
        blackjax = import_module("blackjax")
        window_adaptation = import_module(
            "blackjax.adaptation.window_adaptation")
    except ModuleNotFoundError as exc:
        if exc.name == "blackjax":
            raise RuntimeError(
                "BlackJAX is required for the megamaser sampler. "
                "Install it in the active environment before running "
                "scripts/megamaser/run_maser.py."
            ) from exc
        raise
    return blackjax, window_adaptation


def _step_bar(total, enabled, desc, position=None):
    """tqdm bar counting MCMC steps, or None when disabled/unavailable.

    Steps run in chunks (see ``_chunk_bounds``), but the bar advances by each
    chunk's step count so it reads in steps (``0 .. total``), not chunks.
    """
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, position=position, dynamic_ncols=True)


def _update_mcmc_progress_postfix(progress, info, theta_scale):
    """Show the global step scale without overflowing fixed progress rows."""
    if not hasattr(progress, "set_postfix"):
        return
    theta_scale = float(np.mean(np.asarray(theta_scale)))
    progress.set_postfix({"theta": f"{theta_scale:.2e}"}, refresh=False)


def _scan_fn(body, jit_steps):
    """Wrap a scan body as a (carry, xs) -> (carry, ys) callable."""
    def fn(carry, xs):
        return jax.lax.scan(body, carry, xs)
    return jax.jit(fn) if jit_steps else fn


def _chunk_bounds(n, progress, max_chunks=50):
    """(start, stop) ranges that scan ``n`` steps in chunks for the bar.

    Without a progress bar the whole range is one scan (fastest).  With one,
    ~``max_chunks`` chunks give live feedback; the final short chunk costs one
    extra trace under jit.
    """
    if n <= 0:
        return []
    if not progress:
        return [(0, n)]
    size = max(1, (n + max_chunks - 1) // max_chunks)
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def _concat_dicts(rows):
    """Concatenate a list of identically keyed dicts along axis 0."""
    if not rows:
        return {}
    return {key: np.concatenate([np.asarray(row[key]) for row in rows], axis=0)
            for key in rows[0]}


def _update_chunk_postfix(progress, info_chunk, theta_scale):
    """Show the last step of a scanned chunk in the progress bar."""
    if not hasattr(progress, "set_postfix"):
        return
    last = {key: np.asarray(value)[-1] for key, value in info_chunk.items()}
    _update_mcmc_progress_postfix(progress, last, theta_scale)


def _seeds(model, phys_args):
    """Float32-stable (r_hat, r_min, r_max) from a phys_args tuple.

    ``model.radius_seeds`` is mathematically fine, but in float32 its masked
    closed-form branches can generate enormous inactive derivatives.  In JAX
    reverse mode, ``where`` can still expose those as ``0 * NaN``.  Here the
    inactive branches get finite dummy denominators before masking.
    """
    D_A = phys_args[_I_DA]
    M_BH = phys_args[_I_MBH]
    v_sys = phys_args[_I_VSYS]
    i0 = phys_args[_I_I0]
    r_min, r_max = model.r_ang_range(D_A)

    dtype = jnp.result_type(D_A, M_BH, v_sys, i0)
    sin_i = jnp.maximum(jnp.abs(jnp.sin(i0)),
                        jnp.asarray(1e-6, dtype=dtype))
    is_hv = model.is_highvel
    has_accel = getattr(model, "_all_has_accel",
                        jnp.ones(model.n_spots, dtype=bool))

    dv = model._all_v - v_sys
    dv2_safe = jnp.where(
        is_hv, dv * dv + jnp.asarray(1e-6, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype))
    r_vel = M_BH * (maser_physics.C_v * sin_i) ** 2 / (D_A * dv2_safe)

    use_accel = (~is_hv) & has_accel
    a_safe = jnp.where(
        use_accel, jnp.abs(model._all_a) + jnp.asarray(1e-6, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype))
    r_acc_arg = (
        maser_physics.C_a * M_BH * sin_i / (D_A ** 2 * a_safe))
    r_acc = jnp.sqrt(jnp.maximum(r_acc_arg, jnp.asarray(1e-30, dtype=dtype)))

    r_hat = jnp.where(is_hv, r_vel, r_acc)
    if model._n_sys_uncons > 0:
        r_geo = jnp.sqrt(r_min * r_max)
        r_hat = r_hat.at[model._idx_sys_uncons].set(r_geo)
    r_hat = jnp.clip(r_hat, r_min * 1.01, r_max * 0.99)
    return r_hat, r_min, r_max


def _ll_per_spot(model, r_spots, phys_args, phys_kw, spot_batch=None):
    """Per-spot phi-marginal log-likelihood, shape (n_spots,)."""
    groups = model._spot_groups_from_r(r_spots)
    return model._eval_phi_marginal(
        groups, phys_args, phys_kw, spot_batch=spot_batch)


def _finite_uniform_bounds(prior, name):
    try:
        return float(np.asarray(prior.low)), float(np.asarray(prior.high))
    except AttributeError as exc:
        raise ValueError(
            f"eta mass parameterization requires finite {name} prior "
            "bounds.") from exc


def _eta_support_prior(model, h):
    """Broad sampler support for eta that preserves the log_MBH prior."""
    D_lo, D_hi = _finite_uniform_bounds(model.priors["D"], "D")
    mbh_lo, mbh_hi = _finite_uniform_bounds(
        model.priors["log_MBH"], "log_MBH")
    if model._D_A_uniform:
        D_A_grid = jnp.asarray([D_lo, D_hi])
    else:
        D_grid = jnp.asarray([D_lo, D_hi])
        z_grid = model.distance2redshift(D_grid, h=h)
        D_A_grid = D_grid / (1.0 + z_grid)
    log_D_A_min = float(jnp.min(jnp.log10(D_A_grid)))
    log_D_A_max = float(jnp.max(jnp.log10(D_A_grid)))
    return Uniform(mbh_lo - log_D_A_max, mbh_hi - log_D_A_min)


def _theta_site_prior_pairs(model, h=None):
    """Sampled scalar sites in the single-galaxy megamaser global block."""
    mass_param = getattr(model, "mass_parameterization", "eta")
    if mass_param == "eta":
        if h is None:
            h = float(get_nested(model.config, "model/H0_ref", 73.0)) / 100.0
        mass_pair = ("eta", "eta", _eta_support_prior(model, h))
    else:
        mass_pair = ("log_MBH", "log_MBH")

    distance_pair = ("D_A", "D") if model._D_A_uniform else ("D_c", "D")

    pairs = [
        distance_pair,
        mass_pair,
        ("x0", "x0"),
        ("y0", "y0"),
        ("i0", "i0"),
        ("Omega0", "Omega0"),
        ("dOmega_dr", "dOmega_dr"),
        ("di_dr", "di_dr"),
        ("sigma_x_floor", "sigma_x_floor"),
        ("sigma_y_floor", "sigma_y_floor"),
        ("sigma_v_sys", "sigma_v_sys"),
        ("sigma_v_hv", "sigma_v_hv"),
        ("sigma_a_floor", "sigma_a_floor"),
        ("dv_sys", "dv_sys"),
    ]

    if model.use_ecc:
        if not model.ecc_cartesian:
            raise NotImplementedError(
                "The BlackJAX megamaser sampler currently supports "
                "eccentricity only in ecc_cartesian=True form.")
        pairs.extend([
            ("e_x", "e_x"),
            ("e_y", "e_y"),
            ("dperiapsis_dr", "dperiapsis_dr"),
        ])

    if model.use_quadratic_warp:
        pairs.extend([
            ("d2i_dr2", "d2i_dr2"),
            ("d2Omega_dr2", "d2Omega_dr2"),
        ])

    out = []
    for pair in pairs:
        if len(pair) == 3:
            site, prior_key, prior = pair
        else:
            site, prior_key = pair
            prior = model.priors[prior_key]
        if isinstance(prior, Delta):
            continue
        out.append((site, prior_key, prior))
    return tuple(out)


def _finite_scalar(value):
    value = np.asarray(value)
    if value.shape != ():
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _support_interval(support):
    lower = _finite_scalar(getattr(support, "lower_bound", np.nan))
    upper = _finite_scalar(getattr(support, "upper_bound", np.nan))
    return lower, upper


def nudge_initial_params_inside_support(model, h, init_params, *,
                                        sites=None,
                                        rng_key=None,
                                        margin_fraction=1e-3):
    """Move scalar initial values away from finite prior boundaries.

    Bounded-support transforms map exact boundaries to infinities in the
    unconstrained NUTS coordinate. MAP initialisers often sit exactly on, or
    numerically very close to, floor lower bounds, so keep the physical start
    close while making the unconstrained point finite.
    """
    if sites is None:
        sites = _theta_site_prior_pairs(model, h=h)
    out = dict(init_params)
    adjusted = []
    keys = (
        list(jax.random.split(rng_key, len(sites)))
        if rng_key is not None else [None] * len(sites))

    def offset(margin, key):
        if key is None:
            return margin
        draw = jax.random.uniform(key, (), minval=1.1, maxval=2.0)
        return margin * float(jax.device_get(draw))

    for key, (site, _, prior) in zip(keys, sites):
        if site not in out:
            continue
        value = _finite_scalar(out[site])
        if value is None:
            continue
        lower, upper = _support_interval(prior.support)
        if lower is None and upper is None:
            continue

        scale = max(abs(value), 1.0)
        if lower is not None and upper is not None:
            if not upper > lower:
                continue
            width = upper - lower
            margin = min(
                max(width * margin_fraction, 1e-7 * scale),
                0.25 * width)
            if value <= lower + margin:
                new_value = lower + offset(margin, key)
            elif value >= upper - margin:
                new_value = upper - offset(margin, key)
            else:
                new_value = value
        elif lower is not None:
            margin = max(abs(lower) * margin_fraction, 1e-7 * scale)
            new_value = (
                lower + offset(margin, key)
                if value <= lower + margin else value)
        else:
            margin = max(abs(upper) * margin_fraction, 1e-7 * scale)
            new_value = (
                upper - offset(margin, key)
                if value >= upper - margin else value)

        if new_value != value:
            out[site] = jnp.asarray(
                new_value, dtype=jnp.asarray(out[site]).dtype)
            adjusted.append((site, value, new_value, lower, upper))
    return out, tuple(adjusted)


def init_from_prior_median(model, rng_key, num_samples=100, h=None):
    """Estimate scalar global initial values from prior-sample medians."""
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError("num_samples must be at least one.")

    sites = _theta_site_prior_pairs(model, h=h)
    keys = jax.random.split(rng_key, len(sites))
    init_params = {}
    for key, (site, _, prior) in zip(keys, sites):
        draws = prior.sample(key, sample_shape=(num_samples,))
        init_params[site] = jnp.median(jnp.asarray(draws), axis=0)

    return init_params


def _stack_dicts(rows):
    """Stack a list of identically keyed sample/info dictionaries."""
    if not rows:
        return {}
    return {
        key: np.stack([np.asarray(row[key]) for row in rows], axis=0)
        for key in rows[0]
    }


def _stack_chain_results(results, runtime_seconds, chain_workers):
    """Stack single-chain results into one chain-first result."""
    if not results:
        raise ValueError("at least one chain result is required")

    theta_sites = results[0].theta_sites
    for result in results[1:]:
        if result.theta_sites != theta_sites:
            raise ValueError("all chain results must have the same sites")

    return MaserBlackJaxResult(
        samples=_stack_dicts([r.samples for r in results]),
        log_density=np.stack(
            [np.asarray(r.log_density) for r in results], axis=0),
        info=_stack_dicts([r.info for r in results]),
        warmup_info=_stack_dicts([r.warmup_info for r in results]),
        parameters=_stack_dicts([r.parameters for r in results]),
        theta_sites=theta_sites,
        runtime_seconds=float(runtime_seconds),
        chain_method="parallel" if chain_workers > 1 else "sequential",
        chain_workers=int(chain_workers))


def _chain_worker_count(num_chains, chain_workers=8):
    """Number of chains that fit under the requested and CPU limits."""
    num_chains = int(num_chains)
    chain_workers = int(chain_workers)
    if num_chains < 1 or chain_workers < 1:
        raise ValueError("num_chains and chain_workers must be >= 1.")
    limits = [num_chains, chain_workers]
    for key in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        try:
            value = int(os.environ.get(key, ""))
        except ValueError:
            continue
        if value > 0:
            limits.append(value)
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            if affinity > 0:
                limits.append(affinity)
        except OSError:
            pass
    limits.append(os.cpu_count() or 1)
    return min(limits)


def estimate_radial_eps(model, theta, h, delta=0.02,
                        eps_min=1e-3, eps_max=0.5):
    """Per-spot proposal scales for non-centred log-radius residuals."""
    phys_args, phys_kw = model.phys_from_params_jax(theta, h)
    r_hat, r_min, r_max = _seeds(model, phys_args)
    log_r_hat = jnp.log(r_hat)
    lo = jnp.log(r_min) - log_r_hat
    hi = jnp.log(r_max) - log_r_hat
    step = jnp.minimum(delta, 0.45 * jnp.minimum(-lo, hi))
    step = jnp.maximum(step, jnp.asarray(1e-5, dtype=r_hat.dtype))

    def ll(z):
        return _ll_per_spot(model, r_hat * jnp.exp(z), phys_args, phys_kw)

    z0 = jnp.zeros_like(r_hat)
    l0 = ll(z0)
    lp = ll(z0 + step)
    lm = ll(z0 - step)
    curv = -(lp - 2.0 * l0 + lm) / step**2
    sigma = 1.0 / jnp.sqrt(jnp.clip(curv, 1.0, None))
    return jnp.clip(2.4 * sigma, eps_min, eps_max)


def _r_ang_from_z(model, theta, h, z_r):
    """Map non-centred log-radius residuals to angular radii."""
    phys_args, phys_kw = model.phys_from_params_jax(theta, h)
    r_hat, r_min, r_max = _seeds(model, phys_args)
    return r_hat * jnp.exp(z_r), r_hat, r_min, r_max, phys_args, phys_kw


def _systemic_phi_xy_coefficients(model, r_ang, phys_args, phys_kw):
    """Fixed linear response of systemic phi to the astrometric centre."""
    _, Omega_r = maser_physics.warp_geometry(
        r_ang, phys_args[_I_RREF_I], phys_args[_I_RREF_OMEGA],
        phys_args[_I_I0], phys_args[_I_DI],
        phys_args[_I_OMEGA0], phys_args[_I_DOMEGA],
        phys_kw.get("d2i_dr2", 0.0), phys_kw.get("d2Omega_dr2", 0.0))
    sin_O, cos_O = jnp.sin(Omega_r), jnp.cos(Omega_r)
    radius_uas = 1e3 * r_ang
    dx_dphi = radius_uas * sin_O
    dy_dphi = radius_uas * cos_O
    var_x = model._all_sigma_x2 + phys_args[_I_SXF2]
    var_y = model._all_sigma_y2 + phys_args[_I_SYF2]
    denom = dx_dphi**2 / var_x + dy_dphi**2 / var_y
    denom = jnp.maximum(denom, jnp.finfo(r_ang.dtype).tiny)
    active = ~model.is_highvel
    return (jnp.where(active, dx_dphi / var_x / denom, 0.0),
            jnp.where(active, dy_dphi / var_y / denom, 0.0))


def _ll_fixed_phi_per_spot(model, r_spots, phi, phys_args, phys_kw):
    """Per-spot fixed-phi log-likelihood, shape (n_spots,)."""
    groups = model._spot_groups_from_r(r_spots)
    return model._eval_phi_fixed(groups, phi, phys_args, phys_kw)


def _wrap_interval(x, lo, hi):
    width = hi - lo
    return lo + jnp.mod(x - lo, width)


def _phi_support_arrays(model, dtype):
    lo = jnp.zeros((model.n_spots,), dtype=dtype)
    hi = jnp.zeros((model.n_spots,), dtype=dtype)
    centre = jnp.zeros((model.n_spots,), dtype=dtype)
    for type_key in ("sys", "red", "blue"):
        idx = getattr(model, f"_idx_{type_key}")
        if int(idx.shape[0]) == 0:
            continue
        ranges = model._phi_subranges[type_key]
        for left, right in zip(ranges[:-1], ranges[1:]):
            if not np.isclose(left[1], right[0], rtol=0.0, atol=1e-12):
                raise ValueError(
                    "explicit-phi MCMC requires contiguous phi support; "
                    f"{type_key} has a gap between {left} and {right}.")
        a = jnp.asarray(ranges[0][0], dtype=dtype)
        b = jnp.asarray(ranges[-1][1], dtype=dtype)
        lo = lo.at[idx].set(a)
        hi = hi.at[idx].set(b)
        centre = centre.at[idx].set(0.5 * (a + b))
    return lo, hi, centre


def _initial_phi(model, dtype):
    lo, hi, centre = _phi_support_arrays(model, dtype)
    return _wrap_interval(centre, lo, hi)


# Adaptive-Metropolis latent proposal (Andrieu & Thoms 2008, Alg. 4): the
# per-spot proposal covariance is ``exp(log_s) * Cov`` where ``Cov`` is a
# windowed per-spot empirical 2x2 covariance (scale *and* shape) and ``log_s``
# is a per-spot scalar nudged toward the target acceptance by Robbins-Monro.
_AM_COV_REG = 1e-9          # ridge so the 2x2 Cholesky never fails
_LOG_S_MIN = -9.21          # log(1e-4)
_LOG_S_MAX = 4.61           # log(1e2)
_COV_BURN_FRAC = 0.5        # warmup fraction excluded as transient before the
#                             per-side latent covariance starts accumulating
_COV_BLEND_K = 20.0         # samples for the empirical per-side covariance to
#                             reach half weight vs cov_init (smooth, no hard
#                             switch); a freshly-entered hv mode stays cov_init
# Half of the hv teleports use a per-mode covariance map (matches the target
# mode's shape); the other half stay the plain mirror so a never-visited mode
# can still be discovered.
_MODE_MATCH_MAP_PROB = 0.5


def _latent_init_cov(eps_z, eps_phi):
    """Diagonal per-spot covariance from initial (z, phi) proposal scales."""
    z2 = eps_z ** 2
    p2 = eps_phi ** 2
    zeros = jnp.zeros_like(z2)
    return jnp.stack((
        jnp.stack((z2, zeros), axis=-1),
        jnp.stack((zeros, p2), axis=-1),
    ), axis=-2)


def _latent_cholesky(cov, log_s):
    """Per-spot proposal Cholesky from a covariance shape and log-scale."""
    eye = jnp.eye(2, dtype=cov.dtype)
    L = jnp.linalg.cholesky(cov + _AM_COV_REG * eye)
    return jnp.sqrt(jnp.exp(log_s))[:, None, None] * L


def _welford_init(n_spots, dtype):
    """Per-spot online (count, mean, M2) accumulator for a 2-vector."""
    return (jnp.zeros((n_spots,), dtype=dtype),
            jnp.zeros((n_spots, 2), dtype=dtype),
            jnp.zeros((n_spots, 2, 2), dtype=dtype))


def _welford_update(state, x):
    """Online per-spot mean/M2 update; ``x`` is ``(n_spots, 2)``."""
    count, mean, M2 = state
    count = count + 1.0
    delta = x - mean
    mean = mean + delta / count[:, None]
    delta2 = x - mean
    M2 = M2 + delta[:, :, None] * delta2[:, None, :]
    return count, mean, M2


def _welford_update_masked(state, x, mask):
    """Per-spot Welford update applied only where ``mask`` is True."""
    updated = _welford_update(state, x)

    def pick(new, old):
        m = mask.reshape(mask.shape + (1,) * (new.ndim - 1))
        return jnp.where(m, new, old)

    return jax.tree_util.tree_map(pick, updated, state)


def _welford_cov(state):
    """Per-spot 2x2 covariance, shrunk toward its diagonal for small n.

    Shrinking toward the diagonal (not the identity) keeps the marginal scales
    while damping a noisy, under-sampled off-diagonal correlation.
    """
    count, _, M2 = state
    n = jnp.maximum(count, 1.0)[:, None, None]
    cov = M2 / n
    w = (count / (count + 5.0))[:, None, None]
    eye = jnp.eye(2, dtype=cov.dtype)
    return w * cov + (1.0 - w) * (cov * eye)


def latent_z_phi_rw_sweep(model, theta, z_r, phi, latent_scale, rng_key, h,
                          n_inner=20, reflect_prob=0.25, *,
                          latent_mean, latent_cov):
    """Vectorised explicit ``(z_r, phi)`` random-walk sweep over spots.

    ``latent_scale`` is the stacked per-side proposal Cholesky
    ``(2, n_spots, 2, 2)`` with index ``0`` for the minus mode
    (``phi < phi_c``) and ``1`` for the plus mode (``phi >= phi_c``), each the
    Cholesky of the *signed* covariance ``(z, phi - phi_c)`` on that side.  A
    high-velocity spot has a bimodal phi whose two modes may differ, so:

    * the **local** move uses the active side's Cholesky (its off-diagonal
      already carries the within-mode z-phi correlation *and sign*) and is
      confined to that side (axis-crossing proposals are rejected -- a
      reflecting wall at ``phi_c``);
    * side switches go through the **teleport** move.  Half the teleports use
      the per-mode covariance map ``x' = mu_t + L_t L_s^{-1} (x - mu_s)``
      (``mu``/``L`` the per-side mean / Cholesky of ``(z, phi - phi_c)``), a
      deterministic involution between the two modes accepted with the Jacobian
      ``det L_t / det L_s``; it lands the jump on the target mode's peak *with
      that mode's width*.  The other half is the symmetric mirror
      ``phi -> 2 phi_c - phi``, which always crosses so an unvisited mode is
      still discoverable.  (Mirror is symmetric -> no correction; the map
      carries its Jacobian.)
    """
    _, r_hat, _, _, phys_args, phys_kw = _r_ang_from_z(
        model, theta, h, z_r)
    phi_lo, phi_hi, phi_centre = _phi_support_arrays(model, z_r.dtype)
    is_hv = model.is_highvel

    L_minus = latent_scale[0]                   # phi < phi_c
    L_plus = latent_scale[1]                    # phi >= phi_c
    # Side-symmetric teleport scales so the cross-mode jump is direction
    # independent (no Hastings correction) even when the sides differ.
    z_sd_tele = 0.5 * (L_plus[:, 0, 0] + L_minus[:, 0, 0])
    phi_sd_p = jnp.sqrt(L_plus[:, 1, 0] ** 2 + L_plus[:, 1, 1] ** 2)
    phi_sd_m = jnp.sqrt(L_minus[:, 1, 0] ** 2 + L_minus[:, 1, 1] ** 2)
    phi_sd_tele = 0.5 * (phi_sd_p + phi_sd_m)

    # Per-mode Cholesky/mean of the RAW (unscaled) covariance, used by the
    # mode-matching teleport map.  latent_cov is (2, n, 2, 2), index
    # 0=minus / 1=plus; latent_mean (2, n, 2) is the per-side mean of
    # (z, phi-phi_c).
    eye2 = jnp.eye(2, dtype=z_r.dtype)
    Lraw = jnp.linalg.cholesky(latent_cov + _AM_COV_REG * eye2)  # (2,n,2,2)
    Lraw_m, Lraw_p = Lraw[0], Lraw[1]
    ldet_m = jnp.log(Lraw_m[:, 0, 0]) + jnp.log(Lraw_m[:, 1, 1])
    ldet_p = jnp.log(Lraw_p[:, 0, 0]) + jnp.log(Lraw_p[:, 1, 1])
    mu_m_lat, mu_p_lat = latent_mean[0], latent_mean[1]          # (n,2)

    def side(phi_val):
        return jnp.where(phi_val >= phi_centre, 1.0, -1.0)

    def logp(z_val, phi_val):
        r_val = r_hat * jnp.exp(z_val)
        inside_phi = (phi_val >= phi_lo) & (phi_val <= phi_hi)
        lp = _ll_fixed_phi_per_spot(
            model, r_val, phi_val, phys_args, phys_kw)
        # Flat-in-r_ang Jacobian |dr_ang/dz_r| = r_ang; r_ang > 0 is automatic
        # so there is no [r_min, r_max] wall.  phi keeps its hard box.
        lp = lp + jnp.log(r_val)
        return jnp.where(inside_phi, lp, -jnp.inf)

    lp0 = logp(z_r, phi)
    zeros = jnp.zeros_like(z_r, dtype=jnp.int32)

    def body(_, state):
        z_cur, phi_cur, lp_cur, acc, refl_try, refl_acc, key = state
        key, k_prop, k_mix, k_acc, k_b2 = jax.random.split(key, 5)
        noise = jax.random.normal(k_prop, (z_cur.shape[0], 2),
                                  dtype=z_cur.dtype)
        n0, n1 = noise[:, 0], noise[:, 1]
        s_cur = side(phi_cur)
        is_plus = (s_cur > 0)
        # active per-side Cholesky (sign of the correlation carried by L10).
        L = jnp.where(is_plus[:, None, None], L_plus, L_minus)
        L00, L10, L11 = L[:, 0, 0], L[:, 1, 0], L[:, 1, 1]
        use_reflect = (
            is_hv
            & (jax.random.uniform(k_mix, phi_cur.shape) < reflect_prob))
        # local: active-side correlated step.
        z_local = z_cur + L00 * n0
        phi_local = _wrap_interval(
            phi_cur + (L10 * n0 + L11 * n1), phi_lo, phi_hi)
        # mirror teleport (symmetric; always crosses phi_c).
        z_mir = z_cur + z_sd_tele * n0
        phi_mir = _wrap_interval(
            2.0 * phi_centre - phi_cur + phi_sd_tele * n1, phi_lo, phi_hi)
        # per-mode covariance map x' = mu_t + L_t L_s^{-1} (x - mu_s).
        sel3 = is_plus[:, None, None]
        sel2 = is_plus[:, None]
        L_cur = jnp.where(sel3, Lraw_p, Lraw_m)
        L_tar = jnp.where(sel3, Lraw_m, Lraw_p)
        mu_cur = jnp.where(sel2, mu_p_lat, mu_m_lat)
        mu_tar = jnp.where(sel2, mu_m_lat, mu_p_lat)
        ldet_cur = jnp.where(is_plus, ldet_p, ldet_m)
        ldet_tar = jnp.where(is_plus, ldet_m, ldet_p)
        u0 = (z_cur - mu_cur[:, 0]) / L_cur[:, 0, 0]
        u1 = ((phi_cur - phi_centre - mu_cur[:, 1])
              - L_cur[:, 1, 0] * u0) / L_cur[:, 1, 1]
        z_map = mu_tar[:, 0] + L_tar[:, 0, 0] * u0
        phi_map = phi_centre + (
            mu_tar[:, 1] + L_tar[:, 1, 0] * u0 + L_tar[:, 1, 1] * u1)
        log_jac = ldet_tar - ldet_cur
        # By default half the teleports use the map, half the mirror.
        do_map = (
            jax.random.uniform(k_b2, phi_cur.shape)
            < _MODE_MATCH_MAP_PROB)
        z_tele = jnp.where(do_map, z_map, z_mir)
        phi_tele = jnp.where(do_map, phi_map, phi_mir)
        log_h_tele = jnp.where(do_map, log_jac, 0.0)
        # the map must actually cross to phi_c's other side (else its
        # reverse would not return here); mirror always crosses.
        map_nocross = do_map & (side(phi_map) == s_cur)
        z_prop = jnp.where(use_reflect, z_tele, z_local)
        phi_prop = jnp.where(use_reflect, phi_tele, phi_local)
        lp_prop = logp(z_prop, phi_prop)
        # confine local moves to the current side (reflecting wall at phi_c);
        # reject a mode-matching teleport that failed to cross sides.
        crossed = is_hv & (~use_reflect) & (side(phi_local) != s_cur)
        lp_prop = jnp.where(crossed | (use_reflect & map_nocross),
                            -jnp.inf, lp_prop)
        log_h = jnp.where(use_reflect, log_h_tele, 0.0)
        accept = (
            jnp.log(jax.random.uniform(k_acc, z_cur.shape))
            < (lp_prop - lp_cur + log_h))
        return (
            jnp.where(accept, z_prop, z_cur),
            jnp.where(accept, phi_prop, phi_cur),
            jnp.where(accept, lp_prop, lp_cur),
            acc + accept.astype(jnp.int32),
            refl_try + use_reflect.astype(jnp.int32),
            refl_acc + (accept & use_reflect).astype(jnp.int32),
            key,
        )

    z_new, phi_new, _, acc, refl_try, refl_acc, _ = jax.lax.fori_loop(
        0, int(n_inner), body, (z_r, phi, lp0, zeros, zeros, zeros, rng_key))
    return z_new, phi_new, acc, refl_try, refl_acc


class MaserBlackJaxTarget:
    """Unconstrained global-parameter target for BlackJAX."""

    def __init__(self, model, h, init_params, spot_batch=None,
                 transport_systemic_phi=False):
        if model.use_selection:
            raise RuntimeError(
                "The BlackJAX megamaser sampler currently supports only "
                "single-galaxy disk likelihoods without selection.")
        self.model = model
        self.h = h
        self.spot_batch = spot_batch
        self.mass_parameterization = getattr(
            model, "mass_parameterization", "eta")
        self.sites = _theta_site_prior_pairs(model, h=h)
        self.transforms = tuple(
            biject_to(prior.support) for _, _, prior in self.sites)
        self.names = tuple(site for site, _, _ in self.sites)
        self._check_init(init_params)
        self._phi_transport_xy = None
        if transport_systemic_phi:
            u_theta, z_r = self.initial_state(init_params)
            theta, _ = self.constrain(u_theta)
            r_ang, _, _, _, phys_args, phys_kw = _r_ang_from_z(
                self.model, theta, self.h, z_r)
            self._phi_transport_xy = _systemic_phi_xy_coefficients(
                self.model, r_ang, phys_args, phys_kw)

    def _check_init(self, init_params):
        missing = []
        for site in self.names:
            if site in init_params:
                continue
            missing.append(site)
        if missing:
            raise KeyError(
                "Missing initial value(s) for BlackJAX global block: "
                + ", ".join(missing))
        for site in self.names:
            value = jnp.asarray(init_params[site])
            if value.ndim != 0:
                raise ValueError(
                    "The BlackJAX sampler expects scalar global sites; "
                    f"{site!r} has shape {value.shape}.")

    def unconstrain(self, params):
        vals = []
        for transform, site in zip(self.transforms, self.names):
            vals.append(transform.inv(jnp.asarray(params[site])))
        return jnp.stack(vals)

    def constrain(self, u_theta):
        params = {}
        log_det = jnp.asarray(0.0, dtype=u_theta.dtype)
        for i, (site, transform) in enumerate(zip(self.names,
                                                  self.transforms)):
            x = u_theta[i]
            y = transform(x)
            params[site] = y
            log_det = log_det + transform.log_abs_det_jacobian(x, y)
        params = self.complete_params(params)
        return params, log_det

    def complete_params(self, params):
        """Add deterministic mass parameters to a constrained theta dict."""
        if self.mass_parameterization != "eta" or "eta" not in params:
            return params
        if "D_A" in params:
            D_A = params["D_A"]
        else:
            D_c = params["D_c"]
            z_cosmo = self.model.distance2redshift(
                jnp.atleast_1d(D_c), h=self.h).squeeze()
            D_A = D_c / (1.0 + z_cosmo)
        out = dict(params)
        out["log_MBH"] = params["eta"] + jnp.log10(D_A)
        return out

    def initial_state(self, init_params):
        init_params, _ = nudge_initial_params_inside_support(
            self.model, self.h, init_params, sites=self.sites)
        u_theta = self.unconstrain(init_params)
        theta, _ = self.constrain(u_theta)
        phys_args, _ = self.model.phys_from_params_jax(theta, self.h)
        r_hat, _, _ = _seeds(self.model, phys_args)
        if "z_r" in init_params:
            z_r = jnp.asarray(init_params["z_r"], dtype=u_theta.dtype)
        elif "r_ang" in init_params:
            r_ang = jnp.asarray(init_params["r_ang"], dtype=u_theta.dtype)
            if r_ang.ndim == 1 and int(r_ang.shape[0]) == int(
                    self.model.n_spots):
                z_r = jnp.log(r_ang / r_hat)
            else:
                z_r = jnp.zeros_like(r_hat, dtype=u_theta.dtype)
        else:
            z_r = jnp.zeros_like(r_hat, dtype=u_theta.dtype)
        return u_theta, z_r

    def logdensity_explicit(self, u_theta, z_r, phi):
        theta, log_det = self.constrain(u_theta)
        lp = self.constrained_logdensity_z_phi(theta, z_r, phi)
        return lp + jnp.where(jnp.isfinite(lp), log_det, 0.0)

    def phi_transport_residual(self, theta, phi):
        """Systemic phi residual transported with the astrometric centre."""
        phys_args, _ = self.model.phys_from_params_jax(theta, self.h)
        kx, ky = self._phi_transport_xy
        centre = -(kx * phys_args[_I_X0] + ky * phys_args[_I_Y0])
        lo, hi, _ = _phi_support_arrays(self.model, phi.dtype)
        shifted = _wrap_interval(phi - centre, lo, hi)
        return jnp.where(self.model.is_highvel, phi, shifted)

    def phi_from_transport_residual(self, theta, phi_residual):
        """Physical phi corresponding to an astrometric-centre residual."""
        phys_args, _ = self.model.phys_from_params_jax(theta, self.h)
        kx, ky = self._phi_transport_xy
        centre = -(kx * phys_args[_I_X0] + ky * phys_args[_I_Y0])
        lo, hi, _ = _phi_support_arrays(self.model, phi_residual.dtype)
        shifted = _wrap_interval(phi_residual + centre, lo, hi)
        return jnp.where(self.model.is_highvel, phi_residual, shifted)

    def logdensity_explicit_transported(self, u_theta, z_r, phi_residual):
        """Explicit target with systemic phi held in transported coordinates."""
        theta, log_det_theta = self.constrain(u_theta)
        r_ang, _, r_min, r_max, phys_args, phys_kw = _r_ang_from_z(
            self.model, theta, self.h, z_r)
        kx, ky = self._phi_transport_xy
        centre = -(kx * phys_args[_I_X0] + ky * phys_args[_I_Y0])
        lo, hi, _ = _phi_support_arrays(self.model, z_r.dtype)
        shifted = _wrap_interval(phi_residual + centre, lo, hi)
        phi = jnp.where(self.model.is_highvel, phi_residual, shifted)
        lp = self.constrained_logdensity_r_ang_phi(
            theta, r_ang, phi, r_min=r_min, r_max=r_max,
            phys_args=phys_args, phys_kw=phys_kw)
        lp = lp + jnp.where(jnp.isfinite(lp), jnp.sum(jnp.log(r_ang)), 0.0)
        return lp + jnp.where(jnp.isfinite(lp), log_det_theta, 0.0)

    def constrained_logdensity_z_phi(self, theta, z_r, phi):
        """Log posterior in constrained globals, z radii, and fixed phi."""
        r_ang, _, r_min, r_max, phys_args, phys_kw = _r_ang_from_z(
            self.model, theta, self.h, z_r)
        lp = self.constrained_logdensity_r_ang_phi(
            theta, r_ang, phi, r_min=r_min, r_max=r_max,
            phys_args=phys_args, phys_kw=phys_kw)
        logdet_r = jnp.sum(jnp.log(r_ang))
        return lp + jnp.where(jnp.isfinite(lp), logdet_r, 0.0)

    def constrained_logdensity_r_ang_phi(self, theta, r_ang, phi, *,
                                         r_min=None, r_max=None,
                                         phys_args=None, phys_kw=None):
        """Log posterior in constrained globals and explicit latents."""
        theta = self.complete_params(theta)
        if phys_args is None or phys_kw is None:
            phys_args, phys_kw = self.model.phys_from_params_jax(
                theta, self.h)
        phi_lo, phi_hi, _ = _phi_support_arrays(
            self.model, jnp.asarray(r_ang).dtype)

        lp = jnp.asarray(0.0, dtype=jnp.asarray(r_ang).dtype)
        for site, _, prior in self.sites:
            if site == "eta":
                continue
            lp = lp + prior.log_prob(theta[site])
        if self.mass_parameterization == "eta":
            lp = lp + self.model.priors["log_MBH"].log_prob(
                theta["log_MBH"])

        # r_ang has an improper flat prior (r_ang > 0 only, guaranteed by
        # r_ang = r_hat * exp(z_r)); no [r_min, r_max] bound, so the
        # theta-NUTS step has no r_ang support wall.  phi keeps its hard box.
        inside_phi = jnp.all((phi >= phi_lo) & (phi <= phi_hi))
        groups = self.model._spot_groups_from_r(r_ang)
        ll = self.model._sum_phi_fixed(
            groups, phi, phys_args, phys_kw)
        return jnp.where(inside_phi, lp + ll, -jnp.inf)

    def sample_dict(self, u_theta, z_r):
        theta, _ = self.constrain(u_theta)
        out = dict(theta)
        r_ang, _, _, _, _, _ = _r_ang_from_z(
            self.model, theta, self.h, z_r)
        out["r_ang"] = r_ang
        return out

    def initial_state_explicit(self, init_params):
        u_theta, z_r = self.initial_state(init_params)
        if "phi" in init_params:
            phi = jnp.asarray(init_params["phi"], dtype=u_theta.dtype)
        else:
            phi = _initial_phi(self.model, u_theta.dtype)
        phi_lo, phi_hi, _ = _phi_support_arrays(self.model, u_theta.dtype)
        return u_theta, z_r, _wrap_interval(phi, phi_lo, phi_hi)

    def sample_dict_explicit(self, u_theta, z_r, phi):
        out = self.sample_dict(u_theta, z_r)
        out["phi"] = phi
        return out


class MaserBlackJaxMCMCState(NamedTuple):
    theta: Any
    z_r: Any
    phi: Any
    log_s: Any           # per-spot log proposal scale (Robbins-Monro)
    cov: Any             # per-side (2, n_spots, 2, 2) covariance shapes
    latent_scale: Any    # per-side (2, n_spots, 2, 2) proposal Choleskys
    theta_scale: Any     # global NUTS step size (for the progress bar)
    # per-side (2, n_spots, 2) mean of (z, phi-phi_c) for the mode-matching
    # teleport.
    latent_mean: Any = None


@dataclass
class MaserBlackJaxResult:
    """Return container for one or more BlackJAX megamaser chains."""

    samples: Dict[str, np.ndarray]
    log_density: np.ndarray
    info: Dict[str, np.ndarray]
    warmup_info: Dict[str, np.ndarray]
    parameters: Dict[str, np.ndarray]
    theta_sites: Tuple[str, ...]
    runtime_seconds: float
    chain_method: str = "single"
    chain_workers: int = 1


def _mcmc_info(theta_info, latent_accept, reflect_try, reflect_accept,
               n_inner, dtype):
    latent_rate = latent_accept / jnp.asarray(n_inner, dtype=dtype)
    reflect_rate = reflect_accept / jnp.maximum(reflect_try, 1)
    if hasattr(theta_info, "is_accepted"):
        theta_is_accepted = theta_info.is_accepted
    else:
        theta_is_accepted = ~theta_info.is_divergent
    return {
        "theta_acceptance_rate": theta_info.acceptance_rate,
        "theta_is_accepted": theta_is_accepted,
        "latent_accept_mean": jnp.mean(latent_rate),
        "latent_accept_min": jnp.min(latent_rate),
        "latent_accept_max": jnp.max(latent_rate),
        "reflect_accept_mean": jnp.mean(jnp.where(
            reflect_try > 0, reflect_rate, 0.0)),
        "reflect_try_mean": jnp.mean(reflect_try.astype(dtype)),
    }


def _adapt_latent(state, z_r, phi, l_accept, r_try, r_accept, cov_state,
                  cov_init, is_hv, phi_centre, t, *, n_inner,
                  target_accept_latent, adapt_rate, cov_start):
    """One latent-proposal adaptation step (log_s + per-side Welford cov).

    Shared by the warmup and the latent burn-in; returns the updated
    proposal pieces and the advanced Welford state. Theta is untouched.
    """
    cov_state_m, cov_state_p = cov_state
    dtype = state.log_s.dtype
    # log_s tunes the *local* correlated proposal, so drive Robbins-Monro
    # off the local-move acceptance only -- the teleport is a separate
    # fixed move with its own (lower) acceptance.
    # local = total - teleport.
    local_try = jnp.maximum(
        jnp.asarray(n_inner, dtype=dtype) - r_try.astype(dtype), 1.0)
    local_rate = (l_accept - r_accept).astype(dtype) / local_try
    gamma = adapt_rate / jnp.sqrt(jnp.asarray(t + 1, dtype=dtype))
    log_s = jnp.clip(
        state.log_s + gamma * (local_rate - target_accept_latent),
        _LOG_S_MIN, _LOG_S_MAX)

    # Per-side signed (z, phi - phi_c) Welford so each hv mode learns its
    # own shape/scale; only the second-half window contributes (the early
    # transient is excluded, #3).  Non-hv spots are unimodal, so both sides
    # are fed every sample (the two covariances coincide).
    d = phi - phi_centre
    obs = jnp.stack((z_r, d), axis=-1)
    in_window = t >= cov_start
    on_plus = (~is_hv) | (d >= 0.0)
    on_minus = (~is_hv) | (d < 0.0)
    cov_state_p = _welford_update_masked(
        cov_state_p, obs, in_window & on_plus)
    cov_state_m = _welford_update_masked(
        cov_state_m, obs, in_window & on_minus)
    # Smoothly blend the empirical per-side covariance into cov_init as
    # each side accrues samples (a = n / (n + K)).  With no hard switch a
    # noisy low-n estimate barely moves the proposal and there is no
    # discontinuity for Robbins-Monro to chase; an unvisited hv mode (n=0)
    # stays at cov_init.
    a_p = (cov_state_p[0]
           / (cov_state_p[0] + _COV_BLEND_K))[:, None, None]
    a_m = (cov_state_m[0]
           / (cov_state_m[0] + _COV_BLEND_K))[:, None, None]
    cov_p = (1.0 - a_p) * cov_init + a_p * _welford_cov(cov_state_p)
    cov_m = (1.0 - a_m) * cov_init + a_m * _welford_cov(cov_state_m)
    # Decouple shape from scale: renormalise the pair to the cov_init mean
    # trace so the empirical block carries only shape (and the per-side
    # ratio), leaving log_s as the single, drift-free scale knob.  The
    # common factor cancels in the mode-matching teleport map (L_t L_s^-1),
    # so this does not disturb the cross-mode jump.
    tr_ref = cov_init[:, 0, 0] + cov_init[:, 1, 1]
    s_pair = 0.5 * (cov_p[:, 0, 0] + cov_p[:, 1, 1]
                    + cov_m[:, 0, 0] + cov_m[:, 1, 1])
    norm = (tr_ref / jnp.maximum(s_pair, 1e-12))[:, None, None]
    cov_p = cov_p * norm
    cov_m = cov_m * norm
    cov_shape = jnp.stack((cov_m, cov_p), axis=0)
    latent_scale = jnp.stack(
        (_latent_cholesky(cov_m, log_s), _latent_cholesky(cov_p, log_s)),
        axis=0)
    # per-side mean of (z, phi-phi_c) for the mode-matching teleport.
    latent_mean = jnp.stack((cov_state_m[1], cov_state_p[1]), axis=0)
    return (log_s, cov_shape, latent_scale, latent_mean,
            (cov_state_m, cov_state_p), local_rate)


def _make_mcmc_latent_burnin_step(target, cov_init, is_hv, phi_centre, *,
                                  n_inner, target_accept_latent, adapt_rate,
                                  reflect_prob, cov_start):
    """Scan body for latent-only Gibbs sweeps at FIXED theta.

    Migrates z/phi to their conditional typical set and adapts the
    adaptive-Metropolis proposal, WITHOUT the theta NUTS step or its
    step-size dual-averaging -- so theta adaptation never sees the initial
    divergent transient that otherwise crashes the step size to ~0.

    carry = (state, welford_state, key); xs = step_index.
    """
    def body(carry, t):
        state, cov_state, key = carry
        key, k_latent = jax.random.split(key)
        theta_params, _ = target.constrain(state.theta.position)
        z_r, phi, l_accept, r_try, r_accept = latent_z_phi_rw_sweep(
            target.model, theta_params, state.z_r, state.phi,
            state.latent_scale, k_latent, target.h,
            n_inner=n_inner, reflect_prob=reflect_prob,
            latent_mean=state.latent_mean, latent_cov=state.cov)
        (log_s, cov_shape, latent_scale, latent_mean,
         cov_state_new, _) = _adapt_latent(
            state, z_r, phi, l_accept, r_try, r_accept, cov_state, cov_init,
            is_hv, phi_centre, t, n_inner=n_inner,
            target_accept_latent=target_accept_latent, adapt_rate=adapt_rate,
            cov_start=cov_start)
        new_state = state._replace(
            z_r=z_r, phi=phi, log_s=log_s, cov=cov_shape,
            latent_scale=latent_scale, latent_mean=latent_mean)
        dtype = state.log_s.dtype
        latent_rate = l_accept.astype(dtype) / jnp.asarray(
            n_inner, dtype=dtype)
        # Teleport/reflect acceptance among spots that attempted one -- the
        # mode-jump landing rate, which is what the burn-in buys the bimodal
        # high-velocity spots. Same convention as _mcmc_info.
        reflect_rate = r_accept.astype(dtype) / jnp.maximum(
            r_try.astype(dtype), 1.0)
        info = {"latent_accept_mean": jnp.mean(latent_rate),
                "reflect_accept_mean": jnp.mean(
                    jnp.where(r_try > 0, reflect_rate, 0.0)),
                "log_s_mean": jnp.mean(log_s)}
        return (new_state, cov_state_new, key), info

    return body


def _make_mcmc_warmup_step(blackjax, target, adapt_step, cov_init, is_hv,
                           phi_centre, *, n_inner, target_accept_latent,
                           adapt_rate, reflect_prob, max_num_doublings,
                           cov_start, transport_systemic_phi):
    """Scan body ``(carry, xs) -> (carry, info)`` for one warmup step.

    carry = (state, theta_adaptation_state, welford_state, key);
    xs = (window_schedule_row, step_index).
    """
    nuts_step = blackjax.nuts.build_kernel()

    def body(carry, xs):
        state, adaptation_state, cov_state, key = carry
        sched_row, t = xs
        key, k_latent, k_theta = jax.random.split(key, 3)

        theta_params, _ = target.constrain(state.theta.position)
        z_r, phi, l_accept, r_try, r_accept = latent_z_phi_rw_sweep(
            target.model, theta_params, state.z_r, state.phi,
            state.latent_scale, k_latent, target.h,
            n_inner=n_inner, reflect_prob=reflect_prob,
            latent_mean=state.latent_mean, latent_cov=state.cov)

        if transport_systemic_phi:
            phi_nuts = target.phi_transport_residual(
                theta_params, phi)

            def logdensity_theta(u_theta):
                return target.logdensity_explicit_transported(
                    u_theta, z_r, phi_nuts)
        else:
            def logdensity_theta(u_theta):
                return target.logdensity_explicit(u_theta, z_r, phi)

        theta_state = blackjax.nuts.init(
            state.theta.position, logdensity_theta)
        theta_state, theta_info = nuts_step(
            k_theta, theta_state, logdensity_theta,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            max_num_doublings=max_num_doublings)
        if transport_systemic_phi:
            theta_params, _ = target.constrain(theta_state.position)
            phi = target.phi_from_transport_residual(
                theta_params, phi_nuts)
        adaptation_state = adapt_step(
            adaptation_state, sched_row, theta_state.position,
            theta_info.acceptance_rate)

        dtype = state.log_s.dtype
        (log_s, cov_shape, latent_scale, latent_mean,
         cov_state_new, local_rate) = _adapt_latent(
            state, z_r, phi, l_accept, r_try, r_accept, cov_state, cov_init,
            is_hv, phi_centre, t, n_inner=n_inner,
            target_accept_latent=target_accept_latent, adapt_rate=adapt_rate,
            cov_start=cov_start)

        new_state = MaserBlackJaxMCMCState(
            theta_state, z_r, phi, log_s, cov_shape, latent_scale,
            adaptation_state.step_size, latent_mean)
        info = _mcmc_info(theta_info, l_accept, r_try, r_accept,
                          n_inner, dtype)
        info.update({
            "log_s_mean": jnp.mean(log_s),
            "local_accept_mean": jnp.mean(local_rate),
            "theta_step_size": adaptation_state.step_size,
            "theta_is_divergent": theta_info.is_divergent,
            "theta_num_integration_steps": theta_info.num_integration_steps,
        })
        return (new_state, adaptation_state, cov_state_new, key), info

    return body


def _make_mcmc_sample_step(blackjax, target, sample_parameters, *,
                           n_inner, reflect_prob, transport_systemic_phi):
    """Scan body ``(state, key) -> (state, (sample, log_density, info))``."""
    nuts_step = blackjax.nuts.build_kernel()
    step_size = sample_parameters["step_size"]
    inverse_mass_matrix = sample_parameters["inverse_mass_matrix"]
    max_num_doublings = sample_parameters["max_num_doublings"]

    def body(state, key):
        key_latent, key_theta = jax.random.split(key)
        theta_params, _ = target.constrain(state.theta.position)
        z_r, phi, l_accept, r_try, r_accept = latent_z_phi_rw_sweep(
            target.model, theta_params, state.z_r, state.phi,
            state.latent_scale, key_latent, target.h,
            n_inner=n_inner, reflect_prob=reflect_prob,
            latent_mean=state.latent_mean, latent_cov=state.cov)

        if transport_systemic_phi:
            phi_nuts = target.phi_transport_residual(
                theta_params, phi)

            def logdensity_theta(u_theta):
                return target.logdensity_explicit_transported(
                    u_theta, z_r, phi_nuts)
        else:
            def logdensity_theta(u_theta):
                return target.logdensity_explicit(u_theta, z_r, phi)

        theta_state = blackjax.nuts.init(
            state.theta.position, logdensity_theta)
        theta_state, theta_info = nuts_step(
            key_theta, theta_state, logdensity_theta,
            step_size, inverse_mass_matrix,
            max_num_doublings=max_num_doublings)
        if transport_systemic_phi:
            theta_params, _ = target.constrain(theta_state.position)
            phi = target.phi_from_transport_residual(
                theta_params, phi_nuts)
        sample = target.sample_dict_explicit(theta_state.position, z_r, phi)
        _, log_det_theta = target.constrain(theta_state.position)
        log_density = (
            theta_state.logdensity - log_det_theta
            - jnp.sum(jnp.log(sample["r_ang"]))
        )
        new_state = state._replace(theta=theta_state, z_r=z_r, phi=phi)
        info = _mcmc_info(theta_info, l_accept, r_try, r_accept,
                          n_inner, state.log_s.dtype)
        info.update({
            "theta_is_divergent": theta_info.is_divergent,
            "theta_num_integration_steps": theta_info.num_integration_steps,
        })
        return new_state, (sample, log_density, info)

    return body


def _run_blackjax_mcmc_one(model, init_params, rng_key, *,
                           num_warmup=1000, num_samples=1000, h=None,
                           n_inner=20, target_accept_theta=0.9,
                           target_accept_latent=0.35,
                           theta_step_init=0.01, phi_step_init=0.05,
                           eps_init=None, eps_z_min=1e-3, eps_z_max=0.5,
                           eps_phi_min=1e-4, eps_phi_max=1.0,
                           adapt_rate=0.10, reflect_prob=0.25,
                           max_num_doublings=10, num_latent_burnin=0,
                           sample_n_inner=None,
                           transport_systemic_phi=False,
                           progress_bar=True, jit_steps=True,
                           progress_label="1/1", progress_positions=None):
    """Run one explicit-phi NUTS-within-adaptive-Metropolis chain."""
    if num_warmup < 0 or num_samples < 1:
        raise ValueError(
            "num_warmup must be >= 0 and num_samples must be > 0.")

    blackjax, window_adaptation = _require_blackjax()
    if h is None:
        h = float(get_nested(model.config, "model/H0_ref", 73.0)) / 100.0

    target = MaserBlackJaxTarget(
        model, h, init_params,
        transport_systemic_phi=transport_systemic_phi)
    if transport_systemic_phi:
        sys_ranges = model._phi_subranges["sys"]
        width = float(sys_ranges[-1][1] - sys_ranges[0][0])
        if not np.isclose(width, 2.0 * np.pi, rtol=0.0, atol=1e-12):
            raise ValueError(
                "systemic phi transport requires full 2pi support.")
    u_theta, z_r, phi = target.initial_state_explicit(init_params)
    theta0, _ = target.constrain(u_theta)
    dtype = u_theta.dtype
    if eps_init is None:
        eps_z = estimate_radial_eps(
            model, theta0, h, eps_min=eps_z_min, eps_max=eps_z_max)
    else:
        eps_z = jnp.asarray(eps_init, dtype=dtype)
    eps_z = jnp.broadcast_to(eps_z, (model.n_spots,))
    eps_phi = jnp.clip(
        jnp.full((model.n_spots,), phi_step_init, dtype=dtype),
        eps_phi_min, eps_phi_max)
    is_hv = model.is_highvel
    _, _, phi_centre = _phi_support_arrays(model, dtype)
    cov_init = _latent_init_cov(eps_z, eps_phi)
    log_s = jnp.zeros((model.n_spots,), dtype=dtype)
    L0 = _latent_cholesky(cov_init, log_s)
    latent_scale = jnp.stack((L0, L0), axis=0)
    cov_stacked = jnp.stack((cov_init, cov_init), axis=0)

    def logdensity0(u):
        return target.logdensity_explicit(u, z_r, phi)

    theta_state = blackjax.nuts.init(u_theta, logdensity0)
    state = MaserBlackJaxMCMCState(
        theta_state, z_r, phi, log_s, cov_stacked, latent_scale,
        jnp.asarray(theta_step_init, dtype=dtype),
        jnp.zeros((2, model.n_spots, 2), dtype=dtype))

    adapt_init, adapt_step, adapt_final = window_adaptation.base(
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=target_accept_theta)
    adaptation_state = adapt_init(u_theta, theta_step_init)
    schedule = window_adaptation.build_schedule(int(num_warmup))

    cov_state = (_welford_init(model.n_spots, dtype),
                 _welford_init(model.n_spots, dtype))
    cov_start = int(_COV_BURN_FRAC * int(num_warmup))
    warmup_body = _make_mcmc_warmup_step(
        blackjax, target, adapt_step, cov_init, is_hv, phi_centre,
        n_inner=n_inner, target_accept_latent=target_accept_latent,
        adapt_rate=adapt_rate, reflect_prob=reflect_prob,
        max_num_doublings=max_num_doublings,
        cov_start=cov_start,
        transport_systemic_phi=transport_systemic_phi)
    warmup_scan = _scan_fn(warmup_body, jit_steps)

    t0 = time.time()
    if progress_positions is None:
        progress_positions = (None, None, None)
    # Optional latent-only burn-in at FIXED theta: migrate z/phi to their
    # conditional typical set and seed the per-side covariance before theta
    # NUTS / step-size adaptation begins. The migrated state and accumulated
    # Welford cov_state both flow into warmup, which keeps adapting.
    if int(num_latent_burnin) > 0:
        burnin_body = _make_mcmc_latent_burnin_step(
            target, cov_init, is_hv, phi_centre, n_inner=n_inner,
            target_accept_latent=target_accept_latent, adapt_rate=adapt_rate,
            reflect_prob=reflect_prob,
            cov_start=int(_COV_BURN_FRAC * int(num_latent_burnin)))
        burnin_scan = _scan_fn(burnin_body, jit_steps)
        burnin_idx = jnp.arange(int(num_latent_burnin))
        burnin_bar = _step_bar(
            int(num_latent_burnin), progress_bar,
            f"Compiling latent burn-in {progress_label}",
            position=progress_positions[0])
        bcarry = (state, cov_state, rng_key)
        for c0, c1 in _chunk_bounds(int(num_latent_burnin), progress_bar):
            bcarry, binfo = burnin_scan(bcarry, burnin_idx[c0:c1])
            binfo = jax.device_get(binfo)
            if burnin_bar is not None:
                lat = float(np.asarray(binfo["latent_accept_mean"])[-1])
                refl = float(np.asarray(binfo["reflect_accept_mean"])[-1])
                burnin_bar.set_description(
                    f"Latent burn-in {progress_label}", refresh=False)
                burnin_bar.set_postfix(
                    {"latent_acc": f"{lat:.3f}",
                     "refl_acc": f"{refl:.3f}"},
                    refresh=False)
                burnin_bar.update(c1 - c0)
        if burnin_bar is not None:
            burnin_bar.close()
        state, cov_state, rng_key = bcarry

    carry = (state, adaptation_state, cov_state, rng_key)
    step_idx = jnp.arange(int(num_warmup))
    warmup_rows = []
    warmup_bar = (
        _step_bar(
            int(num_warmup), progress_bar,
            f"Compiling MCMC warmup {progress_label}",
            position=progress_positions[1])
        if int(num_warmup) > 0 else None)
    for c0, c1 in _chunk_bounds(int(num_warmup), progress_bar):
        carry, info_chunk = warmup_scan(
            carry, (schedule[c0:c1], step_idx[c0:c1]))
        info_chunk = jax.device_get(info_chunk)
        warmup_rows.append(info_chunk)
        if warmup_bar is not None:
            warmup_bar.set_description(
                f"MCMC warmup {progress_label}", refresh=False)
        _update_chunk_postfix(
            warmup_bar, info_chunk, jax.device_get(carry[0].theta_scale))
        if warmup_bar is not None:
            warmup_bar.update(c1 - c0)
    if warmup_bar is not None:
        warmup_bar.close()
    state, adaptation_state, cov_state, key = carry

    step_size, inverse_mass_matrix = adapt_final(adaptation_state)
    state = state._replace(theta_scale=step_size)

    sample_parameters = {
        "step_size": step_size,
        "inverse_mass_matrix": inverse_mass_matrix,
        "max_num_doublings": max_num_doublings,
    }
    if sample_n_inner is None:
        sample_n_inner = n_inner
    sample_body = _make_mcmc_sample_step(
        blackjax, target, sample_parameters,
        n_inner=sample_n_inner, reflect_prob=reflect_prob,
        transport_systemic_phi=transport_systemic_phi)
    sample_scan = _scan_fn(sample_body, jit_steps)

    key, sample_master = jax.random.split(key)
    sample_keys = jax.random.split(sample_master, int(num_samples))
    sample_rows = []
    log_density_rows = []
    info_rows = []
    sample_bar = _step_bar(
        int(num_samples), progress_bar,
        f"Compiling sampling {progress_label}",
        position=progress_positions[2])
    for c0, c1 in _chunk_bounds(int(num_samples), progress_bar):
        state, (sample, log_density, info) = sample_scan(
            state, sample_keys[c0:c1])
        sample_rows.append(jax.device_get(sample))
        log_density_rows.append(jax.device_get(log_density))
        info_host = jax.device_get(info)
        info_rows.append(info_host)
        if sample_bar is not None:
            sample_bar.set_description(
                f"Sampling {progress_label}", refresh=False)
        _update_chunk_postfix(
            sample_bar, info_host, jax.device_get(state.theta_scale))
        if sample_bar is not None:
            sample_bar.update(c1 - c0)
    if sample_bar is not None:
        sample_bar.close()
    runtime = time.time() - t0

    warmup_info = _concat_dicts(warmup_rows)
    parameters = {
        "step_size": np.asarray(jax.device_get(step_size)),
        "inverse_mass_matrix": np.asarray(jax.device_get(
            inverse_mass_matrix)),
        "latent_scale": np.asarray(jax.device_get(state.latent_scale)),
        "latent_covariance": np.asarray(jax.device_get(state.cov)),
        "log_s": np.asarray(jax.device_get(state.log_s)),
        "reflect_prob": np.asarray(reflect_prob),
        "sample_n_inner": np.asarray(sample_n_inner),
        "transport_systemic_phi": np.asarray(transport_systemic_phi),
    }
    return MaserBlackJaxResult(
        samples=_concat_dicts(sample_rows),
        log_density=(np.concatenate(log_density_rows, axis=0)
                     if log_density_rows else np.zeros((0,))),
        info=_concat_dicts(info_rows),
        warmup_info=warmup_info,
        parameters=parameters,
        theta_sites=target.names,
        runtime_seconds=runtime)


def run_blackjax_mcmc(model, init_params, rng_key, *,
                      num_warmup=1000, num_samples=1000, num_chains=1,
                      chain_workers=8,
                      h=None, n_inner=20, target_accept_theta=0.9,
                      target_accept_latent=0.35, theta_step_init=0.01,
                      phi_step_init=0.05, eps_init=None, eps_z_min=1e-3,
                      eps_z_max=0.5, eps_phi_min=1e-4, eps_phi_max=1.0,
                      adapt_rate=0.10, reflect_prob=0.25,
                      max_num_doublings=10, num_latent_burnin=0,
                      sample_n_inner=None,
                      transport_systemic_phi=False,
                      progress_bar=True, jit_steps=True):
    """Run explicit-phi chains with bounded CPU parallelism."""
    num_chains = int(num_chains)
    chain_workers = int(chain_workers)
    if num_chains < 1:
        raise ValueError("num_chains must be >= 1.")
    if chain_workers < 1:
        raise ValueError("chain_workers must be >= 1.")
    kwargs = dict(
        num_warmup=num_warmup,
        num_samples=num_samples,
        h=h,
        n_inner=n_inner,
        target_accept_theta=target_accept_theta,
        target_accept_latent=target_accept_latent,
        theta_step_init=theta_step_init,
        phi_step_init=phi_step_init,
        eps_init=eps_init,
        eps_z_min=eps_z_min,
        eps_z_max=eps_z_max,
        eps_phi_min=eps_phi_min,
        eps_phi_max=eps_phi_max,
        adapt_rate=adapt_rate,
        reflect_prob=reflect_prob,
        max_num_doublings=max_num_doublings,
        num_latent_burnin=num_latent_burnin,
        sample_n_inner=sample_n_inner,
        transport_systemic_phi=transport_systemic_phi,
        progress_bar=progress_bar,
        jit_steps=jit_steps,
    )
    if isinstance(init_params, (list, tuple)):
        if len(init_params) != num_chains:
            raise ValueError(
                "init_params must contain one dict per chain when a list is "
                "provided.")
        chain_init_params = list(init_params)
    else:
        chain_init_params = [init_params] * num_chains

    if num_chains == 1:
        return _run_blackjax_mcmc_one(
            model, chain_init_params[0], rng_key, **kwargs)

    chain_keys = jax.random.split(rng_key, num_chains)
    worker_count = _chain_worker_count(num_chains, chain_workers)

    def run_chain(item):
        i, chain_key = item
        # Keep each phase in its own row block, with one blank row between.
        positions = []
        block = 0
        for steps in (num_latent_burnin, num_warmup, num_samples):
            if int(steps) > 0:
                positions.append(block * (num_chains + 1) + i)
                block += 1
            else:
                positions.append(None)
        return _run_blackjax_mcmc_one(
            model, chain_init_params[i], chain_key,
            progress_label=f"{i + 1}/{num_chains}",
            progress_positions=tuple(positions), **kwargs)

    t0 = time.time()
    items = enumerate(chain_keys)
    if worker_count == 1:
        results = [run_chain(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(run_chain, items))
    return _stack_chain_results(
        results, time.time() - t0, worker_count)
