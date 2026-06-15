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
"""BlackJAX Metropolis-within-Gibbs sampler for megamasers.

The sampler targets the Gibbs megamaser disk likelihood:

* global disk parameters are sampled by BlackJAX NUTS in unconstrained
  Euclidean coordinates;
* the distance/mass block is sampled either as ``D_c`` and ``log_MBH``
  or, optionally, ``D_c`` and ``eta = log_MBH - log10(D_A)``;
* ``r_ang`` is represented by non-centred log-radius residuals,
  ``z_r = log(r_ang / r_hat(theta))``, and updated by vectorised
  per-spot random-walk Metropolis steps;
* warmup adapts the global NUTS step size and inverse mass matrix using the
  BlackJAX window-adaptation primitives, and adapts each radial proposal scale
  with a Robbins-Monro update.

The module does not import BlackJAX at import time.  This keeps the rest of the
package usable while BlackJAX is an optional dependency.
"""
import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Delta, Uniform
from numpyro.distributions.transforms import biject_to

from ..util import get_nested
from .maser_physics import C_a, C_v

# phys_args positional indices.
_I_DA, _I_MBH, _I_VSYS, _I_I0, _I_VARVHV, _I_SAF2 = 2, 3, 4, 8, 15, 16


def _require_blackjax():
    """Import BlackJAX and its window-adaptation module with a clear error."""
    try:
        blackjax = import_module("blackjax")
        window_adaptation = import_module(
            "blackjax.adaptation.window_adaptation")
    except ModuleNotFoundError as exc:
        if exc.name == "blackjax":
            raise RuntimeError(
                "BlackJAX is required for the megamaser Gibbs sampler. "
                "Install it in the active environment before running "
                "scripts/megamaser/run_maser.py."
            ) from exc
        raise
    return blackjax, window_adaptation


def _progress_iter(iterable, enabled, desc):
    """Return a tqdm-wrapped iterable when available and requested."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc)


def _update_progress_postfix(progress, info, step_size):
    """Show NUTS diagnostics in a tqdm progress bar."""
    if not hasattr(progress, "set_postfix"):
        return
    n_steps = int(np.asarray(info["nuts_num_integration_steps"]))
    div = int(np.asarray(info["nuts_is_divergent"]))
    acc = float(np.asarray(info["nuts_acceptance_rate"]))
    r_acc = float(np.asarray(info["r_accept_mean"]))
    step_size = float(np.asarray(step_size))
    progress.set_postfix({
        "steps": n_steps,
        "step_size": f"{step_size:.2e}",
        "acc": f"{acc:.3f}",
        "div": div,
        "r_acc": f"{r_acc:.3f}",
    })


def _seeds(model, phys_args):
    """Float32-stable (r_hat, r_min, r_max) from a phys_args tuple.

    ``model.gibbs_seeds`` is mathematically fine, but in float32 its masked
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
    r_vel = M_BH * (C_v * sin_i) ** 2 / (D_A * dv2_safe)

    use_accel = (~is_hv) & has_accel
    a_safe = jnp.where(
        use_accel, jnp.abs(model._all_a) + jnp.asarray(1e-6, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype))
    r_acc_arg = C_a * M_BH * sin_i / (D_A ** 2 * a_safe)
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

    pairs = [
        ("D_c", "D"),
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


def radial_z_rw_sweep(model, theta, z_r, eps, rng_key, h, n_inner=20,
                      spot_batch=None):
    """Vectorised random-walk Metropolis sweep over ``z_r`` sites."""
    r_ang0, r_hat, r_min, r_max, phys_args, phys_kw = _r_ang_from_z(
        model, theta, h, z_r)

    def logp(z_val):
        r_val = r_hat * jnp.exp(z_val)
        inside = (r_val > r_min) & (r_val < r_max)
        lp = _ll_per_spot(model, r_val, phys_args, phys_kw, spot_batch)
        # The chain state is z_r, while the model density is flat in
        # r_ang.  Add |dr_ang/dz_r| = r_ang for the coordinate transform.
        lp = lp + jnp.log(r_val)
        return jnp.where(inside, lp, -jnp.inf)

    valid0 = (r_ang0 > r_min) & (r_ang0 < r_max)
    lp0 = jnp.where(valid0, logp(z_r), -jnp.inf)
    accepted0 = jnp.zeros_like(z_r, dtype=jnp.int32)

    def body(_, state):
        z_cur, lp_cur, accepted, key = state
        key, k_prop, k_acc = jax.random.split(key, 3)
        z_prop = z_cur + eps * jax.random.normal(k_prop, z_cur.shape)
        lp_prop = logp(z_prop)
        log_alpha = lp_prop - lp_cur
        accept = (
            jnp.log(jax.random.uniform(k_acc, z_cur.shape)) < log_alpha)
        return (
            jnp.where(accept, z_prop, z_cur),
            jnp.where(accept, lp_prop, lp_cur),
            accepted + accept.astype(jnp.int32),
            key,
        )

    z_new, _, accepted, _ = jax.lax.fori_loop(
        0, int(n_inner), body, (z_r, lp0, accepted0, rng_key))
    return z_new, accepted


class MaserBlackJaxTarget:
    """Unconstrained global-parameter target for BlackJAX."""

    def __init__(self, model, h, init_params, spot_batch=None):
        if model.mode != "gibbs":
            raise ValueError(
                f"BlackJAX megamaser sampler requires mode='gibbs', "
                f"got {model.mode!r}.")
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
        D_c = params["D_c"]
        z_cosmo = self.model.distance2redshift(
            jnp.atleast_1d(D_c), h=self.h).squeeze()
        D_A = D_c / (1.0 + z_cosmo)
        out = dict(params)
        out["log_MBH"] = params["eta"] + jnp.log10(D_A)
        return out

    def initial_state(self, init_params):
        u_theta = self.unconstrain(init_params)
        theta, _ = self.constrain(u_theta)
        phys_args, _ = self.model.phys_from_params_jax(theta, self.h)
        r_hat, _, _ = _seeds(self.model, phys_args)
        if "r_ang" in init_params:
            r_ang = jnp.asarray(init_params["r_ang"], dtype=u_theta.dtype)
            z_r = jnp.log(r_ang / r_hat)
        elif "z_r" in init_params:
            z_r = jnp.asarray(init_params["z_r"], dtype=u_theta.dtype)
        else:
            z_r = jnp.zeros_like(r_hat, dtype=u_theta.dtype)
        return u_theta, z_r

    def logdensity(self, u_theta, z_r):
        theta, log_det = self.constrain(u_theta)
        lp = self.constrained_logdensity_z(theta, z_r)
        return lp + jnp.where(jnp.isfinite(lp), log_det, 0.0)

    def constrained_logdensity_z(self, theta, z_r):
        """Log posterior in constrained globals and non-centred radii."""
        r_ang, _, r_min, r_max, phys_args, phys_kw = _r_ang_from_z(
            self.model, theta, self.h, z_r)
        lp = self.constrained_logdensity_r_ang(
            theta, r_ang, r_min=r_min, r_max=r_max,
            phys_args=phys_args, phys_kw=phys_kw)
        logdet_r = jnp.sum(jnp.log(r_ang))
        return lp + jnp.where(jnp.isfinite(lp), logdet_r, 0.0)

    def constrained_logdensity(self, theta, r_ang):
        """Log posterior in constrained physical coordinates."""
        return self.constrained_logdensity_r_ang(theta, r_ang)

    def constrained_logdensity_r_ang(self, theta, r_ang, *,
                                     r_min=None, r_max=None,
                                     phys_args=None, phys_kw=None):
        """Log posterior in constrained globals and direct angular radii."""
        theta = self.complete_params(theta)
        if phys_args is None or phys_kw is None:
            phys_args, phys_kw = self.model.phys_from_params_jax(
                theta, self.h)
        if r_min is None or r_max is None:
            r_min, r_max = self.model.r_ang_range(phys_args[_I_DA])

        lp = jnp.asarray(0.0, dtype=jnp.asarray(r_ang).dtype)
        for site, _, prior in self.sites:
            if site == "eta":
                continue
            lp = lp + prior.log_prob(theta[site])
        if self.mass_parameterization == "eta":
            lp = lp + self.model.priors["log_MBH"].log_prob(
                theta["log_MBH"])

        if self.model._D_c_volume:
            lp = lp + 2.0 * jnp.log(theta["D_c"])
        if self.model.use_ecc and self.model.ecc_cartesian:
            r2 = theta["e_x"] ** 2 + theta["e_y"] ** 2
            lp = lp + jnp.log(4.0 / jnp.pi) - 0.5 * jnp.log(r2 + 1e-6)

        inside = jnp.all((r_ang > r_min) & (r_ang < r_max))
        groups = self.model._spot_groups_from_r(r_ang)
        ll = self.model._sum_phi_marginal(
            groups, phys_args, phys_kw, spot_batch=self.spot_batch)
        return jnp.where(inside, lp + ll, -jnp.inf)

    def sample_dict(self, u_theta, z_r):
        theta, _ = self.constrain(u_theta)
        out = dict(theta)
        r_ang, _, _, _, _, _ = _r_ang_from_z(
            self.model, theta, self.h, z_r)
        out["r_ang"] = r_ang
        return out


class MaserBlackJaxState(NamedTuple):
    theta: Any
    z_r: Any
    eps_z: Any


@dataclass
class MaserBlackJaxResult:
    """Return container for a single BlackJAX megamaser chain."""

    samples: Dict[str, np.ndarray]
    info: Dict[str, np.ndarray]
    warmup_info: Dict[str, np.ndarray]
    parameters: Dict[str, np.ndarray]
    theta_sites: Tuple[str, ...]
    runtime_seconds: float


def _make_warmup_step(blackjax, adapt_step, target, *,
                      max_num_doublings, n_inner, spot_batch,
                      target_accept_r, r_adapt_rate, eps_min, eps_max):
    nuts_step = blackjax.nuts.build_kernel()

    def step(state, adaptation_state, rng_key, adaptation_stage, step_index):
        key_r, key_theta = jax.random.split(rng_key)
        theta_params, _ = target.constrain(state.theta.position)
        z_r, r_accept = radial_z_rw_sweep(
            target.model, theta_params, state.z_r, state.eps_z, key_r,
            target.h,
            n_inner=n_inner, spot_batch=spot_batch)

        def logdensity_theta(u_theta):
            return target.logdensity(u_theta, z_r)

        theta_state = blackjax.nuts.init(
            state.theta.position, logdensity_theta)
        theta_state, nuts_info = nuts_step(
            key_theta, theta_state, logdensity_theta,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            max_num_doublings=max_num_doublings)
        adaptation_state = adapt_step(
            adaptation_state, adaptation_stage, theta_state.position,
            nuts_info.acceptance_rate)

        r_accept_rate = r_accept / jnp.asarray(
            n_inner, dtype=state.eps_z.dtype)
        gamma = r_adapt_rate / jnp.sqrt(
            jnp.asarray(step_index + 1, dtype=state.eps_z.dtype))
        eps_z = state.eps_z * jnp.exp(
            gamma * (r_accept_rate - target_accept_r))
        eps_z = jnp.clip(eps_z, eps_min, eps_max)

        new_state = MaserBlackJaxState(theta_state, z_r, eps_z)
        info = {
            "nuts_acceptance_rate": nuts_info.acceptance_rate,
            "nuts_is_divergent": nuts_info.is_divergent,
            "nuts_num_integration_steps": nuts_info.num_integration_steps,
            "r_accept_mean": jnp.mean(r_accept_rate),
            "r_accept_min": jnp.min(r_accept_rate),
            "r_accept_max": jnp.max(r_accept_rate),
            "eps_z_mean": jnp.mean(eps_z),
            "eps_z_min": jnp.min(eps_z),
            "eps_z_max": jnp.max(eps_z),
        }
        return new_state, adaptation_state, info

    return step


def _make_sample_step(blackjax, target, parameters, *,
                      max_num_doublings, n_inner, spot_batch):
    nuts_step = blackjax.nuts.build_kernel()
    step_size = parameters["step_size"]
    inverse_mass_matrix = parameters["inverse_mass_matrix"]

    def step(state, rng_key):
        key_r, key_theta = jax.random.split(rng_key)
        theta_params, _ = target.constrain(state.theta.position)
        z_r, r_accept = radial_z_rw_sweep(
            target.model, theta_params, state.z_r, state.eps_z, key_r,
            target.h,
            n_inner=n_inner, spot_batch=spot_batch)

        def logdensity_theta(u_theta):
            return target.logdensity(u_theta, z_r)

        theta_state = blackjax.nuts.init(
            state.theta.position, logdensity_theta)
        theta_state, nuts_info = nuts_step(
            key_theta, theta_state, logdensity_theta,
            step_size, inverse_mass_matrix,
            max_num_doublings=max_num_doublings)

        new_state = MaserBlackJaxState(theta_state, z_r, state.eps_z)
        r_accept_rate = r_accept / jnp.asarray(
            n_inner, dtype=state.eps_z.dtype)
        sample = target.sample_dict(theta_state.position, z_r)
        info = {
            "nuts_acceptance_rate": nuts_info.acceptance_rate,
            "nuts_is_divergent": nuts_info.is_divergent,
            "nuts_num_integration_steps": nuts_info.num_integration_steps,
            "r_accept_mean": jnp.mean(r_accept_rate),
            "r_accept_min": jnp.min(r_accept_rate),
            "r_accept_max": jnp.max(r_accept_rate),
        }
        return new_state, sample, info

    return step


def run_blackjax_mwg(model, init_params, rng_key, *,
                     num_warmup=1000, num_samples=1000, h=None,
                     n_inner=20, spot_batch=None, dense_mass=True,
                     target_accept_nuts=0.9, target_accept_r=0.44,
                     initial_step_size=0.01, max_num_doublings=10,
                     eps_init=None, eps_min=1e-3, eps_max=0.5,
                     r_adapt_rate=0.10, progress_bar=True,
                     jit_steps=True):
    """Run one BlackJAX Metropolis-within-Gibbs megamaser chain.

    This function always runs exactly one chain.  Use separate invocations for
    independent chains.
    """
    if num_warmup < 0 or num_samples < 1:
        raise ValueError(
            "num_warmup must be >= 0 and num_samples must be > 0.")

    blackjax, window_adaptation = _require_blackjax()
    if h is None:
        h = float(get_nested(model.config, "model/H0_ref", 73.0)) / 100.0

    target = MaserBlackJaxTarget(model, h, init_params, spot_batch=spot_batch)
    u_theta, z_r = target.initial_state(init_params)
    theta0, _ = target.constrain(u_theta)
    if eps_init is None:
        eps_z = estimate_radial_eps(
            model, theta0, h, eps_min=eps_min, eps_max=eps_max)
    else:
        eps_z = jnp.asarray(eps_init, dtype=u_theta.dtype)
    eps_z = jnp.broadcast_to(eps_z, (model.n_spots,))

    def logdensity0(u):
        return target.logdensity(u, z_r)

    theta_state = blackjax.nuts.init(u_theta, logdensity0)
    state = MaserBlackJaxState(theta_state, z_r, eps_z)

    adapt_init, adapt_step, adapt_final = window_adaptation.base(
        is_mass_matrix_diagonal=not dense_mass,
        target_acceptance_rate=target_accept_nuts)
    adaptation_state = adapt_init(u_theta, initial_step_size)
    schedule = window_adaptation.build_schedule(int(num_warmup))

    warmup_step = _make_warmup_step(
        blackjax, adapt_step, target,
        max_num_doublings=max_num_doublings,
        n_inner=n_inner,
        spot_batch=spot_batch,
        target_accept_r=target_accept_r,
        r_adapt_rate=r_adapt_rate,
        eps_min=eps_min,
        eps_max=eps_max)
    t0 = time.time()

    if jit_steps:
        warmup_step = jax.jit(warmup_step)

    warmup_rows = []
    key = rng_key
    warmup_progress = _progress_iter(
        range(int(num_warmup)), progress_bar, "warmup")
    for i in warmup_progress:
        key, subkey = jax.random.split(key)
        state, adaptation_state, info = warmup_step(
            state, adaptation_state, subkey, schedule[i], i)
        info_host = jax.device_get(info)
        warmup_rows.append(info_host)
        _update_progress_postfix(
            warmup_progress, info_host,
            jax.device_get(adaptation_state.step_size))

    step_size, inverse_mass_matrix = adapt_final(adaptation_state)
    sample_parameters = {
        "step_size": step_size,
        "inverse_mass_matrix": inverse_mass_matrix,
    }

    sample_step = _make_sample_step(
        blackjax, target, sample_parameters,
        max_num_doublings=max_num_doublings,
        n_inner=n_inner,
        spot_batch=spot_batch)
    if jit_steps:
        sample_step = jax.jit(sample_step)

    sample_rows = []
    info_rows = []
    sample_progress = _progress_iter(
        range(int(num_samples)), progress_bar, "sample")
    sample_step_size = jax.device_get(sample_parameters["step_size"])
    for _ in sample_progress:
        key, subkey = jax.random.split(key)
        state, sample, info = sample_step(state, subkey)
        sample_rows.append(jax.device_get(sample))
        info_host = jax.device_get(info)
        info_rows.append(info_host)
        _update_progress_postfix(
            sample_progress, info_host, sample_step_size)

    # Synchronise outstanding work before reporting the runtime.
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready()
        if hasattr(x, "block_until_ready") else x,
        state)
    runtime = time.time() - t0

    parameters = {
        "step_size": np.asarray(
            jax.device_get(sample_parameters["step_size"])),
        "inverse_mass_matrix": np.asarray(
            jax.device_get(sample_parameters["inverse_mass_matrix"])),
        "eps_z": np.asarray(jax.device_get(state.eps_z)),
    }
    return MaserBlackJaxResult(
        samples=_stack_dicts(sample_rows),
        info=_stack_dicts(info_rows),
        warmup_info=_stack_dicts(warmup_rows),
        parameters=parameters,
        theta_sites=target.names,
        runtime_seconds=runtime)
