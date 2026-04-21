# Copyright (C) 2025 Richard Stiskalek
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
"""MAP optimizers for NumPyro models.

Two strategies:
  - **Sobol + Adam**: multi-start gradient-based optimizer. Good for
    low-dimensional problems or models without expensive marginalization.
  - **DE** (Differential Evolution): derivative-free global search
    using evosax. Better for high-dimensional problems with expensive
    likelihood (e.g. maser disk model with r+phi marginalization).

Both operate in constrained space for the global survey, then (Adam only)
switch to unconstrained space for gradient optimization.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpyro import handlers
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import (constrain_fn, log_density, potential_energy,
                                unconstrain_fn)
from scipy.stats.qmc import Sobol
from tqdm import trange

from ..util import fprint, fsection

# -----------------------------------------------------------------------
# Bounds extraction
# -----------------------------------------------------------------------


def _prior_bounds(dist, sobol_n_sigma=5):
    """Derive finite optimizer bounds from a numpyro distribution.

    For Uniform priors, uses the support directly. For all other
    distributions (Normal, TruncatedNormal, HalfNormal, ...),
    uses mean +/- sobol_n_sigma * std clipped to any finite support bounds.
    This avoids wasting Sobol points in low-probability tails
    (e.g. TruncatedNormal(2, 5, 0, 100) -> [0, 27] not [0, 100]).

    Delta priors return (None, None) — caller should skip.
    """
    dist_name = dist.__class__.__name__
    if dist_name == "Delta":
        return None, None

    support = dist.support
    lb = getattr(support, "lower_bound", None)
    ub = getattr(support, "upper_bound", None)
    lb = float(lb) if lb is not None else -np.inf
    ub = float(ub) if ub is not None else np.inf

    # Uniform or no tightening requested: use support directly
    if dist_name == "Uniform" or sobol_n_sigma is None:
        return lb, ub

    # Try to tighten with mean +/- sobol_n_sigma * std.
    # Some distributions (e.g. TruncatedNormal) don't implement variance
    # but have a base_dist that does.
    try:
        mu = float(dist.mean)
        sigma = float(dist.variance ** 0.5)
    except (NotImplementedError, AttributeError, TypeError):
        base = getattr(dist, "base_dist", None)
        if base is not None:
            try:
                mu = float(base.mean)
                sigma = float(base.variance ** 0.5)
            except (NotImplementedError, AttributeError, TypeError):
                return lb, ub
        else:
            return lb, ub

    tight_lo = mu - sobol_n_sigma * sigma
    tight_hi = mu + sobol_n_sigma * sigma

    # Clip to support (if finite)
    lb = max(lb, tight_lo) if np.isfinite(lb) else tight_lo
    ub = min(ub, tight_hi) if np.isfinite(ub) else tight_hi
    return lb, ub


def _get_bounds_from_trace(model, model_args, model_kwargs, sobol_n_sigma=5,
                           seed=42):
    """Extract optimizer bounds from model priors.

    Uniform priors use their support. Others use mean +/- n_sigma * std
    clipped to support. Delta priors are skipped.
    """
    substituted = handlers.substitute(
        handlers.seed(model, rng_seed=seed),
        substitute_fn=init_to_median(num_samples=15),
    )
    tr = handlers.trace(substituted).get_trace(*model_args, **model_kwargs)

    names, lo, hi, sizes = [], [], [], []
    for k, v in tr.items():
        if v["type"] == "sample" and not v.get("is_observed", False):
            lb, ub = _prior_bounds(v["fn"], sobol_n_sigma=sobol_n_sigma)
            if lb is None:
                continue
            size = int(np.prod(v["value"].shape))
            lb = np.atleast_1d(np.asarray(lb))
            ub = np.atleast_1d(np.asarray(ub))
            names.append(k)
            lo.append(np.broadcast_to(lb.ravel(), size))
            hi.append(np.broadcast_to(ub.ravel(), size))
            sizes.append(size)

    return names, sizes, np.concatenate(lo), np.concatenate(hi)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _build_logp_flat(model, model_args, model_kwargs, names, sizes):
    """Build a flat-vector log-density function in constrained space."""
    def logp(x):
        params = {}
        offset = 0
        for name, size in zip(names, sizes):
            params[name] = x[offset:offset + size].reshape(()) \
                if size == 1 else x[offset:offset + size]
            offset += size
        ld, _ = log_density(model, model_args, model_kwargs, params)
        return ld
    return logp


def _build_neg_potential_flat(model, model_args, model_kwargs, names, sizes):
    """Build a flat-vector negative potential energy in unconstrained space.

    Uses ``potential_energy`` which automatically includes the Jacobian
    correction for the constrained->unconstrained bijection.
    Returns -U(z) = log p(constrain(z)) + log|det J|.
    """
    def neg_U(z):
        params = {}
        offset = 0
        for name, size in zip(names, sizes):
            params[name] = z[offset:offset + size].reshape(()) \
                if size == 1 else z[offset:offset + size]
            offset += size
        U = potential_energy(model, model_args, model_kwargs, params)
        return -U
    return neg_U


def _constrained_to_flat(params, names, sizes):
    """Pack a param dict into a flat array (constrained space)."""
    parts = []
    for name, size in zip(names, sizes):
        v = jnp.atleast_1d(jnp.asarray(params[name])).ravel()
        parts.append(v)
    return jnp.concatenate(parts)


def _reflect_bounds(x):
    """Reflect out-of-bounds values back into [0, 1].

    Uses triangle-wave folding so that values beyond the boundary are
    reflected rather than clipped. This prevents boundary-attractor
    pathology in DE where clipping causes difference vectors to
    collapse.
    """
    x = jnp.abs(x)
    cycle = jnp.floor(x).astype(jnp.int32)
    frac = x - jnp.floor(x)
    return jnp.where(cycle % 2 == 0, frac, 1.0 - frac)


def _flat_to_dict(x, names, sizes):
    """Unpack a flat array into a param dict."""
    params = {}
    offset = 0
    for name, size in zip(names, sizes):
        params[name] = x[offset:offset + size].reshape(()) \
            if size == 1 else x[offset:offset + size]
        offset += size
    return params


def _constrain_flat(z_flat, model, model_args, model_kwargs, names, sizes):
    """Map flat unconstrained vector -> flat constrained vector."""
    z_dict = _flat_to_dict(z_flat, names, sizes)
    c_dict = constrain_fn(model, model_args, model_kwargs, z_dict)
    return _constrained_to_flat(c_dict, names, sizes)


def _unconstrain_flat(x_flat, model, model_args, model_kwargs, names, sizes):
    """Map flat constrained vector -> flat unconstrained vector."""
    x_dict = _flat_to_dict(x_flat, names, sizes)
    z_dict = unconstrain_fn(model, model_args, model_kwargs, x_dict)
    return _constrained_to_flat(z_dict, names, sizes)


def _select_distinct(points, logp_vals, M, min_dist_frac=0.01):
    """Select top-M points that are sufficiently distinct.

    Greedily picks the best point, then skips any candidate within
    min_dist_frac (L-inf normalised by range) of already selected points.
    Falls back to top-M if fewer than M distinct points found.
    """
    order = np.argsort(-logp_vals)
    selected = [order[0]]
    spread = np.max(points, axis=0) - np.min(points, axis=0)
    spread = np.where(spread > 0, spread, 1.0)

    for idx in order[1:]:
        if len(selected) >= M:
            break
        pt = points[idx]
        too_close = False
        for s in selected:
            if np.all(np.abs(pt - points[s]) / spread < min_dist_frac):
                too_close = True
                break
        if not too_close:
            selected.append(idx)

    if len(selected) < M:
        for idx in order:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= M:
                break

    return np.array(selected[:M])


def _cluster_modes(points, logp_vals, names, sizes, negU_vals=None,
                   scale=None, tol=0.02):
    """Cluster final optimizer points into modes.

    Two points belong to the same mode if their L-inf distance
    (normalised by ``scale``) is < tol for all dimensions.

    ``scale`` should be the prior range (hi - lo). If None, falls back
    to the spread of the final points (unreliable when all converge).

    Returns list of dicts, each with 'logp', 'negU' (optional), 'count',
    'best_idx', 'params' (best point as dict), sorted by logP descending.
    """
    if scale is None:
        scale = np.max(points, axis=0) - np.min(points, axis=0)
    scale = np.where(scale > 0, scale, 1.0)

    order = np.argsort(-logp_vals)
    modes = []  # each: {'indices': [...], 'best_idx': int}

    for idx in order:
        matched = False
        for mode in modes:
            ref = points[mode["best_idx"]]
            if np.all(np.abs(points[idx] - ref) / scale < tol):
                mode["indices"].append(idx)
                matched = True
                break
        if not matched:
            modes.append({"indices": [idx], "best_idx": idx})

    result = []
    for mode in modes:
        best = mode["best_idx"]
        params = {}
        offset = 0
        for name, size in zip(names, sizes):
            val = points[best, offset:offset + size]
            params[name] = float(val.ravel()[0]) if size == 1 else val
            offset += size
        entry = {
            "logp": float(logp_vals[best]),
            "count": len(mode["indices"]),
            "best_idx": best,
            "params": params,
        }
        if negU_vals is not None:
            entry["negU"] = float(negU_vals[best])
        result.append(entry)

    result.sort(key=lambda m: -m["logp"])
    return result


def _short_name(name, max_len=8):
    """Shorten parameter name for table display."""
    _ABBREV = {
        "sigma_x_floor": "sx_fl",
        "sigma_y_floor": "sy_fl",
        "sigma_v_sys": "sv_sys",
        "sigma_v_hv": "sv_hv",
        "sigma_a_floor": "sa_fl",
        "dOmega_dr": "dOm_dr",
        "Omega0": "Om0",
    }
    return _ABBREV.get(name, name)[:max_len]


def _auto_fmt(val):
    """Format a float with adaptive decimal places."""
    av = abs(val)
    if av == 0:
        return f"{val:>8.1f}"
    if av >= 100:
        return f"{val:>8.1f}"
    if av >= 1:
        return f"{val:>8.2f}"
    if av >= 0.01:
        return f"{val:>8.4f}"
    return f"{val:>8.1e}"


def _print_points_table(points, logp_vals, names, sizes,
                        negU_vals=None, max_params=20):
    """Print a table of M points with their logP and -U values."""
    cols, col_offsets = [], []
    offset = 0
    for name, size in zip(names, sizes):
        if size == 1:
            cols.append(name)
            col_offsets.append(offset)
        offset += size
        if len(cols) >= max_params:
            break

    short = [_short_name(c) for c in cols]
    hdr = f"{'#':>3s} {'logP':>8s}"
    if negU_vals is not None:
        hdr += f" {'-U':>8s}"
    for s in short:
        hdr += f" {s:>8s}"
    fprint(hdr)
    fprint("-" * len(hdr))

    order = np.argsort(-logp_vals)
    for rank, idx in enumerate(order):
        row = f"{rank:3d} {logp_vals[idx]:8.1f}"
        if negU_vals is not None:
            row += f" {negU_vals[idx]:8.1f}"
        for col_off in col_offsets:
            row += f" {_auto_fmt(points[idx, col_off])}"
        fprint(row)


def _print_modes(modes, names, sizes, max_params=20):
    """Print mode summary: one block per mode."""
    n_modes = len(modes)
    total = sum(m["count"] for m in modes)

    if n_modes == 1:
        fprint(f"unimodal: all {total} starts converged to the same mode")
    else:
        fprint(f"found {n_modes} modes from {total} starts")
        delta = modes[0]["logp"] - modes[1]["logp"]
        if delta < 5:
            fprint(f"WARNING: top 2 modes differ by only {delta:.1f} nats "
                   f"-- posterior may be multimodal")

    scalar_names = [n for n, s in zip(names, sizes) if s == 1][:max_params]

    for i, mode in enumerate(modes):
        line = f"\nmode {i}: logP = {mode['logp']:.2f}"
        if "negU" in mode:
            line += f", -U = {mode['negU']:.2f}"
        line += f" ({mode['count']}/{total} starts)"
        fprint(line)
        p = mode["params"]
        for name in scalar_names:
            if name in p and isinstance(p[name], (float, int)):
                fprint(f"  {_short_name(name, 15):15s}: "
                       f"{_auto_fmt(p[name])}")


# -----------------------------------------------------------------------
# Inner optimizers (unconstrained space)
# -----------------------------------------------------------------------


def _run_adam(z0, neg_U_fn, n_steps, lr, lr_end, n_restarts, seed,
              verbose):
    """Run parallel Adam optimization in unconstrained space.

    Full state reset (m_t, v_t) at each warm restart prevents stale
    adaptive state from suppressing the Jacobian restoring force in
    the logit-transformed space.
    """
    neg_U_batch = jax.jit(jax.vmap(neg_U_fn))
    steps_per_cycle = n_steps // n_restarts

    def _make_optimizer(lr_init):
        schedule = optax.cosine_decay_schedule(
            init_value=lr_init, decay_steps=steps_per_cycle,
            alpha=lr_end / lr_init)
        return optax.adam(schedule)

    optimizer = _make_optimizer(lr)

    @jax.jit
    def step(z, opt_state):
        def _single(zi, osi):
            g = jax.grad(lambda zz: -neg_U_fn(zz))(zi)
            updates, new_osi = optimizer.update(g, osi)
            zi_new = optax.apply_updates(zi, updates)
            return zi_new, new_osi
        return jax.vmap(_single)(z, opt_state)

    opt_state = jax.vmap(optimizer.init)(z0)

    # Compile
    t0 = time.time()
    z_cur, opt_state = step(z0, opt_state)
    jax.block_until_ready(z_cur)
    if verbose:
        fprint(f"Adam JIT compiled in {time.time() - t0:.1f}s "
               f"({n_restarts} restarts with state reset)")

    # Track best across all restarts
    best_z = z0
    best_negU = np.asarray(neg_U_batch(z0))

    t0 = time.time()
    for restart in range(n_restarts):
        if restart > 0:
            # Full state reset: clear m_t and v_t, keep positions
            opt_state = jax.vmap(optimizer.init)(z_cur)
            if verbose:
                fprint(f"  restart {restart}: reset Adam state")

        start_step = restart * steps_per_cycle
        end_step = (restart + 1) * steps_per_cycle
        desc = f"Adam[{restart}]" if n_restarts > 1 else "Adam"
        pbar = trange(start_step, end_step, desc=desc, disable=not verbose)

        for s in pbar:
            z_cur, opt_state = step(z_cur, opt_state)
            if (s + 1) % 100 == 0:
                jax.block_until_ready(z_cur)
                negU_now = np.asarray(neg_U_batch(z_cur))
                # Update best
                improved = negU_now > best_negU
                if np.any(improved):
                    best_z = jnp.where(
                        jnp.array(improved)[:, None], z_cur, best_z)
                    best_negU = np.where(improved, negU_now, best_negU)
                cycle_step = (s + 1) - restart * steps_per_cycle
                cur_lr = float(optax.cosine_decay_schedule(
                    init_value=lr, decay_steps=steps_per_cycle,
                    alpha=lr_end / lr)(cycle_step))
                pbar.set_postfix(negU=f"{float(best_negU.max()):.1f}",
                                 lr=f"{cur_lr:.1e}")

    jax.block_until_ready(best_z)
    if verbose:
        fprint(f"Adam done in {time.time() - t0:.1f}s")

    return best_z


# -----------------------------------------------------------------------
# Main optimizer
# -----------------------------------------------------------------------


def sobol_optimize(model, model_args=(), model_kwargs=None,
                   log2_N=14, M=10, n_steps=5000,
                   lr=0.1, lr_end=0.005, n_restarts=3,
                   sobol_n_sigma=1, sobol_batch=1024,
                   min_dist_frac=0.01, seed=42, verbose=True):
    """Multi-start MAP optimizer: Sobol survey + parallel Adam.

    Optimization runs in **unconstrained space** using NumPyro's bijective
    transforms, eliminating boundary-sticking artifacts. The Sobol survey
    is done in constrained space (for interpretable bounds), then points
    are transformed to unconstrained space for optimization.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    model_args : tuple
        Positional arguments for the model.
    model_kwargs : dict or None
        Keyword arguments for the model.
    log2_N : int
        log2 of Sobol sample count (default 2^14 = 16384).
    M : int
        Number of parallel starts.
    n_steps : int
        Total gradient evaluations per start.
    lr : float
        Peak learning rate.
    lr_end : float
        Minimum learning rate.
    n_restarts : int
        Number of cosine warm restart cycles.
    sobol_n_sigma : float
        For non-Uniform priors, Sobol bounds are mean +/- sobol_n_sigma * std
        (clipped to support). Use 1 for tight search, 5 for wide.
    sobol_batch : int
        Batch size for Sobol evaluation (memory control).
    min_dist_frac : float
        Minimum L-inf distance (as fraction of range) between starts.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    best_params : dict
        Best MAP parameter values (constrained space).
    best_logp : float
        Log-density at the MAP (constrained space, no Jacobian).
    all_results : dict
        All M results: 'params' (M, D) in constrained space,
        'logp' (M,), 'names', 'sizes', 'modes' (list of mode dicts).
    """
    if model_kwargs is None:
        model_kwargs = {}

    # --- Extract bounds (constrained space, for Sobol survey) ---
    names, sizes, lo_sobol, hi_sobol = _get_bounds_from_trace(
        model, model_args, model_kwargs,
        sobol_n_sigma=sobol_n_sigma, seed=seed)
    D = len(lo_sobol)
    N_sobol = 2**log2_N

    if verbose:
        fsection("Sobol + Adam MAP optimizer (unconstrained)")
        fprint(f"{D}D, {N_sobol} Sobol points, {M} starts "
               f"x {n_steps} steps")
        fprint(f"LR: {lr:.1e} -> {lr_end:.1e} "
               f"(cosine, {n_restarts} restarts)")
        if D <= 50:
            offset = 0
            for name, size in zip(names, sizes):
                s_lo, s_hi = lo_sobol[offset], hi_sobol[offset]
                fprint(f"  {name:20s}: Sobol [{s_lo:.4g}, {s_hi:.4g}]")
                offset += size

    # --- Build log-density functions ---
    logp_fn = _build_logp_flat(
        model, model_args, model_kwargs, names, sizes)
    logp_batch = jax.jit(jax.vmap(logp_fn))

    neg_U_fn = _build_neg_potential_flat(
        model, model_args, model_kwargs, names, sizes)

    # Vectorized constrained <-> unconstrained transforms
    def _to_unconst(x):
        return _unconstrain_flat(
            x, model, model_args, model_kwargs, names, sizes)

    def _to_const(z):
        return _constrain_flat(
            z, model, model_args, model_kwargs, names, sizes)
    to_unconst_batch = jax.jit(jax.vmap(_to_unconst))
    to_const_batch = jax.jit(jax.vmap(_to_const))

    # --- Compile ---
    t0 = time.time()
    x_test = jnp.tile(
        jnp.array(0.5 * (lo_sobol + hi_sobol))[None, :], (4, 1))
    _ = logp_batch(x_test)
    jax.block_until_ready(_)
    if verbose:
        fprint(f"JIT compiled in {time.time() - t0:.1f}s")

    # --- Sobol survey (constrained space) ---
    sampler = Sobol(d=D, scramble=True, seed=seed)
    sobol_01 = sampler.random(N_sobol)
    sobol_points = lo_sobol + sobol_01 * (hi_sobol - lo_sobol)
    sobol_jax = jnp.array(sobol_points)

    t0 = time.time()
    logp_all = []
    n_batches = (N_sobol + sobol_batch - 1) // sobol_batch
    for i in trange(n_batches, desc="Sobol", disable=not verbose):
        start = i * sobol_batch
        end = min(start + sobol_batch, N_sobol)
        vals = logp_batch(sobol_jax[start:end])
        jax.block_until_ready(vals)
        logp_all.append(np.asarray(vals))
    logp_all = np.concatenate(logp_all)
    valid = np.isfinite(logp_all)
    logp_all = np.where(valid, logp_all, -np.inf)

    if verbose:
        fprint(f"Sobol done in {time.time() - t0:.1f}s "
               f"({valid.sum()}/{N_sobol} valid, "
               f"best logP = {logp_all[valid].max():.1f})")

    # --- Select M distinct starts ---
    selected = _select_distinct(sobol_points, logp_all, M, min_dist_frac)
    x0_constrained = jnp.array(sobol_points[selected])

    if verbose:
        fsection("Starting points (constrained)")
        _print_points_table(sobol_points[selected], logp_all[selected],
                            names, sizes)

    # --- Transform to unconstrained space ---
    t0 = time.time()
    z0 = to_unconst_batch(x0_constrained)
    jax.block_until_ready(z0)
    if verbose:
        fprint(f"Transformed to unconstrained space in "
               f"{time.time() - t0:.1f}s")

    # --- Run optimizer ---
    z_final = _run_adam(
        z0, neg_U_fn, n_steps=n_steps, lr=lr, lr_end=lr_end,
        n_restarts=n_restarts, seed=seed, verbose=verbose)

    # --- Transform back to constrained space ---
    x_final = np.asarray(to_const_batch(z_final))

    # Evaluate both constrained logP and unconstrained -U
    neg_U_batch = jax.jit(jax.vmap(neg_U_fn))
    logp_final = np.asarray(logp_batch(jnp.array(x_final)))
    negU_final = np.asarray(neg_U_batch(z_final))

    modes = _cluster_modes(x_final, logp_final, names, sizes,
                           negU_vals=negU_final,
                           scale=hi_sobol - lo_sobol)

    best_idx = np.argmax(logp_final)
    best_logp = float(logp_final[best_idx])
    best_params = modes[0]["params"]

    if verbose:
        fsection("Final points (constrained)")
        _print_points_table(x_final, logp_final, names, sizes,
                            negU_vals=negU_final)
        fsection("Mode summary")
        _print_modes(modes, names, sizes)

    all_results = {
        "params": x_final,
        "logp": logp_final,
        "negU": negU_final,
        "names": names,
        "sizes": sizes,
        "lo_sobol": lo_sobol,
        "hi_sobol": hi_sobol,
        "modes": modes,
    }

    return best_params, best_logp, all_results


# -----------------------------------------------------------------------
# Differential Evolution optimizer (derivative-free)
# -----------------------------------------------------------------------


def de_optimize(model, model_args=(), model_kwargs=None,
                log2_N=16, pop_size=1000, max_generations=1000,
                patience=100, eval_chunk=64,
                sobol_n_sigma=5, min_dist_frac=0.005,
                sobol_bounds_override=None,
                log_every=100, seed=42, verbose=True):
    """Derivative-free MAP optimizer using Differential Evolution.

    Strategy:
      1. Sobol survey in constrained space to seed the initial population.
      2. DE evolves the population until convergence (no improvement
         for ``patience`` generations).

    Uses reflection boundary handling to avoid boundary-attractor
    pathology (clipping causes DE difference vectors to collapse).

    The Sobol survey can use tighter bounds than the DE search space
    via ``sobol_bounds_override`` (e.g. Hubble-flow distance estimate
    for the distance parameter). DE always operates in the full prior
    bounds.

    Operates entirely in constrained space (no gradient needed).
    Requires ``evosax`` (JAX-native evolutionary strategies).

    Parameters
    ----------
    model : callable
        NumPyro model function.
    model_args, model_kwargs
        Arguments for the model.
    log2_N : int
        log2 of Sobol sample count (default 2^16 = 65536).
    pop_size : int
        DE population size.
    max_generations : int
        Maximum number of DE generations.
    patience : int
        Stop if no improvement (> 0.1 nats) for this many generations.
    eval_chunk : int
        Batch size for fitness evaluations (GPU memory control).
    sobol_n_sigma : float
        Sobol bounds width (mean +/- n_sigma * std, clipped to support).
    min_dist_frac : float
        Minimum L-inf distance between initial population members.
    sobol_bounds_override : dict or None
        Override Sobol survey bounds for specific parameters. Keys are
        parameter names, values are (lo, hi) tuples. The DE search
        space is unaffected.
    log_every : int
        Print progress every N generations.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    best_params : dict
        Best MAP parameter values (constrained space).
    best_logp : float
        Log-density at the MAP.
    all_results : dict
        Contains 'names', 'sizes', 'lo_sobol', 'hi_sobol'.
    """
    from evosax.algorithms import DifferentialEvolution

    if not getattr(model, "marginalise_r", False):
        raise ValueError(
            "DE optimizer requires marginalise_r=True. "
            "Set model/marginalise_r = true in the config.")

    if model_kwargs is None:
        model_kwargs = {}

    names, sizes, lo, hi = _get_bounds_from_trace(
        model, model_args, model_kwargs,
        sobol_n_sigma=sobol_n_sigma, seed=seed)
    D = len(lo)
    scale = hi - lo
    N_sobol = 2**log2_N

    # Sobol bounds: default to prior bounds, override where requested
    sobol_lo = lo.copy()
    sobol_hi = hi.copy()
    if sobol_bounds_override is not None:
        offset = 0
        for name, size in zip(names, sizes):
            if name in sobol_bounds_override:
                s_lo, s_hi = sobol_bounds_override[name]
                sobol_lo[offset:offset + size] = max(s_lo, lo[offset])
                sobol_hi[offset:offset + size] = min(s_hi, hi[offset])
            offset += size

    if verbose:
        fsection("DE MAP optimizer")
        fprint(f"{D}D, pop={pop_size}, max_gen={max_generations}, "
               f"patience={patience}")
        if D <= 50:
            offset = 0
            for name, size in zip(names, sizes):
                p_lo, p_hi = lo[offset], hi[offset]
                s_lo, s_hi = sobol_lo[offset], sobol_hi[offset]
                if s_lo != p_lo or s_hi != p_hi:
                    fprint(f"  {name:20s}: [{p_lo:.4g}, {p_hi:.4g}] "
                           f"(Sobol: [{s_lo:.4g}, {s_hi:.4g}])")
                else:
                    fprint(f"  {name:20s}: [{p_lo:.4g}, {p_hi:.4g}]")
                offset += size

    # Build log-density
    logp_fn = _build_logp_flat(model, model_args, model_kwargs, names, sizes)
    _logp_batch_raw = jax.jit(jax.vmap(logp_fn))

    def _fitness_single(x_normed):
        x = lo + x_normed * scale
        return -logp_fn(x)

    _fitness_raw = jax.jit(jax.vmap(_fitness_single))

    def fitness_batch(x_normed):
        n = x_normed.shape[0]
        if n <= eval_chunk:
            return _fitness_raw(x_normed)
        parts = []
        for i in range(0, n, eval_chunk):
            parts.append(_fitness_raw(x_normed[i:i + eval_chunk]))
            jax.block_until_ready(parts[-1])
        return jnp.concatenate(parts)

    # Compile
    t0 = time.time()
    x_test = jnp.tile(jnp.array(0.5 * (lo + hi))[None, :], (4, 1))
    _ = _logp_batch_raw(x_test)
    jax.block_until_ready(_)
    if verbose:
        fprint(f"JIT compiled in {time.time() - t0:.1f}s")

    # Sobol survey (may use tighter bounds than DE)
    sampler = Sobol(d=D, scramble=True, seed=seed)
    sobol_01 = sampler.random(N_sobol)
    sobol_points = sobol_lo + sobol_01 * (sobol_hi - sobol_lo)
    sobol_jax = jnp.array(sobol_points)

    t0 = time.time()
    logp_all = []
    n_batches = (N_sobol + eval_chunk - 1) // eval_chunk
    for i in trange(n_batches, desc="Sobol", disable=not verbose):
        start = i * eval_chunk
        end = min(start + eval_chunk, N_sobol)
        vals = _logp_batch_raw(sobol_jax[start:end])
        jax.block_until_ready(vals)
        logp_all.append(np.asarray(vals))
    logp_all = np.concatenate(logp_all)
    valid = np.isfinite(logp_all)
    logp_all = np.where(valid, logp_all, -np.inf)

    if verbose:
        fprint(f"Sobol done in {time.time() - t0:.1f}s "
               f"({valid.sum()}/{N_sobol} valid, "
               f"best logP = {logp_all[valid].max():.1f})")

    # Seed initial population from top Sobol points
    selected = _select_distinct(sobol_points, logp_all, pop_size,
                                min_dist_frac)
    x0 = sobol_points[selected]
    x0_normed = jnp.array((x0 - lo) / scale)

    # Compile fitness
    t0 = time.time()
    f0 = fitness_batch(x0_normed)
    jax.block_until_ready(f0)
    if verbose:
        fprint(f"Fitness JIT compiled in {time.time() - t0:.1f}s")

    # DE loop
    if verbose:
        fsection(f"DE (pop={pop_size}, max_gen={max_generations})")

    key = jax.random.PRNGKey(seed)
    de = DifferentialEvolution(
        population_size=pop_size, solution=jnp.zeros(D))
    params = de.default_params
    state = de.init(key, x0_normed, f0, params)

    best_logp_so_far = -float(state.best_fitness)
    gens_without_improvement = 0

    t0 = time.time()
    final_gen = max_generations
    for gen in trange(max_generations, desc="DE", disable=not verbose):
        key, ask_key, tell_key = jax.random.split(key, 3)
        population, state = de.ask(ask_key, state, params)
        population = _reflect_bounds(population)
        fitnesses = fitness_batch(population)
        jax.block_until_ready(fitnesses)
        state, metrics = de.tell(
            tell_key, population, fitnesses, state, params)

        current_best = -float(state.best_fitness)
        if current_best > best_logp_so_far + 0.1:
            best_logp_so_far = current_best
            gens_without_improvement = 0
        else:
            gens_without_improvement += 1

        if verbose and (gen + 1) % log_every == 0:
            x_best = np.asarray(lo + state.best_solution * scale)
            true_logp = float(logp_fn(jnp.array(x_best)))
            fprint(f"  gen {gen+1:5d}: logP = {true_logp:.2f}, "
                   f"stale = {gens_without_improvement}/{patience}")

        if gens_without_improvement >= patience:
            if verbose:
                fprint(f"  Converged at gen {gen+1} (no improvement "
                       f"for {patience} generations)")
            final_gen = gen + 1
            break

    de_time = time.time() - t0
    if verbose:
        fprint(f"DE done in {de_time:.1f}s ({final_gen} generations)")

    # Extract best
    x_best = np.asarray(lo + state.best_solution * scale)
    best_logp = float(logp_fn(jnp.array(x_best)))
    best_params = _flat_to_dict(jnp.array(x_best), names, sizes)
    best_params = {
        k: float(v) if jnp.ndim(v) == 0 else v
        for k, v in best_params.items()
    }

    if verbose:
        fsection("DE results")
        fprint(f"Best logP = {best_logp:.2f}")
        for name in names:
            v = best_params[name]
            if isinstance(v, (float, int)):
                fprint(f"  {name:20s} = {v:.4f}")

    all_results = {
        "names": names,
        "sizes": sizes,
        "lo_sobol": lo,
        "hi_sobol": hi,
    }

    return best_params, best_logp, all_results


# -----------------------------------------------------------------------
# Integration with run_H0_inference
# -----------------------------------------------------------------------


def _use_de(model):
    """Check if DE should be used instead of Sobol+Adam.

    DE is preferred for maser disk models with r+phi marginalization,
    where the likelihood is expensive and the posterior is narrow in
    high dimensions.
    """
    return getattr(model, "marginalise_r", False)


def find_MAP(model, model_kwargs=None, seed=42):
    """Find MAP estimate, automatically selecting the optimizer.

    Uses DE for maser disk models with r+phi marginalization
    (derivative-free, handles the narrow 14D posterior well).
    Uses Sobol+Adam for everything else.

    Drop-in replacement for ``find_initial_point``. Reads optimizer
    settings from ``model.config["optimise"]`` (optional).

    Returns
    -------
    best_params : dict
        Constrained-space MAP parameter values, ready for
        ``init_to_value(values=best_params)``.
    """
    if model_kwargs is None:
        model_kwargs = {}

    opt_cfg = model.config.get("optimise", {})

    if _use_de(model):
        # Compute Hubble-flow distance estimate for tighter Sobol seeding
        sobol_bounds_override = None
        v_sys = getattr(model, "v_sys_obs", None)
        if v_sys is not None:
            SPEED_OF_LIGHT = 299792.458
            H0_est = opt_cfg.get("H0_for_D_estimate", 70.0)
            sigma_v = opt_cfg.get("D_sobol_sigma_v", 500.0)
            n_sigma = opt_cfg.get("D_sobol_n_sigma", 5)
            z_est = v_sys / SPEED_OF_LIGHT
            q0 = -0.55
            D_est = (SPEED_OF_LIGHT * z_est / H0_est
                     * (1 + 0.5 * (1 - q0) * z_est))
            sigma_D = sigma_v / H0_est
            D_lo_sobol = D_est - n_sigma * sigma_D
            D_hi_sobol = D_est + n_sigma * sigma_D
            sobol_bounds_override = {"D_c": (D_lo_sobol, D_hi_sobol)}

        kwargs = dict(
            model_kwargs=model_kwargs,
            log2_N=opt_cfg.get("log2_N", 16),
            pop_size=opt_cfg.get("pop_size", 1000),
            max_generations=opt_cfg.get("max_generations", 1000),
            patience=opt_cfg.get("patience", 100),
            eval_chunk=opt_cfg.get("eval_chunk", 64),
            sobol_n_sigma=opt_cfg.get("sobol_n_sigma", 5),
            min_dist_frac=opt_cfg.get("min_dist_frac", 0.005),
            sobol_bounds_override=sobol_bounds_override,
            seed=seed,
        )
        best_params, best_logp, results = de_optimize(model, **kwargs)
    else:
        kwargs = dict(
            model_kwargs=model_kwargs,
            log2_N=opt_cfg.get("log2_N", 14),
            M=opt_cfg.get("M", 10),
            n_steps=opt_cfg.get("n_steps", 5000),
            lr=opt_cfg.get("lr", 0.1),
            lr_end=opt_cfg.get("lr_end", 0.005),
            n_restarts=opt_cfg.get("n_restarts", 3),
            sobol_n_sigma=opt_cfg.get("sobol_n_sigma", 1),
            sobol_batch=opt_cfg.get("sobol_batch", 128),
            min_dist_frac=opt_cfg.get("min_dist_frac", 0.01),
            seed=seed,
        )
        best_params, best_logp, results = sobol_optimize(model, **kwargs)

    # Convert scalar floats to jnp arrays (matching find_initial_point)
    return {k: jnp.array(v) for k, v in best_params.items()}
