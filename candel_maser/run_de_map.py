# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Run profiled MAP optimisation for one megamaser disk.

The objective uses the same global-parameter target as the BlackJAX Gibbs
sampler.  For each global proposal, ``r_ang`` is set to its conditional
phi-marginal MAP before evaluating the constrained-space log posterior.
"""
import argparse
import os
import sys
import tempfile
import time

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

_ENABLE_F64 = "--f64" in sys.argv

import jax

if _ENABLE_F64 and not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)
    print("float64 enabled (--f64)", flush=True)

import jax.numpy as jnp
import numpy as np
import tomli_w
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from tqdm import tqdm, trange

from candel.inference.optimise import (
    _prior_bounds,
    _reflect_bounds,
    _select_distinct,
)
from candel.model.maser_blackjax import (
    MaserBlackJaxTarget,
    init_from_prior_median,
)
from candel.model.model_H0_maser import MaserDiskModel
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import data_path, fprint, fsection, get_nested, results_path


def _h_ref(model):
    return float(get_nested(model.config, "model/H0_ref", 73.0)) / 100.0


def _D_A_from_D_c(model, D_c):
    h = _h_ref(model)
    z_cosmo = model.distance2redshift(jnp.atleast_1d(D_c), h=h).squeeze()
    return D_c / (1.0 + z_cosmo)


def _downsample_spots(data, n_sys, n_red, n_blue):
    n = data["n_spots"]
    is_hv = np.asarray(data["is_highvel"])
    is_blue = np.asarray(data["is_blue"])
    has_a = np.asarray(data["accel_measured"])
    velocity = np.asarray(data["velocity"])

    def pick(idx, count):
        if idx.size <= count:
            return idx
        ordered = idx[np.argsort(velocity[idx])]
        keep = np.round(np.linspace(0, ordered.size - 1, count)).astype(int)
        return np.sort(ordered[keep])

    idx_sys = np.where(~is_hv)[0]
    idx_sys_a = idx_sys[has_a[idx_sys]]
    sel_sys = pick(idx_sys_a if idx_sys_a.size >= n_sys else idx_sys, n_sys)
    sel_red = pick(np.where(is_hv & ~is_blue)[0], n_red)
    sel_blue = pick(np.where(is_hv & is_blue)[0], n_blue)
    selected = np.sort(np.concatenate([sel_sys, sel_red, sel_blue]))

    out = {}
    for key, value in data.items():
        if (isinstance(value, np.ndarray) and value.ndim >= 1
                and value.shape[0] == n):
            out[key] = value[selected]
        else:
            out[key] = value
    out["n_spots"] = int(selected.size)
    fprint(f"downsampled {n} -> {selected.size} spots "
           f"(sys: {sel_sys.size}, red: {sel_red.size}, "
           f"blue: {sel_blue.size})")
    return out


def _clean_init(model, init_cfg):
    init_params = {key: jnp.asarray(value) for key, value in init_cfg.items()}
    init_params.pop("M_BH", None)
    if "D_c" not in init_params:
        raise KeyError("DE init requires D_c.")
    mass_param = getattr(model, "mass_parameterization", "eta")
    if mass_param == "eta":
        if "eta" not in init_params:
            if "log_MBH" not in init_params:
                raise KeyError(
                    "eta mass parameterization requires initial 'eta' or "
                    "'log_MBH'.")
            D_A = _D_A_from_D_c(model, init_params["D_c"])
            init_params["eta"] = init_params["log_MBH"] - jnp.log10(D_A)
    else:
        if "log_MBH" not in init_params:
            if "eta" not in init_params:
                raise KeyError("DE init requires log_MBH.")
            D_A = _D_A_from_D_c(model, init_params["D_c"])
            init_params["log_MBH"] = init_params["eta"] + jnp.log10(D_A)
        init_params.pop("eta", None)
    if not model.use_ecc:
        for key in ("e_x", "e_y", "ecc", "periapsis", "periapsis_rad",
                    "dperiapsis_dr"):
            init_params.pop(key, None)
    if not model.use_quadratic_warp:
        for key in ("d2i_dr2", "d2Omega_dr2"):
            init_params.pop(key, None)
    else:
        init_params.setdefault("d2i_dr2", jnp.asarray(0.0))
        init_params.setdefault("d2Omega_dr2", jnp.asarray(0.0))
    return init_params


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


def _layout(target, sobol_n_sigma):
    names, lo, hi = [], [], []
    for site, _, prior in target.sites:
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


def _theta_to_output(theta, r_ang):
    out = dict(theta)
    out["D_c"] = theta["D_c"]
    out["r_ang"] = r_ang
    return out


def _make_logp(target, names):
    model = target.model
    h = target.h

    def logp_constrained(x):
        theta = _flat_to_theta(x, names)
        phys_args, phys_kw = model.phys_from_params_jax(theta, h)
        # Deterministic inner solve: phi is marginalised numerically and
        # each spot's 1D r_ang objective is maximised with fixed-step
        # bracketing/refinement controlled by n_r_global/n_refine_steps.
        r_ang = jax.lax.stop_gradient(
            model.conditional_r_ang_map(phys_args, phys_kw))
        return target.constrained_logdensity_r_ang(
            theta, r_ang, phys_args=phys_args, phys_kw=phys_kw)

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


def _make_lbfgs_objective(logp):
    value_and_grad = jax.jit(jax.value_and_grad(logp))
    state = {"nfev": 0}

    def objective(x):
        state["nfev"] += 1
        value, grad = value_and_grad(jnp.asarray(x))
        value = float(jax.device_get(value))
        grad = np.asarray(jax.device_get(grad), dtype=float)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            state["logp"] = -np.inf
            state["grad_inf"] = np.inf
            return 1e100, np.zeros_like(np.asarray(x, dtype=float))
        state["logp"] = value
        state["grad_inf"] = float(np.max(np.abs(grad)))
        state["x"] = np.asarray(x, dtype=float)
        return -value, -grad

    return objective, state


def _clip_to_open_bounds(x, lo, hi):
    width = hi - lo
    eps = np.maximum(1e-10 * np.maximum(1.0, width), 0.0)
    return np.minimum(np.maximum(x, lo + eps), hi - eps)


def _make_lbfgs_starts(x0, lo, hi, n_starts, strategy,
                       jitter_scale, seed, logp=None,
                       sobol_candidates=0, eval_chunk=1,
                       min_dist_frac=0.005):
    starts = [x0]
    n_extra = int(n_starts) - 1
    if n_extra <= 0:
        return starts

    scale = hi - lo
    strategy = str(strategy).lower()
    if strategy == "jitter":
        if jitter_scale <= 0:
            raise ValueError(
                "lbfgs_start_strategy='jitter' requires "
                "lbfgs_jitter_scale > 0.")
        rng = np.random.default_rng(seed)
        for _ in range(n_extra):
            starts.append(
                _clip_to_open_bounds(
                    x0 + rng.normal(size=x0.size) * jitter_scale * scale,
                    lo, hi))
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        points = rng.uniform(size=(n_extra, x0.size))
        starts.extend(_clip_to_open_bounds(lo + points * scale, lo, hi))
    elif strategy == "sobol":
        n_candidates = max(n_extra, int(sobol_candidates))
        m = int(np.ceil(np.log2(n_candidates)))
        points = Sobol(d=x0.size, scramble=True, seed=seed).random_base2(m)
        candidates = _clip_to_open_bounds(
            lo + points[:2 ** m] * scale, lo, hi)
        if logp is not None and n_candidates > n_extra:
            fsection("Sobol L-BFGS start screen")
            fprint(f"{candidates.shape[0]} candidates -> {n_extra} starts")
            logp_batch = jax.jit(jax.vmap(logp))
            logp_vals = np.asarray(_eval_chunks(
                logp_batch, jnp.asarray(candidates),
                max(1, int(eval_chunk)), desc="Sobol starts"))
            valid = np.isfinite(logp_vals)
            scores = np.where(valid, logp_vals, -np.inf)
            selected = _select_distinct(
                candidates, scores, n_extra, min_dist_frac)
            best = scores[selected[0]]
            fprint(f"Sobol screen: {valid.sum()}/{scores.size} finite, "
                   f"best logP={best:.2f}")
            candidates = candidates[selected]
        else:
            candidates = candidates[:n_extra]
        starts.extend(candidates)
    else:
        raise ValueError(
            "lbfgs_start_strategy must be 'sobol', 'random', or 'jitter'.")
    return starts


def _run_lbfgs(target, opt_cfg, init_params, seed):
    maxiter = int(opt_cfg.get("lbfgs_maxiter", 500))
    ftol = float(opt_cfg.get("lbfgs_ftol", 1e-6))
    gtol = float(opt_cfg.get("lbfgs_gtol", 1e-5))
    maxls = int(opt_cfg.get("lbfgs_maxls", 50))
    n_starts = max(1, int(opt_cfg.get("lbfgs_n_starts", 1)))
    start_strategy = str(opt_cfg.get("lbfgs_start_strategy", "sobol"))
    sobol_candidates = int(opt_cfg.get("lbfgs_sobol_candidates", 0))
    jitter_scale = float(opt_cfg.get("lbfgs_jitter_scale", 0.0))
    sobol_n_sigma = opt_cfg.get("sobol_n_sigma", 5)
    eval_chunk = int(opt_cfg.get("eval_chunk", 5))
    min_dist_frac = float(opt_cfg.get("min_dist_frac", 0.005))

    names, sizes, lo, hi = _layout(target, sobol_n_sigma)
    del sizes
    scale = hi - lo
    D = len(names)
    x0 = np.asarray([float(np.asarray(init_params[name]))
                     for name in names], dtype=float)
    x0 = _clip_to_open_bounds(x0, lo, hi)

    fsection("Profile L-BFGS MAP optimizer")
    fprint(f"{D}D, maxiter={maxiter}, n_starts={n_starts}, "
           f"start_strategy={start_strategy}, ftol={ftol:g}, "
           f"gtol={gtol:g}, maxls={maxls}")
    for name, lower, upper in zip(names, lo, hi):
        fprint(f"  {name:20s}: [{lower:.4g}, {upper:.4g}]")

    logp = _make_logp(target, names)
    objective, objective_state = _make_lbfgs_objective(logp)
    t0 = time.time()
    f0, g0 = objective(x0)
    fprint(f"JIT compiled in {time.time() - t0:.1f}s; "
           f"initial logP={-f0:.2f}, |grad|_inf={np.max(np.abs(g0)):.3g}")

    starts = _make_lbfgs_starts(
        x0, lo, hi, n_starts, start_strategy, jitter_scale, seed,
        logp=logp, sobol_candidates=sobol_candidates,
        eval_chunk=eval_chunk, min_dist_frac=min_dist_frac)
    bounds = list(zip(lo, hi))
    best_res = None
    best_fun = np.inf
    d_c_idx = names.index("D_c")
    for i, start in enumerate(starts):
        iter_state = {"nit": 0}
        desc = "L-BFGS" if n_starts == 1 else f"L-BFGS {i + 1}/{n_starts}"
        nfev_start = int(objective_state.get("nfev", 0))

        def callback(xk):
            iter_state["nit"] += 1
            progress.update(max(0, min(iter_state["nit"], maxiter)
                                - progress.n))
            logp = objective_state.get("logp", -np.inf)
            grad_inf = objective_state.get("grad_inf", np.inf)
            progress.set_postfix({
                "logP": f"{logp:.2f}",
                "D_c": f"{float(np.asarray(xk)[d_c_idx]):.2f}",
                "|g|inf": f"{grad_inf:.2g}",
                "nfev": int(objective_state.get("nfev", 0)) - nfev_start,
            })

        with tqdm(total=maxiter, desc=desc) as progress:
            res = minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                callback=callback,
                options={
                    "maxiter": maxiter,
                    "ftol": ftol,
                    "gtol": gtol,
                    "maxls": maxls,
                },
            )
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_res = res
        fprint(f"  start {i + 1}/{n_starts}: logP={-float(res.fun):.2f}, "
               f"nit={res.nit}, success={res.success}, "
               f"status={res.status}")

    if best_res is None:
        raise RuntimeError("L-BFGS did not return an optimisation result.")

    x_best = np.asarray(best_res.x, dtype=float)
    theta = target.complete_params(_flat_to_theta(jnp.asarray(x_best), names))
    phys_args, phys_kw = target.model.phys_from_params_jax(theta, target.h)
    r_ang = target.model.conditional_r_ang_map(phys_args, phys_kw)
    best_logp = -float(best_res.fun)
    return (
        _theta_to_output(
            {k: np.asarray(jax.device_get(v)) for k, v in theta.items()},
            np.asarray(jax.device_get(r_ang))),
        best_logp,
        best_res,
    )


def _save_de_checkpoint(path, population, fitness, best_solution,
                        best_fitness, generation, key,
                        gens_without_improvement, best_logp_so_far,
                        lo, hi, names, sizes):
    tmp = path + ".tmp.npz"
    np.savez(
        tmp,
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
    os.replace(tmp, path)


def _load_de_checkpoint(path, lo, hi, names, sizes):
    d = np.load(path)
    if not np.allclose(d["lo"], lo) or not np.allclose(d["hi"], hi):
        raise ValueError("Checkpoint bounds do not match current model.")
    if list(d["names"]) != list(names) or list(d["sizes"]) != list(sizes):
        raise ValueError("Checkpoint parameter layout does not match.")
    return d


def _make_de_initial_population(fitness_batch, lo, hi, pop_size, seed,
                                N_sobol, eval_chunk, min_dist_frac):
    scale = hi - lo
    D = lo.size

    sampler = Sobol(d=D, scramble=True, seed=seed)
    sobol_01 = sampler.random(N_sobol)
    sobol_points = lo + sobol_01 * scale
    sobol_normed = jnp.asarray((sobol_points - lo) / scale)

    t0 = time.time()
    logp_all = -np.asarray(
        _eval_chunks(fitness_batch, sobol_normed, eval_chunk,
                     desc="Sobol"))
    valid = np.isfinite(logp_all)
    logp_all = np.where(valid, logp_all, -np.inf)
    best_sobol = logp_all[valid].max() if np.any(valid) else -np.inf
    fprint(f"Sobol done in {time.time() - t0:.1f}s "
           f"({valid.sum()}/{N_sobol} valid, "
           f"best logP={best_sobol:.1f})")

    selected = _select_distinct(
        sobol_points, logp_all, pop_size, min_dist_frac)
    population = np.asarray((sobol_points[selected] - lo) / scale)
    fitness = np.asarray(_eval_chunks(
        fitness_batch, jnp.asarray(population), eval_chunk))
    jax.block_until_ready(fitness)
    if population.shape[0] != pop_size:
        raise RuntimeError(
            f"DE initial population has {population.shape[0]} members, "
            f"expected {pop_size}.")
    fprint(f"Initial population: {population.shape[0]} Sobol members")
    return jnp.asarray(population), jnp.asarray(fitness)


def _run_de(target, opt_cfg, seed, checkpoint_path=None,
            resume_path=None, checkpoint_interval=900.0):
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

    names, sizes, lo, hi = _layout(target, sobol_n_sigma)
    scale = hi - lo
    D = len(names)
    d_c_idx = names.index("D_c")
    N_sobol = 2 ** log2_N

    fsection("DE MAP optimizer")
    fprint(f"{D}D, pop={pop_size}, max_gen={max_generations}, "
           f"patience={patience}, eval_chunk={eval_chunk}")
    fprint("initial population: scrambled Sobol screen only")
    for name, lower, upper in zip(names, lo, hi):
        fprint(f"  {name:20s}: [{lower:.4g}, {upper:.4g}]")

    logp = _make_logp(target, names)

    def fitness_one(x_normed):
        x = jnp.asarray(lo) + x_normed * jnp.asarray(scale)
        return -logp(x)

    fitness_batch = jax.jit(jax.vmap(fitness_one))

    t0 = time.time()
    _ = fitness_batch(jnp.full((eval_chunk, D), 0.5))
    jax.block_until_ready(_)
    fprint(f"JIT compiled in {time.time() - t0:.1f}s")

    if resume_path is not None:
        ckpt = _load_de_checkpoint(resume_path, lo, hi, names, sizes)
        key = jnp.asarray(ckpt["key"])
        gen_start = int(ckpt["generation_counter"])
        gens_without_improvement = int(ckpt["gens_without_improvement"])
        best_logp_so_far = float(ckpt["best_logp_so_far"])
        population = jnp.asarray(ckpt["population"])
        fitness = jnp.asarray(ckpt["fitness"])
        best_solution = jnp.asarray(ckpt["best_solution"])
        best_fitness = jnp.asarray(ckpt["best_fitness"])
        fprint(f"Resumed from {resume_path} at generation {gen_start}")
    else:
        population, fitness = _make_de_initial_population(
            fitness_batch, lo, hi, pop_size, seed, N_sobol,
            eval_chunk, min_dist_frac)
        key = jax.random.PRNGKey(seed)
        best_idx = int(np.argmin(np.asarray(fitness)))
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        gen_start = 0
        best_logp_so_far = -float(best_fitness)
        gens_without_improvement = 0

    fsection(f"DE (pop={pop_size}, max_gen={max_generations}, "
             f"start={gen_start})")
    last_ckpt = time.time()
    final_gen = max_generations
    de_progress = trange(max_generations - gen_start, desc="DE")
    for step in de_progress:
        gen = gen_start + step
        key, k1, k2, k3, k_cross, k_force = jax.random.split(key, 6)
        idx1 = jax.random.permutation(k1, pop_size)
        idx2 = jax.random.permutation(k2, pop_size)
        idx3 = jax.random.permutation(k3, pop_size)
        mutant = population[idx1] + mutation * (
            population[idx2] - population[idx3])
        mutant = _reflect_bounds(mutant)
        cross = jax.random.uniform(k_cross, (pop_size, D)) < crossover
        forced = jax.random.randint(k_force, (pop_size,), 0, D)
        cross = cross.at[jnp.arange(pop_size), forced].set(True)
        trials = jnp.where(cross, mutant, population)
        trial_fitness = _eval_chunks(fitness_batch, trials, eval_chunk)
        jax.block_until_ready(trial_fitness)
        improved = trial_fitness <= fitness
        population = jnp.where(improved[:, None], trials, population)
        fitness = jnp.where(improved, trial_fitness, fitness)

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
            best_d_c_norm = float(best_solution[d_c_idx])
            best_d_c = float(lo[d_c_idx] + best_d_c_norm * scale[d_c_idx])
            de_progress.set_postfix_str(
                f"logP={current_best:.2f}, D_c={best_d_c:.2f}, "
                f"stale={gens_without_improvement}/{patience}")

        if (checkpoint_path is not None
                and time.time() - last_ckpt >= checkpoint_interval):
            _save_de_checkpoint(
                checkpoint_path, population, fitness, best_solution,
                best_fitness, gen + 1, key, gens_without_improvement,
                best_logp_so_far, lo, hi, names, sizes)
            last_ckpt = time.time()
            fprint(f"  checkpoint: gen {gen + 1}")

        if gens_without_improvement >= patience:
            final_gen = gen + 1
            fprint(f"  converged at gen {final_gen}")
            break

    x_best = np.asarray(lo + best_solution * scale)
    theta = target.complete_params(_flat_to_theta(jnp.asarray(x_best), names))
    phys_args, phys_kw = target.model.phys_from_params_jax(theta, target.h)
    r_ang = target.model.conditional_r_ang_map(phys_args, phys_kw)
    best_logp = -float(best_fitness)
    return _theta_to_output(
        {k: np.asarray(jax.device_get(v)) for k, v in theta.items()},
        np.asarray(jax.device_get(r_ang))), best_logp, final_gen


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run profiled MAP optimisation for one megamaser disk.")
    parser.add_argument("galaxy", type=str)
    parser.add_argument("--optimizer", choices=("de", "lbfgs"),
                        default="de",
                        help="Profiled MAP optimizer to run. Default: de.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
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
    parser.add_argument("--spot-batch", type=int, default=None)
    parser.add_argument("--n-sys", type=int, default=None)
    parser.add_argument("--n-red", type=int, default=None)
    parser.add_argument("--n-blue", type=int, default=None)
    parser.add_argument("--log2-N", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval-chunk", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--lbfgs-maxiter", type=int, default=None)
    parser.add_argument("--lbfgs-ftol", type=float, default=None)
    parser.add_argument("--lbfgs-gtol", type=float, default=None)
    parser.add_argument("--lbfgs-maxls", type=int, default=None)
    parser.add_argument("--lbfgs-n-starts", type=int, default=None)
    parser.add_argument("--lbfgs-sobol-candidates", type=int, default=None)
    parser.add_argument("--lbfgs-start-strategy",
                        choices=("sobol", "random", "jitter"),
                        default=None)
    parser.add_argument("--lbfgs-jitter-scale", type=float, default=None)
    args = parser.parse_args(argv)

    if args.f64 and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
        print("float64 enabled (--f64)", flush=True)

    with open(os.path.join(os.path.dirname(__file__), "config_maser.toml"), "rb") as f:
        master_cfg = tomli.load(f)

    galaxies = master_cfg["model"]["galaxies"]
    if args.galaxy not in galaxies:
        raise SystemExit(
            f"Unknown galaxy {args.galaxy!r}. Available: {list(galaxies)}")
    gcfg = galaxies[args.galaxy]
    seed = args.seed if args.seed is not None else master_cfg["inference"].get(
        "seed", 42)

    _devs = jax.devices()
    _dev_names = ", ".join(d.device_kind for d in _devs)
    _precision = "float64" if jax.config.jax_enable_x64 else "float32"
    fprint(f"JAX platform: {jax.default_backend()}, devices: {_devs} "
           f"({_dev_names}), precision: {_precision}")

    fsection(f"Loading {args.galaxy} data")
    data = load_megamaser_spots(
        data_path("data", "Megamaser"), args.galaxy,
        v_sys_obs=gcfg["v_sys_obs"])
    if "D_lo" in gcfg and "D_hi" in gcfg:
        data["D_lo"] = float(gcfg["D_lo"])
        data["D_hi"] = float(gcfg["D_hi"])
    downsample_counts = (args.n_sys, args.n_red, args.n_blue)
    if any(count is not None for count in downsample_counts):
        if any(count is None for count in downsample_counts):
            raise SystemExit(
                "--n-sys, --n-red, and --n-blue must be provided together.")
        data = _downsample_spots(data, args.n_sys, args.n_red, args.n_blue)

    opt_cfg = dict(master_cfg.get("optimise", {}))
    for arg_name, cfg_name in (
            ("log2_N", "log2_N"),
            ("pop_size", "pop_size"),
            ("max_generations", "max_generations"),
            ("patience", "patience"),
            ("eval_chunk", "eval_chunk"),
            ("log_every", "log_every"),
            ("lbfgs_maxiter", "lbfgs_maxiter"),
            ("lbfgs_maxls", "lbfgs_maxls"),
            ("lbfgs_n_starts", "lbfgs_n_starts"),
            ("lbfgs_sobol_candidates", "lbfgs_sobol_candidates")):
        value = getattr(args, arg_name)
        if value is not None:
            opt_cfg[cfg_name] = int(value)
    for arg_name, cfg_name in (
            ("lbfgs_ftol", "lbfgs_ftol"),
            ("lbfgs_gtol", "lbfgs_gtol"),
            ("lbfgs_jitter_scale", "lbfgs_jitter_scale")):
        value = getattr(args, arg_name)
        if value is not None:
            opt_cfg[cfg_name] = float(value)
    if args.lbfgs_start_strategy is not None:
        opt_cfg["lbfgs_start_strategy"] = args.lbfgs_start_strategy
    config = {
        "inference": master_cfg["inference"],
        "model": dict(master_cfg["model"]),
        "io": master_cfg["io"],
        "optimise": opt_cfg,
    }
    config["model"]["galaxies"] = {
        g: dict(blk) for g, blk in master_cfg["model"]["galaxies"].items()}
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

    inf_cfg = master_cfg.get("inference", {})
    init_params = _make_init(
        model, config["model"]["galaxies"][args.galaxy].get("init", {}),
        inf_cfg.get("init_strategy", "median"),
        int(inf_cfg.get("init_num_samples", 100)),
        jax.random.PRNGKey(seed))
    h = float(get_nested(model.config, "model/H0_ref", 73.0)) / 100.0
    target = MaserBlackJaxTarget(
        model, h, init_params, spot_batch=args.spot_batch)
    fprint("profiled inner r_ang solve: phi marginalised, deterministic "
           f"1D optimiser with n_r_global={model._n_r_global}, "
           f"n_refine_steps={model._n_refine_steps}")

    optimizer_name = "DE" if args.optimizer == "de" else "profile L-BFGS"
    fsection(
        f"{optimizer_name} MAP optimisation "
        f"({args.galaxy}, {data['n_spots']} spots)")
    t0 = time.time()
    if args.optimizer == "de":
        ckpt_dir = results_path(
            master_cfg["io"].get("root_output", "results/Megamaser"),
            "de_checkpoints", args.galaxy)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "de_ckpt_rmap.npz")
        resume_path = (
            ckpt_path if args.resume and os.path.isfile(ckpt_path)
            else None)
        if args.resume and resume_path is None:
            fprint(
                f"--resume: no checkpoint found at {ckpt_path}, "
                "starting fresh")
        init_params, best_logp, run_info = _run_de(
            target, opt_cfg, seed, checkpoint_path=ckpt_path,
            resume_path=resume_path,
            checkpoint_interval=args.checkpoint_interval_minutes * 60.0)
        run_summary = f"generations = {run_info}"
    else:
        if args.resume:
            fprint("--resume is only used by DE checkpoints; ignored.")
        init_params, best_logp, run_info = _run_lbfgs(
            target, opt_cfg, init_params, seed)
        run_summary = (
            f"iterations = {run_info.nit}; success = {run_info.success}; "
            f"status = {run_info.status}")
    dt = time.time() - t0

    fsection(f"MAP results ({args.galaxy}, {dt:.0f}s)")
    fprint(f"best logP = {best_logp:.2f}; {run_summary}")
    for key, value in sorted(init_params.items()):
        value = np.asarray(value)
        if value.ndim == 0:
            fprint(f"  {key:20s} = {float(value):12.4f}")
        else:
            fprint(f"  {key:20s} = [{value.size} values]")

    lines = [f"\n[model.galaxies.{args.galaxy}.init]"]
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
