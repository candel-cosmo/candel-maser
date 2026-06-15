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
"""Unified megamaser runner.

The default sampler is the BlackJAX collapsed-Gibbs chain.  Profiled MAP
optimisers are available as ``--sampler de`` and ``--sampler lbfgs`` through
the same entry point.
"""
import argparse
import copy
import importlib.util
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

if _ENABLE_F64:
    jax.config.update("jax_enable_x64", True)
    print("float64 enabled (--f64)", flush=True)

import jax.numpy as jnp
import numpy as np
import tomli_w
from h5py import File as H5File
from jax import random

from candel.model.maser_blackjax import init_from_prior_median, run_blackjax_mwg
from candel.model.model_H0_maser import MaserDiskModel
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import data_path, fprint, fsection, results_path

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_maser.toml")
with open(_CONFIG_PATH, "rb") as f:
    master_cfg = tomli.load(f)


def _h_ref(model):
    return float(model.config["model"].get("H0_ref", 73.0)) / 100.0


def _D_A_from_D_c(model, D_c):
    h = _h_ref(model)
    z_cosmo = model.distance2redshift(jnp.atleast_1d(D_c), h=h).squeeze()
    return D_c / (1.0 + z_cosmo)


def _select_sampler(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sampler", choices=("gibbs", "de", "lbfgs"),
                        default="gibbs")
    args, _ = parser.parse_known_args(argv)
    if args.sampler == "gibbs":
        return args.sampler, argv

    remaining = []
    skip_next = False
    for value in argv:
        if skip_next:
            skip_next = False
            continue
        if value == "--sampler":
            skip_next = True
            continue
        if value.startswith("--sampler="):
            continue
        remaining.append(value)
    return args.sampler, remaining


def _run_profile_map(argv, optimizer):
    script_dir = os.path.dirname(__file__)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from run_de_map import main as run_profile_map_main
    if optimizer != "de":
        argv = ["--optimizer", optimizer] + list(argv)
    return run_profile_map_main(argv)


def _downsample_spots(data, n_sys, n_red, n_blue):
    """Stratified downsample preserving the systemic/red/blue mix."""
    n = data["n_spots"]
    is_hv = np.asarray(data["is_highvel"])
    is_blue = np.asarray(data["is_blue"])
    has_a = np.asarray(data["accel_measured"])
    velocity = np.asarray(data["velocity"])

    def pick(idx, count):
        if idx.size <= count:
            return idx
        ordered = idx[np.argsort(velocity[idx])]
        keep = np.round(
            np.linspace(0, ordered.size - 1, count)).astype(int)
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
    init_params.pop("H0", None)
    init_params.pop("sigma_pec", None)
    init_params.pop("M_BH", None)
    if "D_c" not in init_params:
        raise KeyError(
            "BlackJAX megamaser initial values must include 'D_c'.")
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
                raise KeyError(
                    "BlackJAX megamaser initial values must include "
                    "'log_MBH'.")
            D_A = _D_A_from_D_c(model, init_params["D_c"])
            init_params["log_MBH"] = init_params["eta"] + jnp.log10(D_A)
        init_params.pop("eta", None)
    if not model.use_ecc:
        for key in ("e_x", "e_y", "ecc", "periapsis", "periapsis_rad",
                    "dperiapsis_dr"):
            init_params.pop(key, None)
    elif model.ecc_cartesian:
        for key in ("ecc", "periapsis", "periapsis_rad"):
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
    raise ValueError(
        "Megamaser init_strategy must be 'median' or 'config'.")


def _print_init(init_params):
    fsection("Initial values")
    for key, value in sorted(init_params.items()):
        value = jnp.asarray(value)
        if value.ndim == 0:
            fprint(f"{key:20s} = {float(value):12.4f}")
        else:
            fprint(f"{key:20s} = [{value.size} values]")


def _ess_1d(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return np.nan
    y = x - x.mean()
    var = np.dot(y, y) / n
    if not np.isfinite(var) or var <= 0:
        return float(n)
    acov = np.correlate(y, y, mode="full")[n - 1:] / n
    rho = acov / acov[0]
    tau = 1.0
    for value in rho[1:]:
        if not np.isfinite(value) or value <= 0:
            break
        tau += 2.0 * float(value)
    return float(max(1.0, min(n, n / tau)))


def _split_rhat_1d(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return np.nan
    half = n // 2
    chains = np.stack([x[:half], x[-half:]], axis=0)
    chain_means = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean()
    if not np.isfinite(W) or W <= 0:
        return 1.0
    B = half * chain_means.var(ddof=1)
    var_hat = ((half - 1) / half) * W + B / half
    return float(np.sqrt(max(var_hat / W, 0.0)))


def _ess_rhat(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return _ess_1d(arr), _split_rhat_1d(arr)
    neff = np.array([_ess_1d(arr[:, i]) for i in range(arr.shape[1])])
    rhat = np.array([_split_rhat_1d(arr[:, i]) for i in range(arr.shape[1])])
    return neff, rhat


def _print_global_summary(samples):
    global_keys = (
        "D_c", "eta", "log_MBH",
        "i0", "di_dr", "Omega0", "dOmega_dr",
        "x0", "y0", "dv_sys", "sigma_x_floor", "sigma_y_floor",
        "sigma_v_sys", "sigma_v_hv", "sigma_a_floor",
    )
    printable = {
        key: np.asarray(samples[key])[None, ...]
        for key in global_keys
        if key in samples and np.asarray(samples[key]).ndim == 1
    }
    if not printable:
        return
    n = next(iter(printable.values())).shape[1]
    if n < 20:
        fprint(f"Global summary skipped: only {n} posterior samples "
               "(need >=20 for diagnostics).")
        return
    fsection("Global Summary")
    header = (
        f"{'':>17s} {'mean':>9s} {'std':>9s} {'median':>9s} "
        f"{'5.0%':>9s} {'95.0%':>9s} {'n_eff':>9s} {'r_hat':>9s}")
    print(header, flush=True)
    for key in global_keys:
        if key not in printable:
            continue
        arr = np.asarray(printable[key]).reshape(-1)
        q05, q50, q95 = np.quantile(arr, [0.05, 0.50, 0.95])
        neff, rhat = _ess_rhat(arr)
        print(
            f"{key:>17s} {arr.mean():9.2f} {arr.std(ddof=1):9.2f} "
            f"{q50:9.2f} {q05:9.2f} {q95:9.2f} "
            f"{neff:9.2f} {rhat:9.2f}",
            flush=True)


def _print_r_ang_summary(samples):
    if "r_ang" not in samples:
        return
    r = np.asarray(samples["r_ang"], dtype=float)
    if r.ndim != 2 or r.shape[0] < 2:
        return
    if r.shape[0] < 20:
        fprint(f"r_ang summary skipped: only {r.shape[0]} posterior samples "
               "(need >=20 for diagnostics).")
        return

    neff, rhat = _ess_rhat(r)
    mean = np.mean(r, axis=0)
    sd = np.std(r, axis=0)
    q05, q50, q95 = np.quantile(r, [0.05, 0.50, 0.95], axis=0)
    move = np.mean(np.abs(np.diff(r, axis=0)) > 0, axis=0)

    fsection("r_ang Summary")
    print(f"  n_spots              = {r.shape[1]}", flush=True)
    print(f"  mean range           = {mean.min():.5f} .. {mean.max():.5f} mas",
          flush=True)
    print(f"  sd range             = {sd.min():.5f} .. {sd.max():.5f} mas",
          flush=True)
    print(f"  q05 range            = {q05.min():.5f} .. {q05.max():.5f} mas",
          flush=True)
    print(f"  median range         = {q50.min():.5f} .. {q50.max():.5f} mas",
          flush=True)
    print(f"  q95 range            = {q95.min():.5f} .. {q95.max():.5f} mas",
          flush=True)
    print(f"  n_eff min/median/max = {np.nanmin(neff):.1f} / "
          f"{np.nanmedian(neff):.1f} / {np.nanmax(neff):.1f}", flush=True)
    print(f"  r_hat max            = {np.nanmax(rhat):.3f}", flush=True)
    print(f"  move frac min/median/max = {move.min():.3f} / "
          f"{np.median(move):.3f} / {move.max():.3f}", flush=True)


def _print_summary(samples, info, result):
    fsection("Results")
    for key in ("D_c", "log_MBH",
                "eta",
                "i0", "di_dr", "Omega0", "dOmega_dr",
                "x0", "y0", "dv_sys", "sigma_x_floor", "sigma_y_floor",
                "sigma_v_sys", "sigma_v_hv", "sigma_a_floor"):
        if key not in samples:
            continue
        arr = np.asarray(samples[key])
        print(f"  {key:20s} = {arr.mean():10.3f} +/- {arr.std():8.3f}",
              flush=True)

    n_div = int(np.asarray(info.get("nuts_is_divergent", [])).sum())
    acc = np.asarray(info.get("nuts_acceptance_rate", []))
    racc = np.asarray(info.get("r_accept_mean", []))
    print(f"\nWall time: {result.runtime_seconds:.1f}s", flush=True)
    print(f"NUTS divergences: {n_div}", flush=True)
    if acc.size:
        print(f"NUTS acceptance: mean={acc.mean():.3f}", flush=True)
    if racc.size:
        print(f"r proposal acceptance: mean={racc.mean():.3f}",
              flush=True)
    _print_global_summary(samples)
    _print_r_ang_summary(samples)


def _save_hdf5(path, result, metadata):
    with H5File(path, "w") as f:
        grp = f.create_group("samples", track_order=True)
        for key in sorted(result.samples):
            grp.create_dataset(
                key, data=np.asarray(result.samples[key]),
                dtype=np.float32)

        info = f.create_group("info", track_order=True)
        for key in sorted(result.info):
            info.create_dataset(key, data=np.asarray(result.info[key]))

        params = f.create_group("sampler_parameters", track_order=True)
        for key in sorted(result.parameters):
            params.create_dataset(key, data=np.asarray(result.parameters[key]))

        f.attrs["sampler"] = "blackjax_mwg"
        f.attrs["theta_sites"] = ",".join(result.theta_sites)
        for key, value in metadata.items():
            f.attrs[key] = value


def _make_corner_plot(samples, path, keys, smooth):
    import matplotlib
    matplotlib.use("Agg")
    from candel.plotting.corner import plot_corner

    present = [
        key for key in keys
        if key in samples and np.asarray(samples[key]).ndim == 1]
    if len(present) < 2:
        fprint("skipping corner plot: fewer than two scalar samples present.")
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        plot_corner(
            samples, show_fig=False, filename=path, smooth=smooth,
            keys=present)
    except ValueError as exc:
        fprint(f"skipping corner plot: {exc}")
        return None
    return path


def _corner_keys(result):
    keys = list(result.theta_sites)
    if "log_MBH" in result.samples and "log_MBH" not in keys:
        idx = keys.index("eta") + 1 if "eta" in keys else len(keys)
        keys.insert(idx, "log_MBH")
    return tuple(keys)


def main(argv=None):
    sampler, argv = _select_sampler(argv)
    if sampler in ("de", "lbfgs"):
        return _run_profile_map(argv, sampler)

    parser = argparse.ArgumentParser(
        description="Run one megamaser disk inference job.")
    parser.add_argument("galaxy", type=str)
    parser.add_argument("--sampler", choices=("gibbs", "de", "lbfgs"),
                        default="gibbs",
                        help="Sampler to run. Default: gibbs.")
    parser.add_argument("--data-root", type=str,
                        default=data_path("data", "Megamaser"))
    parser.add_argument("--num-warmup", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-inner", type=int, default=None)
    parser.add_argument("--spot-batch", type=int, default=None)
    parser.add_argument("--n-sys", type=int, default=None)
    parser.add_argument("--n-red", type=int, default=None)
    parser.add_argument("--n-blue", type=int, default=None)
    parser.add_argument("--target-accept-nuts", type=float, default=None)
    parser.add_argument("--target-accept-r", type=float, default=None)
    parser.add_argument("--initial-step-size", type=float, default=None)
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument("--diagonal-mass", action="store_true",
                        help="Adapt a diagonal global inverse mass matrix.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-jit-steps", action="store_true")
    parser.add_argument("--no-ecc", action="store_true")
    parser.add_argument("--add-ecc", action="store_true")
    parser.add_argument("--no-quadratic-warp", action="store_true")
    parser.add_argument("--add-quadratic-warp", action="store_true")
    parser.add_argument("--mass-parameterization",
                        choices=("eta", "log_mbh"), default=None,
                        help="Global mass coordinate for the sampler. "
                             "Default: config value, eta in config_maser.")
    parser.add_argument("--f64", action="store_true", default=_ENABLE_F64)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    if args.f64 and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
        print("float64 enabled (--f64)", flush=True)

    if importlib.util.find_spec("blackjax") is None:
        raise SystemExit(
            "BlackJAX is not installed in this environment. Install "
            "blackjax before running the megamaser Gibbs sampler.")

    galaxies = master_cfg["model"]["galaxies"]
    if args.galaxy not in galaxies:
        raise SystemExit(
            f"Unknown galaxy {args.galaxy!r}. Available: {list(galaxies)}")

    inf_cfg = master_cfg["inference"]
    seed = args.seed if args.seed is not None else inf_cfg.get("seed", 42)
    num_warmup = (
        args.num_warmup if args.num_warmup is not None
        else inf_cfg.get("num_warmup", 2500))
    num_samples = (
        args.num_samples if args.num_samples is not None
        else inf_cfg.get("num_samples", 2000))
    target_accept_nuts = (
        args.target_accept_nuts if args.target_accept_nuts is not None
        else inf_cfg.get("target_accept_prob", 0.9))
    initial_step_size = (
        args.initial_step_size if args.initial_step_size is not None
        else inf_cfg.get("blackjax_initial_step_size", 0.01))
    max_tree_depth = (
        args.max_tree_depth if args.max_tree_depth is not None
        else inf_cfg.get("max_tree_depth", 10))
    n_inner = (
        args.n_inner if args.n_inner is not None
        else inf_cfg.get("n_inner", 20))
    target_accept_r = (
        args.target_accept_r if args.target_accept_r is not None
        else inf_cfg.get("target_accept_r", 0.44))
    init_strategy = inf_cfg.get("init_strategy", "median")
    init_num_samples = int(inf_cfg.get("init_num_samples", 100))

    gcfg_master = galaxies[args.galaxy]
    fsection(f"Loading {args.galaxy} data")
    data = load_megamaser_spots(
        args.data_root, args.galaxy, v_sys_obs=gcfg_master["v_sys_obs"])
    if "D_lo" in gcfg_master and "D_hi" in gcfg_master:
        data["D_lo"] = float(gcfg_master["D_lo"])
        data["D_hi"] = float(gcfg_master["D_hi"])
    downsample_counts = (args.n_sys, args.n_red, args.n_blue)
    if any(count is not None for count in downsample_counts):
        if any(count is None for count in downsample_counts):
            raise SystemExit(
                "--n-sys, --n-red, and --n-blue must be provided together.")
        data = _downsample_spots(data, args.n_sys, args.n_red, args.n_blue)

    config = {
        "inference": {
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "seed": int(seed),
            "max_tree_depth": int(max_tree_depth),
        },
        "model": copy.deepcopy(master_cfg["model"]),
        "io": master_cfg["io"],
    }
    config["model"]["mode"] = "gibbs"
    config["model"]["galaxies"][args.galaxy].pop("mode", None)
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

    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp)
    tmp.close()
    try:
        model = MaserDiskModel(tmp.name, data)
    finally:
        os.unlink(tmp.name)

    init_key, run_key = random.split(random.PRNGKey(seed))
    init_params = _make_init(
        model, config["model"]["galaxies"][args.galaxy].get("init", {}),
        init_strategy, init_num_samples, init_key)
    fprint(f"init: {init_strategy} ({init_num_samples} prior samples)")
    _print_init(init_params)

    fsection("BlackJAX MWG")
    backend = jax.default_backend()
    precision = "float64" if jax.config.jax_enable_x64 else "float32"
    fprint(f"JAX backend: {backend}; precision: {precision}")
    fprint(f"spots={model.n_spots}; warmup={num_warmup}; "
           f"samples={num_samples}; n_inner={n_inner}")
    fprint(f"target_accept_nuts={target_accept_nuts}; "
           f"target_accept_r={target_accept_r}")
    fprint("phi: marginalised by quadrature")
    fprint("r_ang: sampled as z_r = log(r_ang / r_hat(theta))")
    fprint(f"mass: {model.mass_parameterization}")
    fprint("mass matrix: diagonal" if args.diagonal_mass else
           "mass matrix: dense")

    t0 = time.time()
    result = run_blackjax_mwg(
        model,
        init_params,
        run_key,
        num_warmup=num_warmup,
        num_samples=num_samples,
        n_inner=n_inner,
        spot_batch=args.spot_batch,
        dense_mass=not args.diagonal_mass,
        target_accept_nuts=target_accept_nuts,
        target_accept_r=target_accept_r,
        initial_step_size=initial_step_size,
        max_num_doublings=max_tree_depth,
        progress_bar=not args.no_progress,
        jit_steps=not args.no_jit_steps,
    )
    fprint(f"total runner wall time: {time.time() - t0:.1f}s")

    _print_summary(result.samples, result.info, result)

    outdir = results_path(master_cfg["io"].get(
        "root_output", "results/Maser"))
    os.makedirs(outdir, exist_ok=True)
    suffix = "blackjax_gibbs_rang"
    outpath = args.output or os.path.join(
        outdir, f"{args.galaxy}_{suffix}.hdf5")
    outpath = os.path.abspath(outpath)
    metadata = {
        "galaxy": args.galaxy,
        "mode": model.mode,
        "seed": int(seed),
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "n_inner": int(n_inner),
        "target_accept_nuts": float(target_accept_nuts),
        "target_accept_r": float(target_accept_r),
        "initial_step_size": float(initial_step_size),
        "max_tree_depth": int(max_tree_depth),
        "dense_mass": bool(not args.diagonal_mass),
        "r_parameterization": "log_r_ang_over_seed",
        "mass_parameterization": model.mass_parameterization,
        "runtime_seconds": float(result.runtime_seconds),
        "n_spots": int(model.n_spots),
    }
    _save_hdf5(outpath, result, metadata)
    fprint(f"saved samples to {outpath}")

    corner_path = os.path.splitext(outpath)[0] + "_corner.png"
    corner_unsmoothed_path = (
        os.path.splitext(outpath)[0] + "_corner_unsmoothed.png")
    corner_keys = _corner_keys(result)
    made_smooth = _make_corner_plot(
        result.samples, corner_path, corner_keys, smooth=1)
    made_unsmoothed = _make_corner_plot(
        result.samples, corner_unsmoothed_path, corner_keys, smooth=0)
    if made_smooth is None or made_unsmoothed is None:
        fprint("corner plot could not be generated.")


if __name__ == "__main__":
    main()
