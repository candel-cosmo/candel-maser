# Copyright (C) 2026 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Unified megamaser runner.

The default sampler is the BlackJAX explicit-latent (r, phi) NUTS chain
(``--sampler mcmc``).  ``--sampler de`` runs the 2D-marginal
differential-evolution MAP (``run_de_map``) used as a global search to seed it.
"""
import argparse
import contextlib
import copy
import importlib.util
import io
import os
import sys
import tempfile
import time

import numpy as np
import tomli
import tomli_w
from h5py import File as H5File
from scipy.signal import correlate

from candel_maser.paths import CONFIG_PATH, LOCAL_CONFIG_PATH  # noqa: E402

try:
    with open(LOCAL_CONFIG_PATH, "rb") as f:
        _lcfg = tomli.load(f)
except OSError:
    _lcfg = {}

ld = os.environ.get("LD_LIBRARY_PATH", "")
needed = [p for p in _lcfg.get("gpu_ld_library_path", []) if p not in ld]
if needed:
    os.environ["LD_LIBRARY_PATH"] = ":".join(needed) + (f":{ld}" if ld else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

with open(CONFIG_PATH, "rb") as f:
    master_cfg = tomli.load(f)


def _select_sampler(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sampler", choices=("mcmc", "de"),
                        default="mcmc")
    args, _ = parser.parse_known_args(argv)
    return args.sampler, argv


# MCMC always uses float64. DE retains the explicit/per-galaxy precision
# policy, including forced float64 for NGC4258's extreme geometry.
def _f64_reason_from_argv(argv):
    sampler, _ = _select_sampler(argv)
    if sampler == "mcmc":
        return "MCMC default"
    if "--f64" in argv:
        return "--f64"
    galaxies = master_cfg["model"]["galaxies"]
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
from jax import random  # noqa: E402

if _F64_ENABLED_HERE:
    print(f"float64 enabled ({_F64_REASON})", flush=True)

import candel_maser.maser_physics as maser_physics  # noqa: E402
from .maser_blackjax import MaserBlackJaxTarget  # noqa: E402
from .maser_blackjax import init_from_prior_median  # noqa: E402
from .maser_blackjax import (_ngc5765b_reference_r_ang_bounds,  # noqa: E402
                             nudge_initial_params_inside_support,
                             prepare_floor_init, run_blackjax_mcmc)
from .model_H0_maser import MaserDiskModel  # noqa: E402
from .megamaser_data import load_megamaser_spots  # noqa: E402
from .megamaser_data import maser_data_root  # noqa: E402
from candel.util import fprint, fsection, results_path  # noqa: E402
from .maser_config import (  # noqa: E402
    add_dataset_arg, apply_dataset, check_init_block, variant_init_block)

_REID_CLIGHT = 2.997925e5
_LATENT_SUMMARY_SPOT_CHUNK = 8


class _StdoutTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _capture_stdout(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(_StdoutTee(sys.stdout, buf)):
        func(*args, **kwargs)
    return buf.getvalue()


def _write_run_summary(path, sections):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Megamaser MCMC summary\n")
        for section in sections:
            if not section:
                continue
            f.write("\n")
            f.write(section.rstrip())
            f.write("\n")
    fprint(f"saved text summary to {path}")


def _reid_physics_constants():
    vearth = 29.785
    au2km = 1.496e8
    sun_mass = 1.98892e33
    secinyr = 365.2422 * 86400.0
    g_cgs = 6.674e-8
    cv = vearth * np.sqrt(1.0e7 / 1000.0)
    ca = (cv ** 2 / (au2km * 1000.0)) * secinyr
    cg = (
        2.0 * g_cgs * 1.0e7 * sun_mass
        / (_REID_CLIGHT * 1.0e5) ** 2
        * 1.0e-5 / (1000.0 * au2km)
    )
    return float(cv), float(ca), float(cg)


def _apply_reid_physics_constants():
    cv, ca, cg = _reid_physics_constants()
    maser_physics.C_v = cv
    maser_physics.C_a = ca
    maser_physics.C_g = cg
    maser_physics.SPEED_OF_LIGHT = _REID_CLIGHT
    maser_physics.REID_CIRCULAR_GAMMA = True
    fprint("using Reid fit_disk physics constants "
           f"(C_v={cv:.6g}, C_a={ca:.6g}, C_g={cg:.6g}, "
           f"c={_REID_CLIGHT:.6g}) + circular-speed SR gamma")


def _h_ref(model):
    return float(model.config["model"].get("H0_ref", 73.0)) / 100.0


def _D_A_from_D_c(model, D_c):
    h = _h_ref(model)
    z_cosmo = model.distance2redshift(jnp.atleast_1d(D_c), h=h).squeeze()
    return D_c / (1.0 + z_cosmo)


def _strip_sampler_arg(argv):
    """Drop ``--sampler <value>`` (or ``--sampler=value``) from argv."""
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok == "--sampler":
            skip = True
            continue
        if tok.startswith("--sampler="):
            continue
        out.append(tok)
    return out


def _clean_init(model, init_cfg):
    check_init_block(init_cfg, model)
    init_params = {key: jnp.asarray(value) for key, value in init_cfg.items()}
    init_params.pop("H0", None)
    init_params.pop("sigma_pec", None)
    init_params.pop("M_BH", None)
    if model._D_A_uniform:
        if "D_A" not in init_params:
            if "D_c" not in init_params:
                raise KeyError(
                    "D_A-uniform mode requires initial 'D_A' or 'D_c'.")
            init_params["D_A"] = _D_A_from_D_c(model, init_params["D_c"])
    elif "D_c" not in init_params:
        raise KeyError(
            "BlackJAX megamaser initial values must include 'D_c'.")
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
                raise KeyError(
                    "BlackJAX megamaser initial values must include "
                    "'log_MBH'.")
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
    else:
        for key in ("e_x", "e_y", "periapsis_rad"):
            init_params.pop(key, None)
        init_params.setdefault("ecc", jnp.asarray(0.0))
        init_params.setdefault("periapsis", jnp.asarray(0.0))
        init_params.setdefault("dperiapsis_dr", jnp.asarray(0.0))
    if not model.use_quadratic_warp:
        for key in ("d2i_dr2", "d2Omega_dr2"):
            init_params.pop(key, None)
    else:
        init_params.setdefault("d2i_dr2", jnp.asarray(0.0))
        init_params.setdefault("d2Omega_dr2", jnp.asarray(0.0))
    return prepare_floor_init(model, init_params)


def _paper_init(target, galaxy, master):
    from .reid.pesce_globals import candel_theta_from_point  # noqa: E402
    from .reid.pesce_globals import paper_point

    try:
        point, status = paper_point(galaxy, master)
    except KeyError as exc:
        raise KeyError(f"No Pesce/Reid fixed globals for {galaxy}") from exc
    if point is None:
        raise KeyError(
            f"Cannot build Pesce/Reid fixed globals for {galaxy}: {status}")
    if status:
        fprint("Reid/Pesce init defaulted " + ", ".join(status))
    return candel_theta_from_point(point, galaxy, master, target)


def _make_init(model, init_cfg, strategy, num_samples, rng_key, *,
               galaxy=None, master=None, spot_batch=None):
    strategy = str(strategy).lower()
    if strategy == "median":
        h = _h_ref(model)
        return _clean_init(
            model, init_from_prior_median(
                model, rng_key, num_samples, h=h))
    if strategy == "config":
        return _clean_init(model, init_cfg)
    if strategy == "reid":
        if galaxy is None or master is None:
            raise ValueError("reid init_strategy needs galaxy and master cfg.")
        base_init = _clean_init(model, init_cfg)
        target = MaserBlackJaxTarget(
            model, _h_ref(model), base_init, spot_batch=spot_batch)
        return _paper_init(target, galaxy, master)
    raise ValueError(
        "Megamaser init_strategy must be 'median', 'config', or 'reid'.")


_init_block = variant_init_block


def _print_init(init_params):
    fsection("Initial values")
    for key, value in sorted(init_params.items()):
        value = jnp.asarray(value)
        if value.ndim == 0:
            fprint(f"{key:20s} = {float(value):12.4f}")
        else:
            fprint(f"{key:20s} = [{value.size} values]")


def _nudge_boundary_init(model, h, init_params, rng_key):
    if isinstance(init_params, list):
        keys = random.split(rng_key, len(init_params))
        rows = [
            nudge_initial_params_inside_support(
                model, h, params, rng_key=key)
            for params, key in zip(init_params, keys)
        ]
        return [row[0] for row in rows], [row[1] for row in rows]
    params, adjusted = nudge_initial_params_inside_support(
        model, h, init_params, rng_key=rng_key)
    return params, [adjusted]


def _print_boundary_adjustments(adjustments):
    rows = []
    for i, chain_adjustments in enumerate(adjustments):
        for site, old, new, lower, upper in chain_adjustments:
            rows.append((i, site, old, new, lower, upper))
    if not rows:
        return
    fsection("Initial Boundary Guard")
    for i, site, old, new, lower, upper in rows:
        prefix = f"chain {i + 1}: " if len(adjustments) > 1 else ""
        fprint(
            f"{prefix}{site}: {old:.6g} -> {new:.6g} "
            f"inside [{lower:.6g}, {upper:.6g}]")


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
    acov = correlate(y, y, mode="full", method="fft")[n - 1:] / n
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


def _rhat_chains(chains):
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError("chains must have shape (n_chains, n_samples)")
    finite = np.all(np.isfinite(chains), axis=0)
    chains = chains[:, finite]
    n_chains, n_samples = chains.shape
    if n_chains < 2:
        return _split_rhat_1d(chains.reshape(-1))
    if n_samples < 2:
        return np.nan
    W = chains.var(axis=1, ddof=1).mean()
    if not np.isfinite(W) or W <= 0:
        return 1.0
    B = n_samples * chains.mean(axis=1).var(ddof=1)
    var_hat = ((n_samples - 1) / n_samples) * W + B / n_samples
    return float(np.sqrt(max(var_hat / W, 0.0)))


def _ess_rhat(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return _ess_1d(arr), _split_rhat_1d(arr)
    if arr.ndim == 2:
        neff = np.nansum([_ess_1d(chain) for chain in arr])
        return neff, _rhat_chains(arr)
    if arr.ndim != 3:
        return np.nan, np.nan
    neff = np.array([_ess_rhat(arr[:, :, i])[0]
                     for i in range(arr.shape[2])])
    rhat = np.array([_ess_rhat(arr[:, :, i])[1]
                     for i in range(arr.shape[2])])
    return neff, rhat


def _print_global_summary(samples):
    global_keys = (
        "D_A", "D_c", "eta", "log_MBH",
        "i0", "di_dr", "Omega0", "dOmega_dr",
        "d2i_dr2", "d2Omega_dr2",
        "e_x", "e_y", "dperiapsis_dr",
        "x0", "y0", "dv_sys", "sigma_x_floor", "sigma_y_floor",
        "sigma_v_sys", "sigma_v_hv", "sigma_a_floor",
        "sigma_x_floor_clump2", "sigma_y_floor_clump2",
        "sigma_v_floor_clump2", "sigma_a_floor_clump2",
    )
    printable = {}
    for key in global_keys:
        if key not in samples:
            continue
        arr = np.asarray(samples[key])
        if arr.ndim == 1:
            printable[key] = arr[None, :]
        elif arr.ndim == 2:
            printable[key] = arr
    if not printable:
        return
    n = int(np.prod(next(iter(printable.values())).shape))
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
        chains = np.asarray(printable[key])
        arr = chains.reshape(-1)
        q05, q50, q95 = np.quantile(arr, [0.05, 0.50, 0.95])
        neff, rhat = _ess_rhat(chains)
        print(
            f"{key:>17s} {arr.mean():9.2f} {arr.std(ddof=1):9.2f} "
            f"{q50:9.2f} {q05:9.2f} {q95:9.2f} "
            f"{neff:9.2f} {rhat:9.2f}",
            flush=True)


def _print_r_ang_summary(samples):
    if "r_ang" not in samples:
        return
    r = np.asarray(samples["r_ang"], dtype=float)
    if r.ndim == 2:
        r_chains = r[None, :, :]
    elif r.ndim == 3:
        r_chains = r
    else:
        return
    n_samples = int(r_chains.shape[0] * r_chains.shape[1])
    if n_samples < 20:
        fprint(f"r_ang summary skipped: only {n_samples} posterior samples "
               "(need >=20 for diagnostics).")
        return

    n_spots = r_chains.shape[-1]
    mean = np.empty(n_spots)
    sd = np.empty(n_spots)
    q05 = np.empty(n_spots)
    q50 = np.empty(n_spots)
    q95 = np.empty(n_spots)
    neff = np.empty(n_spots)
    rhat = np.empty(n_spots)
    move = np.empty(n_spots)
    for start in range(0, n_spots, _LATENT_SUMMARY_SPOT_CHUNK):
        idx = slice(start, start + _LATENT_SUMMARY_SPOT_CHUNK)
        block = r_chains[..., idx]
        flat = block.reshape(-1, block.shape[-1])
        neff[idx], rhat[idx] = _ess_rhat(block)
        mean[idx] = np.mean(flat, axis=0)
        sd[idx] = np.std(flat, axis=0)
        q05[idx], q50[idx], q95[idx] = np.quantile(
            flat, [0.05, 0.50, 0.95], axis=0)
        move[idx] = np.mean(
            np.abs(np.diff(block, axis=1)) > 0, axis=(0, 1))

    fsection("r_ang Summary")
    print(f"  n_spots              = {r_chains.shape[-1]}", flush=True)
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


def _print_phi_summary(samples):
    if "phi" not in samples:
        return
    phi = np.asarray(samples["phi"], dtype=float)
    if phi.ndim == 2:
        phi_chains = phi[None, :, :]
    elif phi.ndim == 3:
        phi_chains = phi
    else:
        return
    n_samples = int(phi_chains.shape[0] * phi_chains.shape[1])
    if n_samples < 20:
        fprint(f"phi summary skipped: only {n_samples} posterior samples "
               "(need >=20 for diagnostics).")
        return

    n_spots = phi_chains.shape[-1]
    mean = np.empty(n_spots)
    sd = np.empty(n_spots)
    q05 = np.empty(n_spots)
    q50 = np.empty(n_spots)
    q95 = np.empty(n_spots)
    neff = np.empty(n_spots)
    rhat = np.empty(n_spots)
    move = np.empty(n_spots)
    for start in range(0, n_spots, _LATENT_SUMMARY_SPOT_CHUNK):
        idx = slice(start, start + _LATENT_SUMMARY_SPOT_CHUNK)
        block = phi_chains[..., idx]
        flat = block.reshape(-1, block.shape[-1])
        sin_mean = np.mean(np.sin(flat), axis=0)
        cos_mean = np.mean(np.cos(flat), axis=0)
        centre = np.arctan2(sin_mean, cos_mean)
        centred = centre + np.angle(
            np.exp(1j * (block - centre[None, None, :])))
        neff[idx], rhat[idx] = _ess_rhat(centred)
        np.degrees(centred, out=centred)
        q05[idx], q50[idx], q95[idx] = np.quantile(
            centred.reshape(-1, block.shape[-1]),
            [0.05, 0.50, 0.95], axis=0)
        R = np.clip(np.hypot(sin_mean, cos_mean), 1e-300, 1.0)
        mean[idx] = np.degrees(centre)
        sd[idx] = np.degrees(np.sqrt(-2.0 * np.log(R)))
        del centred
        step = np.abs(np.angle(np.exp(1j * np.diff(block, axis=1))))
        move[idx] = np.mean(step > 0, axis=(0, 1))

    fsection("phi Summary")
    print(f"  n_spots              = {phi_chains.shape[-1]}", flush=True)
    print(f"  mean range           = {mean.min():.2f} .. {mean.max():.2f} deg",
          flush=True)
    print(f"  sd range             = {sd.min():.2f} .. {sd.max():.2f} deg",
          flush=True)
    print(f"  q05 range            = {q05.min():.2f} .. {q05.max():.2f} deg",
          flush=True)
    print(f"  median range         = {q50.min():.2f} .. {q50.max():.2f} deg",
          flush=True)
    print(f"  q95 range            = {q95.min():.2f} .. {q95.max():.2f} deg",
          flush=True)
    print(f"  n_eff min/median/max = {np.nanmin(neff):.1f} / "
          f"{np.nanmedian(neff):.1f} / {np.nanmax(neff):.1f}", flush=True)
    print(f"  r_hat max            = {np.nanmax(rhat):.3f}", flush=True)
    print(f"  move frac min/median/max = {move.min():.3f} / "
          f"{np.median(move):.3f} / {move.max():.3f}", flush=True)


def _print_summary(samples, info, result):
    fsection("Results")
    for key in ("D_A", "D_c", "log_MBH",
                "eta",
                "i0", "di_dr", "Omega0", "dOmega_dr",
                "d2i_dr2", "d2Omega_dr2",
                "e_x", "e_y", "dperiapsis_dr",
                "x0", "y0", "dv_sys", "sigma_x_floor", "sigma_y_floor",
                "sigma_v_sys", "sigma_v_hv", "sigma_a_floor",
                "sigma_x_floor_clump2", "sigma_y_floor_clump2",
                "sigma_v_floor_clump2", "sigma_a_floor_clump2"):
        if key not in samples:
            continue
        arr = np.asarray(samples[key])
        print(f"  {key:20s} = {arr.mean():10.3f} +/- {arr.std():8.3f}",
              flush=True)

    n_div = int(np.asarray(info.get("nuts_is_divergent", [])).sum())
    n_theta_div = int(np.asarray(info.get("theta_is_divergent", [])).sum())
    acc = np.asarray(info.get("nuts_acceptance_rate", []))
    tacc = np.asarray(info.get("theta_acceptance_rate", []))
    racc = np.asarray(info.get("r_accept_mean", []))
    lacc = np.asarray(info.get("latent_accept_mean", []))
    refl = np.asarray(info.get("reflect_accept_mean", []))
    print(f"\nWall time: {result.runtime_seconds:.1f}s", flush=True)
    if "nuts_is_divergent" in info:
        print(f"NUTS divergences: {n_div}", flush=True)
    if acc.size:
        print(f"NUTS acceptance: mean={acc.mean():.3f}", flush=True)
    if "theta_is_divergent" in info:
        print(f"global NUTS divergences: {n_theta_div}", flush=True)
    if tacc.size:
        print(f"global theta acceptance: mean={tacc.mean():.3f}",
              flush=True)
    if racc.size:
        print(f"r proposal acceptance: mean={racc.mean():.3f}",
              flush=True)
    if lacc.size:
        print(f"latent proposal acceptance: mean={lacc.mean():.3f}",
              flush=True)
    if refl.size:
        print(f"reflect proposal acceptance: mean={refl.mean():.3f}",
              flush=True)
    _print_global_summary(samples)
    _print_r_ang_summary(samples)
    _print_phi_summary(samples)


_LATENT_SAMPLE_KEYS = {"r_ang", "phi"}


def _save_hdf5(path, result, metadata, *, save_latents=False):
    with H5File(path, "w") as f:
        grp = f.create_group("samples", track_order=True)
        for key in sorted(result.samples):
            if not save_latents and key in _LATENT_SAMPLE_KEYS:
                continue
            grp.create_dataset(key, data=np.asarray(result.samples[key]))

        f.create_dataset("log_density", data=np.asarray(result.log_density))

        info = f.create_group("info", track_order=True)
        for key in sorted(result.info):
            info.create_dataset(key, data=np.asarray(result.info[key]))

        params = f.create_group("sampler_parameters", track_order=True)
        for key in sorted(result.parameters):
            params.create_dataset(key, data=np.asarray(result.parameters[key]))

        f.attrs["sampler"] = metadata.get("sampler", "blackjax_explicit_mcmc")
        f.attrs["theta_sites"] = ",".join(result.theta_sites)
        for key, value in metadata.items():
            f.attrs[key] = value


def _make_corner_plot(samples, path, keys, smooth, truths=None, points=None,
                      point_label=None, truth_label=None,
                      map_point=None, map_label="MAP"):
    import matplotlib
    matplotlib.use("Agg")
    from candel.plotting.corner import plot_corner

    plot_samples = {}
    present = []
    for key in keys:
        if key not in samples:
            continue
        arr = np.asarray(samples[key])
        if arr.ndim == 1:
            plot_samples[key] = arr
        elif arr.ndim == 2:
            plot_samples[key] = arr.reshape(-1)
        else:
            continue
        present.append(key)
    if len(present) < 2:
        fprint("skipping corner plot: fewer than two scalar samples present.")
        return None
    n_plot_samples = len(plot_samples[present[0]])
    if n_plot_samples <= len(present):
        fprint("skipping corner plot: fewer samples than plotted parameters.")
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        plot_corner(
            plot_samples, show_fig=False, filename=path, smooth=smooth,
            keys=present, truths=truths, points=points,
            point_label=point_label, truth_label=truth_label,
            map_point=map_point, map_label=map_label)
    except (AssertionError, ValueError) as exc:
        fprint(f"skipping corner plot: {exc}")
        return None
    return path


def _corner_keys(result):
    keys = list(result.theta_sites)
    if "log_MBH" in result.samples and "log_MBH" not in keys:
        idx = keys.index("eta") + 1 if "eta" in keys else len(keys)
        keys.insert(idx, "log_MBH")
    return tuple(keys)


def _corner_truths(init_cfg, keys):
    truths = {}
    for key in keys:
        if key not in init_cfg:
            continue
        value = np.asarray(init_cfg[key])
        if value.ndim == 0:
            truths[key] = float(value)
    return truths or None


def _make_log_density_plot(result, path, config_init_logp=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chains = np.asarray(result.log_density, dtype=float)
    if chains.ndim == 1:
        chains = chains[None, :]
    elif chains.ndim != 2:
        chains = chains.reshape(-1, chains.shape[-1])

    finite = chains[np.isfinite(chains)]
    if finite.size == 0:
        fprint("skipping logP plot: no finite log_density values.")
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 5.5), constrained_layout=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]})
    ax_trace, ax_hist = axes

    x = np.arange(chains.shape[1])
    show_chain_labels = 1 < chains.shape[0] <= 4
    for i, chain in enumerate(chains):
        label = f"chain {i + 1}" if show_chain_labels else None
        ax_trace.plot(x, chain, lw=0.7, alpha=0.75, label=label)

    best = float(np.nanmax(finite))
    median = float(np.nanmedian(finite))
    ax_trace.axhline(best, color="0.25", ls=":", lw=1.0,
                     label="max sample")
    ax_hist.axvline(best, color="0.25", ls=":", lw=1.0)
    ax_hist.axvline(median, color="0.55", ls="--", lw=1.0,
                    label="median")
    if config_init_logp is not None and np.isfinite(config_init_logp):
        ax_trace.axhline(config_init_logp, color="crimson", lw=1.2,
                         label="config init")
        ax_hist.axvline(config_init_logp, color="crimson", lw=1.2,
                        label="config init")

    ax_trace.set_xlabel("posterior sample")
    ax_trace.set_ylabel("logP")
    ax_hist.hist(finite, bins=min(80, max(10, int(np.sqrt(finite.size)))),
                 histtype="stepfilled", color="0.72", edgecolor="0.25",
                 alpha=0.8)
    ax_hist.set_xlabel("logP")
    ax_hist.set_ylabel("count")
    ax_trace.legend(loc="best", fontsize=8)
    ax_hist.legend(loc="best", fontsize=8)

    fig.savefig(path, dpi=180)
    plt.close(fig)
    fprint(f"saved logP diagnostic to {path}")
    return path


def _complete_mass_point(model, point):
    point = dict(point)
    D_A = _D_A_from_point(model, point)
    if D_A is None:
        return point
    if "log_MBH" in point:
        if getattr(model, "mass_parameterization", "eta") == "eta":
            point["eta"] = float(point["log_MBH"]) - np.log10(D_A)
    elif "eta" in point:
        point["log_MBH"] = float(point["eta"]) + np.log10(D_A)
    return point


def _median_point(result, model):
    point = {}
    for key, value in result.samples.items():
        arr = np.asarray(value, dtype=float)
        if key in ("r_ang", "phi"):
            point[key] = np.median(arr.reshape(-1, arr.shape[-1]), axis=0)
        elif arr.ndim <= 2:
            point[key] = float(np.median(arr.reshape(-1)))
    return _complete_mass_point(model, point)


def _D_A_from_point(model, point):
    if "D_A" in point:
        return float(np.asarray(point["D_A"]))
    if "D_c" in point:
        return float(_D_A_from_D_c(model, point["D_c"]))
    return None


def _required_inference(cfg, key):
    if key not in cfg:
        raise KeyError(f"Missing [inference].{key} in {CONFIG_PATH}")
    return cfg[key]


_REID_GLOBAL_INIT_KEYS = (
    "D_A", "D_c", "eta", "log_MBH", "dv_sys", "x0", "y0",
    "i0", "di_dr", "d2i_dr2", "Omega0", "dOmega_dr", "d2Omega_dr2",
    "e_x", "e_y", "dperiapsis_dr",
    "sigma_x_floor", "sigma_y_floor", "sigma_v_sys", "sigma_v_hv",
    "sigma_a_floor")


def _reid_loglik_context(galaxy, n_spots, dataset=None, *, use_ecc=False,
                         use_quadratic_warp=False):
    """Import the f2py Reid likelihood and build its data once.

    Returns ``(rp, rr, d, dataset)`` or None if the Reid likelihood (reidlik)
    is not built in this environment (e.g. on the cluster) or the spot count
    differs.
    """
    if dataset is None:
        dataset = master_cfg["io"]["dataset"]
    try:
        from .reid import prepare_reid_data
        import candel_maser.reid.reid_profile as rp
        import candel_maser.reid.run_reid_mcmc as rr
    except ImportError as exc:
        fprint(f"skipping Reid log-likelihood scatter: {exc}")
        return None

    rp.setup_numbers()
    tmp = tempfile.NamedTemporaryFile(
        suffix="_reid_scatter.inp", delete=False)
    inp = tmp.name
    tmp.close()
    try:
        # Without forwarding the dataset this regenerates the .inp from
        # whichever one the config defaults to, silently scoring against the
        # wrong table.
        prepare_args = [galaxy, "--out", inp, "--dataset", dataset]
        if use_ecc:
            prepare_args.append("--add-ecc")
        if use_quadratic_warp:
            prepare_args.append("--add-quadratic-warp")
        prepare_reid_data.main(prepare_args)
        d = rp.build_data(inp)
    finally:
        os.unlink(inp)
    if d["N"] != int(n_spots):
        fprint(f"skipping Reid scatter: spot count mismatch "
               f"(Reid {d['N']} vs CANDEL {n_spots}).")
        return None
    return rp, rr, d, dataset


def _reid_h0_for_D_A(rp, g, D_A):
    """H0 that makes Reid calc_warped_model use the requested D_A exactly."""
    v = float(g["Vsys_km_s"] + g["Vcor_km_s"])
    n_v = int(v + 0.5)
    eq14int, _ = rp.reidlik.dampc(float(n_v), 100.0)
    return _REID_CLIGHT * float(eq14int) / (
        float(D_A) * (1.0 + v / _REID_CLIGHT))


def _point_D_A(model, target, point):
    theta = target.complete_params(
        {name: jnp.asarray(point[name]) for name in target.names})
    phys_args, _ = model.phys_from_params_jax(theta, target.h)
    return float(np.asarray(jax.device_get(phys_args[2])))


def _reid_neg_half_chi2(ctx, galaxy, point, r_ang, phi, D_A=None):
    """Reid fit_disk per-spot data-fit term -0.5*chi^2 at fixed (r_ang, phi).

    CANDEL globals map to Reid's convention via candel_maser/reid (load_config_init)
    (i0 -> 180 - i0, CMB velocity frame); phi is converted rad -> deg and
    r_ang stays in mas.  If ``D_A`` is supplied, Reid's ``H0`` is reset using
    the same integer ``Ez_int(n_v)`` lookup as ``calc_warped_model`` so Reid's
    internal angular-diameter distance equals CANDEL's value.  Returns
    -0.5*chi^2 per spot
    (loader order) using Reid's own predictions (calc_warped_model) and error
    floors (add_error_floors).  We compare -0.5*chi^2 rather than the full
    log-likelihood so the result carries NO normalisation constants: Reid's
    calc_ln_p_data drops the -0.5*ln(2*pi) per data point and its -ln(sigma)
    term differs from CANDEL's only by the uas-vs-mas position unit; both are
    pure additive constants that say nothing about the fit and are excluded
    here.  chi^2 is dimensionless, so the result is convention-free.
    """
    rp, rr, d, dataset = ctx
    r_ang = np.asarray(r_ang, dtype=float)
    init_block = {k: float(point[k]) for k in _REID_GLOBAL_INIT_KEYS
                  if k in point and np.asarray(point[k]).ndim == 0}
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump({"model": {"galaxies": {galaxy: {"init": init_block}}}}, tmp)
    tmp.close()
    try:
        # This fragment carries only the point, so the warp pivots must come
        # from the dataset file; without it they default to 0 and
        # shift_warp_pivots silently skips the shift.
        reid_init = rr.load_toml_init(rr.Path(tmp.name), galaxy, 0.0,
                                      dataset=dataset)
    finally:
        os.unlink(tmp.name)
    g = rp.with_derived(rr.shift_warp_pivots(
        reid_init.values,
        rp.reid_r_ref(d, reid_init.values["x0_mas"],
                      reid_init.values["y0_mas"])))
    if D_A is not None:
        g["H0"] = _reid_h0_for_D_A(rp, g, D_A)
        g = rp.with_derived(g)
    params = rp.globals_to_params(g)
    rp.fill_ez(g["H0"], g["Vsys_km_s"], g["Vcor_km_s"])
    r_ref = rp.reid_r_ref(d, g["x0_mas"], g["y0_mas"])
    # res_err is the per-datum 1-sigma error (data error and floor in
    # quadrature), so chi^2_k = (resid_k / res_err_k)^2.
    res_err = rp.reidlik.add_error_floors(
        params, d["N"], d["vlsr"], d["Vmin"], d["Vmax"],
        d["x_err"], d["y_err"], d["vlsr_err"], d["acc_err"])

    phi_deg = np.degrees(np.asarray(phi, dtype=float))
    data = d["data"]
    measured = np.asarray(d["acc_err"][:d["N"]]) > 0.0
    p = params.copy()
    out = np.zeros(d["N"])
    for i in range(d["N"]):
        base = rp.NUM_GLOBAL + 2 * i
        p[base] = r_ang[i]
        p[base + 1] = phi_deg[i]
        cx, cy, cv, ca = rp.reidlik.calc_warped_model(
            p, rp.NUM_GLOBAL, i + 1, r_ref)
        d4 = data[4 * i:4 * i + 4]
        chi2 = (((d4[0] - cx) / res_err[4 * i]) ** 2
                + ((d4[1] - cy) / res_err[4 * i + 1]) ** 2
                + ((d4[2] - cv) / res_err[4 * i + 2]) ** 2)
        if measured[i]:
            chi2 += ((d4[3] - ca) / res_err[4 * i + 3]) ** 2
        out[i] = -0.5 * chi2
    return out


def _candel_neg_half_chi2(model, target, point, r_ang, phi):
    """CANDEL per-spot -0.5*chi^2 at fixed (r_ang, phi).

    This is ``_eval_phi_fixed`` (the full per-spot Gaussian log-likelihood)
    with its ``-0.5*ln(2*pi*var)`` normalisation removed, leaving only the
    data-fit term so it is directly comparable to ``_reid_neg_half_chi2``.
    """
    theta = target.complete_params(
        {name: jnp.asarray(point[name]) for name in target.names})
    phys_args, phys_kw = model.phys_from_params_jax(theta, target.h)
    groups = list(model._spot_groups_from_r(jnp.asarray(r_ang)))
    full = np.asarray(jax.device_get(
        model._eval_phi_fixed(groups, jnp.asarray(phi), phys_args, phys_kw)),
        dtype=float)
    lnorm = np.zeros(model.n_spots)
    for group in groups:
        type_key, idx, r_g, _ = group
        if int(idx.shape[0]) == 0:
            continue
        rpre = model._r_precompute(
            r_g, idx, *phys_args, **phys_kw,
            has_any_accel=model._group_has_any_accel(type_key))
        lnorm[np.asarray(idx)] = (np.asarray(rpre["lnorm"])
                                  + np.asarray(rpre["lnorm_a"]))
    return full - lnorm


def _variant_suffix(model, args, init_strategy):
    parts = []
    if model.use_ecc:
        parts.append("ecc")
    if model.use_quadratic_warp:
        parts.append("qw")
    if model.galaxy_name == "NGC5765b":
        if model.clump2_acceleration_only:
            parts.append("accelfloor")
        elif not model.use_clump2_floors:
            parts.append("singlefloor")
    if args.da2_prior:
        parts.append("da2")
    # Always tag the resolved init strategy (median/config/reid) last, so runs
    # that differ only by initialisation stay apart and glob cleanly.
    parts.append(f"init{init_strategy}")
    return "_" + "_".join(parts)


def _run_evidence_subprocess(galaxy, chain_path, data_root, spot_batch,
                             dataset=None):
    """Compute the finite-support marginal-objective diagnostic.

    Spawned as a separate process so the float64 reference grid is isolated
    from sampler state. Best-effort: a failure here never aborts sampling.
    """
    import subprocess

    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "evidence_single_galaxy.py")
    cmd = [sys.executable, "-u", script, galaxy, "--chain", chain_path]
    if dataset:
        cmd += ["--dataset", dataset]
    if data_root:
        cmd += ["--data-root", data_root]
    if spot_batch is not None:
        cmd += ["--spot-batch", str(spot_batch)]
    fsection("Single-galaxy marginal-objective diagnostic (harmonic)")
    fprint("running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:                            # never abort the run
        fprint(f"marginal-objective diagnostic failed: {exc}")


def main(argv=None):
    sampler, argv = _select_sampler(argv)

    if sampler == "de":
        from . import run_de_map
        return run_de_map.main(_strip_sampler_arg(argv))

    argv = _strip_sampler_arg(argv)
    parser = argparse.ArgumentParser(
        description="Run one megamaser disk inference job.")
    parser.add_argument("galaxy", type=str)
    add_dataset_arg(parser)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Spot-table directory. Default: the selected "
                             "dataset's directory.")
    parser.add_argument("--num-warmup", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None,
                        help="Number of chains to run "
                             "concurrently, capped by available CPUs.")
    parser.add_argument("--chain-workers", type=int, default=None,
                        help="Maximum chains to run concurrently. Default: "
                             "[inference].chain_workers (8).")
    parser.add_argument("--init-strategy",
                        choices=("median", "config", "reid"),
                        default=None,
                        help="Initial global point strategy. Default: "
                             "inference/init_strategy for any chain count. "
                             "'reid' uses "
                             "reported Pesce/Reid globals; for NGC4258 it "
                             "reads reid_ngc4258_best.toml.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-inner", type=int, default=None)
    parser.add_argument("--latent-burnin", type=int, default=None,
                        help="Latent-only Gibbs burn-in steps at fixed theta "
                             "before warmup (config/reid init only). Default: "
                             "[inference].latent_burnin.")
    parser.add_argument("--spot-batch", type=int, default=None)
    parser.add_argument("--target-accept-r", type=float, default=None)
    parser.add_argument("--target-accept-theta", type=float, default=None,
                        help="Target global NUTS acceptance for "
                             "--sampler mcmc.")
    parser.add_argument("--initial-step-size", type=float, default=None)
    parser.add_argument("--phi-step-size", type=float, default=None,
                        help="Initial phi proposal scale in radians for "
                             "--sampler mcmc.")
    parser.add_argument("--reflect-prob", type=float, default=None,
                        help="High-velocity phi reflection proposal "
                             "probability for --sampler mcmc.")
    phi_transport = parser.add_mutually_exclusive_group()
    phi_transport.add_argument(
        "--transport-systemic-phi", dest="transport_systemic_phi",
        action="store_true", default=None)
    phi_transport.add_argument(
        "--no-transport-systemic-phi", dest="transport_systemic_phi",
        action="store_false")
    cond_mass = parser.add_mutually_exclusive_group()
    cond_mass.add_argument(
        "--conditional-mass", dest="conditional_mass",
        action="store_true", default=None,
        help="Use the inverse conditional Hessian of the NUTS target as the "
             "dense mass matrix instead of BlackJAX's adapted (marginal) "
             "estimate. Default: config value.")
    cond_mass.add_argument(
        "--no-conditional-mass", dest="conditional_mass",
        action="store_false")
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument("--no-ecc", action="store_true")
    parser.add_argument("--add-ecc", action="store_true")
    parser.add_argument("--no-quadratic-warp", action="store_true")
    parser.add_argument("--add-quadratic-warp", action="store_true")
    parser.add_argument("--mass-parameterization",
                        choices=("eta", "log_mbh"), default=None,
                        help="Global mass coordinate for the sampler. "
                             "Default: config value, eta in config_maser.")
    parser.add_argument(
        "--da2-prior", action="store_true",
        help="Use p(D_A) proportional to D_A^2 over the configured D_A "
             "bounds instead of the default uniform D_A prior.")
    parser.add_argument(
        "--f64", action="store_true", default=_ENABLE_F64,
        help="Accepted for compatibility; MCMC always uses float64.")
    floor_mode = parser.add_mutually_exclusive_group()
    floor_mode.add_argument(
        "--single-error-floor", action="store_true",
        help="For NGC5765b, disable the separate clump-2 floors and use "
             "only the standard sampled floor for each observable. "
             "Other galaxies are unchanged.")
    floor_mode.add_argument(
        "--clump2-acceleration-floor-only", action="store_true",
        help="For NGC5765b, sample a separate clump-2 acceleration floor "
             "only; position and velocity use the standard floors.")
    parser.add_argument("--compute-evidence", action="store_true",
                        help="After sampling, compute the single-galaxy "
                             "finite-support harmonic marginal-objective "
                             "diagnostic. Disabled by default.")
    parser.add_argument("--save-latents", action="store_true",
                        help="Write per-spot r_ang/phi samples to HDF5. "
                             "Disabled by default; globals, log_density, and "
                             "sampler_parameters/latent_scale are still "
                             "saved.")
    parser.add_argument("--map-overlay", dest="map_overlay",
                        action="store_true", default=False,
                        help="Overlay the joint MAP (from the DE/config init) "
                             "on the corner plot. Disabled by default.")
    parser.add_argument("--no-map-overlay", dest="map_overlay",
                        action="store_false",
                        help="Disable the MAP overlay on the corner plot.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)
    if args.no_ecc and args.add_ecc:
        raise SystemExit("--no-ecc and --add-ecc are mutually exclusive.")
    if args.no_quadratic_warp and args.add_quadratic_warp:
        raise SystemExit(
            "--no-quadratic-warp and --add-quadratic-warp are mutually "
            "exclusive.")

    dataset = apply_dataset(master_cfg, args.dataset)
    if args.data_root is None:
        args.data_root = maser_data_root(dataset)

    galaxies = master_cfg["model"]["galaxies"]
    if args.galaxy not in galaxies:
        raise SystemExit(
            f"Unknown galaxy {args.galaxy!r}. Available: {list(galaxies)}")
    gcfg_master = galaxies[args.galaxy]
    if importlib.util.find_spec("blackjax") is None:
        raise SystemExit(
            "BlackJAX is not installed in this environment. Install "
            "blackjax before running the megamaser sampler.")

    inf_cfg = master_cfg["inference"]
    seed = args.seed if args.seed is not None else _required_inference(
        inf_cfg, "seed")
    num_warmup = (
        args.num_warmup if args.num_warmup is not None
        else _required_inference(inf_cfg, "num_warmup"))
    num_samples = (
        args.num_samples if args.num_samples is not None
        else _required_inference(inf_cfg, "num_samples"))
    num_chains = int(
        args.num_chains if args.num_chains is not None
        else _required_inference(inf_cfg, "num_chains"))
    chain_workers = int(
        args.chain_workers if args.chain_workers is not None
        else _required_inference(inf_cfg, "chain_workers"))
    if num_chains < 1:
        raise SystemExit("--num-chains must be >= 1.")
    if chain_workers < 1:
        raise SystemExit("--chain-workers must be >= 1.")
    initial_step_size = (
        args.initial_step_size if args.initial_step_size is not None
        else _required_inference(inf_cfg, "blackjax_initial_step_size"))
    max_tree_depth = (
        args.max_tree_depth if args.max_tree_depth is not None
        else _required_inference(inf_cfg, "max_tree_depth"))
    n_inner = (
        args.n_inner if args.n_inner is not None
        else _required_inference(inf_cfg, "n_inner"))
    sample_n_inner = (
        n_inner if args.n_inner is not None
        else gcfg_master.get(
            "mcmc_sample_n_inner", inf_cfg.get("sample_n_inner", n_inner)))
    transport_systemic_phi = (
        bool(args.transport_systemic_phi)
        if args.transport_systemic_phi is not None else bool(
            gcfg_master.get(
                "mcmc_transport_systemic_phi",
                inf_cfg.get("transport_systemic_phi", False))))
    conditional_mass = (
        bool(args.conditional_mass)
        if args.conditional_mass is not None
        else bool(inf_cfg.get("conditional_mass", True)))
    target_accept_r = (
        args.target_accept_r if args.target_accept_r is not None
        else _required_inference(inf_cfg, "target_accept_r"))
    target_accept_theta = (
        args.target_accept_theta if args.target_accept_theta is not None
        else gcfg_master.get(
            "mcmc_target_accept_theta",
            _required_inference(inf_cfg, "target_accept_theta")))
    phi_step_size = (
        args.phi_step_size if args.phi_step_size is not None
        else _required_inference(inf_cfg, "phi_step_size"))
    reflect_prob = (
        args.reflect_prob if args.reflect_prob is not None
        else _required_inference(inf_cfg, "reflect_prob"))
    init_strategy = str(
        args.init_strategy if args.init_strategy is not None
        else _required_inference(inf_cfg, "init_strategy")).lower()
    if num_chains > 1 and init_strategy != "median":
        fprint(
            f"--num-chains {num_chains} with init_strategy '{init_strategy}': "
            "all chains start from the same initial point with independent "
            "per-chain seeds.")
    init_num_samples = int(_required_inference(inf_cfg, "init_num_samples"))
    latent_burnin = int(
        args.latent_burnin if args.latent_burnin is not None
        else inf_cfg.get("latent_burnin", 5000))
    if latent_burnin > 0 and init_strategy not in ("config", "reid"):
        fprint(
            f"latent burn-in disabled: init_strategy '{init_strategy}' is not "
            "config/reid (theta is held fixed during burn-in).")
        latent_burnin = 0

    fsection(f"Loading {args.galaxy} data")
    use_ecc = args.add_ecc or (
        gcfg_master.get("use_ecc", False) and not args.no_ecc)
    use_qw = args.add_quadratic_warp or (
        gcfg_master.get("use_quadratic_warp", False)
        and not args.no_quadratic_warp)
    data = load_megamaser_spots(
        args.data_root, args.galaxy, v_sys_obs=gcfg_master["v_sys_obs"],
        use_ecc=use_ecc, use_quadratic_warp=use_qw)
    if "D_lo" in gcfg_master and "D_hi" in gcfg_master:
        data["D_lo"] = float(gcfg_master["D_lo"])
        data["D_hi"] = float(gcfg_master["D_hi"])

    config = {
        "inference": {
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "num_chains": int(num_chains),
            "seed": int(seed),
            "max_tree_depth": int(max_tree_depth),
        },
        "model": copy.deepcopy(master_cfg["model"]),
        "io": master_cfg["io"],
    }
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
    if args.da2_prior:
        config["model"]["D_c_prior"] = "volume_D_A"
    if args.single_error_floor:
        config["model"]["use_ngc5765b_clump2_floors"] = False
    elif args.clump2_acceleration_floor_only:
        config["model"]["use_ngc5765b_clump2_floors"] = True
        config["model"]["ngc5765b_clump2_acceleration_only"] = True

    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp)
    tmp.close()
    try:
        model = MaserDiskModel(tmp.name, data)
    finally:
        os.unlink(tmp.name)

    init_key, nudge_key, run_key = random.split(random.PRNGKey(seed), 3)
    init_cfg = _init_block(config["model"]["galaxies"][args.galaxy], model)
    if str(init_strategy).lower() == "median" and num_chains > 1:
        init_params = [
            _make_init(
                model, init_cfg, init_strategy, init_num_samples, key,
                galaxy=args.galaxy, master=master_cfg,
                spot_batch=args.spot_batch)
            for key in random.split(init_key, num_chains)
        ]
    else:
        init_params = _make_init(
            model, init_cfg, init_strategy, init_num_samples, init_key,
            galaxy=args.galaxy, master=master_cfg, spot_batch=args.spot_batch)
    init_params, boundary_adjustments = _nudge_boundary_init(
        model, _h_ref(model), init_params, nudge_key)
    if str(init_strategy).lower() == "median":
        fprint(f"init: median ({init_num_samples} prior samples)")
    else:
        fprint(f"init: {init_strategy}")
    _print_boundary_adjustments(boundary_adjustments)
    if isinstance(init_params, list):
        fprint(f"showing chain 1 of {len(init_params)} initial values")
        _print_init(init_params[0])
    else:
        _print_init(init_params)

    fsection("BlackJAX explicit MCMC")
    backend = jax.default_backend()
    precision = "float64" if jax.config.jax_enable_x64 else "float32"
    fprint(f"JAX backend: {backend}; precision: {precision}")
    fprint(f"spots={model.n_spots}; chains={num_chains}; "
           f"chain_workers<={chain_workers}; warmup={num_warmup}; "
           f"samples={num_samples}; n_inner={n_inner}; "
           f"sample_n_inner={sample_n_inner}; "
           f"latent_burnin={latent_burnin}")
    fprint(f"target_accept_theta={target_accept_theta}; "
           f"target_accept_latent={target_accept_r}")
    fprint(f"transport_systemic_phi={transport_systemic_phi}")
    fprint(f"conditional_mass={conditional_mass}")
    fprint(f"global kernel: NUTS; max_tree_depth={max_tree_depth}")
    fprint(f"phi_step_size={phi_step_size}; reflect_prob={reflect_prob}")
    fprint("phi: sampled explicitly")
    if model.galaxy_name == "NGC5765b":
        r_min, r_max, D_A_ref = _ngc5765b_reference_r_ang_bounds(
            model, _h_ref(model))
        fprint(
            f"r_ang support: fixed [{float(r_min):.4f}, "
            f"{float(r_max):.4f}] mas at D_A_ref={float(D_A_ref):.2f} "
            "Mpc from v_cmb and H0_ref; independent of sampled D_A")
    else:
        fprint("r_ang support: improper flat measure for r_ang > 0")
    fprint(f"mass: {model.mass_parameterization}")

    t0 = time.time()
    result = run_blackjax_mcmc(
        model,
        init_params,
        run_key,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_workers=chain_workers,
        n_inner=n_inner,
        target_accept_theta=target_accept_theta,
        target_accept_latent=target_accept_r,
        theta_step_init=initial_step_size,
        phi_step_init=phi_step_size,
        reflect_prob=reflect_prob,
        max_num_doublings=max_tree_depth,
        num_latent_burnin=latent_burnin,
        sample_n_inner=sample_n_inner,
        transport_systemic_phi=transport_systemic_phi,
        conditional_mass=conditional_mass,
        progress_bar=True,
        jit_steps=True,
    )
    fprint(f"chain execution: {result.chain_method}; "
           f"workers={result.chain_workers}")
    fprint(f"total runner wall time: {time.time() - t0:.1f}s")

    outdir = os.path.join(
        results_path(master_cfg["io"].get("root_output", "results/Maser")),
        args.galaxy)
    os.makedirs(outdir, exist_ok=True)
    suffix = f"blackjax_mcmc_rphi{_variant_suffix(model, args, init_strategy)}"
    outpath = args.output or os.path.join(
        outdir, f"{args.galaxy}_{suffix}.hdf5")
    outpath = os.path.abspath(outpath)
    summary_path = os.path.splitext(outpath)[0] + "_summary.txt"
    report_sections = []
    report_sections.append(_capture_stdout(
        _print_summary, result.samples, result.info, result))

    metadata = {
        "sampler": "blackjax_explicit_mcmc",
        "galaxy": args.galaxy,
        "seed": int(seed),
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "num_chains": int(num_chains),
        "init_strategy": str(init_strategy),
        "chain_method": result.chain_method,
        "chain_workers": int(result.chain_workers),
        "chain_worker_limit": int(chain_workers),
        "n_inner": int(n_inner),
        "sample_n_inner": int(sample_n_inner),
        "transport_systemic_phi": bool(transport_systemic_phi),
        "conditional_mass": bool(conditional_mass),
        "latent_burnin": int(latent_burnin),
        "target_accept_r": float(target_accept_r),
        "initial_step_size": float(initial_step_size),
        "r_parameterization": "log_r_ang_over_seed",
        "phi_parameterization": "explicit_wrapped",
        "mass_parameterization": model.mass_parameterization,
        "D_c_prior": model.D_A_prior,
        "dataset": str(dataset),
        "use_quadratic_warp": bool(model.use_quadratic_warp),
        "use_ecc": bool(model.use_ecc),
        "runtime_seconds": float(result.runtime_seconds),
        "n_spots": int(model.n_spots),
        "target_accept_theta": float(target_accept_theta),
        "max_tree_depth": int(max_tree_depth),
        "phi_step_size": float(phi_step_size),
        "reflect_prob": float(reflect_prob),
        "uniform_da_prior": model.D_A_prior == "uniform_D_A",
        "compute_evidence": bool(args.compute_evidence),
        "save_latents": bool(args.save_latents),
        "precision": precision,
    }
    if hasattr(model, "error_floor_policy"):
        metadata["error_floor_policy"] = str(model.error_floor_policy)
    log_density_flat = np.asarray(result.log_density, dtype=float).reshape(-1)
    log_density_finite = log_density_flat[np.isfinite(log_density_flat)]
    if log_density_finite.size:
        metadata["log_density_max"] = float(np.nanmax(log_density_finite))
        metadata["log_density_median"] = float(
            np.nanmedian(log_density_finite))
    _save_hdf5(outpath, result, metadata, save_latents=args.save_latents)
    fprint(f"saved samples to {outpath}")
    if not args.save_latents:
        fprint("skipped saving per-spot latent samples "
               "(pass --save-latents to keep r_ang/phi in HDF5).")

    log_density_path = os.path.splitext(outpath)[0] + "_log_density.png"
    _make_log_density_plot(result, log_density_path)
    _write_run_summary(summary_path, report_sections)
    if args.compute_evidence:
        _run_evidence_subprocess(
            args.galaxy, outpath, args.data_root, args.spot_batch,
            dataset=dataset)

    corner_path = os.path.splitext(outpath)[0] + "_corner.png"
    corner_unsmoothed_path = (
        os.path.splitext(outpath)[0] + "_corner_unsmoothed.png")
    corner_keys = _corner_keys(result)
    corner_truths = _corner_truths(init_cfg, corner_keys)
    corner_median = _median_point(result, model)
    corner_map = None
    if args.map_overlay:
        try:
            from .maser_map import evaluate_at_globals
            ip = (init_params[0] if isinstance(init_params, list)
                  else init_params)
            map_target = MaserBlackJaxTarget(
                model, _h_ref(model), ip, spot_batch=args.spot_batch)
            de_r = ip.get("r_ang")
            map_res = evaluate_at_globals(
                map_target, {n: ip[n] for n in map_target.names},
                init_r_ang=np.asarray(de_r) if de_r is not None else None,
                marginal=False, verbose=False)
            corner_map = map_res["point"]
            chi2_line = f"MAP (DE globals): chi2_CANDEL={map_res['chi2']:.3f}"
            chi2_line += f"  chi2/dof={map_res['chi2_per_dof']:.3f}"
            fprint(chi2_line)
            # Append to the saved summary so the MCP server can read it back.
            try:
                with open(summary_path, "a", encoding="utf-8") as fh:
                    fh.write("\n" + chi2_line + "\n")
            except OSError:
                pass
        except Exception as exc:                       # noqa: BLE001
            fprint(f"MAP overlay skipped: {exc}")
    made_smooth = _make_corner_plot(
        result.samples, corner_path, corner_keys, smooth=1,
        truths=corner_truths, points=corner_median,
        point_label="MCMC median", truth_label="config init",
        map_point=corner_map, map_label="joint MAP")
    made_unsmoothed = _make_corner_plot(
        result.samples, corner_unsmoothed_path, corner_keys, smooth=0,
        truths=corner_truths, points=corner_median,
        point_label="MCMC median", truth_label="config init",
        map_point=corner_map, map_label="joint MAP")
    if made_smooth is None or made_unsmoothed is None:
        fprint("corner plot could not be generated.")


if __name__ == "__main__":
    main()
