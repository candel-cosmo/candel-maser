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
    master_cfg = tomli.load(f)


# Force float64 for galaxies whose config sets ``force_f64`` (e.g. NGC4258).
# Its extreme geometry (D ~ 7.6 Mpc, ~6 mas radii, ultra-sharp position
# likelihood) makes float32 silently fail: the NUTS warmup step size collapses,
# the dense mass matrix never adapts, and the frozen chain misreports
# near-perfect ESS.  The target galaxy is read from argv before any JAX array
# is created so x64 can be enabled in time.
def _f64_reason_from_argv(argv):
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

from numpyro.distributions import Delta  # noqa: E402

import candel.model.maser_blackjax as maser_blackjax  # noqa: E402
import candel.model.maser_physics as maser_physics  # noqa: E402
from candel.model.maser_blackjax import MaserBlackJaxTarget  # noqa: E402
from candel.model.maser_blackjax import init_from_prior_median  # noqa: E402
from candel.model.maser_blackjax import (  # noqa: E402
    nudge_initial_params_inside_support, run_blackjax_mcmc)
from candel.model.model_H0_maser import MaserDiskModel  # noqa: E402
from candel.pvdata.megamaser_data import load_megamaser_spots  # noqa: E402
from candel.pvdata.megamaser_data import (  # noqa: E402
    megamaser_velocity_frame, v_sys_from_cmb)
from candel.util import data_path, fprint, fsection, results_path  # noqa: E402

# Per-observable noise floors held fixed by --fix-floors-pesce, with units.
_PESCE_FLOOR_UNITS = (("sigma_x_floor", "uas"), ("sigma_y_floor", "uas"),
                      ("sigma_v_sys", "km/s"), ("sigma_v_hv", "km/s"),
                      ("sigma_a_floor", "km/s/yr"))
_PESCE_FLOOR_NAMES = tuple(name for name, _ in _PESCE_FLOOR_UNITS)
_REID_CLIGHT = 2.997925e5


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
    # maser_blackjax imports these constants by value for radius seeds.
    maser_blackjax.C_v = cv
    maser_blackjax.C_a = ca
    fprint("match-reid: using Reid fit_disk physics constants "
           f"(C_v={cv:.6g}, C_a={ca:.6g}, C_g={cg:.6g}, "
           f"c={_REID_CLIGHT:.6g}) + circular-speed SR gamma")


def _h_ref(model):
    return float(model.config["model"].get("H0_ref", 73.0)) / 100.0


def _D_A_from_D_c(model, D_c):
    h = _h_ref(model)
    z_cosmo = model.distance2redshift(jnp.atleast_1d(D_c), h=h).squeeze()
    return D_c / (1.0 + z_cosmo)


def _select_sampler(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sampler", choices=("mcmc", "de"),
                        default="mcmc")
    args, _ = parser.parse_known_args(argv)
    return args.sampler, argv


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
    return init_params


def _paper_init(target, galaxy, master):
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


def _init_block(gal_cfg, model):
    """Variant-specific [init...] block (init_ecc / init_qw / init_ecc_qw)
    selected by use_ecc/use_quadratic_warp, falling back to [init]."""
    parts = []
    if model.use_ecc:
        parts.append("ecc")
    if model.use_quadratic_warp:
        parts.append("qw")
    if parts:
        name = "init_" + "_".join(parts)
        if name in gal_cfg:
            fprint(f"init block: [{name}]")
            return gal_cfg[name]
        fprint(f"init block: [{name}] absent, falling back to [init]")
    return gal_cfg.get("init", {})


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

    r_flat = r_chains.reshape(-1, r_chains.shape[-1])
    neff, rhat = _ess_rhat(r_chains)
    mean = np.mean(r_flat, axis=0)
    sd = np.std(r_flat, axis=0)
    q05, q50, q95 = np.quantile(r_flat, [0.05, 0.50, 0.95], axis=0)
    move = np.mean(np.abs(np.diff(r_chains, axis=1)) > 0, axis=(0, 1))

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

    phi_flat = phi_chains.reshape(-1, phi_chains.shape[-1])
    sin_mean = np.mean(np.sin(phi_flat), axis=0)
    cos_mean = np.mean(np.cos(phi_flat), axis=0)
    centre = np.arctan2(sin_mean, cos_mean)
    centred = centre + np.angle(
        np.exp(1j * (phi_chains - centre[None, None, :])))
    centred_deg = np.degrees(centred.reshape(-1, phi_chains.shape[-1]))
    neff, rhat = _ess_rhat(centred)
    R = np.clip(np.hypot(sin_mean, cos_mean), 1e-300, 1.0)
    mean = np.degrees(centre)
    sd = np.degrees(np.sqrt(-2.0 * np.log(R)))
    q05, q50, q95 = np.quantile(
        centred_deg, [0.05, 0.50, 0.95], axis=0)
    step = np.abs(np.angle(np.exp(1j * np.diff(phi_chains, axis=1))))
    move = np.mean(step > 0, axis=(0, 1))

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
                "sigma_v_sys", "sigma_v_hv", "sigma_a_floor"):
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
            grp.create_dataset(
                key, data=np.asarray(result.samples[key]),
                dtype=np.float32)

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


def _sample_point_at(result, model, idx):
    point = {}
    for key, value in result.samples.items():
        arr = np.asarray(value, dtype=float)
        if key in ("r_ang", "phi"):
            point[key] = arr.reshape(-1, arr.shape[-1])[idx]
        elif arr.ndim <= 2:
            point[key] = float(arr.reshape(-1)[idx])
    return _complete_mass_point(model, point)


def _D_A_from_point(model, point):
    if "D_A" in point:
        return float(np.asarray(point["D_A"]))
    if "D_c" in point:
        return float(_D_A_from_D_c(model, point["D_c"]))
    return None


def _D_c_from_D_A(model, D_A, gcfg):
    D_A = float(D_A)

    def f(D_c):
        return float(_D_A_from_D_c(model, D_c)) - D_A

    lo = float(gcfg.get("D_lo", max(1e-6, 0.25 * D_A)))
    hi = float(gcfg.get("D_hi", max(1.0, 3.0 * D_A)))
    while f(lo) > 0:
        lo *= 0.5
    while f(hi) < 0:
        hi *= 1.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _pesce_reported_point(galaxy, master, model):
    if galaxy == "NGC4258":
        base_init = _clean_init(
            model, master["model"]["galaxies"][galaxy].get("init", {}))
        target = MaserBlackJaxTarget(model, _h_ref(model), base_init)
        try:
            theta = _paper_init(target, galaxy, master)
        except KeyError as exc:
            return None, str(exc)
        point = {
            key: float(np.asarray(value))
            for key, value in theta.items()
            if np.asarray(value).ndim == 0
        }
        D_A = _D_A_from_point(model, point)
        if "log_MBH" not in point and "eta" in point:
            point["log_MBH"] = point["eta"] + np.log10(D_A)
        if "eta" not in point and "log_MBH" in point:
            point["eta"] = point["log_MBH"] - np.log10(D_A)
        return point, "ok"

    path = os.path.join(
        os.path.dirname(__file__), "check_reid", "pesce_disk_params.toml")
    if not os.path.exists(path):
        return None, "missing pesce_disk_params.toml"
    with open(path, "rb") as f:
        src = tomli.load(f).get("galaxies", {})
    if galaxy not in src:
        return None, "no Pesce/Reid row"

    gcfg = master["model"]["galaxies"][galaxy]
    p = src[galaxy]
    D_A = float(p["D_Mpc"])
    log_mbh = float(np.log10(p["MBH_1e7"] * 1.0e7))
    ri = float(gcfg["r_ang_ref_i"])
    rO = float(gcfg["r_ang_ref_Omega"])
    di = float(p.get("didr_deg_mas", 0.0))
    dO = float(p.get("dOmegadr_deg_mas", 0.0))
    v_native = v_sys_from_cmb(
        p["v_cmb_kms"], megamaser_velocity_frame(galaxy),
        gcfg["ra"], gcfg["dec"])
    point = {
        "log_MBH": log_mbh,
        "eta": log_mbh - np.log10(D_A),
        "x0": float(p["x0_mas"]) * 1000.0,
        "y0": float(p["y0_mas"]) * 1000.0,
        "i0": float(p["i0_deg"]) + di * ri,
        "di_dr": di,
        "Omega0": float(p["Omega0_deg"]) + dO * rO,
        "dOmega_dr": dO,
        "dv_sys": float(v_native) - float(gcfg["v_sys_obs"]),
        "sigma_x_floor": float(p["sigma_x_mas"]) * 1000.0,
        "sigma_y_floor": float(p["sigma_y_mas"]) * 1000.0,
        "sigma_v_sys": float(p["sigma_vsys_kms"]),
        "sigma_v_hv": float(p["sigma_vhv_kms"]),
        "sigma_a_floor": float(p["sigma_a_kms_yr"]),
    }
    if model._D_A_uniform:
        point["D_A"] = D_A
    else:
        point["D_c"] = _D_c_from_D_A(model, D_A, gcfg)
    if model.use_quadratic_warp:
        point["d2i_dr2"] = 0.0
        point["d2Omega_dr2"] = 0.0
    if model.use_ecc:
        point["e_x"] = 0.0
        point["e_y"] = 0.0
        point["dperiapsis_dr"] = 0.0
    return point, "ok"


def _score_marginal_point(target, point):
    """Return (logP_2d, logZ_2d) at the given globals: logP_2d = logZ_2d +
    log p(globals); logZ_2d is the data-only 2D (r_ang, phi) marginal."""
    from run_de_map import _logp_2d_terms

    theta = {name: jnp.asarray(point[name]) for name in target.names}
    theta = target.complete_params(theta)
    lp, ll, _, _ = _logp_2d_terms(target, theta)
    lp = float(jax.device_get(jax.block_until_ready(lp)))
    ll = float(jax.device_get(jax.block_until_ready(ll)))
    return lp + ll, ll


def _point_summary(model, gcfg, point):
    D_A = _D_A_from_point(model, point)
    log_mbh = point.get("log_MBH")
    if log_mbh is None:
        log_mbh = float(point["eta"]) + np.log10(D_A)
    return {
        "D_A": D_A,
        "log_MBH": float(log_mbh),
        "Vsys": float(gcfg["v_sys_obs"]) + float(point["dv_sys"]),
    }


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    x = float(x)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3f}"


def _required_inference(cfg, key):
    if key not in cfg:
        raise KeyError(f"Missing [inference].{key} in {_CONFIG_PATH}")
    return cfg[key]


def _print_table(rows, cols):
    widths = {
        key: max(len(label), *(len(_fmt(row.get(key))) for row in rows))
        for key, label in cols
    }
    print("  " + "  ".join(
        label.ljust(widths[key]) for key, label in cols), flush=True)
    print("  " + "  ".join("-" * widths[key] for key, _ in cols),
          flush=True)
    for row in rows:
        print("  " + "  ".join(
            _fmt(row.get(key)).ljust(widths[key]) for key, _ in cols),
            flush=True)


_GRID_KEYS = ("n_r_local", "n_r_global", "n_phi_hv_high", "n_phi_hv_low",
              "n_phi_sys")


def _grid_scaled_target(model, h, init, galaxy, data_root, spot_batch, scale):
    """Rebuild the model + target with the 2D-marginal quadrature grids scaled
    by ``scale`` (e.g. 2), for a convergence check on the DE-style logZ."""
    cfg = copy.deepcopy(model.config)

    def _scale(d):
        for k in _GRID_KEYS:
            if k in d:
                d[k] = int(round(int(d[k]) * scale))

    _scale(cfg["model"])
    _scale(cfg["model"]["galaxies"].get(galaxy, {}))
    gcfg = cfg["model"]["galaxies"][galaxy]
    # Silence the data-load + model-build chatter from the rebuild.
    with contextlib.redirect_stdout(io.StringIO()):
        data = load_megamaser_spots(
            data_root, galaxy, v_sys_obs=gcfg["v_sys_obs"])
        if "D_lo" in gcfg and "D_hi" in gcfg:
            data["D_lo"] = float(gcfg["D_lo"])
            data["D_hi"] = float(gcfg["D_hi"])
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".toml", delete=False)
        tomli_w.dump(cfg, tmp)
        tmp.close()
        try:
            m2 = MaserDiskModel(tmp.name, data)
        finally:
            os.unlink(tmp.name)
        target = MaserBlackJaxTarget(m2, h, init, spot_batch=spot_batch)
    return target


def _add_logZ_2x(rows, model, init, galaxy, data_root, spot_batch):
    """Add a 2x-denser-grid logZ_2d to each scored row as a quadrature
    convergence check.  The doubled-grid model is built once here, at the end,
    and released straight after, so the dense grids never clog memory."""
    scored = [r for r in rows if r.get("_point") is not None
              and r.get("logZ_2d") is not None]
    if not scored:
        fprint("2x-grid logZ check: no successfully scored rows.")
        return
    try:
        fprint("2x-grid logZ check: building doubled quadrature target...")
        target2x = _grid_scaled_target(
            model, _h_ref(model), init, galaxy,
            data_root or data_path("data", "Megamaser"), spot_batch, 2)
    except Exception as exc:                           # diagnostic only
        fprint(f"2x-grid logZ unavailable: {exc}")
        return
    for i, r in enumerate(scored, start=1):
        label = r.get("point", f"row {i}")
        fprint(f"2x-grid logZ check: scoring {label} "
               f"({i}/{len(scored)})...")
        try:
            r["logZ_2d_2x"] = _score_marginal_point(target2x, r["_point"])[1]
        except Exception as exc:                       # diagnostic only
            fprint(f"2x-grid logZ failed for {r['point']}: {exc}")
    del target2x
    fprint("2x-grid logZ check: done.")


_REID_GLOBAL_INIT_KEYS = (
    "D_A", "D_c", "eta", "log_MBH", "dv_sys", "x0", "y0",
    "i0", "di_dr", "d2i_dr2", "Omega0", "dOmega_dr", "d2Omega_dr2",
    "e_x", "e_y", "dperiapsis_dr",
    "sigma_x_floor", "sigma_y_floor", "sigma_v_sys", "sigma_v_hv",
    "sigma_a_floor")


_REID_SCATTER_DRAWS = 20


def _reid_loglik_context(galaxy, n_spots):
    """Import the f2py Reid likelihood and build its data once.

    Returns ``(rp, rr, d)`` or None if the Reid likelihood (reidlik) is not
    built in this environment (e.g. on the cluster) or the spot count differs.
    """
    helper_dir = os.path.join(os.path.dirname(__file__), "check_reid")
    for p in (helper_dir, os.path.join(helper_dir, "reidlik_build")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import prepare_reid_data
        import reid_profile as rp
        import run_reid_mcmc as rr
    except ImportError as exc:
        fprint(f"skipping Reid log-likelihood scatter: {exc}")
        return None

    rp.setup_numbers()
    inp = os.path.join(tempfile.gettempdir(), f"{galaxy}_reid_scatter.inp")
    prepare_reid_data.main([galaxy, "--out", inp])
    d = rp.build_data(inp)
    if d["N"] != int(n_spots):
        fprint(f"skipping Reid scatter: spot count mismatch "
               f"(Reid {d['N']} vs CANDEL {n_spots}).")
        return None
    return rp, rr, d


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

    CANDEL globals map to Reid's convention via check_reid/load_config_init
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
    rp, rr, d = ctx
    r_ang = np.asarray(r_ang, dtype=float)
    init_block = {k: float(point[k]) for k in _REID_GLOBAL_INIT_KEYS
                  if k in point and np.asarray(point[k]).ndim == 0}
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump({"model": {"galaxies": {galaxy: {"init": init_block}}}}, tmp)
    tmp.close()
    try:
        reid_init = rr.load_toml_init(rr.Path(tmp.name), galaxy, 0.0)
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


def _make_reid_loglik_scatter(galaxy, model, target, result, path):
    """Per-spot CANDEL vs Reid data-fit scatter (-0.5*chi^2).

    Compares the per-spot DATA-FIT term -0.5*chi^2 (CANDEL ``_eval_phi_fixed``
    minus its Gaussian normalisation vs Reid's residuals over add_error_floors
    sigmas).  chi^2 is dimensionless: it carries no -0.5*ln(2*pi) per data
    point and no uas-vs-mas position-unit zero-point, so the two codes are
    compared on a true 1:1 line with NO arbitrary additive offset, and the
    global priors play no role.  Evaluated at the same per-spot (r_ang, phi),
    globals, and angular-diameter distance for ``_REID_SCATTER_DRAWS`` randomly
    chosen posterior draws (each a coherent global+latent point), pooling all
    per-spot pairs.
    """
    if "r_ang" not in result.samples or "phi" not in result.samples:
        fprint("skipping Reid scatter: no per-spot (r_ang, phi) in samples.")
        return None
    fprint("Reid log-likelihood scatter: preparing Reid likelihood context...")
    ctx = _reid_loglik_context(galaxy, model.n_spots)
    if ctx is None:
        return None

    n_total = int(np.asarray(result.log_density, dtype=float).reshape(-1).size)
    n_draws = min(_REID_SCATTER_DRAWS, n_total)
    idxs = np.random.default_rng(0).choice(
        n_total, size=n_draws, replace=False)
    fprint(f"Reid log-likelihood scatter: evaluating {n_draws} posterior "
           f"draws ({model.n_spots} spots each)...")

    try:
        from tqdm.auto import tqdm
        draw_iter = tqdm(
            idxs, desc="Reid log-likelihood scatter", unit="draw")
    except Exception:
        draw_iter = idxs

    candel, reid = [], []
    for idx in draw_iter:
        point = _sample_point_at(result, model, int(idx))
        if point is None:
            continue
        r_ang = np.asarray(point["r_ang"], dtype=float)
        phi = np.asarray(point["phi"], dtype=float)
        candel.append(_candel_neg_half_chi2(model, target, point, r_ang, phi))
        D_A = _point_D_A(model, target, point)
        reid.append(_reid_neg_half_chi2(
            ctx, galaxy, point, r_ang, phi, D_A=D_A))
    if not candel:
        fprint("skipping Reid scatter: no usable posterior draws.")
        return None
    candel = np.concatenate(candel)
    reid = np.concatenate(reid)

    ok = np.isfinite(candel) & np.isfinite(reid)
    if int(ok.sum()) < 2:
        fprint("skipping Reid scatter: too few finite per-spot values.")
        return None
    diff = candel[ok] - reid[ok]
    mean = float(np.mean(diff))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    med = float(np.median(diff))
    corr = float(np.corrcoef(candel[ok], reid[ok])[0, 1])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    ax.scatter(reid[ok], candel[ok], s=10, alpha=0.4, edgecolor="none")
    lo = float(min(reid[ok].min(), candel[ok].min()))
    hi = float(max(reid[ok].max(), candel[ok].max()))
    ax.plot([lo, hi], [lo, hi], ls=":", color="0.5", lw=1.0)
    ax.set_xlabel(r"Reid per-spot $-\frac{1}{2}\chi^2$")
    ax.set_ylabel(r"CANDEL per-spot $-\frac{1}{2}\chi^2$")
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.04, 0.96,
            f"mean$={mean:.3f}$\nRMS$={rms:.3f}$\n"
            f"median$={med:.3f}$\n$r={corr:.4f}$\n"
            f"{n_draws} draws, $n={int(ok.sum())}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    fprint("Reid data-fit scatter summary: "
           f"mean(CANDEL-Reid)={mean:.3f}, RMS={rms:.3f}, "
           f"median={med:.3f}, r={corr:.4f}, "
           f"n={int(ok.sum())}, draws={n_draws}")
    fprint(f"saved Reid data-fit (chi^2) scatter to {path} "
           f"({n_draws} draws)")
    return path


def _print_point_comparison(galaxy, master, model, result, init_cfg,
                            init_params, spot_batch, data_root=None,
                            scatter_path=None, compare_reid_2x=False):
    target_init = (
        init_params[0] if isinstance(init_params, list) else init_params)
    target = MaserBlackJaxTarget(
        model, _h_ref(model), target_init, spot_batch=spot_batch)
    fsection("MCMC/Pesce comparison setup")
    fprint("Preparing slow Reid/Pesce comparison diagnostics. "
           "The final table is printed after all rows are scored.")

    # Make the data-loglik scatter first, so it is always produced even if the
    # expensive logZ table below is slow, interrupted, or errors.
    if scatter_path is not None:
        try:
            _make_reid_loglik_scatter(galaxy, model, target, result,
                                      scatter_path)
        except Exception as exc:                       # never abort the run
            fprint(f"Reid log-likelihood scatter failed: {exc}")

    gcfg = master["model"]["galaxies"][galaxy]
    rows = []

    def add(label, point):
        row = {"point": label}
        if point is None:
            fprint(f"comparison row {label}: missing.")
            row["status"] = "missing"
            rows.append(row)
            return
        try:
            fprint(f"comparison row {label}: scoring production 2D grid...")
            row.update(_point_summary(model, gcfg, point))
            row["logp"], row["logZ_2d"] = _score_marginal_point(target, point)
            row["_point"] = point
            fprint(f"comparison row {label}: logP_2d={row['logp']:.3f}, "
                   f"logZ_2d={row['logZ_2d']:.3f}")
        except Exception as exc:
            row["status"] = str(exc)
            fprint(f"comparison row {label}: failed ({exc})")
        rows.append(row)

    pesce_point, pesce_status = _pesce_reported_point(galaxy, master, model)
    add("Pesce/Reid", pesce_point)
    if pesce_status != "ok":
        rows[-1]["status"] = pesce_status

    config_point = None
    if init_cfg:
        config_point = _complete_mass_point(
            model, {k: np.asarray(v) for k, v in
                    _clean_init(model, init_cfg).items()})
    add("config init", config_point)

    median_point = _median_point(result, model)
    add("MCMC median", median_point)

    if compare_reid_2x:
        _add_logZ_2x(rows, model, target_init, galaxy, data_root, spot_batch)

    base = next((row.get("logp") for row in rows
                 if row.get("logp") is not None
                 and np.isfinite(row["logp"])), None)
    if base is not None:
        for row in rows:
            if row.get("logp") is not None:
                row["dlogp"] = row["logp"] - base

    fsection("MCMC/Pesce comparison")
    msg = ("logP_2d = logZ_2d + log p(globals) is the 2D-marginal "
           "objective (each spot marginalised over r_ang, phi); logZ_2d is "
           "the data-only marginal")
    if compare_reid_2x:
        msg += ", and logZ_2d(2x) repeats it on a 2x-denser grid"
    fprint(msg + ". dlogP is relative to the first finite row; positive is "
           "preferred over that baseline.")
    cols = [
        ("point", "point"),
        ("logp", "logP_2d"),
        ("dlogp", "dlogP"),
        ("logZ_2d", "logZ_2d"),
        ("D_A", "D_A"),
        ("log_MBH", "log_MBH"),
        ("Vsys", "Vsys"),
        ("status", "status"),
    ]
    if compare_reid_2x:
        cols.insert(4, ("logZ_2d_2x", "logZ_2d(2x)"))
    _print_table(rows, cols)


def _variant_suffix(model, args, init_strategy):
    parts = []
    if model.use_ecc:
        parts.append("ecc")
    if model.use_quadratic_warp:
        parts.append("qw")
    # Flags that change the sampled posterior but are not disc geometry: keep
    # them in the filename so runs varying them don't overwrite each other.
    if args.match_reid:
        parts.append("matchreid")
    if args.fix_floors_pesce:
        parts.append("fixfloors")
    # Always tag the resolved init strategy (median/config/reid) last, so runs
    # that differ only by initialisation stay apart and glob cleanly.
    parts.append(f"init{init_strategy}")
    return "_" + "_".join(parts)


def _run_evidence_subprocess(galaxy, chain_path, data_root, spot_batch):
    """Compute the single-galaxy harmonic evidence on the saved chain.

    Spawned as a separate process so the float64 reference grid (a global JAX
    flag) is genuinely double precision even when the chain was sampled in
    float32.  Best-effort: a failure here never aborts the sampling run.
    """
    import subprocess

    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "evidence_single_galaxy.py")
    cmd = [sys.executable, "-u", script, galaxy, "--chain", chain_path]
    if data_root:
        cmd += ["--data-root", data_root]
    if spot_batch is not None:
        cmd += ["--spot-batch", str(spot_batch)]
    fsection("Single-galaxy evidence (harmonic)")
    fprint("running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:                            # never abort the run
        fprint(f"evidence computation failed: {exc}")


def main(argv=None):
    sampler, argv = _select_sampler(argv)

    if sampler == "de":
        de_dir = os.path.dirname(os.path.abspath(__file__))
        if de_dir not in sys.path:
            sys.path.insert(0, de_dir)
        import run_de_map
        return run_de_map.main(_strip_sampler_arg(argv))

    argv = _strip_sampler_arg(argv)
    parser = argparse.ArgumentParser(
        description="Run one megamaser disk inference job.")
    parser.add_argument("galaxy", type=str)
    parser.add_argument("--data-root", type=str,
                        default=data_path("data", "Megamaser"))
    parser.add_argument("--num-warmup", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None,
                        help="Number of independent chains to run "
                             "sequentially in this job.")
    parser.add_argument("--init-strategy",
                        choices=("median", "config", "reid"),
                        default=None,
                        help="Initial global point strategy. Default: "
                             "config inference/init_strategy. 'reid' uses "
                             "reported Pesce/Reid globals; for NGC4258 it "
                             "reads reid_ngc4258_best.toml. Multi-chain jobs "
                             "require median.")
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
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument("--no-ecc", action="store_true")
    parser.add_argument("--add-ecc", action="store_true")
    parser.add_argument("--no-quadratic-warp", action="store_true")
    parser.add_argument("--add-quadratic-warp", action="store_true")
    parser.add_argument("--mass-parameterization",
                        choices=("eta", "log_mbh"), default=None,
                        help="Global mass coordinate for the sampler. "
                             "Default: config value, eta in config_maser.")
    parser.add_argument("--f64", action="store_true", default=_ENABLE_F64)
    parser.add_argument("--fix-floors-pesce", action="store_true",
                        help="Hold the five error floors (sigma_x_floor, "
                             "sigma_y_floor, sigma_v_sys, sigma_v_hv, "
                             "sigma_a_floor) fixed at the published "
                             "Pesce/Reid values; the MCMC samples all globals "
                             "the per-spot (r, phi) latents.")
    parser.add_argument("--match-reid", action="store_true",
                        help="Diagnostic mode: use Reid fit_disk physical "
                             "constants and circular-speed SR gamma "
                             "(eccentric branch) in CANDEL. Reid scatter "
                             "comparisons "
                             "always evaluate both codes at the same D_A.")
    parser.add_argument("--compare-reid", action="store_true",
                        help="After sampling, run the expensive Pesce/Reid/"
                             "config/MCMC fixed-global comparison scored with "
                             "the DE-style 2D marginal likelihood. Disabled "
                             "by default.")
    parser.add_argument("--compare-reid-2x", action="store_true",
                        help="With --compare-reid, also repeat logZ_2d on a "
                             "2x-denser quadrature grid. Disabled by default.")
    parser.add_argument("--compute-evidence", action="store_true",
                        help="After sampling, compute the single-galaxy "
                             "harmonic evidence. Disabled by default.")
    parser.add_argument("--save-latents", action="store_true",
                        help="Write per-spot r_ang/phi samples to HDF5. "
                             "Disabled by default; globals, log_density, and "
                             "sampler_parameters/latent_scale are still "
                             "saved.")
    parser.add_argument("--map-overlay", dest="map_overlay",
                        action="store_true", default=True,
                        help="Overlay the joint MAP (from the DE/config init) "
                             "on the corner plot. On by default.")
    parser.add_argument("--no-map-overlay", dest="map_overlay",
                        action="store_false",
                        help="Disable the MAP overlay on the corner plot.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)
    if args.compare_reid_2x and not args.compare_reid:
        raise SystemExit("--compare-reid-2x requires --compare-reid.")

    galaxies = master_cfg["model"]["galaxies"]
    if args.galaxy not in galaxies:
        raise SystemExit(
            f"Unknown galaxy {args.galaxy!r}. Available: {list(galaxies)}")
    if args.match_reid:
        _apply_reid_physics_constants()

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
    if num_chains < 1:
        raise SystemExit("--num-chains must be >= 1.")
    initial_step_size = (
        args.initial_step_size if args.initial_step_size is not None
        else _required_inference(inf_cfg, "blackjax_initial_step_size"))
    max_tree_depth = (
        args.max_tree_depth if args.max_tree_depth is not None
        else _required_inference(inf_cfg, "max_tree_depth"))
    n_inner = (
        args.n_inner if args.n_inner is not None
        else _required_inference(inf_cfg, "n_inner"))
    target_accept_r = (
        args.target_accept_r if args.target_accept_r is not None
        else _required_inference(inf_cfg, "target_accept_r"))
    target_accept_theta = (
        args.target_accept_theta if args.target_accept_theta is not None
        else _required_inference(inf_cfg, "target_accept_theta"))
    phi_step_size = (
        args.phi_step_size if args.phi_step_size is not None
        else _required_inference(inf_cfg, "phi_step_size"))
    reflect_prob = (
        args.reflect_prob if args.reflect_prob is not None
        else _required_inference(inf_cfg, "reflect_prob"))
    init_strategy = str(args.init_strategy or _required_inference(
        inf_cfg, "init_strategy")).lower()
    if num_chains > 1 and init_strategy != "median":
        fprint(
            f"--num-chains {num_chains} with init_strategy '{init_strategy}': "
            "all chains start from the same point with independent per-chain "
            "seeds (use --init-strategy median for overdispersed starts).")
    init_num_samples = int(_required_inference(inf_cfg, "init_num_samples"))
    latent_burnin = int(
        args.latent_burnin if args.latent_burnin is not None
        else inf_cfg.get("latent_burnin", 5000))
    if latent_burnin > 0 and init_strategy not in ("config", "reid"):
        fprint(
            f"latent burn-in disabled: init_strategy '{init_strategy}' is not "
            "config/reid (theta is held fixed during burn-in).")
        latent_burnin = 0

    gcfg_master = galaxies[args.galaxy]
    fsection(f"Loading {args.galaxy} data")
    data = load_megamaser_spots(
        args.data_root, args.galaxy, v_sys_obs=gcfg_master["v_sys_obs"])
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

    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp)
    tmp.close()
    try:
        model = MaserDiskModel(tmp.name, data)
    finally:
        os.unlink(tmp.name)

    floor_point = None
    if args.fix_floors_pesce:
        # Delta priors are dropped from the sampled sites (see
        # _theta_site_prior_pairs) and supplied as fixed constants by
        # phys_from_params_jax; set them before the sampler reads the sites.
        floor_point, status = _pesce_reported_point(
            args.galaxy, master_cfg, model)
        if floor_point is None:
            raise SystemExit(
                f"--fix-floors-pesce needs Pesce floors: {status}")
        for name in _PESCE_FLOOR_NAMES:
            model.priors[name] = Delta(jnp.asarray(float(floor_point[name])))
        fsection(f"Error floors fixed at Pesce/Reid values ({args.galaxy})")
        for name, unit in _PESCE_FLOOR_UNITS:
            fprint(f"  {name:16s} = {float(floor_point[name]):8.4g} {unit}")
        fprint("  held fixed; dropped from the sampled sites")

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
    if floor_point is not None:
        # Floors are not sampled; reflect their fixed values in the printed
        # init so the dump is consistent with the held-fixed section above.
        floors = {n: jnp.asarray(float(floor_point[n]))
                  for n in _PESCE_FLOOR_NAMES}
        for ip in (init_params if isinstance(init_params, list)
                   else [init_params]):
            ip.update(floors)
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
    fprint(f"spots={model.n_spots}; chains={num_chains}; warmup={num_warmup}; "
           f"samples={num_samples}; n_inner={n_inner}; "
           f"latent_burnin={latent_burnin}")
    fprint(f"target_accept_theta={target_accept_theta}; "
           f"target_accept_latent={target_accept_r}")
    fprint(f"global kernel: NUTS; max_tree_depth={max_tree_depth}")
    fprint(f"phi_step_size={phi_step_size}; reflect_prob={reflect_prob}")
    fprint("phi: sampled explicitly")
    fprint("r_ang: sampled as z_r = log(r_ang / r_hat(theta))")
    fprint(f"mass: {model.mass_parameterization}")

    t0 = time.time()
    result = run_blackjax_mcmc(
        model,
        init_params,
        run_key,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        n_inner=n_inner,
        target_accept_theta=target_accept_theta,
        target_accept_latent=target_accept_r,
        theta_step_init=initial_step_size,
        phi_step_init=phi_step_size,
        reflect_prob=reflect_prob,
        max_num_doublings=max_tree_depth,
        num_latent_burnin=latent_burnin,
        progress_bar=True,
        jit_steps=True,
    )
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
        "chain_method": "sequential" if num_chains > 1 else "single",
        "n_inner": int(n_inner),
        "latent_burnin": int(latent_burnin),
        "target_accept_r": float(target_accept_r),
        "initial_step_size": float(initial_step_size),
        "r_parameterization": "log_r_ang_over_seed",
        "phi_parameterization": "explicit_wrapped",
        "mass_parameterization": model.mass_parameterization,
        "D_c_prior": "uniform_D_A",
        "use_quadratic_warp": bool(model.use_quadratic_warp),
        "use_ecc": bool(model.use_ecc),
        "runtime_seconds": float(result.runtime_seconds),
        "n_spots": int(model.n_spots),
        "target_accept_theta": float(target_accept_theta),
        "max_tree_depth": int(max_tree_depth),
        "phi_step_size": float(phi_step_size),
        "reflect_prob": float(reflect_prob),
        "fix_floors_pesce": bool(args.fix_floors_pesce),
        "uniform_da_prior": True,
        "match_reid": bool(args.match_reid),
        "compare_reid": bool(args.compare_reid),
        "compare_reid_2x": bool(args.compare_reid_2x),
        "compute_evidence": bool(args.compute_evidence),
        "save_latents": bool(args.save_latents),
    }
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
    if args.compare_reid:
        scatter_path = (
            os.path.splitext(outpath)[0] + "_reid_loglik_scatter.png")
        report_sections.append(_capture_stdout(
            _print_point_comparison,
            args.galaxy, master_cfg, model, result, init_cfg, init_params,
            args.spot_batch, data_root=args.data_root,
            scatter_path=scatter_path, compare_reid_2x=args.compare_reid_2x))
    else:
        text = ("skipping Pesce/Reid fixed-global comparison "
                "(pass --compare-reid to enable).")
        fprint(text)
        report_sections.append("\nMCMC/Pesce comparison\n  " + text + "\n")
    _write_run_summary(summary_path, report_sections)
    if args.compute_evidence:
        _run_evidence_subprocess(
            args.galaxy, outpath, args.data_root, args.spot_batch)

    corner_path = os.path.splitext(outpath)[0] + "_corner.png"
    corner_unsmoothed_path = (
        os.path.splitext(outpath)[0] + "_corner_unsmoothed.png")
    corner_keys = _corner_keys(result)
    corner_truths = _corner_truths(init_cfg, corner_keys)
    corner_median = _median_point(result, model)
    corner_map = None
    if args.map_overlay:
        try:
            from candel.model.maser_map import evaluate_at_globals
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
            # Compute the MAP chi^2 both ways: CANDEL and the original Reid
            # Fortran (reidlik), at the same globals + optimised latents.
            chi2_line = f"MAP (DE globals): chi2_CANDEL={map_res['chi2']:.3f}"
            ctx = _reid_loglik_context(args.galaxy, model.n_spots)
            if ctx is not None:
                reid_nh = _reid_neg_half_chi2(
                    ctx, args.galaxy, map_res["point"], map_res["r_ang"],
                    map_res["phi"], D_A=map_res.get("D_A"))
                chi2_reid = float(-2.0 * np.asarray(reid_nh).sum())
                rel = 100.0 * abs(map_res["chi2"] - chi2_reid) / chi2_reid
                chi2_line += (f"  chi2_Reid_code={chi2_reid:.3f}  "
                              f"(rel {rel:.2f}%)")
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
