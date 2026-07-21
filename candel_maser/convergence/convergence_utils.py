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
"""Brute-force reference kernels for maser disk quadrature diagnostics.

The retained convergence checks compare the production fixed-``r_ang`` phi
marginal against a full-2pi phi reference. The helpers batch over the spot
axis so the intermediate fits on a 12 GB GPU.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from candel.model.integration import trapz_log_weights
from candel.util import get_nested


def cast_floats(x, dtype):
    """Cast floating arrays/scalars in a small pytree to ``dtype``."""
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: cast_floats(v, dtype) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(cast_floats(v, dtype) for v in x)
    if isinstance(x, list):
        return [cast_floats(v, dtype) for v in x]
    arr = jnp.asarray(x)
    if jnp.issubdtype(arr.dtype, jnp.floating):
        return arr.astype(dtype)
    return x


def cast_model_floats(model, dtype):
    """Cast cached floating arrays in a diagnostic model in place."""
    for name, value in list(vars(model).items()):
        if isinstance(value, (np.ndarray, jax.Array)):
            arr = jnp.asarray(value)
            if jnp.issubdtype(arr.dtype, jnp.floating):
                setattr(model, name, arr.astype(dtype))
    model._phi_concat = cast_floats(model._phi_concat, dtype)
    return model


@partial(jax.jit, static_argnames=("model", "has_any_accel"))
def _dense_fixed_r_group(model, idx, r_ang, phys_args, phys_kw,
                         sin_phi, cos_phi, log_w_phi, *, has_any_accel):
    """Fixed-shape dense-phi reference for padded spot batches."""
    def body(_, values):
        idx_b, r_b = values
        r_pre = model._r_precompute(
            r_b, idx_b, *phys_args, **phys_kw,
            has_any_accel=has_any_accel)
        nhc = model._phi_eval(r_pre, sin_phi, cos_phi)
        ll = logsumexp(nhc + log_w_phi, axis=-1)
        return None, r_pre["lnorm"] + r_pre["lnorm_a"] + ll

    _, out = jax.lax.scan(body, None, (idx, r_ang))
    return out.reshape(-1)


@partial(jax.jit, static_argnames=("model", "has_any_accel"))
def _dense_conditional_r_group(model, idx, r_ang, log_w_r,
                               phys_args, phys_kw,
                               sin_phi, cos_phi, log_w_phi, *,
                               has_any_accel):
    """Fixed-shape dense-phi reference on frozen conditional-r nodes."""
    def body(_, values):
        idx_b, r_b, log_w_r_b = values
        r_pre = model._r_precompute(
            r_b, idx_b, *phys_args, **phys_kw,
            has_any_accel=has_any_accel)
        nhc = model._phi_eval(r_pre, sin_phi, cos_phi)
        ll_phi = logsumexp(nhc + log_w_phi, axis=-1)
        ll = logsumexp(ll_phi + log_w_r_b, axis=-1)
        return None, r_pre["lnorm"] + r_pre["lnorm_a"] + ll

    _, out = jax.lax.scan(body, None, (idx, r_ang, log_w_r))
    return out.reshape(-1)


def _padded_group_chunks(idx, values, batch):
    """Pad a spot group by cycling real rows, then split fixed chunks."""
    idx = np.asarray(idx, dtype=np.int32)
    n = len(idx)
    batch = min(max(1, int(batch)), n)
    n_pad = (-n) % batch
    positions = np.arange(n + n_pad) % n
    idx_chunks = idx[positions].reshape(-1, batch)
    value_chunks = [
        np.asarray(value)[idx][positions].reshape(
            -1, batch, *np.asarray(value).shape[1:])
        for value in values
    ]
    return idx_chunks, value_chunks, n


def dense_phi_reference_per_spot(model, phys_args, phys_kw, r_ang,
                                 n_phi, spot_batch, log_w_r=None,
                                 partition_support=False):
    """Per-spot dense phi reference on fixed radial nodes.

    ``r_ang`` is either ``(n_spots,)`` for the fixed-radius profile or
    ``(n_spots, n_r)`` for the conditional-radius profile.  The latter also
    requires matching per-spot ``log_w_r``.  Spot batches are padded and run
    through ``lax.scan`` so every compiled iteration has one static shape.
    The default retains the older full-2pi leakage diagnostic;
    ``partition_support=True`` instead uses the exact half-plane support of
    peak-partition for an apples-to-apples integrator comparison.
    """
    r_ang = np.asarray(r_ang)
    if r_ang.shape[0] != model.n_spots:
        raise ValueError(
            f"r_ang first axis must have {model.n_spots} spots; "
            f"got {r_ang.shape}.")
    conditional = log_w_r is not None
    if conditional and np.asarray(log_w_r).shape != r_ang.shape:
        raise ValueError("log_w_r must match the conditional r_ang shape.")
    if not conditional and r_ang.ndim != 1:
        raise ValueError("fixed-r r_ang must be one-dimensional.")

    dtype = jnp.asarray(phys_args[2]).dtype
    out = np.empty(model.n_spots, dtype=np.float64)

    for type_key, idx in (("sys", model._idx_sys),
                          ("red", model._idx_red),
                          ("blue", model._idx_blue)):
        if not int(idx.shape[0]):
            continue
        values = [r_ang]
        if conditional:
            values.append(log_w_r)
        idx_chunks, chunks, n = _padded_group_chunks(
            idx, values, spot_batch)
        common = (
            model, jnp.asarray(idx_chunks), jnp.asarray(chunks[0], dtype),
        )
        kwargs = {"has_any_accel": model._group_has_any_accel(type_key)}
        if not partition_support:
            ranges = ((0.0, 2.0 * np.pi),)
        elif type_key == "sys":
            ranges = ((-np.pi, 0.0), (0.0, np.pi))
        else:
            subs = model._phi_subranges[type_key]
            ranges = ((subs[0][0], subs[-1][1]),)
        range_values = []
        for phi_lo, phi_hi in ranges:
            phi = jnp.linspace(
                phi_lo, phi_hi, int(n_phi), dtype=dtype)
            sin_phi = jnp.sin(phi)
            cos_phi = jnp.cos(phi)
            log_w_phi = trapz_log_weights(phi)
            if conditional:
                ll = _dense_conditional_r_group(
                    *common, jnp.asarray(chunks[1], dtype),
                    phys_args, phys_kw, sin_phi, cos_phi, log_w_phi, **kwargs)
            else:
                ll = _dense_fixed_r_group(
                    *common, phys_args, phys_kw,
                    sin_phi, cos_phi, log_w_phi, **kwargs)
            range_values.append(np.asarray(
                jax.device_get(jax.block_until_ready(ll)))[:n])
        group_ll = range_values[0]
        for values in range_values[1:]:
            group_ll = np.logaddexp(group_ll, values)
        out[np.asarray(idx)] = group_ll
    return out


def dense_r_phi_reference_per_spot(model, phys_args, phys_kw, n_r, n_phi,
                                   r_chunk, spot_batch,
                                   partition_support=True):
    """Per-spot float64 reference on a full-support log-r x phi grid."""
    n_r, n_phi = int(n_r), int(n_phi)
    r_chunk = int(r_chunk)
    if n_r < 3 or n_phi < 3 or r_chunk < 1:
        raise ValueError("n_r and n_phi must be >= 3; r_chunk must be positive.")

    dtype = jnp.asarray(phys_args[2]).dtype
    if dtype != jnp.float64:
        raise ValueError(
            "the full r x phi acceptance reference requires float64.")
    r_min, r_max = model.r_ang_range(phys_args[2])
    log_r = jnp.linspace(jnp.log(r_min), jnp.log(r_max), n_r, dtype=dtype)
    r_grid = jnp.exp(log_r)
    log_w_r = trapz_log_weights(r_grid)
    total = np.full(model.n_spots, -np.inf, dtype=np.float64)
    for start in range(0, n_r, r_chunk):
        stop = min(start + r_chunk, n_r)
        shape = (model.n_spots, stop - start)
        r_values = np.broadcast_to(
            np.asarray(r_grid[start:stop]), shape)
        log_weights = np.broadcast_to(
            np.asarray(log_w_r[start:stop]), shape)
        partial = dense_phi_reference_per_spot(
            model, phys_args, phys_kw, r_values, n_phi, spot_batch,
            log_w_r=log_weights, partition_support=partition_support)
        total = np.logaddexp(total, partial)
    return total


def bruteforce_ll_fixed_r(model, phys_args, phys_kw, r_ang, ref_cfg):
    """Per-type full-2π φ brute force at a fixed r_ang vector.

    r_ang: shape (n_spots,) in mas.
    ref_cfg: dict with keys n_phi, spot_batch. Runs at the dtype of the
        supplied model parameters.
    Returns dict with keys 'sys', 'red', 'blue', 'total'.
    """
    ll = dense_phi_reference_per_spot(
        model, phys_args, phys_kw, r_ang,
        int(ref_cfg["n_phi"]), int(ref_cfg["spot_batch"]))
    out = {}
    for key, idx in (("sys", model._idx_sys),
                     ("red", model._idx_red),
                     ("blue", model._idx_blue)):
        out[key] = float(np.sum(ll[np.asarray(idx)]))
    out["total"] = out["sys"] + out["red"] + out["blue"]
    return out


# -----------------------------------------------------------------------
# Test-harness helpers (used by the sweep scripts in scripts/megamaser/).
# They build a MaserDiskModel with per-call grid overrides so the sweep
# can vary phi/r grid sizes while holding all other config constant.
# -----------------------------------------------------------------------

def build_model(galaxy, master_cfg, dtype=None, **overrides):
    """Build a MaserDiskModel with global [model] keys overridden.

    Any recognised [model] key may be passed (n_phi_hv_high, n_phi_hv_low,
    n_phi_sys, phi_hv_inner_deg, phi_hv_outer_deg, phi_sys_ranges_deg,
    n_r_local, n_r_global, K_sigma, refine_r_center,
    conditional_spot_batch, ...). Per-galaxy settings in the config normally
    override globals; for the convergence tests we want the GLOBAL values to
    win, so we temporarily strip the galaxy's grid keys from the config copy
    passed to the model.
    """
    import os
    import tempfile

    import tomli_w

    from candel.pvdata.megamaser_data import load_megamaser_spots

    cfg = {k: (v.copy() if isinstance(v, dict) else v)
           for k, v in master_cfg.items()}
    cfg["model"] = dict(master_cfg["model"])
    cfg["model"]["galaxies"] = {
        g: dict(blk) for g, blk in master_cfg["model"]["galaxies"].items()}
    gblk = cfg["model"]["galaxies"][galaxy]

    for key in ("n_phi_hv_high", "n_phi_hv_low", "n_phi_sys",
                "phi_hv_inner_deg", "phi_hv_outer_deg",
                "phi_sys_ranges_deg",
                "n_r_local", "n_r_global", "K_sigma",
                "refine_r_center", "n_refine_steps",
                "conditional_spot_batch"):
        gblk.pop(key, None)

    for k, v in overrides.items():
        cfg["model"][k] = v

    data = load_megamaser_spots(
        master_cfg["io"]["maser_data"]["root"], galaxy=galaxy,
        v_sys_obs=master_cfg["model"]["galaxies"][galaxy]["v_sys_obs"])
    for key in ("D_lo", "D_hi"):
        if key in master_cfg["model"]["galaxies"][galaxy]:
            data[key] = float(master_cfg["model"]["galaxies"][galaxy][key])
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(cfg, tmp)
    tmp.close()
    from candel.model.model_H0_maser import MaserDiskModel
    model = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)
    if dtype is not None:
        cast_model_floats(model, dtype)
    return model


def resolve_grid_for_galaxy(master_cfg, galaxy, profile):
    """Return the phi/r grid used by a diagnostic profile.

    ``profile`` is a label used only by the convergence scripts. The current
    config has one generic phi/r grid plus per-galaxy overrides, so both
    retained profiles resolve through the same production-style hierarchy.
    """
    if profile not in ("fixed_r", "conditional_r"):
        raise ValueError(
            "profile must be 'fixed_r' or 'conditional_r'; "
            f"got {profile!r}")
    gal_cfg = master_cfg["model"]["galaxies"][galaxy]

    def _phi(key):
        if key in gal_cfg:
            return int(gal_cfg[key])
        return int(get_nested(master_cfg, f"model/{key}"))

    def _r(key):
        if key in gal_cfg:
            return int(gal_cfg[key])
        return int(get_nested(master_cfg, f"model/{key}"))

    return dict(
        n_hv_high=_phi("n_phi_hv_high"),
        n_hv_low=_phi("n_phi_hv_low"),
        n_sys=_phi("n_phi_sys"),
        n_r_local=_r("n_r_local"),
        n_r_global=_r("n_r_global"),
    )


# -----------------------------------------------------------------------
# AD-friendly kernels for the summed-gradient convergence tests.
# -----------------------------------------------------------------------

# Parameters differentiated by the gradient convergence checks. Galaxies
# with use_quadratic_warp or use_ecc extend this list via
# ``extend_grad_params``.
GRAD_PARAMS_BASE = (
    "H0", "D_c", "log_MBH", "x0", "y0", "dv_sys",
    "i0", "di_dr", "Omega0", "dOmega_dr",
    "sigma_x_floor", "sigma_y_floor",
    "sigma_v_sys", "sigma_v_hv", "sigma_a_floor",
)


def extend_grad_params(model, sample):
    """Return GRAD_PARAMS_BASE extended with the optional-feature
    parameters present in ``sample`` (quadratic warp, eccentricity)."""
    keys = list(GRAD_PARAMS_BASE)
    if model.use_quadratic_warp:
        for k in ("d2i_dr2", "d2Omega_dr2"):
            if k in sample:
                keys.append(k)
    if model.use_ecc:
        for k in ("ecc", "periapsis", "e_x", "e_y", "dperiapsis_dr"):
            if k in sample:
                keys.append(k)
    return tuple(keys)


def _sample_dtype(sample):
    for v in sample.values():
        arr = jnp.asarray(v)
        if jnp.issubdtype(arr.dtype, jnp.floating):
            return arr.dtype
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def ensure_grad_sample(model, init_block, dtype=None):
    """Populate a jnp-typed sample dict with every parameter used by
    the grad check, filling absent entries with sensible defaults so
    ``jax.grad`` produces a meaningful partial for every key.

    ``H0`` defaults to ``model/H0_ref`` (matches ``jax_phys_from_sample``
    and ``phys_from_sample``); other missing entries default to 0.0,
    which is a valid neighbourhood for the remaining parameters
    (Cartesian offsets, warp rates, noise floors — all small).
    """
    dtype = dtype or (jnp.float64 if jax.config.jax_enable_x64
                      else jnp.float32)
    sample = {k: jnp.asarray(float(v), dtype=dtype)
              for k, v in init_block.items()}
    H0_ref = float(get_nested(model.config, "model/H0_ref", 73.0))
    defaults = {"H0": H0_ref}
    for k in GRAD_PARAMS_BASE:
        sample.setdefault(k, jnp.asarray(defaults.get(k, 0.0), dtype=dtype))
    if model.use_quadratic_warp:
        for k in ("d2i_dr2", "d2Omega_dr2"):
            sample.setdefault(k, jnp.asarray(0.0, dtype=dtype))
    if model.use_ecc:
        if model.ecc_cartesian:
            for k in ("e_x", "e_y"):
                sample.setdefault(k, jnp.asarray(0.0, dtype=dtype))
        else:
            for k in ("ecc", "periapsis"):
                sample.setdefault(k, jnp.asarray(0.0, dtype=dtype))
        sample.setdefault("dperiapsis_dr", jnp.asarray(0.0, dtype=dtype))
    return sample


def jax_phys_from_sample(model, sample):
    """JAX-traceable counterpart of ``MaserDiskModel.phys_from_sample``.

    All non-derived quantities flow through as jnp scalars so jax.grad
    can differentiate w.r.t. any entry of ``sample``.
    """
    dtype = _sample_dtype(sample)

    def g(key, default=None):
        if key in sample:
            return sample[key]
        if default is not None:
            return jnp.asarray(default, dtype=dtype)
        raise KeyError(f"missing '{key}' in sample")

    H0_ref = float(get_nested(model.config, "model/H0_ref", 73.0))
    h = g("H0", H0_ref) / 100.0
    D_c = g("D_c")
    z_cosmo = model.distance2redshift(
        jnp.atleast_1d(D_c), h=h).squeeze()
    D_A = D_c / (1.0 + z_cosmo)
    M_BH = 10.0 ** (g("log_MBH") - 7.0)
    v_sys = model.v_sys_obs + g("dv_sys", 0.0)

    phys_args = (
        g("x0"), g("y0"),
        D_A, M_BH, v_sys,
        jnp.asarray(model._r_ang_ref_i, dtype=dtype),
        jnp.asarray(model._r_ang_ref_Omega, dtype=dtype),
        jnp.asarray(model._r_ang_ref_periapsis, dtype=dtype),
        jnp.deg2rad(g("i0")),
        jnp.deg2rad(g("di_dr")),
        jnp.deg2rad(g("Omega0")),
        jnp.deg2rad(g("dOmega_dr")),
        g("sigma_x_floor") ** 2,
        g("sigma_y_floor") ** 2,
        g("sigma_v_sys") ** 2,
        g("sigma_v_hv") ** 2,
        g("sigma_a_floor") ** 2,
    )
    phys_kw = {}
    if model.use_quadratic_warp:
        phys_kw["d2i_dr2"] = jnp.deg2rad(g("d2i_dr2", 0.0))
        phys_kw["d2Omega_dr2"] = jnp.deg2rad(g("d2Omega_dr2", 0.0))
    if model.use_ecc:
        if "e_x" in sample or "e_y" in sample or model.ecc_cartesian:
            phys_kw["e_x"] = g("e_x", 0.0)
            phys_kw["e_y"] = g("e_y", 0.0)
        else:
            ecc = g("ecc", 0.0)
            peri = jnp.deg2rad(g("periapsis", 0.0))
            phys_kw["e_x"] = ecc * jnp.cos(peri)
            phys_kw["e_y"] = ecc * jnp.sin(peri)
        phys_kw["dperiapsis_dr"] = jnp.deg2rad(g("dperiapsis_dr", 0.0))
    return phys_args, phys_kw


def grad_diff_report(grad_test, grad_ref, param_keys):
    """Compute max_abs and max_rel over a set of parameter keys."""
    max_abs = 0.0
    max_rel = 0.0
    per_param = {}
    for k in param_keys:
        if k not in grad_test or k not in grad_ref:
            continue
        gt = float(np.asarray(grad_test[k]))
        gr = float(np.asarray(grad_ref[k]))
        d = gt - gr
        scale = max(abs(gt), abs(gr), 1e-30)
        rel = abs(d) / scale
        per_param[k] = dict(grad_test=gt, grad_ref=gr,
                            abs_diff=d, rel_diff=rel)
        max_abs = max(max_abs, abs(d))
        max_rel = max(max_rel, rel)
    return dict(max_abs=max_abs, max_rel=max_rel, per_param=per_param)


# ---- Sampled-r production / reference log-L at fixed r_ang ----

def _ll_fixed_r_production(model, sample, r_ang):
    """Scalar sum of phi-marginalised log-L at fixed r_ang."""
    pa, pk = jax_phys_from_sample(model, sample)
    groups = []
    if model._n_sys > 0:
        groups.append(
            ("sys", model._idx_sys, r_ang[model._idx_sys], None))
    if model._n_red > 0:
        groups.append(
            ("red", model._idx_red, r_ang[model._idx_red], None))
    if model._n_blue > 0:
        groups.append(
            ("blue", model._idx_blue, r_ang[model._idx_blue], None))
    ll = model._eval_phi_marginal(groups, pa, pk)
    return jnp.sum(ll)


def _ll_fixed_r_reference(model, sample, r_ang, n_phi, spot_batch):
    """Scalar sum of fixed-r log-L using a full-2π uniform phi reference.

    Spot-batched with jax.checkpoint on each batch's phi integration so
    reverse-mode tape memory stays at one batch's forward activations.
    """
    pa, pk = jax_phys_from_sample(model, sample)
    phi = jnp.linspace(0.0, 2 * jnp.pi, int(n_phi))
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    log_w = trapz_log_weights(phi)

    total = jnp.zeros((), dtype=r_ang.dtype)
    for key, idx in (("sys", model._idx_sys),
                     ("red", model._idx_red),
                     ("blue", model._idx_blue)):
        n = int(idx.shape[0])
        if n == 0:
            continue

        def _batch(b_idx, r_b, pa, pk):
            log_f = model._phi_integrand(
                r_b, sin_phi, cos_phi, b_idx, *pa, **pk)
            return jnp.sum(logsumexp(log_f + log_w, axis=-1))
        _batch_ckpt = jax.checkpoint(_batch)

        for s in range(0, n, int(spot_batch)):
            b = idx[s:s + int(spot_batch)]
            r_b = r_ang[b]
            total = total + _batch_ckpt(b, r_b, pa, pk)
    return total


def grad_fixed_r_production(model, sample, r_ang):
    """Returns (globals_grad_dict, r_ang_grad_vec)."""
    def f(s, r):
        return _ll_fixed_r_production(model, s, r)
    g_glob, g_r = jax.grad(f, argnums=(0, 1))(sample, r_ang)
    return ({k: np.asarray(v) for k, v in g_glob.items()},
            np.asarray(g_r))


def grad_fixed_r_reference(model, sample, r_ang, ref_cfg):
    """Returns (globals_grad_dict, r_ang_grad_vec) on the full-2π ref."""
    n_phi = int(ref_cfg["n_phi"])
    spot_batch = int(ref_cfg["spot_batch"])

    def f(s, r):
        return _ll_fixed_r_reference(model, s, r, n_phi, spot_batch)
    g_glob, g_r = jax.grad(f, argnums=(0, 1))(sample, r_ang)
    return ({k: np.asarray(v) for k, v in g_glob.items()},
            np.asarray(g_r))


def vector_diff_report(vec_test, vec_ref):
    """max_abs and max_rel over a vector diff (e.g. r_ang gradients)."""
    vt = np.asarray(vec_test, dtype=np.float64)
    vr = np.asarray(vec_ref, dtype=np.float64)
    d = vt - vr
    scale = np.maximum(np.maximum(np.abs(vt), np.abs(vr)), 1e-30)
    return dict(
        max_abs=float(np.max(np.abs(d))),
        max_rel=float(np.max(np.abs(d) / scale)),
        argmax_abs=int(np.argmax(np.abs(d))),
        argmax_rel=int(np.argmax(np.abs(d) / scale)))
