# Copyright (C) 2026 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Helpers for stage-2 toy joint-H0 inference from saved distance chains."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from candel.field import (available_mcmc_field_indices, field_mas_directory,
                          name2field_loader)
from candel.field.field_interp import (_get_grid_params,
                                       _trilinear_interp_field,
                                       apply_gaussian_smoothing,
                                       prepare_los_geometry)
from candel.model.utils import log_prob_integrand_sel
from candel.field.field_cache import (_field_cache_dir_from_config,
                                      _field_cache_enabled_from_config,
                                      _field_cache_portable_loader_kwargs,
                                      _field_cache_product_path,
                                      _field_cache_scope)
from candel.field.volume_density import _load_volume_data_for_H0
from candel.util import (R_ICRS_TO_GAL, R_ICRS_TO_SUPERGAL, SHARED_CONFIG_DIR,
                         SPEED_OF_LIGHT, fprint, load_config,
                         radec_to_cartesian)

_R_ICRS_TO_GAL = jnp.asarray(R_ICRS_TO_GAL)
_R_ICRS_TO_SUPERGAL = jnp.asarray(R_ICRS_TO_SUPERGAL)

# -----------------------------------------------------------------------
# Shared paths and galaxy data
# -----------------------------------------------------------------------

# Reconstruction and field-cache paths only, from the shared core fragment.
DEFAULT_FIELD_CONFIG = os.path.join(SHARED_CONFIG_DIR, "config_paths.toml")
FIELD_CACHE_PROJECT = "MMH0"


def _jsonable(value):
    """Convert cache payload values to stable JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _cache_digest(payload, length=24):
    text = json.dumps(_jsonable(payload), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _toy_vlos_cache_path(payload, config):
    recon = str(payload["reconstruction"])
    portable_payload = dict(payload)
    field_kwargs = portable_payload.get("field_kwargs")
    if isinstance(field_kwargs, dict):
        portable_payload["field_kwargs"] = (
            _field_cache_portable_loader_kwargs(field_kwargs))
    digest = _cache_digest(portable_payload)
    return _field_cache_product_path(
        _field_cache_dir_from_config(config), FIELD_CACHE_PROJECT, recon,
        "toy_maser_vlos", _field_cache_scope(payload), f"{digest}.npz")


def _toy_vlos_cache_payload(reconstruction, field_kwargs, field_indices, r,
                            galaxy_data, velocity_smoothing_scale=0.0):
    """Return the complete cache identity for a joint-H0 velocity LOS."""
    return {
        "version": 1,
        "product": "toy_maser_vlos",
        "reconstruction": reconstruction,
        "field_kwargs": field_kwargs,
        "field_indices": [int(index) for index in field_indices],
        "r": np.asarray(r, dtype=np.float32),
        "galaxy_names": [gd["name"] for gd in galaxy_data],
        "RA": np.asarray([gd["RA"] for gd in galaxy_data], dtype=np.float64),
        "dec": np.asarray(
            [gd["dec"] for gd in galaxy_data], dtype=np.float64),
        "velocity_field_smoothing_scale": float(velocity_smoothing_scale),
    }


def _manticore_index_root(reconstruction, field_kwargs):
    root = Path(field_kwargs["fpath_root"])
    if "cola" in reconstruction.lower():
        root = root / field_mas_directory(field_kwargs.get("which_MAS", "CIC"))
    return root


def _available_indices(reconstruction, field_kwargs):
    if reconstruction == "Carrick2015":
        return [0]
    if reconstruction.lower().startswith("manticorelocal"):
        return available_mcmc_field_indices(
            _manticore_index_root(reconstruction, field_kwargs))
    raise ValueError(f"Unsupported reconstruction `{reconstruction}`.")


def _resolve_velocity_beta(reconstruction, velocity_beta):
    if velocity_beta is not None:
        return float(velocity_beta)
    return 0.43 if reconstruction == "Carrick2015" else 1.0


def _resolve_volume_subsample_fraction(reconstruction, fraction, defaults):
    if fraction is not None:
        return float(fraction)
    key = ("manticorelocal"
           if reconstruction.lower().startswith("manticorelocal")
           else reconstruction)
    return float(defaults.get(key, defaults.get("default", 1.0)))


def interpolate_los_velocity(field_loader, r, RA, dec,
                             velocity_field_smoothing_scale=0.0,
                             verbose=True):
    """Interpolate only the radial velocity field on the requested LOS grid."""
    pos_flat, rhat, n_r, n_gal = prepare_los_geometry(
        field_loader, r, RA, dec)
    los_velocity = np.zeros((n_r, n_gal), dtype=np.float32)
    can_load_component = hasattr(field_loader, "load_velocity_component")
    velocity_smooth = float(velocity_field_smoothing_scale or 0.0)

    def _add_component(v_comp, comp):
        ngrid = v_comp.shape[0]
        cellsize, grid_min = _get_grid_params(field_loader, ngrid)
        if velocity_smooth > 0:
            if velocity_smooth <= cellsize:
                raise ValueError(
                    "`velocity_field_smoothing_scale` must exceed the field "
                    f"voxel size {cellsize:g} Mpc/h.")
            v_comp = apply_gaussian_smoothing(
                v_comp, velocity_smooth, field_loader.boxsize,
                make_copy=True)
        v_flat = np.ascontiguousarray(v_comp, dtype=np.float32).ravel()
        los_v = _trilinear_interp_field(
            v_flat, pos_flat, grid_min, cellsize, ngrid, np.float32(0.0))
        los_v = los_v.reshape(n_r, n_gal)
        los_v *= rhat[None, :, comp]
        los_velocity[:] += los_v

    if can_load_component:
        for comp in range(3):
            fprint(f"interpolating velocity component {comp}...",
                   verbose=verbose)
            _add_component(field_loader.load_velocity_component(comp), comp)
        if hasattr(field_loader, "clear_velocity_cache"):
            field_loader.clear_velocity_cache()
    else:
        velocity = field_loader.load_velocity()
        for comp in range(3):
            _add_component(velocity[comp], comp)

    return los_velocity.T, rhat


def _load_or_build_vlos_cache(reconstruction, field_config_path,
                              field_indices, r, galaxy_data,
                              velocity_smoothing_scale=0.0,
                              overwrite=False):
    field_config = load_config(field_config_path)
    field_kwargs = dict(
        field_config["io"]["reconstruction_main"].get(reconstruction, {}))
    if not field_kwargs:
        raise ValueError(
            f"Missing `io.reconstruction_main.{reconstruction}` in "
            f"`{field_config_path}` or local_config.toml.")

    available = _available_indices(reconstruction, field_kwargs)
    if field_indices is None:
        field_indices = available
    else:
        field_indices = [int(i) for i in field_indices]
        missing = [i for i in field_indices if i not in available]
        if missing:
            raise ValueError(
                f"Field indices {missing} are unavailable for "
                f"`{reconstruction}`. Available: {available}.")

    payload = _toy_vlos_cache_payload(
        reconstruction, field_kwargs, field_indices, r, galaxy_data,
        velocity_smoothing_scale)
    names = payload["galaxy_names"]
    RA = payload["RA"]
    dec = payload["dec"]
    cache_path = _toy_vlos_cache_path(payload, field_config)
    if os.path.exists(cache_path) and not overwrite:
        print(f"Loading velocity LOS cache: {cache_path}", flush=True)
        with np.load(cache_path, allow_pickle=False) as f:
            return {
                "r": f["r"],
                "los_velocity": f["los_velocity"],
                "rhat": f["rhat"],
                "field_indices": f["field_indices"],
                "galaxy_names": [str(x) for x in f["galaxy_names"]],
                "coordinate_frame": str(f["coordinate_frame"].item()),
                "cache_path": cache_path,
            }

    print(f"Building velocity LOS cache: {cache_path}", flush=True)
    loader_cls = name2field_loader(reconstruction)
    vlos_fields = []
    rhat_ref = None
    coordinate_frame = None
    for nsim in field_indices:
        kwargs = dict(field_kwargs)
        kwargs["nsim"] = int(nsim)
        loader = loader_cls(**kwargs)
        coordinate_frame = loader.coordinate_frame
        print(f"  field {nsim} ({coordinate_frame})", flush=True)
        vlos, rhat = interpolate_los_velocity(
            loader, r, RA, dec,
            velocity_field_smoothing_scale=velocity_smoothing_scale)
        if rhat_ref is None:
            rhat_ref = rhat
        vlos_fields.append(vlos.astype(np.float32))

    arrays = {
        "r": np.asarray(r, dtype=np.float32),
        "los_velocity": np.stack(vlos_fields).astype(np.float32),
        "rhat": np.asarray(rhat_ref, dtype=np.float32),
        "field_indices": np.asarray(field_indices, dtype=np.int32),
        "galaxy_names": np.asarray(names),
        "coordinate_frame": np.asarray(coordinate_frame),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_", suffix=".npz", dir=os.path.dirname(cache_path))
    os.close(fd)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    arrays["cache_path"] = cache_path
    return arrays


def _load_volume_selection_data(reconstruction, field_config_path,
                                field_indices, radius, geometry,
                                voxel_subsample_fraction,
                                voxel_subsample_seed,
                                field_smoothing_scale,
                                velocity_smoothing_scale,
                                supersample_factor,
                                supersample_radius,
                                supersample_target_dx):
    field_config = load_config(field_config_path)
    field_kwargs = dict(
        field_config["io"]["reconstruction_main"].get(reconstruction, {}))
    if not field_kwargs:
        raise ValueError(
            f"Missing `io.reconstruction_main.{reconstruction}` in "
            f"`{field_config_path}` or local_config.toml.")

    cache_enabled = _field_cache_enabled_from_config(field_config)
    cache_dir = (_field_cache_dir_from_config(field_config)
                 if cache_enabled else None)
    print("Loading 3D selection volume "
          f"({reconstruction}, {len(field_indices)} field(s), "
          f"radius={radius:g} Mpc/h, geometry={geometry}).",
          flush=True)
    out = _load_volume_data_for_H0(
        reconstruction,
        field_kwargs,
        np.asarray(field_indices, dtype=np.int32),
        "powerlaw",  # density unused below (no inhomogeneous Malmquist bias).
        0.315,
        subcube_radius=radius,
        voxel_subsample_fraction=voxel_subsample_fraction,
        voxel_subsample_seed=voxel_subsample_seed,
        load_velocity=True,
        geometry=geometry,
        cache_dir=cache_dir,
        cache_project=FIELD_CACHE_PROJECT,
        cache_enabled=cache_enabled,
        store_rhat=True,
        supersample_factor=supersample_factor,
        supersample_radius=supersample_radius,
        supersample_target_dx=supersample_target_dx,
        field_smoothing_scale=field_smoothing_scale,
        velocity_field_smoothing_scale=velocity_smoothing_scale)
    keep = {
        "log_r_3d", "log_dV_3d",
        "log_volume_weight_3d", "zcosmo_3d", "vrad_3d_fields",
        "rhat_x_3d", "rhat_y_3d", "rhat_z_3d",
        # `rhat_*_3d` are in the field frame, so Vext must be rotated into it.
        "coordinate_frame_3d",
    }
    return {k: v for k, v in out.items() if k in keep}


def _attach_velocity_data(galaxy_data, velocity_data):
    names = list(velocity_data["galaxy_names"])
    if names != [gd["name"] for gd in galaxy_data]:
        raise ValueError("Velocity cache galaxy order does not match data.")
    los_velocity = np.asarray(velocity_data["los_velocity"])
    for i, gd in enumerate(galaxy_data):
        gd["los_r"] = jnp.asarray(velocity_data["r"])
        gd["los_velocity"] = jnp.asarray(los_velocity[:, i, :])
        gd["rhat"] = jnp.asarray(velocity_data["rhat"][i])
        # `prepare_los_geometry` builds rhat in the field's own frame.
        gd["rhat_frame"] = velocity_data["coordinate_frame"]


def _attach_icrs_rhat(galaxy_data):
    rhat = radec_to_cartesian(
        np.asarray([gd["RA"] for gd in galaxy_data]),
        np.asarray([gd["dec"] for gd in galaxy_data]))
    for gd, row in zip(galaxy_data, rhat):
        gd["rhat"] = jnp.asarray(row)
        gd["rhat_frame"] = "icrs"


def _interp_los_velocity(r, los_r, los_velocity, r0_decay_scale=5.0):
    """Interpolate one galaxy's LOS velocities, one value per field."""
    n_steps = los_r.shape[0]
    dr = (los_r[-1] - los_r[0]) / (n_steps - 1)
    idx_cont = jnp.clip((r - los_r[0]) / dr, 0.0, n_steps - 1.0)
    idx_lo = jnp.floor(idx_cont).astype(jnp.int32).clip(0, n_steps - 2)
    t = idx_cont - idx_lo
    y_lin = los_velocity[:, idx_lo] + t * (
        los_velocity[:, idx_lo + 1] - los_velocity[:, idx_lo])
    y_exp = los_velocity[:, -1] * jnp.exp(
        -(r - los_r[-1]) / r0_decay_scale)
    return jnp.where(r > los_r[-1], y_exp, y_lin)


def _logmeanexp(x):
    return jax.scipy.special.logsumexp(x) - jnp.log(x.shape[0])


def healpix_los_vectors(nside=2):
    """Unit vectors at HEALPix pixel centres for isotropic LOS averaging.

    The selection normalisation must be independent of where the observed
    galaxies sit on the sky, so the angular integral is approximated by an
    equal-weight average over these (12*nside**2) directions.
    """
    import healpy as hp
    npix = hp.nside2npix(nside)
    return jnp.asarray(np.asarray(hp.pix2vec(nside, np.arange(npix))).T)


def _volume_log_cell_weight(volume_data, h, flat_dist):
    log_r = volume_data["log_r_3d"]
    out = volume_data["log_dV_3d"] - 3.0 * jnp.log(h)
    if "log_volume_weight_3d" in volume_data:
        out = out + volume_data["log_volume_weight_3d"]
    if flat_dist:
        out = out - 2.0 * (log_r - jnp.log(h))
    return out


def rotate_vext_to_frame(Vext, frame):
    """
    Rotate an ICRS-Cartesian `Vext` into a reconstruction's own frame.  Vext is
    sampled in ICRS, but direction vectors taken from a reconstruction grid
    (the LOS `rhat` and the voxel `rhat_*_3d`) are in the field frame, so the
    two must be brought together before projecting.  Mirrors
    `base_model._vol_sel_Vext_rad_3d`.
    """
    frame = str(frame).lower()
    if frame == "icrs":
        return Vext
    if frame == "galactic":
        return _R_ICRS_TO_GAL @ Vext
    if frame == "supergalactic":
        return _R_ICRS_TO_SUPERGAL @ Vext
    raise ValueError(f"Unsupported coordinate frame for Vext: `{frame}`.")


def _volume_vext_radial(volume_data, Vext):
    Vext = rotate_vext_to_frame(Vext, volume_data["coordinate_frame_3d"])
    return (Vext[0] * volume_data["rhat_x_3d"]
            + Vext[1] * volume_data["rhat_y_3d"]
            + Vext[2] * volume_data["rhat_z_3d"])


def _predict_cz_exact(zcosmo, Vrad):
    beta = Vrad / SPEED_OF_LIGHT
    one_plus_z_pec = jnp.sqrt((1.0 + beta) / (1.0 - beta))
    return SPEED_OF_LIGHT * ((1.0 + zcosmo) * one_plus_z_pec - 1.0)


def _volume_log_Z_distance(volume_data, H0, D_lim, D_width, flat_dist):
    h = H0 / 100.0
    log_r = volume_data["log_r_3d"]
    # Uniform density: we do not model the inhomogeneous Malmquist bias, so the
    # reconstruction density does not weight the selection volume.
    log_cw = _volume_log_cell_weight(volume_data, h, flat_dist)

    # Checkpoint the voxel integrand so reverse-mode AD recomputes it in the
    # backward pass instead of storing the (n_vox,) intermediates (cf. the
    # CH0/TRGB volume selection integrals).  remat is gradient-transparent.
    @jax.checkpoint
    def _integral(D_lim, D_width):
        D_phys = jnp.exp(log_r) / h
        log_P_sel = jax.scipy.stats.norm.logcdf((D_lim - D_phys) / D_width)
        return jax.scipy.special.logsumexp(
            log_P_sel[None, :] + log_cw, axis=-1)

    return _integral(D_lim, D_width)


def _volume_log_Z_redshift(volume_data, H0, sigma_pec, velocity_beta, Vext,
                           cz_lim, cz_width, flat_dist):
    h = H0 / 100.0
    zcosmo = volume_data["zcosmo_3d"]
    vext_rad = _volume_vext_radial(volume_data, Vext)
    # Uniform density: we do not model the inhomogeneous Malmquist bias, so the
    # reconstruction density does not weight the selection volume.
    log_cw = _volume_log_cell_weight(volume_data, h, flat_dist)

    def _one(vrad):
        Vpec = velocity_beta * vrad + vext_rad
        cz_pred = _predict_cz_exact(zcosmo, Vpec)
        log_P_sel = log_prob_integrand_sel(
            cz_pred, sigma_pec, cz_lim, cz_width)
        return jax.scipy.special.logsumexp(log_P_sel + log_cw)

    # Checkpoint the per-field voxel integrand so reverse-mode AD recomputes it
    # in the backward pass instead of storing every (n_vox,) intermediate, and
    # map fields sequentially to cap peak memory at one field (cf. CH0/TRGB).
    # remat is gradient-transparent, so the AD gradient is unchanged.
    return jax.lax.map(jax.checkpoint(_one), volume_data["vrad_3d_fields"],
                       batch_size=1)


def main():
    raise SystemExit(
        "joint_H0_helpers.py is helper-only; use "
        "candel_maser/run_joint_H0.py or submit.sh --infer-H0."
    )


if __name__ == "__main__":
    main()
