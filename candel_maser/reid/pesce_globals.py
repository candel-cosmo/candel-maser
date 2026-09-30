# Copyright (C) 2026 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
"""Published Pesce+2020 / Reid MCP disk globals -> CANDEL ``theta``.

Builds the fixed-global Pesce/Reid point used by ``run_de_map`` (the printed
Pesce baseline and the ``--fix-globals-pesce`` score).  Geometry conventions —
CMB velocity frame, warp-pivot remap (Pesce appendix values are zero-radius
intercepts; CANDEL stores ``i0``/``Omega0`` at per-galaxy reference radii), and
the cosmographic ``D_c`` <-> ``D_A`` inversion — match Pesce+2020.
"""
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import astropy.units as u
from astropy.coordinates import SkyCoord

from ..megamaser_data import megamaser_velocity_frame

SCRIPT_DIR = Path(__file__).resolve().parent
PESCE_DISK_PARAMS = SCRIPT_DIR / "pesce_disk_params.toml"
REID_NGC4258_BEST = SCRIPT_DIR / "reid_ngc4258_best.toml"

SOLAR_LSR_UVW = (11.1, 12.24, 7.25)
CMB_DIPOLE_L = 264.021
CMB_DIPOLE_B = 48.253
CMB_DIPOLE_V = 369.82

_PESCE_KEY_MAP = {
    "D_Mpc": "D_A",
    "MBH_1e7": "M_BH_1e7",
    "v_cmb_kms": "v_cmb",
    "x0_mas": "x0_mas",
    "y0_mas": "y0_mas",
    "i0_deg": "i0_r0_deg",
    "didr_deg_mas": "di_dr_r0",
    "Omega0_deg": "Omega0_r0_deg",
    "dOmegadr_deg_mas": "dOmega_dr_r0",
    "sigma_x_mas": "sigma_x_mas",
    "sigma_y_mas": "sigma_y_mas",
    "sigma_vsys_kms": "sigma_v_sys",
    "sigma_vhv_kms": "sigma_v_hv",
    "sigma_a_kms_yr": "sigma_a",
}


def load_pesce_disk_params(path=PESCE_DISK_PARAMS):
    with path.open("rb") as f:
        rows = tomllib.load(f).get("galaxies", {})
    return {
        galaxy: {
            out: float(values[src])
            for src, out in _PESCE_KEY_MAP.items()
            if src in values
        }
        for galaxy, values in rows.items()
    }


PESCE2020 = load_pesce_disk_params()


def reid_ngc4258_point(master, path=REID_NGC4258_BEST):
    from .run_reid_mcmc import reid_D_A

    with path.open("rb") as f:
        row = tomllib.load(f)["globals"]
    gcfg = master["model"]["galaxies"]["NGC4258"]
    rperi = float(gcfg["r_ang_ref_periapsis"])
    reid_r_ref = float(row["r_ref_mas"])

    i_ref = 180.0 - float(row["i0_deg"])
    di_ref = -float(row["di_dr_deg_mas"])
    d2i = -float(row.get("d2i_dr2_deg_mas2", 0.0))
    Omega_ref = float(row["PA_deg"])
    dOmega_ref = float(row["dPA_dr_deg_mas"])
    d2Omega = float(row.get("d2PA_dr2_deg_mas2", 0.0))
    di_r0 = di_ref - 2.0 * d2i * reid_r_ref
    dOmega_r0 = dOmega_ref - 2.0 * d2Omega * reid_r_ref
    peri_ref = (
        float(row["peri_az_deg"])
        + float(row.get("dperi_dr_deg_mas", 0.0)) * rperi
    )
    peri_rad = math.radians(peri_ref)
    ecc = float(row.get("ecc", 0.0))

    # Reid's inclination and PA are quoted at fit_disk's fixed r_ref. Back
    # them out to the zero-radius convention used by candel_theta_from_point.
    velocity = float(row["Vsys_km_s"]) + float(row["Vcor_km_s"])
    return {
        "D_A": float(reid_D_A(velocity, float(row["H0"]))),
        "M_BH_1e7": float(row["Mbh_1e7Msun"]),
        "v_native": float(row["Vsys_km_s"]),
        "x0_mas": float(row["x0_mas"]),
        "y0_mas": float(row["y0_mas"]),
        "i0_r0_deg": (
            i_ref - di_r0 * reid_r_ref
            - d2i * reid_r_ref * reid_r_ref),
        "di_dr_r0": di_r0,
        "d2i_dr2": d2i,
        "Omega0_r0_deg": (
            Omega_ref - dOmega_r0 * reid_r_ref
            - d2Omega * reid_r_ref * reid_r_ref),
        "dOmega_dr_r0": dOmega_r0,
        "d2Omega_dr2": d2Omega,
        "sigma_x_mas": float(row["sigma_x_mas"]),
        "sigma_y_mas": float(row["sigma_y_mas"]),
        "sigma_v_sys": float(row["sigma_vsys_km_s"]),
        "sigma_v_hv": float(row["sigma_vhv_km_s"]),
        "sigma_a": float(row["sigma_acc_km_s_yr"]),
        "e_x": ecc * math.cos(peri_rad),
        "e_y": ecc * math.sin(peri_rad),
        "dperiapsis_dr": float(row.get("dperi_dr_deg_mas", 0.0)),
    }


def cmb_projection(ra_deg, dec_deg):
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    lon = float(target.galactic.l.rad)
    lat = float(target.galactic.b.rad)
    l_cmb = np.deg2rad(CMB_DIPOLE_L)
    b_cmb = np.deg2rad(CMB_DIPOLE_B)
    return lon, lat, (np.sin(lat) * np.sin(b_cmb)
                      + np.cos(lat) * np.cos(b_cmb) * np.cos(lon - l_cmb))


def to_cmb(v, frame, ra_deg, dec_deg):
    lon, lat, cos_theta = cmb_projection(ra_deg, dec_deg)
    if frame == "lsr":
        U, V, W = SOLAR_LSR_UVW
        v = (v - (U * np.cos(lat) * np.cos(lon)
                  + V * np.cos(lat) * np.sin(lon)
                  + W * np.sin(lat)))
    return v + CMB_DIPOLE_V * cos_theta


def from_cmb(v_cmb, frame, ra_deg, dec_deg):
    return float(v_cmb) - float(to_cmb(0.0, frame, ra_deg, dec_deg))


def da_from_dc(D_c, distance2redshift, h):
    D_c = np.atleast_1d(np.asarray(D_c, dtype=float))
    z = np.asarray(distance2redshift(D_c, h=h))
    out = D_c / (1.0 + z)
    return out[0] if out.size == 1 else out


def dc_bounds(gcfg):
    if "D_de_lo" in gcfg and "D_de_hi" in gcfg:
        return float(gcfg["D_de_lo"]), float(gcfg["D_de_hi"])
    if "D_lo" in gcfg and "D_hi" in gcfg:
        return float(gcfg["D_lo"]), float(gcfg["D_hi"])
    return None, None


def dc_from_da(D_A, distance2redshift, h, lo=None, hi=None):
    """Invert CANDEL's cosmographic D_c -> D_A map by bisection."""
    D_A = float(D_A)

    def f(dc):
        return float(da_from_dc(dc, distance2redshift, h)) - D_A

    lo = float(lo) if lo is not None else max(1e-6, 0.5 * D_A)
    hi = float(hi) if hi is not None else max(1.0, 1.5 * D_A)
    if f(lo) > 0 or f(hi) < 0:
        lo = max(1e-6, 0.25 * D_A)
        hi = max(1.0, 3.0 * D_A)
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


def warp_from_r0(point, gcfg):
    """Move zero-radius warp parameters back to CANDEL config pivots."""
    ri = float(gcfg["r_ang_ref_i"])
    rO = float(gcfg["r_ang_ref_Omega"])
    d2i = float(point.get("d2i_dr2", 0.0))
    d2O = float(point.get("d2Omega_dr2", 0.0))
    return {
        "i0": point["i0_r0_deg"] + point.get("di_dr_r0", 0.0) * ri
        + d2i * ri * ri,
        "di_dr": point.get("di_dr_r0", 0.0) + 2.0 * d2i * ri,
        "Omega0": point["Omega0_r0_deg"]
        + point.get("dOmega_dr_r0", 0.0) * rO + d2O * rO * rO,
        "dOmega_dr": point.get("dOmega_dr_r0", 0.0) + 2.0 * d2O * rO,
        "d2i_dr2": d2i,
        "d2Omega_dr2": d2O,
    }


def candel_theta_from_point(point, galaxy, master, target):
    gcfg = master["model"]["galaxies"][galaxy]
    D_c = point.get("D_c")
    if D_c is None:
        lo, hi = dc_bounds(gcfg)
        D_c = dc_from_da(
            point["D_A"], target.model.distance2redshift, target.h, lo, hi)
    D_A = point.get("D_A")
    if D_A is None:
        D_A = da_from_dc(D_c, target.model.distance2redshift, target.h)
    log_mbh = math.log10(point["M_BH_1e7"] * 1.0e7)
    warp = warp_from_r0(point, gcfg)
    theta = {
        "x0": point["x0_mas"] * 1000.0,
        "y0": point["y0_mas"] * 1000.0,
        "i0": warp["i0"],
        "di_dr": warp["di_dr"],
        "Omega0": warp["Omega0"],
        "dOmega_dr": warp["dOmega_dr"],
        "dv_sys": point["v_native"] - float(gcfg["v_sys_obs"]),
        "sigma_x_floor": point["sigma_x_mas"] * 1000.0,
        "sigma_y_floor": point["sigma_y_mas"] * 1000.0,
        "sigma_v_sys": point["sigma_v_sys"],
        "sigma_v_hv": point["sigma_v_hv"],
        "sigma_a_floor": point["sigma_a"],
    }
    if target.model._D_A_uniform:
        theta["D_A"] = float(D_A)
    else:
        theta["D_c"] = float(D_c)
    if target.mass_parameterization == "eta":
        theta["eta"] = point.get("eta", log_mbh - math.log10(D_A))
    else:
        theta["log_MBH"] = log_mbh
    if target.model.use_quadratic_warp:
        theta["d2i_dr2"] = warp["d2i_dr2"]
        theta["d2Omega_dr2"] = warp["d2Omega_dr2"]
    if target.model.use_ecc:
        if target.model.ecc_cartesian:
            theta["e_x"] = point.get("e_x", 0.0)
            theta["e_y"] = point.get("e_y", 0.0)
            theta["dperiapsis_dr"] = point.get("dperiapsis_dr", 0.0)
        else:
            theta["ecc"] = point.get("ecc", 0.0)
            theta["periapsis"] = point.get("periapsis", 0.0)
            theta["dperiapsis_dr"] = point.get("dperiapsis_dr", 0.0)
    if target.model.use_clump2_floors:
        theta.update({name: theta[source] for name, source in
                      target.model.clump2_floor_pairs})

    missing = [name for name in target.names if name not in theta]
    if missing:
        raise KeyError(f"CANDEL theta missing {missing}")
    return {name: jnp.asarray(theta[name]) for name in target.names}


def paper_point(galaxy, master):
    if galaxy == "NGC4258":
        return reid_ngc4258_point(master), []

    ref = PESCE2020[galaxy]
    needed = [
        "D_A", "M_BH_1e7", "v_cmb", "x0_mas", "y0_mas", "i0_r0_deg",
        "Omega0_r0_deg", "sigma_x_mas", "sigma_y_mas", "sigma_v_sys",
        "sigma_v_hv", "sigma_a",
    ]
    missing = [k for k in needed if k not in ref]
    if missing:
        return None, missing
    gcfg = master["model"]["galaxies"][galaxy]
    frame = megamaser_velocity_frame(galaxy)
    out = dict(ref)
    out["v_native"] = from_cmb(
        ref["v_cmb"], frame, gcfg["ra"], gcfg["dec"])
    defaulted = []
    for key in ("di_dr_r0", "dOmega_dr_r0"):
        if key not in out:
            out[key] = 0.0
            defaulted.append(key)
    out["d2i_dr2"] = 0.0
    out["d2Omega_dr2"] = 0.0
    return out, defaulted
