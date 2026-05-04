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
"""Megamaser spot data loading from AAS machine-readable tables."""
from os.path import join

import numpy as np
from scipy.cluster.vq import kmeans2

from ..util import fprint

_MRT_FILES = {
    "CGCG074-064": "CGCG074-064_Pesce2020_mrt.txt",
}

# Velocity reference frame of the spot data for each galaxy.
# "lsr"         — LSR frame, optical convention (cz)
# "barycentric" — barycentric (heliocentric) frame, optical convention
# "unknown"     — not stated in the source paper
_GALAXY_VELOCITY_FRAME = {
    "NGC6264":     "lsr",         # Kuo+2013, footnote a
    "NGC6323":     "lsr",         # Kuo+2015, footnote a
    "UGC3789":     "lsr",         # Reid+2009/2013, column header
    "CGCG074-064": "barycentric",  # Pesce+2020 MRT header
    "NGC5765b":    "unknown",     # Gao+2016: not stated
    "NGC4258":     "lsr",         # Argon+2007
}

# Solar motion w.r.t. LSR (Schönrich+2010, km/s, Galactic Cartesian U,V,W)
_SOLAR_LSR_UVW = (11.1, 12.24, 7.25)

# CMB dipole (Planck 2018): direction in Galactic coords, amplitude in km/s
_CMB_DIPOLE_L = 264.021   # deg
_CMB_DIPOLE_B = 48.253    # deg
_CMB_DIPOLE_V = 369.82    # km/s


def radio_to_optical_velocity(v_radio):
    """Convert radio-convention velocity to optical convention as Reid does."""
    clight = 2.997925e5
    return v_radio / (1.0 - v_radio / clight)


def v_sys_to_cmb(v_sys_km_s, frame, ra_deg, dec_deg):
    """Convert a systemic velocity to the CMB rest frame.

    Parameters
    ----------
    v_sys_km_s : float
        Systemic velocity in km/s in the original `frame`.
    frame : str
        One of "lsr", "barycentric", or "unknown".
    ra_deg, dec_deg : float
        Galaxy sky position (ICRS, degrees).

    Returns
    -------
    float or None
        CMB-frame velocity in km/s, or None if frame is "unknown".
    """
    if frame == "unknown":
        return None

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    lon = float(target.galactic.l.rad)
    b = float(target.galactic.b.rad)

    # LSR → heliocentric: subtract solar motion projected onto LOS
    if frame == "lsr":
        U, V, W = _SOLAR_LSR_UVW
        v_helio = (v_sys_km_s
                   - (U * np.cos(b) * np.cos(lon)
                      + V * np.cos(b) * np.sin(lon)
                      + W * np.sin(b)))
    else:  # barycentric ≈ heliocentric (sub-km/s difference)
        v_helio = v_sys_km_s

    # Heliocentric → CMB: add CMB dipole projected onto LOS
    l_cmb = np.deg2rad(_CMB_DIPOLE_L)
    b_cmb = np.deg2rad(_CMB_DIPOLE_B)
    cos_theta = (np.sin(b) * np.sin(b_cmb)
                 + np.cos(b) * np.cos(b_cmb) * np.cos(lon - l_cmb))
    return v_helio + _CMB_DIPOLE_V * cos_theta


# Column byte ranges (1-indexed, inclusive) from the MRT header.
_MRT_COLUMNS = {
    "spot_type":      (1, 1),
    "velocity":       (3, 9),
    "flux":           (11, 17),
    "e_flux":         (19, 25),
    "x":              (27, 35),
    "sigma_x":        (37, 44),
    "y":              (46, 54),
    "sigma_y":        (56, 63),
    "a":              (65, 70),
    "sigma_a":        (72, 76),
    "accel_measured":  (78, 78),
}


def _parse_mrt_line(line):
    """Parse a single data line from the MRT file using fixed-width columns."""
    row = {}
    for key, (b0, b1) in _MRT_COLUMNS.items():
        raw = line[b0 - 1:b1].strip()
        if key == "spot_type":
            row[key] = raw
        elif key == "accel_measured":
            row[key] = int(raw)
        else:
            row[key] = float(raw)
    return row


def load_NGC5765b_spots(root, v_sys_obs=None):
    """Load maser spot data for NGC 5765b from Gao+2016 Table 6.

    The table provides velocity, position (x, y), and acceleration for 192
    maser spots. All spots have measured accelerations.

    Parameters
    ----------
    root : str
        Directory containing ``NGC5765b_Gao2016_table6.dat``.
    v_sys_obs : float
        Observed CMB-frame recession velocity in km/s. Required.

    Returns
    -------
    dict with the same keys as ``load_megamaser_spots``.
    """
    fname = "NGC5765b_Gao2016_table6.dat"
    fpath = join(root, fname)
    fprint(f"loading maser spots from '{fpath}'.")

    cols = ["velocity", "x", "sigma_x", "y", "sigma_y", "a", "sigma_a"]
    rows = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append({k: float(v) for k, v in zip(cols, parts)})

    n = len(rows)
    velocity = np.array([r["velocity"] for r in rows])
    x = np.array([r["x"] for r in rows])
    sigma_x = np.array([r["sigma_x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    sigma_y = np.array([r["sigma_y"] for r in rows])
    a = np.array([r["a"] for r in rows])
    sigma_a = np.array([r["sigma_a"] for r in rows])

    # Flag placeholder accelerations:
    # A=1.0, sigma_a=1.0 (sentinel) and A=0.0, sigma_a=0.2 (undetected)
    accel_measured = ~(
        ((a == 1.0) & (sigma_a == 1.0))
        | ((a == 0.0) & (np.abs(sigma_a - 0.2) < 0.01))
    )

    if v_sys_obs is None:
        raise ValueError("v_sys_obs must be provided for NGC5765b.")

    data = {
        "velocity": velocity,
        "x": x,
        "sigma_x": sigma_x,
        "y": y,
        "sigma_y": sigma_y,
        "a": a,
        "sigma_a": sigma_a,
        "accel_measured": accel_measured,
        "n_spots": n,
        "galaxy_name": "NGC5765b",
        "v_sys_obs": float(v_sys_obs),
    }

    # Convert positions from mas to μas for float32 precision
    for key in ("x", "y", "sigma_x", "sigma_y"):
        if key in data:
            data[key] = data[key] * 1000.0

    n_accel = int(accel_measured.sum())
    fprint(f"loaded {n} maser spots for NGC5765b "
           f"({n_accel} with measured acceleration).")
    return data


def _is_missing(s):
    """Check if a table field represents a missing value."""
    s = s.strip()
    return s in ("...", "sdotsdotsdot", "")


def _load_kuo_table2(root, fname, galaxy_label, v_sys_obs=None):
    """Load maser spots from a Kuo+ Table 2 file with inline accelerations.

    The table is tab-delimited with columns: V_op, Theta_x, sigma_x,
    Theta_y, sigma_y, A, sigma_A. Missing accelerations are marked with
    ``...`` or ``sdotsdotsdot``.

    Parameters
    ----------
    root : str
        Directory containing the data file.
    fname : str
        Filename (e.g. ``"NGC6264_Kuo2013_table2.txt"``).
    galaxy_label : str
        Short label used in output messages.
    v_sys_obs : float or None
        Observed CMB-frame recession velocity in km/s. If None, use the
        per-galaxy default.

    Returns
    -------
    dict with the same keys as ``load_megamaser_spots``.
    """
    fpath = join(root, fname)
    fprint(f"loading maser spots from '{fpath}'.")

    with open(fpath) as f:
        lines = f.readlines()

    # Skip header lines (everything before the first numeric data line).
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("Table") or \
                stripped.startswith("(") or stripped.startswith("Notes") or \
                any(c.isalpha() for c in stripped.split("\t")[0].strip()):
            data_start = i + 1
        else:
            break

    velocities, xs, sigma_xs, ys, sigma_ys = [], [], [], [], []
    accels, sigma_accels, has_accel = [], [], []

    for line in lines[data_start:]:
        line = line.rstrip("\n")
        if not line or line.startswith("Notes"):
            break
        fields = line.split("\t")
        if len(fields) < 5:
            continue

        velocities.append(float(fields[0].strip()))
        xs.append(float(fields[1].strip()))
        sigma_xs.append(float(fields[2].strip()))
        ys.append(float(fields[3].strip()))
        sigma_ys.append(float(fields[4].strip()))

        if len(fields) >= 7 and not _is_missing(fields[5]) and \
                not _is_missing(fields[6]):
            accels.append(float(fields[5].strip()))
            sigma_accels.append(float(fields[6].strip()))
            has_accel.append(True)
        else:
            # A or σ_a missing: flag has_accel=False. σ_a = 1e4 is a
            # large placeholder so the model's var_a stays positive;
            # the has_accel mask zeroes the contribution.
            accels.append(0.0)
            sigma_accels.append(1e4)
            has_accel.append(False)

    n = len(velocities)
    if n == 0:
        raise ValueError(f"No spots found in {fpath}.")

    velocity = np.array(velocities)
    x = np.array(xs)
    sigma_x = np.array(sigma_xs)
    y = np.array(ys)
    sigma_y = np.array(sigma_ys)
    a = np.array(accels)
    sigma_a = np.array(sigma_accels)
    accel_measured = np.array(has_accel, dtype=bool)

    if v_sys_obs is None:
        raise ValueError(f"v_sys_obs must be provided for {galaxy_label}.")

    data = {
        "velocity": velocity,
        "x": x,
        "sigma_x": sigma_x,
        "y": y,
        "sigma_y": sigma_y,
        "a": a,
        "sigma_a": sigma_a,
        "accel_measured": accel_measured,
        "n_spots": n,
        "galaxy_name": galaxy_label,
        "v_sys_obs": float(v_sys_obs),
    }

    # Convert positions from mas to μas for float32 precision
    for key in ("x", "y", "sigma_x", "sigma_y"):
        if key in data:
            data[key] = data[key] * 1000.0

    n_accel = int(accel_measured.sum())
    fprint(f"loaded {n} maser spots for {galaxy_label} "
           f"({n_accel} with measured acceleration).")
    return data


def load_NGC6264_spots(root, v_sys_obs=None):
    """Load maser spot data for NGC 6264 from Kuo+2013 Table 2."""
    return _load_kuo_table2(
        root, "NGC6264_Kuo2013_table2.txt", "NGC6264",
        v_sys_obs=v_sys_obs)


def load_NGC6323_spots(root, v_sys_obs=None):
    """Load maser spot data for NGC 6323 from Kuo+2015 Table 2."""
    return _load_kuo_table2(
        root, "NGC6323_Kuo2015_table2.txt", "NGC6323",
        v_sys_obs=v_sys_obs)


def load_UGC3789_spots(root, v_sys_obs=None):
    """Load maser spot data for UGC 3789 from Reid+2013 Table 1."""
    return _load_kuo_table2(
        root, "UGC3789_Reid2013_table1.txt", "UGC3789",
        v_sys_obs=v_sys_obs)


def load_NGC4258_spots(root, v_sys_obs=472.0):
    """Load maser spot data for NGC 4258 from Reid (private comm.).

    Reads ``N4258_disk_data_MarkReid.final``. Comment lines start with ``!``.
    Columns: ID, Vlsr, e_Vlsr, dX, e_dX, dY, e_dY, Acc, e_Acc.
    The Mark Reid file marks velocities as radio convention; central
    velocities are converted to optical convention using Reid's formula.
    Missing accelerations are flagged by negative e_Acc.

    Spot classification by Vlsr: <300 → blue HV, 300–700 → systemic, >700 →
    red HV.

    Returns raw measurement errors (no floors applied).

    Parameters
    ----------
    root : str
        Directory containing ``N4258_disk_data_MarkReid.final``.
    v_sys_obs : float
        Observed CMB-frame recession velocity in km/s.

    Returns
    -------
    dict with the same keys as ``load_megamaser_spots``.
    """
    fpath = join(root, "N4258_disk_data_MarkReid.final")
    fprint(f"loading maser spots from '{fpath}'.")

    velocity, sigma_v, x, sigma_x, y, sigma_y = [], [], [], [], [], []
    a_vals, sigma_a_vals, has_accel, spot_type = [], [], [], []

    with open(fpath) as f:
        for line in f:
            if line.startswith("!") or not line.strip():
                continue
            parts = line.split()
            vlsr = radio_to_optical_velocity(float(parts[1]))
            e_vlsr = float(parts[2])
            dx = float(parts[3])
            e_dx = float(parts[4])
            dy = float(parts[5])
            e_dy = float(parts[6])
            acc = float(parts[7])
            e_acc = float(parts[8])

            # Classification
            if vlsr < 300:
                st = "b"
            elif vlsr > 700:
                st = "r"
            else:
                st = "s"

            velocity.append(vlsr)
            sigma_v.append(e_vlsr)
            x.append(dx)
            sigma_x.append(e_dx)
            y.append(dy)
            sigma_y.append(e_dy)
            spot_type.append(st)

            if e_acc > 0:
                a_vals.append(acc)
                sigma_a_vals.append(e_acc)
                has_accel.append(True)
            else:
                # No measurement: σ_a = 1e4 placeholder (kept positive
                # so the model's var_a is finite); has_accel masks
                # the contribution.
                a_vals.append(0.0)
                sigma_a_vals.append(1e4)
                has_accel.append(False)

    n = len(velocity)
    n_r = spot_type.count("r")
    n_b = spot_type.count("b")
    n_s = spot_type.count("s")
    n_accel = sum(has_accel)
    fprint(f"loaded {n} maser spots for NGC4258 "
           f"({n_r} red, {n_b} blue, {n_s} systemic; "
           f"{n_accel} with measured acceleration).")

    data = {
        "velocity":        np.array(velocity),
        "sigma_v":         np.array(sigma_v),
        "x":               np.array(x),
        "sigma_x":         np.array(sigma_x),
        "y":               np.array(y),
        "sigma_y":         np.array(sigma_y),
        "a":               np.array(a_vals),
        "sigma_a":         np.array(sigma_a_vals),
        "accel_measured":  np.array(has_accel, dtype=bool),
        "spot_type":       spot_type,
        "n_spots":         n,
        "galaxy_name":     "NGC4258",
        "v_sys_obs":       float(v_sys_obs),
    }
    # Convert positions from mas to μas for float32 precision
    for key in ("x", "y", "sigma_x", "sigma_y"):
        if key in data:
            data[key] = data[key] * 1000.0
    return data


def load_megamaser_spots(root, galaxy="CGCG074-064", v_sys_obs=None):
    """Load individual maser spot data for a megamaser galaxy.

    Parameters
    ----------
    root : str
        Directory containing the data file.
    galaxy : str
        Galaxy name: ``"CGCG074-064"``, ``"NGC5765b"``, etc.
    v_sys_obs : float or None
        Observed CMB-frame recession velocity in km/s. Required except for
        NGC4258, where a default is supplied.

    Returns
    -------
    dict with numpy arrays for each spot property, including ``v_sys_obs``.
    """
    if galaxy == "NGC5765b":
        data = load_NGC5765b_spots(root, v_sys_obs=v_sys_obs)
    elif galaxy == "NGC6264":
        data = load_NGC6264_spots(root, v_sys_obs=v_sys_obs)
    elif galaxy == "NGC6323":
        data = load_NGC6323_spots(root, v_sys_obs=v_sys_obs)
    elif galaxy == "UGC3789":
        data = load_UGC3789_spots(root, v_sys_obs=v_sys_obs)
    elif galaxy == "NGC4258":
        data = load_NGC4258_spots(root, v_sys_obs=v_sys_obs)
    else:
        _all = list(_MRT_FILES) + ["NGC5765b", "NGC6264", "NGC6323",
                                   "UGC3789", "NGC4258"]
        if galaxy not in _MRT_FILES:
            raise ValueError(
                f"Unknown galaxy '{galaxy}'. Available: {_all}")

        if v_sys_obs is None:
            raise ValueError(f"v_sys_obs must be provided for {galaxy}.")

        fpath = join(root, _MRT_FILES[galaxy])
        fprint(f"loading maser spots from '{fpath}'.")

        with open(fpath) as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("---"):
                data_start = i + 1

        rows = []
        for line in lines[data_start:]:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            rows.append(_parse_mrt_line(line))

        n = len(rows)
        data = {
            "velocity": np.array([r["velocity"] for r in rows]),
            "x": np.array([r["x"] for r in rows]),
            "sigma_x": np.array([r["sigma_x"] for r in rows]),
            "y": np.array([r["y"] for r in rows]),
            "sigma_y": np.array([r["sigma_y"] for r in rows]),
            "a": np.array([r["a"] for r in rows]),
            "sigma_a": np.array([r["sigma_a"] for r in rows]),
            "accel_measured": np.array(
                [r["accel_measured"] for r in rows], dtype=bool),
            "n_spots": n,
            "galaxy_name": galaxy,
            "v_sys_obs": float(v_sys_obs),
        }

        n_accel = int(data["accel_measured"].sum())
        fprint(f"loaded {n} maser spots for {galaxy} "
               f"({n_accel} with measured acceleration).")

        # Convert positions from mas to μas for float32 precision
        for key in ("x", "y", "sigma_x", "sigma_y"):
            if key in data:
                data[key] = data[key] * 1000.0

    # Classify spots: use spot_type from data if available, else k-means
    n = data["n_spots"]
    if "spot_type" in data:
        st = data["spot_type"]
        labels = np.array([{"b": 0, "s": 1, "r": 2}[t] for t in st])
        method = "spot_type"
    else:
        centroids, lab = kmeans2(
            data["velocity"].astype(np.float64), 3, minit="++", seed=42)
        order = np.argsort(centroids)
        remap = np.empty(3, dtype=int)
        remap[order] = np.arange(3)
        labels = remap[lab]
        method = "k-means on velocity"

    data["velocity_frame"] = _GALAXY_VELOCITY_FRAME.get(galaxy, "unknown")
    data["is_highvel"] = labels != 1

    # Per-spot phi bounds
    phi_lo = np.empty(n)
    phi_hi = np.empty(n)
    for k in range(n):
        if labels[k] == 0:    # blue (approaching)
            phi_lo[k], phi_hi[k] = np.pi, 2 * np.pi
        elif labels[k] == 1:  # systemic
            phi_lo[k], phi_hi[k] = -0.5 * np.pi, 0.5 * np.pi
        else:                 # red (receding)
            phi_lo[k], phi_hi[k] = 0.0, np.pi
    data["phi_lo"] = phi_lo
    data["phi_hi"] = phi_hi
    data["is_blue"] = (labels == 0)

    n_sys = int((labels == 1).sum())
    n_blue = int((labels == 0).sum())
    n_red = int((labels == 2).sum())
    fprint(f"classified spots: {n_sys} systemic, {n_blue} blue, "
           f"{n_red} red ({method}).")

    return data
