#!/usr/bin/env python3
"""Run Mark Reid's ``fit_disk`` MCMC from generated CANDEL inputs.

The Reid Fortran code is intentionally treated as read-only.  This script
creates a separate run directory, writes the hardwired input filenames that
``fit_disk`` expects, compiles the original source into that directory, runs it
there, and post-processes the resulting chain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[3]
REID_DIR = ROOT / "background_info/fit_disk_Reid"
REID_SOURCE = REID_DIR / "fit_disk_v24d_unblinded.f"
REID_CONTROL_TEMPLATE = REID_DIR / "fit_disk_control.inp"
DEFAULT_CONFIG = ROOT / "scripts/megamaser/config_maser.toml"
DEFAULT_DATA = ROOT / "data/Megamaser/N4258_disk_data_MarkReid.final"
DEFAULT_RESULTS = ROOT / "results/Megamaser/reid_mcmc"
DEFAULT_REID_INIT = ROOT / "scripts/megamaser/check_reid/reid_ngc4258_init.toml"

GLOBAL_NAMES = [
    "H0",
    "Mbh_1e7Msun",
    "Vsys_km_s",
    "x0_mas",
    "y0_mas",
    "i0_deg",
    "di_dr_deg_mas",
    "d2i_dr2_deg_mas2",
    "PA_deg",
    "dPA_dr_deg_mas",
    "d2PA_dr2_deg_mas2",
    "ecc",
    "peri_az_deg",
    "dperi_dr_deg_mas",
    "Vcor_km_s",
    "sigma_x_mas",
    "sigma_y_mas",
    "sigma_vsys_km_s",
    "sigma_vhv_km_s",
    "sigma_acc_km_s_yr",
]

DEFAULT_CONTOUR_PARAMS = [
    "H0",
    "Mbh_1e7Msun",
    "Vsys_km_s",
    "i0_deg",
    "PA_deg",
    "ecc",
    "peri_az_deg",
    "D_Mpc",
    "lnP",
]


@dataclass
class ReidInit:
    values: dict[str, float]
    source: str


def load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def first_data_header(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) >= 8 and parts[7].lower().startswith(("r", "o")):
            return stripped
        return None
    return None


def prepared_data_text(data_path: Path, fallback_header: str) -> str:
    """Return a Reid-readable data file without modifying the source data."""
    text = data_path.read_text()
    if first_data_header(text) is not None:
        return text

    # The repository NGC4258 file keeps the Reid data-parameter line commented.
    # Copy it into the generated input as the first non-comment line.
    header_match = re.search(
        r"^!\s*((?:[-+0-9.eEdD]+\s+){7}(?:Radio|Optical|R|O))\s*$",
        text,
        flags=re.MULTILINE,
    )
    header = header_match.group(1) if header_match else fallback_header
    return f"{header}\n{text}"


def parse_data_rows(data_path: Path) -> tuple[dict[str, float | str], np.ndarray]:
    text = prepared_data_text(data_path, "300 700 0.02 0.03 0.01 0.01 0.3 Radio")
    rows: list[list[float]] = []
    header: dict[str, float | str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if header is None:
            header = {
                "Vmin": float(parts[0]),
                "Vmax": float(parts[1]),
                "x_floor": float(parts[2]),
                "y_floor": float(parts[3]),
                "Vsys_floor": float(parts[4]),
                "Vhv_floor": float(parts[5]),
                "A_floor": float(parts[6]),
                "velocity_flag": parts[7],
            }
            continue
        rows.append([float(x) for x in parts[:9]])
    if header is None or not rows:
        raise ValueError(f"Could not parse Reid data rows from {data_path}")
    return header, np.asarray(rows, dtype=float)


def radio_to_optical(v_radio: np.ndarray | float) -> np.ndarray | float:
    c_km_s = 299792.5
    return c_km_s * (v_radio / (c_km_s - v_radio))


def load_reid_init(path: Path, vcor: float | None = None) -> ReidInit:
    init = load_toml(path)["globals"]
    values = {name: float(init[name]) for name in GLOBAL_NAMES}
    if vcor is not None:
        values["Vcor_km_s"] = float(vcor)
    values.update({"_D_c": (values["Vsys_km_s"] + values["Vcor_km_s"]) / values["H0"],
                   "_r_ref_i": 0.0, "_r_ref_PA": 0.0, "_r_ref_peri": 0.0})
    return ReidInit(values=values, source=f"reid-init:{path}")


def load_config_init(config_path: Path, galaxy: str, vcor: float) -> ReidInit:
    cfg = load_toml(config_path)
    gcfg = cfg["model"]["galaxies"][galaxy]
    init = dict(gcfg["init"])
    v_sys = float(gcfg["v_sys_obs"]) + float(init.get("dv_sys", 0.0))
    distance = float(init["D_c"])
    eta = float(init["eta"])
    m_bh = 10.0 ** (eta + math.log10(distance) - 7.0)
    h0 = (v_sys + vcor) / distance

    ecc = float(init.get("ecc", 0.0))
    peri = float(init.get("periapsis", 0.0))
    if "e_x" in init and "e_y" in init:
        ex = float(init["e_x"])
        ey = float(init["e_y"])
        ecc = math.hypot(ex, ey)
        peri = math.degrees(math.atan2(ey, ex)) % 360.0

    return ReidInit(
        values={
            "H0": h0,
            "Mbh_1e7Msun": m_bh,
            "Vsys_km_s": v_sys,
            "x0_mas": float(init.get("x0", 0.0)) / 1000.0,
            "y0_mas": float(init.get("y0", 0.0)) / 1000.0,
            "i0_deg": float(init.get("i0", 86.0)),
            "di_dr_deg_mas": float(init.get("di_dr", 0.0)),
            "d2i_dr2_deg_mas2": float(init.get("d2i_dr2", 0.0)),
            "PA_deg": float(init.get("Omega0", 89.0)),
            "dPA_dr_deg_mas": float(init.get("dOmega_dr", 0.0)),
            "d2PA_dr2_deg_mas2": float(init.get("d2Omega_dr2", 0.0)),
            "ecc": ecc,
            "peri_az_deg": peri,
            "dperi_dr_deg_mas": float(init.get("dperiapsis_dr", 0.0)),
            "Vcor_km_s": vcor,
            "sigma_x_mas": float(init.get("sigma_x_floor", 2.0)) / 1000.0,
            "sigma_y_mas": float(init.get("sigma_y_floor", 20.0)) / 1000.0,
            "sigma_vsys_km_s": float(init.get("sigma_v_sys", 0.5)),
            "sigma_vhv_km_s": float(init.get("sigma_v_hv", 1.0)),
            "sigma_acc_km_s_yr": float(init.get("sigma_a_floor", 0.4)),
            "_D_c": distance,
            "_r_ref_i": float(gcfg.get("r_ang_ref_i", init.get("r_ang_ref", 0.0))),
            "_r_ref_PA": float(
                gcfg.get("r_ang_ref_Omega", init.get("r_ang_ref", 0.0))
            ),
            "_r_ref_peri": float(
                gcfg.get("r_ang_ref_periapsis", init.get("r_ang_ref", 0.0))
            ),
        },
        source=f"config:{config_path}",
    )


def load_de_npz_init(path: Path, config_path: Path, galaxy: str, vcor: float) -> ReidInit:
    base = load_config_init(config_path, galaxy, vcor)
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    lo = np.asarray(z["lo"], dtype=float)
    hi = np.asarray(z["hi"], dtype=float)
    x = np.asarray(z["best_solution"], dtype=float)
    opt = {name: float(lo[i] + x[i] * (hi[i] - lo[i])) for i, name in enumerate(names)}

    cfg = load_toml(config_path)
    gcfg = cfg["model"]["galaxies"][galaxy]
    v_sys = float(gcfg["v_sys_obs"]) + float(opt.get("dv_sys", base.values["Vsys_km_s"] - gcfg["v_sys_obs"]))
    distance = float(opt.get("D_c", base.values["_D_c"]))
    eta = float(opt.get("eta", math.log10(base.values["Mbh_1e7Msun"] * 1e7 / distance)))
    m_bh = 10.0 ** (eta + math.log10(distance) - 7.0)
    h0 = (v_sys + vcor) / distance

    merged = dict(base.values)
    merged.update(
        {
            "H0": h0,
            "Mbh_1e7Msun": m_bh,
            "Vsys_km_s": v_sys,
            "x0_mas": float(opt.get("x0", merged["x0_mas"] * 1000.0)) / 1000.0,
            "y0_mas": float(opt.get("y0", merged["y0_mas"] * 1000.0)) / 1000.0,
            "i0_deg": float(opt.get("i0", merged["i0_deg"])),
            "di_dr_deg_mas": float(opt.get("di_dr", merged["di_dr_deg_mas"])),
            "d2i_dr2_deg_mas2": float(opt.get("d2i_dr2", merged["d2i_dr2_deg_mas2"])),
            "PA_deg": float(opt.get("Omega0", merged["PA_deg"])),
            "dPA_dr_deg_mas": float(opt.get("dOmega_dr", merged["dPA_dr_deg_mas"])),
            "d2PA_dr2_deg_mas2": float(
                opt.get("d2Omega_dr2", merged["d2PA_dr2_deg_mas2"])
            ),
            "Vcor_km_s": vcor,
            "sigma_x_mas": float(opt.get("sigma_x_floor", merged["sigma_x_mas"] * 1000.0))
            / 1000.0,
            "sigma_y_mas": float(opt.get("sigma_y_floor", merged["sigma_y_mas"] * 1000.0))
            / 1000.0,
            "sigma_vsys_km_s": float(opt.get("sigma_v_sys", merged["sigma_vsys_km_s"])),
            "sigma_vhv_km_s": float(opt.get("sigma_v_hv", merged["sigma_vhv_km_s"])),
            "sigma_acc_km_s_yr": float(
                opt.get("sigma_a_floor", merged["sigma_acc_km_s_yr"])
            ),
            "_D_c": distance,
        }
    )
    if "e_x" in opt and "e_y" in opt:
        merged["ecc"] = math.hypot(opt["e_x"], opt["e_y"])
        merged["peri_az_deg"] = math.degrees(math.atan2(opt["e_y"], opt["e_x"])) % 360.0
    else:
        merged["ecc"] = float(opt.get("ecc", merged["ecc"]))
        merged["peri_az_deg"] = float(opt.get("periapsis", merged["peri_az_deg"]))
    merged["dperi_dr_deg_mas"] = float(
        opt.get("dperiapsis_dr", merged["dperi_dr_deg_mas"])
    )
    return ReidInit(values=merged, source=f"de-npz:{path}")


def compute_reid_r_ref(rows: np.ndarray, header: dict[str, float | str], init: dict[str, float]) -> float:
    v = rows[:, 1]
    x = rows[:, 3]
    y = rows[:, 5]
    if str(header["velocity_flag"]).lower().startswith("r"):
        v = radio_to_optical(v)
    hv = (v < float(header["Vmin"])) | (v > float(header["Vmax"]))
    r = np.hypot(x[hv] - init["x0_mas"], y[hv] - init["y0_mas"])
    if not len(r):
        raise ValueError("Cannot infer Reid r_ref: no high-velocity spots found")
    return float(np.mean(r))


def shift_warp_pivots(init: dict[str, float], reid_r_ref: float) -> dict[str, float]:
    out = dict(init)
    ri = float(init.get("_r_ref_i", 0.0))
    rpa = float(init.get("_r_ref_PA", 0.0))
    rperi = float(init.get("_r_ref_peri", 0.0))
    if ri:
        dr = reid_r_ref - ri
        out["i0_deg"] = (
            init["i0_deg"]
            + init["di_dr_deg_mas"] * dr
            + init["d2i_dr2_deg_mas2"] * dr * dr
        )
    if rpa:
        dr = reid_r_ref - rpa
        out["PA_deg"] = (
            init["PA_deg"]
            + init["dPA_dr_deg_mas"] * dr
            + init["d2PA_dr2_deg_mas2"] * dr * dr
        )
    if rperi:
        out["peri_az_deg"] = init["peri_az_deg"] - init["dperi_dr_deg_mas"] * rperi
    return out


def template_control_lines() -> list[str]:
    return REID_CONTROL_TEMPLATE.read_text().splitlines()


def set_control_numbers(line: str, values: Iterable[float | int | str]) -> str:
    suffix = ""
    if "!" in line:
        suffix = " " + line[line.index("!") :]
    fields = []
    for value in values:
        if isinstance(value, str):
            fields.append(value)
        elif isinstance(value, int):
            fields.append(f"{value:d}")
        else:
            fields.append(f"{value:.8g}")
    return " ".join(f"{x:>12s}" for x in fields) + suffix


def write_control(
    path: Path,
    init: dict[str, float],
    *,
    burnin: int,
    trials: int,
    walkers: int,
    h0_low: float,
    h0_high: float,
    seed: int,
    step_fraction: float,
    fit_data: tuple[bool, bool, bool, bool],
) -> None:
    lines = template_control_lines()
    lines[1] = set_control_numbers(lines[1], [burnin])
    lines[2] = set_control_numbers(lines[2], [trials, walkers, h0_low, h0_high])
    lines[3] = set_control_numbers(
        lines[3], [0, 0.5, *(("T" if x else "F") for x in fit_data)]
    )
    lines[4] = set_control_numbers(lines[4], [step_fraction, -abs(seed)])
    global_rows = [
        ("H0",),
        ("Mbh_1e7Msun",),
        ("Vsys_km_s",),
        ("x0_mas",),
        ("y0_mas",),
        ("i0_deg",),
        ("di_dr_deg_mas",),
        ("d2i_dr2_deg_mas2",),
        ("PA_deg",),
        ("dPA_dr_deg_mas",),
        ("d2PA_dr2_deg_mas2",),
        ("ecc",),
        ("peri_az_deg",),
        ("dperi_dr_deg_mas",),
        ("Vcor_km_s",),
        ("sigma_x_mas",),
        ("sigma_y_mas",),
        ("sigma_vsys_km_s",),
        ("sigma_vhv_km_s",),
        ("sigma_acc_km_s_yr",),
    ]
    for i, (name,) in enumerate(global_rows, start=5):
        parts = lines[i].split("!", 1)[0].split()
        prior = float(parts[1])
        post = float(parts[2])
        lines[i] = set_control_numbers(lines[i], [init[name], prior, post])
    path.write_text("\n".join(lines) + "\n")


def global_template_prior_unc() -> dict[str, float]:
    lines = template_control_lines()
    out: dict[str, float] = {}
    for i, name in enumerate(GLOBAL_NAMES, start=5):
        parts = lines[i].split("!", 1)[0].split()
        out[name] = float(parts[1])
    return out


def initial_r_phi(rows: np.ndarray, header: dict[str, float | str], init: dict[str, float]) -> list[tuple[float, float, float, float]]:
    v = rows[:, 1].copy()
    acc = rows[:, 7]
    if str(header["velocity_flag"]).lower().startswith("r"):
        v = radio_to_optical(v)
    D = (init["Vsys_km_s"] + init["Vcor_km_s"]) / init["H0"]
    bh_mass = init["Mbh_1e7Msun"] * 1e7
    vmin = float(header["Vmin"])
    vmax = float(header["Vmax"])
    out: list[tuple[float, float, float, float]] = []
    g_cgs = 6.67e-8
    sun_mass = 1.98892e33
    au_km = 1.496e8
    au_cm = au_km * 1e5
    aearth = g_cgs * sun_mass / au_cm**2 * 1e-5 * 365.2422 * 86400.0
    vearth = 29.785
    for row, vv, aa in zip(rows, v, acc):
        systemic = vmin <= vv <= vmax
        if systemic:
            aabs = abs(float(aa)) if abs(float(aa)) > 1e-6 else 8.0
            r_au = math.sqrt(bh_mass / (aabs / aearth))
            r_mas = 1e-3 * r_au / D
            dv = vv - init["Vsys_km_s"]
            vcirc = vearth * math.sqrt(bh_mass / r_au)
            phi = math.degrees(dv / vcirc)
            sigma_phi = max(1.0, math.degrees(abs(0.5) / vcirc))
            post_phi = 0.5
            post_r = 0.1
        else:
            r_mas = math.hypot(row[3] - init["x0_mas"], row[5] - init["y0_mas"])
            phi = 90.0 if vv > vmax else -90.0
            sigma_phi = 20.0
            post_phi = 5.0
            post_r = 0.005
        out.append((r_mas, post_r, phi, post_phi if sigma_phi > 0 else 0.5))
    return out


def write_burnin_values(path: Path, init: dict[str, float], rphi: list[tuple[float, float, float, float]]) -> None:
    prior_unc = global_template_prior_unc()
    with path.open("w") as f:
        f.write("! Ho value shifted down; reconstruct using  0.000000\n")
        for name in GLOBAL_NAMES:
            post = 0.001 if name == "ecc" else 0.1
            if name == "H0":
                post = 0.7
            elif name == "Mbh_1e7Msun":
                post = 0.05
            elif name.endswith("_mas") or name.endswith("_mas2"):
                post = 0.005
            if prior_unc[name] == 0.0:
                post = 0.0
            f.write(f"{init[name]:12.6f}{post:12.6f}{post:12.6f}\n")
        for r_mas, post_r, phi, post_phi in rphi:
            f.write(f"{r_mas:12.6f}{post_r:12.6f}{post_r:12.6f}\n")
            f.write(f"{phi:12.6f}{post_phi:12.6f}{post_phi:12.6f}\n")


def default_status_interval(trials: int) -> int:
    return max(100000, min(10000000, max(1, trials // 100)))


def instrument_reid_source(run_dir: Path, status_interval: int) -> Path:
    """Write a per-run Reid source copy with less-sparse progress prints."""
    text = REID_SOURCE.read_text()
    status_literal = f"{int(status_interval):d}"

    text = text.replace(
        "         if ( mod(iter,10000000) .eq. 0 )",
        f"         if ( mod(iter,{status_literal}) .eq. 0 )",
        1,
    )

    secondary_marker = "      do ib2 = 1, ib2_max\n\n         do n_w = 1, num_walkers"
    secondary_patch = (
        "      do ib2 = 1, ib2_max\n\n"
        f"         if ( mod(ib2,{status_literal}) .eq. 0 )\n"
        "     +        write (lu_print,1171) ib2, ib2_max\n"
        " 1171    format(' Completed',i13,' of',i13,\n"
        "     +          ' secondary burnin trials.')\n\n"
        "         do n_w = 1, num_walkers"
    )
    if secondary_marker not in text:
        raise RuntimeError("Could not find Reid secondary burn-in loop to instrument")
    text = text.replace(secondary_marker, secondary_patch, 1)

    out = run_dir / "fit_disk_v24d_unblinded_status.f"
    out.write_text(text)
    return out


def compile_reid(run_dir: Path, compiler: str, flags: str, status_interval: int) -> Path:
    exe = run_dir / "fit_disk_reid_v24d"
    source = instrument_reid_source(run_dir, status_interval)
    cmd = [compiler, *flags.split(), "-o", str(exe), str(source)]
    subprocess.run(cmd, cwd=run_dir, check=True)
    return exe


def run_reid(exe: Path, run_dir: Path) -> Path:
    stdout_path = run_dir / "fit_disk.stdout"
    with stdout_path.open("w") as out:
        cmd = [str(exe)]
        if shutil.which("stdbuf") is not None:
            cmd = ["stdbuf", "-oL", "-eL", *cmd]
        proc = subprocess.Popen(
            cmd,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            out.write(line)
            out.flush()
            print(line, end="", flush=True)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
    return stdout_path


def parse_fortran_float(value: bytes | str) -> float:
    """Parse Reid/Fortran numeric output, including D exponent notation."""
    if isinstance(value, bytes):
        text = value.decode("ascii")
    else:
        text = value
    return float(text.replace("D", "E").replace("d", "e"))


def load_chain(path: Path) -> np.ndarray:
    expected = 2 + len(GLOBAL_NAMES) + 1
    data = np.genfromtxt(
        path,
        comments="!",
        converters={expected - 1: parse_fortran_float},
    )
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != expected:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected {expected}")
    dtype = [("iter", "i8"), ("walker", "i4")]
    dtype += [(name, "f8") for name in GLOBAL_NAMES]
    dtype += [("lnP", "f8")]
    arr = np.empty(data.shape[0], dtype=dtype)
    arr["iter"] = data[:, 0].astype(np.int64)
    arr["walker"] = data[:, 1].astype(np.int32)
    for i, name in enumerate(GLOBAL_NAMES, start=2):
        arr[name] = data[:, i]
    arr["lnP"] = data[:, -1]
    arr = append_distance(arr)
    return arr


def append_distance(arr: np.ndarray) -> np.ndarray:
    dtype = arr.dtype.descr + [("D_Mpc", "f8")]
    out = np.empty(arr.shape, dtype=dtype)
    for name in arr.dtype.names:
        out[name] = arr[name]
    out["D_Mpc"] = (arr["Vsys_km_s"] + arr["Vcor_km_s"]) / arr["H0"]
    return out


def best_index(arr: np.ndarray) -> int | None:
    finite = np.isfinite(arr["lnP"])
    if not np.any(finite):
        return None
    idx = np.flatnonzero(finite)
    return int(idx[np.argmax(arr["lnP"][finite])])


def write_chain_csv(arr: np.ndarray, path: Path) -> None:
    names = list(arr.dtype.names or [])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for row in arr:
            writer.writerow([row[name].item() for name in names])


def summary_from_chain(arr: np.ndarray, stdout_path: Path) -> dict:
    best_i = best_index(arr)
    finite_lnP = arr["lnP"][np.isfinite(arr["lnP"])]
    summary = {
        "n_stored": int(len(arr)),
        "best_stored": (
            {name: float(arr[name][best_i]) for name in arr.dtype.names or []}
            if best_i is not None else {}
        ),
        "lnP": {
            "max": float(np.max(finite_lnP)) if len(finite_lnP) else math.nan,
            "median": float(np.median(finite_lnP)) if len(finite_lnP) else math.nan,
            "p05": float(np.percentile(finite_lnP, 5)) if len(finite_lnP) else math.nan,
            "p95": float(np.percentile(finite_lnP, 95)) if len(finite_lnP) else math.nan,
        },
        "parameters": {},
        "stdout": str(stdout_path),
    }
    for name in GLOBAL_NAMES + ["D_Mpc"]:
        x = arr[name]
        summary["parameters"][name] = {
            "median": float(np.nanmedian(x)),
            "p16": float(np.nanpercentile(x, 16)),
            "p84": float(np.nanpercentile(x, 84)),
            "p025": float(np.nanpercentile(x, 2.5)),
            "p975": float(np.nanpercentile(x, 97.5)),
        }
    text = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    for key, pattern in {
        "acceptance_percent": r"Percent trials accepted.*?([0-9.]+)%",
        "best_lnP_reported": r"MCMC trials had best ln\(Probability\) =\s*([-+0-9.Ee]+)",
        "downhill_best_lnP": r"Best global parameter values with ln\(prob\)=\s*([-+0-9.Ee]+)",
    }.items():
        m = re.search(pattern, text)
        if m:
            summary[key] = float(m.group(1))
    return summary


def plot_corner(arr: np.ndarray, params: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = [p for p in params if p in arr.dtype.names]
    n = len(params)
    best_i = best_index(arr)
    fig, axes = plt.subplots(n, n, figsize=(1.85 * n, 1.85 * n), squeeze=False)
    for i, yname in enumerate(params):
        y = arr[yname]
        for j, xname in enumerate(params):
            ax = axes[i, j]
            if i == j:
                ax.hist(y, bins=40, color="#4c72b0", histtype="stepfilled", alpha=0.75)
            elif i > j:
                x = arr[xname]
                ax.hist2d(x, y, bins=45, cmap="Blues")
                if best_i is not None:
                    ax.plot(
                        x[best_i],
                        y[best_i],
                        marker="x",
                        color="crimson",
                        markersize=4,
                    )
            else:
                ax.axis("off")
                continue
            if i == n - 1:
                ax.set_xlabel(xname, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != j:
                ax.set_ylabel(yname, fontsize=8)
            elif j != 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6, length=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def numpyro_summary_text(samples: dict[str, np.ndarray]) -> str:
    from numpyro.diagnostics import summary as numpyro_summary

    stats = numpyro_summary(samples, prob=0.9, group_by_chain=True)
    name_width = max(9, *(len(name) for name in stats))
    lines = [
        f"{'':{name_width}s} {'mean':>10s} {'std':>10s} {'median':>10s} "
        f"{'5.0%':>10s} {'95.0%':>10s} {'n_eff':>10s} {'r_hat':>10s}"
    ]
    for name, values in stats.items():
        lines.append(
            f"{name:{name_width}s} "
            f"{values['mean']:10.4g} {values['std']:10.4g} "
            f"{values['median']:10.4g} {values['5.0%']:10.4g} "
            f"{values['95.0%']:10.4g} {values['n_eff']:10.1f} "
            f"{values['r_hat']:10.2f}"
        )
    return "\n".join(lines)


def collect_chain_batch(chain_dirs: list[Path], output_dir: Path, plot_params: list[str]) -> None:
    chains = []
    for chain_dir in chain_dirs:
        fort7 = chain_dir / "fort.7"
        if not fort7.exists():
            raise FileNotFoundError(f"Missing Reid chain file: {fort7}")
        chains.append(load_chain(fort7))

    if not chains:
        raise ValueError("No chain directories supplied for collection")

    min_draws = min(len(chain) for chain in chains)
    if min_draws < 1:
        raise ValueError("Cannot collect empty Reid chains")
    if any(len(chain) != min_draws for chain in chains):
        print(
            f"Truncating chains to common stored draw count: {min_draws}",
            flush=True,
        )

    summary_names = GLOBAL_NAMES + ["D_Mpc"]
    samples = {
        name: np.stack([chain[name][-min_draws:] for chain in chains], axis=0)
        for name in summary_names
    }
    summary_text = numpyro_summary_text(samples)
    summary_path = output_dir / "global_summary.txt"
    summary_path.write_text(summary_text + "\n")

    combined = np.concatenate([chain[-min_draws:] for chain in chains])
    corner_path = output_dir / "global_corner.png"
    plot_corner(combined, plot_params, corner_path)

    print("\nGlobal parameter summary (combined Reid chains):", flush=True)
    print(summary_text, flush=True)
    print(f"Global summary: {summary_path}", flush=True)
    print(f"Global corner plot: {corner_path}", flush=True)


def parse_bool_quad(value: str) -> tuple[bool, bool, bool, bool]:
    chars = value.replace(",", " ").split()
    if len(chars) == 1 and len(chars[0]) == 4:
        chars = list(chars[0])
    if len(chars) != 4:
        raise argparse.ArgumentTypeError("expected four booleans, e.g. T T T T or TTTF")
    return tuple(c.lower() in {"t", "true", "1", "yes", "y"} for c in chars)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and plot Mark Reid fit_disk MCMC without editing the Reid source."
    )
    parser.add_argument("--galaxy", default="NGC4258")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--init", choices=["reid", "config", "de-npz"], default="reid")
    parser.add_argument(
        "--reid-init",
        type=Path,
        default=DEFAULT_REID_INIT,
        help="TOML file with Reid-style global initial values for --init reid.",
    )
    parser.add_argument(
        "--init-npz",
        type=Path,
        default=ROOT / "results/Megamaser/de_checkpoints/NGC4258/de_ckpt.npz",
        help="DE checkpoint used when --init de-npz.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--burnin",
        type=int,
        default=1000000,
        help="Primary Reid burn-in trials. <=0 skips it via generated burnin_values.dat.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10000000,
        help="Final MCMC trials. Reid v24d requires >=500000 because n_skip=itermax/500000.",
    )
    parser.add_argument("--walkers", type=int, default=1)
    parser.add_argument("--h0-low", type=float, default=None)
    parser.add_argument("--h0-high", type=float, default=None)
    parser.add_argument("--vcor", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=47351937)
    parser.add_argument("--step-fraction", type=float, default=0.015)
    parser.add_argument(
        "--status-interval",
        type=int,
        default=10000,
        help="Progress print interval for secondary burn-in and final MCMC. "
             "Use 0 to choose max(100000, min(10000000, trials/100)).",
    )
    parser.add_argument("--fit-data", type=parse_bool_quad, default=(True, True, True, True))
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--fflags", default="-O2 -std=legacy -fno-automatic")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--plot-params", default=",".join(DEFAULT_CONTOUR_PARAMS))
    parser.add_argument(
        "--collect-chain-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Collect completed chain directories and write a combined summary/corner plot.",
    )
    args = parser.parse_args(argv)

    params = [x.strip() for x in args.plot_params.split(",") if x.strip()]
    if args.collect_chain_dirs is not None:
        if args.output_dir is None:
            parser.error("--output-dir is required with --collect-chain-dirs")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        collect_chain_batch(args.collect_chain_dirs, args.output_dir, params)
        return 0

    if args.trials < 500000:
        raise ValueError(
            "Reid v24d computes n_skip = itermax / 500000 with integer division; "
            "use --trials >= 500000 unless the Fortran source itself is changed."
        )
    status_interval = (
        default_status_interval(args.trials)
        if args.status_interval == 0 else int(args.status_interval)
    )
    if status_interval < 1:
        raise ValueError("--status-interval must be >=1, or 0 for automatic")

    if args.init == "reid":
        reid_init = load_reid_init(args.reid_init, args.vcor)
    elif args.init == "config":
        reid_init = load_config_init(args.config, args.galaxy, args.vcor)
    else:
        reid_init = load_de_npz_init(args.init_npz, args.config, args.galaxy, args.vcor)

    header, data_rows = parse_data_rows(args.data)
    run_init = shift_warp_pivots(reid_init.values, compute_reid_r_ref(data_rows, header, reid_init.values))

    h0_low = args.h0_low
    h0_high = args.h0_high
    if h0_low is None or h0_high is None:
        h0 = run_init["H0"]
        h0_low = h0 - 15.0 if h0_low is None else h0_low
        h0_high = h0 + 15.0 if h0_high is None else h0_high

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = DEFAULT_RESULTS / f"{args.galaxy}_{args.init}_{stamp}"
    else:
        run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "fit_disk_data.inp").write_text(prepared_data_text(args.data, "300 700 0.02 0.03 0.01 0.01 0.3 Radio"))
    write_control(
        run_dir / "fit_disk_control.inp",
        run_init,
        burnin=args.burnin,
        trials=args.trials,
        walkers=args.walkers,
        h0_low=float(h0_low),
        h0_high=float(h0_high),
        seed=args.seed,
        step_fraction=args.step_fraction,
        fit_data=args.fit_data,
    )
    if args.burnin <= 0:
        write_burnin_values(run_dir / "burnin_values.dat", run_init, initial_r_phi(data_rows, header, run_init))

    metadata = {
        "galaxy": args.galaxy,
        "run_dir": str(run_dir),
        "reid_source": str(REID_SOURCE),
        "data_source": str(args.data),
        "init_source": reid_init.source,
        "burnin": args.burnin,
        "trials": args.trials,
        "walkers": args.walkers,
        "status_interval": status_interval,
        "h0_low": h0_low,
        "h0_high": h0_high,
        "initial_globals": {name: run_init[name] for name in GLOBAL_NAMES},
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Prepared Reid run directory: {run_dir}", flush=True)
    if args.prepare_only:
        return 0

    exe = run_dir / "fit_disk_reid_v24d"
    if not args.no_compile:
        exe = compile_reid(run_dir, args.compiler, args.fflags, status_interval)
        print(f"Compiled Reid executable: {exe}", flush=True)
    elif not exe.exists():
        raise FileNotFoundError(f"--no-compile requested but executable is missing: {exe}")

    if not args.no_run:
        stdout = run_reid(exe, run_dir)
        print(f"Reid stdout: {stdout}", flush=True)
    else:
        stdout = run_dir / "fit_disk.stdout"

    fort7 = run_dir / "fort.7"
    if not fort7.exists():
        print(f"No chain file found at {fort7}; skipping post-processing.", file=sys.stderr)
        return 0

    chain = load_chain(fort7)
    write_chain_csv(chain, run_dir / "global_chain.csv")
    summary = summary_from_chain(chain, stdout)
    (run_dir / "likelihood_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_corner(chain, params, run_dir / "global_corner.png")
    print(f"Chain CSV: {run_dir / 'global_chain.csv'}", flush=True)
    print(f"Likelihood summary: {run_dir / 'likelihood_summary.json'}", flush=True)
    print(f"Global contour plot: {run_dir / 'global_corner.png'}", flush=True)
    print(f"Best stored lnP: {summary['lnP']['max']:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
