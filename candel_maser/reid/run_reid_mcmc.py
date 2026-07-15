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
SCRIPT_DIR = Path(__file__).resolve().parent
REID_DIR = ROOT / "background_info/fit_disk_Reid"
REID_SOURCE = REID_DIR / "fit_disk_v24d_unblinded.f"
REID_CONTROL_TEMPLATE = REID_DIR / "fit_disk_control.inp"
DEFAULT_CONFIG = ROOT / "scripts/megamaser/config_maser.toml"
DEFAULT_DATA = ROOT / "data/Megamaser/N4258_disk_data_MarkReid.final"
DEFAULT_RESULTS = ROOT / "results/Megamaser/reid_mcmc"
DEFAULT_REID_INIT = ROOT / "scripts/megamaser/check_reid/reid_ngc4258_init.toml"  # noqa: E501
MAX_CORNER_SAMPLES = 20000
FORT7_WIDTHS = [
    10, 6, 10, 10, 9, 9, 9, 8, 8, 8, 8, 8, 8, 6,
    8, 8, 8, 8, 8, 8, 8, 8, 13,
]
FORT7_ROW_WIDTH = sum(FORT7_WIDTHS)

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

DEFAULT_CONTOUR_PARAMS = [*GLOBAL_NAMES, "D_Mpc"]

assert len(FORT7_WIDTHS) == 2 + len(GLOBAL_NAMES) + 1, (
    "FORT7_WIDTHS must have one entry per fort.7 column (iter, walker, "
    "GLOBAL_NAMES..., lnP)")

# (prior_unc, post_unc) control-file columns used when a template-fixed global
# is unfrozen via --free-params. prior_unc < 0 keeps Reid's flat-prior
# convention (so the prior is unchanged); the step mirrors the analogous
# quadratic-warp term d2PA/dr2 in the template.
FREE_DEFAULTS = {
    "d2i_dr2_deg_mas2": (-1.0, 0.015),
}

PARAM_LABELS = {
    "H0": r"$H_0\ [{\rm km\ s^{-1}\ Mpc^{-1}}]$",
    "Mbh_1e7Msun": r"$M_\bullet\ [10^7\,M_\odot]$",
    "Vsys_km_s": r"$V_{\rm sys}\ [{\rm km\ s^{-1}}]$",
    "x0_mas": r"$x_0\ [{\rm mas}]$",
    "y0_mas": r"$y_0\ [{\rm mas}]$",
    "i0_deg": r"$i_0\ [{\rm deg}]$",
    "di_dr_deg_mas": r"$di/dr\ [{\rm deg\ mas^{-1}}]$",
    "d2i_dr2_deg_mas2": r"$d^2 i/dr^2\ [{\rm deg\ mas^{-2}}]$",
    "PA_deg": r"${\rm PA}\ [{\rm deg}]$",
    "dPA_dr_deg_mas": r"$d{\rm PA}/dr\ [{\rm deg\ mas^{-1}}]$",
    "d2PA_dr2_deg_mas2": r"$d^2{\rm PA}/dr^2\ [{\rm deg\ mas^{-2}}]$",
    "ecc": r"$e$",
    "peri_az_deg": r"$\omega\ [{\rm deg}]$",
    "dperi_dr_deg_mas": r"$d\omega/dr\ [{\rm deg\ mas^{-1}}]$",
    "Vcor_km_s": r"$V_{\rm cor}\ [{\rm km\ s^{-1}}]$",
    "sigma_x_mas": r"$\sigma_x\ [{\rm mas}]$",
    "sigma_y_mas": r"$\sigma_y\ [{\rm mas}]$",
    "sigma_vsys_km_s": r"$\sigma_{v,{\rm sys}}\ [{\rm km\ s^{-1}}]$",
    "sigma_vhv_km_s": r"$\sigma_{v,{\rm hv}}\ [{\rm km\ s^{-1}}]$",
    "sigma_acc_km_s_yr": r"$\sigma_a\ [{\rm km\ s^{-1}\ yr^{-1}}]$",
    "D_Mpc": r"$D\ [{\rm Mpc}]$",
}


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


def parse_data_rows(
        data_path: Path) -> tuple[dict[str, float | str], np.ndarray]:
    text = prepared_data_text(
        data_path, "300 700 0.02 0.03 0.01 0.01 0.3 Radio")
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


def load_reid_init(path: Path, vcor: float | None = None,
                   galaxy: str | None = None,
                   variant: str = "init") -> ReidInit:
    init = load_toml(path)["globals"]
    tag = str(path)
    if any(isinstance(v, dict) for v in init.values()):
        # Merged multi-galaxy file: [globals.<GALAXY>.<variant>].
        if not isinstance(init.get(galaxy), dict):
            available = sorted(k for k, v in init.items()
                               if isinstance(v, dict))
            raise ValueError(
                f"{path} is a merged [globals.<GALAXY>.<variant>] file "
                f"with no [globals.{galaxy}]; available galaxies: "
                f"{available}")
        by_variant = init[galaxy]
        if variant not in by_variant:
            raise ValueError(
                f"{path} has no [globals.{galaxy}.{variant}]; "
                f"available variants: {sorted(by_variant)}")
        init = by_variant[variant]
        tag = f"{path}:{galaxy}:{variant}"
    values = {name: float(init[name]) for name in GLOBAL_NAMES}
    if vcor is not None:
        values["Vcor_km_s"] = float(vcor)
    values.update(
        {
            "_D_c": (
                values["Vsys_km_s"] +
                values["Vcor_km_s"]) /
            values["H0"],
            "_r_ref_i": 0.0,
            "_r_ref_PA": 0.0,
            "_r_ref_peri": 0.0})
    return ReidInit(values=values, source=f"reid-init:{tag}")


def resolve_init_toml(name: str) -> Path:
    path = Path(name)
    if path.name != name or path.is_absolute():
        raise ValueError(
            "--init must be a TOML filename in scripts/megamaser/check_reid")
    if path.suffix != ".toml":
        raise ValueError("--init must end with .toml")
    path = SCRIPT_DIR / path.name
    if not path.exists():
        raise FileNotFoundError(f"Missing init TOML: {path}")
    return path


def load_galaxy_config(config_path: Path, galaxy: str) -> dict:
    cfg = load_toml(config_path)
    gcfg = dict(cfg["model"]["galaxies"][galaxy])
    if "v_sys_obs" in gcfg:
        return gcfg

    default_cfg = load_toml(DEFAULT_CONFIG)
    default_gcfg = dict(default_cfg["model"]["galaxies"][galaxy])
    default_gcfg.update(gcfg)
    return default_gcfg


def load_toml_init(path: Path, galaxy: str, vcor: float,
                   variant: str = "init") -> ReidInit:
    cfg = load_toml(path)
    if "globals" in cfg:
        return load_reid_init(path, vcor, galaxy=galaxy, variant=variant)
    try:
        cfg["model"]["galaxies"][galaxy][variant]
    except KeyError as exc:
        raise ValueError(
            f"{path} must contain [globals] or "
            f"[model.galaxies.{galaxy}.{variant}]"
        ) from exc
    return load_config_init(path, galaxy, vcor, variant=variant)


def load_config_init(config_path: Path, galaxy: str, vcor: float,
                     variant: str = "init") -> ReidInit:
    gcfg = load_galaxy_config(config_path, galaxy)
    init = dict(gcfg[variant])
    v_sys = float(gcfg["v_sys_obs"]) + float(init.get("dv_sys", 0.0))
    distance = float(init["D_c"])
    m_bh = 10.0 ** (float(init["log_MBH"]) - 7.0)
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
            # Reid inclination convention: i_Reid = 180 - i_CANDEL; the di/dr
            # and d2i/dr2 warp gradients are correspondingly sign-flipped.
            "i0_deg": 180.0 - float(init.get("i0", 94.0)),
            "di_dr_deg_mas": -float(init.get("di_dr", 0.0)),
            "d2i_dr2_deg_mas2": -float(init.get("d2i_dr2", 0.0)),
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
            "_r_ref_i": float(gcfg.get("r_ang_ref_i", init.get("r_ang_ref", 0.0))),  # noqa: E501
            "_r_ref_PA": float(
                gcfg.get("r_ang_ref_Omega", init.get("r_ang_ref", 0.0))
            ),
            "_r_ref_peri": float(
                gcfg.get("r_ang_ref_periapsis", init.get("r_ang_ref", 0.0))
            ),
        },
        source=f"config:{config_path}:{variant}",
    )


def compute_reid_r_ref(rows: np.ndarray,
                       header: dict[str,
                                    float | str],
                       init: dict[str,
                                  float]) -> float:
    v = rows[:, 1]
    x = rows[:, 3]
    y = rows[:, 5]
    if str(header["velocity_flag"]).lower().startswith("r"):
        v = radio_to_optical(v)
    hv = (v < float(header["Vmin"])) | (v > float(header["Vmax"]))
    r = np.hypot(x[hv] - init["x0_mas"], y[hv] - init["y0_mas"])
    if not len(r):
        raise ValueError(
            "Cannot infer Reid r_ref: no high-velocity spots found")
    return float(np.mean(r))


def shift_warp_pivots(init: dict[str, float],
                      reid_r_ref: float) -> dict[str, float]:
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
        out["peri_az_deg"] = init["peri_az_deg"] - \
            init["dperi_dr_deg_mas"] * rperi
    return out


def template_control_lines() -> list[str]:
    return REID_CONTROL_TEMPLATE.read_text().splitlines()


def set_control_numbers(line: str, values: Iterable[float | int | str]) -> str:
    suffix = ""
    if "!" in line:
        suffix = " " + line[line.index("!"):]
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
    fixed_params: set[str] | None = None,
    free_params: set[str] | None = None,
) -> None:
    fixed_params = fixed_params or set()
    free_params = free_params or set()
    lines = template_control_lines()
    lines[1] = set_control_numbers(lines[1], [burnin])
    lines[2] = set_control_numbers(
        lines[2], [trials, walkers, h0_low, h0_high])
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
        if name in fixed_params:
            prior = 0.0
            post = 0.0
        elif name in free_params:
            prior, post = FREE_DEFAULTS.get(name, (-1.0, abs(post) or 0.05))
        lines[i] = set_control_numbers(lines[i], [init[name], prior, post])
    path.write_text("\n".join(lines) + "\n")


def initial_r_phi(rows: np.ndarray,
                  header: dict[str,
                               float | str],
                  init: dict[str,
                             float]) -> list[tuple[float,
                                                   float,
                                                   float,
                                                   float]]:
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
            r_mas = math.hypot(
                row[3] - init["x0_mas"],
                row[5] - init["y0_mas"])
            phi = 90.0 if vv > vmax else -90.0
            sigma_phi = 20.0
            post_phi = 5.0
            post_r = 0.005
        out.append((r_mas, post_r, phi, post_phi if sigma_phi > 0 else 0.5))
    return out


def write_burnin_values(path: Path,
                        init: dict[str,
                                   float],
                        rphi: list[tuple[float,
                                         float,
                                         float,
                                         float]],
                        fixed_params: set[str] | None = None) -> None:
    fixed_params = fixed_params or set()
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
            if name in fixed_params:
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

    secondary_marker = "      do ib2 = 1, ib2_max\n\n         do n_w = 1, num_walkers"  # noqa: E501
    secondary_patch = (
        "      do ib2 = 1, ib2_max\n\n"
        f"         if ( mod(ib2,{status_literal}) .eq. 0 )\n"
        "     +        write (lu_print,1171) ib2, ib2_max\n"
        " 1171    format(' Completed',i13,' of',i13,\n"
        "     +          ' secondary burnin trials.')\n\n"
        "         do n_w = 1, num_walkers"
    )
    if secondary_marker not in text:
        raise RuntimeError(
            "Could not find Reid secondary burn-in loop to instrument")
    text = text.replace(secondary_marker, secondary_patch, 1)

    out = run_dir / "fit_disk_v24d_unblinded_status.f"
    out.write_text(text)
    return out


def compile_reid(
        run_dir: Path,
        compiler: str,
        flags: str,
        status_interval: int,
        instrument: bool = False) -> Path:
    exe = run_dir / "fit_disk_reid_v24d"
    if instrument:
        source = instrument_reid_source(run_dir, status_interval)
    else:
        source = run_dir / REID_SOURCE.name
        shutil.copy2(REID_SOURCE, source)
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


def _coerce_float_column(col: np.ndarray) -> np.ndarray:
    """Cast a fixed-width byte column to float64, mapping unparseable
    entries (e.g. a Fortran f10.5-style field overflowing to '****' when
    a run's H0 wanders near zero) to NaN instead of raising -- matching
    genfromtxt's tolerant behaviour."""
    try:
        return col.astype(np.float64)
    except ValueError:
        out = np.full(col.shape, np.nan)
        for i, token in enumerate(col):
            try:
                out[i] = float(token)
            except ValueError:
                pass
        return out


def load_chain(path: Path) -> np.ndarray:
    """Parse a fort.7 chain file: fixed-width Fortran columns that can
    touch (e.g. a full-width negative number abutting the next field, so
    this is NOT whitespace-delimited). Vectorized byte-offset slicing
    instead of genfromtxt's per-row Python parsing -- ~5x faster on the
    million-row production chains this pipeline writes."""
    raw = Path(path).read_bytes().replace(b"D", b"E").replace(b"d", b"e")
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    # Drop comment lines and any partial row left by a job killed mid-write.
    lines = [ln for ln in lines
             if not ln.startswith(b"!") and len(ln) == FORT7_ROW_WIDTH]
    if not lines:
        raise ValueError(f"{path} has no complete fixed-width rows")
    n = len(lines)
    chars = np.array(lines, dtype=f"S{FORT7_ROW_WIDTH}").view("S1").reshape(
        n, FORT7_ROW_WIDTH)
    offsets = np.cumsum([0] + FORT7_WIDTHS)
    cols = [chars[:, s:e].view(f"S{e - s}").ravel()
            for s, e in zip(offsets[:-1], offsets[1:])]

    dtype = [("iter", "i8"), ("walker", "i4")]
    dtype += [(name, "f8") for name in GLOBAL_NAMES]
    dtype += [("lnP", "f8")]
    arr = np.empty(n, dtype=dtype)
    arr["iter"] = cols[0].astype(np.int64)
    arr["walker"] = cols[1].astype(np.int32)
    for i, name in enumerate(GLOBAL_NAMES, start=2):
        arr[name] = _coerce_float_column(cols[i])
    arr["lnP"] = _coerce_float_column(cols[-1])
    arr = append_distance(arr)
    return arr


def _reid_dnum(v):
    """The H0- and D_A-independent factor of Reid fit_disk's ``dampc`` mapping
    (flat LambdaCDM, Omega_m=0.27, Omega_L=0.73; Hogg 1999 eq. 14/18):
    ``c * eq14int(z) / (1+z)`` with z = v/c and v = Vsys+Vcor [km/s]. Both
    directions of the mapping fall out of it -- D_A = _reid_dnum(v)/H0 and its
    inverse H0 = _reid_dnum(v)/D_A.

    Vectorised: eq14int depends only on z, so it is precomputed on a shared
    z-grid and interpolated per sample (grid error <1e-6; agrees with the
    Fortran's per-draw D_A to ~6e-5, the residual being its integer-km/s Ez
    rounding this deliberately smooths), keeping million-row chains
    memory-safe."""
    c, Om, Ol = 299792.458, 0.27, 0.73
    v = np.asarray(v, dtype=np.float64)
    z = v / c
    zmax = float(np.nanmax(z)) if z.size else 0.0
    zg = np.linspace(0.0, zmax if zmax > 0.0 else 1e-6, 20001)
    inv_E = 1.0 / np.sqrt(Om * (1.0 + zg) ** 3 + Ol)
    eq14 = np.concatenate(
        ([0.0], np.cumsum(0.5 * (inv_E[1:] + inv_E[:-1]) * np.diff(zg))))
    return c * np.interp(z, zg, eq14) / (1.0 + z)


def reid_D_A(v, H0):
    """Angular-diameter distance D_A [Mpc] from recession velocity v =
    Vsys+Vcor [km/s] and H0, via Reid fit_disk's own ``dampc`` mapping. This
    is the exact H0->D_A relation the disk model uses internally, so applying
    it to the sampled H0 gives a Reid D_A directly comparable to CANDEL's
    sampled D_A -- unlike the naive Hubble ratio v/H0, which overshoots D_A by
    the cosmological (1+z)/E(z) factor (~3% at MCP redshifts)."""
    return _reid_dnum(v) / np.asarray(H0, dtype=np.float64)


def reid_H0(v, D_A):
    """Inverse of reid_D_A: the cosmological H0 that makes ``dampc`` map
    (v=Vsys+Vcor, H0) onto the given D_A. Applied to CANDEL's sampled D_A it
    puts CANDEL's H0 on the same cosmological footing as Reid's sampled H0,
    instead of the naive v/D_A -- which overshoots the true H0 by the same
    ~3% factor."""
    return _reid_dnum(v) / np.asarray(D_A, dtype=np.float64)


def append_distance(arr: np.ndarray) -> np.ndarray:
    dtype = arr.dtype.descr + [("D_Mpc", "f8")]
    out = np.empty(arr.shape, dtype=dtype)
    for name in arr.dtype.names:
        out[name] = arr[name]
    # D_A via the disk model's dampc mapping (not the naive v/H0), so Reid's
    # distance is the same angular-diameter distance CANDEL samples.
    out["D_Mpc"] = reid_D_A(arr["Vsys_km_s"] + arr["Vcor_km_s"], arr["H0"])
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


def write_lnp_csv(
        arr: np.ndarray,
        path: Path) -> None:
    names = ["iter", "walker", "lnP"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for row in arr:
            writer.writerow([
                row["iter"].item(),
                row["walker"].item(),
                row["lnP"].item()])


def lnp_summary(arr: np.ndarray) -> dict:
    best_i = best_index(arr)
    finite = arr["lnP"][np.isfinite(arr["lnP"])]
    return {
        "n_stored": int(len(arr)),
        "n_finite": int(len(finite)),
        "best_stored": (
            {
                "iter": int(arr["iter"][best_i]),
                "walker": int(arr["walker"][best_i]),
                "lnP": float(arr["lnP"][best_i]),
            }
            if best_i is not None else {}
        ),
        "max": float(np.max(finite)) if len(finite) else math.nan,
        "median": float(np.median(finite)) if len(finite) else math.nan,
        "p05": float(np.percentile(finite, 5)) if len(finite) else math.nan,
        "p95": float(np.percentile(finite, 95)) if len(finite) else math.nan,
    }


def summary_from_chain(arr: np.ndarray, stdout_path: Path) -> dict:
    best_i = best_index(arr)
    summary = {
        "n_stored": int(len(arr)),
        "best_stored": (
            {name: float(arr[name][best_i]) for name in arr.dtype.names or []}
            if best_i is not None else {}
        ),
        "lnP": lnp_summary(arr),
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
    text = stdout_path.read_text(
        errors="replace") if stdout_path.exists() else ""
    for key, pattern in {
        "acceptance_percent": r"Percent trials accepted.*?([0-9.]+)%",
        "best_lnP_reported": r"MCMC trials had best ln\(Probability\) =\s*([-+0-9.Ee]+)",  # noqa: E501
        "downhill_best_lnP": r"Best global parameter values with ln\(prob\)=\s*([-+0-9.Ee]+)",  # noqa: E501
    }.items():
        m = re.search(pattern, text)
        if m:
            summary[key] = float(m.group(1))
    return summary


def plot_corner(arr: np.ndarray, params: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import corner
    import matplotlib.pyplot as plt

    params = [p for p in params if p != "lnP" and p in arr.dtype.names]
    if not params:
        raise ValueError("No valid parameters selected for corner plot")

    data = np.column_stack([arr[name] for name in params])
    finite = np.all(np.isfinite(data), axis=1)
    data = data[finite]
    if len(data) == 0:
        raise ValueError("No finite samples available for corner plot")
    if len(data) > MAX_CORNER_SAMPLES:
        idx = np.linspace(0, len(data) - 1, MAX_CORNER_SAMPLES, dtype=int)
        data = data[idx]

    best_i = best_index(arr)
    truths = None
    if best_i is not None:
        truths = [arr[name][best_i] for name in params]

    if data.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.hist(data[:, 0], bins=40, color="0.25",
                histtype="stepfilled", alpha=0.65)
        ax.set_xlabel(PARAM_LABELS.get(params[0], params[0]))
        ax.set_ylabel("stored samples")
        if truths is not None:
            ax.axvline(truths[0], color="crimson")
            fig.text(
                0.995,
                0.995,
                r"red line: max stored $\ln P$ sample",
                ha="right",
                va="top",
                fontsize=10,
                color="crimson",
            )
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    ranges = []
    for j in range(data.shape[1]):
        lo = float(np.min(data[:, j]))
        hi = float(np.max(data[:, j]))
        if lo == hi:
            pad = max(abs(lo) * 1e-6, 1e-6)
            lo -= pad
            hi += pad
        ranges.append((lo, hi))

    fig = corner.corner(
        data,
        labels=[PARAM_LABELS.get(name, name) for name in params],
        range=ranges,
        bins=40,
        color="0.25",
        truths=truths,
        truth_color="crimson",
        plot_datapoints=False,
        fill_contours=True,
        show_titles=True,
        title_fmt=".4g",
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 8},
    )
    if truths is not None:
        fig.text(
            0.995,
            0.995,
            r"red lines: max stored $\ln P$ sample",
            ha="right",
            va="top",
            fontsize=10,
            color="crimson",
        )
    for ax in fig.axes:
        ax.tick_params(labelsize=6, length=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def numpyro_summary_text(samples: dict[str, np.ndarray]) -> str:
    from numpyro.diagnostics import summary as numpyro_summary

    stats = numpyro_summary(samples, prob=0.9, group_by_chain=True)
    # Drop parameters pinned to a constant (not actually sampled, e.g. a
    # control-file global with prior_unc=0): zero variance everywhere.
    stats = {name: values for name, values in stats.items()
             if values["std"] != 0.0}
    if not stats:
        return "(no sampled parameters)"
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


def collect_chain_batch(
    chain_dirs: list[Path],
    output_dir: Path,
    plot_params: list[str],
    output_prefix: str = "global",
) -> None:
    chains = []
    chain_labels = []
    for chain_dir in chain_dirs:
        fort7 = chain_dir / "fort.7"
        if not fort7.exists():
            raise FileNotFoundError(f"Missing Reid chain file: {fort7}")
        chain = load_chain(fort7)
        chains.append(chain)
        chain_labels.append(chain_dir.name)

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
    output_prefix = output_prefix.strip() or "global"
    summary_text = numpyro_summary_text(samples)
    summary_path = output_dir / f"{output_prefix}_summary.txt"
    summary_path.write_text(summary_text + "\n")

    combined = np.concatenate([chain[-min_draws:] for chain in chains])
    lnp_by_chain = {
        label: lnp_summary(chain[-min_draws:])
        for label, chain in zip(chain_labels, chains)
    }
    lnp_csv_path = output_dir / f"{output_prefix}_lnP.csv"
    with lnp_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chain", "iter", "walker", "lnP"])
        for label, chain in zip(chain_labels, chains):
            for row in chain[-min_draws:]:
                writer.writerow(
                    [
                        label,
                        row["iter"].item(),
                        row["walker"].item(),
                        row["lnP"].item(),
                    ]
                )
    lnp_summary_path = output_dir / f"{output_prefix}_lnP_summary.json"
    lnp_summary_path.write_text(
        json.dumps(
            {
                "common_stored_draws_per_chain": int(min_draws),
                "combined": lnp_summary(combined),
                "chains": lnp_by_chain,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    corner_path = output_dir / f"{output_prefix}_corner.png"
    plot_corner(combined, plot_params, corner_path)

    print("\nGlobal parameter summary (combined Reid chains):", flush=True)
    print(summary_text, flush=True)
    print(f"Global summary: {summary_path}", flush=True)
    print(f"Global lnP values: {lnp_csv_path}", flush=True)
    print(f"Global lnP summary: {lnp_summary_path}", flush=True)
    print(f"Global corner plot: {corner_path}", flush=True)


def parse_bool_quad(value: str) -> tuple[bool, bool, bool, bool]:
    chars = value.replace(",", " ").split()
    if len(chars) == 1 and len(chars[0]) == 4:
        chars = list(chars[0])
    if len(chars) != 4:
        raise argparse.ArgumentTypeError(
            "expected four booleans, e.g. T T T T or TTTF")
    return tuple(c.lower() in {"t", "true", "1", "yes", "y"}
                 for c in chars)  # type: ignore[return-value]


def parse_param_names(value: str) -> set[str]:
    names: set[str] = set()
    for raw in re.split(r"[\s,]+", value.strip()):
        if not raw:
            continue
        if raw not in GLOBAL_NAMES:
            valid = ", ".join(GLOBAL_NAMES)
            raise argparse.ArgumentTypeError(
                f"unknown global parameter '{raw}'. Valid names: {valid}"
            )
        names.add(raw)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and plot Mark Reid fit_disk MCMC without editing the Reid source.")  # noqa: E501
    parser.add_argument("--galaxy", default="NGC4258")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--init",
        default=DEFAULT_REID_INIT.name,
        help=(
            "Init TOML filename in scripts/megamaser/check_reid. "
            "The file may be Reid-style [globals] (flat or "
            "[globals.<GALAXY>.<variant>] merged) or a "
            "[model.galaxies.<NAME>.<variant>] config fragment."
        ),
    )
    parser.add_argument(
        "--variant",
        default="init",
        choices=["init", "init_qw"],
        help="Init variant to select from --init when it is a merged "
             "multi-galaxy TOML or a CANDEL config fragment.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--burnin",
        type=int,
        default=1000000,
        help="Primary Reid burn-in trials. <=0 skips it via generated burnin_values.dat.",  # noqa: E501
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100000000,
        help="Final MCMC trials. Reid v24d requires >=500000 because n_skip=itermax/500000.",  # noqa: E501
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
        default=10000000,
        help="Progress print interval for secondary burn-in and final MCMC. "
             "Use 0 to choose max(100000, min(10000000, trials/100)).",
    )
    parser.add_argument(
        "--fit-data",
        type=parse_bool_quad,
        default=(
            True,
            True,
            True,
            True))
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--fflags", default="-O2 -std=legacy -fno-automatic")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument(
        "--plot-params",
        default=",".join(DEFAULT_CONTOUR_PARAMS))
    parser.add_argument(
        "--fix-params",
        default="",
        help=(
            "Comma/space-separated Reid global parameter names to freeze by "
            "setting their control-file prior and proposal widths to zero."
        ),
    )
    parser.add_argument(
        "--free-params",
        default="",
        help=(
            "Comma/space-separated Reid global parameter names to unfreeze "
            "(template-fixed) by giving them a flat prior and a proposal "
            "width. Used to sample d2i_dr2_deg_mas2 for the quadratic warp."
        ),
    )
    parser.add_argument(
        "--fix-circular",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze eccentricity, periapsis, and periapsis-gradient globals "
             "(default: on; pass --no-fix-circular to model eccentricity).",
    )
    parser.add_argument(
        "--fix-all-globals",
        action="store_true",
        help="Freeze all 20 Reid global parameters; only per-spot latents "
             "(r, phi) are MAP-optimised. Profiles latents at a fixed point.",
    )
    parser.add_argument(
        "--instrument",
        action="store_true",
        help="Compile a per-run source copy with extra progress prints. "
             "Default is the unmodified Reid source.",
    )
    parser.add_argument(
        "--no-instrument",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--linear-warp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the quadratic inclination and PA warp globals "
             "(default: on; pass --no-linear-warp to model quadratic warp).",
    )
    parser.add_argument(
        "--collect-chain-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Collect completed chain directories and write a combined summary/corner plot.",  # noqa: E501
    )
    parser.add_argument(
        "--collect-output-prefix",
        default="global",
        help="Prefix for collected summary/corner filenames.",
    )
    args = parser.parse_args(argv)

    params = [x.strip() for x in args.plot_params.split(",") if x.strip()]
    try:
        fixed_params = parse_param_names(args.fix_params)
        free_params = parse_param_names(args.free_params)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if fixed_params & free_params:
        parser.error(
            "parameters cannot be both fixed and freed: "
            + ", ".join(sorted(fixed_params & free_params)))
    if args.fix_circular:
        fixed_params.update(
            {"ecc", "peri_az_deg", "dperi_dr_deg_mas"} - free_params)
    if args.linear_warp:
        fixed_params.update(
            {"d2i_dr2_deg_mas2", "d2PA_dr2_deg_mas2"} - free_params)
    if args.fix_all_globals:
        fixed_params.update(GLOBAL_NAMES)

    if args.collect_chain_dirs is not None:
        if args.output_dir is None:
            parser.error("--output-dir is required with --collect-chain-dirs")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        collect_chain_batch(
            args.collect_chain_dirs,
            args.output_dir,
            params,
            args.collect_output_prefix,
        )
        return 0

    if args.trials < 500000:
        raise ValueError(
            "Reid v24d computes n_skip = itermax / 500000 with integer division; "  # noqa: E501
            "use --trials >= 500000 unless the Fortran source itself is changed.")  # noqa: E501
    status_interval = (
        default_status_interval(args.trials)
        if args.status_interval == 0 else int(args.status_interval)
    )
    if status_interval < 1:
        raise ValueError("--status-interval must be >=1, or 0 for automatic")

    try:
        init_toml = resolve_init_toml(args.init)
        reid_init = load_toml_init(
            init_toml, args.galaxy, args.vcor, variant=args.variant)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    header, data_rows = parse_data_rows(args.data)
    run_init = shift_warp_pivots(
        reid_init.values, compute_reid_r_ref(
            data_rows, header, reid_init.values))

    h0_low = args.h0_low
    h0_high = args.h0_high
    if h0_low is None or h0_high is None:
        h0 = run_init["H0"]
        h0_low = h0 - 15.0 if h0_low is None else h0_low
        h0_high = h0 + 15.0 if h0_high is None else h0_high

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        variant_tag = f"_{args.variant}" if args.variant != "init" else ""
        run_dir = (
            DEFAULT_RESULTS /
            f"{args.galaxy}_{init_toml.stem}{variant_tag}_{stamp}")
    else:
        run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "fit_disk_data.inp").write_text(prepared_data_text(args.data,
                                                                  "300 700 0.02 0.03 0.01 0.01 0.3 Radio"))  # noqa: E501
    if args.galaxy != "NGC4258":
        print(f"[WARNING] write_control uses REID_CONTROL_TEMPLATE "
              f"({REID_CONTROL_TEMPLATE}, tuned for NGC4258) for every "
              f"non-fixed/non-free global's MCMC proposal step size "
              f"(post_unc column); only the value column is set from "
              f"{args.galaxy}'s init. These step sizes are not tuned for "
              f"{args.galaxy} and may mix poorly.")
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
        fixed_params=fixed_params,
        free_params=free_params,
    )
    if args.burnin <= 0:
        write_burnin_values(
            run_dir /
            "burnin_values.dat",
            run_init,
            initial_r_phi(
                data_rows,
                header,
                run_init),
            fixed_params=fixed_params)

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
        "source_instrumented": bool(args.instrument
                                    and not args.no_instrument),
        "h0_low": h0_low,
        "h0_high": h0_high,
        "fixed_params": sorted(fixed_params),
        "free_params": sorted(free_params),
        "initial_globals": {name: run_init[name] for name in GLOBAL_NAMES},
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata,
                                                          indent=2, sort_keys=True) + "\n")  # noqa: E501

    print(f"Prepared Reid run directory: {run_dir}", flush=True)
    if args.prepare_only:
        return 0

    exe = run_dir / "fit_disk_reid_v24d"
    if not args.no_compile:
        exe = compile_reid(
            run_dir,
            args.compiler,
            args.fflags,
            status_interval,
            instrument=bool(args.instrument and not args.no_instrument))
        print(f"Compiled Reid executable: {exe}", flush=True)
    elif not exe.exists():
        raise FileNotFoundError(
            f"--no-compile requested but executable is missing: {exe}")

    if not args.no_run:
        stdout = run_reid(exe, run_dir)
        print(f"Reid stdout: {stdout}", flush=True)
    else:
        stdout = run_dir / "fit_disk.stdout"

    fort7 = run_dir / "fort.7"
    if not fort7.exists():
        print(
            f"No chain file found at {fort7}; skipping post-processing.",
            file=sys.stderr)
        return 0

    chain = load_chain(fort7)
    write_chain_csv(chain, run_dir / "global_chain.csv")
    write_lnp_csv(chain, run_dir / "lnP.csv")
    summary = summary_from_chain(chain, stdout)
    (run_dir / "likelihood_summary.json").write_text(json.dumps(summary,
                                                                indent=2, sort_keys=True) + "\n")  # noqa: E501
    plot_corner(chain, params, run_dir / "global_corner.png")
    print(f"Chain CSV: {run_dir / 'global_chain.csv'}", flush=True)
    print(f"lnP CSV: {run_dir / 'lnP.csv'}", flush=True)
    print(
        f"Likelihood summary: {run_dir / 'likelihood_summary.json'}",
        flush=True)
    print(f"Global contour plot: {run_dir / 'global_corner.png'}", flush=True)
    print(f"Best stored lnP: {summary['lnP']['max']:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
