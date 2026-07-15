#!/usr/bin/env python3
"""Write a Reid ``fit_disk`` data file for any megamaser galaxy.

Drives off ``load_megamaser_spots`` so every galaxy uses CANDEL's frame and
unit conventions (positions micro-arcsec -> mas, optical-LSR velocities), and
keeps the loader spot order so per-spot quantities line up with CANDEL chains.

Reid's likelihood uses the control-file error floors (params 16-20), so the
floor values written in the data header here are inert placeholders; only the
spot data and the systemic Vmin/Vmax classification matter.
"""
import argparse
import os

import numpy as np
import tomli

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import data_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(SCRIPT_DIR, "..", "config_maser.toml")


def systemic_window(velocity, is_highvel, is_blue):
    """Vmin/Vmax matching the loader's systemic classification in Reid."""
    sys_v = velocity[~is_highvel]
    blue_v = velocity[is_highvel & is_blue]
    red_v = velocity[is_highvel & ~is_blue]
    lo = sys_v.min()
    hi = sys_v.max()
    vmin = 0.5 * (blue_v.max() + lo) if blue_v.size else lo - 1.0
    vmax = 0.5 * (hi + red_v.min()) if red_v.size else hi + 1.0
    return float(vmin), float(vmax)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("galaxy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    with open(CONFIG, "rb") as f:
        master = tomli.load(f)
    gcfg = master["model"]["galaxies"][args.galaxy]
    sigma_v = float(master["model"].get("sigma_v_default", 0.25))

    d = load_megamaser_spots(
        data_path("data", "Megamaser"), args.galaxy,
        v_sys_obs=gcfg["v_sys_obs"])

    v = np.asarray(d["velocity"])
    x = np.asarray(d["x"]) / 1000.0
    sx = np.asarray(d["sigma_x"]) / 1000.0
    y = np.asarray(d["y"]) / 1000.0
    sy = np.asarray(d["sigma_y"]) / 1000.0
    a = np.asarray(d["a"])
    sa = np.asarray(d["sigma_a"])
    measured = np.asarray(d["accel_measured"])
    vmin, vmax = systemic_window(
        v, np.asarray(d["is_highvel"]), np.asarray(d["is_blue"]))

    n_sys = int((~np.asarray(d["is_highvel"])).sum())
    header = (f"{vmin:.6g} {vmax:.6g} 0.001 0.001 0.25 0.25 0.3 Optical")
    lines = [
        header,
        f"! Reid fit_disk data for {args.galaxy} from load_megamaser_spots "
        f"({d['n_spots']} spots, {n_sys} systemic, "
        f"{int(measured.sum())} with acceleration).",
        f"! Velocities optical-LSR (frame={d['velocity_frame']}); "
        "positions in mas; raw velocity error set to sigma_v_default.",
        "! Unmeasured accelerations flagged with sigma_A = -2.",
        "! ID  Vlsr_opt  sigma_V   x  sigma_x   y  sigma_y   Acc  sigma_Acc",
    ]
    for i in range(d["n_spots"]):
        acc = float(a[i]) if measured[i] else 0.0
        sigma_acc = float(sa[i]) if measured[i] else -2.0
        lines.append(
            f"{i + 1:5d} {v[i]:12.5f} {sigma_v:10.5f}"
            f" {x[i]:12.6f} {sx[i]:10.6f} {y[i]:12.6f} {sy[i]:10.6f}"
            f" {acc:12.6f} {sigma_acc:10.6f}")

    out = args.out or data_path(
        "data", "Megamaser", f"{args.galaxy}_loader_reid.inp")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}: {d['n_spots']} spots, {n_sys} systemic, "
          f"Vmin/Vmax = {vmin:.2f}/{vmax:.2f}", flush=True)


if __name__ == "__main__":
    main()
