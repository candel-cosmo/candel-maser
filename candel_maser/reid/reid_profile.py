#!/usr/bin/env python3
"""Profile log-likelihood of Reid's fit_disk model at fixed global parameters.

With the 20 disk globals fixed, each maser spot's likelihood depends only on
its own (r, phi), so the latent fit is separable into independent 2-D problems.
We
optimise each spot's (r, phi) by a coarse grid + Nelder-Mead, evaluating the
likelihood with Mark Reid's *own* Fortran routines (``calc_warped_model`` and
``calc_ln_p_data``) wrapped via f2py -- no physics is reimplemented here.

The sum of per-spot maxima is the profiled data log-likelihood at that global
point (Reid's control file uses flat priors, so his reported best ln(Prob) is
this data term).  Use the same interface for all comparison points; the printed
Reid ``fit_disk.prt`` latents are rounded too aggressively for exact replay.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reidlik_build"))
import reidlik  # noqa: E402
import run_reid_mcmc as rr  # noqa: E402  (reuse data parsing + init loading)

NUM_GLOBAL = 20
MAXF = 401          # Reid max_masers
NDAT = 4 * MAXF     # 1604
FIT_DATA = np.array([True, True, True, True])
N_LATENT_RESTARTS = 10


def parse_name_value(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    name, raw = value.split("=", 1)
    name = name.strip()
    if name not in rr.GLOBAL_NAMES:
        valid = ", ".join(rr.GLOBAL_NAMES)
        raise argparse.ArgumentTypeError(
            f"unknown global parameter '{name}'. Valid names: {valid}")
    return name, float(raw)


def parse_global_names(value):
    try:
        return sorted(rr.parse_param_names(value))
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def default_global_steps():
    lines = rr.template_control_lines()
    steps = {}
    for i, name in enumerate(rr.GLOBAL_NAMES, start=5):
        parts = lines[i].split("!", 1)[0].split()
        post = abs(float(parts[2]))
        prior = abs(float(parts[1]))
        steps[name] = post if post > 0 else prior
    return steps


def with_derived(g):
    out = dict(g)
    out["_D_c"] = (out["Vsys_km_s"] + out["Vcor_km_s"]) / out["H0"]
    return out


def invalid_globals(g):
    if g["H0"] <= 0.0 or g["Mbh_1e7Msun"] <= 0.0:
        return True
    if g["ecc"] < 0.0 or g["ecc"] >= 1.0:
        return True
    for name in ("sigma_x_mas", "sigma_y_mas", "sigma_vsys_km_s",
                 "sigma_vhv_km_s", "sigma_acc_km_s_yr"):
        if g[name] < 0.0:
            return True
    return False


def setup_numbers():
    n = reidlik.numbers
    n.pi = 4.0 * math.atan(1.0)
    n.deg2rad = n.pi / 180.0
    n.rad2deg = 180.0 / n.pi
    n.au2km = 1.496e8
    n.sun_mass = 1.98892e33
    n.vearth = 29.785


def fill_ez(Ho, Vo, Vcor):
    """Populate Ez_int near the galaxy recession velocity via Reid dampc."""
    ez = reidlik.ez_integral
    ez.n_vels = 50000
    n_v = int(Vo + Vcor + 0.5)
    for k in range(max(1, n_v - 3), min(50000, n_v + 4)):
        eq, _ = reidlik.dampc(float(k), Ho)
        ez.ez_int[k - 1] = eq


def globals_to_params(g):
    """Map a Reid-convention global dict to params[0:20]."""
    p = np.zeros(822)
    for i, name in enumerate(rr.GLOBAL_NAMES):
        p[i] = g[name]
    return p


def build_data(data_path):
    header, rows = rr.parse_data_rows(Path(data_path))
    v = rows[:, 1].copy()
    if str(header["velocity_flag"]).lower().startswith("r"):
        v = rr.radio_to_optical(v)
    N = len(rows)
    vlsr = np.zeros(MAXF)
    x_err = np.zeros(MAXF)
    y_err = np.zeros(MAXF)
    vlsr_err = np.zeros(MAXF)
    acc_err = np.zeros(MAXF)
    data = np.zeros(NDAT)
    vlsr[:N] = v
    x_err[:N] = rows[:, 4]
    y_err[:N] = rows[:, 6]
    vlsr_err[:N] = rows[:, 2]
    acc_err[:N] = rows[:, 8]
    for i in range(N):
        data[4 * i + 0] = rows[i, 3]   # x
        data[4 * i + 1] = rows[i, 5]   # y
        data[4 * i + 2] = v[i]         # optical Vlsr
        data[4 * i + 3] = rows[i, 7]   # acc
    return {
        "N": N, "Vmin": float(header["Vmin"]), "Vmax": float(header["Vmax"]),
        "vlsr": vlsr, "x": rows[:, 3], "y": rows[:, 5],
        "x_err": x_err, "y_err": y_err, "vlsr_err": vlsr_err,
        "acc_err": acc_err, "data": data,
    }


def reid_r_ref(d, Xo, Yo):
    hv = (d["vlsr"][:d["N"]] < d["Vmin"]) | (d["vlsr"][:d["N"]] > d["Vmax"])
    r = np.hypot(d["x"][hv] - Xo, d["y"][hv] - Yo)
    return float(np.mean(r)) if len(r) else 0.0


def latent_restart_seeds(sky_i, r_ring):
    radii = [sky_i, r_ring, 0.75 * r_ring, 1.25 * r_ring]
    seeds = [
        (radii[0], 90.0), (radii[0], -90.0),
        (radii[1], 0.0), (radii[1], 180.0),
        (radii[0], 0.0), (radii[0], 180.0),
        (radii[1], 90.0), (radii[1], -90.0),
        (radii[2], 0.0), (radii[3], 180.0),
    ]
    return [(max(float(r), 1e-3), phi) for r, phi in seeds]


def profile_lnp(params, d, r_ref, grid_r=40, grid_phi=73):
    """Sum per-spot maxima of Reid's data log-likelihood over (r, phi)."""
    # Error floors depend only on globals -> compute once.
    res_err = reidlik.add_error_floors(
        params, d["N"], d["vlsr"], d["Vmin"], d["Vmax"],
        d["x_err"], d["y_err"], d["vlsr_err"], d["acc_err"])
    data = d["data"]
    # Geometric seeds for multi-start Nelder-Mead (no physics, no grid).
    # HV spots: sky radius ~ orbital radius, phi ~ +/-90.  Systemic spots sit
    # near the line of sight (small projected radius) but orbit at the
    # disk-ring radius, phi ~ 0/180; their position errors are tiny so the
    # optimum is sharp and a coarse grid misses it.
    hv = (d["vlsr"][:d["N"]] < d["Vmin"]) | (d["vlsr"][:d["N"]] > d["Vmax"])
    sky = np.hypot(d["x"][:d["N"]] - params[3], d["y"][:d["N"]] - params[4])
    r_ring = float(np.median(sky[hv]))
    p = params.copy()
    total = 0.0
    per_spot = np.zeros(d["N"])
    for i in range(d["N"]):
        re4 = np.zeros(NDAT)
        re4[:4] = res_err[4 * i:4 * i + 4]
        d4 = data[4 * i:4 * i + 4]
        base = NUM_GLOBAL + 2 * i

        def neg(rphi):
            p[base] = rphi[0]
            p[base + 1] = rphi[1]
            cx, cy, cv, ca = reidlik.calc_warped_model(p, NUM_GLOBAL, i + 1,
                                                       r_ref)
            resids = np.zeros(NDAT)
            resids[0] = d4[0] - cx
            resids[1] = d4[1] - cy
            resids[2] = d4[2] - cv
            resids[3] = d4[3] - ca
            return -reidlik.calc_ln_p_data(False, 4, resids, re4, FIT_DATA)

        sky_i = max(float(sky[i]), 1e-3)
        seeds = latent_restart_seeds(sky_i, r_ring)[:N_LATENT_RESTARTS]
        best_v = np.inf
        best_x = None
        for s in seeds:
            s = np.asarray(s, float)
            # explicit simplex with real (dr, dphi) steps; a default simplex
            # degenerates when a seed coordinate (e.g. phi=0) is ~0.
            simplex = np.array([s, s + [0.5, 0.0], s + [0.0, 15.0]])
            sol = minimize(neg, s, method="Nelder-Mead",
                           options={"initial_simplex": simplex,
                                    "xatol": 1e-5, "fatol": 1e-5,
                                    "maxiter": 800})
            if sol.fun < best_v:
                best_v = sol.fun
                best_x = sol.x
        per_spot[i] = -best_v
        total += -best_v
        p[base], p[base + 1] = best_x
    return total, per_spot, p


def profile_globals(g, d):
    g = with_derived(g)
    if invalid_globals(g):
        return -np.inf, None, None, np.nan
    params = globals_to_params(g)
    fill_ez(g["H0"], g["Vsys_km_s"], g["Vcor_km_s"])
    r_ref = reid_r_ref(d, g["x0_mas"], g["y0_mas"])
    total, per_spot, params = profile_lnp(params, d, r_ref)
    return total, per_spot, params, r_ref


def optimise_globals(g, d, names, steps, maxiter):
    names = list(names)
    if not names:
        return g, None

    defaults = default_global_steps()
    x0 = np.array([g[name] for name in names], dtype=float)
    scale = np.array([
        steps.get(name, defaults.get(name, 0.0)) for name in names],
        dtype=float)
    for i, name in enumerate(names):
        if scale[i] <= 0.0:
            scale[i] = max(abs(g[name]) * 0.01, 1e-3)

    best = {"lnP": -np.inf, "globals": dict(g)}

    def objective(x):
        trial = dict(g)
        trial.update({name: float(xi) for name, xi in zip(names, x)})
        total, _, _, _ = profile_globals(trial, d)
        if np.isfinite(total) and total > best["lnP"]:
            best["lnP"] = float(total)
            best["globals"] = trial
        return -total if np.isfinite(total) else np.inf

    simplex = np.vstack(
        [x0] + [x0 + np.eye(len(names))[i] * scale[i]
                for i in range(len(names))]
    )
    sol = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": simplex,
            "maxiter": maxiter,
            "xatol": 1e-4,
            "fatol": 1e-3,
        },
    )
    info = {
        "free_globals": names,
        "success": bool(sol.success),
        "message": str(sol.message),
        "nfev": int(sol.nfev),
        "lnP": float(best["lnP"]),
        "values": {name: float(best["globals"][name]) for name in names},
    }
    return best["globals"], info


def result_dict(g, total, per_spot, params, r_ref):
    out = {
        "lnP": float(total),
        "r_ref_mas": float(r_ref),
        "D_Mpc": float((g["Vsys_km_s"] + g["Vcor_km_s"]) / g["H0"]),
        "globals": {name: float(g[name]) for name in rr.GLOBAL_NAMES},
        "latents": [],
    }
    if per_spot is not None and params is not None:
        for i, lnP_i in enumerate(per_spot):
            base = NUM_GLOBAL + 2 * i
            out["latents"].append({
                "spot": i + 1,
                "r_mas": float(params[base]),
                "phi_deg": float(params[base + 1]),
                "lnP": float(lnP_i),
            })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True,
                    help="Reid [globals] or config-init TOML in check_reid/")
    ap.add_argument("--variant", default="init", choices=["init", "init_qw"],
                    help="Init variant to select from --init when it is a "
                         "merged multi-galaxy TOML or a config fragment.")
    ap.add_argument("--galaxy", default="NGC4258")
    ap.add_argument("--data", required=True)
    ap.add_argument("--vcor", type=float, default=0.0)
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        type=parse_name_value,
        metavar="NAME=VALUE",
        help="Override a Reid-convention global parameter after loading init.",
    )
    ap.add_argument(
        "--map-globals",
        default="",
        help="Comma/space-separated Reid globals to optimise before profiling "
             "the per-spot latent r,phi values.",
    )
    ap.add_argument(
        "--map-step",
        action="append",
        default=[],
        type=parse_name_value,
        metavar="NAME=STEP",
        help="Initial simplex step for a --map-globals parameter.",
    )
    ap.add_argument("--map-maxiter", type=int, default=80)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    init_toml = rr.resolve_init_toml(args.init)
    reid_init = rr.load_toml_init(
        init_toml, args.galaxy, args.vcor, variant=args.variant)
    d = build_data(args.data)
    g = rr.shift_warp_pivots(
        reid_init.values, reid_r_ref(d, reid_init.values["x0_mas"],
                                     reid_init.values["y0_mas"]))
    for name, value in args.set:
        g[name] = value
    g = with_derived(g)

    setup_numbers()
    if args.map_globals:
        try:
            names = parse_global_names(args.map_globals)
        except argparse.ArgumentTypeError as exc:
            ap.error(str(exc))
        g, map_info = optimise_globals(
            g, d, names, dict(args.map_step), args.map_maxiter)
    else:
        map_info = None

    g = with_derived(g)
    total, per_spot, params, r_ref = profile_globals(g, d)
    if not np.isfinite(total):
        raise SystemExit("profiled likelihood is not finite")
    result = result_dict(g, total, per_spot, params, r_ref)
    if map_info is not None:
        result["global_map"] = map_info
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2,
                                            sort_keys=True) + "\n")

    print(f"galaxy={args.galaxy} init={init_toml.name}")
    print(f"  globals: H0={g['H0']:.3f} Mbh={g['Mbh_1e7Msun']:.3f} "
          f"i0={g['i0_deg']:.3f} D={g['_D_c']:.4f} r_ref={r_ref:.4f}")
    if map_info is not None:
        values = ", ".join(f"{k}={v:.6g}"
                           for k, v in map_info["values"].items())
        print(f"  global MAP: {values} "
              f"(lnP={map_info['lnP']:.3f}, nfev={map_info['nfev']})")
    print(f"  profiled data ln(Prob) = {total:.3f} over {d['N']} spots")
    if args.json_out is not None:
        print(f"  wrote {args.json_out}")
    return total


if __name__ == "__main__":
    main()
