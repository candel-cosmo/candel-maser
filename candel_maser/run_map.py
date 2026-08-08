"""Fixed-globals latent-MAP / chi^2 diagnostic for one megamaser galaxy.

Fix the disk globals and optimise only the per-spot latents (r_ang, phi):

  * DE   -- globals from the config ``[init]`` block (the DE solution);
  * Reid -- globals from the Pesce/Reid published values (check_reid helper).

Both cases run the same per-spot 2D optimisation, so chi^2 at the DE globals is
directly comparable to chi^2 at Reid's globals.

    python scripts/megamaser/run_map.py NGC6264
"""
import argparse
import json
import os
import sys
import tempfile

import tomli
import tomli_w

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_maser.toml")
with open(_CONFIG_PATH, "rb") as f:
    _MASTER_CFG = tomli.load(f)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from candel.model.maser_blackjax import MaserBlackJaxTarget  # noqa: E402
from candel.model.maser_map import evaluate_at_globals  # noqa: E402
from candel.model.model_H0_maser import MaserDiskModel  # noqa: E402
from candel.pvdata.megamaser_data import (  # noqa: E402
    load_megamaser_spots, maser_data_root)
from candel.util import get_nested, results_path  # noqa: E402
from maser_config import (add_dataset_arg, apply_dataset,  # noqa: E402
                          check_init_block)


def _h_ref(model):
    return float(get_nested(model.config, "model/H0_ref", 73.0)) / 100.0


def _D_A_from_D_c(model, D_c):
    z = model.distance2redshift(jnp.atleast_1d(D_c), h=_h_ref(model)).squeeze()
    return D_c / (1.0 + z)


def _variant_suffix(model):
    parts = []
    if model.use_ecc:
        parts.append("ecc")
    if model.use_quadratic_warp:
        parts.append("qw")
    return "_" + "_".join(parts) if parts else ""


def _init_block(gal_cfg, model):
    suffix = _variant_suffix(model)
    if suffix and ("init" + suffix) in gal_cfg:
        return gal_cfg["init" + suffix]
    return gal_cfg.get("init", {})


def _clean_init(model, init_cfg):
    """Config [init] block -> constrained-globals dict (from run_de_map)."""
    check_init_block(init_cfg, model)
    p = {k: jnp.asarray(v) for k, v in init_cfg.items()}
    p.pop("M_BH", None)
    if model._D_A_uniform:
        if "D_A" not in p:
            p["D_A"] = _D_A_from_D_c(model, p["D_c"])
    elif "D_c" not in p:
        raise KeyError("init requires D_c.")
    D_A = p["D_A"] if model._D_A_uniform else _D_A_from_D_c(model, p["D_c"])
    if getattr(model, "mass_parameterization", "eta") == "eta":
        if "eta" not in p:
            p["eta"] = p["log_MBH"] - jnp.log10(D_A)
    else:
        if "log_MBH" not in p:
            p["log_MBH"] = p["eta"] + jnp.log10(D_A)
        p.pop("eta", None)
    if model._D_A_uniform:
        p.pop("D_c", None)
    if not model.use_ecc:
        for k in ("e_x", "e_y", "ecc", "periapsis", "periapsis_rad",
                  "dperiapsis_dr"):
            p.pop(k, None)
    elif model.ecc_cartesian:
        for k in ("ecc", "periapsis", "periapsis_rad"):
            p.pop(k, None)
        for k in ("e_x", "e_y", "dperiapsis_dr"):
            p.setdefault(k, jnp.asarray(0.0))
    if not model.use_quadratic_warp:
        for k in ("d2i_dr2", "d2Omega_dr2"):
            p.pop(k, None)
    else:
        for k in ("d2i_dr2", "d2Omega_dr2"):
            p.setdefault(k, jnp.asarray(0.0))
    return p


def _build_target(galaxy, gcfg, spot_batch, dataset):
    config = {
        "inference": _MASTER_CFG["inference"],
        "model": dict(_MASTER_CFG["model"]),
        "io": _MASTER_CFG["io"],
    }
    config["model"]["galaxies"] = {
        g: dict(blk) for g, blk in _MASTER_CFG["model"]["galaxies"].items()}
    data = load_megamaser_spots(
        maser_data_root(dataset), galaxy, v_sys_obs=gcfg["v_sys_obs"])
    if "D_de_lo" in gcfg and "D_de_hi" in gcfg:
        data["D_lo"] = float(gcfg["D_de_lo"])
        data["D_hi"] = float(gcfg["D_de_hi"])
    elif "D_lo" in gcfg and "D_hi" in gcfg:
        data["D_lo"], data["D_hi"] = float(gcfg["D_lo"]), float(gcfg["D_hi"])
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp)
    tmp.close()
    try:
        model = MaserDiskModel(tmp.name, data)
    finally:
        os.unlink(tmp.name)
    h = _h_ref(model)
    init = _clean_init(model, _init_block(config["model"]["galaxies"][galaxy],
                                          model))
    target = MaserBlackJaxTarget(model, h, init, spot_batch=spot_batch)
    return model, target, init


def _pesce_globals(target, galaxy):
    """Reid/Pesce published globals as a CANDEL theta dict (target.names)."""
    helper = os.path.join(os.path.dirname(__file__), "check_reid")
    if helper not in sys.path:
        sys.path.insert(0, helper)
    from pesce_globals import candel_theta_from_point, paper_point
    point, status = paper_point(galaxy, _MASTER_CFG)
    if point is None:
        raise KeyError(f"no Pesce/Reid globals: {status}")
    return candel_theta_from_point(point, galaxy, _MASTER_CFG, target)


# Reid+2013 (1207.7292) Table 3 "Basic Disk Model" posteriors for UGC 3789,
# at their reference radius r_ref = 0.60 mas (Reid inclination/PA convention).
_REID2013 = {
    "UGC3789": dict(
        D_A=49.6, M_BH_1e7=1.16, v_cmb=3320.0, x0_mas=-0.402, y0_mas=-0.460,
        i_ref_deg=90.6, di_dr=-7.6, PA_ref_deg=221.5, dPA_dr=-2.0, r_ref=0.60,
        sigma_x_mas=0.01, sigma_y_mas=0.01, sigma_v_sys=1.0, sigma_v_hv=0.3,
        sigma_a=0.57),
}


def _reid2013_globals(galaxy, target):
    """Reid+2013 published globals as a CANDEL theta dict (target.names)."""
    if galaxy not in _REID2013:
        raise KeyError(f"no Reid-2013 globals for {galaxy}")
    helper = os.path.join(os.path.dirname(__file__), "check_reid")
    if helper not in sys.path:
        sys.path.insert(0, helper)
    from pesce_globals import candel_theta_from_point, from_cmb

    from candel.pvdata.megamaser_data import megamaser_velocity_frame
    r = _REID2013[galaxy]
    r0 = r["r_ref"]
    gcfg = _MASTER_CFG["model"]["galaxies"][galaxy]
    frame = megamaser_velocity_frame(galaxy)
    # CANDEL convention: i -> 180 - i (di/dr flips sign); PA/Omega unchanged.
    # candel_theta_from_point expects zero-radius intercepts, so back the
    # r_ref=0.60 values out along the linear warp.
    i_ref = 180.0 - r["i_ref_deg"]
    di = -r["di_dr"]
    point = dict(
        D_A=r["D_A"], M_BH_1e7=r["M_BH_1e7"],
        v_native=from_cmb(r["v_cmb"], frame, gcfg["ra"], gcfg["dec"]),
        x0_mas=r["x0_mas"], y0_mas=r["y0_mas"],
        i0_r0_deg=i_ref - di * r0, di_dr_r0=di,
        Omega0_r0_deg=r["PA_ref_deg"] - r["dPA_dr"] * r0,
        dOmega_dr_r0=r["dPA_dr"],
        sigma_x_mas=r["sigma_x_mas"], sigma_y_mas=r["sigma_y_mas"],
        sigma_v_sys=r["sigma_v_sys"], sigma_v_hv=r["sigma_v_hv"],
        sigma_a=r["sigma_a"])
    return candel_theta_from_point(point, galaxy, _MASTER_CFG, target)


def _reid_fortran_chi2(galaxy, model, results):
    """Add each result's chi^2 from the ORIGINAL Reid Fortran (reidlik), at the
    same globals + CANDEL-optimised latents.  Returns True if available."""
    helper = os.path.join(os.path.dirname(__file__), "check_reid")
    if helper not in sys.path:
        sys.path.insert(0, helper)
    try:
        from reid_chi2 import loglik_context, neg_half_chi2
    except Exception as exc:                           # noqa: BLE001
        print(f"Reid-Fortran cross-check unavailable: {exc}", flush=True)
        return False
    ctx = loglik_context(galaxy, model.n_spots, model.dataset)
    if ctx is None:
        return False
    for s in results.values():
        nh = neg_half_chi2(ctx, galaxy, s["point"], s["r_ang"], s["phi"],
                           D_A=s.get("D_A"))
        s["chi2_reid_code"] = float(-2.0 * np.asarray(nh).sum())
    return True


def _serialise(s):
    return dict(point=s["point"], chi2=s["chi2"],
                chi2_per_dof=s["chi2_per_dof"], logP_marg=s["logP_marg"],
                chi2_reid_code=s.get("chi2_reid_code"),
                r_ang=np.asarray(s["r_ang"]).tolist(),
                phi=np.asarray(s["phi"]).tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("galaxy")
    add_dataset_arg(parser)
    parser.add_argument("--out", default=None)
    parser.add_argument("--nm-maxiter", type=int, default=400)
    parser.add_argument("--n-restarts", type=int, default=5,
                        help="random Nelder-Mead restarts per spot per pass")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the random restarts")
    parser.add_argument("--spot-batch", type=int, default=None)
    parser.add_argument("--no-reid", action="store_true",
                        help="skip the Reid/Pesce fixed-globals comparison")
    parser.add_argument("--no-marginal", action="store_true",
                        help="skip the latent-marginalised ln L (faster)")
    args = parser.parse_args(argv)

    dataset = apply_dataset(_MASTER_CFG, args.dataset)
    galaxies = _MASTER_CFG["model"]["galaxies"]
    if args.galaxy not in galaxies:
        raise SystemExit(
            f"Unknown galaxy {args.galaxy!r}. Available: {list(galaxies)}")

    jax.config.update("jax_enable_x64", True)
    print("float64 enabled; JAX backend:", jax.default_backend(), flush=True)

    model, target, init = _build_target(
        args.galaxy, galaxies[args.galaxy], args.spot_batch, dataset)
    print(f"{args.galaxy}: n_spots={model.n_spots}, "
          f"globals={list(target.names)}", flush=True)

    marginal = not args.no_marginal
    results = {}

    de_globals = {n: init[n] for n in target.names}
    de_r = init.get("r_ang")
    print("Evaluating at DE (config) globals ...", flush=True)
    results["DE"] = evaluate_at_globals(
        target, de_globals,
        init_r_ang=np.asarray(de_r) if de_r is not None else None,
        nm_maxiter=args.nm_maxiter, n_restarts=args.n_restarts,
        seed=args.seed, marginal=marginal)

    if not args.no_reid:
        try:
            reid_globals = _pesce_globals(target, args.galaxy)
            print("Evaluating at Reid/Pesce globals ...", flush=True)
            results["Reid"] = evaluate_at_globals(
                target, reid_globals, init_r_ang=None,
                nm_maxiter=args.nm_maxiter, n_restarts=args.n_restarts,
                seed=args.seed, marginal=marginal)
        except Exception as exc:                       # noqa: BLE001
            print(f"Reid comparison skipped: {exc}", flush=True)

    if args.galaxy in _REID2013 and not args.no_reid:
        try:
            print("Evaluating at Reid+2013 published globals ...", flush=True)
            results["Reid13"] = evaluate_at_globals(
                target, _reid2013_globals(args.galaxy, target),
                init_r_ang=None,
                nm_maxiter=args.nm_maxiter, n_restarts=args.n_restarts,
                seed=args.seed, marginal=marginal)
        except Exception as exc:                       # noqa: BLE001
            print(f"Reid-2013 comparison skipped: {exc}", flush=True)

    # chi^2 cross-check via the original Reid Fortran (reidlik), same point.
    has_reid_code = _reid_fortran_chi2(args.galaxy, model, results)

    dof = results["DE"]["dof"]
    print("\n=== fixed-globals latent-MAP / chi^2 ===", flush=True)
    print(f"dof = {dof}  (N_used={results['DE']['n_used']}, "
          f"N_params={results['DE']['n_params']}: {results['DE']['n_global']} "
          f"globals + 2 x {results['DE']['n_spots']} latents)", flush=True)
    print(f"{'globals':>6s} {'chi2':>10s} {'chi2/dof':>9s} "
          f"{'logP_marg':>12s} {'D_A':>8s} {'logMBH':>8s}", flush=True)
    for label, s in results.items():
        lp = float('nan') if s["logP_marg"] is None else s["logP_marg"]
        key = "D_A" if "D_A" in s["point"] else "D_c"
        print(f"{label:>6s} {s['chi2']:>10.3f} {s['chi2_per_dof']:>9.4f} "
              f"{lp:>12.3f} "
              f"{s['point'].get(key, float('nan')):>8.3f} "
              f"{s['point'].get('log_MBH', float('nan')):>8.4f}", flush=True)
    if "Reid" in results:
        d_chi2 = results['DE']['chi2'] - results['Reid']['chi2']
        print(f"\nchi2(DE) - chi2(Reid)   = {d_chi2:+.3f}  "
              f"(negative => DE fits better)", flush=True)
        if results['DE']['logP_marg'] and results['Reid']['logP_marg']:
            d_lp = results['DE']['logP_marg'] - results['Reid']['logP_marg']
            print(f"logP_marg(DE) - (Reid)  = {d_lp:+.3f}  "
                  f"(positive => DE preferred)", flush=True)

    if has_reid_code:
        print("\n=== chi^2 cross-check: CANDEL vs original Reid Fortran "
              "(same globals + latents) ===", flush=True)
        print(f"{'globals':>6s} {'chi2_CANDEL':>12s} {'chi2_Reid_code':>15s} "
              f"{'abs_diff':>9s} {'rel_%':>7s}", flush=True)
        for label, s in results.items():
            cc, rc = s["chi2"], s["chi2_reid_code"]
            rel = 100.0 * abs(cc - rc) / rc if rc else float("nan")
            flag = "" if rel < 1.0 else "  <-- CHECK"
            print(f"{label:>6s} {cc:>12.3f} {rc:>15.3f} "
                  f"{abs(cc - rc):>9.3f} {rel:>7.3f}{flag}", flush=True)

    outpath = args.out or os.path.join(
        results_path(
            _MASTER_CFG["io"].get("root_output", "results/Megamaser")),
        f"{args.galaxy}_map{_variant_suffix(model)}.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as fh:
        json.dump({"dof": dof,
                   **{k: results["DE"][k] for k in
                      ("n_used", "n_params", "n_spots", "n_accel",
                       "n_global")},
                   "evaluations": {k: _serialise(v)
                                   for k, v in results.items()}}, fh, indent=2)
    print(f"\nsaved {outpath}", flush=True)


if __name__ == "__main__":
    main()
