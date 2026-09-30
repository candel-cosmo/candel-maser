#!/usr/bin/env python3
"""Spot-check that CANDEL and Reid's fit_disk agree on the per-spot data fit.

For each galaxy we load the fiducial chain's *global* posterior samples (the
per-spot latents are not stored), draw a few random global samples, place every
spot at its conditional-MAP radius and phi seed given those globals, and then
evaluate the per-spot data-fit term ``-0.5*chi^2`` with BOTH codes at the SAME
(globals, r, phi, D_A): CANDEL's JAX disc model and Mark Reid's ``fit_disk``
Fortran likelihood (f2py-wrapped as ``reidlik``).  chi^2 is dimensionless, so
the two codes lie on a 1:1 line with no fitted offset.  A handful of random
spots is printed per sample; the codes should agree to <~0.1 nats.

Usage:
    python check_reid_agreement.py [GAL ...] [--n-samples 3] [--n-spots 4]
"""
import argparse
import os
import tempfile

# float64 so the comparison isolates model identity (Reid's fit_disk is f64);
# production CANDEL chains run in f32, where rounding at large chi^2 is larger.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tomli_w  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
from candel_maser.paths import RESULTS_ROOT  # noqa: E402

import candel_maser.run_maser as rm  # noqa: E402

from candel_maser.maser_blackjax import _initial_phi  # noqa: E402
from candel_maser.maser_config import (apply_dataset, check_chain_dataset)  # noqa: E402

DATASET = "original_published"
RESULTS = os.path.join(RESULTS_ROOT, "results", "Megamaser", DATASET)
GALAXIES = ["CGCG074-064", "NGC5765b", "UGC3789", "NGC6264", "NGC6323"]

apply_dataset(rm.master_cfg, DATASET)


def build_model_target(galaxy):
    gcfg = rm.master_cfg["model"]["galaxies"][galaxy]
    data = rm.load_megamaser_spots(
        rm.maser_data_root(DATASET), galaxy, v_sys_obs=gcfg["v_sys_obs"])
    if "D_lo" in gcfg and "D_hi" in gcfg:
        data["D_lo"] = float(gcfg["D_lo"])
        data["D_hi"] = float(gcfg["D_hi"])
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(rm.master_cfg, tmp)
    tmp.close()
    try:
        model = rm.MaserDiskModel(tmp.name, data)
    finally:
        os.unlink(tmp.name)
    return model, gcfg


def load_global_samples(galaxy):
    path = os.path.join(
        RESULTS, galaxy, f"{galaxy}_blackjax_mcmc_rphi_initconfig.hdf5")
    with h5py.File(path, "r") as f:
        check_chain_dataset(f.attrs, DATASET, path)
        return {k: np.asarray(v, dtype=float).ravel()
                for k, v in f["samples"].items()}


def seed_latents(model, target, point):
    """Conditional-MAP (r_ang, phi) seed for all spots given the globals."""
    theta = target.complete_params(
        {name: jnp.asarray(point[name]) for name in target.names})
    phys_args, phys_kw = model.phys_from_params_jax(theta, target.h)
    r_ang = np.asarray(model.conditional_r_ang_map(phys_args, phys_kw))
    phi = np.asarray(_initial_phi(model, jnp.asarray(phys_args[2]).dtype))
    return r_ang, phi


def check_galaxy(galaxy, n_samples, n_spots, rng):
    model, _ = build_model_target(galaxy)
    samples = load_global_samples(galaxy)
    n_draws = len(next(iter(samples.values())))

    init = rm._complete_mass_point(
        model, {k: np.asarray(v[0]) for k, v in samples.items()})
    target = rm.MaserBlackJaxTarget(model, rm._h_ref(model), init)
    ctx = rm._reid_loglik_context(galaxy, model.n_spots, DATASET)
    if ctx is None:
        print(f"{galaxy}: reidlik unavailable, skipping")
        return None

    print(f"\n{'='*64}\n{galaxy}: {model.n_spots} spots, {n_draws} global "
          f"draws\n{'='*64}")
    worst_med = 0.0
    for s in rng.choice(n_draws, size=min(n_samples, n_draws), replace=False):
        point = rm._complete_mass_point(
            model, {k: np.asarray(v[s]) for k, v in samples.items()})
        r_ang, phi = seed_latents(model, target, point)
        D_A = rm._point_D_A(model, target, point)
        candel = rm._candel_neg_half_chi2(model, target, point, r_ang, phi)
        reid = rm._reid_neg_half_chi2(ctx, galaxy, point, r_ang, phi, D_A=D_A)
        diff = np.abs(np.asarray(candel) - np.asarray(reid))
        # median is the robust headline; off-posterior seeds put a few spots
        # on the steep chi^2 wall where any tiny model difference is amplified.
        med, mx = float(np.nanmedian(diff)), float(np.nanmax(diff))
        worst_med = max(worst_med, med)

        spots = rng.choice(model.n_spots, size=min(n_spots, model.n_spots),
                           replace=False)
        print(f"draw {s:6d}  D_A={D_A:7.2f} Mpc   "
              f"median|d|={med:.4f}  max|d|={mx:.4f}")
        print(f"  {'spot':>5} {'r[mas]':>8} {'phi[deg]':>9} "
              f"{'CANDEL':>12} {'Reid':>12} {'diff':>10}")
        for i in spots:
            d = float(candel[i]) - float(reid[i])
            print(f"  {i:5d} {r_ang[i]:8.4f} {np.degrees(phi[i]):9.2f} "
                  f"{float(candel[i]):12.4f} {float(reid[i]):12.4f} {d:10.5f}")
    return worst_med


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("galaxies", nargs="*", default=GALAXIES)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--n-spots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-match-reid", dest="match_reid", action="store_false",
                    help="do NOT swap in Reid's physics constants/gamma "
                         "(leaves CANDEL's native constants; a small "
                         "systematic offset then appears)")
    ap.add_argument("--tol", type=float, default=0.25,
                    help="fail if the per-draw median |CANDEL-Reid| exceeds "
                         "this for any galaxy (a convention/build bug craters "
                         "lnP by thousands, so this cleanly flags breakage)")
    args = ap.parse_args(argv)

    if args.match_reid:
        # Match Reid's physical constants and circular-speed SR gamma before
        # any model is built/traced, so CANDEL evaluates the SAME likelihood;
        # the Reid side is already pinned to the same D_A per draw.
        rm._apply_reid_physics_constants()

    rng = np.random.default_rng(args.seed)
    worst = 0.0
    for g in (args.galaxies or GALAXIES):
        w = check_galaxy(g, args.n_samples, args.n_spots, rng)
        if w is not None:
            worst = max(worst, w)
    print(f"\nworst per-draw median |CANDEL-Reid| over all checks: "
          f"{worst:.4f} nats")
    assert worst < args.tol, (
        f"codes disagree: median |CANDEL-Reid| = {worst:.3f} > {args.tol}")


if __name__ == "__main__":
    main()
