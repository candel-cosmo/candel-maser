"""Quadratic-vs-linear warp comparison from the existing chains -- no MCMC.

Two things, per galaxy, printed as one table:

  * Goodness of fit: the reduced chi^2 of the disc data-fit term at the
    posterior-median globals with the per-spot (r_ang, phi) optimised, for the
    linear and the quadratic warp (same computation as ``chi2_evidence_table``;
    the linear column reproduces ``tab:chi2_pesce``).
  * Savage--Dickey Bayes factor B01 for the two quadratic warp coefficients,
    the nested test (d2i_dr2, d2Omega_dr2) = (0, 0):

        B01 = p((d2i, d2Omega) = 0 | D, M_quad) / pi((d2i, d2Omega) = 0)

    B01 > 1 favours the LINEAR warp.  The joint 2D posterior density at the
    origin is estimated two ways -- a fitted bivariate Gaussian (robust in the
    tail) and a Gaussian KDE (cross-check) -- both divided by the independent
    Normal(0, 90 deg/mas^2) prior density.  The priors are independent across
    parameters, so the Savage--Dickey separability condition holds.

    python scripts/megamaser/warp_model_comparison.py
"""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import run_map as rm  # noqa: E402
from maser_config import apply_dataset  # noqa: E402
from scipy.stats import gaussian_kde, multivariate_normal, norm  # noqa: E402

from candel.model.maser_map import evaluate_at_globals  # noqa: E402

GALAXIES = ["CGCG074-064", "NGC5765b", "UGC3789", "NGC6264", "NGC6323"]
DATASET = "original_published"
QW_COEFS = ("d2i_dr2", "d2Omega_dr2")


def _chain(gal, tag):
    return os.path.join(rm.results_path(rm._MASTER_CFG["io"]["root_output"]),
                        gal, f"{gal}_blackjax_mcmc_rphi_{tag}.hdf5")


def _median_globals(gal, target, tag):
    with h5py.File(_chain(gal, tag), "r") as f:
        s = f["samples"]
        return {n: float(np.median(np.asarray(s[n]).reshape(-1)))
                for n in target.names}


def _gof(gal, use_qw, tag):
    """Reduced chi^2 at the posterior-median globals (latents profiled)."""
    gcfg = rm._MASTER_CFG["model"]["galaxies"][gal]
    gcfg["use_quadratic_warp"] = use_qw
    _, target, _ = rm._build_target(gal, gcfg, None, DATASET)
    med = _median_globals(gal, target, tag)
    res = evaluate_at_globals(target, med, init_r_ang=None,
                              marginal=False, verbose=False)
    return float(res["chi2"]), int(res["dof"]), float(res["chi2_per_dof"])


def _prior_density_at_zero():
    pr = rm._MASTER_CFG["model"]["priors"]
    d = 1.0
    for c in QW_COEFS:
        p = pr[c]
        assert p["dist"] == "normal", f"{c} prior not normal: {p}"
        d *= float(norm.pdf(0.0, loc=p["loc"], scale=p["scale"]))
    return d


def _savage_dickey(gal, tag):
    with h5py.File(_chain(gal, tag), "r") as f:
        s = f["samples"]
        x = np.column_stack([np.asarray(s[c]).reshape(-1) for c in QW_COEFS])
    prior0 = _prior_density_at_zero()
    mu, cov = x.mean(0), np.cov(x, rowvar=False)
    post_g = float(multivariate_normal(mu, cov, allow_singular=True).pdf(
        [0.0, 0.0]))
    post_k = float(gaussian_kde(x.T)(np.zeros((2, 1)))[0])
    # Separation of zero from the mean in standard deviations.
    maha = float(np.sqrt(mu @ np.linalg.solve(cov, mu)))
    return dict(B01_g=post_g / prior0, B01_k=post_k / prior0, maha=maha)


def main():
    apply_dataset(rm._MASTER_CFG, DATASET)
    prior0 = _prior_density_at_zero()
    rows = []
    for gal in GALAXIES:
        chi2_L, nu_L, cnu_L = _gof(gal, False, "initconfig")
        chi2_Q, nu_Q, cnu_Q = _gof(gal, True, "qw_initconfig")
        sd = _savage_dickey(gal, "qw_initconfig")
        rows.append(dict(gal=gal, nu_L=nu_L, cnu_L=cnu_L, nu_Q=nu_Q,
                         cnu_Q=cnu_Q, dchi2=chi2_Q - chi2_L, **sd))
        print(f"  {gal}: done", flush=True)

    print("\nWarp model comparison (existing chains, no MCMC)")
    print("chi^2_nu: disc data-fit at posterior-median globals, latents "
          "profiled (L=linear, Q=quadratic warp)")
    print("dchi2 = chi2_Q - chi2_L (< 0: quadratic fits better)")
    print(f"B01: Savage-Dickey for (d2i_dr2, d2Omega_dr2)=0; prior density at "
          f"0 = {prior0:.3e} (deg/mas^2)^-2; B01 > 1 favours LINEAR")
    print("B01_G bivariate-Gaussian fit; B01_kde KDE cross-check; Mahal = "
          "sigma of origin from posterior mean\n")
    hdr = (f"{'galaxy':13s} {'nu_L':>4s} {'chi2nu_L':>8s} {'nu_Q':>4s} "
           f"{'chi2nu_Q':>8s} {'dchi2':>7s} | {'lnB01_G':>7s} {'B01_G':>9s} "
           f"{'lnB01_k':>7s} {'Mahal':>5s}  favours")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lng = np.log(r["B01_g"])
        lnk = np.log(r["B01_k"]) if r["B01_k"] > 0 else float("-inf")
        fav = "linear" if lng > 0 else "quadratic"
        note = "  (Mahal>3: trust B01_G)" if r["maha"] > 3 else ""
        print(f"{r['gal']:13s} {r['nu_L']:4d} {r['cnu_L']:8.2f} "
              f"{r['nu_Q']:4d} "
              f"{r['cnu_Q']:8.2f} {r['dchi2']:7.1f} | {lng:7.2f} "
              f"{r['B01_g']:9.2e} {lnk:7.2f} {r['maha']:5.1f}  {fav}{note}")


if __name__ == "__main__":
    main()
