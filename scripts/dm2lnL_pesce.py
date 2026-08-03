"""Delta(-2 ln L) for tab:chi2_pesce: the same profile chi^2 comparison as
chi2_evidence_table.py, but retaining the Gaussian log-normalisation
sum ln(2 pi sigma^2_tot) that chi^2 drops.

For each galaxy the posterior-median (ours) and reported P20 globals are held
fixed and the per-spot (r, phi) are profiled by evaluate_at_globals -- the
exact code path behind the chi^2 column.  The full per-spot Gaussian
log-likelihood
(_make_perspot_eval, = lnorm + lnorm_a - 0.5 chi^2) is summed at those same
optimised latents, so -2 ln L = -2 sum(per) = chi^2 + sum ln(2 pi sigma^2_tot)
by construction.  Reports per solution chi^2, normalisation, and -2 ln L, and
the deltas (this work minus P20), and checks

    Delta(-2 ln L) - Delta chi^2 = Delta[ sum ln(2 pi sigma^2_tot) ].

    python scripts/megamaser/dm2lnL_pesce.py
"""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_HERE = os.path.dirname(__file__)
for p in (_HERE, os.path.join(_HERE, "check_reid"),
          os.path.join(_HERE, "check_reid", "reidlik_build")):
    sys.path.insert(0, p)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import run_map as rm  # noqa: E402

from candel.model.maser_map import _make_perspot_eval  # noqa: E402
from candel.model.maser_map import evaluate_at_globals  # noqa: E402

GALAXIES = ["CGCG074-064", "NGC5765b", "UGC3789", "NGC6264", "NGC6323"]


def _median_globals(gal, target):
    path = os.path.join(rm.results_path(rm._MASTER_CFG["io"]["root_output"]),
                        gal, f"{gal}_blackjax_mcmc_rphi_initconfig.hdf5")
    with h5py.File(path, "r") as f:
        s = f["samples"]
        return {n: float(np.median(np.asarray(s[n]).reshape(-1)))
                for n in target.names}


def _eval(target, perspot, gdict):
    """chi^2, normalisation, -2 ln L at fixed globals with profiled latents."""
    res = evaluate_at_globals(target, gdict, init_r_ang=None,
                              marginal=False, verbose=False)
    gp = {n: jnp.asarray(gdict[n], dtype=jnp.float64) for n in target.names}
    ll = perspot(jnp.asarray(res["r_ang"]), jnp.asarray(res["phi"]), gp)
    m2lnL = float(-2.0 * np.asarray(ll).sum())
    return dict(chi2=res["chi2"], norm=m2lnL - res["chi2"], m2lnL=m2lnL,
                dof=int(res["dof"]))


def main():
    rows = []
    for gal in GALAXIES:
        gcfg = rm._MASTER_CFG["model"]["galaxies"][gal]
        _, target, _ = rm._build_target(gal, gcfg, None)
        perspot = _make_perspot_eval(target)
        ours = _eval(target, perspot, _median_globals(gal, target))
        p20 = _eval(target, perspot, rm._pesce_globals(target, gal))
        rows.append((gal, ours, p20))
        print(f"  {gal}: done", flush=True)

    hdr = (f"\n{'galaxy':13s} {'dof':>4s} | "
           f"{'chi2_o':>9s} {'norm_o':>10s} {'-2lnL_o':>10s} | "
           f"{'chi2_p':>9s} {'norm_p':>10s} {'-2lnL_p':>10s} | "
           f"{'Dchi2':>7s} {'Dnorm':>8s} {'D(-2lnL)':>9s} {'check':>7s}")
    print(hdr)
    print("-" * (len(hdr) + 6))
    for gal, o, p in rows:
        dchi2 = o["chi2"] - p["chi2"]
        dnorm = o["norm"] - p["norm"]
        dm2 = o["m2lnL"] - p["m2lnL"]
        chk = dm2 - dchi2 - dnorm            # must be ~0
        print(f"{gal:13s} {o['dof']:4d} | "
              f"{o['chi2']:9.2f} {o['norm']:10.2f} {o['m2lnL']:10.2f} | "
              f"{p['chi2']:9.2f} {p['norm']:10.2f} {p['m2lnL']:10.2f} | "
              f"{dchi2:+7.1f} {dnorm:+8.1f} {dm2:+9.1f} {chk:7.0e}")

    print("\nrounded fill for tab:chi2_pesce (Delta(-2 ln L), this-P20):")
    for gal, o, p in rows:
        print(f"  {gal:13s} {round(o['m2lnL'] - p['m2lnL']):+d}")


if __name__ == "__main__":
    main()
