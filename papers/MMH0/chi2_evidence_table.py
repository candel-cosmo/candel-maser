"""chi^2 (this work vs P20) at the posterior-median disc, paired with the
latent-marginalised logP_2d already stored in the --compare-reid run logs.

No marginal recompute: Delta lnP_2D is read from the logs (analyse_sweep), the
same values behind evidence_vs_distance.pdf.  Only the total chi^2 is computed
here (CANDEL and the original Reid Fortran, with the per-spot latents optimised
at the fixed median / P20 globals).  Prints a table and writes
chi2_evidence_table.json.

    python scripts/megamaser/chi2_evidence_table.py
"""
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_HERE = os.path.dirname(__file__)
_PAPER = os.path.abspath(os.path.join(_HERE, "..", "..", "notebooks",
                                      "paper_MMH0"))
for p in (_HERE, os.path.join(_HERE, "check_reid"),
          os.path.join(_HERE, "check_reid", "reidlik_build"), _PAPER):
    sys.path.insert(0, p)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import analyse_sweep as asw  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import reid_chi2  # noqa: E402
import run_map as rm  # noqa: E402

from candel.model.maser_map import evaluate_at_globals  # noqa: E402

GALAXIES = ["CGCG074-064", "NGC5765b", "UGC3789", "NGC6264", "NGC6323"]
# Published this-work-minus-P20 Delta lnP_2D (main.tex, sec:results_disc),
# for a consistency check against the values read back from the logs.
PAPER_DLNP = {"NGC5765b": 48.9, "CGCG074-064": 18.7, "UGC3789": 14.5,
              "NGC6323": 6.2, "NGC6264": 14.3}


def _median_globals(gal, target):
    path = os.path.join(rm.results_path(rm._MASTER_CFG["io"]["root_output"]),
                        gal, f"{gal}_blackjax_mcmc_rphi_initconfig.hdf5")
    with h5py.File(path, "r") as f:
        s = f["samples"]
        return {n: float(np.median(np.asarray(s[n]).reshape(-1)))
                for n in target.names}


def _logp2d_from_logs(gal):
    """(logP_2d MCMC median, logP_2d Pesce/Reid, D_A median) from run logs."""
    gdir = asw.root_output() / gal
    for init in ("config", "reid"):
        fp = gdir / f"{gal}_blackjax_mcmc_rphi_init{init}.hdf5"
        rows, _ = asw.parse_log(asw.find_log(gdir, str(fp)))
        m = rows.get("MCMC median", {})
        p = rows.get("Pesce/Reid", {})
        if "logP_2d" in m and "logP_2d" in p:
            return m["logP_2d"], p["logP_2d"], m.get("D_A"), init
    return None, None, None, None


def _chi2(gal, target, ctx, gdict):
    res = evaluate_at_globals(target, gdict, init_r_ang=None,
                              marginal=False, verbose=False)
    reid = None
    if ctx is not None:
        nh = reid_chi2.neg_half_chi2(ctx, gal, res["point"], res["r_ang"],
                                     res["phi"], D_A=res.get("D_A"))
        reid = float(-2.0 * np.asarray(nh).sum())
    return dict(chi2_candel=res["chi2"], chi2_reid=reid,
                chi2_per_dof=res["chi2_per_dof"], dof=int(res["dof"]),
                D_A=res.get("D_A"))


def main():
    rows = []
    for gal in GALAXIES:
        try:
            gcfg = rm._MASTER_CFG["model"]["galaxies"][gal]
            model, target, _ = rm._build_target(gal, gcfg, None)
            ctx = reid_chi2.loglik_context(gal, model.n_spots)
            med_glob = _median_globals(gal, target)
            ours = _chi2(gal, target, ctx, med_glob)
            p20 = _chi2(gal, target, ctx, rm._pesce_globals(target, gal))
            lp_m, lp_p, da_log, init = _logp2d_from_logs(gal)
            dlnP = (lp_m - lp_p) if (lp_m is not None and lp_p is not None) \
                else None
            rows.append(dict(
                galaxy=gal, dof=ours["dof"], ours=ours, p20=p20,
                dchi2=ours["chi2_candel"] - p20["chi2_candel"], dlnP_2D=dlnP,
                D_A_med=med_glob["D_A"], D_A_log=da_log, init=init))
            print(f"  {gal}: done ({init}-init log)", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {gal}: FAILED ({exc})", flush=True)

    print(f"\n{'galaxy':13s} {'dof':>4s} | {'chi2_ours':>9s} {'/dof':>5s} "
          f"{'chi2_P20':>9s} {'/dof':>5s} {'dchi2':>8s} | "
          f"{'dlnP_2D':>8s} {'paper':>6s} | {'DA_med':>7s} {'DA_log':>7s}")
    print("-" * 108)
    for r in rows:
        o, p = r["ours"], r["p20"]
        dl = (f"{r['dlnP_2D']:8.2f}"
              if r["dlnP_2D"] is not None else f"{'—':>8s}")
        pp = PAPER_DLNP.get(r["galaxy"], float("nan"))
        dal = (f"{r['D_A_log']:7.2f}"
               if r["D_A_log"] is not None else f"{'—':>7s}")
        print(f"{r['galaxy']:13s} {r['dof']:4d} | "
              f"{o['chi2_candel']:9.2f} {o['chi2_per_dof']:5.2f} "
              f"{p['chi2_candel']:9.2f} {p['chi2_per_dof']:5.2f} "
              f"{r['dchi2']:8.2f} | {dl} {pp:6.1f} | "
              f"{r['D_A_med']:7.2f} {dal}")

    out = os.path.join(rm.results_path(rm._MASTER_CFG["io"]["root_output"]),
                       "chi2_evidence_table.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
