"""Backfill the MAP chi^2 (CANDEL + original Reid Fortran) at the DE/config
globals for the MMH0 galaxies -- no MCMC needed.

Same computation as run_maser's --map-overlay, run standalone so existing runs
don't have to be resampled.  Prints a table and writes map_chi2_table.json.

    python scripts/megamaser/backfill_map_chi2.py
"""
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "check_reid"))
sys.path.insert(0, os.path.join(_HERE, "check_reid", "reidlik_build"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import reid_chi2  # noqa: E402
import run_map as rm  # noqa: E402

from candel.model.maser_map import evaluate_at_globals  # noqa: E402

GALAXIES = ["CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789"]


def main():
    rows = []
    for gal in GALAXIES:
        try:
            gcfg = rm._MASTER_CFG["model"]["galaxies"][gal]
            model, target, init = rm._build_target(gal, gcfg, None)
            ctx = reid_chi2.loglik_context(gal, model.n_spots)   # build once

            # Points to score: DE/config globals (r seeded from config) and the
            # reported Pesce 2020 globals (r from the conditional r-MAP).
            de_r = init.get("r_ang")
            points = [("DE", {n: init[n] for n in target.names},
                       np.asarray(de_r) if de_r is not None else None)]
            try:
                points.append(("Pesce", rm._pesce_globals(target, gal), None))
            except Exception as exc:                   # noqa: BLE001
                print(f"  {gal}: no Pesce globals ({exc})", flush=True)

            for label, gdict, init_r in points:
                res = evaluate_at_globals(
                    target, gdict, init_r_ang=init_r,
                    marginal=True, verbose=False)
                chi2_reid = None
                if ctx is not None:
                    nh = reid_chi2.neg_half_chi2(
                        ctx, gal, res["point"], res["r_ang"], res["phi"],
                        D_A=res.get("D_A"))
                    chi2_reid = float(-2.0 * np.asarray(nh).sum())
                rows.append(dict(
                    galaxy=gal, point=label, n_spots=int(model.n_spots),
                    dof=int(res["dof"]), chi2_candel=res["chi2"],
                    chi2_reid=chi2_reid, chi2_per_dof=res["chi2_per_dof"],
                    logP_marg=res["logP_marg"]))
            print(f"  {gal}: done", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {gal}: FAILED ({exc})", flush=True)

    hdr = (f"{'galaxy':14s} {'point':>6s} {'spots':>5s} {'dof':>4s} "
           f"{'chi2_CANDEL':>11s} {'chi2_Reid':>10s} {'chi2/dof':>8s} "
           f"{'logP_marg':>11s} {'rel_%':>6s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        rc = r["chi2_reid"]
        rel = 100.0 * abs(r["chi2_candel"] - rc) / rc if rc else float("nan")
        rc_s = f"{rc:10.3f}" if rc is not None else f"{'—':>10s}"
        lp = r.get("logP_marg")
        lp_s = f"{lp:11.3f}" if lp is not None else f"{'—':>11s}"
        print(f"{r['galaxy']:14s} {r['point']:>6s} {r['n_spots']:5d} "
              f"{r['dof']:4d} {r['chi2_candel']:11.3f} {rc_s} "
              f"{r['chi2_per_dof']:8.3f} {lp_s} {rel:6.2f}")

    out = os.path.join(
        rm.results_path(rm._MASTER_CFG["io"].get("root_output",
                                                 "results/Megamaser")),
        "map_chi2_table.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
