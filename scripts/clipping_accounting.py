# Copyright (C) 2026 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Spot accounting of our clipping against the MCP cut, both relative to the
unpruned table, under the linear warp.

Reads the stabilised linear-warp clipping masks in `data/Megamaser/clipped/`
and the `clipped_by_pesce` flag of `data/Megamaser/unpruned/provenance.csv`,
both of which are indexed row-for-row against the unpruned table.  No chain or
posterior is involved.
"""
import csv
from collections import Counter, defaultdict

import numpy as np

from candel_maser.megamaser_data import (clipped_mask_path,
                                         load_megamaser_spots,
                                         maser_data_root)
from candel.util import data_path

# Observed systemic velocities, from configs/config_maser.toml.
V_SYS_OBS = {"CGCG074-064": 7175, "NGC5765b": 8465, "UGC3789": 3245,
             "NGC6264": 10145, "NGC6323": 7662}

# NGC5765b rows carrying the a = 1.000 +/- 1.000 placeholder acceleration.
# These are absent from the P20 table because the journal's electronic table
# omitted them, not because the MCP vetting removed them, so they are counted
# separately from the vetted removals.
_PLACEHOLDER_TABLE = "NGC5765b_Gao2016_table6.dat"


def placeholder_mask(galaxy, n_spots):
    """Rows whose published acceleration is the 1.000 +/- 1.000 placeholder."""
    if galaxy != "NGC5765b":
        return np.zeros(n_spots, dtype=bool)
    path = maser_data_root("unpruned") + "/" + _PLACEHOLDER_TABLE
    rows = [line.split() for line in open(path)]
    return np.array([float(r[5]) == 1.0 and float(r[6]) == 1.0 for r in rows])


def load_masks(galaxy, data):
    """Our clip flag, the MCP-absent flag, and our clipping round, per row."""
    root = maser_data_root("unpruned")
    with open(clipped_mask_path(maser_data_root("clipped"), galaxy)) as f:
        rows = list(csv.DictReader(f))
    if len(rows) != data["n_spots"]:
        raise ValueError(f"{galaxy}: mask has {len(rows)} rows, table has "
                         f"{data['n_spots']}.")
    ours = np.array([r["clip"] == "True" for r in rows])
    rounds = [int(r["clipped_at_attempt"]) for r in rows if r["clip"] == "True"]
    with open(data_path("data", "Megamaser", "unpruned", "provenance.csv")) as f:
        prov = [r for r in csv.DictReader(f) if r["galaxy"] == galaxy]
    mcp = np.array([r["clipped_by_pesce"] == "True" for r in prov])
    velocity = np.array([float(r["velocity_km_s"]) for r in prov])
    if not np.allclose(velocity, data["velocity"], atol=1e-6, rtol=0.0):
        raise ValueError(f"{galaxy}: provenance velocities do not match {root}.")
    return ours, mcp, max(rounds) if rounds else 0


def main():
    total = Counter()
    per_galaxy = defaultdict(dict)
    for galaxy, v_sys in V_SYS_OBS.items():
        data = load_megamaser_spots(maser_data_root("unpruned"), galaxy,
                                    v_sys_obs=v_sys)
        n = data["n_spots"]
        systemic = ~np.asarray(data["is_highvel"])
        accel = np.asarray(data["accel_measured"])
        ours, mcp, rounds = load_masks(galaxy, data)
        placeholder = placeholder_mask(galaxy, n)
        vetted = mcp & ~placeholder
        per_galaxy[galaxy] = dict(
            n=n, ours=ours.sum(), mcp=mcp.sum(), vetted=vetted.sum(),
            placeholder=placeholder.sum(), both=(ours & vetted).sum(),
            ours_only=(ours & ~mcp).sum(), mcp_only=(vetted & ~ours).sum(),
            ours_sys=(ours & systemic).sum(), vetted_sys=(vetted & systemic).sum(),
            ours_accel=(ours & accel).sum(), vetted_accel=(vetted & accel).sum(),
            n_systemic=systemic.sum(), n_accel=accel.sum(), rounds=rounds)
        total.update({k: v for k, v in per_galaxy[galaxy].items()
                      if k != "rounds"})

    head = ("galaxy", "N_all", "MCP", "ours", "common", "N_MCP", "N_ours",
            "rounds")
    print(f"{head[0]:12s}" + "".join(f"{h:>8s}" for h in head[1:]))
    for galaxy, g in per_galaxy.items():
        print(f"{galaxy:12s}{g['n']:8d}{g['mcp']:8d}{g['ours']:8d}"
              f"{g['both']:8d}{g['n'] - g['mcp']:8d}{g['n'] - g['ours']:8d}"
              f"{g['rounds']:8d}")
    t = total
    print(f"{'total':12s}{t['n']:8d}{t['mcp']:8d}{t['ours']:8d}{t['both']:8d}"
          f"{t['n'] - t['mcp']:8d}{t['n'] - t['ours']:8d}")

    print(f"\nours:       {t['ours']:3d}/{t['n']} = {100 * t['ours'] / t['n']:.1f} "
          f"per cent, {t['ours_sys']} systemic, {t['ours_accel']} with accelerations")
    print(f"MCP vetted: {t['vetted']:3d}/{t['n']} = {100 * t['vetted'] / t['n']:.1f} "
          f"per cent, {t['vetted_sys']} systemic, {t['vetted_accel']} with accelerations")
    print(f"MCP absent: {t['mcp']:3d}/{t['n']} = {100 * t['mcp'] / t['n']:.1f} "
          f"per cent, including {t['placeholder']} NGC5765b placeholder rows")
    print(f"in common:  {t['both']:3d} = {100 * t['both'] / t['ours']:.0f} per cent "
          f"of ours, {100 * t['both'] / t['vetted']:.0f} per cent of the MCP vetted set")
    print(f"parent:     {t['n_systemic']} systemic and {t['n_accel']} "
          f"acceleration-measured spots of {t['n']}")


if __name__ == "__main__":
    main()
