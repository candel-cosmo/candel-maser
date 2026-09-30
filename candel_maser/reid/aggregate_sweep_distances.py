#!/usr/bin/env python3
"""Consolidate the maser distances from a whole gibbs sweep into one table.

A sweep (submit_gibbs_sweep.sh) lays out
  <sweep_dir>/<galaxy>_<init>/<variant>/chain_*/fort.7
and each (galaxy, init) combo writes its own R-hat + distance tables in its
own sub-directory. This walks every combo x variant, pools the post-burn
chains, and prints ONE table of the distance (median, asymmetric 1sigma)
and its R-hat, so the full sweep is comparable at a glance.

Safe to run while the sweep is still finishing: combos/variants without
chains (or with unreadable fort.7) are skipped with a note.

Usage:
    python aggregate_sweep_distances.py <sweep_dir> [--out FILE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from .run_gibbs_comparison import (VARIANTS, gelman_rubin,
                                   load_variant_chains, thin_chains)


def variant_distance(vdir):
    """(p16, p50, p84, R-hat) of D_Mpc, pooled over a variant's post-burn
    chains; R-hat is nan for a single chain."""
    arrs = load_variant_chains(vdir)
    d = [a["D_Mpc"].astype(np.float64) for a in arrs]
    p16, p50, p84 = np.percentile(np.concatenate(d), [16.0, 50.0, 84.0])
    if len(d) > 1:
        n = min(len(c) for c in d)
        rhat = gelman_rubin(thin_chains(np.array([c[-n:] for c in d])))
    else:
        rhat = float("nan")
    return p16, p50, p84, rhat


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="also write here (default: "
                         "<sweep_dir>/sweep_distances.txt)")
    args = ap.parse_args()
    if not args.sweep_dir.is_dir():
        sys.exit(f"[ERROR] not a directory: {args.sweep_dir}")

    header = (f"{'galaxy':12s} {'init':7s} {'variant':22s} "
              f"{'D_Mpc (med +hi -lo)':>24s} {'R-hat':>8s}")
    lines = [f"Sweep distance summary: {args.sweep_dir}", header]
    n_rows = 0
    for combo in sorted(d for d in args.sweep_dir.iterdir() if d.is_dir()):
        galaxy, sep, init = combo.name.rpartition("_")
        if not sep:                      # not a <galaxy>_<init> directory
            continue
        for vname, *_ in VARIANTS:
            vdir = combo / vname
            if not (vdir.is_dir() and any(vdir.glob("chain_*"))):
                continue
            try:
                p16, p50, p84, rhat = variant_distance(vdir)
            except Exception as exc:     # partial/unreadable -> note, skip
                lines.append(f"{galaxy:12s} {init:7s} {vname:22s} "
                             f"{'(skipped: ' + str(exc)[:24] + ')':>24s}")
                continue
            cell = f"{p50:.1f} +{p84 - p50:.1f} -{p50 - p16:.1f}"
            lines.append(f"{galaxy:12s} {init:7s} {vname:22s} "
                         f"{cell:>24s} {rhat:8.2f}")
            n_rows += 1

    if n_rows == 0:
        lines.append("(no completed variant chains found under sweep_dir)")
    report = "\n".join(lines)
    print(report)
    out = args.out or args.sweep_dir / "sweep_distances.txt"
    out.write_text(report + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
