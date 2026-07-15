#!/usr/bin/env python3
"""Generate reid_control_<GALAXY>.inp for a galaxy from mystart_globals.toml
(see make_candel_globals.py), for use as the --control-template of
run_gibbs_comparison.py / submit_gibbs_comparison.sh.

This is write_control() with the same fixed_params/step/seed defaults
run_reid_mcmc.py --prepare-only uses, except the H0 prior window (the only
field on that line a Gibbs run actually reads -- run_gibbs_chains.sh
overrides trials/walkers/burnin/seed itself) is a fixed, galaxy-independent
range rather than the auto galaxy-centered H0+-15: this mirrors how
reid_control_NGC6323.inp itself was produced (checked against its
run_metadata.json), keeping the flat-H0 prior consistently wide across every
MCP galaxy in the comparison.

The comparison's Pesce/Reid init point (--init pesce in
run_gibbs_comparison.py / submit_gibbs_comparison.sh) is generated from this
file's value column at run time -- it does not need a separate control
template.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from run_reid_mcmc import (ROOT, SCRIPT_DIR, compute_reid_r_ref,
                           load_toml_init, parse_data_rows, shift_warp_pivots,
                           write_control)

FIXED_PARAMS = {
    "ecc", "peri_az_deg", "dperi_dr_deg_mas",  # --fix-circular default
    "d2i_dr2_deg_mas2", "d2PA_dr2_deg_mas2",  # --linear-warp default
}

DEFAULT_INIT = SCRIPT_DIR / "mystart_globals.toml"


def default_data(galaxy: str) -> Path:
    return ROOT / f"data/Megamaser/{galaxy}_loader_reid.inp"


def default_out(galaxy: str) -> Path:
    return SCRIPT_DIR / f"reid_control_{galaxy}.inp"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--galaxy", required=True)
    p.add_argument("--init", type=Path, default=DEFAULT_INIT,
                   help="Globals TOML (default: mystart_globals.toml).")
    p.add_argument("--data", type=Path, default=None,
                   help="Reid-format data file (defaults to "
                        "data/Megamaser/<GALAXY>_loader_reid.inp).")
    p.add_argument("--h0-low", type=float, default=15.0)
    p.add_argument("--h0-high", type=float, default=210.0)
    p.add_argument("--out", type=Path, default=None,
                   help="Default: reid_control_<GALAXY>.inp")
    args = p.parse_args(argv)

    data = args.data or default_data(args.galaxy)
    out = args.out or default_out(args.galaxy)
    for path, label in ((args.init, "init TOML"), (data, "data file")):
        if not path.exists():
            p.error(f"missing {label}: {path}")

    reid_init = load_toml_init(args.init, args.galaxy, vcor=0.0,
                               variant="init")
    header, rows = parse_data_rows(data)
    reid_r_ref = compute_reid_r_ref(rows, header, reid_init.values)
    run_init = shift_warp_pivots(reid_init.values, reid_r_ref)

    write_control(
        out, run_init,
        burnin=100_000, trials=500_000, walkers=4,
        h0_low=args.h0_low, h0_high=args.h0_high,
        seed=47351937, step_fraction=0.015,
        fit_data=(True, True, True, True),
        fixed_params=FIXED_PARAMS,
    )
    out.write_text(
        out.read_text().replace(
            "!  Parameters for NGC 4258",
            f"!  Parameters for {args.galaxy} "
            "(CANDEL globals, Reid convention)",
            1))
    print(f"Wrote {out} (r_ref = {reid_r_ref:.6f} mas, "
          f"H0 in [{args.h0_low}, {args.h0_high}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
