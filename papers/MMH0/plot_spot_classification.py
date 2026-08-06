"""Write figs/spot_classification.pdf (PV diagram of maser spots, coloured by
spectral class) for the MMH0 paper. Plotting code lives in
candel.plotting.spot_classification; this script only loads data and saves.

Run with venv_candel from the CANDEL repo root:
    venv_candel/bin/python notebooks/paper_MMH0/plot_spot_classification.py
"""
import os
import sys
import tomllib
from os.path import abspath, dirname, join

import numpy as np

from candel.pvdata.megamaser_data import load_megamaser_spots, maser_data_root

sys.path.insert(0, dirname(abspath(__file__)))
from spot_classification import plot_spot_classification  # noqa: E402

REPO = abspath(join(dirname(__file__), "..", ".."))
CONFIG = join(REPO, "scripts", "megamaser", "config_maser.toml")
OUTDIR = "/Users/rstiskalek/Papers/MMH0/figs"
DATASET = "original_published"

# Display name -> config key.
GALAXIES = {
    "NGC 5765b": "NGC5765b",
    "NGC 6264": "NGC6264",
    "CGCG 074-064": "CGCG074-064",
}

with open(CONFIG, "rb") as f:
    cfg = tomllib.load(f)["model"]["galaxies"]

root = maser_data_root(DATASET)
galaxies = [
    (disp, load_megamaser_spots(root, key, v_sys_obs=cfg[key]["v_sys_obs"]))
    for disp, key in GALAXIES.items()
]

fig = plot_spot_classification(galaxies)
os.makedirs(OUTDIR, exist_ok=True)
out = join(OUTDIR, "spot_classification.pdf")
fig.savefig(out)
print(f"wrote {out}")

# Sanity check the headline claim: the velocity bands must not overlap.
for disp, d in galaxies:
    sys_mask = ~d["is_highvel"]
    dv = d["velocity"] - np.median(d["velocity"][sys_mask])
    blue_mask, red_mask = d["is_blue"], d["is_highvel"] & ~d["is_blue"]
    separated = (dv[blue_mask].max() < dv[sys_mask].min()
                 and dv[sys_mask].max() < dv[red_mask].min())
    assert separated, f"{disp}: velocity bands overlap"
    print(f"{disp}: blue<sys<red separated = {separated}")
