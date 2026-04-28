# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Run DE MAP optimization for a maser disk galaxy and save results to TOML.

Usage:
    python run_de_map.py <galaxy_name> [--seed 42]
"""
import os
import sys

import tomli

with open(os.path.join(os.path.dirname(__file__), "../../local_config.toml"), "rb") as f:
    _lcfg = tomli.load(f)
ld = os.environ.get("LD_LIBRARY_PATH", "")
needed = [p for p in _lcfg.get("gpu_ld_library_path", []) if p not in ld]
if needed:
    os.environ["LD_LIBRARY_PATH"] = ":".join(needed) + (f":{ld}" if ld else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import tempfile
import time

import tomli

# DE is derivative-free so float32 is sufficient; use --f64 to override.
if "--f64" in sys.argv:
    sys.argv.remove("--f64")
    import jax
    jax.config.update("jax_enable_x64", True)
    print("float64 enabled (--f64)", flush=True)

import jax
import numpy as np
import tomli_w

from candel.inference.optimise import find_MAP
from candel.model.model_H0_maser import MaserDiskModel
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import data_path, fprint, fsection

_devs = jax.devices()
_dev_names = ", ".join(d.device_kind for d in _devs)
_precision = "float64" if jax.config.jax_enable_x64 else "float32"
print(f"JAX platform: {jax.default_backend()}, devices: {_devs} "
      f"({_dev_names}), precision: {_precision}", flush=True)

# ---- Load master config ----
with open("scripts/megamaser/config_maser.toml", "rb") as f:
    master_cfg = tomli.load(f)

# ---- Parse args ----
parser = argparse.ArgumentParser()
parser.add_argument("galaxy", type=str)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint if one exists")
args = parser.parse_args()

galaxy = args.galaxy
seed = args.seed

# ---- Validate galaxy ----
galaxies = master_cfg["model"]["galaxies"]
if galaxy not in galaxies:
    print(f"Unknown galaxy '{galaxy}'. "
          f"Available: {list(galaxies.keys())}", flush=True)
    sys.exit(1)

gcfg = galaxies[galaxy]
v_sys_obs = gcfg["v_sys_obs"]

# ---- Load data ----
fsection(f"Loading {galaxy} data")
_mcfg = master_cfg["model"]
data = load_megamaser_spots(
    data_path("data", "Megamaser"), galaxy, v_sys_obs=v_sys_obs)

if "D_lo" in gcfg and "D_hi" in gcfg:
    data["D_lo"] = float(gcfg["D_lo"])
    data["D_hi"] = float(gcfg["D_hi"])

# ---- Build model (force mode2 — DE requires r+phi marginalisation) ----
opt_cfg = dict(master_cfg.get("optimise", {}))
if "eval_chunk" in gcfg:
    opt_cfg["eval_chunk"] = gcfg["eval_chunk"]

config = {
    "inference": master_cfg["inference"],
    "model": {**master_cfg["model"], "mode": "mode2"},
    "io": master_cfg["io"],
    "optimise": opt_cfg,
}
config["model"]["galaxies"] = {
    g: {k: v for k, v in blk.items() if k != "mode"}
    for g, blk in master_cfg["model"]["galaxies"].items()
}

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()
model = MaserDiskModel(tmp.name, data)
os.unlink(tmp.name)

# ---- Run DE MAP ----
ckpt_dir = os.path.join(
    master_cfg["io"].get("root_output", "results/Megamaser"),
    "de_checkpoints", galaxy)
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "de_ckpt.npz")
resume_path = None
if args.resume and os.path.isfile(ckpt_path):
    resume_path = ckpt_path
    fprint(f"Resuming from {ckpt_path}")
elif args.resume:
    fprint(f"--resume: no checkpoint found at {ckpt_path}, starting fresh")
fprint(f"Checkpoints: {ckpt_dir}")

fsection(f"DE MAP optimization ({galaxy}, {data['n_spots']} spots)")
t0 = time.time()
init_params = find_MAP(model, model_kwargs={}, seed=seed,
                       checkpoint_dir=ckpt_dir, resume_path=resume_path)
dt = time.time() - t0

# ---- Print results ----
fsection(f"MAP results ({galaxy}, {dt:.0f}s)")
for k, v in sorted(init_params.items()):
    v = np.asarray(v)
    if v.ndim == 0:
        fprint(f"  {k:20s} = {float(v):12.4f}")
    else:
        fprint(f"  {k:20s} = [{len(v)} values]")

# ---- Print init values as TOML snippet (do NOT write back to config) ----
lines = [f"\n[model.galaxies.{galaxy}.init]"]
for k, v in sorted(init_params.items()):
    v = np.asarray(v)
    if v.ndim == 0:
        lines.append(f"{k} = {round(float(v), 4)}")
    else:
        vals = ", ".join(str(round(float(x), 4)) for x in v)
        lines.append(f"{k} = [{vals}]")
fprint("MAP init (copy into config_maser.toml manually if desired):")
print("\n".join(lines))
