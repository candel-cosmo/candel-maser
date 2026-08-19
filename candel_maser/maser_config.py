# Copyright (C) 2025 Richard Stiskalek
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
"""Spot-table dataset selection for the megamaser runners.

One switch, applied once per process: `apply_dataset` merges the dataset's
`init_<dataset>.toml` best points into the config and namespaces `root_output`
by dataset, so every downstream path follows without further edits.
"""
from os import environ
from os.path import basename, dirname, join, normpath

import tomli

from candel.pvdata.megamaser_data import MASER_DATASETS
from candel.util import fprint

_HERE = dirname(__file__)
ROOT_OUTPUT_ENV = "CANDEL_MEGAMASER_ROOT_OUTPUT"

# Keys a dataset file may set. Everything else must stay in config_maser.toml:
# the runners read force_f64 / phi_integration / conditional_spot_batch
# straight from sys.argv before argparse runs, so a per-dataset value would be
# silently ignored. Rejecting them keeps that impossible rather than latent.
_ALLOWED_PREFIXES = ("init", "r_ang_ref_")


def dataset_init_path(dataset):
    """Path to `dataset`'s per-galaxy init file."""
    return join(_HERE, f"init_{dataset}.toml")


def add_dataset_arg(parser):
    """Add ``--dataset`` to `parser`; default None means "use the config"."""
    parser.add_argument(
        "--dataset", type=str, default=None, choices=list(MASER_DATASETS),
        help="Megamaser spot-table dataset. Default: [io].dataset from the "
             "config file.")


def resolve_dataset(cfg, dataset=None):
    """Dataset to use: the explicit `dataset`, else the config's."""
    if dataset is None:
        dataset = cfg.get("io", {}).get("dataset")
    if dataset is None:
        raise ValueError(
            "No megamaser dataset selected: pass --dataset or set "
            "[io].dataset in the config file.")
    if dataset not in MASER_DATASETS:
        raise ValueError(
            f"Unknown megamaser dataset '{dataset}'. "
            f"Available: {list(MASER_DATASETS)}.")
    return dataset


def _merge_init(cfg, dataset):
    """Deep-merge `dataset`'s init file into `cfg`, in place."""
    path = dataset_init_path(dataset)
    with open(path, "rb") as f:
        extra = tomli.load(f)

    stray = [k for k in extra if k != "model"]
    stray += [k for k in extra.get("model", {}) if k != "galaxies"]
    for galaxy, blk in extra.get("model", {}).get("galaxies", {}).items():
        stray += [f"model.galaxies.{galaxy}.{k}" for k in blk
                  if not k.startswith(_ALLOWED_PREFIXES)]
    if stray:
        raise ValueError(
            f"'{path}' sets keys outside model.galaxies.<G>.init* and "
            f"model.galaxies.<G>.r_ang_ref_*: {sorted(stray)}. Those belong "
            f"in config_maser.toml — the runners read some of them before "
            f"argparse, so a per-dataset value would be silently ignored.")

    galaxies = cfg.setdefault("model", {}).setdefault("galaxies", {})
    for galaxy, blk in extra["model"]["galaxies"].items():
        galaxies.setdefault(galaxy, {}).update(blk)


def _namespace_root_output(cfg, dataset):
    """Append `dataset` to ``[io].root_output``, idempotently."""
    io = cfg.setdefault("io", {})
    if environ.get(ROOT_OUTPUT_ENV):
        io["root_output"] = environ[ROOT_OUTPUT_ENV]
    root = io.get("root_output")
    if root is None:
        return
    last = basename(normpath(root))
    if last == dataset:
        return
    if last in MASER_DATASETS:
        raise ValueError(
            f"[io].root_output '{root}' is already namespaced to dataset "
            f"'{last}', which is not the selected '{dataset}'. Point "
            f"root_output at the un-namespaced results directory.")
    io["root_output"] = join(root, dataset)


def check_init_block(init_cfg, model):
    """Check an init block against the spot table `model` was built on.

    Called from every runner's ``_clean_init``, the single point every init
    block passes through. Without this a wrong-dataset block runs to
    completion in silence: ``maser_blackjax`` accepts an ``r_ang`` init and
    falls back to zeros when its length does not match ``n_spots``, and
    ``r_ang`` is not among the validated scalar sites.
    """
    galaxy = getattr(model, "galaxy_name", None) or "<unknown galaxy>"
    dataset = getattr(model, "dataset", None) or "<unknown dataset>"

    if not init_cfg:
        raise SystemExit(
            f"No init block for galaxy '{galaxy}' in dataset '{dataset}'. "
            f"Add [model.galaxies.{galaxy}.init] to "
            f"scripts/megamaser/init_{dataset}.toml (run run_maser.py "
            f"{galaxy} --sampler de --dataset {dataset} to produce it). Fresh "
            f"linear and eccentric-only DE searches bootstrap without this "
            f"block; quadratic-warp DE searches require the linear [init].")

    r_ang = init_cfg.get("r_ang")
    n_spots = getattr(model, "n_spots", None)
    if r_ang is not None and n_spots is not None and len(r_ang) != n_spots:
        raise SystemExit(
            f"Init block for galaxy '{galaxy}' has {len(r_ang)} r_ang values "
            f"but dataset '{dataset}' has {n_spots} spots. This block belongs "
            f"to a different dataset — rerun DE for '{galaxy}' on "
            f"'{dataset}' and paste the result into "
            f"scripts/megamaser/init_{dataset}.toml.")
    return init_cfg


def check_chain_dataset(attrs, dataset, path):
    """Require an HDF5 chain to match the selected spot-table dataset."""
    chain_dataset = attrs.get("dataset", "original_published")
    if isinstance(chain_dataset, bytes):
        chain_dataset = chain_dataset.decode()
    chain_dataset = str(chain_dataset)
    if chain_dataset != dataset:
        raise ValueError(
            f"{path} was sampled on dataset {chain_dataset!r} but this run "
            f"selected {dataset!r}. Pass --dataset {chain_dataset}, or use "
            f"a chain sampled on {dataset!r}.")
    return chain_dataset


def apply_dataset(cfg, dataset=None):
    """Select the spot-table dataset for `cfg`, mutating it in place.

    Merges the dataset's init file and namespaces ``[io].root_output``. Call
    this first thing in ``main()``, after argparse and before any other config
    read; never rebind `cfg`, since importers hold the module global.

    Returns the resolved dataset name.
    """
    dataset = resolve_dataset(cfg, dataset)
    _merge_init(cfg, dataset)
    _namespace_root_output(cfg, dataset)
    cfg.setdefault("io", {})["dataset"] = dataset
    cfg.setdefault("model", {})["use_ngc5765b_clump2_floors"] = (
        dataset != "fiducial")
    fprint(f"megamaser dataset: {dataset} "
           f"(root_output '{cfg['io'].get('root_output')}').")
    return dataset


def variant_init_block(gal_cfg, model):
    """Variant-specific [init...] block, preferring the closest match.

    An ecc+quadratic-warp run wants [init_ecc_qw], but the quadratic
    coefficients are the expensive ones to find, so fall back to [init_qw]
    (then [init_ecc], then the linear [init]) rather than dropping straight
    to linear. `_clean_init` zeroes whatever coordinates the chosen block
    lacks and strips any the model does not use.
    """
    if model.use_ecc and model.use_quadratic_warp:
        names = ("init_ecc_qw", "init_qw", "init_ecc")
    elif model.use_quadratic_warp:
        names = ("init_qw",)
    elif model.use_ecc:
        names = ("init_ecc",)
    else:
        names = ()
    for name in names:
        if name in gal_cfg:
            fprint(f"init block: [{name}]")
            return gal_cfg[name]
    if names:
        fprint(f"init block: [{names[0]}] absent, falling back to [init]")
    return gal_cfg.get("init", {})
