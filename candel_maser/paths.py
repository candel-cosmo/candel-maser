# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Filesystem locations of the megamaser package.

Import-light on purpose: the runners read these before importing JAX.
"""
import os
import tomllib
from importlib.util import find_spec

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PACKAGE_ROOT, "configs")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config_maser.toml")
# Core CANDEL checkout holding local_config.toml.
CANDEL_ROOT = os.path.dirname(os.path.dirname(find_spec("candel").origin))
LOCAL_CONFIG_PATH = os.path.join(CANDEL_ROOT, "local_config.toml")


def _local_root(key):
    # Default: the folder holding the checkouts (as candel.util).
    default = os.path.dirname(CANDEL_ROOT)
    try:
        with open(LOCAL_CONFIG_PATH, "rb") as f:
            return tomllib.load(f).get(key, default)
    except FileNotFoundError:
        return default


# Directories holding data/ and results/ (root_data / root_results in
# local_config.toml).
DATA_ROOT = _local_root("root_data")
RESULTS_ROOT = _local_root("root_results")
