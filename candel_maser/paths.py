# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Filesystem locations of the megamaser package.

Import-light on purpose: the runners read these before importing JAX.
"""
import os
from importlib.util import find_spec

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PACKAGE_ROOT, "configs")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config_maser.toml")
# Repository holding local_config.toml, data/ and results/ (as candel.util).
CANDEL_ROOT = os.path.dirname(os.path.dirname(find_spec("candel").origin))
LOCAL_CONFIG_PATH = os.path.join(CANDEL_ROOT, "local_config.toml")
