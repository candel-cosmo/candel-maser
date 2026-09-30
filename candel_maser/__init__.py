# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Megamaser disk model and the two-stage megamaser H0 pipeline.

The runners are modules (``python -m candel_maser.run_maser``); this package
namespace stays import-light so they can set up the GPU environment first.
"""
_LAZY = {
    "DEFAULT_MASER_DATASET": "megamaser_data",
    "MASER_DATASETS": "megamaser_data",
    "load_megamaser_spots": "megamaser_data",
    "maser_data_root": "megamaser_data",
    "MaserDiskModel": "model_H0_maser",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        value = getattr(import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
