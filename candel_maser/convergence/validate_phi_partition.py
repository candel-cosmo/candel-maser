#!/usr/bin/env python
"""Compare fixed-grid and peak-partition against converged 2D references.

The production model supplies every physical prediction, transform, prior,
spot split, conditional-radius grid, and both production phi integrators.
This script only supplies deterministic candidates, dense full-support
r x phi reference levels, convergence gates, timing, caching, and reports.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from jax import config as jax_config  # noqa: E402

jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlib  # noqa: E402
import numpy as np  # noqa: E402
import tomli_w  # noqa: E402

if jax.default_backend() != "gpu":
    # This must precede importing run_de_map, which configures the persistent
    # cache. Old CPU PjRt peak executables can terminate during
    # deserialisation.
    jax.config.update("jax_enable_compilation_cache", False)

from ..paths import PACKAGE_ROOT as _PACKAGE_ROOT  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
MASER_DIR = SCRIPT_DIR.parent
PACKAGE_ROOT = Path(_PACKAGE_ROOT)

from .. import run_de_map as de  # noqa: E402
from ..maser_config import add_dataset_arg, apply_dataset  # noqa: E402

try:  # noqa: E402
    from .convergence_utils import (cast_floats, cast_model_floats,
                                    dense_r_phi_reference_per_spot)
except ImportError:  # direct script execution
    from .convergence_utils import (cast_floats, cast_model_floats,
                                    dense_r_phi_reference_per_spot)


SCHEMA_VERSION = 4
REFERENCE_CACHE_SCHEMA_VERSION = 4
REFERENCE_POLICY = "full_support_log_r_partition_phi_trapezoid_f64_v1"
REFERENCE_CACHE_TTL_SECONDS = 48 * 60 * 60
# Scheme-specific numerical experiments belong in these per-galaxy overrides;
# reports retain the resolved settings separately for each scheme.
INTEGRATION_SCHEMES = {
    "fixed-grid": {"phi_integration": "fixed-grid"},
    "peak-partition": {"phi_integration": "peak-partition"},
}
METHODS = tuple(INTEGRATION_SCHEMES)
COMMON_SCHEME_SETTINGS = {
    "n_r_local": (int, 3),
    "n_r_global": (int, 3),
    "K_sigma": (float, 0.0),
    "n_refine_steps": (int, 1),
    "refine_r_center": (bool, None),
    "global_r_full_support": (bool, None),
    "scan_width_drop": (float, 0.0),
    "asymmetric_r_local": (bool, None),
}
PARTITION_FUNCTION_SETTINGS = {
    "root_capacity": (int, 1),
    "root_steps": (int, 1),
    "root_order": (int, 5),
    "drop": (float, 0.0),
    "drop_steps": (int, 1),
    "core_order": (int, 1),
    "tail_order": (int, 1),
}
SCHEME_SETTINGS = {
    "fixed-grid": {
        **COMMON_SCHEME_SETTINGS,
        "n_phi_sys": (int, 3),
        "n_phi_hv_high": (int, 3),
        "n_phi_hv_low": (int, 3),
    },
    "peak-partition": {
        **COMMON_SCHEME_SETTINGS,
        **PARTITION_FUNCTION_SETTINGS,
        "n_phi_partition_sys": (int, 3),
        "n_phi_partition_hv": (int, 3),
        "peak_r_refine_steps": (int, 0),
        "peak_r_refine_order": (int, 3),
        "peak_r_refine_hv_only": (bool, None),
        "peak_r_width_steps": (int, 0),
    },
}
VARIANTS = {
    "circular": (False, False),
    "eccentric": (True, False),
    "quadratic-warp": (False, True),
    "eccentric-quadratic-warp": (True, True),
}
VARIANT_ALIASES = {
    "ecc": "eccentric",
    "qw": "quadratic-warp",
    "ecc-qw": "eccentric-quadratic-warp",
    "ecc_qw": "eccentric-quadratic-warp",
}
POPULATIONS = ("systemic", "red", "blue")
DEFAULT_REFERENCE_R_LEVELS = (5001, 10001, 20001)
DEFAULT_REFERENCE_PHI_LEVELS = (2501, 5001, 10001)
NGC4258_REFERENCE_R_LEVELS = (20001, 40001, 80001, 160001)
NGC4258_REFERENCE_PHI_LEVELS = (50001,) * 4

# Irrelevance classifier.  Every candidate is judged against the full dense
# reference ladder (computed or loaded exactly from cache); this only labels
# posterior-irrelevant broad Sobol needles so their failures are excused from
# the verdict.  A candidate is flagged when
# its production deficit falls below a gate set at ``GATE_MULTIPLIER`` times
# the worst legitimate (anchor/local-cloud) deficit, floored at ``GATE_FLOOR``
# nats, its finest-reference deficit also clears the gate, and its needle
# geometry rails.
GATE_FLOOR = -1000.0
GATE_MULTIPLIER = 100.0
RAILING_FRACTION_THRESHOLD = 0.1
PHI_EDGE_CELLS = 2


def _source_hash():
    """Hash code that can change the physical dense reference values.

    The validator itself is deliberately excluded: reporting, CLI, and
    production-setting changes must not invalidate an unchanged reference.
    ``REFERENCE_POLICY`` is bumped when the reference algorithm changes.
    """
    paths = (
        MASER_DIR / "model_H0_maser.py",
        MASER_DIR / "run_de_map.py",
        SCRIPT_DIR / "convergence_utils.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PACKAGE_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validator_source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _numerical_settings(model):
    signature = inspect.signature(model._phi_partition_log_integral)
    partition = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    partition["root_capacity"] = int(
        model._phi_partition_root_capacity)
    return {
        "n_phi_sys": int(model._n_phi_sys),
        "n_phi_hv_high": int(model._n_phi_hv_high),
        "n_phi_hv_low": int(model._n_phi_hv_low),
        "n_phi_partition_sys": int(model._n_phi_partition_sys),
        "n_phi_partition_hv": int(model._n_phi_partition_hv),
        "n_r_local": int(model._n_r_local),
        "n_r_global": int(model._n_r_global),
        "K_sigma": float(model._K_sigma),
        "global_r_full_support": bool(model._global_r_full_support),
        "scan_width_drop": float(model._scan_width_drop),
        "asymmetric_r_local": bool(model._asymmetric_r_local),
        "peak_r_refine_steps": int(model._peak_r_refine_steps),
        "peak_r_refine_order": int(model._peak_r_refine_order),
        "peak_r_refine_hv_only": bool(model._peak_r_refine_hv_only),
        "peak_r_width_steps": int(model._peak_r_width_steps),
        "conditional_spot_batch": model._conditional_spot_batch,
        **partition,
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _canonical_json(value):
    return json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False)


def _sha256_json(value):
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _parse_levels(value):
    try:
        levels = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "grid levels must be comma-separated integers") from exc
    if len(levels) < 2 or any(level < 3 for level in levels):
        raise argparse.ArgumentTypeError(
            "grid levels need at least two entries, each >= 3")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise argparse.ArgumentTypeError(
            "grid levels must be strictly increasing")
    return levels


def _variant(value):
    value = VARIANT_ALIASES.get(value, value)
    if value not in VARIANTS:
        raise argparse.ArgumentTypeError(
            f"variant must be one of {', '.join(VARIANTS)}")
    return value


def _parse_scheme_setting(value):
    """Parse and validate ``METHOD.KEY=VALUE`` numerical overrides."""
    try:
        qualified_key, raw = value.split("=", 1)
        method, key = qualified_key.split(".", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "scheme settings must use METHOD.KEY=VALUE") from exc
    if method not in SCHEME_SETTINGS:
        raise argparse.ArgumentTypeError(
            f"unknown integration method {method!r}; choose from "
            f"{', '.join(METHODS)}")
    if key not in SCHEME_SETTINGS[method]:
        allowed = ", ".join(sorted(SCHEME_SETTINGS[method]))
        raise argparse.ArgumentTypeError(
            f"{key!r} is not tunable for {method}; choose from {allowed}")
    expected, minimum = SCHEME_SETTINGS[method][key]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid JSON scalar {raw!r} for {method}.{key}") from exc
    if expected is bool:
        valid_type = isinstance(parsed, bool)
    elif expected is int:
        valid_type = isinstance(parsed, int) and not isinstance(parsed, bool)
    else:
        valid_type = (isinstance(parsed, (int, float))
                      and not isinstance(parsed, bool))
        parsed = float(parsed) if valid_type else parsed
    if not valid_type:
        raise argparse.ArgumentTypeError(
            f"{method}.{key} requires a {expected.__name__} value")
    if minimum is not None:
        valid_value = (
            parsed >= minimum if expected is int else parsed > minimum)
        if not valid_value:
            relation = ">=" if expected is int else ">"
            raise argparse.ArgumentTypeError(
                f"{method}.{key} must be {relation} {minimum:g}")
    return method, key, parsed


def _scheme_overrides(settings):
    overrides = {method: {} for method in METHODS}
    for method, key, value in settings:
        if key in overrides[method]:
            raise ValueError(
                f"duplicate --scheme-setting for {method}.{key}")
        overrides[method][key] = value
    return overrides


def _reference_grids(galaxy, args):
    r_levels = args.reference_r_levels
    if not r_levels:
        r_levels = (NGC4258_REFERENCE_R_LEVELS
                    if galaxy == "NGC4258"
                    else DEFAULT_REFERENCE_R_LEVELS)
    phi_levels = args.reference_phi_levels
    if phi_levels is None:
        phi_levels = (NGC4258_REFERENCE_PHI_LEVELS
                      if galaxy == "NGC4258"
                      else DEFAULT_REFERENCE_PHI_LEVELS)
    if len(r_levels) != len(phi_levels):
        raise ValueError(
            "--reference-r-levels and --reference-phi-levels must have "
            "the same number of entries.")
    return tuple(zip(r_levels, phi_levels))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arg(parser)
    parser.add_argument(
        "--galaxies", nargs="+",
        default=list(de._MASTER_CFG["model"]["galaxies"]),
        help="Galaxies to validate (default: all configured galaxies).")
    parser.add_argument("--variants", nargs="+", type=_variant,
                        default=["circular"])
    parser.add_argument("--sobol-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-sobol", type=int, default=0,
                        help="Extra Sobol points around each available config "
                             "and Pesce/Reid point.")
    parser.add_argument("--local-radius", type=float, default=0.02,
                        help="Half-width of each local cloud in the DE unit "
                             "box (default: 0.02).")
    parser.add_argument("--no-config-point", action="store_true")
    parser.add_argument("--no-pesce-point", action="store_true")
    parser.add_argument(
        "--checkpoint-candidate", action="append", type=Path, default=[],
        help="Add the best solution from a compatible DE checkpoint NPZ as "
             "a fully validated candidate; may be repeated.")
    parser.add_argument(
        "--allow-checkpoint-policy-mismatch", action="store_true",
        help="Permit a checkpoint coordinate saved under a different "
             "numerical objective to be rescored for an explicit method "
             "comparison. Layout, bounds, and finite-value checks remain "
             "strict; the mismatch is recorded in the report.")
    parser.add_argument(
        "--reference-r-levels", type=_parse_levels,
        default=(),
        help="Radial reference levels (default: 5001,10001,20001; "
             "NGC4258: 20001,40001,80001,160001).")
    parser.add_argument(
        "--reference-phi-levels", type=_parse_levels,
        default=None,
        help="Phi reference levels (default: 2501,5001,10001; "
             "NGC4258: 50001 at every radial level).")
    parser.add_argument("--reference-tail-levels", type=int, default=3,
                        help="Number of final levels whose consecutive "
                             "comparisons must pass (default: 3).")
    parser.add_argument("--reference-total-atol", type=float, default=1e-2)
    parser.add_argument("--reference-spot-atol", type=float, default=1e-3)
    parser.add_argument("--reference-rms-atol", type=float, default=1e-4)
    parser.add_argument("--total-atol", type=float, default=0.1)
    parser.add_argument("--spot-atol", type=float, default=0.01)
    parser.add_argument("--spot-p99-atol", type=float, default=0.005)
    parser.add_argument("--spot-rms-atol", type=float, default=0.001)
    parser.add_argument("--ranking-atol", type=float, default=1e-4)
    parser.add_argument("--max-ranking-inversions", type=int, default=0)
    parser.add_argument("--max-root-overflows", type=int, default=0)
    parser.add_argument("--spot-batch", type=int, default=None,
                        help="Production spot-batch override.")
    parser.add_argument("--reference-spot-batch", type=int, default=4)
    parser.add_argument("--reference-r-chunk", type=int, default=32)
    parser.add_argument("--candidate-wave", type=int,
                        choices=(1, 2, 4, 8), default=None,
                        help="Peak-partition candidate-wave override.")
    parser.add_argument(
        "--force-production-f64", action="store_true",
        help="Evaluate both production schemes in float64 even when the "
             "galaxy is normally configured for float32. This is a "
             "diagnostic override; the independent reference is always "
             "float64.")
    parser.add_argument(
        "--scheme-setting", action="append", type=_parse_scheme_setting,
        default=[], metavar="METHOD.KEY=VALUE",
        help="Override one whitelisted production numerical setting; may be "
             "repeated (for example peak-partition.n_phi_partition_sys=257).")
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, default=None)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--no-cache", action="store_true")
    cache.add_argument(
        "--clean-cache", action="store_true",
        help="Delete cached dense references and exit without validating.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Permit a non-GPU backend for development only.")
    return parser


def _validate_args(args):
    known = de._MASTER_CFG["model"]["galaxies"]
    unknown = [galaxy for galaxy in args.galaxies if galaxy not in known]
    if unknown:
        raise ValueError(
            f"unknown galaxies {unknown}; available: {list(known)}")
    if args.sobol_candidates < 0 or args.local_sobol < 0:
        raise ValueError("Sobol candidate counts must be non-negative.")
    if not 0.0 < args.local_radius <= 0.5:
        raise ValueError("--local-radius must be in (0, 0.5].")
    if args.reference_spot_batch < 1 or args.reference_r_chunk < 1:
        raise ValueError("reference spot batch and r chunk must be positive.")
    if args.spot_batch is not None and args.spot_batch < 1:
        raise ValueError("--spot-batch must be positive.")
    if args.timing_repeats < 1 or args.n_devices < 1:
        raise ValueError("timing repeats and device count must be positive.")
    missing_checkpoints = [
        str(path) for path in args.checkpoint_candidate if not path.is_file()]
    if missing_checkpoints:
        raise ValueError(
            "checkpoint candidates do not exist: "
            + ", ".join(missing_checkpoints))
    if args.reference_tail_levels < 2:
        raise ValueError("--reference-tail-levels must be at least 2.")
    for galaxy in args.galaxies:
        grids = _reference_grids(galaxy, args)
        if args.reference_tail_levels > len(grids):
            raise ValueError(
                f"the {galaxy} reference has {len(grids)} levels but "
                f"--reference-tail-levels={args.reference_tail_levels}.")
    _scheme_overrides(args.scheme_setting)


def error_statistics(test, reference):
    """Finite-mask and cancellation-resistant error statistics."""
    test = np.asarray(test, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if test.shape != reference.shape:
        raise ValueError("test and reference arrays must have equal shape.")
    finite_test = np.isfinite(test)
    finite_reference = np.isfinite(reference)
    common = finite_test & finite_reference
    diff = test[common] - reference[common]
    abs_diff = np.abs(diff)
    all_finite = bool(np.all(common))
    signed_total = (float(np.sum(test) - np.sum(reference))
                    if all_finite else np.nan)
    if len(diff):
        maximum = float(np.max(abs_diff))
        median = float(np.median(abs_diff))
        p95 = float(np.percentile(abs_diff, 95))
        p99 = float(np.percentile(abs_diff, 99))
        rms = float(np.sqrt(np.mean(diff * diff)))
        sum_abs = float(np.sum(abs_diff))
        worst = int(np.flatnonzero(common)[np.argmax(abs_diff)])
    else:
        maximum = median = p95 = p99 = rms = sum_abs = np.nan
        worst = None
    return {
        "signed_total_error": signed_total,
        "absolute_total_error": abs(signed_total),
        "sum_absolute_spot_error": sum_abs,
        "max_absolute_spot_error": maximum,
        "median_absolute_spot_error": median,
        "p95_absolute_spot_error": p95,
        "p99_absolute_spot_error": p99,
        "rms_spot_error": rms,
        "finite_test": int(np.count_nonzero(finite_test)),
        "finite_reference": int(np.count_nonzero(finite_reference)),
        "finite_mask_mismatches": int(np.count_nonzero(
            finite_test != finite_reference)),
        "all_spots_finite": all_finite,
        "worst_spot": worst,
    }


def reference_convergence(level_values, tail_levels, criteria):
    """Consecutive-level diagnostics and an explicit convergence verdict."""
    comparisons = []
    for previous, current in zip(level_values, level_values[1:]):
        stats = error_statistics(current, previous)
        stats["passed"] = bool(
            stats["finite_mask_mismatches"] == 0
            and stats["all_spots_finite"]
            and stats["absolute_total_error"] <= criteria["total_atol"]
            and stats["max_absolute_spot_error"] <= criteria["spot_atol"]
            and stats["rms_spot_error"] <= criteria["rms_atol"])
        comparisons.append(stats)
    required = comparisons[-(tail_levels - 1):]
    return {
        "converged": bool(len(required) == tail_levels - 1
                          and all(row["passed"] for row in required)),
        "tail_levels": int(tail_levels),
        "criteria": dict(criteria),
        "comparisons": comparisons,
    }


def ranking_inversions(previous, current, labels, atol):
    """Pairwise ordering flips, ignoring unresolved near-ties."""
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    out = []
    for i in range(len(previous)):
        for j in range(i + 1, len(previous)):
            values = previous[i], previous[j], current[i], current[j]
            if not np.all(np.isfinite(values)):
                continue
            before = previous[i] - previous[j]
            after = current[i] - current[j]
            if abs(before) <= atol or abs(after) <= atol:
                continue
            if np.signbit(before) != np.signbit(after):
                out.append({
                    "candidate_a": labels[i],
                    "candidate_b": labels[j],
                    "previous_separation": float(before),
                    "current_separation": float(after),
                })
    return out


def reference_cache_key(metadata):
    return _sha256_json(metadata)


def _git_metadata():
    def run(*args):
        try:
            result = subprocess.run(
                args, cwd=PACKAGE_ROOT, capture_output=True,
                text=True, check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {
        "revision": revision,
        "dirty": None if dirty is None else bool(dirty),
    }


def _write_model_config(config, data, dtype):
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml",
                                      delete=False)
    try:
        tomli_w.dump(config, tmp)
        tmp.close()
        model = de.MaserDiskModel(tmp.name, copy.deepcopy(data))
    finally:
        if not tmp.closed:
            tmp.close()
        os.unlink(tmp.name)
    return cast_model_floats(model, dtype)


def _apply_partition_overrides(model, overrides):
    settings = {
        key: value for key, value in overrides.items()
        if key in PARTITION_FUNCTION_SETTINGS
    }
    root_capacity = settings.pop("root_capacity", None)
    if root_capacity is not None:
        model._phi_partition_root_capacity = int(root_capacity)
    if settings:
        model._phi_partition_log_integral = partial(
            model._phi_partition_log_integral, **settings)


def _build_case(galaxy, variant, args, seed):
    master = de._MASTER_CFG
    gcfg = master["model"]["galaxies"][galaxy]
    configured_phi_integration = gcfg.get(
        "phi_integration", master["model"].get(
            "phi_integration", "fixed-grid"))
    data = de.load_megamaser_spots(
        de.maser_data_root(master["io"]["dataset"]), galaxy,
        v_sys_obs=gcfg["v_sys_obs"], use_ecc=VARIANTS[variant][0],
        use_quadratic_warp=VARIANTS[variant][1])
    distance_bounds = de._distance_bounds(gcfg)
    if distance_bounds is not None:
        data["D_lo"], data["D_hi"] = distance_bounds[:2]

    config = copy.deepcopy(master)
    case_gcfg = config["model"]["galaxies"][galaxy]
    case_gcfg["use_ecc"], case_gcfg["use_quadratic_warp"] = (
        VARIANTS[variant])
    # ``--spot-batch`` changes only the exact execution tiling of a
    # production objective.  Capture the reference identity before applying
    # that override so GPU batching frontiers reuse the same independently
    # computed dense arrays.  Variant flags stay in the hash because they
    # change the physical predictions.
    reference_config_hash = _sha256_json(config["model"])
    if args.spot_batch is not None:
        case_gcfg["conditional_spot_batch"] = int(args.spot_batch)

    production_f64 = bool(
        gcfg.get("force_f64", False) or args.force_production_f64)
    production_dtype = jnp.float64 if production_f64 else jnp.float32
    method_configs = {}
    models = {}
    scheme_overrides = _scheme_overrides(args.scheme_setting)
    for method, overrides in INTEGRATION_SCHEMES.items():
        method_config = copy.deepcopy(config)
        method_config["model"]["galaxies"][galaxy].update(overrides)
        method_config["model"]["galaxies"][galaxy].update(
            scheme_overrides[method])
        method_configs[method] = method_config
        models[method] = _write_model_config(
            method_config, data, production_dtype)
        _apply_partition_overrides(models[method], scheme_overrides[method])
    model = models["peak-partition"]
    reference_model = _write_model_config(
        method_configs["fixed-grid"], data, jnp.float64)

    suffix = de._variant_suffix(model)
    variant_init_name = "init" + suffix if suffix else "init"
    if variant_init_name in case_gcfg:
        init_block_name = variant_init_name
        init_note = None
    else:
        init_block_name = "init"
        init_note = (
            f"[{variant_init_name}] absent; production fallback [init] used"
            if suffix else None)
    init_status = None
    try:
        init_block = de._init_block(case_gcfg, model)
        init_params = de._make_init(
            model, init_block, "config",
            int(de._required_inference(
                master["inference"], "init_num_samples")),
            jax.random.PRNGKey(seed))
    except (KeyError, ValueError) as exc:
        init_status = str(exc)
        init_params = de._make_init(
            model, {}, "median",
            int(de._required_inference(
                master["inference"], "init_num_samples")),
            jax.random.PRNGKey(seed))
    init_params = cast_floats(init_params, production_dtype)
    if init_status is None:
        try:
            de.MaserBlackJaxTarget(
                model, de._h_ref(model), init_params, spot_batch=None)
        except (KeyError, ValueError) as exc:
            init_status = str(exc)
            init_params = cast_floats(de._make_init(
                model, {}, "median",
                int(de._required_inference(
                    master["inference"], "init_num_samples")),
                jax.random.PRNGKey(seed)), production_dtype)

    configured_batch = case_gcfg.get("conditional_spot_batch")
    if args.spot_batch is None:
        configured_batch = gcfg.get("conditional_spot_batch")
    target_batch, batch_source = de._de_spot_batch_policy(
        galaxy, production_f64, args.spot_batch,
        configured_batch, None)
    targets = {
        method: de.MaserBlackJaxTarget(
            method_model, de._h_ref(method_model), init_params,
            spot_batch=target_batch)
        for method, method_model in models.items()
    }
    target = targets["peak-partition"]
    reference_init = cast_floats(init_params, jnp.float64)
    reference_target = de.MaserBlackJaxTarget(
        reference_model, de._h_ref(reference_model), reference_init,
        spot_batch=args.reference_spot_batch)
    if (any(candidate.names != target.names for candidate in targets.values())
            or target.names != reference_target.names):
        raise RuntimeError("production and reference target layouts differ.")

    sobol_n_sigma = master.get("optimise", {}).get("sobol_n_sigma", 5)
    names, sizes, lo, hi = de._layout(target, sobol_n_sigma)
    return {
        "galaxy": galaxy,
        "variant": variant,
        "config": config,
        "config_hash": reference_config_hash,
        "model": model,
        "models": models,
        "reference_model": reference_model,
        "target": target,
        "targets": targets,
        "reference_target": reference_target,
        "init_params": init_params,
        "init_status": init_status,
        "init_block_name": init_block_name,
        "init_note": init_note,
        "names": names,
        "sizes": sizes,
        "lo": lo,
        "hi": hi,
        "production_dtype": (
            "float64" if production_f64 else "float32"),
        "configured_phi_integration": configured_phi_integration,
        "validated_phi_integrations": list(METHODS),
        "spot_batch": target_batch,
        "spot_batch_source": batch_source,
        "source_hash": _source_hash(),
        "scheme_setting_overrides": scheme_overrides,
        "numerical_settings": {
            method: _numerical_settings(method_model)
            for method, method_model in models.items()
        },
    }


def _sobol(n, dimension, seed):
    if n == 0:
        return np.empty((0, dimension))
    exponent = max(0, (int(n) - 1).bit_length())
    return de.Sobol(d=dimension, scramble=True, seed=seed).random_base2(
        exponent)[:n]


def _reflect_unit_box(values):
    values = np.abs(values)
    cycle = np.floor(values).astype(np.int64)
    fraction = values - np.floor(values)
    return np.where(cycle % 2 == 0, fraction, 1.0 - fraction)


def _load_checkpoint_candidate(path, case, allow_policy_mismatch=False):
    """Load one physical DE point after strict layout/policy validation."""
    required = {
        "best_solution", "best_fitness", "lo", "hi", "names", "sizes",
        "objective_policy",
    }
    with np.load(path, allow_pickle=False) as checkpoint:
        missing = sorted(required.difference(checkpoint.files))
        if missing:
            raise ValueError(
                f"checkpoint candidate {path} is missing {missing}.")
        names = [str(value) for value in checkpoint["names"]]
        if names != list(case["names"]):
            raise ValueError(
                f"checkpoint candidate {path} parameter names do not match "
                f"{case['galaxy']}/{case['variant']}.")
        if not np.array_equal(checkpoint["sizes"], case["sizes"]):
            raise ValueError(
                f"checkpoint candidate {path} parameter sizes do not match.")
        lo = np.asarray(checkpoint["lo"], dtype=np.float64)
        hi = np.asarray(checkpoint["hi"], dtype=np.float64)
        if (not np.allclose(lo, case["lo"], rtol=0.0, atol=0.0)
                or not np.allclose(hi, case["hi"], rtol=0.0, atol=0.0)):
            raise ValueError(
                f"checkpoint candidate {path} DE bounds do not match.")
        saved_policy = str(
            np.asarray(checkpoint["objective_policy"]).item())
        expected_policy = de._objective_policy(
            case["models"]["peak-partition"])
        policy_matches = saved_policy == expected_policy
        if not policy_matches and not allow_policy_mismatch:
            raise ValueError(
                f"checkpoint candidate {path} objective policy "
                f"{saved_policy!r} does not match {expected_policy!r}.")
        unit = np.asarray(checkpoint["best_solution"], dtype=np.float64)
        if (unit.shape != lo.shape or not np.all(np.isfinite(unit))
                or np.any(unit < 0.0) or np.any(unit > 1.0)):
            raise ValueError(
                f"checkpoint candidate {path} best_solution is not a finite "
                "point in the DE unit box.")
        fitness = float(np.asarray(checkpoint["best_fitness"]).item())
        if not np.isfinite(fitness):
            raise ValueError(
                f"checkpoint candidate {path} best_fitness is not finite.")
    return lo + unit * (hi - lo), fitness, {
        "saved": saved_policy,
        "evaluated": expected_policy,
        "matched": policy_matches,
    }


def _candidate_rows(case, args, seed):
    names, lo, hi = case["names"], case["lo"], case["hi"]
    scale = hi - lo
    rows = []
    status = []
    if case["init_note"] is not None:
        status.append(case["init_note"])

    def add(source, source_kind, values):
        values = np.asarray(values, dtype=np.float64)
        source_index = sum(
            row["source_kind"] == source_kind for row in rows)
        rows.append({
            "id": f"{source_kind}-{source_index:04d}",
            "source": source,
            "source_kind": source_kind,
            "values": values,
            "within_de_bounds": bool(np.all(
                (values >= lo) & (values <= hi))),
        })

    # Order anchors and their local clouds before the broad Sobol points so
    # the report reads anchor-first; the irrelevance gate is calibrated on the
    # legitimate high-posterior candidates.  Ids are per-source_kind counters,
    # so this ordering does not change any candidate id.
    anchors = []
    if not args.no_config_point:
        if case["init_status"] is None:
            values = de._theta_to_flat(case["init_params"], names)
            add(f"configured [{case['init_block_name']}]", "config", values)
            anchors.append(("config", values))
        else:
            status.append(
                f"configured point unavailable: {case['init_status']}")

    if not args.no_pesce_point:
        try:
            point, defaults = de._pesce_init(
                case["target"], case["galaxy"], de._MASTER_CFG)
            values = de._theta_to_flat(point, names)
            label = "Pesce/Reid reference"
            if defaults:
                label += "; defaulted " + ", ".join(defaults)
            add(label, "pesce-reid", values)
            anchors.append(("pesce-reid", values))
        except (KeyError, ValueError) as exc:
            status.append(f"Pesce/Reid point unavailable: {exc}")

    for path in args.checkpoint_candidate:
        values, fitness, policy = _load_checkpoint_candidate(
            path, case, args.allow_checkpoint_policy_mismatch)
        mismatch = ("; coordinate rescored under an explicitly different "
                    "objective policy" if not policy["matched"] else "")
        add(f"DE checkpoint {path} (saved logP={-fitness:.6g}{mismatch})",
            "de-checkpoint", values)
        rows[-1]["checkpoint_policy"] = policy

    local_seed_offset = {"config": 0, "pesce-reid": 1}
    for kind, values in anchors:
        centre = (np.asarray(values) - lo) / scale
        cloud = _sobol(
            args.local_sobol, len(names),
            seed + 1000 + local_seed_offset[kind])
        for index, unit in enumerate(cloud):
            local = centre + (2.0 * unit - 1.0) * args.local_radius
            local = _reflect_unit_box(local)
            add(f"local Sobol around {kind} index={index}",
                f"local-{kind}", lo + local * scale)

    broad = _sobol(args.sobol_candidates, len(names), seed)
    for index, unit in enumerate(broad):
        add(f"scrambled Sobol seed={seed} index={index}", "sobol",
            lo + unit * scale)

    if not rows:
        raise ValueError("candidate collection is empty.")
    return rows, status


def _partition_diagnostics(model, type_key, idx, r_ang, phys_args, phys_kw,
                           spot_batch):
    """Root diagnostics in fixed spot batches, separate from the likelihood."""
    n_idx = int(idx.shape[0])
    batch = n_idx if spot_batch is None else min(int(spot_batch), n_idx)

    def evaluate(idx_b, r_b):
        r_pre = model._r_precompute(
            r_b, idx_b, *phys_args, **phys_kw,
            has_any_accel=model._group_has_any_accel(type_key))
        _, roots, overflow = model._phi_partition_group_log_integral(
            type_key, r_pre, model._phi_partition_scan_size(type_key))
        return roots.astype(jnp.int32), overflow

    if batch >= n_idx:
        return evaluate(idx, r_ang)
    n_chunks = (n_idx + batch - 1) // batch
    n_pad = n_chunks * batch - n_idx
    idx_p = jnp.concatenate([idx, idx[:n_pad]]) if n_pad else idx
    r_p = (jnp.concatenate([r_ang, r_ang[:n_pad]], axis=0)
           if n_pad else r_ang)
    idx_c = idx_p.reshape(n_chunks, batch)
    r_c = r_p.reshape(n_chunks, batch, *r_p.shape[1:])

    def body(_, values):
        roots, overflow = evaluate(*values)
        return None, (roots, overflow)

    _, (roots, overflow) = jax.lax.scan(body, None, (idx_c, r_c))
    return (roots.reshape(-1, *roots.shape[2:])[:n_idx],
            overflow.reshape(-1, *overflow.shape[2:])[:n_idx])


def _gate(legit_scores, anchor_score):
    """Posterior-deficit gate: min(floor, multiplier x worst legit deficit)."""
    worst_legit = min(legit_scores) - anchor_score
    return min(GATE_FLOOR, GATE_MULTIPLIER * worst_legit)


def _irrelevant_decision(delta_prod, delta_ref, gate,
                         radius_railed, phi_railed):
    """Flag only needles: both deficits below the gate and a railed axis."""
    return bool(delta_prod < gate and delta_ref < gate
                and (radius_railed or phi_railed))


def _candidate_score(candidate_result):
    """Best production data log-likelihood over the two phi integrators."""
    return max(record["test_total_log_likelihood"]
               for record in candidate_result["methods"].values())


def _calibrate_gate(legit):
    """Irrelevance gate and best-anchor reference from legit candidates.

    ``legit`` are the fully-evaluated config, Pesce/Reid, and local cloud
    candidate results (broad Sobol points excluded).  Returns None when no
    anchor has a converged reference, which disables flagging for the case.
    """
    anchors = [
        cr for cr in legit
        if cr["source_kind"] in ("config", "pesce-reid")
        and cr["methods"]["fixed-grid"][
            "reference_convergence"]["converged"]]
    if not anchors:
        return None
    best = max(anchors, key=_candidate_score)
    anchor_score = _candidate_score(best)
    legit_scores = [_candidate_score(cr) for cr in legit]
    return {
        "gate": _gate(legit_scores, anchor_score),
        "anchor_score": anchor_score,
        "anchor_reference_total": best["methods"]["fixed-grid"][
            "reference_total_log_likelihood"],
        "worst_legit": min(legit_scores) - anchor_score,
    }


def _relevant_pass(candidates):
    """Case pass over candidates NOT flagged irrelevant.

    Irrelevant needles never count toward pass or fail: a failing irrelevant
    candidate cannot fail the case, and an all-irrelevant set contributes
    nothing.  Rankings apply the same filter separately.
    """
    return all(candidate["passed"] for candidate in candidates
               if not candidate["irrelevant"])


def _railing_diagnostics(case, values):
    """Needle-geometry fractions: railed radius seeds and railed phi peaks.

    Reuses the production model kernels on the same physical path the
    evaluator takes.  Radius railing counts closed-form seeds clipped to
    the support edge; phi railing counts red/blue HV peaks argmaxed at the
    phi=0 boundary where the LOS orbital velocity vanishes.  The systemic
    group is never scanned (phi=0 is a partition seam there, not a peak).
    """
    model = case["model"]
    target = case["target"]
    names = case["names"]
    dtype = (jnp.float64 if case["production_dtype"] == "float64"
             else jnp.float32)
    values = jnp.asarray(values, dtype=dtype)
    theta = target.complete_params(de._flat_to_theta(values, names))
    phys_args, phys_kw = model.phys_from_params_jax(theta, target.h)

    r_est, _, r_min, r_max = model._closed_form_seeds(
        phys_args[2], phys_args[3], phys_args[4], phys_args[16],
        phys_args[8], phys_args[15],
        phys_args[20] if len(phys_args) > 20 else phys_args[16])
    r_est = np.asarray(jax.device_get(r_est))
    r_min, r_max = float(r_min), float(r_max)
    valid = np.asarray(jax.device_get(
        model.is_highvel | model._all_has_accel.astype(bool)))
    railed = ((r_est <= 1.01 * r_min * (1.0 + 1e-6))
              | (r_est >= 0.99 * r_max * (1.0 - 1e-6))) & valid
    n_valid = int(np.count_nonzero(valid))
    radius_fraction = (float(np.count_nonzero(railed) / n_valid)
                       if n_valid else 0.0)

    r_est_j = jnp.asarray(r_est, dtype=dtype)
    n_railed_phi = 0
    n_hv = 0
    for type_key, idx in (("red", model._idx_red), ("blue", model._idx_blue)):
        n = int(idx.shape[0])
        if not n:
            continue
        r_pre = model._r_precompute(
            r_est_j[idx], idx, *phys_args, **phys_kw,
            has_any_accel=model._group_has_any_accel(type_key))
        subs = model._phi_subranges[type_key]
        n_scan = int(model._phi_partition_scan_size(type_key))
        phi_scan = jnp.linspace(
            subs[0][0], subs[-1][1], n_scan, dtype=r_pre["r_ang"].dtype)
        arg = np.asarray(jax.device_get(
            jnp.argmax(model._phi_value(r_pre, phi_scan), axis=-1)))
        n_railed_phi += int(np.count_nonzero(
            (arg <= PHI_EDGE_CELLS) | (arg >= n_scan - 1 - PHI_EDGE_CELLS)))
        n_hv += n
    phi_fraction = float(n_railed_phi / n_hv) if n_hv else 0.0
    return radius_fraction, phi_fraction


def _full_result(case, candidate, production, grids, args, cache_dir,
                 record_railing):
    """Full-reference-ladder candidate result with raw per-method verdict."""
    candidate_result = {
        key: value for key, value in candidate.items() if key != "values"}
    candidate_result["parameters"] = {
        name: float(value)
        for name, value in zip(case["names"], candidate["values"])}
    reference, cache_info = _reference_levels(
        case, candidate, grids, args, cache_dir)
    candidate_result["methods"] = {}
    for method in METHODS:
        method_values = production[method]
        record = _method_record(
            case, method, method_values["conditional_ll"],
            method_values["conditional_overflow"],
            method_values["conditional_roots"], reference,
            grids, cache_info, args)
        record["root_capacity_overflow_node_count"] = int(
            method_values["conditional_overflow_nodes"])
        candidate_result["methods"][method] = record
    if record_railing:
        radius_fraction, phi_fraction = _railing_diagnostics(
            case, candidate["values"])
        candidate_result["railing"] = {
            "radius_railed_fraction": radius_fraction,
            "phi_railed_fraction": phi_fraction}
    candidate_result["peak_minus_fixed_log_likelihood"] = (
        candidate_result["methods"]["peak-partition"][
            "test_total_log_likelihood"]
        - candidate_result["methods"]["fixed-grid"][
            "test_total_log_likelihood"])
    candidate_result["passed"] = all(
        value["passed"] for value in candidate_result["methods"].values())
    return candidate_result


def _classify_irrelevant(case, candidate, candidate_result, calibration):
    """Label a fully-evaluated broad Sobol candidate posterior-irrelevant.

    Sets ``irrelevant`` on ``candidate_result`` (and, when flagged, an
    ``irrelevance`` dict).  Deltas use the finest-level reference total that
    the full ladder already produced -- no separate coarse computation.
    """
    candidate_result["irrelevant"] = False
    if candidate_result["source_kind"] != "sobol" or calibration is None:
        return
    delta_prod = _candidate_score(candidate_result) - (
        calibration["anchor_score"])
    delta_ref = (candidate_result["methods"]["fixed-grid"][
        "reference_total_log_likelihood"]
        - calibration["anchor_reference_total"])
    radius_fraction, phi_fraction = _railing_diagnostics(
        case, candidate["values"])
    if _irrelevant_decision(
            delta_prod, delta_ref, calibration["gate"],
            radius_fraction >= RAILING_FRACTION_THRESHOLD,
            phi_fraction >= RAILING_FRACTION_THRESHOLD):
        candidate_result["irrelevant"] = True
        candidate_result["irrelevance"] = {
            "gate": calibration["gate"],
            "anchor_score": calibration["anchor_score"],
            "anchor_reference_total": calibration["anchor_reference_total"],
            "delta_production": delta_prod,
            "delta_reference": delta_ref,
            "radius_railed_fraction": radius_fraction,
            "phi_railed_fraction": phi_fraction,
        }


def _production_evaluator(case, method):
    model = case["models"][method]
    target = case["targets"][method]
    names = case["names"]
    dtype = (jnp.float64 if case["production_dtype"] == "float64"
             else jnp.float32)

    def evaluate(values):
        values = jnp.asarray(values, dtype=dtype)
        theta = target.complete_params(de._flat_to_theta(values, names))
        phys_args, phys_kw = model.phys_from_params_jax(theta, target.h)
        reuse_scan = (method == "peak-partition"
                      or (method == "fixed-grid" and not model.use_ecc))
        built = model._build_conditional_r_grids(
            phys_args[2], phys_args[3], phys_args[4], phys_args[16],
            phys_args[8], phys_args[15], phys_args, phys_kw,
            return_scan_cache=reuse_scan)
        if reuse_scan:
            groups, caches = built
        else:
            groups, caches = built, [None] * len(built)
        conditional_ll = jnp.zeros(model.n_spots, dtype=dtype)
        conditional_roots = jnp.zeros(model.n_spots, dtype=jnp.int32)
        conditional_overflow = jnp.zeros(model.n_spots, dtype=bool)
        conditional_overflow_nodes = jnp.zeros((), dtype=jnp.int32)

        for group, cache in zip(groups, caches):
            type_key, idx, r_union, log_w_r = group
            ll = model._marginal_per_spot_r(
                type_key, idx, r_union, log_w_r,
                model._group_has_any_accel(type_key), phys_args, phys_kw,
                (None if target.spot_batch is None else
                 min(int(target.spot_batch), int(idx.shape[0]))), cache)
            conditional_ll = conditional_ll.at[idx].set(ll)
            if method == "peak-partition":
                roots, overflow_nodes = _partition_diagnostics(
                    model, type_key, idx, r_union, phys_args, phys_kw,
                    target.spot_batch)
                conditional_roots = conditional_roots.at[idx].set(
                    jnp.max(roots, axis=-1).astype(jnp.int32))
                conditional_overflow = conditional_overflow.at[idx].set(
                    jnp.any(overflow_nodes, axis=-1))
                conditional_overflow_nodes += jnp.sum(
                    overflow_nodes, dtype=jnp.int32)

        return {
            "conditional_ll": conditional_ll,
            "conditional_roots": conditional_roots,
            "conditional_overflow": conditional_overflow,
            "conditional_overflow_nodes": conditional_overflow_nodes,
        }

    return jax.jit(evaluate)


def _evaluate_production(evaluator, values):
    out = evaluator(values)
    out = jax.tree_util.tree_map(jax.block_until_ready, out)
    return jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)), out)


def _device_memory():
    rows = []
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        rows.append({
            "device": str(device),
            "bytes_in_use": stats.get("bytes_in_use"),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        })
    return rows


def _time_production(case, candidates, args, method):
    names, lo, hi = case["names"], case["lo"], case["hi"]
    dtype = (jnp.float64 if case["production_dtype"] == "float64"
             else jnp.float32)
    lo_j = jnp.asarray(lo, dtype=dtype)
    scale_j = jnp.asarray(hi - lo, dtype=dtype)
    model = case["models"][method]
    logp = de._make_logp(case["targets"][method], names)

    def fitness_one(unit):
        values = lo_j + jnp.asarray(unit, dtype=dtype) * scale_j
        return -logp(values)

    devices = tuple(jax.local_devices()[:args.n_devices])
    if len(devices) != args.n_devices:
        raise ValueError(
            f"requested {args.n_devices} devices, found {len(devices)}")
    candidate_wave = de._de_candidates_per_wave(
        model, args.candidate_wave if method == "peak-partition" else None)
    batch_eval = de._make_batched_fitness(
        fitness_one, args.n_devices, devices,
        candidates_per_wave=candidate_wave)
    unit = np.stack([
        (candidate["values"] - lo) / (hi - lo)
        for candidate in candidates]).astype(
            np.float64 if dtype == jnp.float64 else np.float32)

    start = time.perf_counter()
    cold = batch_eval(jnp.asarray(unit))
    jax.block_until_ready(cold)
    cold_seconds = time.perf_counter() - start
    warmed = []
    for _ in range(args.timing_repeats):
        start = time.perf_counter()
        values = batch_eval(jnp.asarray(unit))
        jax.block_until_ready(values)
        warmed.append(time.perf_counter() - start)
    steady_seconds = float(np.median(warmed))
    return {
        "cold_compile_and_evaluate_seconds": cold_seconds,
        "estimated_compile_seconds": max(0.0, cold_seconds - steady_seconds),
        "steady_evaluation_seconds": steady_seconds,
        "steady_repeat_seconds": warmed,
        "candidate_count": len(candidates),
        "candidate_wave": candidate_wave,
        "throughput_candidates_per_second": (
            len(candidates) / steady_seconds),
        "device_profile": batch_eval.device_profile(),
        "memory": _device_memory(),
    }


def _timing_comparison(timing):
    """Head-to-head speedup of peak-partition over fixed-grid."""
    fixed = timing["fixed-grid"]
    peak = timing["peak-partition"]
    return {
        "steady_speedup_peak_over_fixed": (
            fixed["steady_evaluation_seconds"]
            / peak["steady_evaluation_seconds"]),
        "cold_speedup_peak_over_fixed": (
            fixed["cold_compile_and_evaluate_seconds"]
            / peak["cold_compile_and_evaluate_seconds"]),
        "throughput_ratio_peak_over_fixed": (
            peak["throughput_candidates_per_second"]
            / fixed["throughput_candidates_per_second"]),
    }


def _time_candidates(case, candidates, args):
    """Per-candidate seconds for the pure production objective, per method.

    Times ``de._make_logp`` only -- NOT ``_production_evaluator``, whose
    peak-partition path also runs root-counting diagnostics that would
    unfairly inflate the peak column.  The pure logp is the number relevant
    to DE throughput.  One warm evaluation compiles; each candidate is then
    the median of ``timing_repeats`` evaluations.  Keyed by candidate id.
    """
    dtype = (jnp.float64 if case["production_dtype"] == "float64"
             else jnp.float32)
    seconds = {candidate["id"]: {} for candidate in candidates}
    for method in METHODS:
        logp = jax.jit(de._make_logp(case["targets"][method], case["names"]))
        values = [jnp.asarray(candidate["values"], dtype=dtype)
                  for candidate in candidates]
        jax.block_until_ready(logp(values[0]))
        for candidate, value in zip(candidates, values):
            reps = []
            for _ in range(args.timing_repeats):
                start = time.perf_counter()
                jax.block_until_ready(logp(value))
                reps.append(time.perf_counter() - start)
            seconds[candidate["id"]][method] = float(np.median(reps))
    return seconds


def _reference_metadata(case, candidate, grids, args):
    return {
        "schema": REFERENCE_CACHE_SCHEMA_VERSION,
        "policy": REFERENCE_POLICY,
        "source_hash": case["source_hash"],
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "backend": jax.default_backend(),
        "device_kinds": sorted({
            device.device_kind for device in jax.devices()}),
        "config_hash": case["config_hash"],
        "galaxy": case["galaxy"],
        "variant": case["variant"],
        "reference_grids": [
            {"n_r": n_r, "n_phi": n_phi} for n_r, n_phi in grids],
        "r_chunk": args.reference_r_chunk,
        "spot_batch": args.reference_spot_batch,
        "dtype": "float64",
        "production_dtype": case["production_dtype"],
        "candidate_names": list(case["names"]),
        "candidate_values": candidate["values"].tolist(),
    }


def _load_reference_cache(cache_dir, metadata):
    key = reference_cache_key(metadata)
    path = cache_dir / f"{key}.npz"
    if not path.is_file():
        return None, path
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return None, path
    if age >= REFERENCE_CACHE_TTL_SECONDS:
        path.unlink(missing_ok=True)
        return None, path
    with np.load(path, allow_pickle=False) as cached:
        saved = str(np.asarray(cached["metadata"]).item())
        expected = _canonical_json(metadata)
        if saved != expected:
            raise RuntimeError(f"reference cache metadata mismatch: {path}")
        values = np.asarray(cached["values"], dtype=np.float64)
    return values, path


def _save_reference_cache(path, metadata, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, metadata=np.asarray(_canonical_json(metadata)),
        values=np.asarray(values, dtype=np.float64))
    os.replace(tmp, path)


def _clean_reference_cache(cache_dir):
    if not cache_dir.is_dir():
        return 0
    paths = list(cache_dir.glob("*.npz"))
    for path in paths:
        path.unlink(missing_ok=True)
    return len(paths)


def _reference_cache_dir(args):
    return args.cache_dir or Path(de.results_path(
        de._MASTER_CFG["io"].get("root_output", "results/Megamaser"),
        "convergence", "reference_cache"))


def _reference_levels(case, candidate, grids, args, cache_dir):
    metadata = _reference_metadata(case, candidate, grids, args)
    cache_path = None
    if not args.no_cache:
        cached, cache_path = _load_reference_cache(cache_dir, metadata)
        if cached is not None:
            expected_shape = (len(grids), case["model"].n_spots)
            if cached.shape != expected_shape:
                raise RuntimeError(
                    f"reference cache shape {cached.shape} does not match "
                    f"{expected_shape}: {cache_path}")
            age_hours = max(
                0.0, (time.time() - cache_path.stat().st_mtime) / 3600.0)
            print(
                f"  Reference cache HIT {cache_path.stem[:12]} "
                f"(age {age_hours:.2f} h): dense ladder skipped.",
                flush=True)
            return cached, {
                "cache_hit": True, "cache_key": cache_path.stem,
                "cache_path": str(cache_path),
                "cache_age_hours": age_hours, "level_seconds": [],
            }

    cache_key = reference_cache_key(metadata)
    if args.no_cache:
        print("  Reference cache disabled: computing dense ladder.",
              flush=True)
    else:
        print(f"  Reference cache MISS {cache_key[:12]}: computing dense "
              "ladder.", flush=True)

    values = jnp.asarray(candidate["values"], dtype=jnp.float64)
    theta = case["reference_target"].complete_params(
        de._flat_to_theta(values, case["names"]))
    phys_args, phys_kw = case["reference_model"].phys_from_params_jax(
        theta, case["reference_target"].h)
    reference = []
    level_seconds = []
    for n_r, n_phi in grids:
        print(f"    Dense reference {n_r}x{n_phi}...", end="", flush=True)
        start = time.perf_counter()
        reference.append(dense_r_phi_reference_per_spot(
            case["reference_model"], phys_args, phys_kw,
            n_r, n_phi, args.reference_r_chunk,
            args.reference_spot_batch, partition_support=True))
        seconds = time.perf_counter() - start
        level_seconds.append(seconds)
        print(f" {seconds:.2f} s", flush=True)
    reference = np.stack(reference)
    if not args.no_cache:
        if cache_path is None:
            cache_path = cache_dir / (cache_key + ".npz")
        _save_reference_cache(cache_path, metadata, reference)
    return reference, {
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_path": None if args.no_cache else str(cache_path),
        "cache_age_hours": None,
        "level_seconds": level_seconds,
    }


def _population_labels(model):
    labels = np.empty(model.n_spots, dtype=object)
    labels[np.asarray(model._idx_sys)] = "systemic"
    labels[np.asarray(model._idx_red)] = "red"
    labels[np.asarray(model._idx_blue)] = "blue"
    return labels


def _method_record(case, method, test, overflow, roots, reference,
                   grids, cache_info, args):
    criteria = {
        "total_atol": args.reference_total_atol,
        "spot_atol": args.reference_spot_atol,
        "rms_atol": args.reference_rms_atol,
    }
    convergence = reference_convergence(
        reference, args.reference_tail_levels, criteria)
    labels = _population_labels(case["model"])
    for previous_grid, current_grid, previous, current, row in zip(
            grids, grids[1:], reference, reference[1:],
            convergence["comparisons"]):
        row["previous_n_r"], row["previous_n_phi"] = previous_grid
        row["current_n_r"], row["current_n_phi"] = current_grid
        row["previous_total_log_likelihood"] = float(np.sum(previous))
        row["current_total_log_likelihood"] = float(np.sum(current))
        row["spots"] = []
        for index, (old, new) in enumerate(zip(previous, current)):
            signed = (float(new - old)
                      if np.isfinite(old) and np.isfinite(new) else np.nan)
            row["spots"].append({
                "spot": index,
                "population": labels[index],
                "previous_log_likelihood": float(old),
                "current_log_likelihood": float(new),
                "signed_change": signed,
                "absolute_change": abs(signed),
                "finite_mask_match": bool(
                    np.isfinite(old) == np.isfinite(new)),
            })
    finest = reference[-1]
    comparison = error_statistics(test, finest)
    by_population = {}
    for population in POPULATIONS:
        mask = labels == population
        by_population[population] = error_statistics(
            test[mask], finest[mask])
        worst_spot = by_population[population]["worst_spot"]
        if worst_spot is not None:
            by_population[population]["worst_spot"] = int(
                np.flatnonzero(mask)[worst_spot])

    comparison_pass = bool(
        comparison["finite_mask_mismatches"] == 0
        and comparison["all_spots_finite"]
        and comparison["absolute_total_error"] <= args.total_atol
        and comparison["max_absolute_spot_error"] <= args.spot_atol
        and comparison["p99_absolute_spot_error"] <= args.spot_p99_atol
        and comparison["rms_spot_error"] <= args.spot_rms_atol)
    overflow_count = int(np.count_nonzero(overflow))
    passed = bool(
        convergence["converged"] and comparison_pass
        and overflow_count <= args.max_root_overflows)
    spots = []
    observed_x = np.asarray(case["model"]._all_x)
    observed_y = np.asarray(case["model"]._all_y)
    observed_v = np.asarray(case["model"]._all_v)
    observed_a = np.asarray(case["model"]._all_a)
    has_acceleration = np.asarray(case["model"]._all_has_accel, dtype=bool)
    for index in range(case["model"].n_spots):
        test_value = float(test[index])
        reference_value = float(finest[index])
        signed_error = (test_value - reference_value
                        if math.isfinite(test_value)
                        and math.isfinite(reference_value) else np.nan)
        spots.append({
            "spot": index,
            "population": labels[index],
            "observed_x": float(observed_x[index]),
            "observed_y": float(observed_y[index]),
            "observed_velocity": float(observed_v[index]),
            "observed_acceleration": float(observed_a[index]),
            "has_measured_acceleration": bool(has_acceleration[index]),
            "test_log_likelihood": test_value,
            "reference_log_likelihood": reference_value,
            "signed_error": signed_error,
            "absolute_error": abs(signed_error),
            "test_finite": math.isfinite(test_value),
            "reference_finite": math.isfinite(reference_value),
            "root_count": int(roots[index]),
            "root_capacity_overflow": bool(overflow[index]),
        })
    return {
        "method": method,
        "reference_levels": [
            {"n_r": grid[0], "n_phi": grid[1],
             "total_log_likelihood": float(np.sum(values)),
             "finite_spots": int(np.count_nonzero(np.isfinite(values)))}
            for grid, values in zip(grids, reference)],
        "reference_convergence": convergence,
        "reference_cache": cache_info,
        "test_total_log_likelihood": float(np.sum(test)),
        "reference_total_log_likelihood": float(np.sum(finest)),
        "comparison": comparison,
        "errors_by_population": by_population,
        "root_capacity_overflow_count": overflow_count,
        "comparison_pass": comparison_pass,
        "passed": passed,
        "spots": spots,
        "_test_total": float(np.sum(test)),
        "_reference_totals": [float(np.sum(values)) for values in reference],
    }


def _add_pesce_deltas(case_result):
    """Record tested log-likelihood differences relative to Pesce/Reid."""
    pesce = next((candidate for candidate in case_result["candidates"]
                  if candidate["source_kind"] == "pesce-reid"), None)
    for candidate in case_result["candidates"]:
        for method, result in candidate["methods"].items():
            if pesce is None or method not in pesce["methods"]:
                delta = None
            else:
                delta = (
                    result["test_total_log_likelihood"]
                    - pesce["methods"][method][
                        "test_total_log_likelihood"])
            result["delta_log_likelihood_vs_pesce"] = delta


def _case_rankings(case_result, args):
    scored = [candidate for candidate in case_result["candidates"]
              if not candidate.get("irrelevant")]
    labels = [candidate["id"] for candidate in scored]
    rankings = {}
    grids = _reference_grids(case_result["galaxy"], args)
    for method in METHODS:
        records = [candidate["methods"][method] for candidate in scored]
        level_totals = np.asarray([
            record["_reference_totals"] for record in records]).T
        consecutive = []
        for index in range(len(level_totals) - 1):
            inversions = ranking_inversions(
                level_totals[index], level_totals[index + 1], labels,
                args.ranking_atol)
            consecutive.append({
                "previous_level_index": index,
                "current_level_index": index + 1,
                "previous_n_r": int(grids[index][0]),
                "previous_n_phi": int(grids[index][1]),
                "current_n_r": int(grids[index + 1][0]),
                "current_n_phi": int(grids[index + 1][1]),
                "count": len(inversions),
                "inversions": inversions,
            })
        test_totals = [record["_test_total"] for record in records]
        test_inversions = ranking_inversions(
            level_totals[-1], test_totals, labels, args.ranking_atol)
        tail = consecutive[-(args.reference_tail_levels - 1):]
        reference_pass = all(
            row["count"] <= args.max_ranking_inversions for row in tail)
        test_pass = len(test_inversions) <= args.max_ranking_inversions
        rankings[method] = {
            "reference_consecutive": consecutive,
            "reference_tail_inversion_count": sum(
                row["count"] for row in tail),
            "reference_pass": reference_pass,
            "test_vs_reference": {
                "count": len(test_inversions),
                "inversions": test_inversions,
                "passed": test_pass,
            },
        }
        if not reference_pass or not test_pass:
            case_result["passed"] = False
    case_result["rankings"] = rankings


def _case_worst(case_result, limit=10):
    scored = [candidate for candidate in case_result["candidates"]
              if not candidate.get("irrelevant")]
    worst = {}
    for method in scored[0]["methods"]:
        candidates = []
        spots = []
        for candidate in scored:
            record = candidate["methods"][method]
            comparison = record["comparison"]
            candidates.append((
                comparison["finite_mask_mismatches"],
                comparison["max_absolute_spot_error"],
                candidate["id"], candidate["source_kind"]))
            for spot in record["spots"]:
                spots.append({"candidate": candidate["id"], **spot})
        candidates.sort(
            key=lambda row: (
                row[0], -np.inf if not np.isfinite(row[1]) else row[1]),
            reverse=True)
        spots.sort(
            key=lambda row: (
                not (row["test_finite"] and row["reference_finite"]),
                (-np.inf if not np.isfinite(row["absolute_error"])
                 else row["absolute_error"])),
            reverse=True)
        mismatches, value, candidate_id, source = candidates[0]
        worst[method] = {
            "candidate": candidate_id,
            "point_source": source,
            "finite_mask_mismatches": mismatches,
            "max_absolute_spot_error": value,
            "spots": spots[:limit],
        }
    case_result["worst"] = worst


def _strip_internal(report):
    if isinstance(report, dict):
        return {key: _strip_internal(value)
                for key, value in report.items()
                if not key.startswith("_")}
    if isinstance(report, list):
        return [_strip_internal(value) for value in report]
    return report


def _aggregate(report):
    grouped = defaultdict(list)
    for case in report["cases"]:
        for candidate in case["candidates"]:
            for method, result in candidate["methods"].items():
                key = (case["galaxy"], case["variant"],
                       candidate["source_kind"], method)
                grouped[key].append(
                    (bool(candidate.get("irrelevant")), result))
    rows = []
    for key, values in sorted(grouped.items()):
        scored = [result for irrelevant, result in values if not irrelevant]
        flagged = [result for irrelevant, result in values if irrelevant]
        comparisons = [value["comparison"] for value in scored]

        def finite_max(field):
            entries = [row[field] for row in comparisons
                       if np.isfinite(row[field])]
            return max(entries) if entries else np.nan

        rows.append({
            "galaxy": key[0],
            "variant": key[1],
            "point_source": key[2],
            "method": key[3],
            "candidate_count": len(values),
            "irrelevant": len(flagged),
            "passed": (all(value["passed"] for value in scored)
                       if scored else None),
            "reference_unconverged": sum(
                not value["reference_convergence"]["converged"]
                for value in scored),
            "unconverged_excused": sum(
                not value["reference_convergence"]["converged"]
                for value in flagged),
            "finite_mask_mismatches": sum(
                row["finite_mask_mismatches"] for row in comparisons),
            "root_capacity_overflows": sum(
                value["root_capacity_overflow_count"] for value in scored),
            "worst_absolute_total_error": finite_max(
                "absolute_total_error"),
            "worst_absolute_spot_error": finite_max(
                "max_absolute_spot_error"),
            "worst_p99_spot_error": finite_max(
                "p99_absolute_spot_error"),
            "worst_rms_spot_error": finite_max("rms_spot_error"),
        })
    return rows


def _fmt(value, precision=".3g"):
    if value is None or not math.isfinite(float(value)):
        return "non-finite"
    return format(float(value), precision)


def _paired(values, precision=".3g", separator=" | "):
    def format_value(value):
        if value is None:
            return "n/a"
        return str(value) if precision is None else _fmt(value, precision)

    return separator.join(format_value(value) for value in values)


def _first_relevant(case):
    """First non-irrelevant candidate; anchors are ordered first and kept."""
    return next(candidate for candidate in case["candidates"]
                if not candidate.get("irrelevant"))


def _count_irrelevant(report):
    return sum(1 for case in report["cases"]
               for candidate in case["candidates"]
               if candidate.get("irrelevant"))


def _conclusion(report):
    verdict = "PASS" if report["passed"] else "FAIL"
    n_irrelevant = _count_irrelevant(report)
    if n_irrelevant:
        verdict += (f" ({n_irrelevant} irrelevant: posterior-irrelevant "
                    "geometry, excused)")
    return verdict


def _candidate_table_row(galaxy, variant, candidate, separator):
    """One candidate row with real numbers; irrelevant rows carry a marker."""
    source = candidate["source_kind"]
    cid = candidate["id"] + (
        " [IRRELEVANT]" if candidate.get("irrelevant") else "")
    results = [candidate["methods"][method] for method in METHODS]
    comparisons = [result["comparison"] for result in results]
    seconds = candidate.get("production_seconds") or {}
    absolute_errors = [r["absolute_total_error"] for r in comparisons]
    spot_errors = [r["max_absolute_spot_error"] for r in comparisons]
    timings = [seconds.get(method) for method in METHODS]
    return (
        f"| {galaxy} | {variant} | {source} | {cid} | "
        f"{_fmt(results[0]['reference_total_log_likelihood'], '.6g')} | "
        f"{_paired(absolute_errors, separator=separator)} | "
        f"{_paired(spot_errors, separator=separator)} | "
        f"{_paired(timings, '.3g', separator)} |")


def _markdown(report):
    lines = [
        "# Megamaser fixed-grid versus peak-partition validation", "",
        f"**Conclusion: {_conclusion(report)}**", "",
        "## Candidate results", "",
        "Errors are absolute log-likelihood differences. Paired values are "
        "`fixed-grid | peak-partition` phi integration. Every candidate is "
        "judged against the full dense reference ladder; rows marked "
        "`[IRRELEVANT]` are "
        "flagged posterior-irrelevant geometry, fully evaluated but excused "
        "from the verdict. `eval s` is the per-candidate pure production "
        "objective (one warm evaluation, median of `timing_repeats`); the "
        "timing section reports the batched throughput. Exact fresh cache "
        "hits load the complete dense ladder instead of recomputing it.",
        "",
        "| Galaxy | Variant | Source | Candidate | Finest reference logL | "
        "abs total error (fixed \\| peak) | "
        "worst spot error (fixed \\| peak) | eval s (fixed \\| peak) |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        for candidate in case["candidates"]:
            lines.append(_candidate_table_row(
                case["galaxy"], case["variant"], candidate, " \\| "))

    flagged_rows = []
    contrast_rows = []
    for case in report["cases"]:
        for candidate in case["candidates"]:
            if candidate.get("irrelevant"):
                flag = candidate["irrelevance"]
                flagged_rows.append(
                    f"- {case['galaxy']}/{case['variant']}/"
                    f"{candidate['id']}: "
                    f"delta_production={_fmt(flag['delta_production'])}, "
                    f"delta_reference={_fmt(flag['delta_reference'])}, "
                    f"radius railed={_fmt(flag['radius_railed_fraction'])}, "
                    f"phi railed={_fmt(flag['phi_railed_fraction'])}")
            elif candidate.get("railing"):
                rail = candidate["railing"]
                contrast_rows.append(
                    f"- {case['galaxy']}/{case['variant']}/"
                    f"{candidate['id']} ({candidate['source_kind']}): "
                    f"radius railed={_fmt(rail['radius_railed_fraction'])}, "
                    f"phi railed={_fmt(rail['phi_railed_fraction'])}")
    if flagged_rows:
        lines.extend(["", "**Flagged posterior-irrelevant:**", ""])
        if contrast_rows:
            lines.append("Anchor geometry for contrast (never flagged):")
            lines.extend(contrast_rows)
            lines.append("")
        lines.append("Flagged candidates:")
        lines.extend(flagged_rows)

    unconverged = []
    excused = []
    overflows = []
    for case in report["cases"]:
        for candidate in case["candidates"]:
            label = f"{case['galaxy']}/{case['variant']}/{candidate['id']}"
            if not candidate["methods"]["fixed-grid"][
                    "reference_convergence"]["converged"]:
                (excused if candidate.get("irrelevant")
                 else unconverged).append(label)
            if candidate.get("irrelevant"):
                continue
            for method, result in candidate["methods"].items():
                spots = result["root_capacity_overflow_count"]
                nodes = result.get(
                    "root_capacity_overflow_node_count", spots)
                if spots or nodes:
                    overflows.append(
                        f"{label} ({method} phi integration): "
                        f"{spots} spots, {nodes} radial nodes")
    if unconverged:
        lines.extend([
            "", "**Reference not converged (FAIL):** "
            + ", ".join(unconverged)])
    if excused:
        lines.extend([
            "", "**Reference not converged (expected for needle geometry, "
            "excused):** " + ", ".join(excused)])
    if overflows:
        lines.extend(["", "**Root-capacity overflows:**", ""])
        lines.extend(f"- {row}" for row in overflows)

    lines.extend([
        "",
        "## Aggregate by galaxy, variant, point source, and phi integration",
        "",
        "| Galaxy | Variant | Source | Phi integration | N | Irrelevant | "
        "Unconverged refs | Unconverged excused | "
        "mask mismatch | overflows | worst abs total | worst spot | "
        "worst p99 | worst RMS | Pass |",
        ("|---|---|---|---|---:|---:|---:|---:|"
         "---:|---:|---:|---:|---:|---:|---:|"),
    ])
    for row in report["aggregate"]:
        lines.append(
            f"| {row['galaxy']} | {row['variant']} | "
            f"{row['point_source']} | {row['method']} | "
            f"{row['candidate_count']} | {row['irrelevant']} | "
            f"{row['reference_unconverged']} | "
            f"{row['unconverged_excused']} | "
            f"{row['finite_mask_mismatches']} | "
            f"{row['root_capacity_overflows']} | "
            f"{_fmt(row['worst_absolute_total_error'])} | "
            f"{_fmt(row['worst_absolute_spot_error'])} | "
            f"{_fmt(row['worst_p99_spot_error'])} | "
            f"{_fmt(row['worst_rms_spot_error'])} | {row['passed']} |")

    lines.extend([
        "", "## Reference convergence", "",
        "| Galaxy | Variant | Candidate | n_r x n_phi | "
        "next n_r x n_phi | "
        "abs total change | max spot | median | p95 | p99 | RMS | "
        "mask mismatch | Pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in report["cases"]:
        for candidate in case["candidates"]:
            result = candidate["methods"]["fixed-grid"]
            for row in result["reference_convergence"]["comparisons"]:
                lines.append(
                    f"| {case['galaxy']} | {case['variant']} | "
                    f"{candidate['id']} | "
                    f"{row['previous_n_r']} x {row['previous_n_phi']} | "
                    f"{row['current_n_r']} x {row['current_n_phi']} | "
                    f"{_fmt(row['absolute_total_error'])} | "
                    f"{_fmt(row['max_absolute_spot_error'])} | "
                    f"{_fmt(row['median_absolute_spot_error'])} | "
                    f"{_fmt(row['p95_absolute_spot_error'])} | "
                    f"{_fmt(row['p99_absolute_spot_error'])} | "
                    f"{_fmt(row['rms_spot_error'])} | "
                    f"{row['finite_mask_mismatches']} | "
                    f"{row['passed']} |")

    lines.extend([
        "", "## Errors by spot population", "",
        "| Galaxy | Variant | Candidate | Phi integration | Population | "
        "signed total | max spot | median | p95 | p99 | RMS | "
        "mask mismatch |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in report["cases"]:
        for candidate in case["candidates"]:
            for method, result in candidate["methods"].items():
                for population, row in result["errors_by_population"].items():
                    lines.append(
                        f"| {case['galaxy']} | {case['variant']} | "
                        f"{candidate['id']} | {method} | {population} | "
                        f"{_fmt(row['signed_total_error'])} | "
                        f"{_fmt(row['max_absolute_spot_error'])} | "
                        f"{_fmt(row['median_absolute_spot_error'])} | "
                        f"{_fmt(row['p95_absolute_spot_error'])} | "
                        f"{_fmt(row['p99_absolute_spot_error'])} | "
                        f"{_fmt(row['rms_spot_error'])} | "
                        f"{row['finite_mask_mismatches']} |")

    lines.extend([
        "", "## Rankings and worst spots", "",
        "Rankings and worst spots exclude irrelevant candidates.", "",
        "| Galaxy | Variant | Phi integration | "
        "Reference ranking inversions | test/reference inversions | "
        "Worst candidate | Worst spot error |",
        "|---|---|---|---:|---:|---|---:|",
    ])
    for case in report["cases"]:
        for method, ranking in case["rankings"].items():
            reference_count = ranking["reference_tail_inversion_count"]
            worst = case["worst"][method]
            lines.append(
                f"| {case['galaxy']} | {case['variant']} | {method} | "
                f"{reference_count} | "
                f"{ranking['test_vs_reference']['count']} | "
                f"{worst['candidate']} | "
                f"{_fmt(worst['max_absolute_spot_error'])} |")
    lines.extend([
        "",
        "| Galaxy | Variant | Phi integration | Candidate | "
        "Spot | Population | "
        "signed error | absolute error | roots | overflow |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for case in report["cases"]:
        for method, worst in case["worst"].items():
            for spot in worst["spots"]:
                lines.append(
                    f"| {case['galaxy']} | {case['variant']} | {method} | "
                    f"{spot['candidate']} | {spot['spot']} | "
                    f"{spot['population']} | {_fmt(spot['signed_error'])} | "
                    f"{_fmt(spot['absolute_error'])} | "
                    f"{spot['root_count']} | "
                    f"{spot['root_capacity_overflow']} |")

    lines.extend(["", "## Timing and memory", ""])
    for case in report["cases"]:
        settings = case["numerical_settings"]
        fixed_settings = settings["fixed-grid"]
        peak_settings = settings["peak-partition"]
        reference_grids = " -> ".join(
            f"{level['n_r']} x {level['n_phi']}"
            for level in _first_relevant(case)["methods"][
                "fixed-grid"]["reference_levels"])
        lines.extend([
            f"### {case['galaxy']} / {case['variant']}", "",
            f"- Production dtype: `{case['production_dtype']}`",
            f"- Configured phi integration: "
            f"`{case['configured_phi_integration']}`",
            "- Radius treatment: `conditional-r` only",
            "- Validation methods: `fixed-grid`, `peak-partition`",
            "- Explicit scheme-setting overrides: "
            f"`{_canonical_json(case['scheme_setting_overrides'])}`",
            "- Fixed-grid phi nodes: "
            f"`{fixed_settings['n_phi_sys']}` per systemic sub-range; "
            f"`{fixed_settings['n_phi_hv_high']}` in the high-velocity core "
            f"and `{fixed_settings['n_phi_hv_low']}` per outer wing",
            "- Peak-partition scan phi nodes: "
            f"`{peak_settings['n_phi_partition_sys']}` systemic and "
            f"`{peak_settings['n_phi_partition_hv']}` high velocity",
            "- Conditional-r nodes: fixed-grid "
            f"`{fixed_settings['n_r_local']}` local + "
            f"`{fixed_settings['n_r_global']}` global; peak-partition "
            f"`{peak_settings['n_r_local']}` local + "
            f"`{peak_settings['n_r_global']}` global",
            f"- Dense full-support r x phi grids: `{reference_grids}`",
            f"- Spot batch: `{case['spot_batch']}` "
            f"({case['spot_batch_source']})",
        ])
        for method, timing in case["timing"].items():
            lines.extend([
                f"- `{method}` cold compile + evaluation: "
                f"{timing['cold_compile_and_evaluate_seconds']:.3f} s",
                f"- `{method}` estimated compile component: "
                f"{timing['estimated_compile_seconds']:.3f} s",
                f"- `{method}` warm steady evaluation: "
                f"{timing['steady_evaluation_seconds']:.3f} s",
                f"- `{method}` throughput: "
                f"{timing['throughput_candidates_per_second']:.3f} "
                "candidates/s",
                f"- `{method}` backend memory stats: "
                f"`{_canonical_json(timing['memory'])}`",
                f"- `{method}` evaluator profile: "
                f"`{_canonical_json(timing['device_profile'])}`",
            ])
        comparison = case["timing_comparison"]
        lines.extend([
            "- Peak-partition vs fixed-grid: steady speedup "
            f"{comparison['steady_speedup_peak_over_fixed']:.2f}x, "
            f"cold {comparison['cold_speedup_peak_over_fixed']:.2f}x",
            "- Backend memory after validation: "
            f"`{_canonical_json(case['memory_after_validation'])}`", ""])

    reproduction = {
        "run": report["metadata"],
        "cases": [{
            "galaxy": case["galaxy"],
            "variant": case["variant"],
            "production_dtype": case["production_dtype"],
            "configured_phi_integration": case[
                "configured_phi_integration"],
            "validated_phi_integrations": case[
                "validated_phi_integrations"],
            "spot_batch": case["spot_batch"],
            "spot_batch_source": case["spot_batch_source"],
            "objective_policies": case["objective_policies"],
            "numerical_settings": case["numerical_settings"],
            "scheme_setting_overrides": case[
                "scheme_setting_overrides"],
            "model_configuration": case["model_configuration"],
            "parameter_names": case["parameter_names"],
            "de_bounds": case["de_bounds"],
        } for case in report["cases"]],
    }
    lines.extend(["## Reproduction metadata", "", "```json",
                  json.dumps(_json_ready(reproduction), indent=2,
                             sort_keys=True, allow_nan=False),
                  "```", ""])
    return "\n".join(lines)


def _print_reference_convergence(case):
    """Console consecutive dense-grid convergence ladder (method-independent).

    The fixed-grid record is canonical: the reference is the same for both
    integrators, so ``_markdown`` reads it too.
    """
    print("\nReference convergence (consecutive dense-grid levels):",
          flush=True)
    header = (f"{'candidate':<20} {'grid pair':<26} {'ref logL':>14} "
              f"{'abs total':>12} {'worst spot':>12} {'RMS':>11} {'pass':>6}")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for candidate in case["candidates"]:
        cid = candidate["id"] + (
            " [IRR]" if candidate.get("irrelevant") else "")
        for row in candidate["methods"]["fixed-grid"][
                "reference_convergence"]["comparisons"]:
            pair = (f"{row['previous_n_r']}x{row['previous_n_phi']} -> "
                    f"{row['current_n_r']}x{row['current_n_phi']}")
            print(
                f"{cid:<20} {pair:<26} "
                f"{_fmt(row['current_total_log_likelihood'], '.6g'):>14} "
                f"{_fmt(row['absolute_total_error']):>12} "
                f"{_fmt(row['max_absolute_spot_error']):>12} "
                f"{_fmt(row['rms_spot_error']):>11} "
                f"{str(row['passed']):>6}", flush=True)


def _print_case(case):
    print("\n" + "=" * 88, flush=True)
    print(f"{case['galaxy']} / {case['variant']} / "
          f"{case['production_dtype']}", flush=True)
    print("=" * 88, flush=True)
    print(
        "Phi integration: configured="
        f"{case['configured_phi_integration']}; comparing="
        "fixed-grid vs peak-partition", flush=True)
    print("Radius treatment: conditional-r only", flush=True)
    print("Explicit scheme-setting overrides: "
          f"{_canonical_json(case['scheme_setting_overrides'])}",
          flush=True)
    settings = case["numerical_settings"]
    fixed_settings = settings["fixed-grid"]
    peak_settings = settings["peak-partition"]
    print(
        "Fixed-grid phi nodes: "
        f"{fixed_settings['n_phi_sys']} per systemic sub-range; "
        f"{fixed_settings['n_phi_hv_high']} high-velocity core + "
        f"{fixed_settings['n_phi_hv_low']} per outer wing", flush=True)
    print(
        "Peak-partition scan phi nodes: "
        f"{peak_settings['n_phi_partition_sys']} (systemic), "
        f"{peak_settings['n_phi_partition_hv']} (high velocity)", flush=True)
    print(
        "Conditional-r grids: fixed-grid="
        f"{fixed_settings['n_r_local']}+{fixed_settings['n_r_global']}; "
        "peak-partition="
        f"{peak_settings['n_r_local']}+{peak_settings['n_r_global']} "
        "(local+global radial nodes)", flush=True)
    reference_grids = " -> ".join(
        f"{level['n_r']}x{level['n_phi']}"
        for level in _first_relevant(case)["methods"][
            "fixed-grid"]["reference_levels"])
    print("Dense full-support r x phi reference grids: "
          f"{reference_grids}", flush=True)
    calibration = case["gate_calibration"]
    if calibration["enabled"]:
        print(
            "Irrelevance gate: enabled; "
            f"gate={_fmt(calibration['gate'], '.6g')}, "
            f"anchor score={_fmt(calibration['anchor_score'], '.6g')}, "
            f"worst legit deficit={_fmt(calibration['worst_legit'])}",
            flush=True)
    else:
        print(f"Irrelevance gate: disabled ({calibration['reason']})",
              flush=True)
    print(
        "Paired values: fixed-grid | peak-partition phi integration",
        flush=True)
    print("Errors are absolute log-likelihood differences", flush=True)
    for message in case["candidate_status"]:
        print(f"NOTE: {message}", flush=True)
    for method, timing in case["timing"].items():
        print(f"Timing {method}: "
              f"cold={timing['cold_compile_and_evaluate_seconds']:.3f}s, "
              f"steady={timing['steady_evaluation_seconds']:.3f}s, "
              f"throughput={timing['throughput_candidates_per_second']:.3f} "
              "candidate/s", flush=True)
    comparison = case["timing_comparison"]
    print("Timing peak-partition vs fixed-grid: steady speedup "
          f"{comparison['steady_speedup_peak_over_fixed']:.2f}x, "
          f"cold {comparison['cold_speedup_peak_over_fixed']:.2f}x",
          flush=True)
    print("Per-candidate 'eval s' = pure production objective (one warm "
          "evaluation, median repeats); case timing above is batched "
          "throughput", flush=True)
    print("", flush=True)
    print("", flush=True)
    header = (f"{'candidate':<20} {'reference logL':>16} "
              f"{'abs total error fixed | peak':>30} "
              f"{'worst spot error fixed | peak':>31} "
              f"{'eval s fixed | peak':>22}")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for candidate in case["candidates"]:
        marker = " [IRR]" if candidate.get("irrelevant") else ""
        results = [candidate["methods"][method] for method in METHODS]
        comparisons = [result["comparison"] for result in results]
        seconds = candidate.get("production_seconds") or {}
        absolute_errors = _paired(
            [r["absolute_total_error"] for r in comparisons])
        spot_errors = _paired(
            [r["max_absolute_spot_error"] for r in comparisons])
        timings = _paired(
            [seconds.get(method) for method in METHODS], ".3g")
        print(
            f"{candidate['id'] + marker:<20} "
            f"{_fmt(results[0]['reference_total_log_likelihood'], '.6g'):>16} "
            f"{absolute_errors:>30} "
            f"{spot_errors:>31} "
            f"{timings:>22}",
            flush=True)
    _print_reference_convergence(case)
    flagged = [candidate for candidate in case["candidates"]
               if candidate.get("irrelevant")]
    if flagged:
        anchors = [candidate for candidate in case["candidates"]
                   if candidate.get("railing") is not None]
        if anchors:
            contrast = ", ".join(
                f"{candidate['id']} "
                f"{_fmt(candidate['railing']['radius_railed_fraction'])}/"
                f"{_fmt(candidate['railing']['phi_railed_fraction'])}"
                for candidate in anchors)
            print(f"Anchor railing contrast (radius/phi): {contrast}",
                  flush=True)
        print("\nFlagged posterior-irrelevant (fully evaluated, excused):",
              flush=True)
        for candidate in flagged:
            flag = candidate["irrelevance"]
            print(
                f"  {candidate['id']}: "
                f"delta_production={_fmt(flag['delta_production'])}, "
                f"delta_reference={_fmt(flag['delta_reference'])}, "
                f"radius railed={_fmt(flag['radius_railed_fraction'])}, "
                f"phi railed={_fmt(flag['phi_railed_fraction'])}", flush=True)
    unconverged = []
    excused = []
    for candidate in case["candidates"]:
        if candidate["methods"]["fixed-grid"][
                "reference_convergence"]["converged"]:
            continue
        (excused if candidate.get("irrelevant") else unconverged).append(
            candidate["id"])
    if unconverged:
        print("\nWARNING: reference not converged (FAIL) for "
              + ", ".join(unconverged), flush=True)
    if excused:
        print("Reference not converged (expected for needle geometry, "
              "excused) for " + ", ".join(excused), flush=True)
    overflow_rows = []
    for candidate in case["candidates"]:
        if candidate.get("irrelevant"):
            continue
        for method, result in candidate["methods"].items():
            spots = result["root_capacity_overflow_count"]
            nodes = result.get("root_capacity_overflow_node_count", spots)
            if spots or nodes:
                overflow_rows.append(
                    f"{candidate['id']} ({method} phi integration): "
                    f"{spots} spots, {nodes} radial nodes")
    if overflow_rows:
        print("\nRoot-capacity overflows:", flush=True)
        for row in overflow_rows:
            print(f"  {row}", flush=True)


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    apply_dataset(de._MASTER_CFG, args.dataset)
    _validate_args(args)
    cache_dir = _reference_cache_dir(args)
    if args.clean_cache:
        count = _clean_reference_cache(cache_dir)
        print(f"Removed {count} cached reference file(s) from {cache_dir}")
        return 0
    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise SystemExit(
            f"GPU backend required; found {jax.default_backend()!r}. "
            "Use --allow-cpu only for development checks.")
    if jax.default_backend() != "gpu":
        # Match run_de_map's conservative peak-partition CPU guard.  An old
        # PjRt executable can terminate the process during cache restore.
        jax.config.update("jax_enable_compilation_cache", False)

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path(de.results_path(
        de._MASTER_CFG["io"].get("root_output", "results/Megamaser"),
        "convergence", f"phi_partition_{stamp}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    git = _git_metadata()
    seed = (args.seed if args.seed is not None
            else int(de._MASTER_CFG["inference"]["seed"]))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": now.isoformat(),
        "argv": raw_argv,
        "git": git,
        "config_path": str(de._CONFIG_PATH),
        "config_sha256": hashlib.sha256(
            Path(de._CONFIG_PATH).read_bytes()).hexdigest(),
        "validator_source_sha256": _validator_source_hash(),
        "reference_source_sha256": _source_hash(),
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "reference_policy": REFERENCE_POLICY,
        "reference_r_levels": list(args.reference_r_levels),
        "reference_phi_levels_override": (
            None if args.reference_phi_levels is None
            else list(args.reference_phi_levels)),
        "reference_grids_by_galaxy": {
            galaxy: [{"n_r": n_r, "n_phi": n_phi}
                     for n_r, n_phi in _reference_grids(galaxy, args)]
            for galaxy in args.galaxies
        },
        "reference_r_chunk": args.reference_r_chunk,
        "reference_spot_batch": args.reference_spot_batch,
        "reference_cache_ttl_seconds": REFERENCE_CACHE_TTL_SECONDS,
        "sobol_seed": seed,
        "sobol_candidate_count": args.sobol_candidates,
        "local_sobol_per_anchor": args.local_sobol,
        "local_radius_de_unit_box": args.local_radius,
        "checkpoint_candidates": [
            str(path) for path in args.checkpoint_candidate],
        "allow_checkpoint_policy_mismatch": bool(
            args.allow_checkpoint_policy_mismatch),
        "radius_treatment": "conditional-r",
        "methods": list(METHODS),
        "scheme_setting_overrides": _scheme_overrides(
            args.scheme_setting),
        "irrelevance_policy": {
            "gate_floor": GATE_FLOOR,
            "gate_multiplier": GATE_MULTIPLIER,
            "railing_fraction_threshold": RAILING_FRACTION_THRESHOLD,
            "phi_edge_cells": PHI_EDGE_CELLS,
        },
        "tolerances": {
            "reference_total_atol": args.reference_total_atol,
            "reference_spot_atol": args.reference_spot_atol,
            "reference_rms_atol": args.reference_rms_atol,
            "test_total_atol": args.total_atol,
            "test_spot_atol": args.spot_atol,
            "test_spot_p99_atol": args.spot_p99_atol,
            "test_spot_rms_atol": args.spot_rms_atol,
            "ranking_atol": args.ranking_atol,
            "max_ranking_inversions": args.max_ranking_inversions,
            "max_root_overflows": args.max_root_overflows,
        },
    }
    report = {"metadata": metadata, "cases": [], "passed": True}

    for galaxy in args.galaxies:
        for variant in args.variants:
            case = _build_case(galaxy, variant, args, seed)
            print(
                f"Preparing {galaxy}/{variant}: configured phi integration="
                f"{case['configured_phi_integration']}; comparing "
                "fixed-grid vs peak-partition", flush=True)
            candidates, candidate_status = _candidate_rows(
                case, args, seed)
            requested = args.sobol_candidates
            grids = _reference_grids(galaxy, args)
            settings = case["numerical_settings"]
            fixed_settings = settings["fixed-grid"]
            peak_settings = settings["peak-partition"]
            source_counts = {}
            for candidate in candidates:
                kind = candidate["source_kind"]
                source_counts[kind] = source_counts.get(kind, 0) + 1
            source_summary = ", ".join(
                f"{kind}={count}" for kind, count in source_counts.items())
            print(
                "Test: complete conditional-r likelihood; production "
                "fixed-grid and peak-partition versus the same independent "
                "float64 full-support r x phi reference", flush=True)
            print(
                "Production fixed-grid phi: "
                f"systemic={fixed_settings['n_phi_sys']} per sub-range, "
                f"HV core={fixed_settings['n_phi_hv_high']}, "
                f"HV wing={fixed_settings['n_phi_hv_low']}", flush=True)
            print(
                "Production peak scan phi: "
                f"systemic={peak_settings['n_phi_partition_sys']}, "
                f"HV={peak_settings['n_phi_partition_hv']}", flush=True)
            print(
                "Production conditional-r grids: fixed-grid="
                f"{fixed_settings['n_r_local']}+"
                f"{fixed_settings['n_r_global']}; peak-partition="
                f"{peak_settings['n_r_local']}+"
                f"{peak_settings['n_r_global']} (local+global)", flush=True)
            print(
                "Reference r x phi grids: "
                + " -> ".join(f"{n_r}x{n_phi}" for n_r, n_phi in grids)
                + f"; r_chunk={args.reference_r_chunk}, "
                f"spot_batch={args.reference_spot_batch}", flush=True)
            print(
                f"Candidates: {len(candidates)} total ({source_summary}); "
                "each requires the full reference ladder (fresh exact cache "
                "hits load it)", flush=True)
            if any(case["scheme_setting_overrides"].values()):
                print("Scheme-setting overrides: "
                      f"{_canonical_json(case['scheme_setting_overrides'])}",
                      flush=True)
            print(
                f"Timing both production methods over "
                f"{len(candidates)} candidates...", flush=True)
            case_result = {
                "galaxy": galaxy,
                "variant": variant,
                "production_dtype": case["production_dtype"],
                "configured_phi_integration": case[
                    "configured_phi_integration"],
                "validated_phi_integrations": case[
                    "validated_phi_integrations"],
                "radius_treatment": "conditional-r",
                "spot_batch": case["spot_batch"],
                "spot_batch_source": case["spot_batch_source"],
                "objective_policies": {
                    method: de._objective_policy(case["models"][method])
                    for method in METHODS
                },
                "numerical_settings": case["numerical_settings"],
                "scheme_setting_overrides": case[
                    "scheme_setting_overrides"],
                "model_configuration": {
                    "defaults": {
                        key: value
                        for key, value in case["config"]["model"].items()
                        if key != "galaxies"
                    },
                    "galaxy": case["config"]["model"]["galaxies"][galaxy],
                },
                "candidate_status": candidate_status,
                "parameter_names": list(case["names"]),
                "de_bounds": {
                    name: [float(lo), float(hi)]
                    for name, lo, hi in zip(
                        case["names"], case["lo"], case["hi"])},
                "timing": {
                    method: _time_production(
                        case, candidates, args, method)
                    for method in METHODS
                },
                "candidates": [],
                "passed": True,
            }
            case_result["timing_comparison"] = _timing_comparison(
                case_result["timing"])
            candidate_seconds = _time_candidates(case, candidates, args)
            evaluators = {
                method: _production_evaluator(case, method)
                for method in METHODS
            }

            # Every candidate is judged against the full dense ladder; exact
            # fresh cache hits load those arrays instead of recomputing them.
            for candidate_index, candidate in enumerate(candidates, start=1):
                print(f"Evaluating {galaxy}/{variant}/{candidate['id']} "
                      f"[{candidate_index}/{len(candidates)}]...", flush=True)
                production = {
                    method: _evaluate_production(
                        evaluators[method], candidate["values"])
                    for method in METHODS
                }
                candidate_result = _full_result(
                    case, candidate, production, grids, args, cache_dir,
                    record_railing=(candidate["source_kind"]
                                    in ("config", "pesce-reid")))
                candidate_result["production_seconds"] = candidate_seconds[
                    candidate["id"]]
                case_result["candidates"].append(candidate_result)

            # Post-processing classifier: flag posterior-irrelevant needles so
            # their failures are excused, without ever skipping their ladder.
            calibration = None
            if requested > 0:
                calibration = _calibrate_gate([
                    cr for cr in case_result["candidates"]
                    if cr["source_kind"] != "sobol"])
            if requested == 0:
                case_result["gate_calibration"] = {
                    "enabled": False, "reason": "no broad Sobol candidates"}
            elif calibration is None:
                case_result["gate_calibration"] = {
                    "enabled": False,
                    "reason": "no converged anchor reference"}
            else:
                case_result["gate_calibration"] = {
                    "enabled": True, **calibration}

            for candidate, candidate_result in zip(
                    candidates, case_result["candidates"]):
                _classify_irrelevant(
                    case, candidate, candidate_result, calibration)
            case_result["passed"] = _relevant_pass(case_result["candidates"])

            _add_pesce_deltas(case_result)
            _case_rankings(case_result, args)
            _case_worst(case_result)
            case_result["memory_after_validation"] = _device_memory()
            _print_case(case_result)
            report["passed"] &= case_result["passed"]
            report["cases"].append(case_result)

    report["aggregate"] = _aggregate(report)
    clean_report = _strip_internal(report)
    json_path = output_dir / "validation.json"
    markdown_path = output_dir / "validation.md"
    json_path.write_text(
        json.dumps(_json_ready(clean_report), indent=2, sort_keys=True,
                   allow_nan=False) + "\n")
    markdown_path.write_text(_markdown(clean_report))

    print("\n" + "=" * 88, flush=True)
    print(f"OVERALL: {_conclusion(report)}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"Markdown: {markdown_path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
