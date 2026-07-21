#!/usr/bin/env python
"""Validate peak-partition phi integration against converged references.

The production model supplies every physical prediction, transform, prior,
spot split, conditional-radius grid, and peak-partition evaluation.  This
script only supplies deterministic candidates, dense matching-support
reference levels, convergence gates, timing, caching, and reports.
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
    # cache. Old CPU PjRt peak executables can terminate during deserialisation.
    jax.config.update("jax_enable_compilation_cache", False)

SCRIPT_DIR = Path(__file__).resolve().parent
MASER_DIR = SCRIPT_DIR.parent
REPO_ROOT = MASER_DIR.parent.parent
if str(MASER_DIR) not in sys.path:
    sys.path.insert(0, str(MASER_DIR))

import run_de_map as de  # noqa: E402
try:  # noqa: E402
    from .convergence_utils import (cast_floats, cast_model_floats,
                                    dense_phi_reference_per_spot)
except ImportError:  # direct script execution
    from convergence_utils import (cast_floats, cast_model_floats,
                                   dense_phi_reference_per_spot)


SCHEMA_VERSION = 1
REFERENCE_POLICY = "partition_support_trapezoid_f64_frozen_r_v1"
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


def _source_hash():
    """Hash objective and validator sources so dirty code invalidates caches."""
    paths = (
        REPO_ROOT / "candel/model/model_H0_maser.py",
        MASER_DIR / "run_de_map.py",
        SCRIPT_DIR / "convergence_utils.py",
        Path(__file__).resolve(),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _numerical_settings(model):
    signature = inspect.signature(model._phi_partition_log_integral)
    partition = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    return {
        "n_phi_partition_sys": int(model._n_phi_partition_sys),
        "n_phi_partition_hv": int(model._n_phi_partition_hv),
        "n_r_local": int(model._n_r_local),
        "n_r_global": int(model._n_r_global),
        "K_sigma": float(model._K_sigma),
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


def _array_sha256(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


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


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--galaxies", nargs="+", default=["NGC6323"])
    parser.add_argument("--variants", nargs="+", type=_variant,
                        default=["circular"])
    parser.add_argument("--profiles", nargs="+",
                        choices=("fixed-r", "conditional-r"),
                        default=["fixed-r", "conditional-r"])
    parser.add_argument("--sobol-candidates", type=int, default=8)
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
        "--fixed-phi-levels", type=_parse_levels,
        default=_parse_levels("25001,50001,100001"))
    parser.add_argument(
        "--conditional-phi-levels", type=_parse_levels,
        default=_parse_levels("12501,25001,50001"))
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
    parser.add_argument("--reference-spot-batch", type=int, default=1)
    parser.add_argument("--candidate-wave", type=int,
                        choices=(1, 2, 4, 8), default=None)
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
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
    if args.reference_spot_batch < 1:
        raise ValueError("--reference-spot-batch must be positive.")
    if args.spot_batch is not None and args.spot_batch < 1:
        raise ValueError("--spot-batch must be positive.")
    if args.timing_repeats < 1 or args.n_devices < 1:
        raise ValueError("timing repeats and device count must be positive.")
    if args.reference_tail_levels < 2:
        raise ValueError("--reference-tail-levels must be at least 2.")
    for profile, levels in (("fixed-r", args.fixed_phi_levels),
                            ("conditional-r",
                             args.conditional_phi_levels)):
        if (profile in args.profiles
                and args.reference_tail_levels > len(levels)):
            raise ValueError(
                f"{profile} has {len(levels)} levels but "
                f"--reference-tail-levels={args.reference_tail_levels}.")


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
        result = subprocess.run(
            args, cwd=MASER_DIR.parent.parent, capture_output=True,
            text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    revision = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {"revision": revision, "dirty": bool(dirty)}


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


def _build_case(galaxy, variant, args, seed):
    master = de._MASTER_CFG
    gcfg = master["model"]["galaxies"][galaxy]
    data = de.load_megamaser_spots(
        de.data_path("data", "Megamaser"), galaxy,
        v_sys_obs=gcfg["v_sys_obs"])
    distance_bounds = de._distance_bounds(gcfg)
    if distance_bounds is not None:
        data["D_lo"], data["D_hi"] = distance_bounds[:2]

    config = copy.deepcopy(master)
    case_gcfg = config["model"]["galaxies"][galaxy]
    case_gcfg["use_ecc"], case_gcfg["use_quadratic_warp"] = (
        VARIANTS[variant])
    case_gcfg["phi_integration"] = "peak-partition"
    if args.spot_batch is not None:
        case_gcfg["conditional_spot_batch"] = int(args.spot_batch)

    production_f64 = bool(gcfg.get("force_f64", False))
    production_dtype = jnp.float64 if production_f64 else jnp.float32
    model = _write_model_config(config, data, production_dtype)
    reference_model = _write_model_config(config, data, jnp.float64)

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
    target = de.MaserBlackJaxTarget(
        model, de._h_ref(model), init_params, spot_batch=target_batch)
    reference_init = cast_floats(init_params, jnp.float64)
    reference_target = de.MaserBlackJaxTarget(
        reference_model, de._h_ref(reference_model), reference_init,
        spot_batch=args.reference_spot_batch)
    if target.names != reference_target.names:
        raise RuntimeError("production and reference target layouts differ.")

    sobol_n_sigma = master.get("optimise", {}).get("sobol_n_sigma", 5)
    names, sizes, lo, hi = de._layout(target, sobol_n_sigma)
    return {
        "galaxy": galaxy,
        "variant": variant,
        "config": config,
        "config_hash": _sha256_json(config["model"]),
        "model": model,
        "reference_model": reference_model,
        "target": target,
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
        "spot_batch": target_batch,
        "spot_batch_source": batch_source,
        "source_hash": _source_hash(),
        "numerical_settings": _numerical_settings(model),
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


def _candidate_rows(case, args, seed):
    names, lo, hi = case["names"], case["lo"], case["hi"]
    scale = hi - lo
    rows = []
    status = []
    if case["init_note"] is not None:
        status.append(case["init_note"])

    def add(source, source_kind, values):
        values = np.asarray(values, dtype=np.float64)
        rows.append({
            "id": f"{source_kind}-{sum(
                row['source_kind'] == source_kind for row in rows):04d}",
            "source": source,
            "source_kind": source_kind,
            "values": values,
            "within_de_bounds": bool(np.all(
                (values >= lo) & (values <= hi))),
        })

    broad = _sobol(args.sobol_candidates, len(names), seed)
    for index, unit in enumerate(broad):
        add(f"scrambled Sobol seed={seed} index={index}", "sobol",
            lo + unit * scale)

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

    for anchor_index, (kind, values) in enumerate(anchors):
        centre = (np.asarray(values) - lo) / scale
        cloud = _sobol(
            args.local_sobol, len(names), seed + 1000 + anchor_index)
        for index, unit in enumerate(cloud):
            local = centre + (2.0 * unit - 1.0) * args.local_radius
            local = _reflect_unit_box(local)
            add(f"local Sobol around {kind} index={index}",
                f"local-{kind}", lo + local * scale)

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


def _production_evaluator(case):
    model = case["model"]
    target = case["target"]
    names = case["names"]
    dtype = (jnp.float64 if case["production_dtype"] == "float64"
             else jnp.float32)
    n_r = model._n_r_local + model._n_r_global

    def evaluate(values):
        values = jnp.asarray(values, dtype=dtype)
        theta = target.complete_params(de._flat_to_theta(values, names))
        phys_args, phys_kw = model.phys_from_params_jax(theta, target.h)
        r_fixed, _, _, _ = model._closed_form_seeds(
            phys_args[2], phys_args[3], phys_args[4], phys_args[16],
            phys_args[8], phys_args[15])

        fixed_groups = model._spot_groups_from_r(r_fixed)
        fixed_ll = model._eval_phi_marginal(
            fixed_groups, phys_args, phys_kw, spot_batch=target.spot_batch)
        fixed_roots = jnp.zeros(model.n_spots, dtype=jnp.int32)
        fixed_overflow = jnp.zeros(model.n_spots, dtype=bool)
        for type_key, idx, r_group, _ in fixed_groups:
            roots, overflow = _partition_diagnostics(
                model, type_key, idx, r_group, phys_args, phys_kw,
                target.spot_batch)
            fixed_roots = fixed_roots.at[idx].set(roots.astype(jnp.int32))
            fixed_overflow = fixed_overflow.at[idx].set(overflow)

        groups, caches = model._build_conditional_r_grids(
            phys_args[2], phys_args[3], phys_args[4], phys_args[16],
            phys_args[8], phys_args[15], phys_args, phys_kw,
            return_scan_cache=True)
        conditional_ll = jnp.zeros(model.n_spots, dtype=dtype)
        conditional_roots = jnp.zeros(model.n_spots, dtype=jnp.int32)
        conditional_overflow = jnp.zeros(model.n_spots, dtype=bool)
        conditional_overflow_nodes = jnp.zeros((), dtype=jnp.int32)
        r_conditional = jnp.zeros((model.n_spots, n_r), dtype=dtype)
        log_w_conditional = jnp.zeros((model.n_spots, n_r), dtype=dtype)

        for group, cache in zip(groups, caches):
            type_key, idx, r_union, log_w_r = group
            ll = model._marginal_per_spot_r(
                type_key, idx, r_union, log_w_r,
                model._group_has_any_accel(type_key), phys_args, phys_kw,
                (None if target.spot_batch is None else
                 min(int(target.spot_batch), int(idx.shape[0]))), cache)
            roots, overflow_nodes = _partition_diagnostics(
                model, type_key, idx, r_union, phys_args, phys_kw,
                target.spot_batch)
            any_overflow = jnp.any(overflow_nodes, axis=-1)
            conditional_ll = conditional_ll.at[idx].set(ll)
            conditional_roots = conditional_roots.at[idx].set(
                jnp.max(roots, axis=-1).astype(jnp.int32))
            conditional_overflow = conditional_overflow.at[idx].set(
                any_overflow)
            conditional_overflow_nodes += jnp.sum(
                overflow_nodes, dtype=jnp.int32)
            r_conditional = r_conditional.at[idx].set(r_union)
            log_w_conditional = log_w_conditional.at[idx].set(log_w_r)

        return {
            "fixed_ll": fixed_ll,
            "fixed_r": r_fixed,
            "fixed_roots": fixed_roots,
            "fixed_overflow": fixed_overflow,
            "conditional_ll": conditional_ll,
            "conditional_r": r_conditional,
            "conditional_log_w_r": log_w_conditional,
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


def _time_production(case, candidates, args):
    names, lo, hi = case["names"], case["lo"], case["hi"]
    dtype = (jnp.float64 if case["production_dtype"] == "float64"
             else jnp.float32)
    lo_j = jnp.asarray(lo, dtype=dtype)
    scale_j = jnp.asarray(hi - lo, dtype=dtype)
    logp = de._make_logp(case["target"], names)

    def fitness_one(unit):
        values = lo_j + jnp.asarray(unit, dtype=dtype) * scale_j
        return -logp(values)

    devices = tuple(jax.local_devices()[:args.n_devices])
    if len(devices) != args.n_devices:
        raise ValueError(
            f"requested {args.n_devices} devices, found {len(devices)}")
    candidate_wave = de._de_candidates_per_wave(
        case["model"], args.candidate_wave)
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


def _reference_metadata(case, candidate, profile, levels, production,
                        git_revision):
    if profile == "fixed-r":
        radial_hash = _array_sha256(production["fixed_r"])
    else:
        radial_hash = _array_sha256(
            production["conditional_r"],
            production["conditional_log_w_r"])
    return {
        "schema": SCHEMA_VERSION,
        "policy": REFERENCE_POLICY,
        "git_revision": git_revision,
        "source_hash": case["source_hash"],
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "backend": jax.default_backend(),
        "device_kinds": sorted({
            device.device_kind for device in jax.devices()}),
        "config_hash": case["config_hash"],
        "objective_policy": de._objective_policy(case["model"]),
        "galaxy": case["galaxy"],
        "variant": case["variant"],
        "profile": profile,
        "phi_levels": list(levels),
        "dtype": "float64",
        "production_dtype": case["production_dtype"],
        "candidate_names": list(case["names"]),
        "candidate_values": candidate["values"].tolist(),
        "radial_hash": radial_hash,
        "radial_settings": {
            "n_r_local": case["model"]._n_r_local,
            "n_r_global": case["model"]._n_r_global,
            "K_sigma": case["model"]._K_sigma,
        },
    }


def _load_reference_cache(cache_dir, metadata):
    key = reference_cache_key(metadata)
    path = cache_dir / f"{key}.npz"
    if not path.is_file():
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


def _reference_levels(case, candidate, profile, levels, production, args,
                      cache_dir, git_revision):
    metadata = _reference_metadata(
        case, candidate, profile, levels, production, git_revision)
    cache_path = None
    if not args.no_cache:
        cached, cache_path = _load_reference_cache(cache_dir, metadata)
        if cached is not None:
            expected_shape = (len(levels), case["model"].n_spots)
            if cached.shape != expected_shape:
                raise RuntimeError(
                    f"reference cache shape {cached.shape} does not match "
                    f"{expected_shape}: {cache_path}")
            return cached, {
                "cache_hit": True, "cache_key": cache_path.stem,
                "cache_path": str(cache_path), "level_seconds": [],
            }

    values = jnp.asarray(candidate["values"], dtype=jnp.float64)
    theta = case["reference_target"].complete_params(
        de._flat_to_theta(values, case["names"]))
    phys_args, phys_kw = case["reference_model"].phys_from_params_jax(
        theta, case["reference_target"].h)
    if profile == "fixed-r":
        r_ang = production["fixed_r"]
        log_w_r = None
    else:
        r_ang = production["conditional_r"]
        log_w_r = production["conditional_log_w_r"]

    reference = []
    level_seconds = []
    for level in levels:
        start = time.perf_counter()
        reference.append(dense_phi_reference_per_spot(
            case["reference_model"], phys_args, phys_kw,
            r_ang, level, args.reference_spot_batch,
            log_w_r=log_w_r, partition_support=True))
        level_seconds.append(time.perf_counter() - start)
    reference = np.stack(reference)
    if not args.no_cache:
        if cache_path is None:
            cache_path = cache_dir / (
                reference_cache_key(metadata) + ".npz")
        _save_reference_cache(cache_path, metadata, reference)
    return reference, {
        "cache_hit": False,
        "cache_key": reference_cache_key(metadata),
        "cache_path": None if args.no_cache else str(cache_path),
        "level_seconds": level_seconds,
    }


def _population_labels(model):
    labels = np.empty(model.n_spots, dtype=object)
    labels[np.asarray(model._idx_sys)] = "systemic"
    labels[np.asarray(model._idx_red)] = "red"
    labels[np.asarray(model._idx_blue)] = "blue"
    return labels


def _profile_record(case, profile, test, overflow, roots, reference,
                    levels, cache_info, args):
    criteria = {
        "total_atol": args.reference_total_atol,
        "spot_atol": args.reference_spot_atol,
        "rms_atol": args.reference_rms_atol,
    }
    convergence = reference_convergence(
        reference, args.reference_tail_levels, criteria)
    labels = _population_labels(case["model"])
    for previous_n, current_n, previous, current, row in zip(
            levels, levels[1:], reference, reference[1:],
            convergence["comparisons"]):
        row["previous_n_phi"] = int(previous_n)
        row["current_n_phi"] = int(current_n)
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
        "profile": profile,
        "reference_levels": [
            {"n_phi": level,
             "total_log_likelihood": float(np.sum(values)),
             "finite_spots": int(np.count_nonzero(np.isfinite(values)))}
            for level, values in zip(levels, reference)],
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


def _case_rankings(case_result, args):
    labels = [candidate["id"] for candidate in case_result["candidates"]]
    rankings = {}
    for profile in args.profiles:
        levels = (args.fixed_phi_levels if profile == "fixed-r"
                  else args.conditional_phi_levels)
        records = [candidate["profiles"][profile]
                   for candidate in case_result["candidates"]]
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
                "previous_n_phi": int(levels[index]),
                "current_n_phi": int(levels[index + 1]),
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
        rankings[profile] = {
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
    worst = {}
    for profile in case_result["candidates"][0]["profiles"]:
        candidates = []
        spots = []
        for candidate in case_result["candidates"]:
            record = candidate["profiles"][profile]
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
        worst[profile] = {
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
            for profile, result in candidate["profiles"].items():
                key = (case["galaxy"], case["variant"],
                       candidate["source_kind"], profile)
                grouped[key].append(result)
    rows = []
    for key, values in sorted(grouped.items()):
        comparisons = [value["comparison"] for value in values]

        def finite_max(field):
            entries = [row[field] for row in comparisons
                       if np.isfinite(row[field])]
            return max(entries) if entries else np.nan

        rows.append({
            "galaxy": key[0],
            "variant": key[1],
            "point_source": key[2],
            "profile": key[3],
            "candidate_count": len(values),
            "passed": all(value["passed"] for value in values),
            "reference_unconverged": sum(
                not value["reference_convergence"]["converged"]
                for value in values),
            "finite_mask_mismatches": sum(
                row["finite_mask_mismatches"] for row in comparisons),
            "root_capacity_overflows": sum(
                value["root_capacity_overflow_count"] for value in values),
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


def _markdown(report):
    lines = [
        "# Megamaser peak-partition validation", "",
        f"**Conclusion: {'PASS' if report['passed'] else 'FAIL'}**", "",
        "## Candidate results", "",
        "| Galaxy | Variant | Source | Candidate | Profile | Ref converged | "
        "Test logL | Ref logL | abs total error | max spot | p99 spot | "
        "RMS spot | mask mismatch | overflows | Pass |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        for candidate in case["candidates"]:
            for profile, result in candidate["profiles"].items():
                comparison = result["comparison"]
                lines.append(
                    f"| {case['galaxy']} | {case['variant']} | "
                    f"{candidate['source_kind']} | {candidate['id']} | "
                    f"{profile} | "
                    f"{result['reference_convergence']['converged']} | "
                    f"{_fmt(result['test_total_log_likelihood'], '.6g')} | "
                    f"{_fmt(result['reference_total_log_likelihood'], '.6g')} | "
                    f"{_fmt(comparison['absolute_total_error'])} | "
                    f"{_fmt(comparison['max_absolute_spot_error'])} | "
                    f"{_fmt(comparison['p99_absolute_spot_error'])} | "
                    f"{_fmt(comparison['rms_spot_error'])} | "
                    f"{comparison['finite_mask_mismatches']} | "
                    f"{result['root_capacity_overflow_count']} | "
                    f"{result['passed']} |")

    lines.extend([
        "", "## Aggregate by galaxy, variant, point source, and profile", "",
        "| Galaxy | Variant | Source | Profile | N | Unconverged refs | "
        "mask mismatch | overflows | worst abs total | worst spot | "
        "worst p99 | worst RMS | Pass |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["aggregate"]:
        lines.append(
            f"| {row['galaxy']} | {row['variant']} | "
            f"{row['point_source']} | {row['profile']} | "
            f"{row['candidate_count']} | {row['reference_unconverged']} | "
            f"{row['finite_mask_mismatches']} | "
            f"{row['root_capacity_overflows']} | "
            f"{_fmt(row['worst_absolute_total_error'])} | "
            f"{_fmt(row['worst_absolute_spot_error'])} | "
            f"{_fmt(row['worst_p99_spot_error'])} | "
            f"{_fmt(row['worst_rms_spot_error'])} | {row['passed']} |")

    lines.extend([
        "", "## Reference convergence", "",
        "| Galaxy | Variant | Candidate | Profile | n_phi | next n_phi | "
        "abs total change | max spot | median | p95 | p99 | RMS | "
        "mask mismatch | Pass |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in report["cases"]:
        for candidate in case["candidates"]:
            for profile, result in candidate["profiles"].items():
                for row in result["reference_convergence"]["comparisons"]:
                    lines.append(
                        f"| {case['galaxy']} | {case['variant']} | "
                        f"{candidate['id']} | {profile} | "
                        f"{row['previous_n_phi']} | {row['current_n_phi']} | "
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
        "| Galaxy | Variant | Candidate | Profile | Population | "
        "signed total | max spot | median | p95 | p99 | RMS | "
        "mask mismatch |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in report["cases"]:
        for candidate in case["candidates"]:
            for profile, result in candidate["profiles"].items():
                for population, row in result["errors_by_population"].items():
                    lines.append(
                        f"| {case['galaxy']} | {case['variant']} | "
                        f"{candidate['id']} | {profile} | {population} | "
                        f"{_fmt(row['signed_total_error'])} | "
                        f"{_fmt(row['max_absolute_spot_error'])} | "
                        f"{_fmt(row['median_absolute_spot_error'])} | "
                        f"{_fmt(row['p95_absolute_spot_error'])} | "
                        f"{_fmt(row['p99_absolute_spot_error'])} | "
                        f"{_fmt(row['rms_spot_error'])} | "
                        f"{row['finite_mask_mismatches']} |")

    lines.extend([
        "", "## Rankings and worst spots", "",
        "| Galaxy | Variant | Profile | Reference ranking inversions | "
        "test/reference inversions | Worst candidate | Worst spot error |",
        "|---|---|---|---:|---:|---|---:|",
    ])
    for case in report["cases"]:
        for profile, ranking in case["rankings"].items():
            reference_count = ranking["reference_tail_inversion_count"]
            worst = case["worst"][profile]
            lines.append(
                f"| {case['galaxy']} | {case['variant']} | {profile} | "
                f"{reference_count} | "
                f"{ranking['test_vs_reference']['count']} | "
                f"{worst['candidate']} | "
                f"{_fmt(worst['max_absolute_spot_error'])} |")
    lines.extend([
        "", "| Galaxy | Variant | Profile | Candidate | Spot | Population | "
        "signed error | absolute error | roots | overflow |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for case in report["cases"]:
        for profile, worst in case["worst"].items():
            for spot in worst["spots"]:
                lines.append(
                    f"| {case['galaxy']} | {case['variant']} | {profile} | "
                    f"{spot['candidate']} | {spot['spot']} | "
                    f"{spot['population']} | {_fmt(spot['signed_error'])} | "
                    f"{_fmt(spot['absolute_error'])} | "
                    f"{spot['root_count']} | "
                    f"{spot['root_capacity_overflow']} |")

    lines.extend(["", "## Timing and memory", ""])
    for case in report["cases"]:
        timing = case["timing"]
        lines.extend([
            f"### {case['galaxy']} / {case['variant']}", "",
            f"- Production dtype: `{case['production_dtype']}`",
            f"- Spot batch: `{case['spot_batch']}` "
            f"({case['spot_batch_source']})",
            f"- Cold compile + evaluation: "
            f"{timing['cold_compile_and_evaluate_seconds']:.3f} s",
            f"- Estimated compile component: "
            f"{timing['estimated_compile_seconds']:.3f} s",
            f"- Warm steady evaluation: "
            f"{timing['steady_evaluation_seconds']:.3f} s",
            f"- Throughput: "
            f"{timing['throughput_candidates_per_second']:.3f} candidates/s",
            f"- Backend memory stats: `{_canonical_json(timing['memory'])}`",
            f"- Backend memory after validation: "
            f"`{_canonical_json(timing['memory_after_validation'])}`",
            f"- Evaluator profile: "
            f"`{_canonical_json(timing['device_profile'])}`",
            "",
        ])

    reproduction = {
        "run": report["metadata"],
        "cases": [{
            "galaxy": case["galaxy"],
            "variant": case["variant"],
            "production_dtype": case["production_dtype"],
            "spot_batch": case["spot_batch"],
            "spot_batch_source": case["spot_batch_source"],
            "objective_policy": case["objective_policy"],
            "numerical_settings": case["numerical_settings"],
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


def _print_case(case):
    print("\n" + "=" * 88, flush=True)
    print(f"{case['galaxy']} / {case['variant']} / "
          f"{case['production_dtype']}", flush=True)
    print("=" * 88, flush=True)
    for message in case["candidate_status"]:
        print(f"SKIP: {message}", flush=True)
    timing = case["timing"]
    print(f"Timing: cold={timing['cold_compile_and_evaluate_seconds']:.3f}s, "
          f"steady={timing['steady_evaluation_seconds']:.3f}s, "
          f"throughput={timing['throughput_candidates_per_second']:.3f} "
          "candidate/s", flush=True)
    header = (f"{'candidate':<20} {'profile':<14} {'ref':<5} "
              f"{'abs total':>11} {'max spot':>11} {'p99':>11} "
              f"{'mask':>5} {'overflow':>8} {'result':>7}")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for candidate in case["candidates"]:
        for profile, result in candidate["profiles"].items():
            comparison = result["comparison"]
            print(
                f"{candidate['id']:<20} {profile:<14} "
                f"{str(result['reference_convergence']['converged']):<5} "
                f"{_fmt(comparison['absolute_total_error']):>11} "
                f"{_fmt(comparison['max_absolute_spot_error']):>11} "
                f"{_fmt(comparison['p99_absolute_spot_error']):>11} "
                f"{comparison['finite_mask_mismatches']:>5} "
                f"{result['root_capacity_overflow_count']:>8} "
                f"{'PASS' if result['passed'] else 'FAIL':>7}",
                flush=True)
    for profile, ranking in case["rankings"].items():
        reference_inversions = ranking["reference_tail_inversion_count"]
        worst = case["worst"][profile]
        print(
            f"{profile}: reference ranking inversions="
            f"{reference_inversions}, test/reference inversions="
            f"{ranking['test_vs_reference']['count']}; worst="
            f"{worst['candidate']} spot error="
            f"{_fmt(worst['max_absolute_spot_error'])}", flush=True)


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    _validate_args(args)
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
    cache_dir = args.cache_dir or Path(de.results_path(
        de._MASTER_CFG["io"].get("root_output", "results/Megamaser"),
        "convergence", "reference_cache"))
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
        "validator_source_sha256": _source_hash(),
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "reference_policy": REFERENCE_POLICY,
        "fixed_phi_levels": list(args.fixed_phi_levels),
        "conditional_phi_levels": list(args.conditional_phi_levels),
        "reference_spot_batch": args.reference_spot_batch,
        "sobol_seed": seed,
        "sobol_candidate_count": args.sobol_candidates,
        "local_sobol_per_anchor": args.local_sobol,
        "local_radius_de_unit_box": args.local_radius,
        "profiles": args.profiles,
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
            candidates, candidate_status = _candidate_rows(case, args, seed)
            case_result = {
                "galaxy": galaxy,
                "variant": variant,
                "production_dtype": case["production_dtype"],
                "spot_batch": case["spot_batch"],
                "spot_batch_source": case["spot_batch_source"],
                "objective_policy": de._objective_policy(case["model"]),
                "numerical_settings": case["numerical_settings"],
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
                "timing": _time_production(case, candidates, args),
                "candidates": [],
                "passed": True,
            }
            evaluator = _production_evaluator(case)
            for candidate in candidates:
                print(f"Evaluating {galaxy}/{variant}/"
                      f"{candidate['id']}...", flush=True)
                production = _evaluate_production(
                    evaluator, candidate["values"])
                candidate_result = {
                    key: value for key, value in candidate.items()
                    if key != "values"
                }
                candidate_result["parameters"] = {
                    name: float(value) for name, value in zip(
                        case["names"], candidate["values"])}
                candidate_result["profiles"] = {}
                if "fixed-r" in args.profiles:
                    levels = args.fixed_phi_levels
                    reference, cache_info = _reference_levels(
                        case, candidate, "fixed-r", levels, production,
                        args, cache_dir, git["revision"])
                    candidate_result["profiles"]["fixed-r"] = (
                        _profile_record(
                            case, "fixed-r", production["fixed_ll"],
                            production["fixed_overflow"],
                            production["fixed_roots"], reference, levels,
                            cache_info, args))
                if "conditional-r" in args.profiles:
                    levels = args.conditional_phi_levels
                    reference, cache_info = _reference_levels(
                        case, candidate, "conditional-r", levels,
                        production, args, cache_dir, git["revision"])
                    record = _profile_record(
                        case, "conditional-r",
                        production["conditional_ll"],
                        production["conditional_overflow"],
                        production["conditional_roots"], reference, levels,
                        cache_info, args)
                    record["root_capacity_overflow_node_count"] = int(
                        production["conditional_overflow_nodes"])
                    candidate_result["profiles"]["conditional-r"] = record
                candidate_result["passed"] = all(
                    value["passed"] for value
                    in candidate_result["profiles"].values())
                case_result["passed"] &= candidate_result["passed"]
                case_result["candidates"].append(candidate_result)

            _case_rankings(case_result, args)
            _case_worst(case_result)
            case_result["timing"]["memory_after_validation"] = (
                _device_memory())
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
    print(f"OVERALL: {'PASS' if report['passed'] else 'FAIL'}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"Markdown: {markdown_path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
