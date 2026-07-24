# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
"""Benchmark exact megamaser DE GPU tiling on fixed checkpoint candidates.

The suite runs every configuration in a fresh child process so compiled JAX
executables and allocator state cannot leak between measurements.  It bypasses
the exact-value archive deliberately: every repeat evaluates the same physical
candidate array with the unchanged production quadrature grids.
"""
import argparse
import atexit
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time


_RESULT_PREFIX = "BATCH_BENCHMARK_JSON="


def _spot_batch_arg(value):
    """Parse a positive spot count or the literal ``all``."""
    if str(value).lower() in ("all", "none"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "spot batch must be a positive integer or 'all'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "spot batch must be a positive integer or 'all'")
    return parsed


def _spot_batch_label(value):
    return "all" if value is None else str(value)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("galaxy")
    parser.add_argument("--suite", action="store_true",
                        help="Run several spot-batch sizes, then repeat the "
                             "baseline, each in a fresh process.")
    parser.add_argument(
        "--spot-batches", default=None,
        help="Comma-separated suite override; each item is a positive spot "
             "count or 'all', for example 8,16,68,all.")
    parser.add_argument("--child", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--spot-batch", type=_spot_batch_arg, default=34)
    parser.add_argument(
        "--phi-integration", choices=("fixed-grid", "peak-partition"),
        default=None,
        help="Production integrator override (default: configured method).")
    parser.add_argument(
        "--candidate-wave", type=int, choices=(1, 2, 4, 8), default=None,
        help="Peak-partition candidates evaluated concurrently per GPU "
             "(default: production policy; fixed grid requires one).")
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Per-configuration timeout in suite mode.")
    parser.add_argument(
        "--trace-memory", action="store_true",
        help="Sample nvidia-smi device memory at 50 ms intervals.")
    parser.add_argument(
        "--include-fixed-score", action="store_true",
        help="Also trace the production-like fixed-global startup score.")
    return parser


def _run_suite(args):
    if args.spot_batches is None:
        spot_batches = [34, 68, None, 34]
    else:
        try:
            spot_batches = [
                _spot_batch_arg(item.strip())
                for item in args.spot_batches.split(",")]
        except argparse.ArgumentTypeError as exc:
            raise ValueError(
                "--spot-batches must contain positive integers or 'all'."
            ) from exc
        if not spot_batches:
            raise ValueError("--spot-batches must not be empty.")
    results = []
    reference_values = None
    for run_index, spot_batch in enumerate(spot_batches, 1):
        spot_label = _spot_batch_label(spot_batch)
        print("\n" + "=" * 72, flush=True)
        print(f"BENCHMARK {run_index}/{len(spot_batches)}: "
              f"spot_batch={spot_label}",
              flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__), args.galaxy,
            "--child", "--spot-batch", spot_label,
            "--candidates", str(args.candidates),
            "--warmups", str(args.warmups),
            "--repeats", str(args.repeats),
            "--n-devices", str(args.n_devices),
            "--seed", str(args.seed),
        ]
        if args.phi_integration is not None:
            cmd.extend(["--phi-integration", args.phi_integration])
        if args.candidate_wave is not None:
            cmd.extend(["--candidate-wave", str(args.candidate_wave)])
        if args.trace_memory:
            cmd.append("--trace-memory")
        if args.include_fixed_score:
            cmd.append("--include-fixed-score")
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=args.timeout,
                check=False)
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            print(output, end="", flush=True)
            result = {
                "spot_batch": spot_batch,
                "status": "timeout",
                "wall_seconds": time.perf_counter() - start,
            }
            results.append(result)
            print(_RESULT_PREFIX + json.dumps(result, sort_keys=True),
                  flush=True)
            continue

        output = completed.stdout + completed.stderr
        print(output, end="", flush=True)
        marker = next(
            (line[len(_RESULT_PREFIX):] for line in output.splitlines()
             if line.startswith(_RESULT_PREFIX)), None)
        if marker is not None:
            result = json.loads(marker)
        else:
            result = {
                "spot_batch": spot_batch,
                "status": f"exit-{completed.returncode}",
            }
        result["suite_wall_seconds"] = time.perf_counter() - start
        values = result.pop("values", None)
        if result.get("status") == "ok" and values is not None:
            if reference_values is None:
                reference_values = values
            if len(values) != len(reference_values):
                result["cross_tiling_shape_match"] = False
                result["cross_tiling_finite_mask_mismatches"] = None
                result["cross_tiling_max_abs_difference"] = None
                result["cross_tiling_rms_difference"] = None
            else:
                finite_pairs = [
                    (math.isfinite(a), math.isfinite(b), a, b)
                    for a, b in zip(reference_values, values)]
                differences = [
                    float(b - a) for fa, fb, a, b in finite_pairs
                    if fa and fb]
                result["cross_tiling_shape_match"] = True
                result["cross_tiling_finite_mask_mismatches"] = sum(
                    fa != fb for fa, fb, _, _ in finite_pairs)
                result["cross_tiling_max_abs_difference"] = (
                    max(map(abs, differences)) if differences else None)
                result["cross_tiling_rms_difference"] = (
                    math.sqrt(sum(value * value for value in differences)
                              / len(differences))
                    if differences else None)
        results.append(result)

    print("\n" + "=" * 72, flush=True)
    print("BATCH_BENCHMARK_SUITE_JSON="
          + json.dumps(results, sort_keys=True), flush=True)
    ok = [row for row in results if row.get("status") == "ok"]
    if ok:
        best = max(ok, key=lambda row: row["candidates_per_second"])
        print("BATCH_BENCHMARK_BEST_JSON="
              + json.dumps(best, sort_keys=True), flush=True)
        digests = {row.get("values_sha256") for row in ok}
        print("BATCH_BENCHMARK_DIGESTS_IDENTICAL="
              + str(len(digests) == 1).lower(), flush=True)


def _build_target(de, galaxy, spot_batch, seed, phi_integration=None):
    import jax
    import tomli_w

    master = de._MASTER_CFG
    galaxies = master["model"]["galaxies"]
    if galaxy not in galaxies:
        raise ValueError(f"Unknown galaxy {galaxy!r}.")
    gcfg = galaxies[galaxy]
    data = de.load_megamaser_spots(
        de.data_path("data", "Megamaser"), galaxy,
        v_sys_obs=gcfg["v_sys_obs"])
    distance_bounds = de._distance_bounds(gcfg)
    if distance_bounds is not None:
        data["D_lo"], data["D_hi"] = distance_bounds[:2]

    config = {
        "inference": copy.deepcopy(master["inference"]),
        "model": copy.deepcopy(master["model"]),
        "io": copy.deepcopy(master["io"]),
        "optimise": copy.deepcopy(master.get("optimise", {})),
    }
    galaxy_config = config["model"]["galaxies"][galaxy]
    if phi_integration is not None:
        galaxy_config["phi_integration"] = phi_integration
    if spot_batch is None:
        galaxy_config.pop("conditional_spot_batch", None)
    else:
        galaxy_config["conditional_spot_batch"] = spot_batch
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml",
                                      delete=False)
    tomli_w.dump(config, tmp)
    tmp.close()
    try:
        model = de.MaserDiskModel(tmp.name, data)
    finally:
        os.unlink(tmp.name)

    init_cfg = de._init_block(config["model"]["galaxies"][galaxy], model)
    init_params = de._make_init(
        model, init_cfg, "config",
        int(de._required_inference(master["inference"], "init_num_samples")),
        jax.random.PRNGKey(seed))
    target = de.MaserBlackJaxTarget(
        model, de._h_ref(model), init_params, spot_batch=spot_batch)
    return model, target, master, init_params


def _checkpoint_points(
        de, model, target, master, galaxy, candidates, seed):
    """Load fixed checkpoint points, or deterministically generate Sobol."""
    import numpy as np

    sobol_n_sigma = master.get("optimise", {}).get("sobol_n_sigma", 5)
    names, sizes, lo, hi = de._layout(target, sobol_n_sigma)
    ckpt_dir = de.results_path(
        master["io"].get("root_output", "results/Megamaser"),
        "de_checkpoints", galaxy)
    ckpt_path = os.path.join(
        ckpt_dir, de._de_checkpoint_filename(model, seed))

    def sobol_fallback(reason):
        exponent = max(0, (int(candidates) - 1).bit_length())
        points = de.Sobol(
            d=len(names), scramble=True, seed=seed).random_base2(exponent)
        points = np.ascontiguousarray(points[:candidates], dtype=np.float32)
        source = f"scrambled Sobol seed={seed}; {reason} at {ckpt_path}"
        return points, names, lo, hi, source

    if not os.path.isfile(ckpt_path):
        return sobol_fallback("compatible checkpoint absent")

    checkpoint = None
    try:
        checkpoint = de._load_de_checkpoint(
            ckpt_path, lo, hi, names, sizes)
        de._validate_de_checkpoint_policy(
            checkpoint, ckpt_path, de._objective_policy(model),
            optimizer_seed=seed)
    except (KeyError, ValueError) as exc:
        if checkpoint is not None:
            checkpoint.close()
        return sobol_fallback(
            "checkpoint incompatible (" + " ".join(str(exc).split()) + ")")
    population = np.asarray(checkpoint["population"], dtype=np.float32)
    if candidates > len(population):
        checkpoint.close()
        raise ValueError(
            f"Requested {candidates} candidates from a checkpoint population "
            f"of {len(population)}.")
    indices = np.linspace(0, len(population) - 1, candidates, dtype=int)
    points = np.ascontiguousarray(population[indices])
    checkpoint.close()
    return points, names, lo, hi, ckpt_path


def _device_memory(devices):
    rows = []
    for device in devices:
        stats = device.memory_stats() or {}
        rows.append({
            "device": str(device),
            "bytes_in_use": int(stats.get("bytes_in_use", 0)),
            "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
        })
    return rows


def _memory_geometry(model, dtype_bytes):
    """First-order live scan geometry for one candidate.

    Peak partition never materialises the legacy fixed-phi tensor described
    by ``_phi_concat``.  Its local and cached-global radial passes are also
    separate, so report the larger pass and the configured partition scan.
    JAX ``peak_bytes_in_use`` remains the authoritative whole-executable
    measurement because root refinement and quadrature use additional,
    shorter-lived arrays.
    """
    peak_partition = model.phi_integration == "peak-partition"
    n_r = int(max(model._n_r_local, model._n_r_global)
              if peak_partition
              else model._n_r_local + model._n_r_global)
    groups = {}
    for name in ("sys", "red", "blue"):
        n_spots = int(getattr(model, f"_n_{name}"))
        n_half_planes = 2 if peak_partition and name == "sys" else 1
        n_phi = int(
            model._phi_partition_scan_size(name)
            if peak_partition
            else model._phi_concat[name]["sin_phi"].shape[0])
        bytes_per_spot = (
            n_half_planes * n_r * n_phi * int(dtype_bytes))
        groups[name] = {
            "n_spots": n_spots,
            ("n_phi_scan" if peak_partition else "n_phi"): n_phi,
            "n_half_planes": n_half_planes,
            "bytes_per_spot_candidate": bytes_per_spot,
            "all_spots_one_candidate_bytes": n_spots * bytes_per_spot,
        }
    return {
        "policy": ("peak-partition largest radial scan pass"
                   if peak_partition else "fixed-grid radial union"),
        "n_r": n_r,
        "dtype_bytes": int(dtype_bytes),
        "groups": groups,
    }


class _NvidiaSmiSampler:
    """Continuously sample aggregate per-device state with one subprocess."""

    def __init__(self, interval_ms=50):
        self.interval_ms = int(interval_ms)
        self._lock = threading.Lock()
        self._rows = {}
        self._process = None
        self._thread = None
        self._stderr = ""

    def _reader(self):
        assert self._process is not None
        assert self._process.stdout is not None
        for line in self._process.stdout:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            try:
                index, memory, utilisation, temperature = map(int, fields)
            except ValueError:
                continue
            with self._lock:
                row = self._rows.setdefault(index, {
                    "samples": 0,
                    "min_memory_mib": memory,
                    "max_memory_mib": memory,
                    "max_utilisation_percent": utilisation,
                    "max_temperature_c": temperature,
                })
                row["samples"] += 1
                row["min_memory_mib"] = min(row["min_memory_mib"], memory)
                row["max_memory_mib"] = max(row["max_memory_mib"], memory)
                row["max_utilisation_percent"] = max(
                    row["max_utilisation_percent"], utilisation)
                row["max_temperature_c"] = max(
                    row["max_temperature_c"], temperature)

    def start(self):
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-lms", str(self.interval_ms),
        ]
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def reset(self):
        with self._lock:
            self._rows = {}

    def snapshot(self):
        with self._lock:
            rows = copy.deepcopy(self._rows)
        for row in rows.values():
            row["memory_span_mib"] = (
                row["max_memory_mib"] - row["min_memory_mib"])
        return {
            "interval_ms": self.interval_ms,
            "devices": {str(key): value
                        for key, value in sorted(rows.items())},
        }

    def stop(self):
        if self._process is None:
            return
        process = self._process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if process.stderr is not None:
            self._stderr = process.stderr.read().strip()
        self._process = None

    @property
    def stderr(self):
        return self._stderr


def _run_child(args):
    if args.candidates < 1 or args.warmups < 3 or args.repeats < 1:
        raise ValueError("Need candidates >= 1, warmups >= 3, repeats >= 1.")

    import jax
    import jax.numpy as jnp
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_de_map as de

    model, target, master, init_params = _build_target(
        de, args.galaxy, args.spot_batch, args.seed,
        phi_integration=args.phi_integration)
    points, names, lo, hi, candidate_source = _checkpoint_points(
        de, model, target, master, args.galaxy, args.candidates, args.seed)
    n_dev, devices = de._resolve_n_devices(args.n_devices)
    if n_dev != args.n_devices:
        raise RuntimeError(
            f"Requested {args.n_devices} devices, found {n_dev}.")

    sampler = _NvidiaSmiSampler() if args.trace_memory else None
    fixed_score_seconds = None
    fixed_score_logp = None
    fixed_score_trace = None
    if sampler is not None:
        sampler.start()
        atexit.register(sampler.stop)

    logp = de._make_logp(target, names)
    lo_jax = jnp.asarray(lo)
    scale_jax = jnp.asarray(hi - lo)

    def fitness_one(x_normed):
        return -logp(lo_jax + x_normed * scale_jax)

    if model.phi_integration == "fixed-grid":
        if args.candidate_wave not in (None, 1):
            raise ValueError(
                "fixed-grid benchmarking requires candidate-wave 1")
        candidate_wave = de._de_candidates_per_wave(model)
    else:
        candidate_wave = de._de_candidates_per_wave(
            model, args.candidate_wave)
    evaluate = de._make_batched_fitness(
        fitness_one, n_dev, devices,
        candidates_per_wave=candidate_wave)
    if args.include_fixed_score:
        if sampler is not None:
            sampler.reset()
        fixed_point = de._normalise_theta_point(
            init_params, names, lo, hi)[None]
        start = time.perf_counter()
        fixed_fitness = np.asarray(evaluate(jnp.asarray(fixed_point)))
        fixed_score_logp = -float(fixed_fitness[0])
        fixed_score_seconds = time.perf_counter() - start
        if sampler is not None:
            fixed_score_trace = sampler.snapshot()
        print("BENCHMARK_FIXED_SCORE "
              f"seconds={fixed_score_seconds:.6f} "
              f"logp={fixed_score_logp:.9f} "
              f"trace={json.dumps(fixed_score_trace, sort_keys=True)}",
              flush=True)

    points_jax = jnp.asarray(points)
    reference = None
    warmup_seconds = []
    timed_seconds = []
    max_abs_difference = 0.0

    print(f"BENCHMARK_CONFIG galaxy={args.galaxy} "
          f"phi_integration={model.phi_integration} "
          f"spot_batch={_spot_batch_label(args.spot_batch)} "
          f"candidates_per_gpu_wave={candidate_wave} "
          f"device_block_size={de._DEVICE_LOCAL_BLOCK_SIZE} "
          f"candidates={args.candidates} warmups={args.warmups} "
          f"repeats={args.repeats} devices={devices}", flush=True)
    print(f"BENCHMARK_CANDIDATES {candidate_source}", flush=True)
    print("BENCHMARK_GRIDS "
          f"n_phi_hv_high={model._n_phi_hv_high} "
          f"n_phi_hv_low={model._n_phi_hv_low} "
          f"n_phi_sys={model._n_phi_sys} "
          f"n_r_local={model._n_r_local} "
          f"n_r_global={model._n_r_global} "
          f"n_refine_steps={model._n_refine_steps}", flush=True)
    if model.phi_integration == "peak-partition":
        print("BENCHMARK_PARTITION "
              f"n_phi_partition_sys={model._n_phi_partition_sys} "
              f"n_phi_partition_hv={model._n_phi_partition_hv} "
              f"root_capacity={model._phi_partition_root_capacity}",
              flush=True)
    geometry = _memory_geometry(
        model, 8 if jax.config.jax_enable_x64 else 4)
    print("BENCHMARK_MEMORY_GEOMETRY="
          + json.dumps(geometry, sort_keys=True), flush=True)

    if sampler is not None:
        sampler.reset()

    for index in range(args.warmups + args.repeats):
        start = time.perf_counter()
        values = np.asarray(evaluate(points_jax))
        elapsed = time.perf_counter() - start
        if reference is None:
            reference = values.copy()
        else:
            finite = np.isfinite(reference) & np.isfinite(values)
            if np.any(finite):
                max_abs_difference = max(
                    max_abs_difference,
                    float(np.max(np.abs(reference[finite] - values[finite]))))
            if not np.array_equal(
                    np.isfinite(reference), np.isfinite(values)):
                raise RuntimeError(
                    "Finite-value mask changed between repeats.")
        phase = "warmup" if index < args.warmups else "timed"
        phase_index = (index + 1 if phase == "warmup"
                       else index - args.warmups + 1)
        (warmup_seconds if phase == "warmup" else timed_seconds).append(
            elapsed)
        print(f"BENCHMARK_PASS phase={phase} index={phase_index} "
              f"seconds={elapsed:.6f} "
              f"finite={int(np.isfinite(values).sum())}", flush=True)

    if hasattr(evaluate, "device_profile"):
        profile = evaluate.device_profile()
    else:
        # The production single-device evaluator intentionally has no
        # multi-device balancing state. Populate the common reporting fields
        # from the last completed pass so suite output remains comparable.
        profile = {
            "assignment_weights": np.ones(1),
            "profile_weights": np.ones(1),
            "last_seconds": np.asarray([warmup_seconds[-1]
                                        if not timed_seconds
                                        else timed_seconds[-1]]),
            "profile_samples": 0,
            "block_size": 1,
            "rebalance_attempts": 0,
            "rebalances": 0,
            "last_rebalance_gain": 0.0,
        }
    evaluator_trace = sampler.snapshot() if sampler is not None else None
    if sampler is not None:
        sampler.stop()
        atexit.unregister(sampler.stop)
        if sampler.stderr:
            print(f"BENCHMARK_NVIDIA_SMI_WARNING {sampler.stderr}",
                  flush=True)
    median_seconds = float(np.median(timed_seconds))
    values_sha256 = hashlib.sha256(
        np.ascontiguousarray(reference).tobytes()).hexdigest()
    result = {
        "status": "ok",
        "galaxy": args.galaxy,
        "phi_integration": model.phi_integration,
        "candidate_source": candidate_source,
        "spot_batch": args.spot_batch,
        "candidates_per_gpu_wave": candidate_wave,
        "candidates": args.candidates,
        "warmup_seconds": warmup_seconds,
        "timed_seconds": timed_seconds,
        "median_seconds": median_seconds,
        "candidates_per_second": args.candidates / median_seconds,
        "assignment_weights": profile["assignment_weights"].tolist(),
        "profile_weights": profile["profile_weights"].tolist(),
        "last_device_seconds": profile["last_seconds"].tolist(),
        "profile_samples": int(profile["profile_samples"]),
        "device_block_size": int(profile["block_size"]),
        "rebalance_attempts": int(profile["rebalance_attempts"]),
        "rebalances": int(profile["rebalances"]),
        "last_rebalance_gain": float(profile["last_rebalance_gain"]),
        "max_abs_difference": max_abs_difference,
        "values_sha256": values_sha256,
        "values": np.asarray(reference).tolist(),
        "fixed_score_seconds": fixed_score_seconds,
        "fixed_score_logp": fixed_score_logp,
        "fixed_score_trace": fixed_score_trace,
        "evaluator_trace": evaluator_trace,
        "memory_geometry": geometry,
        "device_memory": _device_memory(devices),
    }
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.suite and args.child:
        raise SystemExit("--suite and --child are mutually exclusive.")
    if args.suite:
        _run_suite(args)
    else:
        _run_child(args)


if __name__ == "__main__":
    main()
