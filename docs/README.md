# Megamaser Scripts

Supported megamaser workflow:

- `run_maser.py`: unified megamaser runner. It defaults to `--sampler mcmc` (the explicit-latent `(r_ang, phi)` NUTS chain); use `--sampler de` for the 2D-marginal differential-evolution MAP (delegates to `run_de_map.py`), the global search used to seed the MCMC.
- `run_de_map.py`: 2D-marginal MAP optimiser. Per spot it marginalises `(r_ang, phi)` jointly on a conditional per-spot r-grid (`_build_conditional_r_grids` + `_sum_phi_marginal`; not phi at a profiled `r_ang`, which overfits `D_A`) and optimises the globals with differential evolution. Phi integration defaults to the fixed grids in `config_maser.toml`; DE can opt into the fixed-shape peak-partition routine with `--phi-integration peak-partition`. Reached via `run_maser.py --sampler de`.
- `benchmark_de_batching.py`: fixed-candidate exact-likelihood benchmark for
  spot batching. Suite mode uses fresh child processes, bypasses SQLite, and
  reads candidates from the current compatible DE checkpoint, or uses a
  deterministic scrambled-Sobol fallback when no checkpoint exists.
- `run_joint_H0.py`: toy joint MCP megamaser H0 inference using KDE distance likelihoods from the saved single-galaxy MCMC chains.
- `submit.sh`: cluster/local submission helper for `--sampler mcmc`, `--sampler de`, or the toy joint H0 via `--infer-H0`.
- `check_reid/`: standalone comparison against the Reid-style parameterisation.

## Production Sampling

Run a single galaxy locally:

```bash
python scripts/megamaser/run_maser.py NGC5765b
```

Useful development flags:

```bash
python scripts/megamaser/run_maser.py NGC6264 \
    --num-warmup 100 --num-samples 100
```

Use `run_maser.py --help` for MCMC options and
`run_maser.py --sampler de --help` for the DE MAP options.

```bash
python scripts/megamaser/run_maser.py NGC6264 --sampler mcmc \
    --num-warmup 100 --num-samples 100
```

## Cluster Jobs

Submit production MCMC jobs:

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b --sampler mcmc
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b,NGC6264 --sampler mcmc
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy all --sampler mcmc
```

`--galaxy all` expands to `CGCG074-064,NGC5765b,NGC6264,NGC6323,UGC3789`.
It deliberately excludes `NGC4258`; submit `NGC4258` explicitly for
single-galaxy MCMC/DE, not joint H0.

Submit DE MAP jobs (the global search to seed MCMC):

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b --sampler de
```

DE now always uses L-SHADE: current-to-pbest/1 mutation, success-history
adaptation of the mutation and crossover rates, a displaced-parent archive,
and linear population reduction.  Population reduction follows the number of
DE-population fitness evaluations, not the generation counter.  Consequently,
raising `max_generations` as a safety ceiling does not slow the reduction
schedule.  The production population is 2000 -> 128 over 3,400,000 such
evaluations; after that it remains at 128 until patience or the generation
ceiling stops the run.  The 3,400,000 value is only the population-reduction
horizon, not an NFE limit: evaluations continue beyond it.  There is no
classic/hybrid selector and no Adam polishing path.

The initial population contains only the data-derived ridge and scrambled
Sobol points.  The Pesce/Reid point is never inserted, including through the
DE initialisation strategy.  Its exact all-spot unnormalised log posterior
density is still printed as an independent reference and is scored through the
same compiled DE objective rather than a separate startup executable.  (A
single point has zero probability mass in a continuous posterior.) Runs use
the explicit
`*_lshade_nopesce.npz` checkpoint plus a SQLite exact-evaluation sidecar, and
`--resume` restores both without accepting an older seeded or
generation-scheduled checkpoint.

Every L-SHADE proposal is evaluated with the exact all-spot objective and
deduplicated in the sidecar.  Spot batching remains allowed because it is an
exact sum. The five standard float32 galaxies (`CGCG074-064`, `NGC5765b`,
`NGC6264`, `NGC6323`, and `UGC3789`) default to true all-spots evaluation;
an explicit `--spot-batch` or per-galaxy setting still overrides this. The f64
`NGC4258` path retains its configured/planned spot batching. GPU runs use
one shared `pmap` executable when the same GPU model supplies every device, so
cold startup compiles the objective once rather than once per GPU. Phi
integrands are materialised before their log-sum reductions so XLA does not
build the very slow fused reduction kernels seen on the 60,001-point NGC4258
grid. Each device runs immutable eight-candidate blocks, so population
shrinkage cannot trigger new input shapes. Fixed-grid entries remain
sequential to cap memory; peak-partition evaluates all eight concurrently to
fill the GPU with its smaller working set. Peak-partition accepts
`--peak-candidates-per-wave 1|2|4|8` for card-specific throughput calibration;
the setting changes only batching, not the objective. Heterogeneous devices
retain concurrent device-local JITs, learn bounded per-device throughput weights, and
adopt a weighted assignment only when its block-aware predicted makespan
improves by at least 2%. Fixed-grid candidate batching remains an implementation
invariant. Padding is evaluated but
excluded from the archive and algorithmic NFE count.
The SQLite sidecar persists deterministic 64-bit
fingerprints, so a resume loads the compact fingerprint table instead of every
full point key.  Possible matches are still verified against the complete BLOB
key, preserving exact cache semantics even under a fingerprint collision.
Checkpoint logs report exact-evaluation, trial-generation, archive lookup/write,
device balance, update, and checkpoint timings.  Pass budget overrides after
`--` when using `submit.sh`, for example:

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC6264 \
    --sampler de --gpu-count 4 --cpus 2 -- \
    --population-reduction-evaluations 3400000 \
    --max-generations 5000 --patience 500
```

For a development-only batching benchmark inside an existing two-GPU
allocation, run:

```bash
python scripts/megamaser/benchmark_de_batching.py UGC3789 --suite \
    --candidates 512 --n-devices 2
```

The benchmark reports compile/adaptation passes separately from steady timed
passes. It does not change the checkpoint or exact-evaluation sidecar. To
trace the production-like fixed/config score and DE evaluator at 50 ms
resolution, while isolating allocator state in one fresh process per setting,
use for example:

```bash
python scripts/megamaser/benchmark_de_batching.py UGC3789 --suite \
    --spot-batches 8,16,34,68,all,68 \
    --candidates 128 --warmups 3 --repeats 1 --n-devices 2 \
    --trace-memory --include-fixed-score
```

The JSON records both JAX live/peak bytes and sampled `nvidia-smi` resident
memory. It also reports the raw grid geometry by spot class. For scan-based
spot batching, the first-order live-array scale is
`spot_batch * n_r * n_phi * dtype_bytes`. The truly unbatched path selected by
`--spot-batch all` can have a different, more strongly fused XLA memory plan,
so benchmark it directly rather than extrapolating a scan-batch fit. Resident
memory can also jump in allocator buckets; use the sampled peak when
establishing a card-specific limit.

For GPU jobs, `--gpu-count N` requests N GPUs and `--cpus C` means C CPU cores
per GPU. Use `--cpus 2` for multi-GPU Glamdring jobs with the default 7 GB per
CPU: two RTX 2080 Ti GPUs then request 28 GB total and four RTX 3090 GPUs 56
GB, fitting one node. The generic omitted default is 4 CPU cores per GPU; with
7 GB per CPU that can force the scheduler to spread a nominal multi-GPU job
across nodes, where one JAX process cannot use the remote GPUs. `--mem` remains
GB per CPU.

Submit one joint H0 chain over several galaxies:

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b,NGC6264 --infer-H0 --selection redshift
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy all --infer-H0 --selection redshift --distance-prior volume
```

Common forwarded options are `--spot-batch`, `--f64`, `--add-ecc`, `--add-quadratic-warp`, and `--fix-floors-pesce`. DE accepts `--init-strategy median|config`; MCMC also accepts `reid`, which uses reported Pesce/Reid globals (NGC4258 reads `reid_ngc4258_best.toml`). MCMC quick overrides are `--num-warmup` and `--num-samples`; MCMC also accepts opt-in `--save-latents`, `--compare-reid`, `--match-reid`, and `--compare-reid-2x`. DE operational options are `--resume`, `--fix-globals`, and `--fix-globals-pesce`; pass DE budget overrides after the `submit.sh` `--` separator. Submit single-galaxy evidence separately with `submit.sh --evidence` after the chain exists. Joint H0 accepts `--distance-prior distance|volume`; selection runs require the volume prior. The joint H0 run (`--infer-H0`) uses the matching saved per-galaxy `samples/D_A` chains (legacy `samples/D_c` chains are converted to D_A) as KDE distance likelihoods and prints source/support-edge diagnostics. Sampler, optimiser, and model defaults live in `config_maser.toml`.

Automatic retries use the watcher wrapper. The `--max-retries` shortcut
launches the watcher in a detached `screen`/`tmux` session and prints the
reattach command and log path.

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy all --sampler de --max-retries 4

bash scripts/megamaser/watch_and_resubmit.sh --marker "MAP init" -- \
    bash scripts/megamaser/submit.sh -q cmbgpu --galaxy all --sampler de

bash scripts/megamaser/watch_and_resubmit.sh --marker "saved samples to" -- \
    bash scripts/megamaser/submit.sh -q cmbgpu --galaxy all --sampler mcmc
```

Short galaxy aliases are accepted by `submit.sh`: `5765b`, `6264`, `6323`, `3789`, and `4258`.

MCMC jobs also write `*_log_density.png` beside the HDF5 and corner plots.
Pass `--compare-reid` to additionally print the slow compact
Pesce/Reid-reported/config/MCMC-median comparison table scored with the same
2D marginal disk likelihood used by the DE objective; per-spot
`(r_ang, phi)` latents are integrated out. Add `--compare-reid-2x` for the
2x-denser-grid logZ check. Submit harmonic evidence separately with
`submit.sh -q short --galaxy NGC6323 --evidence --init-strategy config`.

## Configuration

The main config is `config_maser.toml`.

The DE objective uses `phi_integration = "fixed-grid"` by default. To use the
GPU-shaped peak-partition quadrature, pass:

```bash
./scripts/megamaser/submit.sh -q short --sampler de --galaxy NGC4258 \
    --gpu-mem 32 --gpu-count 2 --spot-batch 47 \
    --phi-integration peak-partition
```

This mode always searches the two systemic half-planes independently using
513 nodes per half-plane, and uses 257 nodes for each red/blue half-plane. It
then locates extrema from neighbouring likelihood values, refines all fixed-size
brackets in parallel, and integrates peak/tail partitions. All 128 global
radii are scanned concurrently, and radius-only position, velocity, and
acceleration terms are precomputed once. The global-radius scan uses a
three-point log-radius interpolation for the local-grid centre, avoiding the
former nested 32-step radial Brent solve. Its phi marginals are reused in the
final local/global union, including for eccentric models. On float32 paths,
the global extrema also seed the 256 local-radius nodes; a 65-node guard and a
fixed 256-pair full-scan fallback retain the original result when interpolation
is unsafe. Float64 retains the full local half-plane scans: NGC4258 tests found
that extrema reuse could otherwise shift very sharp eccentric likelihoods by
more than float64 rounding. Peak-partition v3 therefore rejects older peak
checkpoints rather than mixing objective values.
GPU jobs retain the persistent JAX compilation cache; CPU runs keep the
conservative cache-disable guard after an earlier PjRt deserialisation failure.
Its GPU memory planner is not yet calibrated, so explicit `--spot-batch` and
the per-galaxy config remain authoritative. To compare candidate concurrency
on a particular GPU, rerun with `--peak-candidates-per-wave 2`, `4`, and `8`
and compare the steady Sobol `cand/s`; 8 remains the default.

Important explicit-latent controls:

```toml
phi_sys_ranges_deg = [[-180, 180]]
```

MCMC samples non-centred log-radius residuals,
`z_r = log(r_ang / r_hat(theta))`. Per-spot `r_ang`/`phi` samples are not
written by default; pass `--save-latents` to keep them in the HDF5 output.
The DE MAP marginalises `(r_ang, phi)` rather than sampling them.

The default mass coordinate is `mass_parameterization = "eta"`, i.e. `eta = log_MBH - log10(D_A)`. Saved samples still include derived `log_MBH`, and the original `log_MBH` prior is applied to that derived value. Change `mass_parameterization` in `config_maser.toml` to sample `log_MBH` directly. The single-galaxy megamaser distance is always sampled as uniform `D_A` over the configured distance bounds (no flag).

## Reid Likelihood Queries

`check_reid/reid_profile.py` evaluates Mark Reid's unmodified `fit_disk` likelihood through the local f2py wrapper.  The data file and init TOML are the inputs; per-spot `(r, phi)` latents are MAP-profiled for each query.

```bash
python scripts/megamaser/check_reid/reid_profile.py \
    --init reid_ngc4258_best.toml \
    --galaxy NGC4258 \
    --data data/Megamaser/N4258_disk_data_MarkReid.final \
    --set H0=63.0 \
    --json-out /tmp/reid_query.json
```

If a global is not reported, provide a starting value in the init TOML and pass it to `--map-globals`, for example `--map-globals "H0 sigma_vhv_km_s"`.
