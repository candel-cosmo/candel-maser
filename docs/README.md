# Megamaser Scripts

Supported megamaser workflow:

- `run_maser.py`: unified megamaser runner. It defaults to `--sampler mcmc` (the explicit-latent `(r_ang, phi)` NUTS chain); use `--sampler de` for the 2D-marginal differential-evolution MAP (delegates to `run_de_map.py`), the global search used to seed the MCMC.
- `run_de_map.py`: 2D-marginal MAP optimiser. Per spot it marginalises `(r_ang, phi)` jointly on a conditional per-spot r-grid (`_build_conditional_r_grids` + `_sum_phi_marginal`; not phi at a profiled `r_ang`, which overfits `D_A`) and optimises the globals with differential evolution. Phi integration defaults to the GPU-shaped peak-partition routine in `config_maser.toml`; the legacy fixed grids remain selectable with `--phi-integration fixed-grid`. Reached via `run_maser.py --sampler de`.
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
schedule.  The production population is 2000 -> 1024 over 5,000,000 such
evaluations; after that it remains at 1024 until patience or the generation
ceiling stops the run.  The 5,000,000 value is only the population-reduction
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
    --population-reduction-evaluations 5000000 \
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
    --phi-integration peak-partition --candidate-wave 8 \
    --spot-batches 8,16,34,68,all,68 \
    --candidates 128 --warmups 3 --repeats 1 --n-devices 2 \
    --trace-memory --include-fixed-score
```

The JSON records both JAX live/peak bytes and sampled `nvidia-smi` resident
memory. It also reports method-aware first-order scan geometry by spot class:
the fixed method uses its local/global radial union and dense phi grid, while
peak partition reports the larger separate radial pass and its small
partition scan. Root refinement and quadrature create shorter-lived arrays,
so JAX `peak_bytes_in_use` remains the authoritative whole-executable number.
For scan-based spot batching, the first-order live-array scale is
`spot_batch * n_r * n_phi * half_planes * dtype_bytes` (two independent
half-planes for systemic peak partition, one otherwise). The truly unbatched path selected by
`--spot-batch all` can have a different, more strongly fused XLA memory plan,
so benchmark it directly rather than extrapolating a scan-batch fit. Resident
memory can also jump in allocator buckets; use the sampled peak when
establishing a card-specific limit.

`--phi-integration peak-partition` permits a head-to-head benchmark before a
galaxy is switched in the TOML configuration. `--candidate-wave 1|2|4|8`
overrides only the peak-partition GPU wave width; omitting it uses the same
production policy as `run_de_map.py` (currently eight for peak partition and
one for fixed grid). Every suite child reports the resolved integrator and
wave width alongside the output digest. Because different exact spot tilings
can change float32 reduction order, the suite also reports cross-tiling maximum
and RMS differences plus finite-mask mismatches; repeated copies of the same
tiling retain the stricter bitwise digest check.

For GPU jobs, `--gpu-count N` requests N GPUs and `--cpus C` means C CPU cores
per GPU. Omitting `--cpus` defaults to two cores per GPU. With the default 7 GB
per CPU, two RTX 2080 Ti GPUs request 28 GB total and four RTX 3090 GPUs 56 GB,
fitting one node. `--mem` remains GB per CPU.

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

The all-galaxy validated DE default is `phi_integration = "peak-partition"`.
The legacy dense fixed grid remains available for controlled comparisons:

```bash
./scripts/megamaser/submit.sh -q short --sampler de --galaxy NGC4258 \
    --gpu-mem 32 --gpu-count 2 --spot-batch 47 \
    --phi-integration fixed-grid
```

Peak partition searches the two systemic half-planes independently using
129 nodes per half-plane, and uses 65 nodes for each red/blue half-plane;
validated per-galaxy overrides can raise either scan independently. It
then locates extrema from neighbouring likelihood values, refines all fixed-size
brackets in parallel, and integrates peak/tail partitions. All 128 global
radii span the full physical support and are scanned concurrently; radius-only
position, velocity, and acceleration terms are precomputed once. A three-point
log-radius interpolation supplies the local-grid centre, avoiding the former
nested 32-step radial Brent solve and its width searches. Optional fixed-shape
value-only radial stencils are off by default. The
`peak_r_refine_hv_only` selector can restrict an enabled stencil to red/blue
spots when population diagnostics show that systemic radii do not need it;
the numerical objective policy records that scope.
The global phi marginals are reused in the final local/global union, including
for eccentric models. Independent left and right local-radius spans retain
useful quadrature support when a seed is close to a physical boundary. At every
precision, every local-radius node performs a fresh per-galaxy half-plane
scan; local phi-extrema are not reused.  The standard local grid has 256
nodes, with validated radial overrides listed below.
Circular disks use the structural four-root capacity, while eccentric disks
retain eight because their rational velocity factor breaks the circular
trigonometric-polynomial bound. A capacity overflow falls back to the already
computed scan trapezoid, producing a finite poor-fit objective while retaining
an explicit overflow diagnostic. Peak-partition v6 and fixed-grid radial-policy
v2 therefore reject older checkpoints rather than mixing objective values.
For a galaxy whose radial likelihood is exceptionally narrow, the optional
peak-radius path first narrows a log-radius bracket with fixed value-only
stencils, takes a guarded three-point quadratic vertex, and can solve the two
`Delta logL = K_sigma^2 / 2` half-widths by fixed-count bisection. The width
solve is batched over spots and both directions, uses no derivatives or
circular-orbit formula, and is therefore also valid for eccentric models.
GPU jobs retain the persistent JAX compilation cache; CPU runs keep the
conservative cache-disable guard after an earlier PjRt deserialisation failure.
Explicit `--spot-batch` and the per-galaxy config remain authoritative. To
compare candidate concurrency
on a particular GPU, rerun with `--peak-candidates-per-wave 2`, `4`, and `8`
and compare the steady Sobol `cand/s`; 8 remains the default.

The checked-in profiles use 97/49 scans for CGCG074-064, NGC5765b, and
UGC3789; NGC6264 and NGC6323 retain 129/65.  NGC5765b uses a centred
321-node local-radius grid.  UGC3789 uses 384 local nodes and a
`scan_width_drop = 50` support envelope.  NGC4258 uses 513/65 scans, three
all-class radial refinements, an eight-step value-only width solve, and a
32-spot tile.  NGC6264 also uses a 32-spot tile; the other float32 DE targets
use all spots.  The complete circular/eccentric validation, pathological-point
audit, sustained GPU speedups, memory measurements, and caveats are in
`docs/notes/megamaser_phi_integration_all_galaxies.md`; the preceding
algorithm-development study is
`docs/notes/megamaser_phi_integration_research.md`.

### Fixed-grid versus peak-partition validation

`convergence/validate_phi_partition.py` compares the two production phi
integrators, `fixed-grid` and `peak-partition`, inside the complete
conditional-radius likelihood. It constructs the production DE target,
parameter layout, constrained bounds, variant-specific configured point, and
Pesce/Reid point through `run_de_map.py`; it does not implement a second
physical likelihood. The default candidate set is 4 scrambled Sobol points
from the DE box plus the config and Pesce/Reid anchors.

Both production methods are compared per spot with the same independent
float64 reference. That reference integrates a log-uniform radial grid over
the full physical support and a uniform phi grid over the matching physical
half-plane support. Its default paired convergence sequence is
`5001 x 2501`, `10001 x 5001`, and `20001 x 10001` radial-by-phi nodes;
NGC4258 instead uses `20001 x 50001`, `40001 x 50001`,
`80001 x 50001`, and `160001 x 50001`. Individual-spot refinement shows that
NGC4258's former failure was radial: 50001 phi nodes are converged, while the
uniform radial grid enters its asymptotic regime only beyond 40001 nodes.
The default tail-three gate therefore requires the final two radial
transitions to pass without weakening any tolerance. Explicit reference-level
flags override these defaults. The radial axis is chunked to control memory.
Production uses the configured galaxy precision; NGC4258 remains forced to
float64.

Submit the default circular validation for every configured galaxy:

```bash
bash scripts/megamaser/convergence/validate_phi_partition.sh -q cmbgpu
```

Restrict the run to one galaxy with `--galaxies`:

```bash
bash scripts/megamaser/convergence/validate_phi_partition.sh -q cmbgpu \
    --galaxies NGC6323
```

On a laptop, use explicit local mode. This permits the CPU backend; restricting
the run to one galaxy is recommended for the first check:

```bash
bash scripts/megamaser/convergence/validate_phi_partition.sh --local \
    --galaxies NGC6323
```

The terminal, JSON, and Markdown outputs report the configured phi-integration
default separately from both tested methods. They show the fixed-grid phi
sizes, peak-partition scan sizes, production conditional-radius grid, the
paired dense reference grids, each method minus the reference, each method
relative to its Pesce/Reid likelihood, and peak-partition minus fixed-grid.
Candidate tables use one row per point and show the finest reference
log-likelihood, total absolute error, worst-spot absolute error, and a
per-candidate `eval s` column timing the pure production objective (one warm
evaluation, median of `timing_repeats`) for each method; method-dependent
values are printed as `fixed-grid | peak-partition`. Reference-convergence
failures and root-capacity overflows are listed below the table.
The global config default is `peak-partition`; the validator still constructs
and compares both production methods without mutating the checked-in config.

To validate the actual best point from a completed DE run, add its checkpoint
as a candidate. The loader requires identical parameter names, sizes, bounds,
and peak-objective policy before converting the saved unit-box point. The
archived NGC4258 reconnaissance below predates v6, so its coordinate is an
explicit cross-policy experiment:

```bash
bash scripts/megamaser/convergence/validate_phi_partition.sh -q cmbgpu \
    --galaxies NGC4258 --sobol-candidates 0 --no-config-point \
    --no-pesce-point \
    --checkpoint-candidate \
    results/Megamaser/de_checkpoints/NGC4258/de_ckpt_rmap_peakpartition_lshade_nopesce.npz \
    --allow-checkpoint-policy-mismatch
```

For a deliberate cross-policy numerical experiment, add
`--allow-checkpoint-policy-mismatch`. This only permits the saved coordinate
to be rescored under the tested objective; layout, bounds, unit-box finiteness,
and saved-fitness finiteness remain hard checks, and both policy strings plus
the mismatch are written to the report. Normal checkpoint loading remains
strict.

Every candidate is judged against the full dense float64 reference ladder.
On a cache miss every level is computed; an exact cache hit loads the complete
ladder instead. Production compilation, timing, and both production integrator
evaluations are intentionally rerun, so seeing `Timing both production methods`
or `Evaluating ...` does not imply that the dense reference is being recomputed.
The terminal prints `Reference cache HIT ... dense ladder skipped` before a
cached result is used. Broad Sobol points from the DE box can still land
on needle-like posterior geometry whose dense reference never converges (one
trapezoid node holds the whole integral), even though the point is tens of
thousands of nats below the config anchor and thus posterior-irrelevant. The
gate and railing machinery is a pure classifier: after every candidate is
fully evaluated, the converged anchors (config, Pesce/Reid) and local clouds
calibrate a per-case deficit gate at 100x the worst legitimate deficit, floored
at 1000 nats. A broad Sobol point is flagged posterior-irrelevant when both its
production and finest-reference log-likelihoods fall below that gate and its
geometry rails (closed-form radius seeds clipped to the support edge, or
red/blue high-velocity phi peaks argmaxed at the phi=0 boundary). Flagged
points are fully reported with their real reference log-likelihood and error
columns, marked `[IRRELEVANT]`, and their failures -- unconverged references or
comparison mismatches -- are excused from the verdict and listed separately
from hard failures. They never count toward the case or overall PASS/FAIL in
either direction. A candidate that takes the full ladder and genuinely FAILs
without being flagged irrelevant still fails the case: excusal is reserved for
posterior-irrelevant needle geometry, never for integrator mismatches on
relevant points. Classification is disabled for a case when no anchor reference
converges, in which case nothing is flagged. The headline reads e.g. `PASS
(K irrelevant: posterior-irrelevant geometry, excused)`.

The default all-galaxy run is intentionally expensive: it evaluates 6 points
per galaxy (4 Sobol + config + Pesce/Reid) at three very large 2D reference
grids. Run one galaxy first when calibrating memory and wall time.

Submit a lightweight NGC6323 GPU smoke test with only the config point and
reduced reference grids:

```bash
bash scripts/megamaser/convergence/validate_phi_partition.sh -q cmbgpu -- \
    --galaxies NGC6323 --variants circular \
    --sobol-candidates 0 --no-pesce-point \
    --reference-r-levels 501,1001,2001 \
    --reference-phi-levels 251,501,1001 \
    --reference-spot-batch 2 --timing-repeats 2
```

Then exercise the forced-float64 eccentric NGC4258 path with a reduced
candidate set:

```bash
bash scripts/megamaser/convergence/validate_phi_partition.sh -q cmbgpu \
    --mem 32 -- \
    --galaxies NGC4258 --variants eccentric --sobol-candidates 0 \
    --spot-batch 4 --reference-spot-batch 1 \
    --candidate-wave 2 --timing-repeats 2
```

For a quick laptop orchestration check, keep one config point and use small
grids; these are not acceptance-quality references:

```bash
bash scripts/megamaser/convergence/validate_phi_partition.sh --local \
    --galaxies NGC6323 --sobol-candidates 0 --no-pesce-point \
    --reference-r-levels 51,101,201 \
    --reference-phi-levels 51,101,201 --timing-repeats 1
```

Accuracy, ranking, overflow, and reference-convergence tolerances are explicit
command-line options shown by `--help`. Any unconverged reference, finite-mask
mismatch, excessive per-spot or total error, ranking inversion, or root
overflow makes the command exit non-zero.

Each run writes `validation.json` (all candidates, consecutive reference
levels, and every per-spot diagnostic) and `validation.md` (candidate,
population, convergence, ranking, worst-spot, aggregate, timing, memory, and
reproduction tables). Expensive reference arrays are cached by candidate,
galaxy, variant, paired grid sequence, chunking, dtype, physical configuration,
backend, and a hash of the model/reference sources. Production objective
policies, integration-node overrides, the git revision, and validator-only
reporting code are excluded because they cannot change the dense reference.
Entries expire after 48 hours; `--no-cache` disables reuse, while
`--clean-cache` deletes the cache and exits without running validation.

```bash
bash scripts/megamaser/convergence/validate_phi_partition.sh --clean-cache
```

### Numerical-setting experiments

The validator accepts repeatable, method-specific numerical overrides without
editing `config_maser.toml`. For example, test a cheaper peak scan for NGC6264
against the same candidates and cached references with:

```bash
bash scripts/megamaser/convergence/validate_phi_partition.sh --local \
    --galaxies NGC6264 \
    --scheme-setting peak-partition.n_phi_partition_sys=257 \
    --scheme-setting peak-partition.n_phi_partition_hv=129
```

The whitelist is shown below. Fixed-grid exposes its three phi node counts;
peak-partition exposes its two scan counts plus root refinement, radial
stencil/width, and quadrature controls; both expose the
config-backed conditional-radius controls. Overrides are recorded separately
for each method in the terminal, JSON, and Markdown reports. Keep the galaxy,
variant, seed, candidate options, and reference options fixed across trials.
An agent can treat a zero exit code and `passed: true` as the feasibility gate,
then rank feasible trials by the selected method's warmed throughput in
`cases[].timing`. Local CPU trials are suitable for accuracy and workflow
checks; final performance choices must be repeated on the production GPU.
The completed all-galaxy multi-loop benchmark, physical analysis, pass/fail
boundaries, and production recommendation are recorded in
`docs/notes/megamaser_phi_integration_all_galaxies.md`.

The production objective timing separates the cold compile/evaluation from
warmed throughput and records backend peak memory when JAX exposes it.

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
CANDEL config inputs are converted automatically using their `D_A` and active `eta`/`log_mbh` mass coordinate.  Moving a quadratic warp to Reid's data-derived reference radius preserves the complete polynomial by shifting both its intercept and linear coefficient.

```bash
python scripts/megamaser/check_reid/reid_profile.py \
    --init reid_ngc4258_best.toml \
    --galaxy NGC4258 \
    --data data/Megamaser/N4258_disk_data_MarkReid.final \
    --set H0=63.0 \
    --json-out /tmp/reid_query.json
```

If a global is not reported, provide a starting value in the init TOML and pass it to `--map-globals`, for example `--map-globals "H0 sigma_vhv_km_s"`.
