# Megamaser Scripts

Supported megamaser workflow:

- `run_maser.py`: unified megamaser runner. It defaults to `--sampler mcmc` (the explicit-latent `(r_ang, phi)` NUTS chain); use `--sampler de` for the 2D-marginal differential-evolution MAP (delegates to `run_de_map.py`), the global search used to seed the MCMC.
- `run_de_map.py`: 2D-marginal MAP optimiser. Per spot it marginalises `(r_ang, phi)` jointly on a conditional per-spot r-grid (`_build_conditional_r_grids` + `_sum_phi_marginal`; not phi at a profiled `r_ang`, which overfits `D_A`) and optimises the globals with differential evolution. The phi/r grid is read from the per-galaxy `config_maser.toml` (same grid as the MCMC and convergence checks). Reached via `run_maser.py --sampler de`.
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

The production default remains classic DE.  Two opt-in adaptive modes are
available for validation:

```bash
python scripts/megamaser/run_maser.py NGC6264 --sampler de \
    --de-algorithm lshade
python scripts/megamaser/run_maser.py NGC6264 --sampler de \
    --de-algorithm hybrid --adam-steps 200
```

`lshade` uses success-history mutation/crossover adaptation and linearly
reduces the population.  `hybrid` additionally applies Adam to a small set of
diverse elites using independently reshuffled, stratified spot batches.  Adam
never changes the integration grids, and its endpoints enter the DE population
only after improving the exact all-spot objective.  Every exact optimiser
proposal is deduplicated in a SQLite sidecar next to the algorithm-specific DE
checkpoint, and `--resume` restores both.  GPU runs continue to split exact
population evaluations over the visible devices with `pmap`; pass
experimental runner options after `--` when using `submit.sh`, for example:

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC6264 \
    --sampler de --gpu-count 4 -- \
    --de-algorithm hybrid --min-pop-size 16 --adam-steps 200
```

For GPU jobs, `--gpu-count N` requests N GPUs on one node and `--cpus C`
means C CPU cores per GPU. If `--cpus` is omitted, GPU jobs use 4 CPU cores
per GPU. For example, `--gpu-count 8` requests 32 CPU cores by default, while
`--gpu-count 8 --cpus 2` requests 16 CPU cores. `--mem` remains GB per CPU.

Submit one joint H0 chain over several galaxies:

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b,NGC6264 --infer-H0 --selection redshift
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy all --infer-H0 --selection redshift --distance-prior volume
```

Common forwarded options are `--init-strategy median|config|reid`, `--spot-batch`, `--f64`, `--add-ecc`, `--add-quadratic-warp`, and `--fix-floors-pesce`. `reid` uses reported Pesce/Reid globals, with NGC4258 read from `reid_ngc4258_best.toml`. MCMC quick overrides are `--num-warmup` and `--num-samples`; MCMC also accepts opt-in `--save-latents`, `--compare-reid`, `--match-reid`, and `--compare-reid-2x`. DE operational options are `--resume`, `--fix-globals`, and `--fix-globals-pesce`; pass adaptive-DE options after the `submit.sh` `--` separator. Submit single-galaxy evidence separately with `submit.sh --evidence` after the chain exists. Joint H0 accepts `--distance-prior distance|volume`; selection runs require the volume prior. The joint H0 run (`--infer-H0`) uses the matching saved per-galaxy `samples/D_A` chains (legacy `samples/D_c` chains are converted to D_A) as KDE distance likelihoods and prints source/support-edge diagnostics. Sampler, optimiser, and model defaults live in `config_maser.toml`.

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
