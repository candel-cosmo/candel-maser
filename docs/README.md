# Megamaser disk inference

Scripts for fitting the warped Keplerian disk model (`MaserDiskModel`) to
VLBI maser spot data in order to infer angular-diameter distances and
ultimately `H0`. Model code lives in `candel/model/model_H0_maser.py`;
data loaders in `candel/pvdata/megamaser_data.py`.

All settings (priors, sampler, grid sizes, per-galaxy overrides) live in
`config_maser.toml`. Most CLI flags below are overrides on top of that.

## Galaxies

- **MCP five** — CGCG074-064, NGC5765b, NGC6264, NGC6323, UGC3789. Run
  in `mode2` (r and φ both marginalised analytically). Default sampler
  is NSS (nested sampling, GPU).
- **NGC4258** — anchor galaxy. Position errors are ~10× tighter than
  the MCP five, so `mode2` (shared r, φ grids) cannot resolve the peaks.
  Runs in `mode1` (sample `r_ang` per spot, marginalise φ on dense
  per-type brute-force grids) with NUTS on GPU, or in mode2-like
  CPU+MPI (one spot per rank) via `run_mode2_mpi.py`.

## Python entry points

| Script | Purpose |
|---|---|
| `run_maser_disk.py` | Generic single-galaxy runner (NUTS or NSS). Driven by `config_maser.toml`. |
| `run_n4258_mode1.py` | NGC4258 Mode 1 NUTS with dense 100k φ grid (no ecc, no quadratic warp). Historical/dev. |
| `run_mode2_mpi.py` | CPU+MPI runner, one spot per rank, DE MAP or ultranest posterior. Built for NGC4258 where GPU memory can't hold the required grids. |
| `run_de_map.py` | DE MAP optimizer (mode2 only) for one galaxy, saves result to TOML. |
| `run_mock_maser.py` | Short single-mock closure test on one synthetic galaxy. |
| `run_mock_maser_disk.py` | Batch mock closure tests over many seeds, with NUTS and KS-style summary. |
| `toy_joint_H0.py` | Joint `H0` inference from per-galaxy NSS `D_c` posteriors via KDE + numpyro hierarchical model (with/without phenomenological selection). |

## Shell submission scripts

All submit through `addqueue` on glamdring. Both GPU and CPU/MPI jobs
use the single `venv_candel` (JAX with CUDA, CPU fallback when no GPU
is visible).

### `submit_all.sh` — MCP five (NSS or NUTS)

Loops over the five MCP galaxies (or a single `--galaxy`) and submits
one GPU job each.

```bash
bash scripts/megamaser/submit_all.sh --sampler nss                 # NSS, all five
bash scripts/megamaser/submit_all.sh --sampler nuts --num-chains 4 # NUTS, 4 vectorised chains
bash scripts/megamaser/submit_all.sh --sampler nss --galaxy NGC5765b
```

Options: `--mode {mode1|mode2}` (NSS requires mode2),
`--f-grid F` (grid-density scaling), `--init-method {config|median|sample}`,
`-q QUEUE` (default `gpulong`).

### `submit_ngc4258.sh` — NGC4258 NUTS

GPU NUTS job for NGC4258 specifically. Defaults to `mode1` with per-type
brute-force φ grids, full model (eccentricity + quadratic warp).

```bash
bash scripts/megamaser/submit_ngc4258.sh                                  # defaults
bash scripts/megamaser/submit_ngc4258.sh --warmup 5000 --samples 4000
bash scripts/megamaser/submit_ngc4258.sh --no-ecc --no-quadratic-warp     # circular + linear
```

Flags: `-q QUEUE` (default `optgpu`), `--warmup`, `--samples`,
`--init {config|median|sample}`, `--mode`, `--no-ecc`, `--no-quadratic-warp`.
NSS is *not* supported here (358 per-spot `r_ang` parameters); DE MAP
doesn't support mode1.

### `submit_mode2_mpi.sh` — CPU+MPI mode2 (primarily NGC4258)

Submits `run_mode2_mpi.py` via `addqueue -n 1x64` on the `redwood`
queue. Reads the real spot count from the data file (optionally capped
by `--max-spots`) and launches one rank per spot round-robin.

```bash
bash scripts/megamaser/submit_mode2_mpi.sh --galaxy NGC4258 --method de  # DE MAP
bash scripts/megamaser/submit_mode2_mpi.sh --galaxy NGC4258 --method ns  # ultranest posterior
bash scripts/megamaser/submit_mode2_mpi.sh --galaxy NGC4258 --method de --resume --out-dir RESULTS_DIR
```

Forwards DE tuning (`--de-popsize`, `--de-maxiter`, `--de-F`,
`--de-CR`, `--checkpoint-every`) and ultranest tuning
(`--ns-min-live`, `--ns-max-ncalls`, `--ns-stepsampler`, `--ns-nsteps`).
Exports the OpenMPI TCP/shm transport knobs required on glamdring
(OFI MTL crashes at init).

### `run_de_map.sh` — DE MAP on mode2 galaxies

GPU DE MAP optimizer for one or more mode2 galaxies (defaults to the
full mode2 set read from the config).

```bash
bash scripts/megamaser/run_de_map.sh                   # all mode2 galaxies
bash scripts/megamaser/run_de_map.sh NGC5765b UGC3789  # subset
bash scripts/megamaser/run_de_map.sh -q cmbgpu
```

### `toy_joint_H0.sh` — joint `H0` from saved per-galaxy posteriors

Submits `toy_joint_H0.py` with 1000 warmup + 4000 samples × 8 chains on
GPU. Produces a GetDist corner plot overlaying the with/without-selection
runs.

```bash
bash scripts/megamaser/toy_joint_H0.sh              # volumetric D^2 prior
bash scripts/megamaser/toy_joint_H0.sh --flat-dist  # flat D prior
```

### `python.sh`

Legacy `addqueue`-generated wrapper. Not actively used; kept only for
reference to the environment setup (IB locked-memory ulimit, PML/MTL
transport flags).

## Convergence tests

Shared-grid settings in `config_maser.toml` have been validated against
brute-force references. Scripts live in `convergence/`
(`convergence_grids.py`, `convergence_phi_marginal.py`, …). Summary:

- **Mode 2, MCP five** — all within 0.08 nats of a 10001² brute-force
  reference after Simpson HV + two-cluster systemic. Production grids
  are adequate.
- **NGC4258, Mode 2** — shared grids cannot resolve σ_φ ~ 0.001 rad
  peaks; error is ≈ -4200 nats, almost entirely from the systemic
  spots. This is why NGC4258 uses Mode 1 or mode2-MPI (per-spot grids).
- **NGC4258, Mode 1** — per-type brute-force φ grids converge to the
  dense reference when `r_ang` is sampled explicitly; NUTS handles the
  r exploration.

Reproduction commands live in the convergence scripts themselves.

## Current status (2026-04-21)

- **MCP five (NSS, mode2):** the production path. Per-galaxy posteriors
  feed `toy_joint_H0.py`, which shares `H0` across galaxies with a
  phenomenological selection model. Reproduces Pesce+2020-style `D_c`
  posteriors.
- **NGC4258:** still the most fragile piece.
  - GPU NUTS `mode1` runs cleanly, but currently converges on the
    inclination branch `i ≈ 94.9°` rather than Reid+2019's `i = 87.05°`
    (the near-edge-on symmetry is only weakly broken by eccentricity
    ≈ 0.007). D ≈ 7.51 Mpc vs Reid 7.58; periapsis offset by ~87°.
    Fix candidates: tight prior on i, sign check on the `periapsis`
    convention, dense-mass block over the warp parameters.
  - CPU+MPI mode2 runner (`run_mode2_mpi.py` + `submit_mode2_mpi.sh`)
    exists as an alternative that sidesteps GPU-memory limits by
    parallelising one spot per rank. Two MPI gotchas on glamdring are
    handled inside the submit script (OFI MTL → TCP/shm; ultranest's
    own MPI mode disabled on rank 0).
- **Joint `H0`:** `toy_joint_H0.py` produces a first-pass combined
  posterior from the per-galaxy NSS chains. Still a "toy" combiner —
  per-galaxy `D_c` posteriors are KDE'd and the volumetric prior divided
  out; a fully hierarchical joint fit is future work.
- **Paper draft:** `/mnt/users/rstiskalek/Papers/MMH0/main.tex`.

## Related docs

- `docs/maser_numerical_accuracy.md` — quadrature/grid accuracy notes.
- `docs/mode2_mpi_runner.md` — grid sizes, memory budget, full CLI for
  `run_mode2_mpi.py`.
- `instructions/maser_disk_jobs.md` — runner, config, submission.
- `instructions/glamdring_gpu_jobs.md` — queues and `addqueue` syntax.
