# Megamaser Scripts

This directory now keeps only the supported megamaser workflow:

- `run_maser_blackjax.py`: production BlackJAX collapsed-Gibbs sampler. The global disk block is sampled with BlackJAX, while each spot radius `r_ang` is updated by direct Gibbs/Metropolis steps and `phi` is marginalised numerically.
- `run_maser_disk.py`: compatibility wrapper that forwards supported options to `run_maser_blackjax.py`.
- `run_de_map.py`: differential-evolution MAP optimiser used for initialisation and deterministic conditional-`r_ang` diagnostics.
- `submit.sh`: cluster submission helper for `--sampler gibbs` or `--sampler de`.
- `toy_joint_H0.py` and `toy_joint_H0.sh`: retained toy post-processing combiner for saved per-galaxy distance posteriors.
- `convergence/`: numerical accuracy diagnostics for the `phi` and conditional-`r` integrals.
- `check_reid/`: standalone comparison against the Reid-style parameterisation.

## Production Sampling

Run a single galaxy locally:

```bash
python scripts/megamaser/run_maser_blackjax.py NGC5765b
```

Useful development flags:

```bash
python scripts/megamaser/run_maser_blackjax.py NGC6264 \
    --num-warmup 100 --num-samples 100 --n-sys 6 --n-red 7 --n-blue 7
```

The compatibility wrapper is still available for existing job templates:

```bash
python scripts/megamaser/run_maser_disk.py NGC5765b
```

Use `run_maser_blackjax.py --help` for the supported production options.

## Cluster Jobs

Submit production Gibbs jobs:

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b --sampler gibbs
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b,NGC6264 --sampler gibbs
```

Submit DE MAP jobs:

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b --sampler de
```

Common forwarded options are `--num-warmup`, `--num-samples`, `--n-inner`, `--seed`, `--spot-batch`, `--n-sys`, `--n-red`, `--n-blue`, `--f64`, `--no-progress`, `--no-ecc`, `--add-ecc`, `--no-quadratic-warp`, and `--add-quadratic-warp`.

## Configuration

The main config is `config_maser.toml`.

Important grid controls:

```toml
n_phi_hv_high = 5001
n_phi_hv_low = 2501
n_phi_sys = 5001
n_r_local = 256
n_r_global = 128
conditional_spot_batch = 16
```

`n_phi_*` controls the numerical `phi` marginalisation. `n_r_local`, `n_r_global`, and `conditional_spot_batch` are used by the DE initialiser and conditional-`r` diagnostics; production Gibbs sampling still samples `r_ang` directly.

The `[convergence.*]` blocks are kept because the diagnostic scripts use them for high-resolution numerical references.

## Numerical Diagnostics

The following scripts are intentionally retained:

```bash
python scripts/megamaser/convergence/convergence_phi_marginal.py --galaxies NGC6264 --no-grad
python scripts/megamaser/convergence/convergence_grids.py --galaxies NGC6264 --timing-attempts 0
python scripts/megamaser/convergence/test_mode2_grad_vs_numerical.py --galaxy NGC6264
python scripts/megamaser/convergence/r_ang_posteriors.py --galaxies NGC6264
python scripts/megamaser/convergence/diagnose_mode2_delta.py --galaxies NGC6264
```

Despite some historical filenames, these are now diagnostic-only checks for quadrature and gradients. They are not alternative megamaser samplers.
