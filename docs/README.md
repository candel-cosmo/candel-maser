# Megamaser Scripts

Supported megamaser workflow:

- `run_maser.py`: unified megamaser runner. It defaults to the BlackJAX collapsed-Gibbs sampler; use `--sampler de` for the differential-evolution MAP optimiser or `--sampler lbfgs` for the fast profiled L-BFGS optimiser.
- `submit.sh`: cluster submission helper for `--sampler gibbs`, `--sampler de`, or `--sampler lbfgs`.
- `toy_joint_H0.py` and `toy_joint_H0.sh`: toy post-processing combiner for saved per-galaxy distance posteriors.
- `convergence/`: numerical accuracy diagnostics for the `phi` and conditional-`r` integrals.
- `check_reid/`: standalone comparison against the Reid-style parameterisation.

## Production Sampling

Run a single galaxy locally:

```bash
python scripts/megamaser/run_maser.py NGC5765b
```

Useful development flags:

```bash
python scripts/megamaser/run_maser.py NGC6264 \
    --num-warmup 100 --num-samples 100 --n-sys 6 --n-red 7 --n-blue 7
```

Use `run_maser.py --help` for the default Gibbs options and
`run_maser.py --sampler de --help` for profiled MAP options.

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

Submit profiled L-BFGS MAP jobs:

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b --sampler lbfgs
```

Common forwarded options are `--seed`, `--spot-batch`, `--n-sys`, `--n-red`, `--n-blue`, `--f64`, `--no-ecc`, `--add-ecc`, `--no-quadratic-warp`, `--add-quadratic-warp`, and `--mass-parameterization eta|log_mbh`. Gibbs-specific options include `--num-warmup`, `--num-samples`, `--n-inner`, and `--no-progress`. DE-specific options include `--log2-N`, `--pop-size`, `--max-generations`, `--patience`, `--eval-chunk`, and `--log-every`. L-BFGS-specific options include `--lbfgs-maxiter`, `--lbfgs-ftol`, `--lbfgs-gtol`, `--lbfgs-maxls`, `--lbfgs-n-starts`, `--lbfgs-sobol-candidates`, `--lbfgs-start-strategy`, and `--lbfgs-jitter-scale`.

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

`n_phi_*` controls the numerical `phi` marginalisation. Production Gibbs sampling uses non-centred log-radius residuals, `z_r = log(r_ang / r_hat(theta))`, while saved samples remain in `r_ang`. `n_r_local`, `n_r_global`, and `conditional_spot_batch` are used by the profiled MAP objectives and conditional-`r` diagnostics. The DE initial population is always the best distinct points from a screened global scrambled-Sobol cloud; the configured init is not injected. The L-BFGS objective profiles `r_ang` with a stop-gradient radius map, so it is intended as a fast physical-MAP initialiser rather than a replacement posterior sampler. Multi-start L-BFGS keeps the first start at the configured init and draws additional starts using `sobol`, `random`, or local `jitter` starts. With `--lbfgs-sobol-candidates N`, Sobol starts are selected by screening `N` candidates with the profiled log posterior.

The default mass coordinate is `mass_parameterization = "eta"`, i.e. `eta = log_MBH - log10(D_A)`. Saved samples still include derived `log_MBH`, and the original `log_MBH` prior is applied to that derived value. Use `--mass-parameterization log_mbh` to sample `log_MBH` directly.

The `[convergence.fixed_r_*]` blocks provide high-resolution numerical references for the phi-marginal diagnostic scripts.

## Numerical Diagnostics

Numerical diagnostic scripts:

```bash
python scripts/megamaser/convergence/convergence_phi_marginal.py --galaxies NGC6264 --no-grad
python scripts/megamaser/convergence/check_conditional_r_grad_vs_numerical.py --galaxy NGC6264
python scripts/megamaser/convergence/r_ang_posteriors.py --galaxies NGC6264
python scripts/megamaser/convergence/check_conditional_r_delta.py --galaxies NGC6264
```

These are diagnostic-only checks for quadrature and gradients. They are not alternative megamaser samplers.
