# Reproducing the P20 headline H0 from the archived distance posteriors

Status: 2026-08-12.

## Bottom line

The available P20 distance sequences do recover the headline result of
Pesce et al. (2020, hereafter P20), but only after reproducing the full P20
stage-2 model. Fixing the peculiar-velocity dispersion to
`250 km/s` is required for their fiducial method, but fixing it alone is not
what moves the current result to `73.9 km/s/Mpc`.

The three decisive changes are:

1. P20 fixes the peculiar-velocity dispersion to `250 km/s`.
2. P20 assigns flat priors to the latent true recession velocities, whereas
   the CANDEL runner samples galaxy distances. The corresponding
   velocity-to-distance Jacobian raises the five-galaxy median by
   `0.63 km/s/Mpc`.
3. P20 includes NGC4258; the CANDEL joint-H0 runner explicitly excludes it.
   Adding its published `7.58 +/- 0.11 Mpc` constraint raises the reproduced
   median by another `0.26 km/s/Mpc`.

Using the five archived P20 distance chains, the P20 velocities and their
quoted statistical errors, the P20 distance-redshift relation, and a Gaussian
approximation to the published NGC4258 distance gives

```text
H0 = 73.92 -2.96/+3.06 km/s/Mpc,
```

in direct agreement with P20's `73.9 +/- 3.0 km/s/Mpc`.

## Reproduction

Run the lightweight deterministic quadrature from the repository root:

```bash
venv_candel/bin/python packages/candel-maser/scripts/reproduce_p20_h0.py
```

It uses the same KDE implementation and support convention as
`packages/candel-maser/candel_maser/run_joint_H0.py`, then evaluates P20 Equations 1--4 with
`sigma_pec = 250 km/s`. The five archived files reproduce the medians and
central intervals in P20 Table 1. NGC4258 is represented by the Gaussian
distance likelihood quoted in that table, including its statistical and
systematic uncertainties in quadrature.

The complete leave-one-out check is:

| fit | reproduced H0 | P20 H0 | median difference |
|---|---:|---:|---:|
| drop UGC3789 | 75.75 -3.27/+3.40 | 75.8 | -0.05 |
| drop NGC6264 | 73.75 -3.10/+3.20 | 73.8 | -0.05 |
| drop NGC6323 | 73.87 -2.99/+3.09 | 73.8 | +0.07 |
| drop NGC5765b | 74.34 -4.42/+4.63 | 74.1 | +0.24 |
| drop CGCG074-064 | 72.52 -3.23/+3.32 | 72.5 | +0.02 |
| drop NGC4258 | 73.66 -2.97/+3.06 | 73.6 | +0.06 |
| all six | 73.92 -2.96/+3.06 | 73.9 | +0.02 |

All values are in `km/s/Mpc`. The maximum median discrepancy is
`0.24 km/s/Mpc`; six of seven medians agree within `0.07 km/s/Mpc`. This
agreement across the full jackknife table is a much stronger validation than
matching only the headline value.

## What P20 actually fits

Let `q_i(D_A)` denote the archived stage-1 distance posterior and let `v_i` be
the latent cosmological recession velocity. P20 method (1) has the target

```text
p(H0, {v_i} | data) proportional to p(H0)
    product_i q_i[D_A(v_i, H0)]
              Normal(vhat_i | v_i, sqrt(sigma_v,i^2 + 250^2)),
```

with flat priors on `H0` and each `v_i`. The angular-diameter distance
`D_A(v_i, H0)` is P20 Equation 1 for flat LCDM with `Omega_m = 0.315`.

Changing the latent integration coordinate from `v_i` to `D_A` gives

```text
q_i(D_A) Normal[vhat_i | v(D_A, H0), sigma_i]
    |dv_i / dD_A| dD_A.
```

At these redshifts, `|dv_i / dD_A|` is approximately `H0`. It therefore
contributes an approximately `H0^5` factor for the five archived galaxies and
`H0^6` when NGC4258 is included. This is not an optional numerical detail: it
is the measure implied by P20's stated flat-velocity nuisance prior.

The local MCP material records the archived stage-1 chains as having an
effective prior uniform in `log(D_A)`. P20 then inserts those posterior
densities directly as its distance likelihood. Thus the distance-dependent
part is effectively log-flat relative to the underlying disk likelihood, but
the flat-velocity stage-2 measure also contributes the H0-dependent Jacobian
above. Calling the P20 prior simply "flat in distance" loses this distinction.

## What the CANDEL joint-H0 runner fits

For the no-reconstruction path, `packages/candel-maser/candel_maser/run_joint_H0.py` instead
samples `H0`, a shared `sigma_pec`, and one distance coordinate per galaxy. It
predicts the cosmological redshift from the comoving distance and evaluates

```text
Normal(vhat_i | vpred(D_c,i, H0), sigma_pec).
```

The relevant differences from P20 method (1) are:

| component | P20 headline model | current CANDEL joint-H0 model |
|---|---|---|
| galaxies | six, including NGC4258 | five; NGC4258 is rejected by the runner |
| peculiar scatter | fixed `250 km/s` | sampled from `Maxwell(scale=156.7 km/s)`, whose mean is approximately `250 km/s` |
| latent coordinate | true cosmological velocity `v_i` | `D_c`, or `log(D_A)` for the explicit log-distance prior |
| latent measure | flat in `v_i` | explicit distance, volume, or log-distance prior |
| P20 distance chains | posterior density used directly | uniform-log stage-1 prior is removed first, then the requested stage-2 prior is applied |
| selection | none | configurable; default CLI choice is redshift selection |
| velocity correction | none | none, Carrick2015, or ManticoreLocalCOLA |
| velocity error | `sqrt(250^2 + sigma_v,i^2)` | `sigma_pec`; the `0.4--1.9 km/s` statistical errors are omitted |
| cosmology | flat LCDM, `Omega_m=0.315` | same `Omega_m`, numerically inverted distance-redshift relation |
| H0 prior | reported flat | `Uniform(10, 200) km/s/Mpc` |

The missing statistical velocity errors are numerically irrelevant beside
`250 km/s`, but the galaxy set, velocity prior, latent measure, distance prior,
and any active selection or reconstruction are not.

In particular, `--reconstruction none` does not by itself select the P20
model. The CLI still defaults to `--selection redshift`, which forces the
uniform-in-volume distance prior. Even an explicit `--selection none` defaults
to the uniform-in-distance prior, not the measure used by P20.

## Controlled numerical decomposition

The locally saved P20-source chains give:

| model | H0 |
|---|---:|
| CANDEL, no selection/reconstruction, volume prior, sampled `sigma_pec` | 71.43 -2.91/+3.10 |
| CANDEL, no selection/reconstruction, distance prior, sampled `sigma_pec` | 72.54 -2.96/+3.11 |
| CANDEL, no selection/reconstruction, log-distance prior, sampled `sigma_pec` | 72.98 -2.95/+3.07 |
| same archived-posterior measure, but fix `sigma_pec=250 km/s` | 73.04 -2.95/+3.05 |
| P20 flat-velocity measure, fixed `250 km/s`, five galaxies | 73.66 -2.97/+3.06 |
| P20 flat-velocity measure, fixed `250 km/s`, all six | 73.92 -2.96/+3.06 |

For the current log-distance chain, the sampled scatter is
`245 -95/+111 km/s`, so the data do not strongly update the Maxwell prior.
Fixing it to `250 km/s` changes the median by only `+0.06 km/s/Mpc`. The answer
to "do I need to fix sigma_v to 250?" is therefore:

- yes, for a literal reproduction of P20 method (1);
- no, it is not the reason the current result misses `73.9`;
- the flat-velocity Jacobian and NGC4258 supply almost all of the remaining
  shift.

P20's method (2), which fits `sigma_pec`, is also not the current model: P20
uses the outlier-robust Sivia--Skilling likelihood and obtains
`74.4 -3.4/+3.9 km/s/Mpc`, whereas CANDEL uses a Gaussian velocity likelihood
with a Maxwell prior on the shared scatter.

## Interpretation and recommended use

The current CANDEL model is not failing to reproduce P20 because of a coding
error in the distance KDEs. It answers a different statistical question. Its
explicit distance-population prior, selection normalisation, optional velocity
reconstruction, sampled peculiar scatter, and five-galaxy sample are deliberate
forward-model choices.

The P20 result should therefore be used as a compatibility diagnostic, not as
a target that the production model must recover. If a P20 compatibility mode
is ever added to `run_joint_H0.py`, it must change the galaxy set, fix
`sigma_pec`, and reproduce the flat-velocity nuisance measure together.
Adding only a `--fix-sigma-pec 250` flag would be incomplete and misleading.

The MMH0 manuscript currently describes P20's prior as constant in distance.
That should be revisited before publication: P20's reported parameterisation,
the MCP clarification that the disk-distance prior was effectively log-flat,
and the successful reproduction here support the more precise description
above.

## Inputs, approximations, and provenance

- P20 paper and equations: <https://arxiv.org/pdf/2001.09213>.
- Archived distance inputs:
  `data/Megamaser/external/Dom_data/D_archivedP20_*.txt`.
- Current runner and KDE implementation:
  `packages/candel-maser/candel_maser/run_joint_H0.py`.
- Reproduction diagnostic:
  `packages/candel-maser/scripts/reproduce_p20_h0.py`.
- Current sweep values and convergence caveats:
  `packages/candel-maser/docs/megamaser_joint_h0_sweep_summary.md`.
- Saved comparison chains:
  `results/Megamaser/fiducial/H0/joint_H0_toy_all_none_none_*_p20.hdf5`.

The archived text files contain no independent-chain labels or stage-2 code.
NGC4258 is not among them, so its published distance posterior is approximated
as Gaussian. The resulting reproduction is nevertheless robustly identified:
it matches the all-galaxy result to `0.02 km/s/Mpc`, the P20 five-galaxy result
to `0.06 km/s/Mpc`, and every published leave-one-out median to at worst
`0.24 km/s/Mpc`.
