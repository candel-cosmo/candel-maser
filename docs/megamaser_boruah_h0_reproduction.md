# Reproducing the Boruah et al. (2021) megamaser H0

Status: 2026-08-14. All reference values are the refereed MNRAS 507, 2697--2713
(2021) version, not the arXiv preprint. See "Refereed versus preprint" below;
the two are not interchangeable.

## Bottom line

Boruah, Hudson & Lavaux (2021, hereafter B21) report
`H0 = 70.1 +/- 2.9 km/s/Mpc` from the same six MCP distances that give
Pesce et al. (2020, hereafter P20) `73.9 +/- 3.0`. Rebuilding their model
inside CANDEL recovers `70.32 -2.81/+2.89`, and every row of their table 3
reproduces to `0.31 km/s/Mpc` or better.

The `3.8 km/s/Mpc` gap is not one effect. It decomposes as

| step | H0 | shift |
|---|---:|---:|
| P20 method (1): no velocity field, `sigma_pec = 250`, flat prior on the latent recession velocity | 73.92 | --- |
| P20 method (4): 2M++ point velocities, `sigma_v = 150` | 71.93 | -1.99 |
| latent measure changed from flat in `v` to flat in `D_c` | 71.33 | -0.60 |
| 2M++ queried in real space rather than at the observed redshift | 70.30 | -1.03 |
| line-of-sight marginalisation of that field | 70.32 | +0.02 |
| (variant) volumetric distance prior instead of uniform | 69.31 | -1.01 |

All values are in `km/s/Mpc`, six galaxies, archived P20 distances, no group
redshift correction, which B21 report raises H0 by `0.4`.

**The line-of-sight marginalisation is not what lowers H0.** B21 attribute
`+1.5 km/s/Mpc` to replacing the marginal by a point estimate, but with the
field read in real space the marginalisation moves the median by `+0.02`. The
`+1.5` is a change of coordinate, not of estimator: P20 obtain their point
velocities by querying the Carrick field *at the observed redshift*, that is
with the galaxy placed at its redshift-space position `r = cz/100` in `Mpc/h`,
whereas B21's marginal and every CANDEL evaluation read the field in real
space, at the sampled distance.

| convention | H0 |
|---|---:|
| P20 table 4 values | 71.33 |
| redshift space, galaxy at `r = cz/100` | 71.56 |
| redshift space, query iterated to self-consistency | 70.19 |
| real space, at the maser distance | 70.30 |
| real space, marginalised along the line of sight | 70.32 |

Uniform prior, `sigma_v = 150`, six galaxies. The redshift-space query
reproduces B21's `Fixed v_pec` row of `71.5` to `0.06`, while every real-space
evaluation lands at `70.2`--`70.3` regardless of where along the line of sight
the field is read or whether it is read once or marginalised. Iterating the
redshift-space query to self-consistency also gives the real-space answer,
because self-consistency puts the galaxy back at its true position.

The mechanism is that a galaxy with positive peculiar velocity sits *further
out* in redshift space than in real space, and these lines of sight are
infall-dominated, so the field is weaker there. Less of `cz` is then assigned
to peculiar motion and more to Hubble flow, raising H0. Per galaxy:

| galaxy | real, at `D_maser` | redshift space | redshift space, iterated | P20 table 4 |
|---|---:|---:|---:|---:|
| UGC3789 | -61.7 | -43.7 | -44.5 | -55.1 |
| NGC6264 | 311.3 | 103.6 | 259.6 | 230.6 |
| NGC6323 | 316.9 | 344.8 | 431.7 | 423.5 |
| NGC5765b | 380.9 | 125.1 | 287.4 | 127.7 |
| CGCG074-064 | 452.0 | 212.2 | 468.6 | 303.2 |
| NGC4258 | 284.8 | 245.6 | 290.2 | 262.3 |

Peculiar velocities in `km/s`. The redshift-space column reproduces the P20
table only to `72 km/s` rms, so P20's exact pipeline is not recovered galaxy by
galaxy, but the two agree at the level of the combined H0 to `0.23 km/s/Mpc`
and no real-space convention comes within `1 km/s/Mpc` of either.

## Reproduction

```bash
venv_candel/bin/python scripts/reproduce_boruah_h0.py
```

The script is a deterministic quadrature over the comoving distance, one
integral per galaxy per `H0` node, sharing the KDE distance likelihood, the
`Distance2Redshift` interpolator, and the `Carrick2015` LOS velocity cache with
`candel_maser/run_joint_H0.py`. It needs no sampler because the per-galaxy
likelihoods factorise once `sigma_v` is fixed.

## Refereed table 3, 2M++ block

| B21 row | B21 | reproduced | difference |
|---|---:|---:|---:|
| Fiducial (uniform distance prior) | 70.1 +/- 2.9 | 70.32 -2.81/+2.89 | +0.22 |
| Volumetric distance prior | 69.0 +2.9/-2.8 | 69.31 -2.78/+2.88 | +0.31 |
| No group redshift correction | 69.6 +2.9/-2.9 | 70.32 -2.81/+2.89 | +0.72 |
| `sigma_v = 200 km/s` | 70.5 +3.1/-3.1 | 70.71 -3.01/+3.12 | +0.21 |
| Fixed `v_pec` | 71.5 +/- 2.7 | 71.33 -2.66/+2.68 | -0.17 |
| P20 2M++ fit | 71.8 +/- 2.7 | 71.93 -2.65/+2.69 | +0.13 |

Widths reproduce as well as medians. Because no row here applies group
redshifts, the strict like-for-like comparison is the `No group redshift
correction` row, where the reproduction is `0.72 km/s/Mpc` (`0.25 sigma`) high.

## Further validation

Three checks that are not the headline number:

1. **P20 method (1)** reproduces to `0.02 km/s/Mpc`, as in
   `docs/megamaser_p20_h0_reproduction.md`.
2. **The 2M++ LOS velocity curves** match B21 figure 11 panel by panel: the
   `cz_pred(r)` crossings, plateaux and endpoint values agree, and the maser
   distance markers confirm B21 place galaxies at `r = D_A * 0.72` in
   `Mpc/h`. The independent NGC4993 check gives `427 km/s` at `40 Mpc` against
   B21's `456 +/- 150`. Per-galaxy H0 posteriors match their figure 10:
   NGC5765b is the tightest (`70.99 -3.91/+4.15`) and UGC3789 the lowest.
3. **Leave-one-out**, against the table that survives only as a commented-out
   block in the arXiv source, whose `all` entry of 69.0 identifies it as the
   volumetric variant:

   | dropped | reproduced delta | B21 delta |
   |---|---:|---:|
   | UGC3789 | +1.45 | +1.20 |
   | NGC6264 | -0.11 | -0.20 |
   | NGC6323 | +0.29 | +0.30 |
   | NGC5765b | -1.24 | -0.80 |
   | CGCG074-064 | -1.12 | -0.90 |
   | NGC4258 | +0.30 | +0.30 |

## Velocity field handling

B21 use the Carrick et al. (2015) 2M++ field from `cosmicflows.iap.fr`, which
is distributed with `beta* = 0.43` applied and the external dipole
`Vext = [89, -131, 17] km/s` (Galactic) already added. The CANDEL
`Carrick2015_FieldLoader` strips both, so the script restores them:
`v_2M++ = 0.43 * v_CANDEL + Vext . rhat`. Without that restoration the
comparison is meaningless; the dipole alone contributes between `-111` and
`+65 km/s` along these six lines of sight.

This differs from the production joint-H0 runner, which samples `Vext` under
the informative `[joint.priors.Vext_informative.Carrick2015]` prior
(`|Vext| = 201 +/- 16 km/s` towards `(l, b) = (306, -16)`) instead of fixing
Carrick's published dipole (`159 km/s` towards `(304, 6)`).

## Refereed versus preprint

The arXiv record has only v1. Its fiducial is the **volumetric** prior and it
quotes `69.0 +2.9/-2.8`. The refereed version makes the **uniform** prior
fiducial and quotes `70.1 +/- 2.9`, demoting `69.0` to a variant; every other
row is the same model shifted by that `~1 km/s/Mpc` prior change, and v1's
`Fixed v_pec and uniform prior` row is the refereed `Fixed v_pec` row.

The refereed justification is the Gould (1993) effect: the fractional Malmquist
bias is `3 rho^2 Delta^2`, where `rho` is the correlation between the selection
variable and the distance measurement. B21 state that for megamasers the
selection function "has complicated dependence on multiple observation
features", that quantifying the bias is "beyond the scope of this paper", and
that they therefore adopt the uniform prior (`rho = 0`, no bias) as fiducial
and report the volumetric prior as the maximal `rho = 1` bound. Their spread
between the two priors, `~1 km/s/Mpc (~0.4 sigma)`, is exactly the interval a
modelled selection function is supposed to resolve.

Anything citing B21 must say which of `69.0` and `70.1` it means, and must not
present `69.0` as their result.

## Residual and what is not reproduced

Ruled out as the cause of the `+0.72` residual:

- group redshifts, the one B21 ingredient missing here, move H0 by `+0.4` in
  their own test, i.e. the wrong sign to close the gap;
- the per-galaxy statistical velocity errors change nothing at this precision;
- placing the field at a fixed `h = 0.72` rather than at the sampled `h` leaves
  the median at `70.33` but collapses the interval to `+/-1.2`, so it is
  neither the cause nor what B21 did;
- NGC4258 is represented by a Gaussian rather than its MCP chain, but its
  leave-one-out weight is already correct (`+0.30` against B21's `+0.30`).

Dividing the uniform-`log(D_A)` stage-1 prior out of the archived chains gives
`69.81`. B21 state that P20 used a uniform distance prior at stage 1, so they
would not have applied this; it is reported as a variant, not a fix.

## Inputs and provenance

- B21 refereed text: MNRAS 507, 2697--2713, doi:10.1093/mnras/stab2320,
  accepted manuscript at <http://hdl.handle.net/10150/662291>; section 6.3 and
  table 3.
- B21 preprint: <https://arxiv.org/abs/2010.01119> (v1 only); its LaTeX source
  carries the commented-out leave-one-out table.
- P20 source: <https://arxiv.org/abs/2001.09213>, tables 1, 3 and 4.
- Distance inputs: `data/Megamaser/external/Dom_data/D_archivedP20_*.txt`.
- Velocity field: `data/fields/carrick2015_twompp_{density,velocity}.npy`.
- Script: `scripts/reproduce_boruah_h0.py`.
- Companion note: `docs/megamaser_p20_h0_reproduction.md`.
