# Reproducing the Watkins & Feldman (2026) megamaser H0

Status: 2026-08-20. Reference values are arXiv:2608.06247**v4** (14 Aug 2026),
which silently replaced the velocities, the headline, and the tables of v1-v3.
See "Version history" below; the two are not interchangeable.

## Bottom line

Watkins & Feldman (hereafter WF26) report `H0 = 68.8 +/- 2.6 km/s/Mpc` from the
six P20 distances with Manticore (M25) peculiar velocities, against
`71.5 +/- 2.6` with Carrick et al. (2015, C15), and conclude that the Hubble
tension is a distance-ladder systematic. `scripts/megamaser/reproduce_watkins_h0.py`
rebuilds their model and recovers every published value to `0.17 km/s/Mpc` or
better.

**The `-2.80 km/s/Mpc` C15 -> M25 shift is one galaxy.** Swapping NGC 5765b's
velocity alone accounts for `-2.28`; the other five together give `-0.52`.

| galaxy swapped C15 -> M25 | H0 | shift | dv (km/s) |
|---|---:|---:|---:|
| NGC 5765b | 69.25 | **-2.28** | +483 |
| UGC 3789 | 71.29 | -0.24 | +99 |
| NGC 6264 | 71.44 | -0.09 | +215 |
| CGCG 074-064 | 71.46 | -0.07 | +37 |
| NGC 4258 | 71.48 | -0.05 | +23 |
| NGC 6323 | 71.50 | -0.03 | +280 |

All at `sigma_v = 150 km/s`, six galaxies. NGC 5765b dominates because it has
the smallest fractional distance error among the five distant hosts (4.5 per
cent) and so carries 53-58 per cent of the Fisher information on H0 across
their whole `sigma_v` range, and because it is where the two reconstructions
disagree most violently. Note that NGC 6323 has a *larger* velocity change
(+280) and moves H0 by `-0.03`, because its distance error is 27 per cent.

## Their model

Three equations over six distance-velocity pairs taken from P20 table 1, with
asymmetric distance uncertainties symmetrised by averaging the two sides. No
VLBI spot data, no re-derivation of any distance, no distance prior, no
selection function, no Malmquist treatment; the words "prior", "selection" and
"Malmquist" do not appear in the v4 inference.

```
L(H0) = prod_i N(d_i | d_m(z_i; H0), sigma_i)                      Eq. (1)
d_m   = flat-LCDM angular-diameter distance, Om = 0.3               Eq. (2)
sigma_i^2 = sigma_di^2 + (sigma_v / c * dd_m/dz|_{z_i})^2           Eq. (3)
```

with `cz_i = cz_obs,i - v_pec,i`, H0 the only free parameter, and `sigma_v`
fixed by hand.

### Conventions needed to reproduce them

Both were determined by matching their published numbers, and neither is stated
in the paper:

- **Point estimate is the maximum of `L(H0)`, not its mean.** `L` is skewed and
  the two differ by `0.2 km/s/Mpc`.
- **The Gaussian prefactor is dropped.** `sigma_i` depends on H0 through
  Eq. (3), so `-2 log L` is not `chi^2`, but only the `chi^2` form reproduces
  them: keeping `log(sigma_i^2)` moves the C15 headline to 71.66 against their
  71.5 and the M25 headline to 68.86 against their 68.8. Their text describes
  the procedure as minimising the exponent, so they drop it too.

`d_m` must be the angular-diameter distance. The comoving and luminosity forms
give 73.5 and 75.4 for the C15 row against their 71.5.

| row | reproduced | WF26 v4 |
|---|---:|---:|
| C15, `sigma_v = 150` | 71.53 +/- 2.60 | 71.5 +/- 2.6 |
| M25, `sigma_v = 150` | 68.73 +/- 2.51 | 68.8 +/- 2.6 |
| M25, `sigma_v = 100` | 68.38 +/- 2.37 | 68.4 +/- 2.5 |
| M25, `sigma_v = 50` | 67.08 +/- 2.11 | 66.9 +/- 2.5 |
| M25 v1-v3 velocities, `sigma_v = 150` | 69.66 +/- 2.56 | 69.7 +/- 2.6 |

At `sigma_v = 250`, the P20 value and the one WF26 do not tabulate, the same
model gives `69.07 +/- 2.80` with M25 and `71.81 +/- 2.89` with C15. Their
claim is `1.6 sigma` there, and their tabulated grid stops one row short of it.

## The sigma_v weighting knob

`dd_m/dz ~ c / H0` near `z = 0`, so the velocity term of Eq. (3) adds the *same*
distance error in Mpc to every galaxy: 2.1 Mpc at `sigma_v = 150`, which is 28
per cent of NGC 4258's 7.58 Mpc but 1.6 per cent of NGC 6264's 132.1 Mpc.
Lowering `sigma_v` therefore transfers the fit onto the nearest host rather
than tightening it uniformly. Fisher weight on H0 is `d_i^2 / sigma_i^2`, not
the `1 / sigma_i^2` of the distance-space chi^2.

| `sigma_v` | NGC 4258 | NGC 5765b | other four |
|---:|---:|---:|---:|
| 0 | **84.9** | 9.0 | 6.1 |
| 20 | 41.6 | 34.7 | 23.7 |
| 50 | 11.4 | 52.6 | 36.0 |
| 100 | 3.3 | 57.2 | 39.5 |
| 150 | 1.6 | 57.8 | 40.6 |
| 250 | 0.7 | 56.9 | 42.4 |

Per cent, M25 v4 velocities.

## Leave-one-out

M25 v4 velocities, maximum-likelihood H0:

| subsample | `sv=0` | `sv=50` | `sv=100` | `sv=150` | `sv=250` |
|---|---:|---:|---:|---:|---:|
| six galaxies | **55.16** | 67.08 | 68.38 | 68.73 | 69.07 |
| drop NGC 4258 | 68.86 | 68.88 | 68.92 | 69.00 | 69.19 |
| drop UGC 3789 | 54.92 | 67.74 | 69.35 | 69.76 | 70.06 |
| drop CGCG 074-064 | 54.43 | 65.50 | 66.78 | 67.11 | 67.36 |
| drop NGC 6323 | 55.14 | 67.12 | 68.45 | 68.82 | 69.19 |
| drop NGC 5765b | 53.48 | 65.81 | 68.41 | 69.22 | 69.97 |
| drop NGC 6264 | 55.00 | 66.89 | 68.23 | 68.59 | 68.92 |
| drop NGC 4258 + NGC 5765b | 69.59 | 69.63 | 69.75 | 69.91 | 70.28 |

At their fiducial `sigma_v = 150` the leave-one-out values span `67.11` to
`69.76`, a range of `2.65 km/s/Mpc`. That is comparable to their quoted
`+/- 2.6` and close to the `-2.80` C15 -> M25 shift they report as a detection,
so no single galaxy can be removed without moving H0 by about as much as the
effect being claimed. The extremes are CGCG 074-064, whose implied
single-galaxy H0 is the highest in the sample at `78.1`, and UGC 3789, whose
`63.6` is the lowest of the five distant hosts. At `sigma_v = 0` the same range
is `53.48` to `68.86`.

Two further things follow.

**NGC 4258 carries the entire `sigma_v` dependence.** Without it the fit is flat
to `0.33 km/s/Mpc` from `sigma_v = 0` to `250`. With it, the six-galaxy fit
descends to `55.2` at `sigma_v = 0` because M25 assigns NGC 4258 a peculiar
velocity of `285 +/- 94 km/s` where `122` is required for `H0 = 73.5` and `168`
for `67.4` — an overshoot of 1.7 realisation standard deviations at the one
host whose distance is known to 1.5 per cent, so the check is available.
v1-v3 published the `sigma_v <= 40` rows and reported `68.4` at `sigma_v = 0`,
because the old velocity was `162`. With the v4 velocity the same row reads
`55.2`, and v4 deleted the row, the `chi^2_v` column, and the sentence that did
this arithmetic.

**Removing both galaxies removes the result.** The four-galaxy fit is `69.6` to
`70.3` across the whole `sigma_v` range with `+/- 3.9` to `4.4`, consistent with
both the CMB and the distance ladder.

The same structure holds for C15 velocities, so it is a property of Eq. (3) and
of the sample, not of Manticore: the C15 six-galaxy fit at `sigma_v = 0` is
`58.1`.

## Their Manticore velocities are correct; the point evaluation is not

`scripts/megamaser/compare_watkins_velocities.py` evaluates both Manticore
products of the `2MPP_MULTIBIN_N256_DES_V2` generation along the six maser
lines of sight over all 80 realisations: the `forward_fields/PCS` BORG grid
(`ManticoreLocalCOLA`) and the `SWIFT_velocity_fields` momentum products
(`ManticoreLocalSWIFT`, `p / rho`). Both are `681 Mpc/h` boxes at `N = 256` in
ICRS with the observer at the box centre; the SWIFT files carry no attributes,
so `ngrid` and `boxsize` must be set in `local_config.toml`.

At the P20 point distance, mean and scatter over the 80 realisations:

| galaxy | `D` [Mpc] | PCS | SWIFT | WF26 v4 |
|---|---:|---:|---:|---:|
| NGC 4258 | 7.58 | 255 +/- 63 | 311 +/- 78 | 285 +/- 94 |
| UGC 3789 | 51.5 | 54 +/- 35 | 63 +/- 41 | 45 +/- 48 |
| CGCG 074-064 | 87.6 | 336 +/- 45 | 370 +/- 45 | 334 +/- 61 |
| NGC 6323 | 109.4 | 579 +/- 57 | **708 +/- 72** | **694 +/- 88** |
| NGC 5765b | 112.2 | 537 +/- 59 | 573 +/- 80 | 607 +/- 53 |
| NGC 6264 | 132.1 | 392 +/- 51 | 427 +/- 58 | 439 +/- 75 |

**Their velocities reproduce.** Every galaxy agrees within about half their
quoted scatter, and NGC 6323 identifies the product: the two Manticore fields
differ by `130 km/s` there and WF26 sit `14 km/s` from SWIFT against `115` from
PCS. They used the SWIFT momentum fields and read them correctly. Any critique
that they mis-queried the reconstruction is wrong.

**NGC 5765b sits on a velocity cliff, and P20's distance lands on its peak.**
Along that line of sight the reconstructed velocity rises to a maximum within a
Mpc of `112.2 Mpc` and then collapses:

| `D` [Mpc] | PCS | SWIFT |
|---:|---:|---:|
| 97.2 | 409 +/- 34 | 427 +/- 32 |
| 107.2 | 528 +/- 42 | 564 +/- 56 |
| **112.2** | **537 +/- 59** | **573 +/- 80** |
| 117.2 | 374 +/- 68 | 320 +/- 88 |
| 122.2 | 166 +/- 70 | 40 +/- 67 |
| 127.2 | 25 +/- 70 | -68 +/- 86 |

The gradient just beyond the peak is `-33` to `-56 km/s` per Mpc. For this host
`dH0/dv = -1/D = -0.009 km/s/Mpc` per `km/s`, so the `640 km/s` fall between
`112` and `127 Mpc` is `5.7 km/s/Mpc` in its implied H0, on a galaxy carrying
58 per cent of the Fisher weight.

Marginalising the same field over a Gaussian distance posterior, median and
central 68 per cent interval:

| `sigma_D` [Mpc] | PCS | SWIFT |
|---:|---:|---:|
| 0 (point) | 539 +59/-65 | 599 +44/-129 |
| 5 (P20) | 504 +71/-149 | 524 +93/-232 |
| 11 (ours) | 430 +114/-296 | 440 +141/-401 |

At P20's own `+/- 5 Mpc` the point estimate already overstates the correction by
`50` to `75 km/s` and understates its uncertainty by a factor of two to three:
the realisation scatter WF26 adopt, `+/- 53 km/s`, is the spread of the field at
a fixed position, whereas the dominant uncertainty is where along the cliff the
galaxy sits. Our own stage-1 posterior is about twice as wide, which takes the
median to `440 km/s` with a `-400 km/s` lower tail. The remainder of the gap to
our tabulated `229 +250/-185` is the stage-2 reweighting, which applies the
volume prior and the selection function and so pushes the sampled distance
further out along the falling branch; that part has not been checked directly
against the chain.

### Their model with our own query of the same reconstruction

Substituting our 80-realisation velocities into their Eq. (1), all six galaxies:

| velocities | `sv=50` | `sv=100` | `sv=150` | `sv=250` |
|---|---:|---:|---:|---:|
| WF26 tabulated M25 | 67.08 | 68.38 | **68.73** | 69.07 |
| SWIFT, at the P20 distance | 66.78 | 68.31 | **68.72** | 69.08 |
| PCS, at the P20 distance | 67.77 | 68.83 | **69.13** | 69.44 |
| SWIFT, marginalised over `D` | 67.33 | 68.92 | **69.35** | 69.72 |
| PCS, marginalised over `D` | 68.11 | 69.22 | **69.53** | 69.84 |
| C15, for reference | 69.83 | 71.18 | **71.53** | 71.81 |

Our SWIFT query reproduces their headline to `0.01 km/s/Mpc` at their fiducial
`sigma_v`, which is the cleanest possible confirmation that the disagreement is
not about the field. Swapping SWIFT for the PCS forward grid adds `0.4`, and
marginalising over the distance instead of evaluating at a point adds a further
`0.4` to `0.6`. Both together give `69.53` against their `68.73`.

Those marginalised rows are a lower bound on the effect, because they push the
marginal *mean* through a likelihood that still has no distance-dependent
velocity: the marginal scatter is `120` to `280 km/s`, not the `50` to `150`
assumed. Propagating the width as well, rather than only the mean, is what our
own population model does, and it is worth a further `1.5 km/s/Mpc` in the
gap to our `71.5`, the rest being the volume prior and the selection function.

The same mechanism, more weakly, affects NGC 6323 and NGC 6264: marginalising
over their P20 distance errors moves the assigned velocity from `579` to `239`
and from `392` to `299 km/s`. It does not affect NGC 4258, UGC 3789 or
CGCG 074-064, where the field is locally flat.

## Why their 68.8 is 2.8 below our 71.5 on the same distances

**It is not the selection function.** Their model with the velocity field
switched off gives `73.37` at `sigma_v = 250`, and `73.63` with `sigma_v`
sampled under our own Maxwell prior. That is where P20 (`73.9`), our
replication of P20 (`73.7`), and our redshift-selection run on the same
distances (`73.3`) all sit -- not where our no-selection run sits (`71.4`).
Their distance-space `chi^2` with a distance-independent velocity error already
carries an H0 dependence equivalent to the flat-latent-velocity prior of P20,
which is the same approximate redshift-selection accounting described in
section 6.3 of the paper. So the baselines agree, and

| | no field | with Manticore | correction |
|---|---:|---:|---:|
| WF26, `sigma_v = 150` | 73.37 | 68.72 | **-4.65** |
| WF26, `sigma_v = 250` | 73.37 | 69.08 | **-4.30** |
| ours, redshift selection, volume prior | 73.30 | 71.50 | **-1.80** |

**The entire gap is that their peculiar-velocity correction is two and a half
times ours, from the same reconstruction.**

Their model throughout, one change at a time:

| step | H0 | shift |
|---|---:|---:|
| WF26 as published: SWIFT, point velocity, six galaxies, `sigma_v = 150` | 68.72 | --- |
| + PCS forward grid instead of the SWIFT momentum products | 69.13 | +0.41 |
| + velocity marginalised over the P20 distance error | 69.53 | +0.40 |
| + five galaxies, NGC 4258 dropped | 69.75 | +0.22 |
| + `sigma_v = 250`, the P20 value | 69.94 | +0.19 |
| + external bulk flow projected on each line of sight | 70.06 | +0.12 |

That is `+1.34`, about half the gap, and no single term dominates. The `+0.41`
and `+0.40` are the two velocity-bookkeeping choices of the previous section;
the rest are sample, `sigma_v`, and the missing `Vext` term.

### Sampling sigma_v

The `chi^2` form is unnormalised in `sigma_v` and is maximised by
`sigma_v -> infinity`, so the Gaussian prefactor must be reinstated before
`sigma_v` can be inferred at all. Marginalising over it, with the median of the
`sigma_v` posterior alongside:

| velocities | Maxwell mean 250 [ours] | uniform [0, 150] | uniform [0, 700] |
|---|---|---|---|
| SWIFT, point, six galaxies | 69.08 (`sv=223`) | 68.44 (`sv=115`) | 69.12 (`sv=272`) |
| PCS, marginalised, five | 69.97 (`sv=236`) | 69.66 (`sv=76`) | 70.02 (`sv=303`) |
| no velocity field, six | 73.63 (`sv=222`) | 73.86 (`sv=107`) | 73.78 (`sv=271`) |

Under the uniform priors the `sigma_v` median tracks the prior upper bound, and
H0 moves by `0.7` between `[0, 150]` and `[0, 700]`: the six galaxies do not
constrain `sigma_v`, exactly as our own `sigma_pec` is prior-dominated
(section 8 of the paper). Sampling it rather than fixing it at 150 raises their
value by `0.36` and leaves the conclusion unchanged; it is not the explanation
either.

### What is left: where the galaxy is placed in the reconstruction box

The residual is not the distance posterior. Our stage-2 median for NGC 5765b is
`D_A = 111.98 Mpc`, within `0.2 Mpc` of the P20 value WF26 use. It is the radius
at which the field is queried, and two separate conventions separate us:

1. **Comoving, not angular-diameter.** The pipeline places each galaxy at
   `D_c`, which for NGC 5765b is `115.09` against `D_A = 111.98`.
2. **The sampled h, not the box h.** The lookup radius is `D_c * h` with `h`
   from the *sampled* H0, so `0.716`, not the Manticore box value `0.681`. That
   is a further `5.1 per cent` outward in box units.

Together the field is read at `82.56 Mpc/h` rather than the `76.4 Mpc/h` that
`D_A(P20) * 0.681` gives -- `9 Mpc` further out, and NGC 5765b's cliff is
`9 Mpc` wide. Interpolating the PCS field over the chain and all 80
realisations:

| galaxy | pipeline `D_c*h` | `D_c*0.681` | `D_A(P20)*0.681` | paper table 4 |
|---|---:|---:|---:|---:|
| CGCG 074-064 | 362 +70/-129 | 361 +63/-73 | 338 +40/-44 | 362 |
| NGC 5765b | **207 +249/-184** | 454 +108/-233 | 539 +60/-67 | **207** |
| NGC 6264 | 242 +123/-205 | 337 +107/-147 | 388 +58/-46 | 243 |
| NGC 6323 | 553 +81/-158 | 530 +86/-108 | 578 +45/-53 | 553 |
| UGC 3789 | 53 +39/-40 | 66 +40/-38 | 55 +42/-37 | 53 |

The first column reproduces table 4 of the paper exactly, so the decomposition
is closed. Feeding each convention back through their model, five galaxies:

| velocities | `sv=150` | `sv=250` | shift |
|---|---:|---:|---:|
| WF26, SWIFT at `D_A(P20)` | 69.05 | 69.23 | --- |
| PCS at `D_A(P20)*0.681` | 69.34 | 69.52 | +0.29 |
| PCS at `D_c*0.681` | 69.70 | 69.88 | +0.36 |
| PCS at `D_c*h_sampled` [ours] | 70.98 | 71.12 | **+1.25** |
| our published headline | | 71.50 | +0.38 |

**The single largest term in the whole disagreement is our own choice to place
galaxies in the box with the sampled h rather than the box h**, worth
`1.25 km/s/Mpc`, and it acts almost entirely through NGC 5765b. The remaining
`0.38` is the volume prior, the selection function, latent-distance sampling
and `Vext`, none of which their model can express.

### Which placement convention is right

Convention A is correct, and the identity is exact rather than a low-z
approximation. A structure observed at redshift `z` is placed in the box at

    r_box = h_fid * D_c^fid(z) = (c/100) * Int_0^z dz'/E(z'; Om_fid),

in which `h_fid` cancels: the box radial coordinate is a redshift coordinate
relabelled through `E(z)`, and only `Om` enters it. For a maser host the model
says `D_c = (c/H0) Int_0^{z_cos} dz'/E`, so dividing the two expressions gives

    r = (H0_sampled / 100) * D_c   exactly,

for the same `Om` on both sides. This is what the code does, and it does it in
one place: `run_joint_H0.py` passes `D_c * h` both to `distance2redshift` and to
`_interp_los_velocity`, and `Distance2Redshift.__call__(r, h)` is literally
`f(r * h)` with `f` inverted from an `H0 = 100` cosmology. The field lookup
radius and the argument of the cosmological redshift are the same variable.

The closure argument makes the failure of B concrete. For a host whose
environment 2M++ resolves, the reconstruction encodes that environment at
real-space coordinate `(cz_obs - v)/100`. At the true `(H0, D_c)` Convention A
queries `H0 D_c/100 = cz_cos/100`, which is that coordinate, and the predicted
redshift closes on `cz_obs`. Convention B queries `0.681 D_c`, which is the
host's own coordinate only if `H0 = 68.1`; otherwise it reads a structure `5`
per cent away in radius -- for NGC 5765b, `4 Mpc/h`, or `590 km/s` of redshift
coordinate, a different part of the flow. B is a unit inconsistency, not an
alternative convention, and it should not appear in the error budget.

The empirical check confirms this. The h-free, redshift-anchored queries that
the peculiar-velocity literature actually uses bracket Convention A, and
Convention B sits outside that family:

| query radius | NGC 5765b [Mpc/h] | v [km/s] | H0, their model, five galaxies |
|---|---:|---:|---:|
| `cz_obs/100`, redshift space | 85.3 | 87 | 72.06 |
| iterated to real space (Yahil) | 84.0 | 135 | 71.31 |
| **`D_c * h_sampled` [ours]** | **82.6** | **207** | **71.12** |
| `D_c * 0.681` | 78.4 | 454 | 69.88 |
| `D_A(P20) * 0.681` [approximately WF26] | 76.4 | 539 | 69.52 |

Our hierarchical placement agrees with the classical iterated real-space
construction to `0.19 km/s/Mpc`, which is the like-for-like comparison: both
put the galaxy at its real-space redshift coordinate, ours by sampling the
latent distance rather than by iteration. The honest robustness spread is
therefore the `71.1` to `72.1` between the real-space and redshift-space
queries, not anything involving `h_fid`.

Two supporting points. There is no accompanying velocity rescaling: in linear
theory `v[km/s] = 100 f E(a) a delta / k[h/Mpc]`, in which `h` cancels exactly
as it does in the coordinates, so the km/s field is redshift-anchored too and
the residual fiducial-cosmology dependence is an `f sigma_8`-type amplitude
question handled by the `beta` scaling, not an `h` one. And our own Cepheid
paper already states Convention A in print, on the identical Manticore product:
distances are converted from Mpc to `h^-1 Mpc` using the sampled H0.

The decisive test for a referee is a mock closure run: take one Manticore
realisation as truth, choose `h_true = 0.716`, place mock masers at box radius
`r` with `D_c = r/h_true` and `cz_obs = 100 r + v(r) + noise`, and run the
pipeline both ways. A should recover `H0 = 71.6` unbiased and B should not.
This has not been run.

The one irreducible caveat is second order: the reconstruction's prior power
spectrum was set at the fiducial cosmology, so a large `h_true - h_fid`
mismatch would distort the field where the 2M++ data are weak. Removing it
needs the reconstruction re-run or marginalised over cosmology.

## Version history

v1 (6 Aug) -> v2 (8 Aug) is a commented-out title and a bib entry; v2 -> v3
(11 Aug) is a float move. v3 -> v4 (14 Aug) is a substantive rewrite with no
note added, no changelog, and a byte-identical abstract:

- M25 velocities changed from a single realisation to 80-realisation means:
  NGC 4258 `162 -> 285 +/- 94`, CGCG 074-064 `149 -> 334 +/- 61`, NGC 5765b
  `546 -> 607 +/- 53`, NGC 6323 `671 -> 694 +/- 88`, NGC 6264 `434 -> 439 +/- 75`,
  UGC 3789 `54 -> 45 +/- 48`.
- Headline moved `69.7 -> 68.8` at `sigma_v = 150`.
- The MCMC over a uniform `sigma_v` prior, its corner plot, and the marginalised
  `68.7 +/- 2` with `P(> H_DL) = 0.029` were deleted, with nothing replacing them.
- The `sigma_v` grid changed from `0, 10, 20, 40, 100` to `50, 100, 150`; the
  `chi^2_v` and `P(< H_CMB)` columns were removed.

The `-0.93` headline shift decomposes as CGCG 074-064 `-0.40`, NGC 5765b
`-0.28`, NGC 4258 `-0.26`. CGCG 074-064's velocity moved by `3.0` times the
realisation standard deviation that v4 then adopts as its velocity error, which
is the internal inconsistency: inter-realisation scatter is a consistency width,
not an accuracy, and their own revision, the C15-M25 disagreement (up to `483
km/s` for NGC 5765b against its `+/- 53`), and the NGC 4258 test all exceed it.

## Tension arithmetic

`P(H0 >= 73.5) = 0.041` reproduced against their `0.047`, both one-sided, which
is `1.7 sigma`, not the "greater than 2 sigma" of the abstract. The `+/- 0.81`
on the distance-ladder value is not folded in.

| `sigma_v` | `P(H0 >= 73.5)` | one-sided |
|---:|---:|---:|
| 50 | 0.003 | 2.8 sigma |
| 100 | 0.024 | 2.0 sigma |
| 150 | 0.041 | 1.7 sigma |
| 250 | 0.072 | 1.5 sigma |

The claim clears `2 sigma` only at `sigma_v <= 100 km/s`, below the value P20
adopt and below the `235 km/s` RMS at which C15 and M25 disagree over these six
hosts.

## What is defensible

- Averaging the likelihood over realisations rather than averaging velocities
  first is correct and preserves the inter-galaxy velocity correlation within a
  realisation. It is a genuine improvement over v1-v3 and over a mean field.
- Reproducing P20's method (5) (`71.5 +/- 2.6` against P20's `71.8 +/- 2.9`)
  validates their machinery.
- The sign of the reconstruction effect matches ours.

Section 9 of the script emulates the realisation averaging with independent
Gaussian draws, since only the per-galaxy means and standard deviations are
published. That discards the inter-galaxy correlation, so it bounds the width
rather than reproducing their procedure; it moves H0 by `+0.2` to `+0.4`.
