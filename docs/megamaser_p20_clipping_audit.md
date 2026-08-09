# P20 spot clipping: forensics, and how to treat the outliers

Status as of 2026-08-09. Dom has since clarified that the vetting used an
iterative approximately 3-sigma cut on per-coordinate normalised residuals,
sometimes removing coherent emission regions rather than independent spots.
A subsequent broad Gaussian outlier-mixture implementation was rejected after
empirical testing and is not part of either the DE or MCMC model.

Source material: MCP response PDF (`~/Downloads/response.pdf`, 2026-08-03) and
`fiducial_tables.zip`, already unpacked into `data/Megamaser/fiducial/`.
Dataset provenance is in `data/Megamaser/README`; the `--dataset` switch is
documented in `scripts/megamaser/README.md`.


## 1. What differs between the two datasets

Matched spot-by-spot on velocity (exact in every table) using the repo loaders.

| galaxy | `original_published` | `fiducial` | change |
|---|---|---|---|
| CGCG074-064 | 165 | 165 | byte-identical |
| NGC4258 | 358 | 358 | byte-identical |
| NGC5765b | 212 | 169 | 20 placeholders excluded, 23 removed in vetting |
| NGC6264 | 66 | 61 | 5 removed, sigma_a replaced |
| NGC6323 | 68 | 87 | 19 added, **and all 68 originals modified** |
| UGC3789 | 156 | 153 | 3 removed |

Every count in Dom's letter now reproduces exactly. Gao et al. (2016) Table 6
has 212 rows, 20 of which carry the `a = 1.000 +/- 1.000` placeholder (all
systemic, 8241-8346 km/s); the journal's electronic table omitted those rows
and retained only 192. `original_published` restores the 20 printed rows while
masking their accelerations, so 212 - 20 - 23 = 169 for the P20 fit.

Non-issues, checked and dismissed: UGC3789's matched-row differences are pure
tabulation rounding; NGC5765b's sigma_a differs on 51/54 accelerating rows but
only at +/-10 per cent (re-derivation noise, not replacement).

Real, and **not mentioned in Dom's letter**:

- **NGC6323's 68 published spots were modified, not merely augmented.** x
  shifted on 65/68 (median +1.5 uas, max 12), y on 39/68 (max 38 uas) — 0.45
  and 0.70 sigma rms in units of the published errors. Uncertainties changed on
  ~15-18 spots, 4 accelerations differ, 3 acceleration flags lost and 3 gained.
- The 19 added Kuo+2011 spots are not a neutral extension: 15 blue / 4 red, no
  systemic, median positional error 17.7 uas vs 8.2 uas for the Kuo+2015 set,
  and only 1/19 has a measured acceleration.
- NGC6264's sigma_a replacement is large: 33/34 rows changed, median ratio
  2.9x, range 0.34-16.5x.

### Unpruned table for clipping tests

`scripts/megamaser/build_unpruned_dataset.py` constructs the `unpruned`
dataset as the spot-wise union of the two tables. For NGC5765b and NGC6264 it
keeps every `original_published` row and its published astrometry, replaces
acceleration fields from `fiducial` where the velocity matches, and retains
the published acceleration for a spot absent from `fiducial`. UGC3789 uses the
complete fiducial row for each of its 153 matches and the published row for
each of the three spots absent from the fiducial table.

NGC6323 is deliberately different. Dom's response reports 19 added spots but
no clipped spots, so its `unpruned` table is now field-for-field and row-order
identical to the 87-row `fiducial` table. The earlier hybrid of Kuo et al.
(2015) rows, raw Kuo et al. (2011) astrometry and fiducial accelerations was
internally inconsistent with that provenance and also misaligned a fiducial
`r_ang` initialisation with the reordered rows. The resulting counts are
165, 358, 212, 66, 87 and 156 for CGCG074-064, NGC4258, NGC5765b, NGC6264,
NGC6323 and UGC3789, respectively. A generated `provenance.csv` records the
source of the astrometry and acceleration for every row, plus a
`clipped_by_pesce` flag for published velocities absent from the fiducial
table.

The loader now retains and masks the 20 `1.000 +/- 1.000` systemic placeholder
rows. It separately masks the 123 high-velocity `a = 0.000 +/- 0.200` rows;
neither convention contributes an acceleration likelihood term.

The quantitative NGC5765b diagnostics below were computed before that
restoration, on the journal's 192-row electronic table. They must be rerun
before being quoted as results for the complete 212-row `original_published`
dataset.


## 2. Are the removed spots special?

Per-spot chi^2 from the CANDEL MAP fitted to the **full** published table (so
the fit is dragged toward the outliers — conservative).

| galaxy | kept median | removed median | mean ratio | Mann-Whitney p |
|---|---|---|---|---|
| NGC5765b | 0.38 | 3.05 | 4.5x | 3e-6 |
| NGC6264 | 0.68 | 4.22 | 3.5x | 1e-3 |
| UGC3789 | 0.83 | 12.10 | 9.1x | 1e-5 |

UGC3789's three removals rank 1st, 2nd and 6th worst of 156. NGC6264's rank
1, 4, 6, 13, 25 of 66. So the removals **are** disk-model outliers and the
fit-and-clip story is corroborated independently.

But the clip is not neutral, in three ways:

1. **NGC5765b's clip is not clean.** 7 of its 23 removed spots fit fine at
   convergence (chi^2 ranks 76, 95, 97, 100, 139, 160, 182 of 192; three below
   the *kept* median). They are not >3 sigma outliers of any converged fit —
   consistent with clipping against an early, poor fit and never reinstating.
   **The final table is therefore not the fixed point of any stated criterion.**
2. **It preferentially deleted the most informative spots.** NGC5765b's removed
   spots have median positional uncertainty 10.5 uas vs 24.5 uas kept (27th
   percentile of the kept distribution) and 65 per cent carry measured
   accelerations vs 32 per cent of kept.
3. **It concentrates in the distance-bearing subsample.**

Weight accounting (positional weights use the fitted floors):

| galaxy | spots removed | positional weight | acceleration measurements |
|---|---|---|---|
| NGC5765b | 12.0% | 18.9% | 21.7% |
| NGC6264 | 7.6% | 7.4% | 8.1% |
| UGC3789 | 1.9% | 1.5% | 2.5% |

Class/radius concentration: NGC5765b lost 25 per cent of its systemic spots
(10/40) against 4 per cent of blue — systemic spots carry the accelerations
that set D (hypergeometric p = 7.6e-3, 2.4 sigma). NGC6264's removals sit at
radial ranks 1, 4, 5, 8, 60 of 66; 4-of-5 in the outermost 8 has p = 4.6e-4 —
direct leverage on the rotation curve.

**A distance-blind criterion does not imply a distance-neutral outcome.** That
is the point to press, not the vetting itself. Do not over-claim: these show
the outcome was strongly non-exchangeable and concentrated in the
distance-bearing data. The *sign* of the induced shift in D is an empirical
question, answered by the run trio in section 5.

Also avoid ratifying the clip's frame. "The removals are genuine outliers"
concedes too much — the population has elevated chi^2, but at least 7/23
individuals are not outliers by P20's own converged standard.


## 3. Why hard clipping fails here specifically

- **The arithmetic refuses the noise-tail reading.** The NGC5765b electronic
  table used for this calculation has 192 spots x
  ~3.3 channels ~ 640 measurements; a two-sided 3 sigma clip on calibrated
  Gaussians expects ~1.7 excursions. Twenty-three were removed. The clip was
  removing structure the model cannot represent — which is exactly the MCP's
  own physical account (off-disk emission), and that account is the strongest
  argument for modelling it as a population rather than deleting it.
- **The clip and the fitted error floors fight each other.** Clip high-chi^2
  spots -> kept residuals truncated -> fitted floors shrink (they solve roughly
  sum z^2 ~ N on the kept set) -> likelihood sharpens -> more spots exceed
  3 sigma -> clip again. A coupled fixed-point iteration that was never run to
  a fixed point. Even at one it gives floors biased low and a **D posterior
  that is too narrow** — and precision is the entire product here.
- **The selection is uncorrectable.** With a fixed stated criterion you could
  write the truncated likelihood (divide each kept spot by Pr(survive | theta)).
  Because the procedure is iterative, order-dependent and non-convergent, the
  selection function is not reconstructible even by its authors. The likelihood
  of the clipped table is not the likelihood of any well-defined
  data-generating process.
- **Effective dof accounting is zero when it should be ~n.** Clipping is MAP
  estimation over n binary inclusion indicators, optimised greedily against fit
  quality, then priced at zero. Post-selection inference (Berk et al. 2013).


## 4. Rejected treatment: per-spot good/bad mixture

This section records the mixture model that was considered and later tested.
It did not work adequately on the megamaser data and was removed; it is not a
current recommendation or an implemented inference option.

Box & Tiao (1968); Hogg, Bovy & Lang (2010). Per spot, marginalise a Bernoulli
label analytically:

    p_i = f * L_good,i + (1 - f) * B_i

`L_good,i` is **exactly the per-spot (r, phi) marginal the code already
computes**. Rejected alternatives, with reasons, in section 4.4.

### 4.1 Where it goes in the code

All four consumers build a per-spot array and then reduce it. One line each:

| function | approx line | current | becomes |
|---|---|---|---|
| `_sum_phi_marginal` | 2134 | `total + jnp.sum(ps)` | `total + jnp.sum(logaddexp(log_f + ps, log1mf + logB[idx]))` |
| `_sum_phi_fixed` | 2101 | `total + jnp.sum(ps)` | same |
| `_eval_phi_marginal` | 2033 | `result.at[idx].set(ps)` | set the mixed value |
| `_eval_phi_fixed` | 2086 | `result.at[idx].set(ps)` | same |

(all in `candel/model/model_H0_maser.py`)

`logB` is a precomputed length-`n_spots` constant vector with two distinct
values, 3-channel and 4-channel. Membership falls out free:
`q_i = exp(log_f + ps_i - mixed_i)`.

Parameters added per galaxy: **one** (`f`, bad fraction, prior U(0, 0.5)).
`B` is fixed, never sampled. Error floors stay free with their existing
informative priors. The `q_i` are derived, not sampled.

### 4.2 Choosing B — the only real judgement call

B must be a normalised density over the same observables in the same units as
`L_good,i`. Two objects per galaxy: 3-channel spots and 4-channel spots.

Because the bad spot is still *measured* by VLBI, the honest form is
`B_i = integral N(d_i | s, Sigma_i) Q(s) ds`, but Q is broad compared with
Sigma_i so `B_i ~ Q(d_i)`, and with uniform Q it is **literally a constant**.
You are choosing three numbers per galaxy:

- **A_field** — the VLBI imaging field, from the observing paper.
- **V_v** — the spectrometer bandwidth searched.
- **V_a** — the acceleration range the multi-epoch drift fit could return.

These are the **detection volume**: where a spot could have been found. An
instrumental lookup, not a prior. **Trap:** do not use the observed span of
detected spots — that is what was found, not what could have been. It
understates V, inflates B, and over-flags.

B enters the threshold logarithmically:

    chi2_crit ~ 2 * { ln[L_peak / B] + ln[f / (1-f)] }

so a factor 3 error in B moves the effective cut by 2 ln 3 ~ 2.2 in chi^2 —
about 3 sigma to 2.8 sigma. That is the precise sense in which the choice does
not matter much.

**Key correction:** the two latents saturate the positions, so the position
channels contribute only ln(A_field / A_disk) ~ O(1), not the raw
span-over-sigma ratio. Essentially all discrimination lives in v and a.

Implied thresholds using the fitted floors and observed spans as placeholders
(so these are lower bounds on V, hence upper bounds on flagging):

| galaxy | n | n_accel | V_v (km/s) | sigma_v,hv | V_a | sigma_a | chi2_crit 3-ch (1 dof) | chi2_crit 4-ch (2 dof) |
|---|---|---|---|---|---|---|---|---|
| NGC5765b | 192 | 69 | 1544 | 2.81 | 1.83 | 0.22 | 17.2 | 19.5 |
| NGC6264 | 66 | 37 | 1384 | 3.01 | 4.87 | 0.18 | 16.8 | 21.6 |
| UGC3789 | 156 | 80 | 1441 | 2.06 | 8.80 | 0.31 | 17.7 | 22.5 |
| NGC6323 | 68 | 29 | 1060 | 1.74 | 2.07 | 0.30 | 17.4 | 19.4 |
| CGCG074-064 | 165 | 145 | 1768 | 2.09 | 9.17 | 1.03 | 18.0 | 20.6 |

All land at ~4.0-4.2 sigma equivalent — consistent across five quite different
setups, which suggests the construction is stable rather than accidental. Note
it is **stricter** than the nominal 3 sigma P20 describes, so a principled B
should flag noticeably fewer spots than were removed. That is a finding, not a
defect.

**Never fit B's scale.** A free width shrinks onto the outliers, becomes a
second "good" component, and takes f and the floors degenerate with it.

**Useful inversion:** ask what B would be needed to make P20's 23 NGC5765b
removals the flagged set, then compare the implied detection volume with the
real field of view. Converts "too aggressive" into a number.

### 4.3 Interaction with the (r, phi) latents

**Membership must be defined on the marginal — this is structural, not taste.**
The label sits above the latents (good -> latents -> obs; bad -> obs ~ B, no
disk coordinates), so `L_good,i` is the (r, phi) marginal.

Consequence: because the phi posterior is bimodal under reflection, **a spot
whose point estimate lands on the wrong branch looks like a gross outlier while
its marginal is healthy.** Some of the 7 reinstatable NGC5765b spots may be
exactly this. Checkable: compute per-branch marginal masses for the clipped
spots and report how many are branch flips.

The good component's latent freedom does *not* swallow the bad component: the
2 latent dof are the same for every spot, do not grow with model complexity,
and the marginal carries an Occam penalty (fitted sliver / prior volume). The
real degeneracy is the **triangle floors <-> f <-> B-width**, broken by scale
separation: informative floor priors, B fixed orders of magnitude wider than
any admissible floor, f bounded below 0.5.

### 4.4 Rejected alternatives

- **Student-t / hierarchical per-spot inflation** — the same idea twice (a t
  *is* a Gamma scale mixture). Non-redescending influence (decays like 1/z, so
  precise off-disk spots still pull D); models heavy-tailed *noise* when the
  problem is a distinct *population*; and it destroys
  `_neg_half_chi2_quadform` and the coefficient hoisting, needing a per-spot
  lambda grid outside the (r, phi) grids. 10-50x the work for weaker
  robustness. **Not worth it, not even as an appendix variant.**
- **Trimmed likelihood / LTS with inferred fraction** — not a probability model;
  hard clipping with extra steps. **Write one pre-emptive paragraph rejecting
  it, because a referee will suggest it.**
- **Explicit physical off-disk population** — unidentifiable at 3-23
  contaminants. Fold the physics into B's *shape* instead.
- **Escape hatch if the volume choice is unpalatable:** bad component = the same
  disk model with errors inflated by fixed kappa (~10). Units automatic, no
  support needed. But not redescending, and it swaps V for kappa rather than
  escaping the choice. Legitimate cross-check, not the primary model.

### 4.5 DE vs MCMC

- **DE (`--sampler de`, runs on `_sum_phi_marginal`) — clean.** `ps` there *is*
  the per-spot marginal, the correct object. `f` becomes one more L-SHADE
  dimension; gradient-free so nothing to differentiate; `remat=False`
  unaffected. **Pre-empt the unboundedness objection:** mixture MAP is
  unbounded when a component collapses, but B is fixed and the good component's
  widths are floors under informative priors, so nothing can collapse. Cost:
  `f` correlates with the floors, adding a soft ridge — budget more iterations.
- **MCMC (`--sampler mcmc`, explicit-latent (r, phi) NUTS) — exact but
  pathological.** The mixture on `_sum_phi_fixed` is valid (the joint factorises
  as `p(r,phi)[f p(d|r,phi) + (1-f) B]`), but for q_i ~ 0 the latent conditional
  collapses to the prior. Then: bad spots' latents random-walk; the per-side
  (2,n,2,2) covariance adaptation estimates prior width; teleport acceptance
  -> 1 and the diagnostic goes uninformative; and **borderline spots
  (q_i ~ 0.5) give a sharp-mode-plus-flat-basin geometry NUTS handles badly.**
  Also the conditional-r grids are seeded around good-model solutions and may
  under-cover a flat conditional (the union with the global scan should
  protect, but verify).
- **Resolution: run the MCMC on the globals-only marginal**, the same target DE
  uses. Not new infrastructure — `_sum_phi_marginal` already takes
  `remat=True` "to save memory in the backward pass", i.e. it is already
  differentiable for a gradient-based caller. Dimension collapses from
  globals + 2n latents to globals alone (NGC5765b: ~15 + 384 -> ~15). Steps
  cost more, geometry is far better. **And it independently removes the phi
  bimodality — the sampling pathology we criticise P20 for.**
- Evidence comes free: `maser_map.marginal_loglik` calls `_sum_phi_marginal`,
  so the harmonic estimator and Savage-Dickey inherit the change.


## 5. The warp-order degeneracy

**The heart of the matter.** On the 192-row NGC5765b electronic table there is a
significant preference for a quadratic warp (negative d2i/dr2). On P20's
clipped table (169) it vanishes by every measure they computed, and permitting
it shifts D by only +1.4 Mpc there.

So outlier removal and model flexibility are **competing explanations for the
same residuals**, and hard clipping at fixed model order silently picks one.

**Layer 1, for the headline D: do not select, marginalise.** The quadratic warp
is a continuous superset (zero is an interior point) and `use_quadratic_warp`
already exists. Fit the full table with mixture *and* quadratic warp under a
physically motivated prior, and quote D marginalised over d2i/dr2, f and the
memberships. If the joint posterior shows an **f <-> d2i/dr2 anticorrelation
ridge, that ridge is the result** — the +1.4 Mpc becomes a propagated error
term instead of an analyst's choice. Show that 2D posterior.

**Layer 2, for interpretation.** With the mixture in place, never on clipped
data: Savage-Dickey on d2i/dr2 with the prior-scale sweep shown, not buried;
membership-ranked deletion scan (drop the k lowest-q spots, k = 1..10);
per-radial-bin Delta ln L profile (real curvature is smooth and distributed,
spot-driven curvature spikes); cross-channel coherence through sin i / cos i;
and **injection tests measuring the confusion rate** between "linear warp +
planted contaminants" and "quadratic warp, no contaminants". Without the last,
"Savage-Dickey favours X" has unknown error rates in exactly the entangled
regime that matters. Evidence that the quadratic warp is real requires all
four; P20's implicit adjudication (clip, then find no curvature) fails all
four.

### Run trio (the paper's core argument)

1. clipped table + Gaussian likelihood — P20 replication
2. full table + Gaussian likelihood — the damage taken naively
3. full table + mixture — the result

Quadratic warp active and marginalised in all three. **Headline number:
(3) - (1) on D for NGC5765b.**

**Pre-registered check:** the 7 reinstatable NGC5765b spots must return
q_i ~ 1, and the 3-12x chi^2 spots q_i ~ 0.


## 6. Diagnostics to ship

- **PSIS-LOO** (Vehtari, Gelman & Gabry 2017). Store per-draw *per-spot*
  log p_i (already computed groupwise — scatter, do not sum). Pareto k-hat > 0.7
  *is* the influence screen; those spots get exact case-deletion refits.
- **Per-spot Delta D_i influence table**, annotated with q_i, class, radius,
  has-accel. This is the honest replacement for a hidden data cut.
- Leave-one-class-out and leave-radial-bin-out (targets the NGC6264 outer-spot
  issue directly).
- Posterior predictive checks: channel-wise PIT/rank histograms of q-weighted
  residuals. The clipped analysis fails these by construction.
- **One sensitivity table.** Rows: B volume x3 and /3; f prior Beta(1,9) vs
  U(0,0.5); per-class f; accel-channel mixture on/off; floor prior widths x2;
  linear vs quadratic vs marginalised warp. Columns: D, f, sigma-floors.


## 7. Reproducing the clip forensically (complementary, not a substitute)

**Frame it as a multiverse / specification curve** (Steegen et al. 2016;
Simonsohn et al. 2020), not as reproduction. Exact recovery of 23/5/3 is
unlikely — undocumented statistic, threshold, round structure, reduction
vintage, plus human judgement. Aiming at their set and missing looks like
failure. Aim at the *family* of datasets a defensible >3 sigma fit-and-clip
could produce, and locate P20's inside it. That spread is the systematic.

```
active = all spots
for it in range(max_iter):
    fit = MAP(spots=active, warp=W, floors=F)     # warm-start after round 1
    z   = residual_stat(fit, ALL_spots)           # score ALL, always
    flagged = {i : z_i > thresh}
    active = (complement(flagged) if reinstate
              else active - (flagged & active))
    record(active, D, floors, z)
    if active unchanged: break
```

**The detail that matters most:** score every spot every round, including
already-removed ones. Costs nothing (evaluations at fixed globals) and gives
the "would this have been reinstated?" flag free — the direct test of the
7-spot anomaly.

Axes, highest-value first:

| axis | options | why |
|---|---|---|
| reinstatement | one-way vs re-test | explains the 7-spot anomaly |
| warp order while clipping | none / linear / quadratic | the degeneracy, operationalised |
| floors refit each round | yes / held fixed | isolates the shrinking-floor cascade |
| clip statistic | see section 8 | Dom says only "a residual statistic" |
| threshold | 2.5 / 3.0 / 3.5 / 4.0 | "approximately >3 sigma" |
| removal per round | all above / worst-1 / worst-k | worst-1 is maximally path-dependent |
| initial fit quality | converged vs under-converged | Dom says "*initial* disk fits" |

The under-converged option may be **more** faithful: the 7 reinstatable spots
are what you get from clipping against an early poor fit and never re-testing.

**Build it on `fit_disk`, not DE.** (i) Faithfulness — the vetting predates P20
and used MCP code; `fit_disk` is that code, and DE optimises a marginal
objective nobody had in 2019. (ii) Cost — Fortran, cheap enough to brute-force
the grid; DE would be tens of GPU-hours. (iii) It is already wired up:
`check_reid/prepare_reid_data.py` generates `*_loader_reid.inp` (generated,
untracked, dataset-namespaced), so the loop just rewrites the `.inp` with the
active subset each round. `reid_chi2.loglik_context(galaxy, n_spots, dataset)`
takes `n_spots`, so rebuild per round. Confirm a few endpoints with DE.
Bookkeeping: `init.r_ang` is per-spot, so subsetting the active set means
subsetting the init array.

Plots: specification curve of D with P20 marked; Jaccard overlap with P20's
removed set per configuration; D trajectory across iterations (drift = "blind
but not neutral", made visible); floor trajectory (does the cascade fire?);
and the money plot, **P(quadratic warp preference survives | configuration)**.

It **cannot** prove what they did. Say so. Division of labour: the sweep is
forensic and motivates the mixture; the mixture is inferential and is the
answer. That is also the order they should appear in the paper.


## 8. The residual statistic

`reid_chi2.neg_half_chi2` gives per-spot -0.5 chi^2 from Reid's
`calc_warped_model` + `add_error_floors` at fixed (r_ang, phi), i.e. four
standardized residuals z_x, z_y, z_v, z_a with floor-inflated sigmas.

**The subtlety that decides everything:** two fitted latents per spot against
3-4 observables. So per-spot residual dof is **1** (no accel) or **2** (with);
the z's are shrunk and mutually correlated because the latents absorb them
((r, phi) -> (X, Y) is a generically invertible 2->2 map, so the fit can nearly
zero the positions and push residual into v and a). **"3 sigma" is a nominal
label, not a tail probability**, and must be calibrated. This cuts in our
favour on the headline anomaly: shrinkage makes |z| > 3 rarer than Gaussian,
so 23 removals from ~640 measurements is even more excessive than it looks.

Candidates:

| # | statistic | clip at | comment |
|---|---|---|---|
| A | max(\|z_x\|,\|z_y\|,\|z_v\|,\|z_a\|) | 3 | most literal reading |
| D | position only | 3 | see below |
| E | velocity only | 3 | |
| F | studentized version of any | — | corrects latent leverage |

Dom's follow-up selects A as the closest literal reconstruction: inspect each
coordinate's normalised residual at approximately 3 sigma. The historical
procedure was iterative and sometimes removed coherent regions, so it still
cannot be reconstructed as a deterministic spot-wise rule from this statement
alone.

### Conditional latent-posterior diagnostic now implemented

After DE, `run_de_map.py` fixes the globals to their MAP values and integrates
the Dom-style statistic over each spot's existing deterministic conditional
`(r_ang, phi)` grids:

    z_ij(r, phi) = (d_ij - m_ij(theta_MAP, r, phi))
                   / sqrt(sigma_ij^2 + sigma_floor,j,MAP^2)
    S_i = integral L_xyv,i(r, phi | theta_MAP)
                   max_(j in {x,y,v}) |z_ij(r, phi)| dr dphi
          / integral L_xyv,i(r, phi | theta_MAP) dr dphi

Acceleration never enters either the maximum or the latent-posterior weights.
Each DE run writes the x, y and velocity posterior mean absolute residuals and
`S_i` beside its checkpoint as
`*_posterior_outliers.csv` and `.png`, marking `S_i >= 3` as a diagnostic
flag. This averages the residual magnitude, not the signed residual or a
three-sigma exceedance indicator, so latent-posterior sign changes do not
cancel. It uses no latent sampling and is not run by MCMC. Ordinary DE only
reports it; clipping and refitting require the explicit iterative-clipping
flag.

**Why the channel choice is the whole ballgame.** Positions are *angular*, so a
position residual genuinely carries no distance information. But the velocity
residual depends on M/D and the acceleration residual on M/D^2 — and D is
determined precisely by the consistency between those two. So **a statistic
using both v and a is clipping directly on the comparison that determines the
distance**: blinded in Dom's sense (no D in the formula), and not blind at all
in the sense that matters. A and F have this property; D does not. Which
channels entered therefore decides whether the blindness claim is strong or
vacuous — and it is empirically testable.

**Cheap way to identify it, no iteration needed.** We already know the answer
set (23/5/3). At a converged fit on the full table, rank spots by each
candidate and compare the top-k with P20's actual removals (Jaccard / rank
overlap). Whichever ranking reproduces the known set is very likely theirs.
This needs only a per-channel decomposition of `_fixed_phi_per_spot` (which
currently sums the four terms) plus the comparison — an afternoon, and it
collapses seven sweep axes to one or two before any GPU time. Encouraging
prior: the single-fit total-chi^2 ranking already put UGC3789's three removals
at ranks 1, 2, 6 of 156.

Then calibrate the threshold by simulation: mocks from the fitted model at the
observed layout and precision, refit including latents, record the statistic.
"3 sigma" = the 99.73rd percentile of *that* distribution.

**Do this first:** the vetting happened in the era of the source papers, and
`Kuo2011_MCP_III_ReadMe.txt` is already in `data/Megamaser/`. Reid+2009/2013,
Kuo+2011/2013/2015, Gao+2016 and Humphreys+2013 may document their own outlier
rejection explicitly, which would replace the whole reconstruction with a
citation.


## 9. Remaining open questions for Dom

1. **Were removed spots ever re-tested** against later fits, or was removal
   one-way? (Directly explains our 7 reinstatable NGC5765b spots.)
2. **Were the error floors refitted between clip rounds?** If so the
   shrinking-floor cascade applies.
3. **What model was used for the vetting fits** — which warp order, and was it
   the same parameterisation as P20?
4. **NGC6323: why were the 68 published spots' positions and uncertainties
   modified**, not merely augmented with 19 Kuo+2011 spots? This is not
   mentioned in the letter (section 1).
5. **NGC6264: what is the internal acceleration analysis** that replaced the
   published sigma_a (median 2.9x, up to 16.5x)?
6. Are the pre-2019 vetting fits or logs archived, and can the clip code be
   shared?
7. Do the earlier MCP papers document this rejection, and if so where?

Also outstanding from their side, unrelated to clipping: whether our 2M++
peculiar velocities match theirs (their Table 4 mean is positive and moves H0
73.9 -> 71.8; ours shows essentially no change).


## 10. Where things stand

The `unpruned` dataset builder, MAP-conditional posterior-mean sigma
diagnostic, and cumulative iterative DE fit-and-clip loop are implemented.
The loop is launched with `--dataset unpruned --iterative-clip-sigma`, using
2.5 sigma by default, and stops on an unchanged mask or a maximum attempt
count. A stabilised mask is installed for the `clipped` dataset, which filters
the unpruned rows without modifying source tables; last-attempt flags from an
unstabilised run remain explicitly pending. The mixture
likelihood was tested and removed; calibration of the clipping threshold and a
reinstating variant remain proposals. The numbers above are reproducible from
`load_megamaser_spots` on the datasets plus
`candel.model.maser_map.evaluate_at_globals` at the DE MAP points in
`scripts/megamaser/init_original_published.toml`.

Suggested order when picking this up:

1. Literature check on stated outlier rejection in the MCP source papers
   (section 8, last paragraph) — may moot much of section 7.
2. Compare the posterior-mean sigma ranking against the known removed sets
   (section 8) — cheap, high value.
3. Calibrate its three-sigma threshold with fitted-model simulations.
4. If clipping is retained, add a reinstating comparison and report distance
   sensitivity to the threshold.
5. The fit_disk multiverse sweep (section 7), scoped by what steps 1-3 find.
