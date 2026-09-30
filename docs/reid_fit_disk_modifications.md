# Modifications to Reid's `fit_disk` (v24d) in `fit_disk_Reid_reflection`

Notes on every change in
`background_info/fit_disk_Reid_reflection/fit_disk_v24d_unblinded.f`
relative to Reid's original code (kept pristine in
`background_info/fit_disk_Reid/fit_disk_v24d_unblinded.f`).
Both trees are deliberately untracked; sync to the cluster with
`background_info/sync_fit_disk_to_glamdring.sh` before submitting
(a stale binary silently ignores newer control tokens).

**Bitwise-identity guarantee:** with every toggle off (`use_gibbs=F`,
`p_reflect=0`, or a legacy 2-token control line) the modified binary
reproduces the original sampler's `fort.7` byte for byte (md5 gate,
re-verified after each change, most recently 2026-07-02 for the 2-, 5-,
6- and 7-token control lines).

## Control-file interface

All switches live on one trailing control line, parsed with an
iostat+backspace cascade so older files keep working:

```
use_gibbs  p_reflect  [n_inner]  [use_gcov]  [track_lat]  [use_eta]  [n_write_thin]
    T        0.25        20          T            F           T          1
```

Defaults when a token is absent: `n_inner=1`, `use_gcov=F`,
`track_lat=F`, `use_eta=F`, `n_write_thin=1`. The shell tooling
(`run_gibbs_chains.sh`) writes all seven tokens and **defaults
`use_eta` to T** (`--eta F` / `--no-eta` for the original
parameterization); the raw control-file token still defaults to
`n_write_thin=1` (no thinning, preserving the legacy-file bitwise
guarantee), but the shell tooling (`run_gibbs_chains.sh`,
`run_gibbs_comparison.py`, `submit_gibbs_comparison.sh`) defaults
`--write-thin` to **10**.

## Sampler changes (the ladder)

Each change targets one failure mode of the original sampler found
while comparing against CANDEL's converged NGC 6323 posterior
(NUTS on globals + Gibbs on latents).

### 1. Metropolis-within-Gibbs restructuring (`use_gibbs`, 2026-06/07)

Original: ALL parameters (20 globals + 2 per spot, 150–800+
dimensions) proposed in ONE joint diagonal Metropolis block with a
single accept/reject; acceptance collapses with dimension and
systemic-spot phi never mixes (inflated `sigma_vsys` floor ~14 km/s,
wrong `y0` sign).

Modified (`emcee_gibbs_exploration`): (a) a globals block scored
against the full data set; (b) every spot's `(r, phi)` swept as its
own 2-D block scored only against that spot's `(x,y,V,a)` quartet via
`calc_ln_p_one_spot` — valid because spots are conditionally
independent given the globals, and O(1) per spot.

### 2. Azimuthal reflection move (`p_reflect`)

HV spots are near-degenerate under phi -> reflection about the phi
support centre (mode boundary at exactly +-90 deg, where the original
code *seeds* them). A probability-`p_reflect` move proposes the mirror
mode using per-spot, per-side 2x2 `(r, dphi)` covariances (Welford
accumulation during burn-in, per-side Cholesky), detailed-balanced;
systemic spots are excluded (measured acceleration breaks the
degeneracy). Fixes HV bimodality; exposes the true (broader)
posterior.

### 3. Inner latent sweeps (`n_inner`)

Repeat the full per-spot sweep `n_inner` times per Gibbs iteration
(CANDEL's `n_inner` analog; latents equilibrate against fixed globals
between global proposals). `n_inner=20` drops every R-hat; latent
sweeps are cheap relative to the globals' full-data evaluation.

### 4. Adaptive global covariance (`use_gcov`)

Original global proposal is a diagonal walk plus a hard-wired
straight-line correlated Ho–M/Ho–Vcor heuristic (`HoM_trials`
control fraction). Neither tracks the curved H0–D–Mbh degeneracy.
`use_gcov` (requires `use_gibbs`): Welford mean/co-moments of the free
globals over the secondary burn-in first half -> midpoint Cholesky
activation at Roberts–Gelman–Gilks scale 2.38/sqrt(d) ->
Robbins–Monro scalar tuning to 23.4% acceptance over the second half
-> end-of-burn-in covariance refresh (tuned scale kept) -> frozen for
production (valid MH). Diagonal-walk fallback on any failure.

### 5. Eta reparameterization (`use_eta`, 2026-07-02)

The endpoint. Even with gcov, the wide-open Ho window exposes the
ridge M ∝ 1/H0 — a *hyperbola* in Reid's sampled `(H0, M)` plane that
no fixed-shape random walk can traverse (4x1M chains: R-hat 2.7,
n_eff ~2 on H0/D/Mbh). `use_eta` switches to CANDEL's coordinates:

- `params(2)` becomes `eta = log10(Mbh/Msun) - log10(D_A/Mpc)`.
  Rotation velocities constrain only `Mbh/D_A`
  (v ~ sqrt(M/(D*theta))), so eta is velocity-pinned (posterior std
  ~0.004) and the ridge is axis-aligned in `(H0, eta)`.
- Init conversion uses the same integer `Ez_int` lookup as
  `calc_warped_model`, so the init point round-trips exactly
  (NGC 6323: init eta = 4.96747 vs CANDEL config eta = 4.9676).
  Step size converted via d(eta) = dM/(M ln10).
- `calc_warped_model` forms `BH_mass = 10^eta * D_A` after `D_A`.
- `calc_ln_P_priors` adds `-2 ln Ho` — the |dD_A/dHo| = D_A/Ho
  Jacobian up to a <1e-3-nat ln A(V) term — so flat-(Ho, eta) sampling
  targets a measure **flat in (D_A, eta)**: exactly CANDEL's
  `uniform_D_A` distance prior. This supersedes the old post-hoc
  `--reweight-da2` corner reweighting (which overshot; see below).
- `HoM_trials` is forced to 0 (the anti-correlation line is
  meaningless in eta); the multi-strand "Mbh anti-correlated with Ho"
  seeding is skipped (dispersing Ho alone spreads strands along the
  ridge); a Gaussian prior on the mass line triggers a clean STOP
  (flat required).
- `fort.7` keeps the **Mbh column** (eta back-converted at the write),
  so `load_chain` and all downstream comparison tooling are unchanged.
  The `.prt` diagnostics print eta with parameter name
  `eta=lg(M/D_A)`.

## Robustness / infrastructure changes

- **NaN-bootstrap guard** (`emcee_gibbs_exploration`): non-finite
  proposals are rejected outright; on a walker's first call a NaN
  proposal can no longer be "accepted" against the 0.0 bootstrap
  sentinel (which permanently poisoned the walker); if the walker's
  own state is NaN it falls back to the primary burn-in point.
- **Progress heartbeat**: flushed burn-in stage lines, secondary
  burn-in progress every 25%, production every 5% (`tail -f`
  friendly on cluster nodes).
- **fort.74 latent tracking** (`track_lat`, default F): full `(r,
  dphi)` chains for up to 5 HV spots at fort.7 cadence (~1.5x fort.7
  volume), enabling latent-level Gelman–Rubin. dphi is wrapped to
  (-180, 180] about the spot's phi centre.
- **Dimensioning**: `max_stored` 500000 -> 1000000 (1M stored
  production samples per chain).
- **Output thinning** (`n_write_thin`, control-file token defaults to
  1, shell tooling defaults `--write-thin` to 10, 2026-07-02): writes
  only every `n_write_thin`-th STORED sample to `fort.7`/`fort.74`.
  Decoupled from the `params_stored`/`n_s` bookkeeping used by the
  Ho-M correlated-trial bootstrap, which still sees every stored
  sample regardless -- `n_write_thin` only shrinks the chain files
  (and the collect step's parse time) without changing the sampler's
  internal state. At the pipeline's default `--num-samples 1000000`
  (== `max_stored`, so `n_skip=1`, every iteration stored) each
  `fort.7` is ~195 MB/1M rows; `n_write_thin=10` cuts that 10x.
  Exposed via `run_gibbs_chains.sh --write-thin N`,
  `run_gibbs_comparison.py --write-thin N`, and
  `submit_gibbs_comparison.sh --write-thin N`.

## What is NOT changed

The physics and likelihood are untouched: `calc_warped_model`'s disk
model equations (warp, Kepler, relativistic terms), `dampc` /
`Ez_int` cosmology, the error-floor treatment (`add_error_floors`),
the Gaussian/outlier-tolerant data likelihood, and the Gaussian prior
machinery. `use_eta` changes only the *coordinates* and the *measure*
of the target, plus which built-in proposal heuristics are active.

## Priors / measure (Reid original vs use_eta vs CANDEL)

| | sampled coords | induced measure on (D_A, eta) |
|---|---|---|
| Reid original | flat (H0, M), hard Ho window | ∝ 1/D_A |
| `use_eta` | flat (H0, eta) + `-2 ln Ho` Jacobian, hard Ho window | flat — matches CANDEL |
| CANDEL | flat (D_A, eta) (`uniform_D_A`), D box | flat |

Hence `--reweight-da2` (D_A^2 on top of the original) overshoots by
one power of D_A; do not combine it with `use_eta`. Error-floor
priors: Reid flat vs CANDEL truncated Gaussians (N(2,1) km/s etc.);
`run_gibbs_chains.sh --match-priors T` rewrites the floor lines with
the CANDEL priors. The hard Ho window (control line 3 +-10 km/s/Mpc)
is part of the posterior; keep it wide or map it onto CANDEL's per-
galaxy D box.

## Validation status (NGC 6323)

- Ladder steps 1–3 (`eta_gibbs`, n_inner=20, narrow Ho window):
  converged, R-hat <= 1.04 everywhere; matches CANDEL on H0/D/y0/
  sigma_vhv; flat-prior sigma_vsys ~5 vs CANDEL ~2 is the floor
  prior, not mixing.
- Step 5 smoke (2026-07-02, single chain 300k+300k, Gibbs+reflection+gcov config,
  wide-open Ho window): `use_eta=F` stays on a narrow ridge segment
  (D 5–95% [112, 167] Mpc — the R-hat-2.7 pathology); `use_eta=T`
  traverses the full posterior (D [84, 260], H0 [30, 93]) and lands
  on CANDEL: H0 median 54.1 vs 53.6, D_A 145.2 -> D_c ~149 vs CANDEL
  149 [96, 262]. Corner:
  `results/Megamaser/reid_mcmc/NGC6323_eta_smoke_corner.png`.
- Residual known metastability: HV mode-configuration hopping in the
  reflection variants (`sigma_vhv` n_eff ~9 in the smoke run) —
  latent-space, unaffected by global proposals; `eta_gibbs` avoids
  it.

## Tooling entry points

- `scripts/reid/run_gibbs_chains.sh` — compile once,
  launch N independent one-strand chains; owns the control-line
  rewrite (`--gibbs --reflect --n-inner --global-cov --eta
  --track-latents --match-priors --write-thin`); eta defaults ON,
  write-thin defaults to 10.
- `candel_maser/reid/run_gibbs_comparison.py` — variants
  reid_original / eta_reparam / eta_gibbs / eta_gibbs_reflection,
  multi-chain R-hat, numpyro summaries, CANDEL overlay corner
  (`--eta/--no-eta`, default on; corner H0 axis fixed to [5, 200] via
  `compare_reid_candel.PLOT_RANGES`). `reid_original` is Reid's true
  original formulation — joint one-block sampler AND flat-(H0, M)
  coordinates — pinned eta-off regardless of the flag, so the default
  four-variant run always carries the untouched-baseline reference;
  `eta_reparam` is the joint sampler in the coordinates selected by
  `--eta`. `--reweight-da2` is applied only to eta-off variants.
- `scripts/reid/submit_gibbs_comparison.sh` — one
  cluster job per variant + dependent collect job (`--eta/--no-eta`,
  default on).
