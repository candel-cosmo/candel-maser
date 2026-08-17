# Megamaser joint-H0 sweep summary

Status as of 2026-08-13. This note summarises the outputs pulled locally after
running the CANDEL- and P20-distance sweeps on Glamdring:

```bash
./scripts/megamaser/submit_sweep_H0.sh -q cmbgpu \
    --dataset fiducial,unpruned,clipped,original_published --cpus 2
./scripts/megamaser/submit_sweep_H0.sh -q cmbgpu \
    --dataset fiducial --distance-source p20 --cpus 2
```

The local audit covers 55 stage-2 HDF5 chains below
`results/Megamaser/<dataset>/H0/`. The CANDEL-distance sweep is complete, with
all 44 expected chains: 11 modelling variants for each of the four spot-table
datasets. All 11 expected P20-distance chains are present under `fiducial`, so
that sweep is also complete. All 55 chains are finite and readable. Each used
one chain, 3,000 warmup steps and 15,000 retained draws.

The joint inference uses KDE likelihoods constructed from single-galaxy
angular-diameter-distance posteriors for CGCG074-064, NGC5765b, NGC6264,
NGC6323 and UGC3789. NGC4258 is not part of the joint H0 inference.

## Dataset definitions

The dataset choice changes the maser-spot tables used in the stage-1
single-galaxy fits. Stage 2 then uses the matching saved `D_A` chains; it does
not mix distance chains between datasets.

| dataset | definition and purpose |
|---|---|
| `original_published` | Complete final distance-fitting tables from the source papers. For NGC5765b this includes all 212 printed rows, including 20 systemic spots without measured accelerations; their positions and velocities are retained, while their placeholder accelerations are masked from the acceleration likelihood. |
| `fiducial` | The exact spot tables actually fitted by Pesce et al. (2020), supplied by the MCP with its 2026 response and planned erratum. This is the operational default. The MCP's pre-fit vetting removed spots from NGC5765b, NGC6264 and UGC3789, added 19 earlier-epoch spots to NGC6323, and replaced the NGC6264 acceleration uncertainties. |
| `unpruned` | The clipping-test reconstruction. It restores all published spots and astrometry for NGC5765b, NGC6264 and UGC3789 while retaining the fiducial acceleration fields for matched rows. NGC6323 is identical to `fiducial`, because the MCP reported no clipped NGC6323 spots. Row-level origins are recorded in `data/Megamaser/unpruned/provenance.csv`. |
| `clipped` | A virtual dataset: load `unpruned`, then apply the matching stabilised iterative-DE clipping mask. The source chains in this sweep use the base circular-disc, linear-warp masks. |

The loaded spot counts for the five galaxies entering this joint H0 analysis
are:

| galaxy | `original_published` | `fiducial` | `unpruned` | `clipped` | removed by clipping |
|---|---:|---:|---:|---:|---:|
| CGCG074-064 | 165 | 165 | 165 | 152 | 13 |
| NGC5765b | 212 | 169 | 212 | 206 | 6 |
| NGC6264 | 66 | 61 | 66 | 65 | 1 |
| NGC6323 | 68 | 87 | 87 | 83 | 4 |
| UGC3789 | 156 | 153 | 156 | 148 | 8 |
| **total** | **667** | **635** | **686** | **654** | **32** |

Thus `unpruned` is not simply another name for `original_published`: most
notably, it keeps the 19 additional NGC6323 spots from the fiducial table.
CGCG074-064 is identical across the three source datasets, so its only change
is the subsequent iterative clipping. NGC4258 has 358 spots and is unchanged
across the source datasets, but it is an anchor-system fit and is excluded
from this joint H0 sweep.

## Distance-posterior sources and priors

The `candel` source uses the matching saved CANDEL stage-1 posterior for each
spot-table dataset. Those stage-1 chains were derived with a prior uniform in
`D_A`, so their KDE densities are already proportional to the distance
likelihood within the configured support.

The `p20` source instead uses the five archived P20 `D_A` sequences in
`data/Megamaser/external/Dom_data/`. These sequences were derived with a prior
uniform in `log(D_A)`, for which the density is proportional to `1 / D_A`.
The stage-2 runner therefore multiplies each P20 KDE by `D_A` to remove the
stage-1 prior before applying the requested stage-2 prior. Consequently, the
P20 and CANDEL rows below use the same stated stage-2 prior and can be compared
like for like. The `fiducial` path of the P20 outputs is only their stage-2
results namespace; no fiducial CANDEL distance posterior enters those runs.

The stage-2 priors labelled `volume` and `distance` are proportional to
`D_c^2` and 1 in the sampled comoving-distance coordinate, while
`log-distance` samples uniformly in `log(D_A)`. Selected-population runs
require the volume prior. The no-selection baselines additionally test the
distance and log-distance priors, both without reconstruction and with
Carrick2015.

## Headline result

For the CANDEL-distance source, the most fully modelled combination, redshift
selection with the ManticoreLocalCOLA reconstruction, gives

| dataset | H0 [km/s/Mpc] |
|---|---:|
| `fiducial` | 71.63 -3.41/+3.79 |
| `unpruned` | 70.02 -3.49/+3.79 |
| `clipped` | 70.05 -3.39/+3.67 |
| `original_published` | 67.70 -3.49/+3.74 |

These are posterior medians with central 68 per cent credible intervals. The
fiducial and original-published variants are sampler-suspect because more than
1 per cent of their retained transitions diverged; see the convergence
section. The corresponding P20-distance result is
`71.51 -3.41/+3.64 km/s/Mpc`, only `0.12 km/s/Mpc` below the CANDEL-distance
fiducial result, and passes the working convergence gates below.

## All H0 results

Values are posterior medians with central 68 per cent credible intervals, in
km/s/Mpc. A `[suspect]` marker denotes more than 1 per cent divergent retained
transitions.

### CANDEL distances

| selection | reconstruction | distance prior | `fiducial` | `unpruned` | `clipped` | `original_published` |
|---|---|---|---:|---:|---:|---:|
| none | none | volume | 71.19 -3.15/+3.27 | 69.39 -3.25/+3.40 | 69.27 -3.24/+3.25 | 66.93 -3.25/+3.36 |
| none | none | distance | 72.46 -3.18/+3.34 | 70.72 -3.18/+3.37 | 70.54 -3.19/+3.28 | 68.27 -3.36/+3.37 |
| none | none | log-distance | 73.07 -3.29/+3.37 | 71.28 -3.21/+3.39 | 71.10 -3.11/+3.23 | 68.90 -3.25/+3.47 |
| none | Carrick2015 | distance | 71.12 -3.33/+3.44 | 69.43 -3.41/+3.65 | 69.57 -3.25/+3.39 | 67.07 -3.26/+3.52 |
| none | Carrick2015 | log-distance | 71.71 -3.22/+3.56 | 70.02 -3.35/+3.69 | 70.17 -3.27/+3.51 | 67.74 -3.32/+3.55 |
| distance | none | volume | 72.92 -3.32/+3.42 | 71.11 -3.23/+3.51 | 70.93 -3.25/+3.34 | 68.75 -3.36/+3.45 |
| distance | Carrick2015 | volume | 71.60 -3.32/+3.55 | 69.93 -3.38/+3.54 | 70.03 -3.29/+3.48 | 67.60 -3.32/+3.50 |
| distance | ManticoreLocalCOLA | volume | 70.81 -3.42/+3.68 | 69.24 -3.51/+3.65 | 69.28 -3.38/+3.53 | 66.86 -3.41/+3.54 |
| redshift | none | volume | 73.42 -3.34/+3.58 | 71.65 -3.28/+3.61 | 71.48 -3.23/+3.42 | 69.38 -3.47/+3.46 [suspect] |
| redshift | Carrick2015 | volume | 72.41 -3.40/+3.65 [suspect] | 70.75 -3.47/+3.82 | 70.67 -3.23/+3.58 | 68.24 -3.35/+3.66 |
| redshift | ManticoreLocalCOLA | volume | 71.63 -3.41/+3.79 [suspect] | 70.02 -3.49/+3.79 | 70.05 -3.39/+3.67 | 67.70 -3.49/+3.74 [suspect] |

### P20 distances

These runs are stored under `fiducial`, but use only the archived P20
distance posteriors in their stage-1 distance likelihoods.

| selection | reconstruction | distance prior | H0 |
|---|---|---|---:|
| none | none | volume | 71.43 -2.91/+3.10 |
| none | none | distance | 72.54 -2.96/+3.10 |
| none | none | log-distance | 72.98 -2.95/+3.07 |
| none | Carrick2015 | distance | 71.16 -3.14/+3.25 |
| none | Carrick2015 | log-distance | 71.68 -3.20/+3.30 |
| distance | none | volume | 72.83 -3.02/+3.12 |
| distance | Carrick2015 | volume | 71.50 -3.15/+3.40 |
| distance | ManticoreLocalCOLA | volume | 70.72 -3.26/+3.49 |
| redshift | none | volume | 73.31 -3.07/+3.18 [suspect] |
| redshift | Carrick2015 | volume | 72.19 -3.12/+3.53 |
| redshift | ManticoreLocalCOLA | volume | 71.51 -3.41/+3.64 |

## Main comparisons

- The fiducial result is consistently 3.93--4.26 km/s/Mpc above the
  original-published result across all 11 like-for-like CANDEL-distance
  variants.
- The clipped and unpruned results are effectively identical: every matched
  median differs by less than 0.19 km/s/Mpc. Iterative clipping therefore has
  negligible impact on the combined H0 result in this sweep.
- Across all 11 like-for-like fiducial comparisons, replacing the
  CANDEL distance posteriors with P20 changes the H0 median by between -0.22
  and +0.24 km/s/Mpc. The P20 central 68 per cent intervals are narrower by
  0.16--0.67 km/s/Mpc in total width.
- Relative to no velocity reconstruction, Carrick2015 lowers the median H0 by
  0.81--1.31 km/s/Mpc and ManticoreLocalCOLA lowers it by 1.43--2.11
  km/s/Mpc in the CANDEL-distance sweep.
- Redshift selection raises the median H0 by 0.5--0.84 km/s/Mpc relative to
  distance selection in the CANDEL-distance sweep.
- For the CANDEL no-selection baseline without reconstruction, changing from
  the volume prior to the distance prior raises the median H0 by 1.26--1.35
  km/s/Mpc. Changing from the distance to the log-distance prior raises it by
  a further 0.56--0.63 km/s/Mpc; with Carrick2015, the latter shift is
  0.58--0.67 km/s/Mpc.
- The corresponding fiducial P20 prior shifts are similar: volume to distance
  raises H0 by 1.11 km/s/Mpc without reconstruction, while distance to
  log-distance raises it by 0.45 km/s/Mpc without reconstruction and 0.52
  km/s/Mpc with Carrick2015.
- These modelling shifts remain smaller than the individual posterior
  uncertainty of approximately 3.0--3.8 km/s/Mpc.

## Stage-2 convergence

Because every final stage-2 run contains only one chain, genuine between-chain
R-hat is unavailable. The values below use single-chain split-R-hat only as a
stationarity diagnostic; they do not demonstrate convergence between
independently initialised chains.

Across the 55 available chains:

- H0 effective sample size ranges from 3,743 to 17,677;
- the maximum H0 split-R-hat is 1.00133;
- across all sampled parameters, the minimum ESS is 438 and the maximum
  split-R-hat is 1.00337; and
- no retained draw reached the configured maximum tree depth.

No final chain therefore fails the working ESS < 400 or split-R-hat > 1.01
gate. Five runs are nevertheless sampler-suspect because more than 1 per cent
of their retained transitions diverged:

| distance source | dataset | selection | reconstruction | divergences | H0 ESS | H0 split-R-hat | worst all-parameter ESS / split-R-hat |
|---|---|---|---|---:|---:|---:|---:|
| CANDEL | `fiducial` | redshift | Carrick2015 | 329/15,000 = 2.19% | 4,786 | 1.00029 | 438 / 1.00337 |
| CANDEL | `fiducial` | redshift | ManticoreLocalCOLA | 459/15,000 = 3.06% | 5,599 | 1.00010 | 485 / 1.00266 |
| P20 | `fiducial` | redshift | none | 491/15,000 = 3.27% | 3,743 | 1.00005 | 644 / 1.00039 |
| CANDEL | `original_published` | redshift | none | 151/15,000 = 1.01% | 8,228 | 1.00010 | 2,125 / 1.00078 |
| CANDEL | `original_published` | redshift | ManticoreLocalCOLA | 183/15,000 = 1.22% | 8,660 | 0.99995 | 2,090 / 1.00039 |

The newly recovered P20-distance redshift-selection + ManticoreLocalCOLA chain
has 104/15,000 = 0.69 per cent divergences, H0 ESS = 6,274 and H0 split-R-hat
= 1.00038. Its worst all-parameter ESS is 2,916 and its maximum split-R-hat is
1.00073, so it does not add to the sampler-suspect table.

The H0 sequences themselves have high ESS and split-R-hat near one in these
runs, but divergent transitions can still bias their posteriors. Before using
them as final results, rerun them with multiple independent chains and a higher
target acceptance.

## Stage-1 distance-posterior caveats

Stage 2 consumes only the `D_A` samples from each single-galaxy posterior. For
the CANDEL source, the `original_published` NGC6323 input is the only such
sequence that fails the ESS gate: it has two chains of 2,000 draws, D_A ESS =
230, split-R-hat =
1.00043 and 56/4,000 = 1.40 per cent divergent transitions. Every
original-published H0 result should therefore remain provisional until that
single-galaxy distance chain is improved.

Three further input chains pass the D_A ESS/R-hat gate but have more than 1
per cent divergences: fiducial NGC6323 (1.03 per cent), unpruned NGC5765b
(1.23 per cent) and unpruned NGC6323 (1.03 per cent).

The archived P20 inputs do not retain independent-chain or divergence
metadata. Treating each file as one sequence gives:

| galaxy | draws | D_A ESS | single-sequence split-R-hat |
|---|---:|---:|---:|
| CGCG074-064 | 15,012 | 1,275 | 1.00287 |
| NGC5765b | 94,617 | 14,378 | 0.99999 |
| NGC6264 | 91,874 | 12,767 | 1.00007 |
| NGC6323 | 96,460 | 1,801 | 1.00372 |
| UGC3789 | 95,677 | 3,688 | 1.00005 |

These sequence diagnostics pass the working ESS and stationarity gates, but
cannot establish between-chain convergence. All CANDEL source `D_A` draws lie
within their stage-2 support. For P20, 3/15,012 CGCG074-064 draws and
11/96,460 NGC6323 draws lie outside it; the other three files have none. The
14 out-of-support tail draws constitute less than 0.004 per cent of the
combined P20 input.

## Failed and superseded jobs

Six earlier `original_published` logs, jobs 818628, 818629, 818631, 818632,
818634 and 818635, failed before sampling because the CGCG074-064 input chain
was missing. The subsequent complete rerun, jobs 818637--818644, succeeded and
produced the eight original-published HDF5 files from that sweep. Jobs
827379--827390 then added the three new no-selection variants for each of the
four CANDEL datasets.

The failed attempts are retained in the scheduler logs for provenance but do
not contribute to the reported results. P20 job 827396 supplies the previously
missing redshift-selection + ManticoreLocalCOLA chain and completes the P20
sweep.

## Implementation note

These jobs post-date the Carrick coordinate-frame correction. The live joint
model rotates the ICRS Cartesian `Vext` vector into the reconstruction's
coordinate frame before projecting it onto the line of sight and onto the 3-D
selection volume. The earlier Carrick frame-mismatch warning therefore does
not apply to this sweep.

## Source files and diagnostic convention

- Runner: `scripts/megamaser/run_joint_H0.py`.
- Sweep wrapper: `scripts/megamaser/submit_sweep_H0.sh`.
- CANDEL stage-1 inputs:
  `results/Megamaser/<dataset>/<galaxy>/*_blackjax_mcmc_rphi_initconfig.hdf5`.
- P20 stage-1 inputs: `data/Megamaser/external/Dom_data/D_archivedP20_*.txt`.
- Final outputs: `results/Megamaser/<dataset>/H0/*.hdf5`; P20 variants carry
  the `_p20` suffix.
- Scheduler logs: `results/Megamaser/<dataset>/H0/logs/*.out`.
- H0 summaries use the 16th, 50th and 84th sample percentiles.
- ESS and split-R-hat were recomputed directly from the saved HDF5 samples
  with NumPyro's diagnostics; divergences and integration-step counts were
  read from the saved `info/` arrays.
