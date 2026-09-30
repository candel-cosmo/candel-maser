#!/usr/bin/env python
"""Reproduce the Watkins & Feldman (2026) megamaser H0 and its sample effects.

Watkins & Feldman (arXiv:2608.06247, hereafter WF26) reanalyse the six
Megamaser Cosmology Project distances of Pesce et al. (2020, hereafter P20)
with peculiar velocities from the Manticore reconstruction (M25) instead of
Carrick et al. (2015, C15), report H0 = 68.8 +/- 2.6 km/s/Mpc, and conclude
that the Hubble tension is a distance-ladder systematic.  This script rebuilds
their model and asks how much of that conclusion is carried by two galaxies:
NGC 4258, whose 1.5 per cent distance makes it the highest-weight object once
the velocity error is small, and NGC 5765b, where C15 and M25 disagree most.

Their model is three equations over six distance-velocity pairs.  Writing
d_i for the P20 angular-diameter distance and z_i for the peculiar-velocity
corrected redshift, their Eq. (1) is

    L(H0) = prod_i N(d_i | d_m(z_i; H0), sigma_i),

with d_m the flat-LCDM angular-diameter distance at Om = 0.3 (their Eq. 2) and

    sigma_i^2 = sigma_di^2 + (sigma_v / c * dd_m/dz|_{z_i})^2               (3)

their Eq. (3).  H0 is the only free parameter; sigma_v is fixed by hand.  Note
what Eq. (3) does: dd_m/dz ~ c / H0 near z = 0, so the velocity term adds the
same distance error in Mpc to every galaxy regardless of its distance.  It is
therefore a weighting knob, not a noise knob, and lowering sigma_v transfers
the fit onto the nearest host.  Section 3 below quantifies that transfer.

Versions matter here.  v1-v3 used a single M25 realisation and reported
69.7 +/- 2.6 at sigma_v = 150 km/s, a marginalised 68.7 +/- 2 over a uniform
sigma_v prior, and a sigma_v grid reaching 0.  v4 silently replaced the
velocities with 80-realisation ensemble means, moved the headline to 68.8,
deleted the marginalisation, the corner plot, the chi^2 column and the
sigma_v <= 40 rows, and left the abstract byte-identical.  Both velocity sets
are tabulated below so the revision can be decomposed (section 6).

WF26 average the likelihood over the 80 realisations rather than averaging the
velocities first.  Only the per-galaxy means and standard deviations are
published, so section 9 emulates the averaging with independent Gaussian draws.
That discards the inter-galaxy correlation within a realisation, which is the
part of their procedure that is genuinely better than a single mean field, so
treat section 9 as a bound on the width, not as a reproduction.

Run from the repository root with::

    venv_candel/bin/python packages/candel-maser/scripts/reproduce_watkins_h0.py
"""
import numpy as np
from scipy.stats import norm

from candel.cosmo.cosmography import Redshift2Distance
from candel.util import SPEED_OF_LIGHT

# WF26 table 1 (ta:par), "Parameters for the megamaser galaxies from P20".
# Distances with asymmetric uncertainties are symmetrised by WF26 by averaging
# the two sides; the values below are theirs, not the P20 posteriors.
GALAXIES = ("NGC4258", "UGC3789", "CGCG074-064", "NGC6323", "NGC5765b",
            "NGC6264")
DISTANCE = np.array([7.58, 51.5, 87.6, 109.4, 112.2, 132.1])
DISTANCE_ERROR = np.array([0.11, 4.2, 7.5, 30.0, 5.0, 20.0])
CZ_OBS = np.array([679.3, 3319.9, 7172.2, 7801.5, 8525.7, 10192.6])

# Peculiar velocities, WF26 table 1.  C15 is unchanged across all versions;
# M25 was replaced wholesale between v3 and v4.
V_C15 = np.array([262.0, -54.0, 297.0, 414.0, 124.0, 224.0])
V_M25 = np.array([285.0, 45.0, 334.0, 694.0, 607.0, 439.0])
V_M25_SD = np.array([94.0, 48.0, 61.0, 88.0, 53.0, 75.0])
V_M25_V3 = np.array([162.0, 54.0, 149.0, 671.0, 546.0, 434.0])

# Our own query of the same Manticore generation, 80-realisation means from
# `compare_watkins_velocities.py`.  `POINT` evaluates at the P20 distance, as
# WF26 do; `MARG` marginalises over a Gaussian of the P20 distance error.  The
# SWIFT momentum products are the ones WF26 used; PCS is the BORG forward grid.
V_SWIFT_POINT = np.array([311.0, 63.0, 370.0, 708.0, 573.0, 427.0])
V_PCS_POINT = np.array([255.0, 54.0, 336.0, 579.0, 537.0, 392.0])
V_SWIFT_MARG = np.array([311.0, 51.0, 356.0, 253.0, 475.0, 305.0])
V_PCS_MARG = np.array([255.0, 57.0, 337.0, 239.0, 470.0, 299.0])
# Projection of the sampled bulk flow onto each line of sight, from table 4 of
# the paper (Manticore column).  WF26 have no external-flow term.  NGC 4258 is
# not in our five-galaxy sample, so this is applied only once it is dropped.
VEXT_LOS = np.array([np.nan, -97.0, 15.0, -57.0, 22.0, -28.0])

# WF26 table 2 (ta:vel), M25 block, and their C15 quote in section "Results".
WF_M25 = {50: (66.9, 2.5), 100: (68.4, 2.5), 150: (68.8, 2.6)}
WF_C15 = {150: (71.5, 2.6)}
WF_M25_V3 = {150: (69.7, 2.6)}
H0_DISTANCE_LADDER = 73.5

OM = 0.3
H0_GRID = np.linspace(40.0, 110.0, 14001)
_Z2D = Redshift2Distance(Om0=OM, zmax_interp=0.1)


def _shape(z, eps=1e-6):
    """d_A and dd_A/dz at H0 = 100, in Mpc.

    Redshift2Distance returns the comoving distance as f(z) / h exactly, so
    both quantities scale as 1 / H0 and the H0 grid needs no cosmology calls.
    """
    z = np.atleast_1d(z)
    D_A = _Z2D(z, h=1.0) / (1.0 + z)
    grad = (_Z2D(z + eps, h=1.0) / (1.0 + z + eps)
            - _Z2D(z - eps, h=1.0) / (1.0 + z - eps)) / (2.0 * eps)
    return D_A, grad


def _angular_diameter_distance(z, H0):
    """Flat-LCDM angular-diameter distance in Mpc, WF26 Eq. (2)."""
    return _shape(z)[0] * 100.0 / H0


def _sigma(z, H0, sigma_d, sigma_v):
    """Total distance uncertainty, WF26 Eq. (3)."""
    grad = _shape(z)[1] * 100.0 / H0
    return np.sqrt(sigma_d**2 + (sigma_v / SPEED_OF_LIGHT * grad)**2)


def _posterior(v_pec, sigma_v, keep=None, grid=H0_GRID):
    """Normalised L(H0) of WF26 Eq. (1) over the retained galaxies.

    The sigma_i in Eq. (3) depends on H0, so the Gaussian prefactor of Eq. (1)
    is not constant.  WF26 nonetheless describe their procedure as minimising
    over the exponent alone, and only the chi^2 form recovers their published
    values: the log-variance term shifts the C15 headline from 71.53 to 71.66
    against their 71.5, and the M25 headline from 68.73 to 68.86 against 68.8.
    The term is therefore dropped here, as they drop it.
    """
    keep = np.arange(len(GALAXIES)) if keep is None else np.asarray(keep)
    z = (CZ_OBS[keep] - np.asarray(v_pec)[keep]) / SPEED_OF_LIGHT
    D_A, grad = _shape(z)
    scale = (100.0 / grid)[:, None]
    model = D_A[None, :] * scale
    var = (DISTANCE_ERROR[keep][None, :]**2
           + (sigma_v / SPEED_OF_LIGHT * grad[None, :] * scale)**2)
    log_L = -0.5 * np.sum((DISTANCE[keep][None, :] - model)**2 / var, axis=1)
    p = np.exp(log_L - log_L.max())
    return grid, p / np.trapezoid(p, grid)


def _log_like_2d(v_pec, sigma_v_grid, keep=None, grid=H0_GRID):
    """log L(H0, sigma_v) with the Gaussian prefactor retained.

    The chi^2 form of `_posterior` is unnormalised in sigma_v and is maximised
    by sigma_v -> infinity, so any inference on sigma_v needs this version.
    """
    keep = np.arange(len(GALAXIES)) if keep is None else np.asarray(keep)
    z = (CZ_OBS[keep] - np.asarray(v_pec)[keep]) / SPEED_OF_LIGHT
    D_A, grad = _shape(z)
    scale = (100.0 / grid)[:, None, None]
    model = D_A[None, None, :] * scale
    vel = (sigma_v_grid[None, :, None] / SPEED_OF_LIGHT
           * grad[None, None, :] * scale)
    var = DISTANCE_ERROR[keep][None, None, :]**2 + vel**2
    return -0.5 * np.sum((DISTANCE[keep][None, None, :] - model)**2 / var
                         + np.log(var), axis=2)


def _marginalise_sigma_v(v_pec, log_prior, sigma_v_grid, keep=None):
    """H0 posterior marginalised over sigma_v, and the sigma_v posterior."""
    log_L = _log_like_2d(v_pec, sigma_v_grid, keep)
    log_post = log_L + log_prior[None, :]
    post = np.exp(log_post - log_post.max())
    p_H0 = np.trapezoid(post, sigma_v_grid, axis=1)
    p_H0 /= np.trapezoid(p_H0, H0_GRID)
    p_sv = np.trapezoid(post, H0_GRID, axis=0)
    p_sv /= np.trapezoid(p_sv, sigma_v_grid)
    mean = np.trapezoid(p_H0 * H0_GRID, H0_GRID)
    sd = np.sqrt(np.trapezoid(p_H0 * (H0_GRID - mean)**2, H0_GRID))
    sv_med = np.interp(
        0.5, np.cumsum(p_sv) / np.sum(p_sv), sigma_v_grid)
    return H0_GRID[p_H0.argmax()], sd, sv_med


def _fit(v_pec, sigma_v, keep=None):
    """Maximum-likelihood H0 and the width of L(H0).

    WF26 quote the maximum, not the mean; the two differ by 0.2 km/s/Mpc
    because L(H0) is skewed.
    """
    grid, p = _posterior(v_pec, sigma_v, keep)
    mean = np.trapezoid(p * grid, grid)
    sd = np.sqrt(np.trapezoid(p * (grid - mean)**2, grid))
    return grid[p.argmax()], sd


def _row(label, mean, sd, reference=None):
    line = f"  {label:52s} {mean:6.2f} +/- {sd:4.2f}"
    if reference is not None:
        line += (f"   WF26 {reference[0]:5.2f} +/- {reference[1]:4.2f}"
                 f"   delta {mean - reference[0]:+5.2f}")
    print(line)


def _fisher_weights(v_pec, sigma_v, H0=69.0):
    """Per-galaxy share of the Fisher information on H0.

    d_m ~ c z / H0 at low z, so dd_m/dH0 = -d_m / H0 and the information is
    sum_i d_i^2 / sigma_i^2 / H0^2.  The weight is therefore d_i^2 / sigma_i^2,
    not the 1 / sigma_i^2 of the distance-space chi^2.
    """
    z = (CZ_OBS - v_pec) / SPEED_OF_LIGHT
    w = DISTANCE**2 / _sigma(z, H0, DISTANCE_ERROR, sigma_v)**2
    return w / w.sum()


def _drop(*names):
    return [i for i, g in enumerate(GALAXIES) if g not in names]


def main():
    print("\n1. Reproduction of the published WF26 values")
    for sigma_v, reference in sorted(WF_C15.items()):
        _row(f"C15, six galaxies, sigma_v = {sigma_v}",
             *_fit(V_C15, sigma_v), reference)
    for sigma_v, reference in sorted(WF_M25.items()):
        _row(f"M25 v4 ensemble means, sigma_v = {sigma_v}",
             *_fit(V_M25, sigma_v), reference)
    for sigma_v, reference in sorted(WF_M25_V3.items()):
        _row(f"M25 v1-v3 single realisation, sigma_v = {sigma_v}",
             *_fit(V_M25_V3, sigma_v), reference)
    print("  the P20 value of sigma_v, which WF26 do not tabulate:")
    _row("C15, six galaxies, sigma_v = 250", *_fit(V_C15, 250.0))
    _row("M25 v4 ensemble means, sigma_v = 250", *_fit(V_M25, 250.0))
    _row("no velocity field at all, sigma_v = 250",
         *_fit(np.zeros(len(GALAXIES)), 250.0))

    print("\n2. Single-galaxy H0 = (cz - v_pec) / d")
    print(f"  {'galaxy':14s} {'d [Mpc]':>9s} {'C15':>8s} {'M25 v4':>8s} "
          f"{'M25 v1-v3':>10s}")
    for i, galaxy in enumerate(GALAXIES):
        print(f"  {galaxy:14s} {DISTANCE[i]:9.2f} "
              f"{(CZ_OBS[i] - V_C15[i]) / DISTANCE[i]:8.1f} "
              f"{(CZ_OBS[i] - V_M25[i]) / DISTANCE[i]:8.1f} "
              f"{(CZ_OBS[i] - V_M25_V3[i]) / DISTANCE[i]:10.1f}")

    print("\n3. Fisher weight for H0 [per cent], M25 v4 velocities")
    print("  sigma_v " + "".join(f"{g:>14s}" for g in GALAXIES))
    for sigma_v in (0, 20, 50, 100, 150, 250):
        w = 100.0 * _fisher_weights(V_M25, sigma_v)
        print(f"  {sigma_v:7d} " + "".join(f"{x:14.1f}" for x in w))
    print("  (the v4 table spans sigma_v = 50 to 150 only; v1-v3 reached 0)")

    print("\n4. Leave-one-out, M25 v4 velocities")
    subsamples = [("six galaxies", None)]
    subsamples += [(f"drop {g}", _drop(g)) for g in GALAXIES]
    subsamples += [("drop NGC4258 + NGC5765b",
                    _drop("NGC4258", "NGC5765b"))]
    grid_v = (0, 50, 100, 150, 250)
    header = "".join(f"{'sv=' + str(s):>16s}" for s in grid_v)
    print(f"  {'subsample':22s}" + header)
    for label, keep in subsamples:
        cells = []
        for sigma_v in grid_v:
            mean, sd = _fit(V_M25, sigma_v, keep)
            cells.append(f"{mean:6.2f}+/-{sd:4.2f}")
        print(f"  {label:22s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n   the same, C15 velocities")
    for label, keep in subsamples:
        cells = []
        for sigma_v in grid_v:
            mean, sd = _fit(V_C15, sigma_v, keep)
            cells.append(f"{mean:6.2f}+/-{sd:4.2f}")
        print(f"  {label:22s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n5. C15 -> M25 shift, one galaxy swapped at a time, sigma_v = 150")
    base = _fit(V_C15, 150.0)[0]
    full = _fit(V_M25, 150.0)[0]
    print(f"  all C15 {base:.2f}, all M25 {full:.2f}, "
          f"total {full - base:+.2f}")
    for i, galaxy in enumerate(GALAXIES):
        v = V_C15.copy()
        v[i] = V_M25[i]
        mean = _fit(v, 150.0)[0]
        print(f"  swap {galaxy:14s} {mean:6.2f}   delta {mean - base:+5.2f}"
              f"   (dv = {V_M25[i] - V_C15[i]:+5.0f} km/s)")

    print("\n6. v1-v3 -> v4 velocity revision, one galaxy at a time, "
          "sigma_v = 150")
    base3 = _fit(V_M25_V3, 150.0)[0]
    base4 = _fit(V_M25, 150.0)[0]
    print(f"  v1-v3 {base3:.2f}, v4 {base4:.2f}, total {base4 - base3:+.2f}")
    for i, galaxy in enumerate(GALAXIES):
        v = V_M25_V3.copy()
        v[i] = V_M25[i]
        mean = _fit(v, 150.0)[0]
        dv = V_M25[i] - V_M25_V3[i]
        print(f"  swap {galaxy:14s} {mean:6.2f}   delta {mean - base3:+5.2f}"
              f"   (dv = {dv:+5.0f} km/s = {dv / V_M25_SD[i]:+4.1f} sd)")

    print("\n7. NGC4258 consistency test, distance known to 1.5 per cent")
    for H0 in (H0_DISTANCE_LADDER, 70.0, 67.4):
        print(f"  v_pec required for H0 = {H0:5.1f}: "
              f"{CZ_OBS[0] - H0 * DISTANCE[0]:6.1f} km/s")
    overshoot = V_M25[0] - (CZ_OBS[0] - H0_DISTANCE_LADDER * DISTANCE[0])
    print(f"  M25 assigns {V_M25[0]:.0f} +/- {V_M25_SD[0]:.0f} km/s, an "
          f"overshoot of {overshoot:.0f} km/s = {overshoot / V_M25_SD[0]:.1f} "
          f"realisation sd")

    print("\n8. Tension arithmetic, M25 v4")
    for sigma_v in (50, 100, 150, 250):
        grid, p = _posterior(V_M25, sigma_v)
        tail = float(np.trapezoid(p * (grid >= H0_DISTANCE_LADDER), grid))
        print(f"  sigma_v = {sigma_v:3d}   P(H0 >= {H0_DISTANCE_LADDER}) = "
              f"{tail:.3f}, one-sided {norm.isf(tail):.2f} sigma")
    print("  WF26 quote 0.012, 0.030, 0.047 for sigma_v = 50, 100, 150 and "
          "call the\n  range '> 2 sigma'; sigma_v = 250, the P20 value, is "
          "not tabulated")

    print("\n9. Realisation averaging emulated with independent draws")
    print("   (WF26 average 80 correlated realisations; only means and "
          "standard\n    deviations are published, so this bounds the width "
          "rather than\n    reproducing it)")
    rng = np.random.default_rng(42)
    for sigma_v in (50, 100, 150):
        stack = np.zeros(H0_GRID.size)
        for _ in range(80):
            _, p = _posterior(rng.normal(V_M25, V_M25_SD), sigma_v)
            stack += p
        stack /= np.trapezoid(stack, H0_GRID)
        mean = np.trapezoid(stack * H0_GRID, H0_GRID)
        sd = np.sqrt(np.trapezoid(stack * (H0_GRID - mean)**2, H0_GRID))
        _row(f"averaged likelihood, sigma_v = {sigma_v}", mean, sd,
             WF_M25[sigma_v])

    print("\n10. Their model with our own query of the same reconstruction")
    print("    (velocities from compare_watkins_velocities.py, 80-realisation"
          "\n     means; SWIFT is the product WF26 used, PCS the forward "
          "grid)")
    variants = (("WF26 tabulated M25", V_M25),
                ("SWIFT, at the P20 distance", V_SWIFT_POINT),
                ("PCS, at the P20 distance", V_PCS_POINT),
                ("SWIFT, marginalised over the distance", V_SWIFT_MARG),
                ("PCS, marginalised over the distance", V_PCS_MARG),
                ("C15, for reference", V_C15))
    grid_v = (50, 100, 150, 250)
    header = "".join(f"{'sv=' + str(s):>16s}" for s in grid_v)
    print(f"  {'velocities':38s}" + header)
    for label, v_pec in variants:
        cells = []
        for sigma_v in grid_v:
            mean, sd = _fit(v_pec, sigma_v)
            cells.append(f"{mean:6.2f}+/-{sd:4.2f}")
        print(f"  {label:38s}" + "".join(f"{c:>16s}" for c in cells))
    print("    the marginalised rows substitute the marginal mean into a "
          "model that\n    has no distance-dependent velocity, so they "
          "understate the width: the\n    marginal scatter is 120 to "
          "280 km/s, not the 50 to 150 assumed here")

    print("\n11. The gap is the size of the velocity correction, not the "
          "baseline")
    print("    (their model with no velocity field already sits where P20 and"
          "\n     our selection run sit, so the selection function is not "
          "what\n     separates us)")
    keep5 = _drop("NGC4258")
    zero = np.zeros(len(GALAXIES))
    for label, v_pec, sigma_v in (
            ("their model, no velocity field, sigma_v = 250", zero, 250.0),
            ("their model, SWIFT point velocities, sigma_v = 250",
             V_SWIFT_POINT, 250.0),
            ("their model, SWIFT point velocities, sigma_v = 150",
             V_SWIFT_POINT, 150.0)):
        _row(label, *_fit(v_pec, sigma_v))
    print(f"  {'our model, redshift selection, no reconstruction':52s} "
          f"{73.3:6.2f}")
    print(f"  {'our model, redshift selection, Manticore':52s} {71.5:6.2f}")
    print(f"  {'P20 as published':52s} {73.9:6.2f}")
    print("    their Manticore correction is -4.65 at sigma_v = 150 and "
          "-4.30 at 250;\n    ours is -1.80")

    print("\n11b. WF26 -> our configuration, one change at a time")
    v_ext = V_PCS_MARG + np.nan_to_num(VEXT_LOS)
    ladder = (
        ("WF26 as published: SWIFT, point, six galaxies, sv=150",
         V_SWIFT_POINT, 150.0, None),
        ("  + PCS forward grid instead of SWIFT", V_PCS_POINT, 150.0, None),
        ("  + velocity marginalised over the P20 distance error",
         V_PCS_MARG, 150.0, None),
        ("  + five galaxies, NGC 4258 dropped", V_PCS_MARG, 150.0, keep5),
        ("  + sigma_v = 250, the P20 value", V_PCS_MARG, 250.0, keep5),
        ("  + external bulk flow projected on each line of sight",
         v_ext, 250.0, keep5),
    )
    previous = None
    for label, v_pec, sigma_v, keep in ladder:
        mean, sd = _fit(v_pec, sigma_v, keep)
        shift = "" if previous is None else f"   {mean - previous:+5.2f}"
        print(f"  {label:56s} {mean:6.2f} +/- {sd:4.2f}{shift}")
        previous = mean
    print(f"  {'our model, no selection, no reconstruction, volume prior':56s}"
          f" {71.4:6.2f}")
    print(f"  {'our model, redshift selection, no reconstruction':56s}"
          f" {73.3:6.2f}   +1.90")
    print(f"  {'our model, redshift selection, Manticore [headline]':56s}"
          f" {71.5:6.2f}   -1.80")

    print("\n12. Sampling sigma_v instead of fixing it")
    print("    (the Gaussian prefactor is reinstated here; without it the "
          "likelihood\n     is maximised by sigma_v -> infinity.  Our own "
          "prior is Maxwell with\n     mean 250 km/s; WF26 v1-v3 used uniform "
          "[0, sigma_lim] and deleted it in v4)")
    sv_grid = np.linspace(1.0, 700.0, 700)
    maxwell_a = 250.0 / (2.0 * np.sqrt(2.0 / np.pi))
    priors = (
        ("Maxwell, mean 250 [ours]",
         2.0 * np.log(sv_grid) - sv_grid**2 / (2.0 * maxwell_a**2)),
        ("uniform [0, 150] [WF26 v1-v3]",
         np.where(sv_grid <= 150.0, 0.0, -np.inf)),
        ("uniform [0, 250]", np.where(sv_grid <= 250.0, 0.0, -np.inf)),
        ("uniform [0, 700]", np.zeros_like(sv_grid)),
    )
    for v_label, v_pec, keep in (("SWIFT, point, six galaxies",
                                  V_SWIFT_POINT, None),
                                 ("PCS, marginalised, five galaxies",
                                  V_PCS_MARG, keep5),
                                 ("no velocity field, six galaxies",
                                  np.zeros(len(GALAXIES)), None)):
        print(f"  {v_label}")
        for p_label, log_prior in priors:
            mean, sd, sv_med = _marginalise_sigma_v(
                v_pec, log_prior, sv_grid, keep)
            print(f"    {p_label:34s} H0 = {mean:6.2f} +/- {sd:4.2f}"
                  f"   sigma_v = {sv_med:5.0f} km/s")

    mean, _ = _fit(V_C15, 150.0)
    assert abs(mean - WF_C15[150][0]) < 0.15, f"C15 reproduction: {mean}"
    mean, _ = _fit(V_M25, 150.0)
    assert abs(mean - WF_M25[150][0]) < 0.15, f"M25 reproduction: {mean}"
    print("\n  [self-check: the published C15 and M25 headlines reproduce to "
          "better than 0.15 km/s/Mpc]")


if __name__ == "__main__":
    main()
