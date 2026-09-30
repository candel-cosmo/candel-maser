#!/usr/bin/env python
"""Reproduce Boruah, Hudson & Lavaux (2021) megamaser H0 from P20 distances.

Boruah, Hudson & Lavaux (2021, MNRAS 507, 2697, hereafter B21) reanalyse the
six Megamaser Cosmology Project distances of Pesce et al. (2020, hereafter P20)
and report H0 = 70.1 +/- 2.9 km/s/Mpc, some 3.8 km/s/Mpc below the P20 headline
73.9 +/- 3.0.  This script rebuilds their model inside CANDEL and walks the
P20 -> B21 ladder one change at a time, so the shift can be attributed to
mechanisms rather than to a single unexplained offset.

All reference values are the refereed table 3, not the arXiv preprint.  The two
differ: arXiv:2010.01119v1 makes the volumetric distance prior fiducial and
quotes 69.0 +2.9/-2.8, whereas the refereed version makes the uniform prior
fiducial, on the grounds that the Gould (1993) bias vanishes when the selection
variable is uncorrelated with the distance measurement, and demotes 69.0 to a
variant.  Every other row is the same model shifted by that prior change.

B21 differ from P20 in four stated ways (their section 6.3): line-of-sight
marginalisation of the peculiar velocity instead of a point estimate, the
distance prior, group-corrected redshifts, and sigma_v = 150 km/s instead of
250 km/s.  A fifth, unstated difference dominates: the P20 headline applies no
velocity-field correction at all, whereas every B21 row does.

Group-corrected redshifts are the one B21 ingredient not reproduced here: their
2M++ group assignments are not tabulated, so every row below uses the P20
CMB-frame systemic velocities.  The like-for-like target is therefore their
`No group redshift correction` row, and B21 report that grouping raises H0 by
0.4 km/s/Mpc.

Run from the repository root with::

    venv_candel/bin/python packages/candel-maser/scripts/reproduce_boruah_h0.py
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")


import numpy as np  # noqa: E402
import candel_maser.run_joint_H0 as joint  # noqa: E402
from candel_maser.joint_H0_helpers import DEFAULT_FIELD_CONFIG  # noqa: E402
from candel_maser.joint_H0_helpers import (  # noqa: E402
    _load_or_build_vlos_cache)

from candel.cosmo.cosmography import Distance2Redshift  # noqa: E402
from candel.util import SPEED_OF_LIGHT  # noqa: E402

# The Carrick+2015 field distributed by cosmicflows.iap.fr, which B21 use, is
# the 2M++ prediction with beta* = 0.43 and the external dipole already added.
# CANDEL's loader strips both, so they are restored here.
BETA_2MPP = 0.43
VEXT_GAL = np.array([89.0, -131.0, 17.0])

GALAXIES = ("UGC3789", "NGC6264", "NGC6323", "NGC5765b",
            "CGCG074-064", "NGC4258")

# P20 table 1: CMB-frame recession velocity and its statistical error.
P20_VELOCITIES = {
    "UGC3789": (3319.9, 0.8),
    "NGC6264": (10192.6, 0.8),
    "NGC6323": (7801.5, 1.5),
    "NGC5765b": (8525.7, 0.7),
    "CGCG074-064": (7172.2, 1.9),
    "NGC4258": (679.3, 0.4),
}
# P20 table 4, "2M++" column: the recession velocity their treatment (4)
# substitutes for the observed one.  These are point estimates of the Carrick
# field at each galaxy's sky position and redshift, and they differ from the
# field evaluated at the maser distance by up to 250 km/s.
P20_2MPP_RECESSION = {
    "UGC3789": 3375.0,
    "NGC6264": 9962.0,
    "NGC6323": 7378.0,
    "NGC5765b": 8398.0,
    "CGCG074-064": 6869.0,
    "NGC4258": 417.0,
}
# Reid et al. (2019) NGC4258 distance, statistical and systematic combined.
NGC4258_DISTANCE = (7.58, 0.11)

N_D_GRID = 1024
H0_GRID = np.linspace(45.0, 105.0, 2401)
VLOS_R = np.arange(0.1, 250.0 + 0.25, 0.5, dtype=np.float32)


def _galaxy_items():
    """Per-galaxy distance likelihood, sky position, and CMB-frame velocity."""
    root = os.path.join(joint.ROOT, "data", "Megamaser", "external",
                        "Dom_data")
    items = []
    for galaxy in GALAXIES:
        cfg = joint.MASTER_CFG["model"]["galaxies"][galaxy]
        DA_lo, DA_hi = joint._toy_D_A_bounds(cfg["D_lo"], cfg["D_hi"])
        if galaxy == "NGC4258":
            # No archived spot-level chain; use the published Gaussian.
            D_A = np.linspace(DA_lo, DA_hi, N_D_GRID)
            mu, sd = NGC4258_DISTANCE
            log_L = -0.5 * ((D_A - mu) / sd) ** 2
            log_L_raw = log_L                 # no stage-1 prior to remove
            D_med = mu
        else:
            samples = np.loadtxt(
                os.path.join(root, f"D_archivedP20_{galaxy}.txt"))
            D_A, log_L_raw, _, _ = joint._build_log_distance_likelihood(
                samples, DA_lo, DA_hi, N_D_GRID,
                joint.TOY_DISTANCE_MAX_SAMPLES, "uniform_D_A")
            _, log_L, _, _ = joint._build_log_distance_likelihood(
                samples, DA_lo, DA_hi, N_D_GRID,
                joint.TOY_DISTANCE_MAX_SAMPLES, "uniform_log_D_A")
            D_med = float(np.median(samples))
        velocity, stat_error = P20_VELOCITIES[galaxy]
        items.append({
            "name": galaxy,
            "RA": float(cfg["ra"]),
            "dec": float(cfg["dec"]),
            "D_lo": float(cfg["D_lo"]),
            "D_hi": float(cfg["D_hi"]),
            "D_A_grid": np.asarray(D_A, dtype=np.float64),
            # `log_L`: archived density with its uniform-log(D_A) stage-1 prior
            # divided out.  `log_L_raw`: the archived density at face value,
            # which is what B21 insert as their distance likelihood.
            "log_L": np.asarray(log_L, dtype=np.float64),
            "log_L_raw": np.asarray(log_L_raw, dtype=np.float64),
            "D_A_median": D_med,
            "cz_obs": velocity,
            "stat_error": stat_error,
        })
    return items


def _attach_2mpp(items):
    """2M++ line-of-sight velocity, as published by Carrick et al. (2015)."""
    stub = [{"name": it["name"], "RA": it["RA"], "dec": it["dec"]}
            for it in items]
    data = _load_or_build_vlos_cache(
        "Carrick2015", DEFAULT_FIELD_CONFIG, None, VLOS_R, stub)
    if str(data["coordinate_frame"]) != "galactic":
        raise RuntimeError("expected the Carrick2015 field in Galactic "
                           "coordinates")
    vlos = np.asarray(data["los_velocity"])[0]        # (n_gal, n_r)
    rhat = np.asarray(data["rhat"])                   # (n_gal, 3), Galactic
    for i, it in enumerate(items):
        it["los_r"] = np.asarray(data["r"], dtype=np.float64)
        it["vext_los"] = float(VEXT_GAL @ rhat[i])
        it["v_2mpp"] = (BETA_2MPP * vlos[i].astype(np.float64)
                        + it["vext_los"])


def _attach_grids(items):
    """Precompute the (H0, D_c) geometry shared by every model variant."""
    om = float(joint.MASTER_CFG["model"]["Om"])
    d2z = Distance2Redshift(Om0=om)
    h = H0_GRID / 100.0
    for it in items:
        D_c = np.linspace(it["D_lo"], it["D_hi"], N_D_GRID)
        z = np.asarray(d2z(D_c[None, :] * h[:, None]))    # (n_H0, n_D)
        D_A = D_c[None, :] / (1.0 + z)
        it["D_c"] = D_c
        it["z"] = z
        it["D_A"] = D_A
        # P20 place a flat prior on the latent recession velocity, so the
        # measure in D_c carries |dv / dD_c| = c |dz / dD_c|.
        it["log_jacobian"] = np.log(
            SPEED_OF_LIGHT * np.gradient(z, D_c, axis=1))
        it["v_los"] = np.interp(
            D_c[None, :] * h[:, None], it["los_r"], it["v_2mpp"])
        # B21 state h = 0.72 when placing galaxies in the reconstruction box,
        # so the field lookup may not track the sampled H0.
        it["v_los_h72"] = np.interp(
            D_c[None, :] * 0.72, it["los_r"], it["v_2mpp"])
        # `fixed`: the field evaluated once, at the stage-1 median distance.
        z_med = np.asarray(d2z(np.full(H0_GRID.size, it["D_A_median"]) * h))
        it["v_fixed"] = np.interp(
            it["D_A_median"] * (1.0 + z_med) * h, it["los_r"],
            it["v_2mpp"])[:, None]
        # `zspace`: the field queried at the observed redshift, i.e. with the
        # galaxy placed at its redshift-space position.  Needs no H0 because
        # r [Mpc/h] = c z / (100 h) * h = cz / 100.
        it["v_zspace"] = float(np.interp(
            it["cz_obs"] / 100.0, it["los_r"], it["v_2mpp"]))
        # `fixed_z`: the same query iterated to self-consistency, which puts
        # the galaxy back near its real-space position.
        it["v_fixed_z"] = _redshift_space_velocity(it)
        # `p20table`: P20's tabulated 2M++ recession velocity, expressed as the
        # constant peculiar velocity that reproduces it.
        it["v_p20table"] = it["cz_obs"] - P20_2MPP_RECESSION[it["name"]]


def _redshift_space_velocity(item):
    """2M++ velocity where 100 r + v(r) = cz_obs, averaged over the roots.

    The relation is triple-valued behind an overdensity, so all crossings are
    averaged, which is the convention of the 2M++ velocity calculator.
    """
    r, v = item["los_r"], item["v_2mpp"]
    residual = 100.0 * r + v - item["cz_obs"]
    idx = np.flatnonzero(np.sign(residual[:-1]) != np.sign(residual[1:]))
    if idx.size == 0:
        raise RuntimeError(f"no redshift-space solution for {item['name']}")
    t = residual[idx] / (residual[idx] - residual[idx + 1])
    roots = r[idx] + t * (r[idx + 1] - r[idx])
    item["r_redshift_space"] = roots
    return float(np.mean(np.interp(roots, r, v)))


def _predict_cz(zcosmo, vrad):
    beta = vrad / SPEED_OF_LIGHT
    return SPEED_OF_LIGHT * (
        (1.0 + zcosmo) * np.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0)


def _log_likelihood(item, *, sigma_v, add_stat_error, prior, velocity,
                    remove_stage1_prior):
    """log L_i(H0) from a deterministic quadrature over comoving distance.

    The latent coordinate is the comoving distance D_c, matching
    `run_joint_H0.py`.  `prior` selects the measure: ``flat`` is uniform in
    D_c, ``volume`` is proportional to D_c^2, and ``velocity`` is P20's flat
    prior on the latent recession velocity.  `velocity` selects the
    peculiar-velocity treatment: ``none`` predicts pure Hubble flow, ``fixed``
    evaluates the 2M++ field once at the stage-1 median distance, and ``los``
    evaluates it at every point of the integral.
    """
    D_c, z = item["D_c"], item["z"]
    log_L = item["log_L"] if remove_stage1_prior else item["log_L_raw"]
    sigma = (np.hypot(sigma_v, item["stat_error"]) if add_stat_error
             else sigma_v)
    log_q = np.interp(item["D_A"].ravel(), item["D_A_grid"], log_L,
                      left=-np.inf, right=-np.inf).reshape(item["D_A"].shape)
    vrad = {"none": 0.0, "fixed": item["v_fixed"],
            "fixed_z": item["v_fixed_z"], "zspace": item["v_zspace"],
            "p20table": item["v_p20table"],
            "los": item["v_los"], "los_h72": item["v_los_h72"]}[velocity]
    log_w = -0.5 * ((_predict_cz(z, vrad) - item["cz_obs"]) / sigma) ** 2
    log_measure = {"flat": 0.0, "volume": 2.0 * np.log(D_c)[None, :],
                   "velocity": item["log_jacobian"]}[prior]
    total = log_q + log_w + log_measure
    offset = np.max(total, axis=1, keepdims=True)
    integral = np.trapezoid(np.exp(total - offset), D_c, axis=1)
    return np.log(np.maximum(integral, 1e-300)) + offset[:, 0]


def _summary(log_p):
    p = np.exp(log_p - np.max(log_p))
    cdf = np.zeros_like(H0_GRID)
    cdf[1:] = np.cumsum((p[1:] + p[:-1]) * 0.5 * np.diff(H0_GRID))
    cdf /= cdf[-1]
    return np.interp((0.16, 0.5, 0.84), cdf, H0_GRID)


def _fit(items, galaxies=None, **kwargs):
    names = set(GALAXIES if galaxies is None else galaxies)
    total = np.zeros_like(H0_GRID)
    for item in items:
        if item["name"] in names:
            total = total + _log_likelihood(item, **kwargs)
    return _summary(total)


def _row(label, values, reference=None):
    q16, med, q84 = values
    text = f"{label:50s} {med:5.2f} -{med - q16:4.2f}/+{q84 - med:4.2f}"
    if reference is not None:
        text += f"   B21 {reference:5.1f}   {med - reference:+.2f}"
    print(text)


def main():
    items = _galaxy_items()
    _attach_2mpp(items)
    _attach_grids(items)

    base = dict(add_stat_error=True, remove_stage1_prior=False)
    # The refereed fiducial is the uniform distance prior; B21 demote the
    # volumetric prior to a variant because the Gould (1993) bias vanishes when
    # the selection variable is uncorrelated with the distance measurement.
    fiducial = dict(sigma_v=150.0, prior="flat", velocity="los")

    print("\n2M++ peculiar velocity per galaxy, four point estimates (km/s)")
    print(f"{'galaxy':13s} {'cz_obs':>8s} {'real @Dmaser':>13s} "
          f"{'zspace':>8s} {'zspace iter':>12s} {'P20 table 4':>12s}")
    i70 = int(np.argmin(np.abs(H0_GRID - 70.0)))
    for item in items:
        print(f"  {item['name']:11s} {item['cz_obs']:8.1f} "
              f"{float(item['v_fixed'][i70, 0]):13.1f} "
              f"{item['v_zspace']:8.1f} {item['v_fixed_z']:12.1f} "
              f"{item['v_p20table']:12.1f}"
              f"   (Vext.rhat = {item['vext_los']:5.1f})")

    print("\nP20 -> B21 ladder, six galaxies, archived P20 distances")
    steps = [
        ("P20 (1): flat v prior, sigma=250, no field", dict(
            sigma_v=250.0, prior="velocity", velocity="none"), 73.9),
        ("P20 (4): flat v prior, sigma=150, 2M++ table", dict(
            sigma_v=150.0, prior="velocity", velocity="p20table"), 71.8),
        ("  + latent measure flat in D_c [B21 fixed v_pec]", dict(
            sigma_v=150.0, prior="flat", velocity="p20table"), 71.5),
        ("  + field queried in real space, not redshift space", dict(
            sigma_v=150.0, prior="flat", velocity="fixed"), None),
        ("  + LOS marginalisation [B21 no group cz]", dict(
            **fiducial), 69.6),
        ("  + volumetric prior [B21 volumetric variant]", dict(
            sigma_v=150.0, prior="volume", velocity="los"), 69.0),
    ]
    for label, kwargs, reference in steps:
        _row(label, _fit(items, **base, **kwargs), reference)

    print("\nB21 table 3, 2M++ block (refereed version)")
    checks = [
        ("fiducial: LOS marginalisation, uniform prior, sigma=150", dict(
            **fiducial), 70.1),
        ("volumetric distance prior", dict(
            sigma_v=150.0, prior="volume", velocity="los"), 69.0),
        ("no group redshift correction", dict(**fiducial), 69.6),
        ("sigma_v = 200 km/s", dict(
            sigma_v=200.0, prior="flat", velocity="los"), 70.5),
        ("fixed v_pec", dict(
            sigma_v=150.0, prior="flat", velocity="p20table"), 71.5),
        ("P20 2M++ fit", dict(
            sigma_v=150.0, prior="velocity", velocity="p20table"), 71.8),
    ]
    for label, kwargs, reference in checks:
        _row(label, _fit(items, **base, **kwargs), reference)
    print("  (every reproduced row omits the group redshifts, so the "
          "like-for-like\n   comparison is the `no group redshift "
          "correction` row)")

    # The fixed-versus-marginal comparison conflates two changes.  Redshift
    # space against real space is the one that moves H0; where in real space
    # the field is read, and whether it is read once or marginalised, does not.
    print("\nWhat the fixed-versus-marginal comparison actually isolates")
    for label, kwargs in [
            ("P20 tabulated 2M++ velocities", dict(velocity="p20table")),
            ("redshift space, galaxy placed at r = cz/100", dict(
                velocity="zspace")),
            ("redshift space, query iterated to self-consistency", dict(
                velocity="fixed_z")),
            ("real space, at the maser distance", dict(velocity="fixed")),
            ("real space, marginalised along the LOS", dict(velocity="los")),
            ("no velocity field, sigma_v = 250 km/s", dict(
                velocity="none", sigma_v=250.0))]:
        kwargs.setdefault("sigma_v", 150.0)
        _row(label, _fit(items, prior="flat", **base, **kwargs))

    # The leave-one-out table survives only as a commented-out block in the
    # arXiv source; its `all` entry is 69.0, so it is the volumetric variant.
    print("\nLeave-one-out under the volumetric variant")
    volumetric = dict(sigma_v=150.0, prior="volume", velocity="los")
    b21_loo = {"UGC3789": 70.2, "NGC6264": 68.8, "NGC4258": 69.3,
               "NGC5765b": 68.2, "NGC6323": 69.3, "CGCG074-064": 68.1,
               None: 69.0}
    all_med = _fit(items, **base, **volumetric)[1]
    for dropped in (*GALAXIES, None):
        kept = tuple(g for g in GALAXIES if g != dropped)
        q16, med, q84 = _fit(items, galaxies=kept, **base, **volumetric)
        label = "all six" if dropped is None else f"drop {dropped}"
        print(f"  {label:24s} {med:5.2f} -{med - q16:4.2f}/+{q84 - med:4.2f}"
              f"   delta {med - all_med:+5.2f}   B21 delta "
              f"{b21_loo[dropped] - b21_loo[None]:+5.2f}")

    print("\nResidual budget: implementation choices B21 do not pin down")
    _row("fiducial", _fit(items, **base, **fiducial))
    _row("field placed with a fixed h = 0.72", _fit(
        items, sigma_v=150.0, prior="flat", velocity="los_h72", **base))
    _row("no per-galaxy statistical velocity error", _fit(
        items, add_stat_error=False, remove_stage1_prior=False, **fiducial))

    print("\nStage-1 prior treatment (fiducial B21 model, six galaxies)")
    for remove, label in (
            (False, "archived density used as the likelihood [as B21]"),
            (True, "uniform-log(D_A) stage-1 prior divided out")):
        _row(label, _fit(items, add_stat_error=True,
                         remove_stage1_prior=remove, **fiducial))

    print("\nSample dependence (fiducial B21 model)")
    _row("six galaxies", _fit(items, **base, **fiducial))
    _row("five galaxies, NGC4258 dropped", _fit(
        items, galaxies=joint.MCP_GALAXIES, **base, **fiducial))

    print("\nPer-galaxy H0, fiducial B21 model vs no velocity field")
    for item in items:
        one = (item["name"],)
        q16, med, q84 = _fit(items, galaxies=one, **base, **fiducial)
        _, med0, _ = _fit(items, galaxies=one, sigma_v=150.0, prior="flat",
                          velocity="none", **base)
        print(f"  {item['name']:12s} {med:5.2f} -{med - q16:4.2f}/"
              f"+{q84 - med:4.2f}   no field {med0:5.2f}   "
              f"shift {med - med0:+.2f}")


if __name__ == "__main__":
    main()
