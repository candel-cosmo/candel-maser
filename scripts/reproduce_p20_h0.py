#!/usr/bin/env python
"""Reproduce Pesce et al. (2020) H0 method (1) from archived distances.

Run from the repository root with::

    venv_candel/bin/python packages/candel-maser/scripts/reproduce_p20_h0.py
"""
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

import candel_maser.run_joint_H0 as joint  # noqa: E402


C_KM_S = 299792.458
P20_GALAXIES = (
    "UGC3789", "NGC6264", "NGC6323", "NGC5765b",
    "CGCG074-064", "NGC4258",
)
P20_VELOCITIES = {
    "UGC3789": (3319.9, 0.8),
    "NGC6264": (10192.6, 0.8),
    "NGC6323": (7801.5, 1.5),
    "NGC5765b": (8525.7, 0.7),
    "CGCG074-064": (7172.2, 1.9),
    "NGC4258": (679.3, 0.4),
}
P20_REPORTED_MEDIANS = {
    "UGC3789": 75.8,
    "NGC6264": 73.8,
    "NGC6323": 73.8,
    "NGC5765b": 74.1,
    "CGCG074-064": 72.5,
    "NGC4258": 73.6,
    None: 73.9,
}


def _distance_posteriors():
    grids = {}
    root = Path(joint.ROOT) / "data" / "Megamaser" / "external" / "Dom_data"
    for galaxy in P20_GALAXIES[:-1]:
        samples = np.loadtxt(root / f"D_archivedP20_{galaxy}.txt")
        cfg = joint.MASTER_CFG["model"]["galaxies"][galaxy]
        lo, hi = joint._toy_D_A_bounds(cfg["D_lo"], cfg["D_hi"])
        D_A, log_q, _, _ = joint._build_log_distance_likelihood(
            samples, lo, hi, joint.TOY_DISTANCE_GRID_SIZE,
            joint.TOY_DISTANCE_MAX_SAMPLES, "uniform_D_A")
        grids[galaxy] = np.asarray(D_A), np.exp(np.asarray(log_q))

    D_A = np.linspace(6.8, 8.4, 2048)
    grids["NGC4258"] = D_A, np.exp(-0.5 * ((D_A - 7.58) / 0.11) ** 2)
    return grids


def _relation_grid():
    """P20 Equation 1 and its derivative, with z as the grid coordinate."""
    omega_m = float(joint.MASTER_CFG["model"]["Om"])
    z = np.linspace(0.0, 0.25, 200001)
    a = 3.0 * omega_m / 4.0
    b = omega_m * (9.0 * omega_m - 4.0) / 8.0
    poly = 1.0 - a * z + b * z ** 2
    scaled_D_A = z * poly / (1.0 + z)
    derivative = (
        (poly + z * (-a + 2.0 * b * z)) * (1.0 + z) - z * poly
    ) / (1.0 + z) ** 2
    return z, scaled_D_A, derivative


def _log_likelihoods(grids, H0, flat_velocity):
    z_grid, scaled_D_A, derivative = _relation_grid()
    out = {}
    for galaxy, (D_A, q_D_A) in grids.items():
        velocity, statistical_error = P20_VELOCITIES[galaxy]
        sigma = np.hypot(250.0, statistical_error)
        pieces = []
        for start in range(0, H0.size, 100):
            H = H0[start:start + 100]
            scaled = H[:, None] * D_A[None, :] / C_KM_S
            z = np.interp(scaled, scaled_D_A, z_grid)
            model_velocity = C_KM_S * z
            integrand = q_D_A[None, :] * np.exp(
                -0.5 * ((model_velocity - velocity) / sigma) ** 2)
            if flat_velocity:
                # P20 samples true velocity with a flat prior. Changing the
                # integration coordinate to D_A therefore requires dv/dD_A.
                integrand *= H[:, None] / np.interp(
                    z, z_grid, derivative)
            pieces.append(np.trapezoid(integrand, D_A, axis=1))
        out[galaxy] = np.log(np.maximum(np.concatenate(pieces), 1e-300))
    return out


def _summary(H0, log_likelihoods, galaxies):
    log_p = sum(
        (log_likelihoods[galaxy] for galaxy in galaxies),
        start=np.zeros_like(H0),
    )
    p = np.exp(log_p - np.max(log_p))
    cdf = np.zeros_like(H0)
    cdf[1:] = np.cumsum((p[1:] + p[:-1]) * 0.5 * np.diff(H0))
    cdf /= cdf[-1]
    return np.interp((0.16, 0.5, 0.84), cdf, H0)


def main():
    H0 = np.linspace(45.0, 105.0, 3001)
    grids = _distance_posteriors()
    p20 = _log_likelihoods(grids, H0, flat_velocity=True)
    flat_D_A = _log_likelihoods(grids, H0, flat_velocity=False)

    print("P20 method (1): fixed sigma_pec=250 km/s, flat true velocities")
    print("fit                         reproduced          P20    delta")
    differences = []
    for dropped in (*P20_GALAXIES, None):
        galaxies = tuple(g for g in P20_GALAXIES if g != dropped)
        q16, median, q84 = _summary(H0, p20, galaxies)
        reported = P20_REPORTED_MEDIANS[dropped]
        differences.append(abs(median - reported))
        label = "all six" if dropped is None else f"drop {dropped}"
        print(
            f"{label:24s} {median:5.2f} -{median-q16:4.2f}/+{q84-median:4.2f}"
            f"   {reported:4.1f}   {median-reported:+.2f}")

    five = P20_GALAXIES[:-1]
    flat_q = _summary(H0, flat_D_A, five)[1]
    p20_five = _summary(H0, p20, five)[1]
    p20_six = _summary(H0, p20, P20_GALAXIES)[1]
    print("\nControlled decomposition")
    print(f"five galaxies, q(D_A) dD_A measure: {flat_q:.2f}")
    print(f"+ P20 flat-velocity Jacobian:        {p20_five:.2f} "
          f"({p20_five-flat_q:+.2f})")
    print(f"+ NGC4258:                           {p20_six:.2f} "
          f"({p20_six-p20_five:+.2f})")

    if abs(p20_six - 73.9) > 0.1 or max(differences) > 0.3:
        raise RuntimeError("P20 reproduction is outside its validation gate")
    print(f"\ncheck passed: maximum median discrepancy = "
          f"{max(differences):.2f} km/s/Mpc")


if __name__ == "__main__":
    main()
