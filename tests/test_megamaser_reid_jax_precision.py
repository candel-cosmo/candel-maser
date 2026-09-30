"""Reid f2py calc_warped_model vs CANDEL JAX disk physics.

This convention regression aligns CANDEL's constants, angular-diameter
distance rounding, and Reid's circular-speed SR gamma before comparing
position, velocity, and acceleration in both circular and eccentric branches.
Production CANDEL intentionally uses its own constants and the true eccentric
orbital speed in the SR gamma.

Run:  venv_candel/bin/python -m pytest \
    packages/candel-maser/tests/test_megamaser_reid_jax_precision.py
"""
import math
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "candel_mpl_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import jax  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

try:
    from candel_maser.reid import reid_profile as rp
except ModuleNotFoundError as exc:  # private f2py module is gitignored
    if exc.name == "reidlik":
        pytest.skip(
            "private reidlik f2py module is not built.",
            allow_module_level=True)
    raise

import candel_maser.maser_physics as phys  # noqa: E402
from candel_maser.reid.run_reid_mcmc import reid_D_A, reid_H0  # noqa: E402


CLIGHT_REID = 2.997925e5
SEC_PER_YR_REID = 365.2422 * 86400.0
G_CGS_REID = 6.674e-8


def reid_candel_constants():
    n = rp.reidlik.numbers
    cv = n.vearth * math.sqrt(1.0e7 / 1000.0)
    ca = (cv ** 2 / (n.au2km * 1000.0)) * SEC_PER_YR_REID
    cg = (
        2.0 * G_CGS_REID * 1.0e7 * n.sun_mass
        / (CLIGHT_REID * 1.0e5) ** 2
        * 1.0e-5 / (1000.0 * n.au2km)
    )
    return cv, ca, cg


def reid_distance(g):
    v = g["Vsys_km_s"] + g["Vcor_km_s"]
    rp.fill_ez(g["H0"], g["Vsys_km_s"], g["Vcor_km_s"])
    n_v = int(v + 0.5)
    z = v / CLIGHT_REID
    return (
        CLIGHT_REID * rp.reidlik.ez_integral.ez_int[n_v - 1]
        / (g["H0"] * (1.0 + z))
    )


def reid_model(g, r_ref, r_mas, phi_deg):
    p = rp.globals_to_params(g)
    p[rp.NUM_GLOBAL] = r_mas
    p[rp.NUM_GLOBAL + 1] = phi_deg
    return rp.reidlik.calc_warped_model(p, rp.NUM_GLOBAL, 1, r_ref)


def candel_model(g, r_ref, r_mas, phi_deg, D_A):
    dr = r_mas - r_ref
    inc_reid = (
        g["i0_deg"]
        + g["di_dr_deg_mas"] * dr
        + g["d2i_dr2_deg_mas2"] * dr ** 2
    )
    inc = math.radians(180.0 - inc_reid)
    pa = math.radians(
        g["PA_deg"]
        + g["dPA_dr_deg_mas"] * dr
        + g["d2PA_dr2_deg_mas2"] * dr ** 2
    )
    phi = math.radians(phi_deg)
    peri = math.radians(
        g["peri_az_deg"] + g["dperi_dr_deg_mas"] * r_mas)
    ecc = g["ecc"]
    sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)
    sin_i, cos_i = jnp.sin(inc), jnp.cos(inc)
    sin_pa, cos_pa = jnp.sin(pa), jnp.cos(pa)

    x, y = phys.predict_position(
        r_mas, sin_phi, cos_phi,
        g["x0_mas"] * 1000.0, g["y0_mas"] * 1000.0,
        sin_i, cos_i, sin_pa, cos_pa)
    v_rel = phys.predict_velocity_los(
        r_mas, sin_phi, cos_phi, D_A, g["Mbh_1e7Msun"],
        g["Vsys_km_s"], 0.0, sin_i, ecc2=ecc * ecc,
        ecc_cos_om=ecc * math.cos(peri),
        ecc_sin_om=ecc * math.sin(peri))
    acc = phys.predict_acceleration_los(
        r_mas, sin_phi, cos_phi, D_A, g["Mbh_1e7Msun"], sin_i)
    return (
        float(x) / 1000.0,
        float(y) / 1000.0,
        float(v_rel) + g["Vsys_km_s"],
        float(acc),
    )


def main():
    rp.setup_numbers()
    g = {
        "H0": 70.0,
        "Mbh_1e7Msun": 4.0,
        "Vsys_km_s": 1000.0,
        "x0_mas": 0.1,
        "y0_mas": -0.2,
        "i0_deg": 87.0,
        "di_dr_deg_mas": 1.2,
        "d2i_dr2_deg_mas2": -0.1,
        "PA_deg": 88.0,
        "dPA_dr_deg_mas": 2.1,
        "d2PA_dr2_deg_mas2": 0.05,
        "ecc": 0.0,
        "peri_az_deg": 0.0,
        "dperi_dr_deg_mas": 0.0,
        "Vcor_km_s": 0.0,
        "sigma_x_mas": 0.0,
        "sigma_y_mas": 0.0,
        "sigma_vsys_km_s": 0.0,
        "sigma_vhv_km_s": 0.0,
        "sigma_acc_km_s_yr": 0.0,
    }
    r_ref = 0.5
    D_A = reid_distance(g)
    saved = (
        phys.C_v, phys.C_a, phys.C_g, phys.SPEED_OF_LIGHT,
        phys.REID_CIRCULAR_GAMMA,
    )
    phys.C_v, phys.C_a, phys.C_g = reid_candel_constants()
    phys.SPEED_OF_LIGHT = CLIGHT_REID
    phys.REID_CIRCULAR_GAMMA = True
    try:
        max_abs = 0.0
        labels = ("x", "y", "v", "a")
        for ecc, peri, slope in ((0.0, 0.0, 0.0), (0.17, 37.0, -2.3)):
            g["ecc"] = ecc
            g["peri_az_deg"] = peri
            g["dperi_dr_deg_mas"] = slope
            for r_mas, phi_deg in (
                    (0.7, 73.0), (1.1, -24.0), (0.35, 165.0)):
                rr = reid_model(g, r_ref, r_mas, phi_deg)
                cc = candel_model(g, r_ref, r_mas, phi_deg, D_A)
                diff = [abs(a - b) for a, b in zip(rr, cc)]
                max_abs = max(max_abs, *diff)
                print(
                    f"e={ecc:.2f} r={r_mas:.3f} mas "
                    f"phi={phi_deg:7.2f} deg "
                    + " ".join(f"d{label}={value:.3e}"
                               for label, value in zip(labels, diff)))
                assert diff[0] < 1e-12
                assert diff[1] < 1e-12
                assert diff[2] < 1e-8
                assert diff[3] < 1e-10
    finally:
        (phys.C_v, phys.C_a, phys.C_g, phys.SPEED_OF_LIGHT,
         phys.REID_CIRCULAR_GAMMA) = saved

    print(f"PASS: Reid/CANDEL physics agrees; max abs={max_abs:.3e}")


def test_reid_jax_precision():
    main()


def test_scalar_reid_h0_replays_fortran_distance():
    rp.setup_numbers()
    v = 667.0 - 259.7458
    D_A = 8.1421
    H0 = reid_H0(v, D_A)
    rp.fill_ez(H0, v, 0.0)
    n_v = int(v + 0.5)
    replay = (
        CLIGHT_REID * rp.reidlik.ez_integral.ez_int[n_v - 1]
        / (H0 * (1.0 + v / CLIGHT_REID))
    )
    assert replay == pytest.approx(D_A, abs=2e-14)
    assert reid_D_A(v, H0) == pytest.approx(D_A, abs=2e-14)


if __name__ == "__main__":
    main()
