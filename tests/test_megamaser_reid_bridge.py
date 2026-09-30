"""Pure-Python regressions for the Reid/CANDEL convention bridge."""
from pathlib import Path

import numpy as np
import pytest
import tomli


import candel_maser.reid.pesce_globals as pg
import candel_maser.reid.run_reid_mcmc as rr


def test_scalar_reid_distance_uses_literal_fortran_mapping():
    v = 472.911
    H0 = 62.733
    expected = rr._reid_dnum_scalar(v) / H0
    assert rr.reid_D_A(v, H0) == pytest.approx(expected, abs=1e-14)
    assert rr.reid_H0(v, expected) == pytest.approx(H0, abs=1e-13)
    assert expected != pytest.approx(v / H0, abs=1e-3)


def test_vector_reid_distance_replays_each_integer_lookup():
    velocity = np.array([472.1, 472.9, 473.2, 7000.25])
    H0 = np.array([62.0, 63.0, 64.0, 73.0])
    got = rr.reid_D_A(velocity, H0)
    expected = np.array([
        rr._reid_dnum_scalar(v) / h for v, h in zip(velocity, H0)])
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


def test_ngc4258_reid_warp_is_moved_from_its_own_fixed_pivot():
    path = Path(pg.REID_NGC4258_BEST)
    with path.open("rb") as f:
        row = tomli.load(f)["globals"]
    master = {
        "model": {
            "galaxies": {
                "NGC4258": {
                    "r_ang_ref_periapsis": 5.1,
                },
            },
        },
    }
    point = pg.reid_ngc4258_point(master, path)
    r_ref = float(row["r_ref_mas"])

    for radius in (0.0, 5.1, r_ref, 8.0):
        i_candel = (
            point["i0_r0_deg"] + point["di_dr_r0"] * radius
            + point["d2i_dr2"] * radius**2)
        i_reid = (
            float(row["i0_deg"])
            + float(row["di_dr_deg_mas"]) * (radius - r_ref)
            + float(row["d2i_dr2_deg_mas2"]) * (radius - r_ref)**2)
        pa_candel = (
            point["Omega0_r0_deg"] + point["dOmega_dr_r0"] * radius
            + point["d2Omega_dr2"] * radius**2)
        pa_reid = (
            float(row["PA_deg"])
            + float(row["dPA_dr_deg_mas"]) * (radius - r_ref)
            + float(row["d2PA_dr2_deg_mas2"]) * (radius - r_ref)**2)
        assert i_candel == pytest.approx(180.0 - i_reid, abs=1e-12)
        assert pa_candel == pytest.approx(pa_reid, abs=1e-12)

    peri_at_pivot = np.degrees(np.arctan2(
        point["e_y"], point["e_x"]))
    slope = float(row["dperi_dr_deg_mas"])
    for radius in (0.0, 5.1, r_ref, 8.0):
        peri_candel = peri_at_pivot + slope * (radius - 5.1)
        peri_reid = float(row["peri_az_deg"]) + slope * radius
        delta = (peri_candel - peri_reid + 180.0) % 360.0 - 180.0
        assert delta == pytest.approx(0.0, abs=1e-12)

    expected_distance = rr.reid_D_A(
        float(row["Vsys_km_s"]) + float(row["Vcor_km_s"]),
        float(row["H0"]))
    assert point["D_A"] == pytest.approx(expected_distance, abs=1e-14)
