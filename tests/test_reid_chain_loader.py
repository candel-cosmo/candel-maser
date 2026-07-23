"""Regression tests for the Reid bridge and fort.7 chain parsing."""
import math
import os
import sys

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REID_DIR = os.path.join(REPO_ROOT, "scripts", "megamaser", "check_reid")
if REID_DIR not in sys.path:
    sys.path.insert(0, REID_DIR)

from run_reid_mcmc import (FORT7_WIDTHS, load_chain, load_config_init,  # noqa: E402
                           reid_H0, shift_warp_pivots)


VALUES = [
    70.0,
    3.5,
    780.0,
    123.4567,
    -123.4567,
    92.0,
    1.25,
    -0.125,
    220.0,
    -0.75,
    0.025,
    0.0,
    15.0,
    -1.0,
    50.0,
    0.01,
    0.02,
    2.0,
    3.0,
    0.3,
]


@pytest.mark.parametrize(
    ("mass_parameterization", "expected_log_mbh"),
    [("eta", 6.8041 + math.log10(8.1421)), ("log_mbh", 7.5)],
)
def test_config_init_uses_active_distance_mass_and_quadratic_pivot(
        tmp_path, mass_parameterization, expected_log_mbh):
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[model]
mass_parameterization = "{mass_parameterization}"

[model.galaxies.NGC4258]
v_sys_obs = 667.0
r_ang_ref_i = 5.1
r_ang_ref_Omega = 5.1
r_ang_ref_periapsis = 5.1

[model.galaxies.NGC4258.init_qw]
D_A = 8.1421
eta = 6.8041
log_MBH = 7.5
dv_sys = -259.7458
i0 = 94.2829
di_dr = 0.3154
d2i_dr2 = 0.4053
Omega0 = 85.9489
dOmega_dr = 2.5803
d2Omega_dr2 = -0.3628
"""
    )

    init = load_config_init(
        config, "NGC4258", 0.0, variant="init_qw").values
    assert init["H0"] == pytest.approx(
        reid_H0(init["Vsys_km_s"], 8.1421), rel=1e-14)
    assert init["Mbh_1e7Msun"] == pytest.approx(
        10.0 ** (expected_log_mbh - 7.0), rel=1e-14)

    reid_r_ref = 6.276756628362692
    shifted = shift_warp_pivots(init, reid_r_ref)
    for r in (1.8, 5.1, reid_r_ref, 8.9):
        dr_candel = r - 5.1
        dr_reid = r - reid_r_ref
        i_candel = 94.2829 + 0.3154 * dr_candel + 0.4053 * dr_candel**2
        pa_candel = 85.9489 + 2.5803 * dr_candel - 0.3628 * dr_candel**2
        i_reid = (
            shifted["i0_deg"]
            + shifted["di_dr_deg_mas"] * dr_reid
            + shifted["d2i_dr2_deg_mas2"] * dr_reid**2
        )
        pa_reid = (
            shifted["PA_deg"]
            + shifted["dPA_dr_deg_mas"] * dr_reid
            + shifted["d2PA_dr2_deg_mas2"] * dr_reid**2
        )
        assert i_reid == pytest.approx(180.0 - i_candel, abs=2e-14)
        assert pa_reid == pytest.approx(pa_candel, abs=2e-14)


def _make_row(iter_=100, walker=1, values=VALUES, lnp=-1234.56789):
    row = (
        f"{iter_:10d}{walker:6d}"
        f"{values[0]:10.5f}{values[1]:10.5f}"
        f"{values[2]:9.2f}{values[3]:9.4f}{values[4]:9.4f}"
        f"{values[5]:8.2f}{values[6]:8.3f}{values[7]:8.3f}"
        f"{values[8]:8.2f}{values[9]:8.3f}{values[10]:8.3f}"
        f"{values[11]:6.3f}{values[12]:8.2f}{values[13]:8.2f}"
        f"{values[14]:8.2f}{values[15]:8.4f}{values[16]:8.4f}"
        f"{values[17]:8.4f}{values[18]:8.4f}{values[19]:8.4f}"
        f"{lnp:13.5E}".replace("E", "D")
    )
    assert len(row) == sum(FORT7_WIDTHS)
    return row


def test_load_chain_reads_touching_fixed_width_fields(tmp_path):
    row = _make_row()
    assert len(row.split()) == 22

    path = tmp_path / "fort.7"
    path.write_text("! Ho values shifted down; reconstruct using 0.000000\n"
                    + row + "\n")

    arr = load_chain(path)
    assert arr.shape == (1,)
    assert arr["iter"][0] == 100
    assert arr["walker"][0] == 1
    assert arr["x0_mas"][0] == VALUES[3]
    assert arr["y0_mas"][0] == VALUES[4]
    assert arr["lnP"][0] == -1234.57


def test_load_chain_maps_field_overflow_to_nan(tmp_path):
    """A run whose H0 wanders near zero can blow up a downstream column
    (e.g. Mbh) past its f10.5 field width; Fortran fills such an
    overflowing field with '*' rather than truncating it."""
    row = _make_row()
    field_start = sum(FORT7_WIDTHS[:3])  # Mbh_1e7Msun field (index 3)
    field_width = FORT7_WIDTHS[3]
    row = row[:field_start] + "*" * field_width + row[field_start + field_width:]

    path = tmp_path / "fort.7"
    path.write_text("! Ho values shifted down; reconstruct using 0.000000\n"
                    + row + "\n")

    arr = load_chain(path)
    assert arr.shape == (1,)
    assert math.isnan(arr["Mbh_1e7Msun"][0])
    assert arr["x0_mas"][0] == VALUES[3]  # other columns unaffected


def test_load_chain_drops_truncated_last_row(tmp_path):
    """A job killed mid-write leaves a short final line; it should be
    dropped rather than corrupting the parse or the array shape."""
    complete = _make_row(iter_=100)
    truncated = _make_row(iter_=200)[:50]

    path = tmp_path / "fort.7"
    path.write_text("! Ho values shifted down; reconstruct using 0.000000\n"
                    + complete + "\n" + truncated)

    arr = load_chain(path)
    assert arr.shape == (1,)
    assert arr["iter"][0] == 100
