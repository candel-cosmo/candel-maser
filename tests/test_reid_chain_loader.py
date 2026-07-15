"""Regression tests for Reid fort.7 chain parsing."""
import math
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REID_DIR = os.path.join(REPO_ROOT, "scripts", "megamaser", "check_reid")
if REID_DIR not in sys.path:
    sys.path.insert(0, REID_DIR)

from run_reid_mcmc import FORT7_WIDTHS, load_chain  # noqa: E402


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
