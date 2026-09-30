"""Vext coordinate-frame regressions for the joint megamaser H0 model.

Vext is sampled in ICRS-Cartesian km/s, but the LOS `rhat` and the voxel
`rhat_*_3d` come from a reconstruction grid and are in that field's own frame
(Galactic for Carrick2015/Lilow2024, Supergalactic for CF4/CLONES/HAMLET).  The
radial projection is a physical scalar, so it must not depend on which frame it
is evaluated in.
"""

import numpy as np
import pytest

from candel.util import (radec_to_cartesian, radec_to_galactic,
                         radec_to_supergalactic)

from candel_maser.joint_H0_helpers import rotate_vext_to_frame

# The six megamaser hosts, and the Carrick2015 informative Vext prior mean
# (ICRS-Cartesian km/s) from config_maser.toml.
GALAXIES = {
    "CGCG074-064": (210.768491, 8.947582),
    "NGC4258": (184.74009, 47.30372),
    "NGC5765b": (222.71461, 5.11449),
    "NGC6264": (254.3172, 27.8496),
    "NGC6323": (258.32519, 43.78244),
    "UGC3789": (109.87872, 59.35513),
}
VEXT_ICRS = np.array([-36.1490, -19.6859, -196.4901])


def _rhat_in_frame(ra, dec, frame):
    """Build rhat the way `prepare_los_geometry` does for each frame."""
    ra, dec = np.atleast_1d(ra), np.atleast_1d(dec)
    if frame == "icrs":
        return radec_to_cartesian(ra, dec)[0]
    if frame == "galactic":
        return radec_to_cartesian(*radec_to_galactic(ra, dec))[0]
    if frame == "supergalactic":
        return radec_to_cartesian(*radec_to_supergalactic(ra, dec))[0]
    raise ValueError(frame)


@pytest.mark.parametrize("frame", ["galactic", "supergalactic"])
@pytest.mark.parametrize("galaxy", sorted(GALAXIES))
def test_vext_radial_projection_is_frame_invariant(galaxy, frame):
    """rhat_frame . R(Vext) must equal the ICRS projection rhat_icrs . Vext.

    The rotation matrices are `jnp` arrays and JAX runs in float32 here, so the
    tolerance is set well above that rounding (~2e-6 km/s) and well below the
    23-203 km/s error the missing rotation used to introduce.
    """
    ra, dec = GALAXIES[galaxy]
    reference = _rhat_in_frame(ra, dec, "icrs") @ VEXT_ICRS
    rotated = _rhat_in_frame(ra, dec, frame) @ np.asarray(
        rotate_vext_to_frame(VEXT_ICRS, frame))
    assert rotated == pytest.approx(reference, abs=1e-3)
