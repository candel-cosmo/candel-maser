"""Metadata regressions for single-galaxy megamaser evidence."""

import numpy as np


import candel_maser.evidence_single_galaxy as ev


class DummyTarget:
    def __init__(self, names):
        self.names = tuple(names)


def test_chain_attrs_restore_uniform_da_prior():
    cfg = {
        "model": {
            "D_c_prior": "uniform",
            "galaxies": {
                "NGC6264": {
                    "mass_parameterization": "log_mbh",
                    "use_ecc": False,
                    "use_quadratic_warp": True,
                },
            },
        },
    }
    attrs = {
        "uniform_da_prior": True,
        "mass_parameterization": b"eta",
        "use_ecc": np.bool_(True),
        "use_quadratic_warp": "false",
    }

    gblk = ev._apply_chain_attrs_to_config(cfg, "NGC6264", attrs)

    assert cfg["model"]["D_c_prior"] == "uniform_D_A"
    assert gblk["mass_parameterization"] == "eta"
    assert gblk["use_ecc"] is True
    assert gblk["use_quadratic_warp"] is False


def test_sampled_global_names_ignore_deterministic_extras():
    target = DummyTarget(("D_A", "eta", "x0"))
    samples = {
        "D_A": np.array([1.0, 2.0]),
        "eta": np.array([3.0, 4.0]),
        "x0": np.array([5.0, 6.0]),
        "D_c": np.array([7.0, 8.0]),
        "log_MBH": np.array([9.0, 10.0]),
    }

    names = ev._sampled_global_names(target, samples)
    X = ev._stack_globals(samples, names)

    assert names == ("D_A", "eta", "x0")
    np.testing.assert_allclose(X, [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
