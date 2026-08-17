import os
import sys

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEGAMASER_DIR = os.path.join(REPO_ROOT, "scripts", "megamaser")
if MEGAMASER_DIR not in sys.path:
    sys.path.insert(0, MEGAMASER_DIR)

import run_joint_H0 as joint  # noqa: E402


def test_distance_prior_selection_compatibility():
    assert joint._resolve_distance_prior("none", None) == "distance"
    assert joint._resolve_distance_prior("redshift", None) == "volume"
    assert joint._resolve_distance_prior("none", "log-distance") == (
        "log-distance")
    for prior in ("distance", "log-distance"):
        with pytest.raises(ValueError, match="requires.*uniform-in-volume"):
            joint._resolve_distance_prior("redshift", prior)


def test_log_distance_samples_log_da(monkeypatch):
    monkeypatch.setattr(
        joint, "healpix_los_vectors",
        lambda _: jnp.asarray([[1.0, 0.0, 0.0]]))
    item = {
        "name": "test",
        "D_lo": 40.0,
        "D_hi": 80.0,
        "DA_lo": 39.0,
        "DA_hi": 78.0,
        "DA_init": 55.0,
        "D_init": 56.0,
        "rhat_frame": "icrs",
    }
    priors = {
        "H0": dist.Uniform(10.0, 200.0),
        "sigma_pec": dist.Uniform(10.0, 1000.0),
        "Vext_mag": dist.Uniform(0.0, 1000.0),
    }
    target = joint.ToyDistanceTarget(
        [item], priors, "none", None, "log-distance", 1.0, False,
        False, 1)

    site, prior = target.specs[-1]
    assert site == "test__log_D_A"
    np.testing.assert_allclose(
        [prior.low, prior.high], np.log([item["DA_lo"], item["DA_hi"]]))
    init = joint._toy_init(target)
    assert float(jnp.exp(init[site])) == pytest.approx(item["DA_init"])
    params = {"H0": jnp.asarray(70.0), site: init[site]}
    assert float(target._D_A(params, item)) == pytest.approx(item["DA_init"])
    assert float(target._D_c(params, item)) > item["DA_init"]


def test_p20_log_prior_is_removed_from_kde():
    samples = np.linspace(45.0, 75.0, 200)
    grid, log_uniform, _, _ = joint._build_log_distance_likelihood(
        samples, 40.0, 80.0, 64, 0)
    _, log_p20, _, _ = joint._build_log_distance_likelihood(
        samples, 40.0, 80.0, 64, 0, "uniform_log_D_A")

    delta = np.asarray(log_p20 - log_uniform)
    np.testing.assert_allclose(
        np.diff(delta), np.diff(np.log(np.asarray(grid))), atol=1e-6)
