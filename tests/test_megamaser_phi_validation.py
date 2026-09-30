"""CPU checks for peak-partition validation orchestration."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import candel_maser.model_H0_maser as maser_module  # noqa: E402
from candel.model.integration import trapz_log_weights  # noqa: E402
from candel_maser.model_H0_maser import MaserDiskModel  # noqa: E402
from candel_maser.convergence.convergence_utils import (  # noqa: E402
    dense_phi_reference_per_spot, dense_r_phi_reference_per_spot)
from candel_maser.convergence.validate_phi_partition import (  # noqa: E402
    _irrelevant_decision, error_statistics, reference_convergence)


def test_per_spot_gate_catches_cancelled_total_error():
    stats = error_statistics([1.0, -1.0], [0.0, 0.0])
    assert stats["absolute_total_error"] == 0.0
    assert stats["max_absolute_spot_error"] == 1.0

    verdict = reference_convergence(
        [np.zeros(2), np.array([1.0, -1.0])], 2,
        {"total_atol": 0.1, "spot_atol": 0.5, "rms_atol": 1.0})
    assert not verdict["converged"]


def test_float32_phi_eval_does_not_take_float64_quadform(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("float32 entered the float64 quadratic form")

    monkeypatch.setattr(maser_module, "_neg_half_chi2_quadform", forbidden)
    model = object.__new__(MaserDiskModel)
    model._predict_on_grid = lambda *args: (
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.zeros((1, 3), dtype=jnp.float32),
        None)
    r_pre = {
        "ecc2": None,
        "r_ang": jnp.ones(1, dtype=jnp.float32),
        "all_x": jnp.zeros(1, dtype=jnp.float32),
        "all_y": jnp.zeros(1, dtype=jnp.float32),
        "all_v_rel": jnp.zeros(1, dtype=jnp.float32),
        "var_x": jnp.ones(1, dtype=jnp.float32),
        "var_y": jnp.ones(1, dtype=jnp.float32),
        "var_v": jnp.ones(1, dtype=jnp.float32),
        "weight_x": -0.5 * jnp.ones(1, dtype=jnp.float32),
        "weight_y": -0.5 * jnp.ones(1, dtype=jnp.float32),
        "weight_v": -0.5 * jnp.ones(1, dtype=jnp.float32),
        "has_any_accel": False,
    }
    got = model._phi_eval(
        r_pre, jnp.zeros(3, dtype=jnp.float32),
        jnp.ones(3, dtype=jnp.float32))
    assert got.dtype == jnp.float32
    assert got.shape == (1, 3)


def test_dense_acceptance_reference_uses_partition_support():
    class Model:
        n_spots = 1
        _idx_sys = jnp.array([], dtype=int)
        _idx_red = jnp.array([0])
        _idx_blue = jnp.array([], dtype=int)
        _phi_subranges = {"red": ((0.0, np.pi, 3),)}

        @staticmethod
        def _group_has_any_accel(type_key):
            del type_key
            return False

        @staticmethod
        def _r_precompute(r_ang, idx, *args, **kwargs):
            del idx, args, kwargs
            return {
                "r_ang": r_ang,
                "lnorm": jnp.zeros(r_ang.shape[0], dtype=r_ang.dtype),
                "lnorm_a": jnp.zeros(r_ang.shape[0], dtype=r_ang.dtype),
            }

        @staticmethod
        def _phi_eval(r_pre, sin_phi, cos_phi):
            del cos_phi
            return jnp.zeros_like(r_pre["r_ang"])[..., None] + 3 * sin_phi

    model = Model()
    phys_args = (None, None, jnp.asarray(1.0, dtype=jnp.float64))
    got = dense_phi_reference_per_spot(
        model, phys_args, {}, np.ones(1), 1001, 1,
        partition_support=True)
    phi = jnp.linspace(0.0, jnp.pi, 1001, dtype=jnp.float64)
    expected = jax.scipy.special.logsumexp(
        3 * jnp.sin(phi) + trapz_log_weights(phi))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)

    full = dense_phi_reference_per_spot(
        model, phys_args, {}, np.ones(1), 1001, 1)
    assert not np.isclose(full[0], got[0])


def test_dense_2d_reference_integrates_full_radial_support_in_chunks():
    class Model:
        n_spots = 2
        _idx_sys = jnp.array([], dtype=int)
        _idx_red = jnp.array([0, 1])
        _idx_blue = jnp.array([], dtype=int)
        _phi_subranges = {"red": ((0.0, np.pi, 3),)}

        @staticmethod
        def r_ang_range(distance):
            del distance
            return 1.0, 2.0

        @staticmethod
        def _group_has_any_accel(type_key):
            del type_key
            return False

        @staticmethod
        def _r_precompute(r_ang, idx, *args, **kwargs):
            del idx, args, kwargs
            return {
                "r_ang": r_ang,
                "lnorm": jnp.zeros(r_ang.shape[0], dtype=r_ang.dtype),
                "lnorm_a": jnp.zeros(r_ang.shape[0], dtype=r_ang.dtype),
            }

        @staticmethod
        def _phi_eval(r_pre, sin_phi, cos_phi):
            del cos_phi
            return -r_pre["r_ang"][..., None] + 3 * sin_phi

    phys_args = (None, None, jnp.asarray(1.0, dtype=jnp.float64))
    got = dense_r_phi_reference_per_spot(
        Model(), phys_args, {}, 101, 201, 16, 1)
    r = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(2.0), 101))
    phi = jnp.linspace(0.0, jnp.pi, 201)
    expected = jax.scipy.special.logsumexp(
        -r[:, None] + 3 * jnp.sin(phi)[None, :]
        + trapz_log_weights(r)[:, None]
        + trapz_log_weights(phi)[None, :])
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
    selected = dense_r_phi_reference_per_spot(
        Model(), phys_args, {}, 101, 201, 16, 1, spot_indices=[1])
    assert np.isneginf(selected[0])
    np.testing.assert_allclose(
        selected[1], expected, rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="requires float64"):
        dense_r_phi_reference_per_spot(
            Model(), (None, None, jnp.asarray(1.0, dtype=jnp.float32)),
            {}, 3, 3, 2, 1)


def test_irrelevant_decision_requires_both_deficits_and_railing():
    gate = -1000.0
    assert _irrelevant_decision(-2000.0, -2000.0, gate, True, False)
    assert _irrelevant_decision(-2000.0, -2000.0, gate, False, True)
    # Both deficits below the gate but nothing rails -> keep as relevant.
    assert not _irrelevant_decision(-2000.0, -2000.0, gate, False, False)
    # Only the production deficit clears the gate -> keep.
    assert not _irrelevant_decision(-2000.0, -500.0, gate, True, True)
    # Only the reference deficit clears the gate -> keep.
    assert not _irrelevant_decision(-500.0, -2000.0, gate, True, True)
