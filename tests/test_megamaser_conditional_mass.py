"""Regression checks for the conditional-Hessian mass matrix."""
import jax
import jax.numpy as jnp
import numpy as np

from candel.model.maser_blackjax import _conditional_inverse_mass

jax.config.update("jax_enable_x64", True)


def test_conditional_inverse_mass_inverts_a_positive_definite_hessian():
    # log N(0, C) has -H = C^-1, so the returned metric must be C itself.
    C = np.array([[4.0, 1.0, 0.0],
                  [1.0, 9.0, 2.0],
                  [0.0, 2.0, 1.0]])
    P = np.linalg.inv(C)

    def logdensity(u):
        return -0.5 * u @ jnp.asarray(P) @ u

    got = np.asarray(_conditional_inverse_mass(
        logdensity, jnp.zeros(3, dtype=jnp.float64)))
    np.testing.assert_allclose(got, C, rtol=1e-8, atol=1e-10)


def test_conditional_inverse_mass_is_positive_definite_at_a_saddle():
    # The DE-MAP point can sit on a prior bound, leaving a negative curvature
    # direction.  |w| must give it a width matched to the curvature scale, so
    # the metric stays positive definite and the preconditioned curvature is
    # O(1) rather than the O(1e2) a hard floor leaves behind.
    w_true = np.array([1.0e6, 4.0, -0.5])
    H_neg = np.diag(w_true)                       # this is -H

    def logdensity(u):
        return -0.5 * u @ jnp.asarray(H_neg) @ u

    M = np.asarray(_conditional_inverse_mass(
        logdensity, jnp.zeros(3, dtype=jnp.float64)))
    assert np.all(np.linalg.eigvalsh(M) > 0), "metric must be positive definite"
    # the flipped direction gets width 1/|w|, not 1/floor
    np.testing.assert_allclose(np.diag(M), 1.0 / np.abs(w_true), rtol=1e-8)


def test_conditional_inverse_mass_floors_an_exactly_flat_direction():
    w_true = np.array([1.0e8, 1.0, 0.0])

    def logdensity(u):
        return -0.5 * u @ jnp.asarray(np.diag(w_true)) @ u

    M = np.asarray(_conditional_inverse_mass(
        logdensity, jnp.zeros(3, dtype=jnp.float64)))
    assert np.all(np.isfinite(M))
    assert np.all(np.linalg.eigvalsh(M) > 0)
    # floor is relative to the largest |eigenvalue| (1e-8 * 1e8 = 1)
    np.testing.assert_allclose(M[2, 2], 1.0, rtol=1e-6)


def test_conditional_inverse_mass_rejects_a_non_finite_hessian():
    """Fail loudly rather than return a NaN metric that poisons the chain."""
    import pytest

    def logdensity(u):
        return -0.5 * jnp.sum(u ** 2) / jnp.sum(u)   # singular at u = 0

    with pytest.raises(FloatingPointError, match="not finite"):
        _conditional_inverse_mass(logdensity, jnp.zeros(3, dtype=jnp.float64))
