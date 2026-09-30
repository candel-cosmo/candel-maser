"""Regression checks for the conditional-Hessian mass matrix."""
import jax
import jax.numpy as jnp
import numpy as np

from candel_maser.maser_blackjax import _conditional_inverse_mass

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


def test_conditional_inverse_mass_averages_over_latent_draws():
    """A sequence of log-densities is averaged as PRECISIONS, not widths."""
    P1 = np.diag([100.0, 4.0, 1.0])
    P2 = np.diag([100.0, 4.0, 9.0])

    def make(P):
        return lambda u: -0.5 * u @ jnp.asarray(P) @ u

    M = np.asarray(_conditional_inverse_mass(
        [make(P1), make(P2)], jnp.zeros(3, dtype=jnp.float64)))
    np.testing.assert_allclose(M, np.linalg.inv(0.5 * (P1 + P2)), rtol=1e-8,
                               atol=1e-12)
    # mean of precisions, not mean of covariances: 1/5 != 0.5*(1/1 + 1/9)
    assert not np.isclose(M[2, 2], 0.5 * (1.0 + 1.0 / 9.0))


def test_conditional_inverse_mass_averaging_repairs_an_indefinite_draw():
    """One bad latent draw must not set the metric on its own.

    Measured on NGC4258: across decorrelated latent draws at fixed theta the
    softest curvature spans 0.014 to 27.2, and single draws come out
    indefinite in the error-floor directions.
    """
    bad = np.diag([1.0e6, -0.5])           # indefinite
    good = np.diag([1.0e6, 4.0])

    def make(P):
        return lambda u: -0.5 * u @ jnp.asarray(P) @ u

    u0 = jnp.zeros(2, dtype=jnp.float64)
    M_bad = np.asarray(_conditional_inverse_mass([make(bad)], u0))
    M_avg = np.asarray(_conditional_inverse_mass([make(bad), make(good)], u0))
    assert np.all(np.linalg.eigvalsh(M_avg) > 0)
    # the single bad draw reports width 1/|-0.5|; the average reports 1/1.75
    np.testing.assert_allclose(M_bad[1, 1], 1.0 / 0.5, rtol=1e-8)
    np.testing.assert_allclose(M_avg[1, 1], 1.0 / 1.75, rtol=1e-8)
