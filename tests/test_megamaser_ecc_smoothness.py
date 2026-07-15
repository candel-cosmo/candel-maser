"""Eccentric megamaser velocity must be differentiable at e=0.

The Cartesian eccentricity (e_x, e_y) is sampled by the BlackJAX global block.
``predict_velocity_los`` therefore has to receive eccentricity in the smooth
combinations e_x, e_y -> (ecc*cos w, ecc*sin w, ecc^2); the old polar route via
sqrt/arctan2 produced NaN gradients at the origin (e=0), froze the global NUTS
step (every trajectory divergent), and is the bug this guards against.

Run:  venv_candel/bin/python -m pytest tests/test_megamaser_ecc_smoothness.py
"""
import jax
import jax.numpy as jnp

from candel.model.maser_physics import predict_velocity_los


def _v_sum(e_x, e_y, dperiapsis_dr=0.3):
    """Sum LOS velocity over a phi grid, eccentricity in Cartesian form.

    Mirrors how ``_r_precompute`` feeds the velocity: rotate (e_x, e_y) by the
    radial warp and pass the products, never the magnitude/direction split.
    """
    r = jnp.linspace(0.3, 0.9, 8)
    phi = jnp.linspace(-1.5, 1.5, 11)[:, None]
    sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)
    delta = dperiapsis_dr * (r - 0.5)
    cos_d, sin_d = jnp.cos(delta), jnp.sin(delta)
    V = predict_velocity_los(
        r, sin_phi, cos_phi, 50.0, 3.0, 1500.0, 0.0, jnp.sin(1.2),
        ecc2=e_x * e_x + e_y * e_y,
        ecc_cos_om=e_x * cos_d - e_y * sin_d,
        ecc_sin_om=e_x * sin_d + e_y * cos_d)
    return jnp.sum(V)


def test_velocity_gradient_finite_at_origin():
    g = jax.grad(_v_sum, argnums=(0, 1))(0.0, 0.0)
    assert jnp.isfinite(g[0]) and jnp.isfinite(g[1]), g


def test_circular_limit_matches_zero_ecc():
    # e=0 must reproduce the circular branch (ecc2 literal 0.0) exactly.
    ecc = _v_sum(0.0, 0.0)
    r = jnp.linspace(0.3, 0.9, 8)
    phi = jnp.linspace(-1.5, 1.5, 11)[:, None]
    circ = jnp.sum(predict_velocity_los(
        r, jnp.sin(phi), jnp.cos(phi), 50.0, 3.0, 1500.0, 0.0, jnp.sin(1.2)))
    assert jnp.allclose(ecc, circ, atol=1e-4), (ecc, circ)


if __name__ == "__main__":
    test_velocity_gradient_finite_at_origin()
    test_circular_limit_matches_zero_ecc()
    print("ok")
