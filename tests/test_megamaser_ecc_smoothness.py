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
from numpyro.distributions import Delta, Uniform

from candel.model import maser_physics
from candel.model.model_H0_maser import MaserDiskModel
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


def test_eccentric_model_accepts_a_fixed_periapsis_warp():
    """A Delta prior fixes the warp rather than being refused.

    Making (e_x, e_y) smooth through e=0 did not remove the funnel, it moved
    it: the likelihood sees the triple only through R(delta) . e, so the
    rotation angle delta = dperiapsis_dr * (r - r_ref) is unidentified as
    |e| -> 0.  Sampling it anyway is what broke NGC6323 (--add-ecc, 5000
    draws): dperiapsis_dr came back at its prior (posterior sd 208.69 vs
    prior sd 207.85) while dragging dv_sys/x0/y0 to r_hat 1.53/1.16/1.28.
    Fixing it restored r_hat <= 1.11 on every site.
    """
    model = object.__new__(MaserDiskModel)
    model.config = {"model": {"use_ecc": True}}
    model.priors = {"dperiapsis_dr": Delta(0.0)}
    model._configure_features({})
    assert model.sample_periapsis_warp is False

    model.priors = {"dperiapsis_dr": Uniform(-360.0, 360.0)}
    model._configure_features({})
    assert model.sample_periapsis_warp is True


def test_reid_speed_constant_reaches_optimised_eccentric_velocity():
    model = object.__new__(MaserDiskModel)
    r_pre = {
        "ecc2": jnp.asarray(0.04),
        "r_ang": jnp.asarray([0.7]),
        "sin_i": jnp.asarray([0.99]),
        "velocity_kep": jnp.asarray([910.0]),
        "velocity_los_scale": jnp.asarray([900.0]),
        "velocity_beta_c2": jnp.asarray([1e-5]),
        "velocity_zg": jnp.asarray([2e-6]),
        "velocity_scale": jnp.asarray(3e5),
        "dv_sys": jnp.asarray(0.0),
        "ecc_cos_om": jnp.asarray([0.15]),
        "ecc_sin_om": jnp.asarray([0.1]),
    }
    args = (r_pre, jnp.asarray(0.6), jnp.asarray(0.8), jnp.asarray(0))
    baseline = model._predict_velocity_on_grid(*args)
    saved = maser_physics.SPEED_OF_LIGHT
    try:
        maser_physics.SPEED_OF_LIGHT *= 0.99
        changed = model._predict_velocity_on_grid(*args)
    finally:
        maser_physics.SPEED_OF_LIGHT = saved
    assert not jnp.allclose(changed, baseline)


if __name__ == "__main__":
    test_velocity_gradient_finite_at_origin()
    test_circular_limit_matches_zero_ecc()
    print("ok")
