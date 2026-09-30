import jax
import jax.numpy as jnp
import numpy as np

from candel_maser.maser_blackjax import (_bounded_r_from_z,
                                         _r_ang_seed_bounds,
                                         _z_from_bounded_r)


def test_bounded_radius_transform_and_jacobian():
    z = jnp.array([-10.0, 0.0, 10.0])
    r_hat = jnp.array([0.8, 1.0, 1.2])
    r, log_jac = _bounded_r_from_z(z, r_hat, 0.1, 2.0)

    assert np.all(np.asarray(r) > 0.1)
    assert np.all(np.asarray(r) < 2.0)
    np.testing.assert_allclose(r[1], r_hat[1], rtol=1e-6)
    np.testing.assert_allclose(
        _z_from_bounded_r(r, r_hat, 0.1, 2.0), z,
        rtol=2e-3, atol=1e-12)

    derivative = jax.vmap(jax.grad(
        lambda value, centre: _bounded_r_from_z(
            value, centre, 0.1, 2.0)[0]))(z, r_hat)
    np.testing.assert_allclose(
        derivative, jnp.exp(log_jac), rtol=2e-4)


def test_ngc5765b_angular_bounds_do_not_follow_sampled_distance():
    class Model:
        galaxy_name = "NGC5765b"
        config = {"model": {"galaxies": {"NGC5765b": {
            "v_cmb_kms": 8525.7}}}}

        @staticmethod
        def redshift2distance(value, h, is_velocity=False):
            assert is_velocity
            return value / (100.0 * h)

        @staticmethod
        def r_ang_range(D_A):
            return 0.01 / D_A, 1.5 / D_A

    bounds_near = _r_ang_seed_bounds(Model(), jnp.asarray(90.0), 0.73)
    bounds_far = _r_ang_seed_bounds(Model(), jnp.asarray(150.0), 0.73)
    np.testing.assert_allclose(bounds_near, bounds_far)
