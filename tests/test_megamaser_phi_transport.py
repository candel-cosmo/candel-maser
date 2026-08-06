"""Regression check for the systemic-phi global-block transport."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from candel.model.maser_blackjax import MaserBlackJaxTarget


def test_systemic_phi_transport_roundtrip_and_shift():
    def phys_from_params_jax(theta, _):
        args = [jnp.asarray(0.0)] * 17
        args[0], args[1] = theta["x0"], theta["y0"]
        return tuple(args), {}

    model = SimpleNamespace(
        n_spots=2,
        _idx_sys=np.array([0]),
        _idx_red=np.array([1]),
        _idx_blue=np.array([], dtype=int),
        _phi_subranges={
            "sys": [(-np.pi, np.pi, 3)],
            "red": [(0.0, np.pi, 3)],
            "blue": [(-np.pi, 0.0, 3)],
        },
        is_highvel=jnp.array([False, True]),
        phys_from_params_jax=phys_from_params_jax,
    )
    target = object.__new__(MaserBlackJaxTarget)
    target.model, target.h = model, 0.7
    target._phi_transport_xy = (
        jnp.array([0.1, 0.0]), jnp.array([0.2, 0.0]))

    theta = {"x0": jnp.asarray(1.0), "y0": jnp.asarray(2.0)}
    phi = jnp.array([3.0, np.pi])
    residual = target.phi_transport_residual(theta, phi)
    np.testing.assert_allclose(
        target.phi_from_transport_residual(theta, residual), phi,
        atol=1e-6)

    shifted = target.phi_from_transport_residual(
        {"x0": jnp.asarray(2.0), "y0": jnp.asarray(2.0)}, residual)
    np.testing.assert_allclose(shifted, [2.9, np.pi], atol=1e-6)
