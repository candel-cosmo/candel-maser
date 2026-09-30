"""Regression check for the systemic-phi global-block transport."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from candel_maser.maser_blackjax import MaserBlackJaxTarget


def test_systemic_phi_transport_roundtrip_and_shift():
    def phys_from_params_jax(theta, _):
        args = [jnp.asarray(0.0)] * 17
        args[0], args[1] = theta["x0"], theta["y0"]
        args[4] = theta.get("v_sys", jnp.asarray(0.0))
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
    target._phi_transport_k = (
        jnp.array([0.1, 0.0]), jnp.array([0.2, 0.0]),
        jnp.array([0.05, 0.0]))

    theta = {"x0": jnp.asarray(1.0), "y0": jnp.asarray(2.0),
             "v_sys": jnp.asarray(4.0)}
    phi = jnp.array([3.0, np.pi])
    residual = target.phi_transport_residual(theta, phi)
    np.testing.assert_allclose(
        target.phi_from_transport_residual(theta, residual), phi,
        atol=1e-6)

    shifted = target.phi_from_transport_residual(
        {"x0": jnp.asarray(2.0), "y0": jnp.asarray(2.0),
         "v_sys": jnp.asarray(4.0)}, residual)
    np.testing.assert_allclose(shifted, [2.9, np.pi], atol=1e-6)

    # the dv_sys leg moves systemic phi by -kv * delta_v_sys (0.05 * 4 = 0.2),
    # hv phi untouched
    shifted_v = target.phi_from_transport_residual(
        {"x0": jnp.asarray(1.0), "y0": jnp.asarray(2.0),
         "v_sys": jnp.asarray(8.0)}, residual)
    np.testing.assert_allclose(shifted_v, [2.8, np.pi], atol=1e-6)


def test_systemic_phi_transport_coefficients_include_the_velocity_channel():
    """kx, ky must be damped by the velocity term, and kv must be non-zero.

    The least-squares response of systemic phi to (x0, y0, dv_sys) divides by
    the TOTAL Fisher information in phi.  Dropping the velocity channel (which
    for NGC4258 carries ~92 per cent of it) inflates kx, ky and loses the
    dv_sys leg entirely.
    """
    from candel_maser import maser_blackjax as mb
    from candel_maser import maser_physics

    n = 3
    r_ang = jnp.array([3.9, 4.1, 5.0])
    model = SimpleNamespace(
        n_spots=n,
        is_highvel=jnp.array([False, False, True]),
        _all_sigma_x2=jnp.full((n,), 1.0),
        _all_sigma_y2=jnp.full((n,), 1.0),
        _all_sigma_v2=jnp.full((n,), 1e-4),
        use_clump2_floors=False,
    )
    phys = [jnp.asarray(0.0)] * 19
    phys[mb._I_RREF_I] = jnp.asarray(5.1)
    phys[mb._I_RREF_OMEGA] = jnp.asarray(5.1)
    phys[mb._I_I0] = jnp.asarray(np.deg2rad(95.7))
    phys[mb._I_OMEGA0] = jnp.asarray(np.deg2rad(86.2))
    phys[mb._I_DA] = jnp.asarray(7.42)
    phys[mb._I_MBH] = jnp.asarray(10.0 ** (7.59 - 7.0))
    phys[mb._I_VSYS] = jnp.asarray(472.0)
    phys[mb._I_SXF2] = jnp.asarray(4.0)
    phys[mb._I_SYF2] = jnp.asarray(36.0)
    phys[mb._I_VARVSYS] = jnp.asarray(0.13)

    kx, ky, kv = mb._systemic_phi_coefficients(
        model, r_ang, tuple(phys), {})

    # high-velocity spots are never transported
    assert float(kx[2]) == 0.0 and float(ky[2]) == 0.0 and float(kv[2]) == 0.0
    # the dv_sys leg exists and is the dominant response
    assert np.all(np.abs(np.asarray(kv)[:2]) > 0.0)
    assert np.abs(float(kv[0])) > np.abs(float(kx[0]))

    # kx must be strictly smaller than the x/y-only coefficient it replaces
    _, Omega_r = maser_physics.warp_geometry(
        r_ang, phys[mb._I_RREF_I], phys[mb._I_RREF_OMEGA], phys[mb._I_I0],
        phys[mb._I_DI], phys[mb._I_OMEGA0], phys[mb._I_DOMEGA], 0.0, 0.0)
    dx = 1e3 * r_ang * jnp.sin(Omega_r)
    dy = 1e3 * r_ang * jnp.cos(Omega_r)
    var_x = model._all_sigma_x2 + phys[mb._I_SXF2]
    var_y = model._all_sigma_y2 + phys[mb._I_SYF2]
    denom_xy = dx ** 2 / var_x + dy ** 2 / var_y
    kx_xy_only = np.asarray(dx / var_x / denom_xy)
    assert np.all(np.abs(np.asarray(kx)[:2]) < np.abs(kx_xy_only[:2]))
