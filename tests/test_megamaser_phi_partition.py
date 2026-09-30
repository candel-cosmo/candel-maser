"""Focused checks for the opt-in peak-partition phi quadrature."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from candel_maser.model_H0_maser import (MaserDiskModel,  # noqa: E402
                                         _quadratic_log_grid_peak,
                                         neg_half_chi2_acceleration,
                                         neg_half_chi2_position,
                                         neg_half_chi2_velocity)


class _ToyModel:
    _phi_value = MaserDiskModel._phi_value
    _refine_phi_roots = MaserDiskModel._refine_phi_roots
    _integrate_phi_roots = MaserDiskModel._integrate_phi_roots
    _phi_partition_log_integral = MaserDiskModel._phi_partition_log_integral
    _phi_partition_group_log_integral = (
        MaserDiskModel._phi_partition_group_log_integral)

    def _phi_eval(self, r_pre, sin_phi, cos_phi):
        mu = r_pre["mu"]
        return r_pre["kappa"][..., None] * (
            cos_phi * jnp.cos(mu)[..., None]
            + sin_phi * jnp.sin(mu)[..., None])


def _eccentric_r_pre():
    r_ang = jnp.array([[0.28, 0.34], [0.39, 0.47]])
    i = jnp.deg2rad(jnp.array([[88.0, 89.0], [91.0, 92.0]]))
    Omega = jnp.deg2rad(jnp.array([[20.0, 21.0], [19.0, 18.0]]))
    ex, ey = 0.035, -0.02
    var_x = jnp.array([16.0, 25.0])
    var_y = jnp.array([16.0, 25.0])
    var_v = jnp.array([4.0, 4.0])
    var_a = jnp.array([0.01, 1.0])
    has_a = jnp.array([1.0, 0.0])
    return dict(
        r_ang=r_ang,
        sin_i=jnp.sin(i), cos_i=jnp.cos(i),
        sin_O=jnp.sin(Omega), cos_O=jnp.cos(Omega),
        ecc_cos_om=jnp.full_like(r_ang, ex),
        ecc_sin_om=jnp.full_like(r_ang, ey),
        ecc2=jnp.asarray(ex * ex + ey * ey),
        x0=jnp.asarray(2.0), y0=jnp.asarray(-1.0),
        D=jnp.asarray(110.0), M_BH=jnp.asarray(3.0),
        v_sys=jnp.asarray(7003.0), dv_sys=jnp.asarray(3.0),
        all_x=jnp.array([120.0, -160.0]),
        all_y=jnp.array([70.0, -90.0]),
        all_v_rel=jnp.array([650.0, -590.0]),
        all_a=jnp.array([0.4, 0.0]),
        var_x=var_x, var_y=var_y, var_v=var_v, var_a=var_a,
        weight_x=-0.5 / var_x, weight_y=-0.5 / var_y,
        weight_v=-0.5 / var_v, weight_a=-0.5 * has_a / var_a,
        has_a=has_a,
        has_any_accel=True,
    )


def _precomputed_model_fields(eccentric, shared_r=False):
    model = object.__new__(MaserDiskModel)
    model._all_x = jnp.array([120.0, -160.0])
    model._all_y = jnp.array([70.0, -90.0])
    model._all_v_rel = jnp.array([650.0, -590.0])
    model._all_a = jnp.array([0.4, 0.0])
    model._all_sigma_x2 = jnp.array([16.0, 25.0])
    model._all_sigma_y2 = jnp.array([16.0, 25.0])
    model._all_sigma_v2 = jnp.array([0.25, 0.25])
    model._all_sigma_a2 = jnp.array([0.01, 1.0])
    model._all_has_accel = jnp.array([True, False])
    model._all_is_clump2 = jnp.array([False, False])
    model.is_highvel = jnp.array([False, True])
    r_ang = (jnp.array([0.28, 0.34]) if shared_r else
             jnp.array([[0.28, 0.34], [0.39, 0.47]]))
    return model, model._r_precompute(
        r_ang, jnp.arange(2),
        2.0, -1.0, 110.0, 3.0, 7003.0,
        0.35, 0.35, 0.2,
        jnp.deg2rad(89.0), jnp.deg2rad(2.0),
        jnp.deg2rad(20.0), jnp.deg2rad(3.0),
        1.0, 1.0, 4.0, 4.0, 0.01,
        e_x=0.035 if eccentric else None,
        e_y=-0.02 if eccentric else None,
        dperiapsis_dr=jnp.deg2rad(4.0), dv_sys=3.0)


def test_clump2_floors_replace_standard_floors():
    model, _ = _precomputed_model_fields(eccentric=False)
    model._all_is_clump2 = jnp.array([True, False])
    r_pre = model._r_precompute(
        jnp.array([0.28, 0.39]), jnp.arange(2),
        2.0, -1.0, 110.0, 3.0, 7003.0,
        0.35, 0.35, 0.2,
        jnp.deg2rad(89.0), jnp.deg2rad(2.0),
        jnp.deg2rad(20.0), jnp.deg2rad(3.0),
        1.0, 1.0, 4.0, 4.0, 0.01,
        9.0, 16.0, 25.0, 0.09)

    np.testing.assert_allclose(r_pre["var_x"], [25.0, 26.0])
    np.testing.assert_allclose(r_pre["var_y"], [32.0, 26.0])
    np.testing.assert_allclose(r_pre["var_v"], [25.25, 4.25])
    np.testing.assert_allclose(r_pre["var_a"], [0.10, 1.01])


def test_partition_integrates_sharp_peaks_in_both_phi_halfplanes():
    model = _ToyModel()
    r_pre = {
        "r_ang": jnp.ones((2, 3)),
        "mu": jnp.array([[0.2, -1.0, 2.5], [0.0, 1.2, -2.4]]),
        "kappa": jnp.array([[10.0, 100.0, 1000.0],
                            [1.0, 50.0, 1.0e10]]),
    }
    got, roots, overflow = jax.jit(
        lambda rp: model._phi_partition_group_log_integral(
            "sys", rp, 257))(r_pre)
    ref = (jnp.log(2.0 * jnp.pi)
           + jnp.log(jax.scipy.special.i0e(r_pre["kappa"]))
           + r_pre["kappa"])

    np.testing.assert_allclose(got, ref, rtol=3e-16, atol=1e-9)
    assert np.all(np.asarray(roots) <= 2)
    assert not np.any(np.asarray(overflow))


def test_eccentric_value_only_partition_converges():
    model = object.__new__(MaserDiskModel)
    r_pre = _eccentric_r_pre()
    test, roots, overflow = model._phi_partition_group_log_integral(
        "sys", r_pre, 513)
    ref, _, ref_overflow = model._phi_partition_group_log_integral(
        "sys", r_pre, 1025,
        root_capacity=16, root_steps=8, root_order=17, drop_steps=24,
        core_order=48, tail_order=16)
    np.testing.assert_allclose(test, ref, rtol=0.0, atol=2e-8)
    assert np.all(np.asarray(roots) <= 8)
    assert not np.any(np.asarray(overflow))
    assert not np.any(np.asarray(ref_overflow))


def test_root_capacity_overflow_uses_finite_scan_fallback():
    class OscillatoryModel(_ToyModel):
        _phi_partition_root_capacity = 4

        def _phi_eval(self, r_pre, sin_phi, cos_phi):
            del r_pre
            phi = jnp.arctan2(sin_phi, cos_phi)
            return 10.0 * jnp.cos(10.0 * phi)

    model = OscillatoryModel()
    got, roots, overflow = model._phi_partition_group_log_integral(
        "sys", {"r_ang": jnp.asarray(1.0)}, 257)
    expected = np.log(2.0 * np.pi * np.i0(10.0))

    assert int(np.asarray(roots)) > model._phi_partition_root_capacity
    assert bool(np.asarray(overflow))
    assert np.isfinite(np.asarray(got))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-10)


def test_radius_only_precompute_preserves_circular_and_eccentric_integrands():
    hoisted_keys = {
        "pos_x_s", "pos_x_c", "pos_y_s", "pos_y_c", "accel_c",
        "velocity_0", "velocity_s", "velocity_kep", "velocity_los_scale",
        "velocity_beta_c2", "velocity_zg", "velocity_scale",
    }
    phi = jnp.linspace(-jnp.pi, jnp.pi, 41)
    for eccentric in (False, True):
        model, r_pre = _precomputed_model_fields(eccentric)
        legacy = {key: value for key, value in r_pre.items()
                  if key not in hoisted_keys}
        got = model._phi_eval(r_pre, jnp.sin(phi), jnp.cos(phi))
        expected = model._phi_eval(legacy, jnp.sin(phi), jnp.cos(phi))
        np.testing.assert_array_equal(got, expected)


def test_eccentric_f64_hybrid_matches_full_residual_kernel():
    model, r_pre = _precomputed_model_fields(True)
    phi = jnp.linspace(-jnp.pi, jnp.pi, 41)
    sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)
    rpad = (slice(None),) * r_pre["r_ang"].ndim + (None,)
    dpad = (slice(None),) + (None,) * r_pre["r_ang"].ndim
    X, Y, V, A = model._predict_on_grid(
        r_pre, sin_phi, cos_phi, rpad)
    expected = neg_half_chi2_position(
        r_pre["all_x"][dpad], r_pre["all_y"][dpad], X, Y,
        r_pre["var_x"][dpad], r_pre["var_y"][dpad])
    expected += neg_half_chi2_velocity(
        r_pre["all_v_rel"][dpad], V, r_pre["var_v"][dpad])
    expected += neg_half_chi2_acceleration(
        r_pre["all_a"][dpad], A, r_pre["var_a"][dpad],
        r_pre["has_a"][dpad])

    model._predict_on_grid = lambda *args: (_ for _ in ()).throw(
        AssertionError("hybrid path rebuilt every residual channel"))
    got = model._phi_eval(r_pre, sin_phi, cos_phi)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-9)


def test_eccentric_f64_shared_r_hybrid_matches_residual_kernel():
    model, r_pre = _precomputed_model_fields(True, shared_r=True)
    phi = jnp.linspace(-jnp.pi, jnp.pi, 41)
    sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)
    expected = model._phi_eval_shared_r(
        r_pre, sin_phi, cos_phi, use_quadform=False)
    got = model._phi_eval_shared_r(
        r_pre, sin_phi, cos_phi, use_quadform=True)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-9)


def test_quadratic_log_grid_peak_interpolates_and_guards_boundaries():
    r = jnp.exp(jnp.linspace(jnp.log(0.1), jnp.log(1.0), 17))
    target = jnp.asarray([0.23, 0.61])
    values = -(jnp.log(r)[None, :] - jnp.log(target)[:, None]) ** 2
    best = jnp.argmax(values, axis=-1)
    got = _quadratic_log_grid_peak(r, values, best)
    np.testing.assert_allclose(got, target, rtol=0.0, atol=2e-15)

    boundary_values = jnp.stack((-jnp.arange(17.0), jnp.arange(17.0)))
    boundary_best = jnp.argmax(boundary_values, axis=-1)
    boundary = _quadratic_log_grid_peak(
        r, boundary_values, boundary_best)
    np.testing.assert_array_equal(boundary, r[boundary_best])
