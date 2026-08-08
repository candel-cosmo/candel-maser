"""Focused checks for the opt-in peak-partition phi quadrature."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from candel.model.model_H0_maser import (  # noqa: E402
    MaserDiskModel, _quadratic_log_grid_peak, _scan_log_radius_scale,
    neg_half_chi2_acceleration, neg_half_chi2_position,
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


def test_scan_radius_scale_covers_disconnected_active_envelope():
    r = jnp.exp(jnp.linspace(jnp.log(0.1), jnp.log(1.0), 11))
    values = jnp.array([
        [-100.0, -1.0, 0.0, -2.0, -100.0, -100.0,
         -3.0, -1.0, -100.0, -100.0, -100.0],
    ])
    centre = r[jnp.array([2])]
    scale = _scan_log_radius_scale(r, values, centre, 4.0, 10.0)
    expected_half_span = jnp.log(r[8]) - jnp.log(r[2])
    np.testing.assert_allclose(10.0 * scale, expected_half_span)


def test_asymmetric_local_grid_does_not_collapse_at_support_edge():
    model = object.__new__(MaserDiskModel)
    model._K_sigma = 10.0
    model._asymmetric_r_local = True
    t = jnp.linspace(-jnp.arcsinh(10.0), jnp.arcsinh(10.0), 9)
    model._sinh_t_frozen = jnp.sinh(t)
    r, _ = model._build_local_sinh(
        jnp.array([0.1]), jnp.array([0.1]), 0.1, 1.0)
    np.testing.assert_allclose(r[0, 0], 0.1, rtol=0.0, atol=1e-15)
    assert r[0, -1] > 0.1
    assert np.all(np.diff(np.asarray(r[0])) >= 0.0)


def test_global_radius_count_is_integration_and_variant_agnostic():
    for integration, eccentric, quadratic_warp in (
            ("peak-partition", False, False),
            ("peak-partition", True, False),
            ("peak-partition", False, True),
            ("peak-partition", True, True),
            ("fixed-grid", True, True)):
        model = object.__new__(MaserDiskModel)
        model.config = {"model": {"n_r_local": 256, "n_r_global": 176}}
        model.phi_integration = integration
        model.use_ecc = eccentric
        model.use_quadratic_warp = quadratic_warp
        model._build_r_config({}, {})
        assert model._n_r_global == 176


def test_peak_radius_stencil_refines_coarse_group_centres():
    class Model:
        phi_integration = "peak-partition"
        _peak_r_refine_steps = 1
        _peak_r_refine_order = 7
        _peak_r_refine_hv_only = False
        _peak_r_width_steps = 0
        _n_r_global = 128
        _K_sigma = 10.0
        _refine_r_center_group = MaserDiskModel._refine_r_center_group

        @staticmethod
        def _group_has_any_accel(type_key):
            del type_key
            return False

        @staticmethod
        def _r_precompute(r_ang, idx, *args, **kwargs):
            del args, kwargs
            target = jnp.array([0.23, 0.61])[idx]
            return {"r_ang": r_ang, "target": target}

        @staticmethod
        def _phi_partition_scan_size(type_key):
            del type_key
            return 17

        @staticmethod
        def _phi_partition_group_log_integral(
                type_key, r_pre, n_scan):
            del type_key, n_scan
            value = -(jnp.log(r_pre["r_ang"])
                      - jnp.log(r_pre["target"][:, None])) ** 2
            return value, jnp.zeros_like(value, dtype=int), jnp.zeros_like(
                value, dtype=bool)

    model = Model()
    target = jnp.array([0.23, 0.61])
    got, scale = model._refine_r_center_group(
        "sys", jnp.arange(2), target * jnp.array([1.04, 0.96]),
        jnp.array([0.02, 0.03]), 0.1, 1.0, (), {})
    # The final three-point quadratic vertex is exact for this log-Gaussian;
    # a plain bracket midpoint is not.
    np.testing.assert_allclose(got, target, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(
        scale, jnp.full(2, 1.0 / np.sqrt(2.0)), rtol=2e-3, atol=0.0)

    # The optional value-only drop solve must recover a much narrower width
    # without relying on the three-point curvature.
    model._peak_r_width_steps = 18

    def narrow_log_gaussian(type_key, r_pre, n_scan):
        del type_key, n_scan
        value = -20000.0 * (
            jnp.log(r_pre["r_ang"])
            - jnp.log(r_pre["target"][:, None])) ** 2
        return (value, jnp.zeros_like(value, dtype=int),
                jnp.zeros_like(value, dtype=bool))

    model._phi_partition_group_log_integral = narrow_log_gaussian
    got, scale = model._refine_r_center_group(
        "sys", jnp.arange(2), target * jnp.array([1.04, 0.96]),
        jnp.array([0.02, 0.03]), 0.1, 1.0, (), {})
    np.testing.assert_allclose(got, target, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(
        scale, jnp.full(2, 1.0 / 200.0), rtol=2e-3, atol=0.0)


def test_peak_radius_stencil_can_skip_systemic_group():
    model = object.__new__(MaserDiskModel)
    model.phi_integration = "peak-partition"
    model._peak_r_refine_steps = 4
    model._peak_r_refine_order = 7
    model._peak_r_refine_hv_only = True
    model._peak_r_width_steps = 0
    centre = jnp.array([0.2, 0.4])
    width = jnp.array([0.05, 0.06])

    got_centre, got_width = model._refine_r_center_group(
        "sys", jnp.array([0, 1]), centre, width, 0.1, 1.0,
        (), {})

    np.testing.assert_array_equal(got_centre, centre)
    np.testing.assert_array_equal(got_width, width)


def test_peak_partition_global_scan_evaluates_all_radii_together():
    class Model:
        phi_integration = "peak-partition"
        _phi_concat = {"sys": {}}
        _scan_width_drop = 0.0
        _scan_on_global_grid = MaserDiskModel._scan_on_global_grid

        def __init__(self):
            self.radial_shapes = []

        @staticmethod
        def _group_has_any_accel(type_key):
            del type_key
            return False

        def _r_precompute(self, r_ang, idx, *args, **kwargs):
            del idx, args, kwargs
            self.radial_shapes.append(r_ang.shape[-1])
            return {"r_ang": r_ang}

        @staticmethod
        def _phi_partition_scan_size(type_key):
            del type_key
            return 17

        @staticmethod
        def _phi_partition_group_log_integral(
                type_key, r_pre, n_scan):
            del type_key, n_scan
            ll = -r_pre["r_ang"] ** 2
            return ll, jnp.zeros_like(ll, dtype=int), jnp.zeros_like(
                ll, dtype=bool)

    model = Model()
    r_global = jnp.linspace(0.1, 1.0, 17)
    model._scan_on_global_grid(
        "sys", jnp.arange(2), r_global, (), {}, r_chunk=4)
    assert model.radial_shapes == [17]

    model_cached = Model()
    _, _, cache, _ = model_cached._scan_on_global_grid(
        "sys", jnp.arange(2), r_global.astype(jnp.float64), (), {},
        r_chunk=4, cache_scan=True)
    assert len(cache) == 2
    assert model_cached.radial_shapes == [17]


def test_peak_partition_cached_global_r_columns_are_exact():
    class Model:
        phi_integration = "peak-partition"
        _phi_concat = {"sys": {}}
        _marginal_per_spot_r = MaserDiskModel._marginal_per_spot_r

        @staticmethod
        def _r_precompute(r_ang, idx, *args, **kwargs):
            del idx, args, kwargs
            n_spot = r_ang.shape[0]
            return {"r_ang": r_ang,
                    "lnorm": jnp.zeros(n_spot),
                    "lnorm_a": jnp.zeros(n_spot)}

        @staticmethod
        def _phi_partition_scan_size(type_key):
            del type_key
            return 17

        @staticmethod
        def _phi_partition_group_log_integral(
                type_key, r_pre, n_scan):
            del type_key, n_scan
            ll = -r_pre["r_ang"] ** 2
            return ll, jnp.zeros_like(ll, dtype=int), jnp.zeros_like(
                ll, dtype=bool)

    model = Model()
    idx = jnp.arange(2)
    r_local = jnp.array([[0.2, 0.4, 0.8], [0.3, 0.5, 0.9]])
    r_global = jnp.array([0.1, 0.7])
    r_nodes = jnp.concatenate(
        [r_local, jnp.broadcast_to(r_global, (2, 2))], axis=-1)
    order = jnp.argsort(r_nodes, axis=-1)
    r_union = jnp.take_along_axis(r_nodes, order, axis=-1)
    log_w_r = jnp.log(jnp.array(
        [[0.1, 0.2, 0.3, 0.25, 0.15],
         [0.15, 0.2, 0.25, 0.3, 0.1]]))
    ll_global = jnp.broadcast_to(-r_global ** 2, (2, 2))
    no_overflow = jnp.zeros_like(ll_global, dtype=bool)
    cache = (r_local, ll_global, order, no_overflow)

    got = model._marginal_per_spot_r(
        "sys", idx, r_union, log_w_r, False, (), {}, None, cache)
    expected = model._marginal_per_spot_r(
        "sys", idx, r_union, log_w_r, False, (), {}, None, None)
    np.testing.assert_array_equal(got, expected)

    overflow = no_overflow.at[0, 0].set(True)
    recovered = model._marginal_per_spot_r(
        "sys", idx, r_union, log_w_r, False, (), {}, None,
        (r_local, ll_global, order, overflow))
    np.testing.assert_array_equal(recovered, expected)
