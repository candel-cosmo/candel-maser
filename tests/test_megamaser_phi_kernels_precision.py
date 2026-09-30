"""Precision regression for the phi-marginal kernels.

Verifies the optimised + f32-stabilised integrand kernels:
  (1) r-only coefficient hoisting in predict_position/velocity/accel,
  (2) reciprocal-folded neg_half_chi2_* helpers,
  (3) the circular quadratic-form integrand `_neg_half_chi2_quadform`,
  (4) the velocity-relative-to-v_sys_obs reformulation (Tier 2): velocities
      are computed and compared relative to v_sys_obs (≈7000 km/s) so no
      ~v_sys-magnitude number is ever subtracted, making the integrand
      float32-stable.

`predict_velocity_los` now returns V − v_sys_obs and the χ² uses the data
residual all_v − v_sys_obs (precomputed exactly on the host). The oracle is
the *true* −½χ² in float64; the production kernels are checked in float64
(deep agreement) and float32 (now accurate, vs the old absolute approach
which was cancellation-bound at the ~v_sys scale).

Run:  venv_candel/bin/python -m pytest \
    packages/candel-maser/tests/test_megamaser_phi_kernels_precision.py
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from candel_maser.maser_physics import (SPEED_OF_LIGHT,  # noqa: E402
                                        centripetal_acceleration,
                                        gravitational_redshift_factor,
                                        keplerian_speed, lorentz_factor,
                                        predict_acceleration_los,
                                        predict_position,
                                        predict_velocity_los)
from candel_maser.model_H0_maser import (  # noqa: E402
                                         _combine_cached_phi_marginal,
                                         _neg_half_chi2_quadform,
                                         neg_half_chi2_acceleration,
                                         neg_half_chi2_position,
                                         neg_half_chi2_velocity)

V_SYS_OBS = 7000.0   # known observation constant (km/s)
DV_SYS = 3.0         # small fitted systemic offset; v_sys = V_SYS_OBS + DV_SYS


def make_inputs(seed=0):
    """Realistic single-group (red HV) grid with on-disk spots."""
    rng = np.random.default_rng(seed)
    N, n_r, n_phi = 24, 80, 64

    D = 100.0
    M_BH = 4.0
    v_sys = V_SYS_OBS + DV_SYS
    x0, y0 = 3.0, -2.0
    i0 = np.deg2rad(82.0)
    Om0 = np.deg2rad(31.0)
    di_dr = np.deg2rad(5.0)
    dOm_dr = np.deg2rad(-3.0)

    r_true = rng.uniform(0.12, 0.45, size=N)
    phi_true = np.deg2rad(90.0 + rng.uniform(-25.0, 25.0, size=N))

    def warp(r):
        return i0 + di_dr * r, Om0 + dOm_dr * r

    i_t, Om_t = warp(r_true)
    R = r_true * 1e3
    X = x0 + R * (np.sin(phi_true) * np.sin(Om_t)
                  - np.cos(phi_true) * np.cos(Om_t) * np.cos(i_t))
    Y = y0 + R * (np.sin(phi_true) * np.cos(Om_t)
                  + np.cos(phi_true) * np.sin(Om_t) * np.cos(i_t))
    vk = 2978.8656 * np.sqrt(M_BH / (r_true * D))
    V = v_sys + np.sin(i_t) * vk * np.sin(phi_true)
    a_mag = 1.872e3 * M_BH / (r_true ** 2 * D ** 2)
    A = a_mag * np.cos(phi_true) * np.sin(i_t)

    sx, sy, sv, sa = 5.0, 5.0, 2.0, 0.05
    all_x = X + sx * rng.standard_normal(N)
    all_y = Y + sy * rng.standard_normal(N)
    all_v = V + sv * rng.standard_normal(N)
    all_a = A + sa * rng.standard_normal(N)
    has_a = (rng.random(N) < 0.5).astype(float)

    r_grid = np.geomspace(0.08, 0.6, n_r)
    r_ang = np.broadcast_to(r_grid[None, :], (N, n_r)).copy()
    i_r, Om_r = warp(r_ang)
    phi = np.deg2rad(np.linspace(65.0, 115.0, n_phi))

    return dict(
        r_ang=r_ang, sin_i=np.sin(i_r), cos_i=np.cos(i_r),
        sin_O=np.sin(Om_r), cos_O=np.cos(Om_r),
        x0=x0, y0=y0, D=D, M_BH=M_BH, v_sys=v_sys, dv_sys=DV_SYS,
        all_x=all_x, all_y=all_y, all_v=all_v, all_a=all_a,
        # velocity data relative to v_sys_obs, computed exactly (host f64):
        all_v_rel=all_v - V_SYS_OBS,
        var_x=np.full(N, sx ** 2), var_y=np.full(N, sy ** 2),
        var_v=np.full(N, sv ** 2), var_a=np.full(N, sa ** 2), has_a=has_a,
        sin_phi=np.sin(phi), cos_phi=np.cos(phi))


def oracle_nhc(d):
    """True −½χ² on the (N, n_r, n_phi) grid, float64 (absolute residual)."""
    f = lambda a: jnp.asarray(a, dtype=jnp.float64)  # noqa: E731
    r = f(d["r_ang"])[:, :, None]
    s, c = f(d["sin_phi"]), f(d["cos_phi"])
    sin_i = f(d["sin_i"])[:, :, None]
    cos_i = f(d["cos_i"])[:, :, None]
    sin_O = f(d["sin_O"])[:, :, None]
    cos_O = f(d["cos_O"])[:, :, None]
    R = r * 1e3
    X = d["x0"] + R * (s * sin_O - c * cos_O * cos_i)
    Y = d["y0"] + R * (s * cos_O + c * sin_O * cos_i)
    vk = keplerian_speed(r, f(d["D"]), f(d["M_BH"]))
    zg = gravitational_redshift_factor(r, f(d["D"]), f(d["M_BH"]))
    beta2 = (vk / SPEED_OF_LIGHT) ** 2
    vz = sin_i * vk * s
    V = SPEED_OF_LIGHT * (
        lorentz_factor(beta2) * (1.0 + vz / SPEED_OF_LIGHT)
        * zg * (1.0 + f(d["v_sys"]) / SPEED_OF_LIGHT) - 1.0)
    A = centripetal_acceleration(r, f(d["D"]), f(d["M_BH"])) * c * sin_i
    dp = (slice(None), None, None)
    nhc = -0.5 * ((f(d["all_x"])[dp] - X) ** 2 / f(d["var_x"])[dp]
                  + (f(d["all_y"])[dp] - Y) ** 2 / f(d["var_y"])[dp])
    nhc = nhc - 0.5 * (f(d["all_v"])[dp] - V) ** 2 / f(d["var_v"])[dp]
    nhc = nhc - 0.5 * (f(d["all_a"])[dp] - A) ** 2 / f(d["var_a"])[dp] \
        * f(d["has_a"])[dp]
    return nhc


def old_absolute_nhc(d, dt):
    """Pre-Tier-2 path: absolute velocity + all_v − V at the ~v_sys scale."""
    f = lambda a: jnp.asarray(a, dtype=dt)  # noqa: E731
    r = f(d["r_ang"])[:, :, None]
    s, c = f(d["sin_phi"]), f(d["cos_phi"])
    sin_i = f(d["sin_i"])[:, :, None]
    cos_i = f(d["cos_i"])[:, :, None]
    sin_O = f(d["sin_O"])[:, :, None]
    cos_O = f(d["cos_O"])[:, :, None]
    R = r * 1e3
    X = d["x0"] + R * (s * sin_O - c * cos_O * cos_i)
    Y = d["y0"] + R * (s * cos_O + c * sin_O * cos_i)
    vk = keplerian_speed(r, f(d["D"]), f(d["M_BH"]))
    zg = gravitational_redshift_factor(r, f(d["D"]), f(d["M_BH"]))
    beta2 = (vk / SPEED_OF_LIGHT) ** 2
    vz = sin_i * vk * s
    V = SPEED_OF_LIGHT * (
        lorentz_factor(beta2) * (1.0 + vz / SPEED_OF_LIGHT)
        * zg * (1.0 + f(d["v_sys"]) / SPEED_OF_LIGHT) - 1.0)
    A = centripetal_acceleration(r, f(d["D"]), f(d["M_BH"])) * c * sin_i
    dp = (slice(None), None, None)
    nhc = -0.5 * ((f(d["all_x"])[dp] - X) ** 2 / f(d["var_x"])[dp]
                  + (f(d["all_y"])[dp] - Y) ** 2 / f(d["var_y"])[dp])
    nhc = nhc - 0.5 * (f(d["all_v"])[dp] - V) ** 2 / f(d["var_v"])[dp]
    nhc = nhc - 0.5 * (f(d["all_a"])[dp] - A) ** 2 / f(d["var_a"])[dp] \
        * f(d["has_a"])[dp]
    return nhc


def per_channel_nhc(d, dt):
    """−½χ² via predict_* + neg_half_chi2_* (items 1, 2; velocity relative)."""
    f = lambda a: jnp.asarray(a, dtype=dt)  # noqa: E731
    r = f(d["r_ang"])[:, :, None]
    s, c = f(d["sin_phi"]), f(d["cos_phi"])
    si = f(d["sin_i"])[:, :, None]
    ci = f(d["cos_i"])[:, :, None]
    so = f(d["sin_O"])[:, :, None]
    co = f(d["cos_O"])[:, :, None]
    X, Y = predict_position(r, s, c, f(d["x0"]), f(d["y0"]), si, ci, so, co)
    V = predict_velocity_los(r, s, c, f(d["D"]), f(d["M_BH"]),
                             f(d["v_sys"]), f(d["dv_sys"]), si)
    A = predict_acceleration_los(r, s, c, f(d["D"]), f(d["M_BH"]), si)
    dp = (slice(None), None, None)
    nhc = neg_half_chi2_position(
        f(d["all_x"])[dp], f(d["all_y"])[dp], X, Y,
        f(d["var_x"])[dp], f(d["var_y"])[dp])
    nhc = nhc + neg_half_chi2_velocity(
        f(d["all_v_rel"])[dp], V, f(d["var_v"])[dp])
    nhc = nhc + neg_half_chi2_acceleration(
        f(d["all_a"])[dp], A, f(d["var_a"])[dp], f(d["has_a"])[dp])
    return nhc


def quadform_nhc(d, dt, shared_r=False):
    """−½χ² via the quadratic-form kernel (item 3, velocity relative)."""
    f = lambda a: jnp.asarray(a, dtype=dt)  # noqa: E731
    s, c = f(d["sin_phi"]), f(d["cos_phi"])
    row = 0 if shared_r else slice(None)
    r_pre = dict(
        r_ang=f(d["r_ang"])[row], sin_i=f(d["sin_i"])[row],
        cos_i=f(d["cos_i"])[row], sin_O=f(d["sin_O"])[row],
        cos_O=f(d["cos_O"])[row],
        x0=f(d["x0"]), y0=f(d["y0"]), D=f(d["D"]), M_BH=f(d["M_BH"]),
        v_sys=f(d["v_sys"]), dv_sys=f(d["dv_sys"]),
        all_x=f(d["all_x"]), all_y=f(d["all_y"]),
        all_v_rel=f(d["all_v_rel"]), all_a=f(d["all_a"]),
        var_x=f(d["var_x"]), var_y=f(d["var_y"]),
        var_v=f(d["var_v"]), var_a=f(d["var_a"]), has_a=f(d["has_a"]),
        weight_x=-0.5 / f(d["var_x"]),
        weight_y=-0.5 / f(d["var_y"]),
        weight_v=-0.5 / f(d["var_v"]),
        weight_a=-0.5 * f(d["has_a"]) / f(d["var_a"]),
        has_any_accel=True)
    return _neg_half_chi2_quadform(
        r_pre, s, c, s * s, c * c, s * c, shared_r=shared_r)


def run(dt):
    name = "float64" if dt == jnp.float64 else "float32"
    print(f"\n[{name}]")
    d = make_inputs()
    a = lambda x: np.asarray(x, np.float64)  # noqa: E731
    ref = a(oracle_nhc(d))
    e_old = float(np.max(np.abs(a(old_absolute_nhc(d, dt)) - ref)))
    e_pc = float(np.max(np.abs(a(per_channel_nhc(d, dt)) - ref)))
    e_qf = float(np.max(np.abs(a(quadform_nhc(d, dt)) - ref)))
    e_qfs = float(np.max(np.abs(a(quadform_nhc(d, dt, True)) - ref)))
    print(f"  max|Δ(-½χ²)| vs true  old-absolute={e_old:.3e}  "
          f"per-channel={e_pc:.3e}  quad-form={e_qf:.3e}  "
          f"shared-quad-form={e_qfs:.3e}")
    return dict(name=name, e_old=e_old, e_pc=e_pc, e_qf=e_qf,
                e_qfs=e_qfs)


def main():
    res = {}
    if jax.config.jax_enable_x64:
        res["f64"] = run(jnp.float64)
    res["f32"] = run(jnp.float32)

    print("\n=== assertions ===")
    ok = True
    if "f64" in res:
        r = res["f64"]
        c1 = (r["e_pc"] < 1e-6 and r["e_qf"] < 1e-6
              and r["e_qfs"] < 1e-6)
        ok &= c1
        print(f"f64 per-channel & quad-form vs true < 1e-6: {c1} "
              f"({r['e_pc']:.2e}, {r['e_qf']:.2e})")
    r = res["f32"]
    # Tier 2: velocity-relative formulation makes f32 accurate. Both paths
    # must be far better than the old absolute approach and below ~0.5 in
    # −½χ² even at the worst (off-fit, |Δv|~1700) gridpoint.
    c2 = r["e_pc"] < 0.5 and r["e_qf"] < 0.5
    c3 = r["e_pc"] < r["e_old"] / 20.0 and r["e_qf"] < r["e_old"] / 20.0
    ok &= c2 and c3
    print(f"f32 per-channel & quad-form vs true < 0.5: {c2} "
          f"({r['e_pc']:.2e}, {r['e_qf']:.2e})")
    print(f"f32 >=20x better than old absolute: {c3} "
          f"(old={r['e_old']:.2e})")

    print("\nPASS" if ok else "\nFAIL")
    assert ok


def test_phi_kernels_precision():
    main()


def test_cached_phi_marginal_matches_flat_reduction():
    rng = np.random.default_rng(42)
    n_spot, n_local, n_global, n_phi = 4, 7, 5, 11
    nhc_local = rng.normal(size=(n_spot, n_local, n_phi))
    nhc_global = rng.normal(size=(n_spot, n_global, n_phi))
    order = np.argsort(
        rng.uniform(size=(n_spot, n_local + n_global)), axis=-1)
    log_w_r = rng.normal(size=(n_spot, n_local + n_global))
    log_w_phi = rng.normal(size=n_phi)

    for dtype in (jnp.float64, jnp.float32):
        local = jnp.asarray(nhc_local, dtype=dtype)
        global_ = jnp.asarray(nhc_global, dtype=dtype)
        wr = jnp.asarray(log_w_r, dtype=dtype)
        wp = jnp.asarray(log_w_phi, dtype=dtype)
        ll_global = jax.scipy.special.logsumexp(global_ + wp, axis=-1)
        got = _combine_cached_phi_marginal(
            local, ll_global, jnp.asarray(order), wr, wp)

        nhc = jnp.concatenate([local, global_], axis=-2)
        nhc = jnp.take_along_axis(
            nhc, jnp.asarray(order)[..., None], axis=-2)
        expected = jax.scipy.special.logsumexp(
            nhc + wr[..., None] + wp, axis=(-2, -1))
        np.testing.assert_array_max_ulp(got, expected, maxulp=2)


if __name__ == "__main__":
    main()
