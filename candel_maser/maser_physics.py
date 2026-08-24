# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Deterministic megamaser disk physics (Pesce et al. 2020).

Pure JAX functions and unit-conversion constants only: no NumPyro, no
priors, no quadrature, no config parsing, no random numbers. Extracted
from model_H0_maser.py.
"""
import jax
import jax.numpy as jnp

from ..util import SPEED_OF_LIGHT

# -----------------------------------------------------------------------
# Disk physics constants
# -----------------------------------------------------------------------

# Internal units: M_BH in 1e7 M_sun, sky positions in μas,
# r_ang in mas, D in Mpc.  These retain the rounded constants used by the
# production model and its validated chains. They are not CODATA-precision
# values.
C_v = 2978.8656    # km/s: sqrt(G * 1e7 M_sun / (1 mas * 1 Mpc))
C_a = 1.872e3      # km/s/yr: 1e7 M_sun * G * yr / (1 mas * 1 Mpc)^2
C_g = 1.974e-4     # dimensionless: 2*G * 1e7 M_sun / (c^2 * 1 mas * 1 Mpc)
# When True, the eccentric SR gamma uses the circular speed Vcirc (Reid
# fit_disk convention) instead of the true orbital speed. Read at trace time;
# used by the standalone Reid-validation scripts. Affects only the eccentric
# branch.
REID_CIRCULAR_GAMMA = False
LOG_2PI = 1.8378770664093453  # jnp.log(2 * pi), precomputed

# Matching rounded production conversion: 1 mas at 1 Mpc = 4.848e-3 pc.
PC_PER_MAS_MPC = 4.848e-3

# Floor applied to trapezoidal step widths before taking log, so that a
# zero-width interval at a grid edge yields a very-negative (finite)
# weight rather than -inf.
W_LOG_FLOOR = 1e-30

# Denominator regulariser for the data-driven r-estimate helpers. Added
# to |Δv| and |a| so spots with anomalously small measurement values
# produce a finite (large but clipped) r estimate rather than inf/NaN.
R_EST_EPS = 1e-30


# -----------------------------------------------------------------------
# Disk physics functions
# -----------------------------------------------------------------------


def keplerian_speed(r_ang, D, M_BH):
    """Keplerian orbital speed v_kep = sqrt(G M / r), in km/s.

    ``r_ang`` in mas, ``D`` in Mpc, ``M_BH`` in units of 1e7 M_sun.
    """
    return C_v * jnp.sqrt(M_BH / (r_ang * D))


def lorentz_factor(beta_sq):
    """γ = 1 / sqrt(1 - β²), clipped to guard against β → 1.

    Takes β² (not β) so callers can pass the full eccentric β² directly
    without branching.
    """
    return 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta_sq, 1e-6))


def gravitational_redshift_factor(r_ang, D, M_BH):
    """Schwarzschild (1 + z_g) = 1 / sqrt(1 - 2GM / (r c²)).

    Clipped for r approaching the Schwarzschild radius.
    """
    return 1.0 / jnp.sqrt(
        jnp.maximum(1.0 - C_g * M_BH / (r_ang * D), 1e-6))


def centripetal_acceleration(r_ang, D, M_BH):
    """Circular centripetal acceleration |a| = G M / r², in km/s/yr."""
    return C_a * M_BH / (r_ang ** 2 * D ** 2)


def gamma_minus_one(beta_sq):
    """γ − 1 = 1/√(1−β²) − 1, computed cancellation-free.

    Rationalising, γ−1 = β²/(q(1+q)) with q = √(1−β²): a sum/quotient of
    positive terms, so no float catastrophic cancellation as β → 0 (unlike
    forming ``1/√(1−β²) − 1`` directly).
    """
    q = jnp.sqrt(jnp.maximum(1.0 - beta_sq, 1e-6))
    return beta_sq / (q * (1.0 + q))


def gravitational_redshift_minus1(r_ang, D, M_BH):
    """z_g = 1/√(1−x) − 1 with x = 2GM/(rc²), computed cancellation-free."""
    x = C_g * M_BH / (r_ang * D)
    q = jnp.sqrt(jnp.maximum(1.0 - x, 1e-6))
    return x / (q * (1.0 + q))


def velocity_rel_affine(r_ang, D, M_BH, v_sys, dv_sys, sin_i):
    """Circular V − v_sys_obs as affine coefficients (V0, B) in sinφ.

    The systemic velocity composes multiplicatively in redshift, so
        V = c·z_og·(1+z0) + v_sys,   z0 = v_sys/c,
        z_og = z_D + z_g + z_D·z_g   (orbital+gravitational redshift),
    hence  V − v_sys_obs = c·z_og·(1+z0) + dv_sys  with dv_sys = v_sys −
    v_sys_obs. Computing z_og from the cancellation-free z_D, z_g (sums of
    small terms, never ``prod − 1``) keeps the result float32-stable — no
    ~v_sys-magnitude number ever appears in the residual. V is affine in
    sinφ; the returned (V0, B) give V_rel = V0 + B·sinφ.
    """
    v_kep = keplerian_speed(r_ang, D, M_BH)
    z0 = v_sys / SPEED_OF_LIGHT
    zg = gravitational_redshift_minus1(r_ang, D, M_BH)
    gm1 = gamma_minus_one((v_kep / SPEED_OF_LIGHT) ** 2)
    g = 1.0 + gm1
    k = sin_i * v_kep / SPEED_OF_LIGHT          # v_z/c = k·sinφ
    z_og0 = gm1 + zg + gm1 * zg                  # φ-independent part of z_og
    z_og1 = g * (1.0 + zg) * k                   # sinφ coefficient of z_og
    fac = SPEED_OF_LIGHT * (1.0 + z0)
    return fac * z_og0 + dv_sys, fac * z_og1


def radius_from_los_velocity(v_los, sin_i, D, M_BH):
    """Solve the Keplerian LOS velocity relation for r_ang at phi = ±π/2.

    ``|v_LOS - v_sys| ≈ v_kep · sin(i)`` at a high-velocity spot →
    r_ang = M · (C_v · sin_i)² / (D · v_LOS²) (circular, no relativistic
    corrections — suitable for grid centring / initialisation only).
    """
    return M_BH * (C_v * sin_i) ** 2 / (D * v_los ** 2)


def radius_from_los_acceleration(a_los, sin_i, D, M_BH):
    """Solve the centripetal LOS acceleration relation for r_ang at phi≈0.

    ``|A_LOS| ≈ a_mag · sin(i)`` at a systemic spot →
    r_ang = √(C_a · M · sin_i / (D² · |A_LOS|)) (same circular-orbit
    approximation as `radius_from_los_velocity`).
    """
    return jnp.sqrt(C_a * M_BH * sin_i / (D ** 2 * a_los))


def warp_geometry(r_ang, r_ang_ref_i, r_ang_ref_Omega,
                  i0_rad, di_dr_rad,
                  Omega0_rad, dOmega_dr_rad,
                  d2i_dr2_rad=0.0, d2Omega_dr2_rad=0.0):
    """Evaluate warped inclination and position angle at angular radius.

    Each warp has its own pivot radius: i is expanded about
    ``r_ang_ref_i`` and Omega about ``r_ang_ref_Omega`` (both in mas).
    The warp rates di/dr and dOmega/dr are in radians per mas; the
    optional quadratic terms are in radians per mas^2.
    """
    dr_i = r_ang - r_ang_ref_i
    dr_O = r_ang - r_ang_ref_Omega
    i = i0_rad + di_dr_rad * dr_i + d2i_dr2_rad * (dr_i * dr_i)
    Omega = (Omega0_rad + dOmega_dr_rad * dr_O
             + d2Omega_dr2_rad * (dr_O * dr_O))
    return i, Omega


def predict_position(r_ang, sin_phi, cos_phi, x0, y0,
                     sin_i, cos_i, sin_O, cos_O):
    """Predict sky-plane position (X, Y) of a disk point in μas.

    Reid+2019 phi convention: phi = +π/2 at the red HV locus, phi = -π/2
    at the blue HV locus, phi = 0 and phi = π at systemic. ``sin_O, cos_O``
    are sin/cos of the sky position angle; ``sin_i, cos_i`` sin/cos of
    the inclination — precomputed by the caller so the same trig values
    are shared across the position / velocity / acceleration channels.
    All inputs broadcast element-wise.
    """
    R = r_ang * 1e3  # mas → μas for position projection
    # Fold every r-only factor into (…,1) coefficients so each φ-broadcast
    # multiply hits the (…, n_phi) tensor exactly once (fewer big ops; the
    # reassociation only perturbs the result at the rounding level).
    R_sinO = R * sin_O
    R_cosO = R * cos_O
    R_cosO_ci = R_cosO * cos_i
    R_sinO_ci = R_sinO * cos_i
    X = x0 + R_sinO * sin_phi - R_cosO_ci * cos_phi
    Y = y0 + R_cosO * sin_phi + R_sinO_ci * cos_phi
    return X, Y


def predict_velocity_los(r_ang, sin_phi, cos_phi, D, M_BH, v_sys, dv_sys,
                         sin_i, ecc2=0.0, ecc_cos_om=0.0, ecc_sin_om=0.0):
    """Predict LOS velocity RELATIVE to v_sys_obs (``V − v_sys_obs``), km/s.

    Combines Keplerian (possibly eccentric) orbital motion, special-
    relativistic Doppler, and Schwarzschild gravitational redshift,
    composed with the systemic recession via
        (1 + z_obs) = (1 + z_D)(1 + z_grav)(1 + v_sys / c).
    Rather than the absolute velocity (~v_sys ≈ thousands of km/s), it
    returns ``V − v_sys_obs = c·z_og·(1+z0) + dv_sys`` (``v_sys =
    v_sys_obs + dv_sys``, ``z0 = v_sys/c``, ``z_og = z_D + z_g + z_D·z_g``).
    The orbital+gravitational redshift z_og is built from cancellation-free
    z_D, z_g (sums of small terms, never ``prod − 1``), so no ~v_sys-scale
    number enters and the velocity residual stays accurate in float32.
    Callers must compare against the data residual ``all_v − v_sys_obs``.

    Eccentricity enters only through the smooth Cartesian combinations
    ``ecc_cos_om = e·cos ω``, ``ecc_sin_om = e·sin ω`` (ω the argument of
    periapsis, Reid convention) and ``ecc2 = e²``.  Passing these products
    rather than ``(e, ω)`` keeps the velocity differentiable at e=0, where the
    polar direction ω is undefined.  The circular branch (literal
    ``ecc2 == 0.0``) is affine in sinφ and shares its coefficients with
    ``velocity_rel_affine``.
    """
    z0 = v_sys / SPEED_OF_LIGHT
    zg = gravitational_redshift_minus1(r_ang, D, M_BH)

    if isinstance(ecc2, (int, float)) and ecc2 == 0.0:
        V0, B = velocity_rel_affine(r_ang, D, M_BH, v_sys, dv_sys, sin_i)
        return V0 + B * sin_phi

    v_kep = keplerian_speed(r_ang, D, M_BH)
    beta_c2 = (v_kep / SPEED_OF_LIGHT) ** 2
    # phi - omega via angle-subtraction; only cos_d is needed because the
    # tangential/radial decomposition collapses algebraically:
    #   v_t·sin_phi - v_r·cos_phi = v_kep · (sin_phi + ecc·sin_om) / E.
    # Keep eccentricity in Cartesian form (e·cos ω, e·sin ω) so e·cos_d is a
    # smooth polynomial in (e_x, e_y) -- no e·(direction) split at the origin.
    ecc_cos_d = cos_phi * ecc_cos_om + sin_phi * ecc_sin_om
    # ecc→1 at anti-periapsis sends denom→0; clip so residuals stay finite.
    denom = jnp.maximum(1.0 + ecc_cos_d, 1e-6)
    inv_sqrt_denom = jax.lax.rsqrt(denom)
    inv_denom = inv_sqrt_denom * inv_sqrt_denom
    v_z = sin_i * v_kep * (sin_phi + ecc_sin_om) * inv_sqrt_denom

    # Reid puts the circular speed in the SR gamma; the physical choice is the
    # true orbital speed v_orb² = Vcirc²·(1+e²+2e·cos_d)/(1+e·cos_d).
    beta_g2 = beta_c2 if REID_CIRCULAR_GAMMA else \
        beta_c2 * (1.0 + ecc2 + 2.0 * ecc_cos_d) * inv_denom
    # z_D = γ(1 + v_z/c) − 1 = (γ−1) + γ·v_z/c, cancellation-free.
    gm1 = gamma_minus_one(beta_g2)
    z_D = gm1 + (1.0 + gm1) * (v_z / SPEED_OF_LIGHT)
    z_og = z_D + zg + z_D * zg
    return SPEED_OF_LIGHT * (1.0 + z0) * z_og + dv_sys


def predict_acceleration_los(r_ang, sin_phi, cos_phi, D, M_BH, sin_i):
    """Predict LOS centripetal acceleration in km/s/yr.

    Projects |a| = G M / r² onto the line of sight:
    ``A = |a| · cos(phi) · sin(i)``. ``sin_phi`` is accepted for
    interface uniformity with the other ``predict_*`` functions but is
    unused here.
    """
    del sin_phi
    coef = centripetal_acceleration(r_ang, D, M_BH) * sin_i  # r-only
    return coef * cos_phi
