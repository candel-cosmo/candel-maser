# Copyright (C) 2025 Richard Stiskalek
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
"""Megamaser disk forward model for the BlackJAX megamaser samplers.

Implements the warped Keplerian disk model from Pesce et al. (2020),
arXiv:2001.04581. All JAX functions are JIT-compilable and auto-differentiable.

Phi integration:
  - Red HV (φ=+π/2 peak): 3 uniform sub-ranges around +π/2
        (N_low, N_high, N_low points). Trapezoidal in each.
  - Blue HV (φ=-π/2 peak): mirror of red, centred on -π/2.
  - Systemic (φ=0 and φ=π): configurable list of uniform sub-ranges
        (default [[-45°, 45°], [135°, 225°]], N_sys points each).
  - Sub-ranges within a type are disjoint → combined in log-space via
    logsumexp over sub-integrals.
  Knobs in [model] config: phi_hv_inner_deg, phi_hv_outer_deg,
    n_phi_hv_high, n_phi_hv_low, phi_sys_ranges_deg, n_phi_sys.

The per-spot angular radii ``r_ang`` are marginalised by the DE objective and
sampled explicitly (with the per-spot ``phi``) by the NUTS chain.

Phi convention (Reid+2019): phi=+pi/2 at the redshifted HV locus,
phi=-pi/2 at the blueshifted HV locus, phi=0 and phi=pi at systemic
(front and back of disk along LOS). Argument of periapsis omega is in
the same convention. LOS velocity v_z ∝ sin(phi)·sin(i); LOS
acceleration A ∝ cos(phi)·sin(i).

All operations fully batched over spots — no vmap or lax.scan.
All angles in RADIANS inside physics functions.
"""
import jax
import jax.numpy as jnp
import numpy as _np
from jax.scipy.special import logsumexp
from numpyro.distributions import Delta, Uniform

from ..util import fprint, fsection, get_nested
from .base_model import ModelBase
from .integration import trapz_log_weights
from .maser_physics import (LOG_2PI, PC_PER_MAS_MPC, R_EST_EPS, W_LOG_FLOOR,
                            centripetal_acceleration, predict_acceleration_los,
                            predict_position, predict_velocity_los,
                            radius_from_los_acceleration,
                            radius_from_los_velocity, velocity_rel_affine,
                            warp_geometry)
from .optim1d import brent_1d
from .utils import load_priors


def _compile_friendly_logsumexp(x, axis=-1):
    """Keep the phi integrand out of XLA's slow input-reduction fusion."""
    return logsumexp(jax.lax.optimization_barrier(x), axis=axis)


# -----------------------------------------------------------------------
# Gaussian -1/2 chi^2 helpers
# -----------------------------------------------------------------------


def neg_half_chi2_position(x_obs, y_obs, X_pred, Y_pred, var_x, var_y):
    """Per-gridpoint −½χ² from the (X, Y) position channel.

    Returns only the residual term ``-½(dx²/var_x + dy²/var_y)``. The
    Gaussian normalisation ``-½·log(2π·var)`` is added once per spot by
    the caller after the (r, φ) integration — this keeps the values
    entering logsumexp bounded in χ² space so max-subtraction retains
    float32 precision.
    """
    dx = x_obs - X_pred
    dy = y_obs - Y_pred
    # Fold -0.5/var into per-spot coefficients: turns two big-tensor
    # divisions into multiplies and drops the trailing -0.5 scale.
    return dx * dx * (-0.5 / var_x) + dy * dy * (-0.5 / var_y)


def neg_half_chi2_velocity(v_obs, V_pred, var_v):
    """Per-gridpoint −½χ² from the LOS velocity channel (no norm)."""
    dv = v_obs - V_pred
    return dv * dv * (-0.5 / var_v)


def neg_half_chi2_acceleration(a_obs, A_pred, var_a, has_a):
    """Per-gridpoint −½χ² from the LOS acceleration channel (no norm).

    ``has_a`` multiplicatively zeroes the contribution for spots without
    an accel measurement; the caller must still supply a finite
    ``var_a`` (the factor still multiplies 0 when has_a=0).
    """
    da = a_obs - A_pred
    return da * da * (-0.5 * has_a / var_a)


def _neg_half_chi2_quadform(r_pre, sin_phi, cos_phi, sin2, cos2, sincos,
                            shared_r=False):
    """−½χ² on the (r, φ) grid via a quadratic form in (sinφ, cosφ).

    Circular-orbit only. Every residual channel is low-order in φ with
    coefficients that depend only on r:

        X = x0 + Px_s·sinφ + Px_c·cosφ,   Y = y0 + Py_s·sinφ + Py_c·cosφ
        V_rel = V0 + Bv·sinφ               (velocity relative to v_sys_obs)
        A = Ba·cosφ

    so −½χ² collapses to ``C0 + Cs·s + Cc·c + Css·s² + Ccc·c² + Csc·s·c``
    with six r-only coefficients evaluated against the precomputed φ-basis
    ``(sinφ, cosφ, sin²φ, cos²φ, sinφ·cosφ)``. This is algebraically exact
    but expands the constant C0 ≈ −½Σ(data/σ)², so the basis terms cancel
    to recover the small residual. The velocity uses the relative
    formulation (``velocity_rel_affine`` + ``all_v_rel``) so C0 stays at the
    position/orbital scale rather than the ~v_sys² scale; the residual
    cancellation is still milder in float64, so the caller (``_phi_eval``)
    gates it to x64.

    Mirrors ``_phi_eval``'s broadcasting: ``r_pre`` r-only fields share
    ``r_ang``'s shape; per-spot data/variances carry a single leading axis.
    Returns the residual term only — lnorm/lnorm_a are added by the caller
    after the logsumexp.
    """
    r_ang = r_pre["r_ang"]
    if shared_r:
        r_ang = r_ang[None, :]
        dp = (slice(None), None)
        sin_i = r_pre["sin_i"][None, :]
        cos_i = r_pre["cos_i"][None, :]
        sin_O = r_pre["sin_O"][None, :]
        cos_O = r_pre["cos_O"][None, :]
    else:
        dp = (slice(None),) + (None,) * (r_ang.ndim - 1)
        sin_i = r_pre["sin_i"]
        cos_i = r_pre["cos_i"]
        sin_O = r_pre["sin_O"]
        cos_O = r_pre["cos_O"]
    D = r_pre["D"]
    M_BH = r_pre["M_BH"]

    R = r_ang * 1e3
    R_sinO = R * sin_O
    R_cosO = R * cos_O
    px_s = R_sinO
    px_c = -R_cosO * cos_i
    py_s = R_cosO
    py_c = R_sinO * cos_i

    # Velocity relative to v_sys_obs: V_rel = v0 + bv·sinφ (f32-stable).
    v0, bv = velocity_rel_affine(
        r_ang, D, M_BH, r_pre["v_sys"], r_pre["dv_sys"], sin_i)

    kx = (-0.5 / r_pre["var_x"])[dp]
    ky = (-0.5 / r_pre["var_y"])[dp]
    kv = (-0.5 / r_pre["var_v"])[dp]
    ex = r_pre["all_x"][dp] - r_pre["x0"]
    ey = r_pre["all_y"][dp] - r_pre["y0"]
    ev = r_pre["all_v_rel"][dp] - v0

    C0 = kx * ex * ex + ky * ey * ey + kv * ev * ev
    Cs = -2.0 * (kx * ex * px_s + ky * ey * py_s + kv * ev * bv)
    Cc = -2.0 * (kx * ex * px_c + ky * ey * py_c)
    Css = kx * px_s * px_s + ky * py_s * py_s + kv * bv * bv
    Ccc = kx * px_c * px_c + ky * py_c * py_c
    Csc = 2.0 * (kx * px_s * px_c + ky * py_s * py_c)

    if r_pre["has_any_accel"]:
        ba = centripetal_acceleration(r_ang, D, M_BH) * sin_i
        ka = (-0.5 * r_pre["has_a"] / r_pre["var_a"])[dp]
        ea = r_pre["all_a"][dp]
        C0 = C0 + ka * ea * ea
        Cc = Cc - 2.0 * (ka * ea * ba)
        Ccc = Ccc + ka * ba * ba

    return (C0[..., None]
            + Cs[..., None] * sin_phi + Cc[..., None] * cos_phi
            + Css[..., None] * sin2 + Ccc[..., None] * cos2
            + Csc[..., None] * sincos)


def _combine_cached_phi_marginal(nhc_local, ll_global, order,
                                 log_w_r, log_w_phi):
    """Combine local columns with cached global phi-marginals.

    This is algebraically the original joint logsumexp, associated as phi
    then radius so the global-radius scan can be reused.
    """
    ll_local = _compile_friendly_logsumexp(nhc_local + log_w_phi)
    ll_nodes = jnp.concatenate([ll_local, ll_global], axis=-1)
    ll_sorted = jnp.take_along_axis(ll_nodes, order, axis=-1)
    return logsumexp(ll_sorted + log_w_r, axis=-1)


def _conditional_global_r_window(valid, r_cf, r_min, r_max):
    """Global conditional-r scan window from valid closed-form seeds."""
    r_for_min = jnp.where(valid, r_cf, jnp.inf)
    r_for_max = jnp.where(valid, r_cf, 0.0)
    return (
        jnp.maximum(r_min, jnp.min(r_for_min) * 0.25),
        jnp.minimum(r_max, jnp.max(r_for_max) * 4.0),
    )


class MaserDiskModel(ModelBase):
    """Megamaser disk model with explicit and marginalised latent helpers."""

    def __init__(self, config_path, data):
        super().__init__(config_path)
        fsection("Maser Disk Model")
        self._load_and_set_priors()
        self._resolve_per_galaxy_priors(data)

        self.n_spots = data["n_spots"]
        self.is_highvel = jnp.asarray(data["is_highvel"])

        if "sigma_v" not in data:
            sv_default = float(get_nested(
                self.config, "model/sigma_v_default", 0.25))
            data["sigma_v"] = sv_default * _np.ones(data["n_spots"])
            fprint(f"sigma_v not in data, defaulting to {sv_default} km/s.")

        self._set_data_arrays(
            data, skip_keys=("accel_measured", "is_highvel", "is_systemic",
                             "is_blue", "is_red", "spot_type",
                             "phi_lo", "phi_hi",
                             "n_spots", "galaxy_name", "v_sys_obs"))

        if "v_sys_obs" not in data:
            raise ValueError(
                "data must contain 'v_sys_obs' (CMB-frame recession "
                "velocity in km/s).")
        self.v_sys_obs = float(data["v_sys_obs"])

        accel_meas = self._build_spot_indices(data)
        self._build_allspot_arrays(data, accel_meas)
        gal_cfg = self._configure_galaxy(data)
        self._build_phi_subranges(gal_cfg)
        self._build_r_config(data, gal_cfg)
        self._precompute_sinh_template(data)
        self._print_summary()

    # ---- priors / per-galaxy ----

    def _resolve_per_galaxy_priors(self, data):
        """Set per-galaxy D prior and optional prior overrides."""
        gname = data.get("galaxy_name")
        gal_cfg = get_nested(self.config, f"model/galaxies/{gname}", {})
        gal_priors = gal_cfg.get("priors", {})
        if gal_priors:
            priors, prior_dist_name = load_priors(gal_priors)
            self.priors.update(priors)
            self.prior_dist_name.update(prior_dist_name)
            fprint(f"prior overrides for {gname}: "
                   f"{', '.join(sorted(gal_priors))}")

        # Megamaser distance is ALWAYS sampled as uniform D_A. The legacy
        # uniform-D_c sampling path was removed; D_c_prior is no longer a knob.
        self._D_A_uniform = True
        if "D_lo" in data and "D_hi" in data:
            D_prior_type = get_nested(
                self.config, "model/D_c_prior", "uniform_D_A")
            if D_prior_type != "uniform_D_A":
                raise ValueError(
                    "Megamaser distance is always sampled as uniform D_A; "
                    f"model/D_c_prior={D_prior_type!r} is no longer supported "
                    "(remove the key or set it to 'uniform_D_A').")
            lo, hi = float(data["D_lo"]), float(data["D_hi"])
            # Config D_lo/D_hi are comoving bounds; convert to D_A bounds
            # at the fiducial cosmology (H0_ref, Om) so the uniform-D_A
            # prior brackets the same physical distance range.
            h_ref = float(get_nested(
                self.config, "model/H0_ref", 73.0)) / 100.0
            z_bounds = self.distance2redshift(jnp.asarray([lo, hi]), h=h_ref)
            lo_DA = float(lo / (1.0 + z_bounds[0]))
            hi_DA = float(hi / (1.0 + z_bounds[1]))
            self.priors["D"] = Uniform(lo_DA, hi_DA)
            fprint(f"D prior: D_A ~ Uniform({lo_DA:.1f}, {hi_DA:.1f}) Mpc "
                   f"(from D_c [{lo:.1f}, {hi:.1f}] at H0_ref="
                   f"{100 * h_ref:.0f}, Om={self.Om:.3f})")
            fprint("distance coordinate: D_A sampled directly; D_c init "
                   "values are converted once; no Jacobian term")

    # ---- spot indexing ----

    def _build_spot_indices(self, data):
        """Build spot type index arrays.

        Returns the accel_measured numpy array. Also populates:
            _idx_sys, _idx_red, _idx_blue      — per type
            _idx_sys_cons    — sys WITH accel
            _idx_sys_uncons  — sys WITHOUT accel
        """
        is_hv_np = _np.asarray(data["is_highvel"])
        is_blue_np = _np.asarray(data.get("is_blue", _np.zeros(
            self.n_spots, dtype=bool)))
        is_red_np = is_hv_np & ~is_blue_np
        is_sys_np = ~is_hv_np

        if "accel_measured" not in data:
            raise KeyError(
                "data dict must carry an explicit 'accel_measured' "
                "boolean array per spot. The σ_a-based fallback was "
                "removed to stop relying on loader sentinels.")
        accel_meas = _np.asarray(data["accel_measured"])

        self._idx_sys = jnp.where(jnp.asarray(is_sys_np))[0]
        self._idx_red = jnp.where(jnp.asarray(is_red_np))[0]
        self._idx_blue = jnp.where(jnp.asarray(is_blue_np))[0]
        self._idx_sys_cons = jnp.where(
            jnp.asarray(is_sys_np & accel_meas))[0]
        self._idx_sys_uncons = jnp.where(
            jnp.asarray(is_sys_np & ~accel_meas))[0]

        self._n_sys = int(is_sys_np.sum())
        self._n_red = int(is_red_np.sum())
        self._n_blue = int(is_blue_np.sum())
        self._n_sys_cons = int((is_sys_np & accel_meas).sum())
        self._n_sys_uncons = int((is_sys_np & ~accel_meas).sum())

        # Static (Python-side) flag: does any spot in this group have an
        # accel measurement? Used to gate da*da/var_a in _phi_eval so the
        # zero contribution never enters the JIT-compiled kernel.
        #   - sys (sampled-r path combines cons+uncons): any sys spot w/ accel
        #   - sys_cons: True by construction (accel_measured=True)
        self._group_has_accel = {
            "sys": bool((is_sys_np & accel_meas).any()),
            "sys_cons": True if self._n_sys_cons > 0 else False,
            "red": bool((is_red_np & accel_meas).any()),
            "blue": bool((is_blue_np & accel_meas).any()),
        }

        fprint(
            f"spot split: {self._n_sys} sys "
            f"({self._n_sys_cons} w/accel, {self._n_sys_uncons} w/o), "
            f"{self._n_red} red, {self._n_blue} blue.")
        return accel_meas

    def _build_allspot_arrays(self, data, accel_meas):
        """Build all-spot arrays in original data order."""
        self._all_x = jnp.asarray(data["x"])
        self._all_y = jnp.asarray(data["y"])
        self._all_sigma_x2 = jnp.asarray(data["sigma_x"])**2
        self._all_sigma_y2 = jnp.asarray(data["sigma_y"])**2
        self._all_v = jnp.asarray(self.velocity)
        # Velocity relative to the (exactly known) systemic constant, formed
        # once on the host in float64 so the ~v_sys-scale subtraction never
        # happens in float32. The φ integrand compares against this.
        self._all_v_rel = jnp.asarray(
            _np.asarray(self.velocity, dtype=_np.float64) - self.v_sys_obs,
            dtype=self._all_v.dtype)
        self._all_a = jnp.asarray(self.a)
        self._all_sigma_a = jnp.asarray(self.sigma_a)
        self._all_has_accel = jnp.asarray(accel_meas)
        self._all_sigma_a2 = self._all_sigma_a**2
        self._all_sigma_v2 = jnp.asarray(self.sigma_v)**2

    # ---- galaxy / feature config ----

    def _configure_galaxy(self, data):
        """Configure per-galaxy feature flags and warp pivots.

        Returns the galaxy config dict.
        """
        gname = data.get("galaxy_name", "")
        gal_cfg = get_nested(self.config, f"model/galaxies/{gname}", {})
        self._configure_features(gal_cfg)
        self._configure_mass_parameterization(gal_cfg)
        self._configure_warp_pivots(gal_cfg)
        return gal_cfg

    def _configure_features(self, gal_cfg):
        use_ecc = get_nested(self.config, "model/use_ecc", False)
        use_qw = get_nested(self.config, "model/use_quadratic_warp", False)
        self.use_ecc = gal_cfg.get("use_ecc", use_ecc)
        self.ecc_cartesian = gal_cfg.get("ecc_cartesian", True)
        self.use_quadratic_warp = gal_cfg.get("use_quadratic_warp", use_qw)
        flags = []
        if self.use_ecc:
            flags.append("ecc" + ("(cart)" if self.ecc_cartesian else ""))
        if self.use_quadratic_warp:
            flags.append("quad_warp")
        fprint("features: " + (", ".join(flags) if flags else "none"))

    def _configure_mass_parameterization(self, gal_cfg):
        mass_param = gal_cfg.get(
            "mass_parameterization",
            get_nested(self.config, "model/mass_parameterization",
                       "eta"))
        valid = ("eta", "log_mbh")
        if mass_param not in valid:
            raise ValueError(
                f"mass_parameterization must be one of {valid}; "
                f"got {mass_param!r}.")
        self.mass_parameterization = mass_param
        fprint(f"mass parameterization: {mass_param}")

    def _configure_warp_pivots(self, gal_cfg):
        keys = ("r_ang_ref_i", "r_ang_ref_Omega", "r_ang_ref_periapsis")
        missing = [key for key in keys if key not in gal_cfg]
        if missing:
            raise ValueError(
                "Megamaser warp pivot radii must be set in the galaxy "
                f"config; missing: {', '.join(missing)}.")

        self._r_ang_ref_i = float(gal_cfg["r_ang_ref_i"])
        self._r_ang_ref_Omega = float(gal_cfg["r_ang_ref_Omega"])
        self._r_ang_ref_periapsis = float(gal_cfg["r_ang_ref_periapsis"])
        fprint(
            f"r_ang_ref_i={self._r_ang_ref_i:.3f} mas "
            f"r_ang_ref_Omega={self._r_ang_ref_Omega:.3f} mas "
            f"r_ang_ref_periapsis={self._r_ang_ref_periapsis:.3f} mas")

    # ---- phi sub-ranges ----

    def _build_phi_subranges(self, gal_cfg):
        """Parse per-type φ sub-ranges from config.

        Stores self._phi_subranges as a dict mapping spot-type ("red",
        "blue", "sys") to a list of (lo_rad, hi_rad, n_phi) triplets.
        Each sub-range is evaluated with a uniform linspace and
        trapezoidal weights; disjoint sub-ranges for the same spot type
        are combined in log-space via logsumexp.
        """
        def _get(key, default):
            if key in gal_cfg:
                return gal_cfg[key]
            return get_nested(self.config, f"model/{key}", default)

        phi_integration = str(_get(
            "phi_integration", "fixed-grid")).lower()
        if phi_integration not in ("fixed-grid", "peak-partition"):
            raise ValueError(
                "phi_integration must be 'fixed-grid' or "
                f"'peak-partition'; got {phi_integration!r}.")
        n_partition_sys = int(_get("n_phi_partition_sys", 513))
        n_partition_hv = int(_get("n_phi_partition_hv", 257))
        if min(n_partition_sys, n_partition_hv) < 3:
            raise ValueError(
                "n_phi_partition_sys and n_phi_partition_hv must be >= 3.")

        hv_inner_deg = float(_get("phi_hv_inner_deg", 45.0))
        hv_outer_deg = float(_get("phi_hv_outer_deg", 90.0))
        n_high = int(_get("n_phi_hv_high", 401))
        n_low = int(_get("n_phi_hv_low", 101))

        if not (0 < hv_inner_deg < hv_outer_deg <= 180.0):
            raise ValueError(
                "Require 0 < phi_hv_inner_deg < phi_hv_outer_deg <= 180°; "
                f"got inner={hv_inner_deg}, outer={hv_outer_deg}.")
        if min(n_high, n_low) < 3:
            raise ValueError(
                "n_phi_hv_high and n_phi_hv_low must be >= 3.")

        i_rad = _np.deg2rad(hv_inner_deg)
        o_rad = _np.deg2rad(hv_outer_deg)
        pi2 = _np.pi / 2.0

        def _hv_subranges(peak):
            """3 sub-ranges around HV peak: low-dense wings + n_high core."""
            return [
                (peak - o_rad, peak - i_rad, n_low),
                (peak - i_rad, peak + i_rad, n_high),
                (peak + i_rad, peak + o_rad, n_low),
            ]

        red = _hv_subranges(pi2)
        blue = _hv_subranges(-pi2)

        sys_ranges_deg = _get(
            "phi_sys_ranges_deg", [[-45.0, 45.0], [135.0, 225.0]])
        n_sys = int(_get("n_phi_sys", 2001))
        if n_sys < 3:
            raise ValueError("n_phi_sys must be >= 3.")
        sys_rng = []
        for lo, hi in sys_ranges_deg:
            lo, hi = float(lo), float(hi)
            if hi <= lo:
                raise ValueError(
                    f"phi_sys_ranges_deg sub-range [{lo}, {hi}] must be "
                    "strictly increasing.")
            sys_rng.append((_np.deg2rad(lo), _np.deg2rad(hi), n_sys))

        self._phi_subranges = {"red": red, "blue": blue, "sys": sys_rng}

        # Precompute concatenated (sin, cos, log-weights) per type so
        # _eval_phi_marginal can do one _phi_eval + one logsumexp per group.
        # Uniform linspace + trapezoidal weights; disjoint sub-range weights
        # concatenate cleanly (endpoints keep h/2, interior h).
        def _nodes_log_w(lo, hi, n):
            phi = jnp.linspace(lo, hi, n)
            return phi, trapz_log_weights(phi)

        self._phi_concat = {}
        for key, subs in self._phi_subranges.items():
            sin_parts, cos_parts, w_parts = [], [], []
            for lo, hi, n in subs:
                phi, log_w = _nodes_log_w(lo, hi, n)
                sin_parts.append(jnp.sin(phi))
                cos_parts.append(jnp.cos(phi))
                w_parts.append(log_w)
            sin_c = jnp.concatenate(sin_parts)
            cos_c = jnp.concatenate(cos_parts)
            self._phi_concat[key] = dict(
                sin_phi=sin_c,
                cos_phi=cos_c,
                log_w_phi=jnp.concatenate(w_parts),
                # φ-basis for the quadratic-form integrand (theta-independent).
                sin2_phi=sin_c * sin_c,
                cos2_phi=cos_c * cos_c,
                sincos_phi=sin_c * cos_c,
            )

        # Save for summary/printing
        self._phi_hv_inner_deg = hv_inner_deg
        self._phi_hv_outer_deg = hv_outer_deg
        self._n_phi_hv_high = n_high
        self._n_phi_hv_low = n_low
        self._phi_sys_ranges_deg = sys_ranges_deg
        self._n_phi_sys = n_sys
        self.phi_integration = phi_integration
        self._n_phi_partition_sys = n_partition_sys
        self._n_phi_partition_hv = n_partition_hv

    # ---- r_ang support + conditional r-MAP support ----

    def _build_r_config(self, data, gal_cfg):
        """Set physical radius bounds and conditional r-MAP grid knobs."""
        def _get(key, default):
            if key in gal_cfg:
                return gal_cfg[key]
            return get_nested(self.config, f"model/{key}", default)

        _R_lo = float(_get("R_phys_lo", 0.01))
        _R_hi = float(_get("R_phys_hi", 2.0))
        self._R_phys_lo = _R_lo
        self._R_phys_hi = _R_hi

        self._n_r_local = int(_get("n_r_local", 151))
        self._n_r_global = int(_get("n_r_global", 301))
        self._K_sigma = float(_get("K_sigma", 5.0))
        if min(self._n_r_local, self._n_r_global) < 3:
            raise ValueError(
                "n_r_local and n_r_global must be >= 3.")

        self.use_selection = get_nested(
            self.config, "model/use_selection", False)

        self._refine_r_center = bool(_get("refine_r_center", True))
        self._n_refine_steps = int(_get("n_refine_steps", 32))
        sb = _get("conditional_spot_batch", None)
        self._conditional_spot_batch = int(sb) if sb is not None else None

    def _precompute_sinh_template(self, data):
        """Cache the sinh quadrature template used by r-MAP helpers."""
        del data
        T_max = float(_np.arcsinh(self._K_sigma))
        t = _np.linspace(-T_max, T_max, self._n_r_local)
        self._sinh_t_frozen = jnp.asarray(_np.sinh(t))

    # ---- summary ----
    def _print_summary(self):
        if self.phi_integration == "peak-partition":
            fprint(
                "phi integration: peak-partition "
                f"(systemic: two {self._n_phi_partition_sys}-node "
                "half-plane scans; "
                f"HV: {self._n_phi_partition_hv}-node half-plane scan)")
        else:
            fprint("phi integration: fixed-grid")
        fprint(
            f"φ HV: inner ±{self._phi_hv_inner_deg:.0f}° "
            f"(n={self._n_phi_hv_high}), outer wings to "
            f"±{self._phi_hv_outer_deg:.0f}° "
            f"(n={self._n_phi_hv_low} per wing)")
        rng_str = " ∪ ".join(
            f"[{lo:.0f}°, {hi:.0f}°]"
            for lo, hi in self._phi_sys_ranges_deg)
        fprint(f"φ sys: {rng_str}, n={self._n_phi_sys} per sub-range")
        fprint(
            f"r_ang support: bounds track D_A via "
            f"R_phys in [{self._R_phys_lo:.3f}, "
            f"{self._R_phys_hi:.3f}] pc")
        refine_str = "on" if self._refine_r_center else "off"
        fprint(
            f"conditional r-MAP grid: n_r_local={self._n_r_local}, "
            f"n_r_global={self._n_r_global}, refinement={refine_str}, "
            f"K={self._K_sigma}")

    def r_ang_range(self, D_A):
        """r_ang range in mas corresponding to physical R_phys bounds at D_A.

        Used by the latent radial updates to keep the physical/angular
        conversion in one place.
        """
        conv = D_A * PC_PER_MAS_MPC
        return self._R_phys_lo / conv, self._R_phys_hi / conv

    # ---- r estimation (per-spot physics-based centre + scale) ----

    def _closed_form_seeds(self, D_A, M_BH, v_sys, sigma_a_floor2,
                           i0, var_v_hv):
        """Closed-form seed + propagated-noise width for every spot.

        HV:    velocity → r_vel = M·(C_v·sin_i)² / (D·Δv²)
        sys+a: acceleration → r_acc = √(C_a·M·sin_i / (D²·|a|))
        sys-a: placeholder — `r_acc` sentinel value and a loose
               propagated-noise width; overwritten by the scan-based
               seed in `_compute_seeds` for sys-no-accel spots.
        Returns (r_est, s_prop, r_min, r_max) — all shape (n_spots,).
        """
        r_min, r_max = self.r_ang_range(D_A)

        sin_i = jnp.abs(jnp.sin(i0))

        dv = self._all_v - v_sys
        r_vel = radius_from_los_velocity(
            jnp.sqrt(dv ** 2 + R_EST_EPS), sin_i, D_A, M_BH)
        r_vel = jnp.clip(r_vel, r_min, r_max)

        r_acc = radius_from_los_acceleration(
            jnp.abs(self._all_a) + R_EST_EPS, sin_i, D_A, M_BH)
        r_acc = jnp.clip(r_acc, r_min, r_max)

        r_est = jnp.where(self.is_highvel, r_vel, r_acc)
        r_est = jnp.clip(r_est, r_min * 1.01, r_max * 0.99)

        sigma_v_eff = jnp.sqrt(var_v_hv)
        sigma_a_eff = jnp.sqrt(sigma_a_floor2)
        s_vel = 2.0 * sigma_v_eff / (jnp.abs(dv) + R_EST_EPS)
        s_acc = sigma_a_eff / (2.0 * jnp.abs(self._all_a) + R_EST_EPS)
        s_prop = jnp.where(
            self.is_highvel,
            jnp.maximum(s_vel, 0.05),
            jnp.maximum(s_acc, 0.1))
        return r_est, s_prop, r_min, r_max

    def radius_seeds(self, D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv):
        """Differentiable closed-form radius seeds (r_hat plus support bounds).

        Same closed forms as `_closed_form_seeds` (velocity inversion
        for HV spots, centripetal inversion for sys spots with an accel
        measurement); sys spots without an accel measurement get the
        geometric mean of the support, which is theta-smooth and weakly
        informative.

        Returns (r_hat, r_min, r_max).
        """
        r_hat, _, r_min, r_max = self._closed_form_seeds(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
        if self._n_sys_uncons > 0:
            r_geo = jnp.sqrt(r_min * r_max)
            r_hat = r_hat.at[self._idx_sys_uncons].set(r_geo)
        return r_hat, r_min, r_max

    def _scan_on_global_grid(self, type_key, idx, r_global,
                             phys_args, phys_kw, r_chunk=32,
                             spot_chunk=None, cache_scan=False):
        """Per-spot argmax on the phi-marginalised global radius grid."""
        n = int(idx.shape[0])
        if n == 0:
            return None, None, None
        n_r = int(r_global.shape[0])
        pc = self._phi_concat[type_key]
        has_any_accel = self._group_has_any_accel(type_key)
        if spot_chunk is None:
            spot_chunk = n
        spot_chunk = max(1, int(spot_chunk))

        r_parts = []
        ll_parts = []
        ll_grid_parts = []
        for i0 in range(0, n, spot_chunk):
            idx_chunk = idx[i0:i0 + spot_chunk]
            scan_parts = []
            for r0 in range(0, n_r, r_chunk):
                r_chunk_arr = r_global[r0:r0 + r_chunk]
                r_eval = (jnp.broadcast_to(
                    r_chunk_arr[None, :],
                    (idx_chunk.shape[0], r_chunk_arr.shape[0]))
                    if self.phi_integration == "peak-partition"
                    else r_chunk_arr)
                r_pre = self._r_precompute(
                    r_eval, idx_chunk, *phys_args, **phys_kw,
                    has_any_accel=has_any_accel)
                if self.phi_integration == "peak-partition":
                    ll_chunk, _, overflow = (
                        self._phi_partition_group_log_integral(
                            type_key, r_pre,
                            self._phi_partition_scan_size(type_key)))
                    ll_chunk = jnp.where(overflow, -jnp.inf, ll_chunk)
                else:
                    nhc = self._phi_eval_shared_r(
                        r_pre, pc["sin_phi"], pc["cos_phi"],
                        pc["sin2_phi"], pc["cos2_phi"], pc["sincos_phi"],
                        use_quadform=cache_scan)
                    ll_chunk = _compile_friendly_logsumexp(
                        nhc + pc["log_w_phi"][None, None, :])
                scan_parts.append(ll_chunk)
            ll_scan = jnp.concatenate(scan_parts, axis=-1)

            best = jnp.argmax(ll_scan, axis=-1)
            r_parts.append(r_global[best])
            ll_parts.append(jnp.take_along_axis(
                ll_scan, best[:, None], axis=-1).squeeze(-1))
            if cache_scan:
                ll_grid_parts.append(ll_scan)
        ll_grid = jnp.concatenate(ll_grid_parts) if cache_scan else None
        return (jnp.concatenate(r_parts), jnp.concatenate(ll_parts), ll_grid)

    def _compute_seeds(self, D_A, M_BH, v_sys, sigma_a_floor2,
                       i0, var_v_hv, phys_args, phys_kw, r_global,
                       cache_scan=False):
        """Per-spot seed and fallback width for conditional r-MAP."""
        r_est, s_prop, r_min, r_max = self._closed_form_seeds(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
        scan_cache = {}
        for type_key, idx in [("sys", self._idx_sys),
                              ("red", self._idx_red),
                              ("blue", self._idx_blue)]:
            r_scan, ll_scan_best, ll_scan = self._scan_on_global_grid(
                type_key, idx, r_global, phys_args, phys_kw,
                spot_chunk=self._conditional_spot_batch,
                cache_scan=cache_scan)
            if r_scan is None:
                continue
            if cache_scan:
                scan_cache[type_key] = ll_scan
            r_cf = r_est[idx]
            has_any_accel = self._group_has_any_accel(type_key)
            r_pre_cf = self._r_precompute(
                r_cf, idx, *phys_args, **phys_kw,
                has_any_accel=has_any_accel)
            if self.phi_integration == "peak-partition":
                ll_cf, _, overflow = self._phi_partition_group_log_integral(
                    type_key, r_pre_cf,
                    self._phi_partition_scan_size(type_key))
                ll_cf = jnp.where(overflow, -jnp.inf, ll_cf)
            else:
                pc = self._phi_concat[type_key]
                nhc_cf = self._phi_eval(
                    r_pre_cf, pc["sin_phi"], pc["cos_phi"])
                ll_cf = _compile_friendly_logsumexp(
                    nhc_cf + pc["log_w_phi"])
            scan_wins = ll_scan_best >= ll_cf
            r_best = jnp.where(scan_wins, r_scan, r_cf)
            r_est = r_est.at[idx].set(r_best)
        if self._n_sys_uncons > 0:
            log_bin = ((jnp.log(r_global[-1]) - jnp.log(r_global[0]))
                       / (r_global.shape[0] - 1))
            s_uc = jnp.full((self._n_sys_uncons,), 3.0 * log_bin,
                            dtype=r_est.dtype)
            s_prop = s_prop.at[self._idx_sys_uncons].set(s_uc)
        return r_est, s_prop, r_min, r_max, scan_cache

    def _build_conditional_r_grids(self, D_A, M_BH, v_sys, sigma_a_floor2,
                                   i0, var_v_hv,
                                   phys_args=None, phys_kw=None,
                                   return_scan_cache=False):
        """Build conditional r-grid objects for phi/r diagnostics.

        ``return_scan_cache`` also returns the global-radius phi marginals
        and their positions in each sorted local/global union.
        """
        have_phys = phys_args is not None and phys_kw is not None
        if not have_phys:
            raise ValueError(
                "_build_conditional_r_grids requires phys_args and phys_kw; "
                "pass the same values used by _eval_phi_marginal.")
        if phys_kw is None:
            phys_kw = {}

        r_cf, _, r_min, r_max = self._closed_form_seeds(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
        valid = self.is_highvel | self._all_has_accel.astype(bool)
        r_lo_data, r_hi_data = _conditional_global_r_window(
            valid, r_cf, r_min, r_max)
        r_global, _ = self._build_global_r_grid(r_lo_data, r_hi_data)

        r_est, s_fallback, r_min, r_max, scan_values = self._compute_seeds(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv,
            phys_args, phys_kw, r_global, cache_scan=return_scan_cache)

        def _refine(type_key, idx):
            r0 = r_est[idx]
            s0 = s_fallback[idx]
            if not self._refine_r_center:
                return r0, s0
            return self._refine_r_center_group(
                type_key, idx, r0, s0, r_min, r_max,
                phys_args, phys_kw)

        def _group(type_key, idx, n):
            if n == 0:
                return None
            r_c, s_spot = _refine(type_key, idx)
            r_local, _ = self._build_local_sinh(
                r_c, s_spot, r_min, r_max)
            if return_scan_cache:
                r_union, log_w_union, order = self._build_union(
                    r_local, r_global, return_order=True)
                cache = (jax.lax.stop_gradient(r_local),
                         jax.lax.stop_gradient(scan_values[type_key]),
                         jax.lax.stop_gradient(order))
            else:
                r_union, log_w_union = self._build_union(
                    r_local, r_global)
                cache = None
            return ((type_key, idx,
                     jax.lax.stop_gradient(r_union),
                     jax.lax.stop_gradient(log_w_union)), cache)

        groups = []
        caches = []
        for entry in (
                _group("sys", self._idx_sys, self._n_sys),
                _group("red", self._idx_red, self._n_red),
                _group("blue", self._idx_blue, self._n_blue)):
            if entry is not None:
                group, cache = entry
                groups.append(group)
                caches.append(cache)
        return (groups, caches) if return_scan_cache else groups

    def _build_local_sinh(self, r_c, s, r_min, r_max):
        """Per-spot sinh grid of shape (N, n_r_local)."""
        log_r_min = jnp.log(r_min)
        log_r_max = jnp.log(r_max)
        log_r_c = jnp.log(r_c)
        s_cap = (jnp.minimum(log_r_c - log_r_min,
                             log_r_max - log_r_c) / self._K_sigma)
        s = jnp.minimum(s, s_cap)
        log_r = (log_r_c[:, None]
                 + self._sinh_t_frozen[None, :] * s[:, None])
        r = jnp.clip(jnp.exp(log_r), r_min, r_max)
        return r, _trapz_log_w_per_spot(r)

    def _build_global_r_grid(self, r_min, r_max):
        """Shared log-uniform grid of shape (n_r_global,)."""
        dtype = jnp.result_type(r_min, r_max)
        log_r = jnp.linspace(
            jnp.log(r_min), jnp.log(r_max), self._n_r_global,
            dtype=dtype)
        r = jnp.exp(log_r)
        return r, trapz_log_weights(r)

    def _build_union(self, r_local, r_global, return_order=False):
        """Sorted per-spot union of local and global nodes."""
        N = r_local.shape[0]
        r_global_b = jnp.broadcast_to(
            r_global[None, :], (N, r_global.shape[0]))
        r_union = jnp.concatenate([r_local, r_global_b], axis=-1)
        if return_order:
            order = jnp.argsort(r_union, axis=-1)
            r_sorted = jnp.take_along_axis(r_union, order, axis=-1)
            return r_sorted, _trapz_log_w_per_spot(r_sorted), order
        r_sorted = jnp.sort(r_union, axis=-1)
        return r_sorted, _trapz_log_w_per_spot(r_sorted)

    def get_conditional_r_centres(self, phys_args, phys_kw=None,
                                  with_width=True):
        """Return per-group conditional radius centres and widths."""
        if phys_kw is None:
            phys_kw = {}
        D_A = phys_args[2]
        M_BH = phys_args[3]
        v_sys = phys_args[4]
        i0 = phys_args[8]
        var_v_hv = phys_args[15]
        sigma_a_floor2 = phys_args[16]
        r_cf, _, r_min, r_max = self._closed_form_seeds(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
        valid = self.is_highvel | self._all_has_accel.astype(bool)
        r_lo_data, r_hi_data = _conditional_global_r_window(
            valid, r_cf, r_min, r_max)
        r_global, _ = self._build_global_r_grid(r_lo_data, r_hi_data)

        r_est, s_fallback, r_min, r_max, _ = self._compute_seeds(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv,
            phys_args, phys_kw, r_global)

        def _one(type_key, idx, n):
            if n == 0:
                return None
            r0 = r_est[idx]
            s0 = s_fallback[idx]
            if self._refine_r_center:
                r_c, s = self._refine_r_center_group(
                    type_key, idx, r0, s0, r_min, r_max,
                    phys_args, phys_kw, with_width=with_width)
            else:
                r_c, s = r0, s0
            return dict(r_c=r_c, s=s)

        return dict(
            r_min=r_min, r_max=r_max,
            sys=_one("sys", self._idx_sys, self._n_sys),
            red=_one("red", self._idx_red, self._n_red),
            blue=_one("blue", self._idx_blue, self._n_blue))

    def conditional_r_ang_map(self, phys_args, phys_kw=None):
        """Return a length-``n_spots`` conditional MAP radius vector."""
        centres = self.get_conditional_r_centres(
            phys_args, phys_kw, with_width=False)
        dtype = jnp.asarray(phys_args[2]).dtype
        r_ang = jnp.zeros((self.n_spots,), dtype=dtype)
        for type_key in ("sys", "red", "blue"):
            group = centres[type_key]
            if group is None:
                continue
            idx = getattr(self, f"_idx_{type_key}")
            r_ang = r_ang.at[idx].set(group["r_c"])
        return r_ang

    def _refine_r_center_group(self, type_key, idx, r_est_group,
                               s_fallback, r_min, r_max,
                               phys_args, phys_kw, with_width=True):
        """Refine per-spot grid centre via Brent's method in log(r)."""
        pc = self._phi_concat[type_key]
        sin_phi = pc["sin_phi"]
        cos_phi = pc["cos_phi"]
        log_w_phi = pc["log_w_phi"]
        has_any_accel = self._group_has_any_accel(type_key)

        (x0, y0, D_A, M_BH, v_sys,
         r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
         i0, di_dr, Omega0, dOmega_dr,
         sigma_x_floor2, sigma_y_floor2, var_v_sys, var_v_hv,
         sigma_a_floor2) = phys_args
        d2i_dr2 = phys_kw.get("d2i_dr2", 0.0)
        d2Omega_dr2 = phys_kw.get("d2Omega_dr2", 0.0)
        e_x = phys_kw.get("e_x", None)
        e_y = phys_kw.get("e_y", 0.0)
        dperiapsis_dr = phys_kw.get("dperiapsis_dr", 0.0)
        dv_sys = phys_kw.get("dv_sys", 0.0)

        dtype = r_est_group.dtype
        x_g = self._all_x[idx].astype(dtype)
        y_g = self._all_y[idx].astype(dtype)
        v_g = self._all_v_rel[idx].astype(dtype)
        a_g = self._all_a[idx].astype(dtype)
        sx2_g = self._all_sigma_x2[idx].astype(dtype)
        sy2_g = self._all_sigma_y2[idx].astype(dtype)
        sv2_g = self._all_sigma_v2[idx].astype(dtype)
        sa2_g = self._all_sigma_a2[idx].astype(dtype)
        has_a_g = self._all_has_accel[idx].astype(dtype)
        is_hv_g = self.is_highvel[idx]

        ell_lo = jnp.log(r_min * 1.01)
        ell_hi = jnp.log(r_max * 0.99)
        ell_est = jnp.log(r_est_group)

        def f_one(ell, spot):
            (xi, yi, vi, ai, sx2i, sy2i, sv2i, sa2i, hai, ishvi) = spot
            r = jnp.exp(ell)

            i_r, Om_r = warp_geometry(
                r, r_ang_ref_i, r_ang_ref_Omega,
                i0, di_dr, Omega0, dOmega_dr,
                d2i_dr2, d2Omega_dr2)
            sin_i = jnp.sin(i_r)
            cos_i = jnp.cos(i_r)
            sin_O = jnp.sin(Om_r)
            cos_O = jnp.cos(Om_r)

            X, Y = predict_position(
                r, sin_phi, cos_phi, x0, y0,
                sin_i, cos_i, sin_O, cos_O)
            if e_x is None:
                V = predict_velocity_los(
                    r, sin_phi, cos_phi, D_A, M_BH, v_sys, dv_sys, sin_i)
            else:
                delta = dperiapsis_dr * (r - r_ang_ref_periapsis)
                cos_del = jnp.cos(delta)
                sin_del = jnp.sin(delta)
                V = predict_velocity_los(
                    r, sin_phi, cos_phi, D_A, M_BH, v_sys, dv_sys, sin_i,
                    ecc2=e_x * e_x + e_y * e_y,
                    ecc_cos_om=e_x * cos_del - e_y * sin_del,
                    ecc_sin_om=e_x * sin_del + e_y * cos_del)

            var_x = sx2i + sigma_x_floor2
            var_y = sy2i + sigma_y_floor2
            var_v = sv2i + jnp.where(ishvi, var_v_hv, var_v_sys)

            nhc = neg_half_chi2_position(xi, yi, X, Y, var_x, var_y)
            nhc = nhc + neg_half_chi2_velocity(vi, V, var_v)
            if has_any_accel:
                A = predict_acceleration_los(
                    r, sin_phi, cos_phi, D_A, M_BH, sin_i)
                var_a = sa2i + sigma_a_floor2
                nhc = nhc + neg_half_chi2_acceleration(
                    ai, A, var_a, hai)
            return -_compile_friendly_logsumexp(nhc + log_w_phi)

        if self.phi_integration == "peak-partition":
            def f_partition(ell, spot_idx):
                r_pre = self._r_precompute(
                    jnp.exp(ell)[None], spot_idx[None],
                    *phys_args, **phys_kw,
                    has_any_accel=has_any_accel)
                ll, _, overflow = self._phi_partition_group_log_integral(
                    type_key, r_pre,
                    self._phi_partition_scan_size(type_key))
                return -jnp.where(overflow[0], -jnp.inf, ll[0])

            f_one = f_partition

        log_bin = ((jnp.log(r_max) - jnp.log(r_min))
                   / (self._n_r_global - 1))
        bracket_half = 3.0 * log_bin
        a_bracket = jnp.maximum(ell_est - bracket_half, ell_lo)
        b_bracket = jnp.minimum(ell_est + bracket_half, ell_hi)

        spot_data = (idx if self.phi_integration == "peak-partition" else
                     (x_g, y_g, v_g, a_g,
                      sx2_g, sy2_g, sv2_g, sa2_g,
                      has_a_g, is_hv_g))

        def optim_one(a, b, spot):
            return brent_1d(
                lambda ell: f_one(ell, spot), a, b,
                n_steps=self._n_refine_steps)

        ell_opt = jax.vmap(optim_one, in_axes=(0, 0, 0))(
            a_bracket, b_bracket, spot_data)
        r_opt = jnp.exp(ell_opt)
        if not with_width:
            bad = ~jnp.isfinite(r_opt)
            r_c = jnp.where(bad, r_est_group, r_opt)
            return r_c, s_fallback

        K = self._K_sigma
        target_rise = K * K / 2.0
        f_0 = jax.vmap(f_one)(ell_opt, spot_data)
        log_half_range = 0.5 * (ell_hi - ell_lo)

        def _find_half_width(ell_c, f_c, spot, direction):
            lo = jnp.zeros_like(ell_c)
            hi = jnp.full_like(ell_c, log_half_range)

            def body(_, state):
                lo, hi = state
                mid = 0.5 * (lo + hi)
                f_mid = f_one(ell_c + direction * mid, spot)
                rise = f_mid - f_c
                return (jnp.where(rise < target_rise, mid, lo),
                        jnp.where(rise < target_rise, hi, mid))

            lo, hi = jax.lax.fori_loop(0, 20, body, (lo, hi))
            return 0.5 * (lo + hi)

        s_right = jax.vmap(_find_half_width, in_axes=(0, 0, 0, None))(
            ell_opt, f_0, spot_data, 1.0)
        s_left = jax.vmap(_find_half_width, in_axes=(0, 0, 0, None))(
            ell_opt, f_0, spot_data, -1.0)
        s = jnp.maximum(s_right, s_left) / K

        bad = ~jnp.isfinite(r_opt) | ~jnp.isfinite(s)
        r_c = jnp.where(bad, r_est_group, r_opt)
        s = jnp.where(bad, s_fallback, s)
        return r_c, s

    # ---- unified φ integrand (1-D or 2-D r_ang) ----

    def _r_precompute(self, r_ang, idx,
                      x0, y0, D_A, M_BH, v_sys,
                      r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
                      i0, di_dr, Omega0, dOmega_dr,
                      sigma_x_floor2, sigma_y_floor2,
                      var_v_sys, var_v_hv, sigma_a_floor2,
                      d2i_dr2=0.0, d2Omega_dr2=0.0,
                      e_x=None, e_y=None, dperiapsis_dr=0.0,
                      dv_sys=0.0, has_any_accel=True):
        """Gather per-spot data and warped angles for the φ integrand.

        Returned pytree is consumed by _phi_eval or _phi_eval_shared_r.
        ``r_ang`` accepts ``(N,)`` for sampled radii, ``(N, n_r)`` for
        per-spot diagnostic grids, or ``(n_r,)`` for shared-r scans.
        Warped angles i(r), Omega(r), omega(r) share r_ang's shape; the
        phi-eval step broadcasts them against the (n_phi,) sin/cos grid
        by padding a trailing axis.
        """
        all_x = self._all_x[idx]
        all_y = self._all_y[idx]
        all_v_rel = self._all_v_rel[idx]
        all_a = self._all_a[idx]
        sx2 = self._all_sigma_x2[idx]
        sy2 = self._all_sigma_y2[idx]
        sv2 = self._all_sigma_v2[idx]
        sa2 = self._all_sigma_a2[idx]
        has_a = self._all_has_accel[idx].astype(r_ang.dtype)
        is_hv = self.is_highvel[idx]

        i_r, Om_r = warp_geometry(
            r_ang, r_ang_ref_i, r_ang_ref_Omega,
            i0, di_dr, Omega0, dOmega_dr,
            d2i_dr2, d2Omega_dr2)
        sin_i_r = jnp.sin(i_r)
        cos_i_r = jnp.cos(i_r)
        sin_O_r = jnp.sin(Om_r)
        cos_O_r = jnp.cos(Om_r)

        if e_x is not None:
            # Rotate the Cartesian eccentricity by the radius-dependent warp
            # delta = dperiapsis_dr·(r - r_ref); ecc·cos ω(r) and ecc·sin ω(r)
            # stay smooth polynomials in (e_x, e_y) (no e·direction split).
            delta = dperiapsis_dr * (r_ang - r_ang_ref_periapsis)
            cos_del = jnp.cos(delta)
            sin_del = jnp.sin(delta)
            ecc_cos_om_r = e_x * cos_del - e_y * sin_del
            ecc_sin_om_r = e_x * sin_del + e_y * cos_del
            ecc2 = e_x * e_x + e_y * e_y
        else:
            ecc_cos_om_r = None
            ecc_sin_om_r = None
            ecc2 = None

        var_x = sx2 + sigma_x_floor2
        var_y = sy2 + sigma_y_floor2
        var_v = sv2 + jnp.where(is_hv, var_v_hv, var_v_sys)
        # Per-spot Gaussian normalisation (added after the φ/r integral
        # — see the neg_half_chi2_* docstrings for the precision rationale).
        lnorm = -0.5 * (3 * LOG_2PI + jnp.log(var_x) +
                        jnp.log(var_y) + jnp.log(var_v))
        if has_any_accel:
            # Loaders supply a large (but finite) placeholder σ_a for
            # spots without a real acceleration measurement, so var_a
            # stays strictly positive. has_a then zeroes out both the
            # log-norm and the residual contribution for those spots.
            var_a = sa2 + sigma_a_floor2
            lnorm_a = -0.5 * (LOG_2PI + jnp.log(var_a)) * has_a
        else:
            var_a = None
            lnorm_a = jnp.zeros_like(lnorm)

        return dict(
            r_ang=r_ang,
            sin_i=sin_i_r, cos_i=cos_i_r,
            sin_O=sin_O_r, cos_O=cos_O_r,
            ecc_cos_om=ecc_cos_om_r, ecc_sin_om=ecc_sin_om_r, ecc2=ecc2,
            x0=x0, y0=y0, D=D_A, M_BH=M_BH, v_sys=v_sys, dv_sys=dv_sys,
            all_x=all_x, all_y=all_y, all_v_rel=all_v_rel, all_a=all_a,
            var_x=var_x, var_y=var_y, var_v=var_v, var_a=var_a,
            has_a=has_a, lnorm=lnorm, lnorm_a=lnorm_a,
            has_any_accel=has_any_accel,
        )

    def _predict_on_grid(self, r_pre, sin_phi, cos_phi, rpad):
        """Evaluate predict_* on an (r, φ) grid broadcast by ``rpad``.

        Returns (X, Y, V, A) with A = None when no spot in this group has
        an accel measurement.
        """
        r_b = r_pre["r_ang"][rpad]
        sin_i_b = r_pre["sin_i"][rpad]
        cos_i_b = r_pre["cos_i"][rpad]
        sin_O_b = r_pre["sin_O"][rpad]
        cos_O_b = r_pre["cos_O"][rpad]

        X, Y = predict_position(
            r_b, sin_phi, cos_phi, r_pre["x0"], r_pre["y0"],
            sin_i_b, cos_i_b, sin_O_b, cos_O_b)

        ecc2 = r_pre["ecc2"]
        if ecc2 is None:
            V = predict_velocity_los(
                r_b, sin_phi, cos_phi, r_pre["D"], r_pre["M_BH"],
                r_pre["v_sys"], r_pre["dv_sys"], sin_i_b)
        else:
            V = predict_velocity_los(
                r_b, sin_phi, cos_phi, r_pre["D"], r_pre["M_BH"],
                r_pre["v_sys"], r_pre["dv_sys"], sin_i_b,
                ecc2=ecc2, ecc_cos_om=r_pre["ecc_cos_om"][rpad],
                ecc_sin_om=r_pre["ecc_sin_om"][rpad])

        if r_pre["has_any_accel"]:
            A = predict_acceleration_los(
                r_b, sin_phi, cos_phi,
                r_pre["D"], r_pre["M_BH"], sin_i_b)
        else:
            A = None
        return X, Y, V, A

    def _phi_eval(self, r_pre, sin_phi, cos_phi,
                  sin2=None, cos2=None, sincos=None):
        """−½χ² at every (r, φ) gridpoint for per-spot r.

        r_pre   : pytree from _r_precompute with r_ang shape (N,) for
                  sampled radii or (N, n_r) for per-spot diagnostic grids.
        sin_phi, cos_phi : shape (n_phi,).
        Returns shape (N, [n_r,] n_phi) — residual term only; callers
        add lnorm/lnorm_a after logsumexp.

        Circular orbits under x64 use the quadratic-form integrand
        (`_neg_half_chi2_quadform`): ~2.5× fewer big-tensor ops. It expands
        a cancellation-prone constant, so it is restricted to float64 (the
        production regime); float32 and the eccentric branch keep the
        residual-stable predict/chi² path. `sin2/cos2/sincos` are the
        precomputed φ-basis; computed inline if omitted.
        """
        if r_pre["ecc2"] is None and jax.config.jax_enable_x64:
            if sin2 is None:
                sin2 = sin_phi * sin_phi
                cos2 = cos_phi * cos_phi
                sincos = sin_phi * cos_phi
            return _neg_half_chi2_quadform(
                r_pre, sin_phi, cos_phi, sin2, cos2, sincos)
        r_ang = r_pre["r_ang"]
        rpad = (slice(None),) * r_ang.ndim + (None,)
        dpad = (slice(None),) + (None,) * r_ang.ndim

        X, Y, V, A = self._predict_on_grid(r_pre, sin_phi, cos_phi, rpad)

        nhc = neg_half_chi2_position(
            r_pre["all_x"][dpad], r_pre["all_y"][dpad], X, Y,
            r_pre["var_x"][dpad], r_pre["var_y"][dpad])
        nhc = nhc + neg_half_chi2_velocity(
            r_pre["all_v_rel"][dpad], V, r_pre["var_v"][dpad])
        if r_pre["has_any_accel"]:
            nhc = nhc + neg_half_chi2_acceleration(
                r_pre["all_a"][dpad], A,
                r_pre["var_a"][dpad], r_pre["has_a"][dpad])
        return nhc

    def _phi_value(self, r_pre, phi):
        return self._phi_eval(r_pre, jnp.sin(phi), jnp.cos(phi))

    def _phi_value_slope(self, r_pre, phi):
        """Evaluate the phi integrand and its exact JAX directional slope."""
        value = lambda p: self._phi_value(r_pre, p)  # noqa: E731
        return jax.jvp(value, (phi,), (jnp.ones_like(phi),))

    def _phi_partition_scan_size(self, type_key):
        return (self._n_phi_partition_sys if type_key == "sys"
                else self._n_phi_partition_hv)

    def _phi_partition_log_integral(
            self, r_pre, phi_lo, phi_hi, n_scan,
            root_capacity=8, root_steps=16, drop=24.0,
            drop_steps=16, core_order=24, tail_order=8):
        """Log-integrate phi after partitioning at numerical extrema.

        ``r_pre`` must use per-spot radii, with shape ``(N,)`` or
        ``(N, n_r)``.  The fixed scan and masked root buffer keep all shapes
        static under JIT while ``jax.jvp`` differentiates the existing
        circular or eccentric likelihood without a second physics formula.

        Returns ``(log_integral, root_count, overflow)``.
        """
        if n_scan < 3:
            raise ValueError("n_scan must be >= 3.")
        if root_capacity < 1 or root_capacity >= n_scan:
            raise ValueError(
                "root_capacity must be in [1, n_scan - 1].")

        dtype = r_pre["r_ang"].dtype
        phi_scan = jnp.linspace(phi_lo, phi_hi, n_scan, dtype=dtype)
        _, slope = self._phi_value_slope(r_pre, phi_scan)

        finite = jnp.isfinite(slope)
        negative = slope < 0.0
        crossing = ((negative[..., :-1] != negative[..., 1:])
                    & finite[..., :-1] & finite[..., 1:])
        root_count = jnp.sum(crossing, axis=-1)

        n_cells = n_scan - 1
        cell = jnp.arange(n_cells)
        score = jnp.where(crossing, -cell, -n_cells - cell)
        top_score, bracket = jax.lax.top_k(score, root_capacity)
        valid_root = top_score > -n_cells

        a = phi_scan[bracket]
        b = phi_scan[bracket + 1]
        fa = jnp.take_along_axis(slope[..., :-1], bracket, axis=-1)

        def bisect_root(_, state):
            a, b, fa = state
            mid = 0.5 * (a + b)
            _, fm = self._phi_value_slope(r_pre, mid)
            same_side = (fa < 0.0) == (fm < 0.0)
            move_a = valid_root & same_side
            return (jnp.where(move_a, mid, a),
                    jnp.where(valid_root & ~same_side, mid, b),
                    jnp.where(move_a, fm, fa))

        a, b, _ = jax.lax.fori_loop(
            0, root_steps, bisect_root, (a, b, fa))
        roots = jnp.where(valid_root, 0.5 * (a + b), phi_hi)

        edge_shape = roots.shape[:-1] + (1,)
        lo = jnp.full(edge_shape, phi_lo, dtype=dtype)
        hi = jnp.full(edge_shape, phi_hi, dtype=dtype)
        bounds = jnp.concatenate((lo, roots, hi), axis=-1)
        int_lo = bounds[..., :-1]
        int_hi = bounds[..., 1:]
        n_kept = jnp.minimum(root_count, root_capacity)
        valid_interval = (
            jnp.arange(root_capacity + 1) <= n_kept[..., None])

        ell_lo = self._phi_value(r_pre, int_lo)
        ell_hi = self._phi_value(r_pre, int_hi)
        peak_left = ell_lo >= ell_hi
        peak_phi = jnp.where(peak_left, int_lo, int_hi)
        far_phi = jnp.where(peak_left, int_hi, int_lo)
        peak_ell = jnp.where(peak_left, ell_lo, ell_hi)
        far_ell = jnp.where(peak_left, ell_hi, ell_lo)
        target = peak_ell - jnp.asarray(drop, dtype=dtype)
        has_drop = far_ell <= target

        def bisect_drop(_, state):
            t_lo, t_hi = state
            t = 0.5 * (t_lo + t_hi)
            phi = peak_phi + t * (far_phi - peak_phi)
            ell = self._phi_value(r_pre, phi)
            above = ell >= target
            return (jnp.where(has_drop & above, t, t_lo),
                    jnp.where(has_drop & ~above, t, t_hi))

        t0 = jnp.zeros_like(peak_phi)
        t1 = jnp.ones_like(peak_phi)
        t_lo, t_hi = jax.lax.fori_loop(
            0, drop_steps, bisect_drop, (t0, t1))
        t_split = jnp.where(has_drop, 0.5 * (t_lo + t_hi), 1.0)
        split = peak_phi + t_split * (far_phi - peak_phi)

        def integrate(a, b, order):
            nodes_np, weights_np = _np.polynomial.legendre.leggauss(order)
            nodes = jnp.asarray(nodes_np, dtype=dtype)
            log_weights = jnp.log(jnp.asarray(weights_np, dtype=dtype))
            mid = 0.5 * (a + b)
            half = 0.5 * (b - a)
            phi = mid[..., None] + half[..., None] * nodes
            flat_shape = phi.shape[:-2] + (-1,)
            ell = self._phi_value(r_pre, phi.reshape(flat_shape))
            ell = ell.reshape(phi.shape)
            log_half = jnp.where(
                half > 0.0, jnp.log(half), -jnp.inf)
            return logsumexp(
                ell + log_weights + log_half[..., None], axis=-1)

        core_lo = jnp.minimum(peak_phi, split)
        core_hi = jnp.maximum(peak_phi, split)
        tail_lo = jnp.minimum(split, far_phi)
        tail_hi = jnp.maximum(split, far_phi)
        log_core = integrate(core_lo, core_hi, core_order)
        log_tail = integrate(tail_lo, tail_hi, tail_order)
        log_interval = jnp.logaddexp(log_core, log_tail)
        log_interval = jnp.where(valid_interval, log_interval, -jnp.inf)
        return (logsumexp(log_interval, axis=-1), root_count,
                root_count > root_capacity)

    def _phi_partition_group_log_integral(
            self, type_key, r_pre, n_scan, **partition_kw):
        """Partition-integrate one group on independent phi half-planes.

        Systemic spots always use ``[-pi, 0]`` and ``[0, pi]`` separately;
        red and blue each use their configured half-plane.  The returned root
        count is the maximum in any contributing half-plane, so capacity and
        overflow remain per-half-plane diagnostics.
        """
        if type_key == "sys":
            ranges = ((-jnp.pi, 0.0), (0.0, jnp.pi))
        elif type_key in ("red", "blue"):
            subs = self._phi_subranges[type_key]
            ranges = ((subs[0][0], subs[-1][1]),)
        else:
            raise ValueError(f"Unknown maser type {type_key!r}.")

        bounds = jnp.asarray(ranges, dtype=r_pre["r_ang"].dtype)

        def integrate_half(bound):
            return self._phi_partition_log_integral(
                r_pre, bound[0], bound[1], n_scan, **partition_kw)

        values, roots, overflows = jax.vmap(integrate_half)(bounds)
        return (logsumexp(values, axis=0),
                jnp.max(roots, axis=0),
                jnp.any(overflows, axis=0))

    def _phi_eval_shared_r(self, r_pre, sin_phi, cos_phi,
                           sin2=None, cos2=None, sincos=None,
                           use_quadform=False):
        """Shared-r variant of `_phi_eval` for scan/reference grids."""
        if (use_quadform and r_pre["ecc2"] is None
                and jax.config.jax_enable_x64):
            if sin2 is None:
                sin2 = sin_phi * sin_phi
                cos2 = cos_phi * cos_phi
                sincos = sin_phi * cos_phi
            return _neg_half_chi2_quadform(
                r_pre, sin_phi, cos_phi, sin2, cos2, sincos,
                shared_r=True)
        rpad = (slice(None), None)
        X, Y, V, A = self._predict_on_grid(r_pre, sin_phi, cos_phi, rpad)

        dpad = (slice(None), None, None)
        X3, Y3, V3 = X[None], Y[None], V[None]

        nhc = neg_half_chi2_position(
            r_pre["all_x"][dpad], r_pre["all_y"][dpad], X3, Y3,
            r_pre["var_x"][dpad], r_pre["var_y"][dpad])
        nhc = nhc + neg_half_chi2_velocity(
            r_pre["all_v_rel"][dpad], V3, r_pre["var_v"][dpad])
        if r_pre["has_any_accel"]:
            nhc = nhc + neg_half_chi2_acceleration(
                r_pre["all_a"][dpad], A[None],
                r_pre["var_a"][dpad], r_pre["has_a"][dpad])
        return nhc

    def _phi_integrand(self, r_ang, sin_phi, cos_phi, idx,
                       x0, y0, D_A, M_BH, v_sys,
                       r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
                       i0, di_dr, Omega0, dOmega_dr,
                       sigma_x_floor2, sigma_y_floor2,
                       var_v_sys, var_v_hv, sigma_a_floor2,
                       d2i_dr2=0.0, d2Omega_dr2=0.0,
                       e_x=None, e_y=None, dperiapsis_dr=0.0):
        """Convenience wrapper for dense phi-reference diagnostics."""
        r_pre = self._r_precompute(
            r_ang, idx, x0, y0, D_A, M_BH, v_sys,
            r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
            i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor2, sigma_y_floor2,
            var_v_sys, var_v_hv, sigma_a_floor2,
            d2i_dr2=d2i_dr2, d2Omega_dr2=d2Omega_dr2,
            e_x=e_x, e_y=e_y, dperiapsis_dr=dperiapsis_dr,
            dv_sys=v_sys - self.v_sys_obs)
        neg_half_chi2 = self._phi_eval(r_pre, sin_phi, cos_phi)
        rpad = (slice(None),) * r_ang.ndim + (None,)
        return (r_pre["lnorm"] + r_pre["lnorm_a"])[rpad] + neg_half_chi2

    # ---- phi marginal ----

    def _group_has_any_accel(self, type_key):
        """Is there at least one accel-measured spot in this group?

        For "sys", uses "sys_cons" when there are no sys-uncons spots
        (every sys spot then has an accel measurement) and "sys"
        otherwise (the group mixes accel-measured and not).
        """
        if type_key == "sys":
            key = "sys_cons" if self._n_sys_uncons == 0 else "sys"
        else:
            key = type_key
        return self._group_has_accel[key]

    def _marginal_per_spot_r(self, type_key, idx, r_ang, log_w_r,
                             has_any_accel, phys_args, phys_kw, batch,
                             scan_cache=None):
        """Per-spot log-marginal for groups with a per-spot r grid.

        Used with ``r_ang`` shape ``(N,)`` and ``log_w_r is None``.

        ``batch is None`` (or ``batch >= n_idx``) -> single-shot
        evaluation (one XLA op). Otherwise the spot axis is chunked
        with ``jax.lax.scan`` so the largest live intermediate is
        ``O(batch · [n_r ·] n_phi)``. Inputs are padded to a multiple
        of ``batch`` so every scan iteration sees identical shapes
        (single compile, no per-residual recompile); padding is
        sliced off the output. Returns shape ``(N_group,)``.

        ``scan_cache`` avoids re-evaluating the global-radius columns in a
        conditional grid; it is used only by the circular DE path.
        """
        pc = self._phi_concat[type_key]
        n_idx = int(idx.shape[0])

        def _eval(idx_b, r_b, lwr_b, cache_b):
            r_eval = r_b if cache_b is None else cache_b[0]
            r_pre = self._r_precompute(
                r_eval, idx_b, *phys_args, **phys_kw,
                has_any_accel=has_any_accel)
            lnorm_b = r_pre["lnorm"] + r_pre["lnorm_a"]
            if self.phi_integration == "peak-partition":
                ll_phi, _, overflow = (
                    self._phi_partition_group_log_integral(
                        type_key, r_pre,
                        self._phi_partition_scan_size(type_key)))
                if lwr_b is None:
                    return lnorm_b + jnp.where(
                        overflow, -jnp.inf, ll_phi)
                ll = logsumexp(ll_phi + lwr_b, axis=-1)
                return lnorm_b + jnp.where(
                    jnp.any(overflow, axis=-1), -jnp.inf, ll)
            # _phi_eval returns −½χ² only; lnorm is added after
            # logsumexp so the max-subtraction acts on bounded χ²
            # differences (protects float32 precision).
            nhc = self._phi_eval(
                r_pre, pc["sin_phi"], pc["cos_phi"],
                pc["sin2_phi"], pc["cos2_phi"], pc["sincos_phi"])
            if lwr_b is None:
                return lnorm_b + _compile_friendly_logsumexp(
                    nhc + pc["log_w_phi"])
            if cache_b is not None:
                return lnorm_b + _combine_cached_phi_marginal(
                    nhc, cache_b[1], cache_b[2], lwr_b,
                    pc["log_w_phi"])
            w2d = lwr_b[:, :, None] + pc["log_w_phi"][None, None, :]
            return lnorm_b + _compile_friendly_logsumexp(
                nhc + w2d, axis=(-2, -1))

        if batch is None or batch >= n_idx:
            return _eval(idx, r_ang, log_w_r, scan_cache)

        n_chunks = (n_idx + batch - 1) // batch
        n_pad = n_chunks * batch - n_idx
        if n_pad:
            idx_p = jnp.concatenate([idx, idx[:n_pad]])
            r_p = jnp.concatenate([r_ang, r_ang[:n_pad]], axis=0)
            lwr_p = (None if log_w_r is None else
                     jnp.concatenate([log_w_r, log_w_r[:n_pad]], axis=0))
        else:
            idx_p, r_p, lwr_p = idx, r_ang, log_w_r
        if scan_cache is not None:
            cache_p = tuple(
                jnp.concatenate([x, x[:n_pad]], axis=0) if n_pad else x
                for x in scan_cache)
        else:
            cache_p = None
        idx_c = idx_p.reshape(n_chunks, batch)
        r_c = r_p.reshape(n_chunks, batch, *r_p.shape[1:])

        if cache_p is not None:
            lwr_c = lwr_p.reshape(n_chunks, batch, *lwr_p.shape[1:])
            cache_c = tuple(
                x.reshape(n_chunks, batch, *x.shape[1:]) for x in cache_p)

            def body(_, x):
                cache_b = (x[3], x[4], x[5])
                return None, _eval(x[0], x[1], x[2], cache_b)

            xs = (idx_c, r_c, lwr_c, *cache_c)
        elif lwr_p is None:
            def body(_, x):
                return None, _eval(x[0], x[1], None, None)

            xs = (idx_c, r_c)
        else:
            lwr_c = lwr_p.reshape(n_chunks, batch, *lwr_p.shape[1:])

            def body(_, x):
                return None, _eval(x[0], x[1], x[2], None)

            xs = (idx_c, r_c, lwr_c)
        _, ps_chunks = jax.lax.scan(body, None, xs)
        return ps_chunks.reshape(-1)[:n_idx]

    def _spot_groups_from_r(self, r_spots):
        """Per-class `(type_key, idx, r_ang, None)` groups from a
        length-`n_spots` sampled radius vector."""
        groups = []
        for type_key in ("sys", "red", "blue"):
            idx = getattr(self, f"_idx_{type_key}")
            if int(idx.shape[0]) > 0:
                groups.append((type_key, idx, r_spots[idx], None))
        return groups

    def _eval_phi_marginal(self, spot_groups, phys_args, phys_kw=None,
                           spot_batch=None):
        """Compute the per-spot log-marginal likelihood, scattered into
        a length-`n_spots` array in original data order.

        `spot_groups` is the list of `(type_key, idx, r_ang, None)`
        tuples produced by `_spot_groups_from_r`. Each group describes
        one spot class (red / blue / sys) and is dispatched to
        `_marginal_per_spot_r`.

        `spot_batch` optionally caps the (N_batch, n_phi) intermediates.
        """
        if phys_kw is None:
            phys_kw = {}
        dtype = jnp.asarray(phys_args[2]).dtype
        result = jnp.zeros(self.n_spots, dtype=dtype)

        for group in spot_groups:
            type_key, idx, r_ang, log_w_r = group
            n_idx = int(idx.shape[0])
            if n_idx == 0:
                continue

            has_any_accel = self._group_has_any_accel(type_key)
            batch = (None if spot_batch is None
                     else min(int(spot_batch), n_idx))
            ps = jax.checkpoint(
                self._marginal_per_spot_r,
                static_argnums=(0, 4, 7))(
                type_key, idx, r_ang, log_w_r,
                has_any_accel, phys_args, phys_kw, batch, None)
            result = result.at[idx].set(ps)

        return result

    def _fixed_phi_per_spot(self, idx, r_ang, phi, has_any_accel,
                            phys_args, phys_kw):
        r_pre = self._r_precompute(
            r_ang, idx, *phys_args, **phys_kw,
            has_any_accel=has_any_accel)
        sin_phi = jnp.sin(phi)
        cos_phi = jnp.cos(phi)
        X, Y = predict_position(
            r_ang, sin_phi, cos_phi, r_pre["x0"], r_pre["y0"],
            r_pre["sin_i"], r_pre["cos_i"],
            r_pre["sin_O"], r_pre["cos_O"])

        ecc2 = r_pre["ecc2"]
        if ecc2 is None:
            V = predict_velocity_los(
                r_ang, sin_phi, cos_phi, r_pre["D"], r_pre["M_BH"],
                r_pre["v_sys"], r_pre["dv_sys"], r_pre["sin_i"])
        else:
            V = predict_velocity_los(
                r_ang, sin_phi, cos_phi, r_pre["D"], r_pre["M_BH"],
                r_pre["v_sys"], r_pre["dv_sys"], r_pre["sin_i"], ecc2=ecc2,
                ecc_cos_om=r_pre["ecc_cos_om"], ecc_sin_om=r_pre["ecc_sin_om"])

        nhc = neg_half_chi2_position(
            r_pre["all_x"], r_pre["all_y"], X, Y,
            r_pre["var_x"], r_pre["var_y"])
        nhc = nhc + neg_half_chi2_velocity(
            r_pre["all_v_rel"], V, r_pre["var_v"])
        if has_any_accel:
            A = predict_acceleration_los(
                r_ang, sin_phi, cos_phi,
                r_pre["D"], r_pre["M_BH"], r_pre["sin_i"])
            nhc = nhc + neg_half_chi2_acceleration(
                r_pre["all_a"], A, r_pre["var_a"], r_pre["has_a"])
        return r_pre["lnorm"] + r_pre["lnorm_a"] + nhc

    def _eval_phi_fixed(self, spot_groups, phi, phys_args, phys_kw=None):
        """Per-spot fixed-phi log likelihood for explicit-angle MCMC."""
        if phys_kw is None:
            phys_kw = {}
        dtype = jnp.asarray(phys_args[2]).dtype
        result = jnp.zeros(self.n_spots, dtype=dtype)
        for group in spot_groups:
            type_key, idx, r_ang, _ = group
            if int(idx.shape[0]) == 0:
                continue
            ps = self._fixed_phi_per_spot(
                idx, r_ang, phi[idx],
                self._group_has_any_accel(type_key), phys_args, phys_kw)
            result = result.at[idx].set(ps)
        return result

    def _sum_phi_fixed(self, spot_groups, phi, phys_args, phys_kw=None):
        """Total fixed-phi log likelihood for explicit-angle MCMC."""
        if phys_kw is None:
            phys_kw = {}
        total = jnp.asarray(0.0, dtype=jnp.asarray(phys_args[2]).dtype)
        for group in spot_groups:
            type_key, idx, r_ang, _ = group
            if int(idx.shape[0]) == 0:
                continue
            ps = self._fixed_phi_per_spot(
                idx, r_ang, phi[idx],
                self._group_has_any_accel(type_key), phys_args, phys_kw)
            total = total + jnp.sum(ps)
        return total

    def _sum_phi_marginal(self, spot_groups, phys_args, phys_kw=None,
                          spot_batch=None, remat=True, scan_cache=None):
        """Compute the total phi-marginal log-likelihood.

        This mirrors `_eval_phi_marginal` but avoids scattering group results
        into a full per-spot vector when only the scalar total is needed.

        `remat` wraps each group in `jax.checkpoint` to save memory in the
        backward pass; set it False for gradient-free callers (the DE MAP),
        where rematerialisation only adds overhead.
        """
        if phys_kw is None:
            phys_kw = {}
        total = jnp.asarray(0.0, dtype=jnp.asarray(phys_args[2]).dtype)

        for i, group in enumerate(spot_groups):
            type_key, idx, r_ang, log_w_r = group
            n_idx = int(idx.shape[0])
            if n_idx == 0:
                continue

            has_any_accel = self._group_has_any_accel(type_key)
            batch = (None if spot_batch is None
                     else min(int(spot_batch), n_idx))
            fn = (jax.checkpoint(self._marginal_per_spot_r,
                                 static_argnums=(0, 4, 7))
                  if remat else self._marginal_per_spot_r)
            cache = None if scan_cache is None else scan_cache[i]
            ps = fn(type_key, idx, r_ang, log_w_r,
                    has_any_accel, phys_args, phys_kw, batch, cache)
            total = total + jnp.sum(ps)

        return total

    def phys_from_sample(self, sample):
        """Reconstruct (phys_args, phys_kw, diag) from a single posterior draw.

        `sample` is a dict mapping param name -> scalar or small array.
        If `sample["r_ang"]` is present, it has shape (n_spots,) and should
        be used by the caller directly; this helper does NOT consume it.
        """
        def g(key, default=None):
            if key in sample:
                return float(_np.asarray(sample[key]))
            if default is not None:
                return default
            raise KeyError(f"missing '{key}' in posterior sample")

        H0_ref = float(get_nested(self.config, "model/H0_ref", 73.0))
        h = g("H0", H0_ref) / 100.0

        if "D_A" in sample:
            D_A = g("D_A")
        else:
            D_c = g("D_c")
            z_cosmo = float(self.distance2redshift(
                jnp.atleast_1d(D_c), h=h).squeeze())
            D_A = D_c / (1.0 + z_cosmo)
        if "eta" in sample and (
                "log_MBH" not in sample
                or self.mass_parameterization == "eta"):
            log_MBH = g("eta") + _np.log10(D_A)
        else:
            log_MBH = g("log_MBH")
        M_BH = 10.0 ** (log_MBH - 7.0)

        v_sys = self.v_sys_obs + g("dv_sys", 0.0)

        phys_args = (
            g("x0"), g("y0"),
            D_A, M_BH, v_sys,
            self._r_ang_ref_i, self._r_ang_ref_Omega,
            self._r_ang_ref_periapsis,
            _np.deg2rad(g("i0")),
            _np.deg2rad(g("di_dr")),
            _np.deg2rad(g("Omega0")),
            _np.deg2rad(g("dOmega_dr")),
            g("sigma_x_floor") ** 2,
            g("sigma_y_floor") ** 2,
            g("sigma_v_sys") ** 2,
            g("sigma_v_hv") ** 2,
            g("sigma_a_floor") ** 2,
        )

        phys_kw = {"dv_sys": g("dv_sys", 0.0)}
        if self.use_quadratic_warp:
            phys_kw["d2i_dr2"] = _np.deg2rad(g("d2i_dr2", 0.0))
            phys_kw["d2Omega_dr2"] = _np.deg2rad(g("d2Omega_dr2", 0.0))
        if self.use_ecc:
            phys_kw["e_x"] = g("e_x", 0.0)
            phys_kw["e_y"] = g("e_y", 0.0)
            phys_kw["dperiapsis_dr"] = _np.deg2rad(g("dperiapsis_dr", 0.0))

        diag = dict(D_A=D_A, M_BH=M_BH, v_sys=v_sys)
        return phys_args, phys_kw, diag

    def phys_from_params_jax(self, par, h):
        """JAX-traceable (phys_args, phys_kw) from constrained values.

        `par` maps site name -> jnp scalar. This mirrors the deterministic
        transformations used by the BlackJAX megamaser target.
        """
        missing = object()

        def g(site, prior_key=None, default=missing):
            if site in par:
                return par[site]
            prior = self.priors.get(prior_key or site)
            if isinstance(prior, Delta):
                return jnp.asarray(prior.v)
            if default is not missing:
                return jnp.asarray(default)
            raise KeyError(f"missing '{site}' in HMC sites")

        if "D_A" in par:
            D_A = g("D_A", "D")
        else:
            D_c = g("D_c", "D")
            z_cosmo = self.distance2redshift(
                jnp.atleast_1d(D_c), h=h).squeeze()
            D_A = D_c / (1.0 + z_cosmo)
        if "eta" in par and (
                "log_MBH" not in par
                or self.mass_parameterization == "eta"):
            log_MBH = g("eta") + jnp.log10(D_A)
        else:
            log_MBH = g("log_MBH")
        M_BH = 10.0 ** (log_MBH - 7.0)
        v_sys = self.v_sys_obs + g("dv_sys", default=0.0)

        phys_args = (
            g("x0"), g("y0"), D_A, M_BH, v_sys,
            self._r_ang_ref_i, self._r_ang_ref_Omega,
            self._r_ang_ref_periapsis,
            jnp.deg2rad(g("i0")), jnp.deg2rad(g("di_dr")),
            jnp.deg2rad(g("Omega0")), jnp.deg2rad(g("dOmega_dr")),
            g("sigma_x_floor") ** 2, g("sigma_y_floor") ** 2,
            g("sigma_v_sys") ** 2, g("sigma_v_hv") ** 2,
            g("sigma_a_floor") ** 2)

        phys_kw = {"dv_sys": g("dv_sys", default=0.0)}
        if self.use_quadratic_warp:
            phys_kw["d2i_dr2"] = jnp.deg2rad(g("d2i_dr2", default=0.0))
            phys_kw["d2Omega_dr2"] = jnp.deg2rad(
                g("d2Omega_dr2", default=0.0))
        if self.use_ecc:
            # Carry eccentricity Cartesian (e_x, e_y); never sqrt/arctan2 the
            # sampled components -- polar magnitude/direction is singular at
            # e=0 and its gradient poisons the whole near-circular region.
            phys_kw["e_x"] = g("e_x", default=0.0)
            phys_kw["e_y"] = g("e_y", default=0.0)
            phys_kw["dperiapsis_dr"] = jnp.deg2rad(
                g("dperiapsis_dr", default=0.0))
        return phys_args, phys_kw

    def __call__(self):
        raise RuntimeError(
            "Megamaser NumPyro inference has been removed; use "
            "scripts/megamaser/run_maser.py.")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _trapz_log_w_per_spot(r):
    """Per-spot trapezoidal log-weights for non-uniform r grids."""
    h = jnp.diff(r, axis=-1)
    N = r.shape[0]
    zeros = jnp.zeros((N, 1), dtype=r.dtype)
    h_left = jnp.concatenate([zeros, h], axis=-1)
    h_right = jnp.concatenate([h, zeros], axis=-1)
    w = (h_left + h_right) / 2
    return jnp.log(jnp.maximum(w, W_LOG_FLOOR))
