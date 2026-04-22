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
"""Megamaser disk forward model for H0 inference.

Implements the warped Keplerian disk model from Pesce et al. (2020),
arXiv:2001.04581. All JAX functions are JIT-compilable and auto-differentiable.

Phi integration (identical for Mode 1 and Mode 2):
  - Red HV (φ=+π/2 peak): 3 uniform sub-ranges around +π/2
        (N_low, N_high, N_low points). Trapezoidal in each.
  - Blue HV (φ=-π/2 peak): mirror of red, centred on -π/2.
  - Systemic (φ=0 and φ=π): configurable list of uniform sub-ranges
        (default [[-45°, 45°], [135°, 225°]], N_sys points each).
  - Sub-ranges within a type are disjoint → combined in log-space via
    logsumexp over sub-integrals.
  Knobs in [model] config: phi_hv_inner_deg, phi_hv_outer_deg,
    n_phi_hv_high, n_phi_hv_low, phi_sys_ranges_deg, n_phi_sys.

R integration (Mode 2 only):
  - HV and systemic-with-accel-measurement: per-spot sinh-spaced grid in
    log-r centred on a physics-based r estimate; N_r_local points.
  - Systemic-without-accel-measurement: log-uniform r grid spanning the
    full [R_phys_lo, R_phys_hi] range with N_r_brute points.
  Knobs: n_r_local, n_r_brute, K_sigma.

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
from jax.scipy.stats import norm as jax_norm
from numpyro import deterministic, factor, handlers, plate, sample
from numpyro.distributions import ImproperUniform, Uniform, VonMises
from numpyro.distributions import constraints as dist_constraints

from ..util import SPEED_OF_LIGHT, fprint, fsection, get_nested
from .base_model import ModelBase
from .integration import ln_trapz_precomputed, trapz_log_weights
from .pv_utils import rsample
from .utils import normal_logpdf_var

# -----------------------------------------------------------------------
# Disk physics constants
# -----------------------------------------------------------------------

# Internal units: M_BH in 1e7 M_sun, sky positions in μas,
# r_ang in mas, D in Mpc.
C_v = 2978.8656    # km/s: sqrt(G * 1e7 M_sun / (1 mas * 1 Mpc))
C_a = 1.872e3      # km/s/yr: 1e7 M_sun * G * yr / (1 mas * 1 Mpc)^2
C_g = 1.974e-4     # dimensionless: 2*G * 1e7 M_sun / (c^2 * 1 mas * 1 Mpc)
LOG_2PI = 1.8378770664093453  # jnp.log(2 * pi), precomputed

# Conversion: 1 mas at 1 Mpc = 4.848e-3 pc
PC_PER_MAS_MPC = 4.848e-3


# -----------------------------------------------------------------------
# Disk physics functions
# -----------------------------------------------------------------------


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


def predict_position(r_ang, phi, x0, y0, i, Omega):
    """Predict sky-plane position of maser spots. Mock-generator use."""
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    sin_O = jnp.sin(Omega)
    cos_O = jnp.cos(Omega)
    cos_i = jnp.cos(i)

    R = r_ang * 1e3  # mas → μas for position projection
    X = x0 + R * (sin_phi * sin_O - cos_phi * cos_O * cos_i)
    Y = y0 + R * (sin_phi * cos_O + cos_phi * sin_O * cos_i)
    return X, Y


def predict_velocity_los(r_ang, phi, D, M_BH, v_sys, i, ecc=0.0, omega=0.0):
    """Predict line-of-sight velocity (optical convention). Mock use."""
    v_kep = C_v * jnp.sqrt(M_BH / (r_ang * D))

    cos_d = jnp.cos(phi - omega)
    sin_d = jnp.sin(phi - omega)
    E = jnp.sqrt(1.0 + ecc * cos_d)
    v_r = v_kep * ecc * sin_d / E
    v_t = v_kep * E
    v_z = jnp.sin(i) * (v_t * jnp.sin(phi) - v_r * jnp.cos(phi))

    beta_c2 = (v_kep / SPEED_OF_LIGHT)**2
    beta_e2 = (beta_c2 * (1.0 + ecc**2 + 2.0 * ecc * cos_d)
               / (1.0 + ecc * cos_d))
    gamma = 1.0 / jnp.sqrt(1.0 - beta_e2)

    one_plus_z_D = gamma * (1.0 + v_z / SPEED_OF_LIGHT)
    # Floor prevents 1/sqrt(0)=inf for sub-Schwarzschild radii; the
    # resulting large velocity residual drives chi2 → ∞, down-weighting.
    one_plus_z_g = 1.0 / jnp.sqrt(
        jnp.maximum(1.0 - C_g * M_BH / (r_ang * D), 1e-6))
    z_0 = v_sys / SPEED_OF_LIGHT

    return SPEED_OF_LIGHT * (one_plus_z_D * one_plus_z_g * (1.0 + z_0) - 1.0)


def predict_acceleration_los(r_ang, phi, D, M_BH, i):
    """Predict LOS centripetal acceleration. Mock use."""
    a_mag = C_a * M_BH / (r_ang**2 * D**2)
    return a_mag * jnp.cos(phi) * jnp.sin(i)


def _precompute_r_quantities(r_ang, D, M_BH, sin_i, cos_i, sin_O, cos_O):
    """Precompute r-dependent quantities for the phi integration."""
    rD = r_ang * D
    v_kep = C_v * jnp.sqrt(M_BH / rD)
    beta = v_kep / SPEED_OF_LIGHT
    # Unphysical-proposal guards: clamp sqrt arguments to prevent 1/sqrt(0)=inf
    # cascading to chi2=inf and log_f=-inf in the phi marginal.  The physical
    # regimes where these clip (sub-Schwarzschild r, v_kep → c) are well below
    # the prior lower bound; NUTS never stays there.
    gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta * beta, 1e-6))
    one_plus_z_g = 1.0 / jnp.sqrt(jnp.maximum(1.0 - C_g * M_BH / rD, 1e-6))
    a_mag = v_kep * v_kep / rD * (C_a / (C_v * C_v))

    # Position projection coefficients (Reid phi convention):
    # X = x0 + R·(sin_phi·pA + cos_phi·pB),
    # Y = y0 + R·(sin_phi·pC + cos_phi·pD).
    pA = sin_O
    pB = -cos_O * cos_i
    pC = cos_O
    pD = sin_O * cos_i

    return v_kep, gamma, one_plus_z_g, a_mag, pA, pB, pC, pD


def _observables_from_precomputed(sin_phi, cos_phi, x0, y0, v_sys,
                                  sin_i, r_ang,
                                  v_kep, gamma, one_plus_z_g, a_mag,
                                  pA, pB, pC, pD):
    """Compute observables using precomputed r-dependent quantities."""
    R = r_ang * 1e3
    X = x0 + R * (sin_phi * pA + cos_phi * pB)
    Y = y0 + R * (sin_phi * pC + cos_phi * pD)

    v_z = v_kep * sin_phi * sin_i
    one_plus_z_D = gamma * (1.0 + v_z / SPEED_OF_LIGHT)
    V = SPEED_OF_LIGHT * (
        one_plus_z_D * one_plus_z_g * (1.0 + v_sys / SPEED_OF_LIGHT)
        - 1.0)

    A = a_mag * cos_phi * sin_i

    return X, Y, V, A


# -----------------------------------------------------------------------
# Model class
# -----------------------------------------------------------------------


class MaserDiskModel(ModelBase):
    """Megamaser disk H0 model with phi [+ r] marginalisation.

    Mode 1 (marginalise_r=False): sample per-spot r_ang; marginalise φ
      with uniform trapezoidal grids per sub-range.
    Mode 2 (marginalise_r=True): marginalise both r and φ. HV and
      sys-with-accel spots use sinh r-grids of n_r_local points; sys
      spots without accel measurement use a log-uniform r-grid of
      n_r_brute points.
    The φ grid structure (sub-ranges and point counts) is identical
    between the two modes — only the r step differs.
    """

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
        self._print_summary()

    # ---- priors / per-galaxy ----

    def _resolve_per_galaxy_priors(self, data):
        """Set per-galaxy D prior from data dict."""
        self._D_c_volume = False
        if "D_lo" in data and "D_hi" in data:
            lo, hi = float(data["D_lo"]), float(data["D_hi"])
            self.priors["D"] = Uniform(lo, hi)
            D_prior_type = get_nested(
                self.config, "model/D_c_prior", "uniform")
            self._D_c_volume = D_prior_type == "volume"
            fprint(f"D prior: {'volume' if self._D_c_volume else 'uniform'}"
                   f"({lo:.1f}, {hi:.1f})")

    # ---- spot indexing ----

    def _build_spot_indices(self, data):
        """Build spot type index arrays.

        Returns the accel_measured numpy array. Also populates:
            _idx_sys, _idx_red, _idx_blue      — per type
            _idx_sys_cons    — sys WITH accel (sinh r-grid)
            _idx_sys_uncons  — sys WITHOUT accel (brute r-grid)
        """
        is_hv_np = _np.asarray(data["is_highvel"])
        is_blue_np = _np.asarray(data.get("is_blue", _np.zeros(
            self.n_spots, dtype=bool)))
        is_red_np = is_hv_np & ~is_blue_np
        is_sys_np = ~is_hv_np

        accel_meas = _np.asarray(data.get(
            "accel_measured", self.sigma_a < 1e4))

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
        #   - sys (Mode 1 combines cons+uncons): any sys spot w/ accel
        #   - sys_cons: True by construction (accel_measured=True)
        #   - sys_uncons: False by construction
        self._group_has_accel = {
            "sys": bool((is_sys_np & accel_meas).any()),
            "sys_cons": True if self._n_sys_cons > 0 else False,
            "sys_uncons": False,
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
        self._reject_legacy_keys(gal_cfg)
        self._configure_features(gal_cfg)
        self._configure_mode(gal_cfg)
        self._configure_warp_pivots(data, gal_cfg)
        return gal_cfg

    @staticmethod
    def _reject_legacy_keys(gal_cfg):
        legacy = {
            "adaptive_phi": "removed (Mode 1 uses the same unified φ grid "
                            "as Mode 2).",
            "phi_method": "removed (single unified φ method).",
            "n_phi_red": "replaced by n_phi_hv_high / n_phi_hv_low.",
            "n_phi_blue": "replaced by n_phi_hv_high / n_phi_hv_low.",
            "phi_range_red_deg": "replaced by phi_hv_inner_deg / "
                                 "phi_hv_outer_deg.",
            "phi_range_blue_deg": "replaced by phi_hv_inner_deg / "
                                  "phi_hv_outer_deg.",
            "phi_range_sys_deg": "renamed to phi_sys_ranges_deg.",
            "adaptive_r": "removed — adaptive sinh is always used for "
                          "spots with a constrained r (HV and sys+accel).",
            "G_phi_half": "removed — HV grid is now 3 uniform sub-ranges.",
            "n_inner_sys": "removed — systemic grid uses n_phi_sys per "
                           "sub-range from phi_sys_ranges_deg.",
            "n_wing_sys": "removed.",
            "inner_deg_sys": "removed.",
            "n_r": "removed — Mode 2 r-grid is per-spot (n_r_local, "
                   "n_r_brute).",
            "marginalise_r": "replaced by `mode = 'mode0' | 'mode1' | "
                             "'mode2'`.",
        }
        for key, msg in legacy.items():
            if key in gal_cfg:
                raise ValueError(f"legacy config key '{key}': {msg}")

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

    def _configure_mode(self, gal_cfg):
        """Resolve the sampling mode for this galaxy.

        mode = "mode0": sample per-spot r AND phi (no marginalisation).
        mode = "mode1": sample per-spot r, marginalise phi numerically.
        mode = "mode2": marginalise both r and phi numerically.
        """
        valid = ("mode0", "mode1", "mode2")
        mode_global = get_nested(self.config, "model/mode", "mode2")
        mode = gal_cfg.get("mode", mode_global)
        if mode not in valid:
            raise ValueError(
                f"Invalid mode '{mode}'; expected one of {valid}.")
        if mode == "mode2" and gal_cfg.get("forbid_marginalise_r", False):
            raise ValueError(
                "This galaxy has `forbid_marginalise_r = true`; mode2 is "
                "not supported. Set mode = 'mode0' or 'mode1'.")
        self.mode = mode
        self.marginalise_r = (mode == "mode2")
        self.marginalise_phi = (mode in ("mode1", "mode2"))

    def _configure_warp_pivots(self, data, gal_cfg):
        r_common = gal_cfg.get("r_ang_ref", None)
        if r_common is not None:
            r_base = float(r_common)
            base_src = "config"
        else:
            is_hv_np = _np.asarray(data["is_highvel"])
            x_hv = _np.asarray(data["x"])[is_hv_np]
            y_hv = _np.asarray(data["y"])[is_hv_np]
            if x_hv.size == 0:
                raise ValueError(
                    "_configure_warp_pivots: no HV spots and r_ang_ref not "
                    "set in gal_cfg; cannot derive pivot radius.")
            r_ang_hv = _np.sqrt(x_hv**2 + y_hv**2) / 1e3  # μas → mas
            r_base = float(_np.median(r_ang_hv))
            base_src = f"median projected radius of {x_hv.size} HV spots"

        self._r_ang_ref_i = float(gal_cfg.get("r_ang_ref_i", r_base))
        self._r_ang_ref_Omega = float(gal_cfg.get("r_ang_ref_Omega", r_base))
        self._r_ang_ref_periapsis = float(
            gal_cfg.get("r_ang_ref_periapsis", r_base / 2.0))
        self._r_ang_ref = r_base

        overrides = [k for k in ("r_ang_ref_i", "r_ang_ref_Omega",
                                 "r_ang_ref_periapsis") if k in gal_cfg]
        if overrides:
            extras = " ".join(
                f"{k}={float(gal_cfg[k]):.3f}" for k in overrides)
            fprint(f"r_ang_ref = {r_base:.3f} mas ({base_src}); "
                   f"overrides: {extras}")
        else:
            fprint(f"r_ang_ref = {r_base:.3f} mas ({base_src})")

    # ---- phi sub-ranges (single source of truth for both modes) ----

    def _build_phi_subranges(self, gal_cfg):
        """Parse per-type φ sub-ranges from config.

        Stores self._phi_subranges as a dict mapping spot-type ("red",
        "blue", "sys") to a list of (lo_rad, hi_rad, n_phi) triplets.
        Each sub-range is evaluated with a uniform linspace and
        trapezoidal weights; disjoint sub-ranges for the same spot type
        are combined in log-space via logsumexp.
        """
        mode_suffix = "mode2" if self.marginalise_r else "mode1"

        def _get(key, default):
            if key in gal_cfg:
                return gal_cfg[key]
            # Mode-specific global (e.g. n_phi_hv_high_mode1) takes
            # precedence over the generic key when set.
            mode_val = get_nested(
                self.config, f"model/{key}_{mode_suffix}", None)
            if mode_val is not None:
                return mode_val
            return get_nested(self.config, f"model/{key}", default)

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
        pi = _np.pi
        pi2 = pi / 2.0

        # Red (peak at φ=+π/2)
        red = [
            (pi2 - o_rad, pi2 - i_rad, n_low),
            (pi2 - i_rad, pi2 + i_rad, n_high),
            (pi2 + i_rad, pi2 + o_rad, n_low),
        ]
        # Blue (peak at φ=-π/2)
        blue = [
            (-pi2 - o_rad, -pi2 - i_rad, n_low),
            (-pi2 - i_rad, -pi2 + i_rad, n_high),
            (-pi2 + i_rad, -pi2 + o_rad, n_low),
        ]

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

        # Precompute concatenated (sin, cos, log-trapz-weights) per type
        # so _eval_phi_marginal can do one _phi_eval + one logsumexp per
        # group. Disjoint sub-range trapezoidal weights concatenate
        # cleanly: each endpoint retains its h/2 weight, interior h.
        self._phi_concat = {}
        for key, subs in self._phi_subranges.items():
            sin_parts, cos_parts, w_parts = [], [], []
            for lo, hi, n in subs:
                phi = jnp.linspace(lo, hi, n)
                sin_parts.append(jnp.sin(phi))
                cos_parts.append(jnp.cos(phi))
                w_parts.append(trapz_log_weights(phi))
            self._phi_concat[key] = dict(
                sin_phi=jnp.concatenate(sin_parts),
                cos_phi=jnp.concatenate(cos_parts),
                log_w_phi=jnp.concatenate(w_parts),
            )

        # Save for summary/printing
        self._phi_hv_inner_deg = hv_inner_deg
        self._phi_hv_outer_deg = hv_outer_deg
        self._n_phi_hv_high = n_high
        self._n_phi_hv_low = n_low
        self._phi_sys_ranges_deg = sys_ranges_deg
        self._n_phi_sys = n_sys

        # Per-spot phi-prior bounds for Mode 0. Each spot has up to two
        # disjoint sub-ranges (lo0..hi0) and (lo1..hi1); for spots with
        # only one allowed range (red/blue), the second has zero width.
        # An auxiliary u ∈ [0, 1] is mapped to phi deterministically via
        # _phi_from_u, yielding a uniform prior on the union.
        pi = _np.pi
        lo0 = _np.zeros(self.n_spots)
        hi0 = _np.zeros(self.n_spots)
        lo1 = _np.zeros(self.n_spots)
        hi1 = _np.zeros(self.n_spots)
        # Red: contiguous [-outer, outer] (the 3 integration sub-ranges
        # union cleanly into one interval). For single-range spots we
        # set lo1 = hi1 = hi0 so that _phi_from_u at u=1 (which falls
        # into the second branch by a zero-width margin) returns hi0
        # rather than 0.
        if self._n_red > 0:
            idx_red = _np.asarray(self._idx_red)
            lo0[idx_red] = pi2 - o_rad
            hi0[idx_red] = pi2 + o_rad
            lo1[idx_red] = pi2 + o_rad
            hi1[idx_red] = pi2 + o_rad
        if self._n_blue > 0:
            idx_blue = _np.asarray(self._idx_blue)
            lo0[idx_blue] = -pi2 - o_rad
            hi0[idx_blue] = -pi2 + o_rad
            lo1[idx_blue] = -pi2 + o_rad
            hi1[idx_blue] = -pi2 + o_rad
        if self._n_sys > 0:
            if len(sys_rng) == 1:
                (s_lo, s_hi, _) = sys_rng[0]
                idx_sys = _np.asarray(self._idx_sys)
                lo0[idx_sys] = s_lo
                hi0[idx_sys] = s_hi
            elif len(sys_rng) == 2:
                (s0_lo, s0_hi, _), (s1_lo, s1_hi, _) = sys_rng
                idx_sys = _np.asarray(self._idx_sys)
                lo0[idx_sys] = s0_lo
                hi0[idx_sys] = s0_hi
                lo1[idx_sys] = s1_lo
                hi1[idx_sys] = s1_hi
            else:
                raise ValueError(
                    "Mode 0 supports at most 2 systemic phi sub-ranges; "
                    f"got {len(sys_rng)}.")
        self._phi_spot_lo0 = jnp.asarray(lo0)
        self._phi_spot_hi0 = jnp.asarray(hi0)
        self._phi_spot_lo1 = jnp.asarray(lo1)
        self._phi_spot_hi1 = jnp.asarray(hi1)

    def _phi_from_u(self, u):
        """Map per-spot auxiliary u ∈ [0, 1] to phi_i respecting the
        same sub-range union used by the Mode 1/2 integrator."""
        w0 = self._phi_spot_hi0 - self._phi_spot_lo0
        w1 = self._phi_spot_hi1 - self._phi_spot_lo1
        total = w0 + w1
        s = u * total
        in_first = s < w0
        return jnp.where(
            in_first,
            self._phi_spot_lo0 + s,
            self._phi_spot_lo1 + (s - w0))

    # ---- r config (Mode 2 grids + Mode 1 r_ang bounds) ----

    def _build_r_config(self, data, gal_cfg):
        """Set r-grid constants and Mode 1 r_ang prior bounds."""
        def _get(key, default):
            if key in gal_cfg:
                return gal_cfg[key]
            return get_nested(self.config, f"model/{key}", default)

        _R_lo = float(_get("R_phys_lo", 0.01))
        _R_hi = float(_get("R_phys_hi", 2.0))
        self._R_phys_lo = _R_lo
        self._R_phys_hi = _R_hi

        # Mode 2 r-grid knobs.
        self._n_r_local = int(_get("n_r_local", 101))
        self._n_r_brute = int(_get("n_r_brute", 501))
        self._K_sigma = float(_get("K_sigma", 5.0))
        if self._n_r_local < 3 or self._n_r_brute < 3:
            raise ValueError("n_r_local and n_r_brute must be >= 3.")

        # Selection function grid (same for all modes/galaxies).
        D_min = float(get_nested(self.config, "model/priors/D/low", 10.0))
        D_max = float(get_nested(self.config, "model/priors/D/high", 200.0))
        self._sel_D_grid = jnp.linspace(D_min, D_max, 501)
        self._sel_log_w = jnp.asarray(trapz_log_weights(self._sel_D_grid))
        self._sel_lp_vol = 2.0 * jnp.log(self._sel_D_grid)
        self.use_selection = get_nested(
            self.config, "model/use_selection", False)

        # Fixed r_ang bounds for Mode 1 prior (cosmographic D_A at z_sys).
        z_est = self.v_sys_obs / SPEED_OF_LIGHT
        q0 = -0.55
        D_c_est = (SPEED_OF_LIGHT * z_est / 73.0
                   * (1 + 0.5 * (1 - q0) * z_est))
        D_A_est = D_c_est / (1 + z_est)
        self._r_ang_lo = self._R_phys_lo / (D_A_est * PC_PER_MAS_MPC)
        self._r_ang_hi = self._R_phys_hi / (D_A_est * PC_PER_MAS_MPC)

    # ---- summary ----
    def _print_summary(self):
        mode_desc = {
            "mode0": "sample r, sample φ",
            "mode1": "sample r, marginalise φ",
            "mode2": "marginalise r+φ",
        }[self.mode]
        fprint(f"mode: {self.mode} ({mode_desc})")
        if self.marginalise_phi:
            fprint(
                f"φ HV: inner ±{self._phi_hv_inner_deg:.0f}° "
                f"(n={self._n_phi_hv_high}), outer wings to "
                f"±{self._phi_hv_outer_deg:.0f}° "
                f"(n={self._n_phi_hv_low} per wing)")
            rng_str = " ∪ ".join(
                f"[{lo:.0f}°, {hi:.0f}°]"
                for lo, hi in self._phi_sys_ranges_deg)
            fprint(f"φ sys: {rng_str}, n={self._n_phi_sys} per sub-range")
        if self.marginalise_r:
            fprint(
                f"r grid: n_r_local={self._n_r_local} (sinh), "
                f"n_r_brute={self._n_r_brute} (log-uniform, "
                f"{self._n_sys_uncons} sys-no-accel spots), "
                f"K={self._K_sigma}")
        else:
            fprint(
                f"r_ang prior bounds: "
                f"[{self._r_ang_lo:.3f}, {self._r_ang_hi:.3f}] mas")

    # ---- r estimation (per-spot physics-based centre + scale) ----

    def _estimate_adaptive_r(self, D_A, M_BH, v_sys, sigma_a_floor2,
                             i0, var_v_hv):
        """Per-spot r_ang centre and sinh-grid scale.

        HV:    velocity → r_vel = M·(C_v·sin_i)² / (D·Δv²)
        sys+a: acceleration → r_acc = √(C_a·M·sin_i / (D²·|a|))
        sys-a: (unused here — sys-without-accel go through the brute grid)
        Returns (r_est, scale) with broadcasting to all spots.
        """
        conv = D_A * PC_PER_MAS_MPC
        r_min = self._R_phys_lo / conv
        r_max = self._R_phys_hi / conv

        sin_i = jnp.abs(jnp.sin(i0))
        eps = 1e-30

        dv = self._all_v - v_sys
        r_vel = M_BH * (C_v * sin_i) ** 2 / (D_A * (dv ** 2 + eps))
        r_vel = jnp.clip(r_vel, r_min, r_max)

        r_acc = jnp.sqrt(
            C_a * M_BH * sin_i / (D_A ** 2 * (jnp.abs(self._all_a) + eps)))
        r_acc = jnp.clip(r_acc, r_min, r_max)

        # Centering: HV → r_vel, sys → r_acc (only sys+accel goes through
        # this path; sys-no-accel uses the brute-force log-uniform grid).
        r_est = jnp.where(self.is_highvel, r_vel, r_acc)
        r_est = jnp.clip(r_est, r_min * 1.01, r_max * 0.99)

        # Scale: set half-width in log-r to the propagated measurement
        # error, floored to avoid pathologically narrow grids.
        sigma_v_eff = jnp.sqrt(var_v_hv)
        sigma_a_eff = jnp.sqrt(sigma_a_floor2)
        sigma_log_vel = 2.0 * sigma_v_eff / (jnp.abs(dv) + eps)
        sigma_log_acc = sigma_a_eff / (2.0 * jnp.abs(self._all_a) + eps)
        scale = jnp.where(
            self.is_highvel,
            jnp.maximum(sigma_log_vel, 0.05),
            jnp.maximum(sigma_log_acc, 0.1))
        return r_est, scale, r_min, r_max

    # ---- r grids for Mode 2 ----

    def _build_r_grids_mode2(self, D_A, M_BH, v_sys, sigma_a_floor2,
                             i0, var_v_hv):
        """Return list of (type_key, idx, r_ang, log_w_r) for all spot
        groups. Constrained spots (HV, sys+accel) get per-spot sinh grids
        with n_r_local points centred on their physics-based r_est; sys
        spots without accel measurement get a log-uniform r grid with
        n_r_brute points spanning the full [R_phys_lo, R_phys_hi] range.
        """
        r_est, scale, r_min, r_max = self._estimate_adaptive_r(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)

        def _sinh_grid(idx, n_r):
            r_c = r_est[idx]
            s = scale[idx]
            logr_c = jnp.log(r_c)
            logr_lo = jnp.log(r_min)
            logr_hi = jnp.log(r_max)
            t_lo = jnp.arcsinh((logr_lo - logr_c) / s)
            t_hi = jnp.arcsinh((logr_hi - logr_c) / s)
            u = jnp.linspace(0.0, 1.0, n_r)
            t = t_lo[:, None] + (t_hi - t_lo)[:, None] * u[None, :]
            r = jnp.exp(logr_c[:, None] + jnp.sinh(t) * s[:, None])
            return r, _trapz_log_w_per_spot(r)

        def _loguniform_grid_shared(n_r):
            # Shared across all spots in the group: no broadcast to (N, n_r).
            logr = jnp.linspace(jnp.log(r_min), jnp.log(r_max), n_r)
            r1 = jnp.exp(logr)
            w1 = trapz_log_weights(r1)
            return r1, w1

        # Grid positions are a numerical quadrature tool, not part of the
        # model — stop_gradient prevents NaN gradients from flowing back
        # through the adaptive grid construction (r_acc path overflows in
        # float32 for spots with a=0).  Integrand gradients are unaffected.
        def _sg(r, lw):
            return jax.lax.stop_gradient(r), jax.lax.stop_gradient(lw)

        groups = []
        # Systemic with accel → sinh centred on r_acc (per-spot).
        if self._n_sys_cons > 0:
            r, lw = _sg(*_sinh_grid(self._idx_sys_cons, self._n_r_local))
            groups.append(("sys", self._idx_sys_cons, r, lw, False))
        # Systemic without accel → log-uniform brute (shared r).
        if self._n_sys_uncons > 0:
            r1, w1 = _sg(*_loguniform_grid_shared(self._n_r_brute))
            groups.append(("sys", self._idx_sys_uncons, r1, w1, True))
        # Red HV → sinh centred on r_vel (per-spot).
        if self._n_red > 0:
            r, lw = _sg(*_sinh_grid(self._idx_red, self._n_r_local))
            groups.append(("red", self._idx_red, r, lw, False))
        # Blue HV → sinh centred on r_vel (per-spot).
        if self._n_blue > 0:
            r, lw = _sg(*_sinh_grid(self._idx_blue, self._n_r_local))
            groups.append(("blue", self._idx_blue, r, lw, False))
        return groups

    # ---- unified φ integrand (1-D or 2-D r_ang) ----

    def _r_precompute(self, r_ang, idx,
                      x0, y0, D_A, M_BH, v_sys,
                      r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
                      i0, di_dr, Omega0, dOmega_dr,
                      sigma_x_floor2, sigma_y_floor2,
                      var_v_sys, var_v_hv, sigma_a_floor2,
                      d2i_dr2=0.0, d2Omega_dr2=0.0,
                      ecc=None, periapsis0=None, dperiapsis_dr=0.0,
                      has_any_accel=True):
        """Precompute r-only quantities and gather per-spot data.

        Returned pytree is consumed by _phi_eval (or _phi_eval_shared_r).
        r_ang accepts (N,) [Mode 1], (N, n_r) [Mode 2 per-spot], or
        (n_r,) [Mode 2 shared-r, sys-uncons only].
        """
        all_x = self._all_x[idx]
        all_y = self._all_y[idx]
        all_v = self._all_v[idx]
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
        sin_i = jnp.sin(i_r)
        cos_i = jnp.cos(i_r)
        sin_O = jnp.sin(Om_r)
        cos_O = jnp.cos(Om_r)
        v_kep, gamma, z_g, a_mag, pA, pB, pC, pD = \
            _precompute_r_quantities(
                r_ang, D_A, M_BH, sin_i, cos_i, sin_O, cos_O)

        var_x = sx2 + sigma_x_floor2
        var_y = sy2 + sigma_y_floor2
        var_v = sv2 + jnp.where(is_hv, var_v_hv, var_v_sys)
        lnorm = -0.5 * (3 * LOG_2PI + jnp.log(var_x) +
                        jnp.log(var_y) + jnp.log(var_v))
        if has_any_accel:
            var_a = sa2 + sigma_a_floor2
            lnorm_a = -0.5 * (LOG_2PI + jnp.log(var_a)) * has_a
        else:
            var_a = None
            lnorm_a = jnp.zeros_like(lnorm)

        ecc_pre = None
        if ecc is not None:
            omega_r = (periapsis0
                       + dperiapsis_dr * (r_ang - r_ang_ref_periapsis))
            sw = jnp.sin(omega_r)
            cw = jnp.cos(omega_r)
            beta_c2 = (v_kep / SPEED_OF_LIGHT) ** 2
            ecc_pre = dict(ecc=ecc, sw=sw, cw=cw, beta_c2=beta_c2)

        return dict(
            r_ang=r_ang, sin_i=sin_i,
            v_kep=v_kep, gamma=gamma, z_g=z_g, a_mag=a_mag,
            pA=pA, pB=pB, pC=pC, pD=pD,
            all_x=all_x, all_y=all_y, all_v=all_v, all_a=all_a,
            var_x=var_x, var_y=var_y, var_v=var_v, var_a=var_a,
            has_a=has_a, lnorm=lnorm, lnorm_a=lnorm_a,
            ecc_pre=ecc_pre,
            x0=x0, y0=y0, v_sys=v_sys,
            has_any_accel=has_any_accel,
        )

    def _phi_eval(self, r_pre, sin_phi, cos_phi):
        """Evaluate log-integrand at (r, phi) grid points (per-spot r).

        r_pre   : dict from _r_precompute with per-spot r-prefix shape
                  (N,) [Mode 1] or (N, n_r) [Mode 2 per-spot].
        sin_phi, cos_phi : shape (n_phi,).
        Returns log_f shape (N, [n_r,] n_phi).
        """
        r_ang = r_pre["r_ang"]
        rpad = (slice(None),) * r_ang.ndim + (None,)
        dpad = (slice(None),) + (None,) * r_ang.ndim

        sin_i = r_pre["sin_i"]
        v_kep = r_pre["v_kep"]
        ecc_pre = r_pre["ecc_pre"]

        # Position + acceleration are r-and-phi dependent but do not
        # differ between circular and eccentric orbits in this model.
        R = r_ang[rpad] * 1e3
        X = r_pre["x0"] + R * (sin_phi * r_pre["pA"][rpad]
                               + cos_phi * r_pre["pB"][rpad])
        Y = r_pre["y0"] + R * (sin_phi * r_pre["pC"][rpad]
                               + cos_phi * r_pre["pD"][rpad])
        A = r_pre["a_mag"][rpad] * cos_phi * sin_i[rpad]

        # V branch: compute exactly once (circular OR eccentric).
        if ecc_pre is None:
            v_z = v_kep[rpad] * sin_phi * sin_i[rpad]
            one_plus_z_D = r_pre["gamma"][rpad] * (
                1.0 + v_z / SPEED_OF_LIGHT)
        else:
            ecc = ecc_pre["ecc"]
            sw = ecc_pre["sw"][rpad]
            cw = ecc_pre["cw"][rpad]
            beta_c2 = ecc_pre["beta_c2"][rpad]
            cos_d = cos_phi * cw + sin_phi * sw
            # Same guard class: ecc→1 sends denom→0 at anti-periapsis,
            # and beta_e2 can exceed 1 if ecc is sampled out of bounds.
            denom = jnp.maximum(1.0 + ecc * cos_d, 1e-6)
            ecc_fac = (sin_phi + ecc * sw) / jnp.sqrt(denom)
            v_z = v_kep[rpad] * ecc_fac * sin_i[rpad]
            beta_e2 = (beta_c2 * (1.0 + ecc ** 2 + 2.0 * ecc * cos_d) / denom)
            gamma_e = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta_e2, 1e-6))
            one_plus_z_D = gamma_e * (1.0 + v_z / SPEED_OF_LIGHT)
        V = SPEED_OF_LIGHT * (
            one_plus_z_D * r_pre["z_g"][rpad]
            * (1.0 + r_pre["v_sys"] / SPEED_OF_LIGHT) - 1.0)

        dx = r_pre["all_x"][dpad] - X
        dy = r_pre["all_y"][dpad] - Y
        dv = r_pre["all_v"][dpad] - V
        chi2 = (dx * dx / r_pre["var_x"][dpad]
                + dy * dy / r_pre["var_y"][dpad]
                + dv * dv / r_pre["var_v"][dpad])
        if r_pre["has_any_accel"]:
            da = r_pre["all_a"][dpad] - A
            chi2 = chi2 + (da * da / r_pre["var_a"][dpad]
                           * r_pre["has_a"][dpad])

        return -0.5 * chi2

    def _phi_eval_shared_r(self, r_pre, sin_phi, cos_phi):
        """Shared-r variant of _phi_eval (sys-uncons).

        r_pre is built from a (n_r,)-shaped r_ang (no spot axis).
        Predictions X, Y, V, A are computed at (n_r, n_phi); the
        (N, n_r, n_phi) cost appears only at the residual step.
        Returns log_f of shape (N, n_r, n_phi).
        """
        sin_i = r_pre["sin_i"]                             # (n_r,)
        v_kep = r_pre["v_kep"]                             # (n_r,)
        ecc_pre = r_pre["ecc_pre"]

        # X, Y, A are the same under circular and eccentric orbits.
        R = r_pre["r_ang"][:, None] * 1e3                  # (n_r, 1)
        X = r_pre["x0"] + R * (sin_phi * r_pre["pA"][:, None]
                               + cos_phi * r_pre["pB"][:, None])
        Y = r_pre["y0"] + R * (sin_phi * r_pre["pC"][:, None]
                               + cos_phi * r_pre["pD"][:, None])
        A = r_pre["a_mag"][:, None] * cos_phi * sin_i[:, None]
        # X, Y, A: (n_r, n_phi).

        if ecc_pre is None:
            v_z = v_kep[:, None] * sin_phi * sin_i[:, None]
            one_plus_z_D = r_pre["gamma"][:, None] * (
                1.0 + v_z / SPEED_OF_LIGHT)
        else:
            ecc = ecc_pre["ecc"]
            sw = ecc_pre["sw"][:, None]
            cw = ecc_pre["cw"][:, None]
            beta_c2 = ecc_pre["beta_c2"][:, None]
            cos_d = cos_phi * cw + sin_phi * sw
            # Same guard class: ecc→1 sends denom→0 at anti-periapsis,
            # and beta_e2 can exceed 1 if ecc is sampled out of bounds.
            denom = jnp.maximum(1.0 + ecc * cos_d, 1e-6)
            ecc_fac = (sin_phi + ecc * sw) / jnp.sqrt(denom)
            v_z = v_kep[:, None] * ecc_fac * sin_i[:, None]
            beta_e2 = (beta_c2 * (1.0 + ecc ** 2 + 2.0 * ecc * cos_d) / denom)
            gamma_e = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta_e2, 1e-6))
            one_plus_z_D = gamma_e * (1.0 + v_z / SPEED_OF_LIGHT)
        V = SPEED_OF_LIGHT * (
            one_plus_z_D * r_pre["z_g"][:, None]
            * (1.0 + r_pre["v_sys"] / SPEED_OF_LIGHT) - 1.0)

        # Broadcast to (N, n_r, n_phi) at the residual step.
        dx = r_pre["all_x"][:, None, None] - X[None]
        dy = r_pre["all_y"][:, None, None] - Y[None]
        dv = r_pre["all_v"][:, None, None] - V[None]
        chi2 = (dx * dx / r_pre["var_x"][:, None, None]
                + dy * dy / r_pre["var_y"][:, None, None]
                + dv * dv / r_pre["var_v"][:, None, None])
        if r_pre["has_any_accel"]:
            da = r_pre["all_a"][:, None, None] - A[None]
            chi2 = chi2 + (da * da / r_pre["var_a"][:, None, None]
                           * r_pre["has_a"][:, None, None])
        return -0.5 * chi2

    def _phi_integrand(self, r_ang, sin_phi, cos_phi, idx,
                       x0, y0, D_A, M_BH, v_sys,
                       r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
                       i0, di_dr, Omega0, dOmega_dr,
                       sigma_x_floor2, sigma_y_floor2,
                       var_v_sys, var_v_hv, sigma_a_floor2,
                       d2i_dr2=0.0, d2Omega_dr2=0.0,
                       ecc=None, periapsis0=None, dperiapsis_dr=0.0):
        """Backward-compat wrapper: precompute then evaluate.

        Production Mode 1/2 paths call _r_precompute and _phi_eval
        directly; this wrapper is kept for bruteforce_ll_mode1 and
        Mode 0 sampling.

        r_ang must be 1-D, shape (N,).  lnorm from _r_precompute has
        shape (N,), and rpad = (:, None) appends the phi axis correctly.
        Mode 2 callers with 2-D r_ang use _eval_phi_marginal directly.
        """
        r_pre = self._r_precompute(
            r_ang, idx, x0, y0, D_A, M_BH, v_sys,
            r_ang_ref_i, r_ang_ref_Omega, r_ang_ref_periapsis,
            i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor2, sigma_y_floor2,
            var_v_sys, var_v_hv, sigma_a_floor2,
            d2i_dr2=d2i_dr2, d2Omega_dr2=d2Omega_dr2,
            ecc=ecc, periapsis0=periapsis0, dperiapsis_dr=dperiapsis_dr)
        neg_half_chi2 = self._phi_eval(r_pre, sin_phi, cos_phi)
        rpad = (slice(None),) * r_ang.ndim + (None,)
        return (r_pre["lnorm"] + r_pre["lnorm_a"])[rpad] + neg_half_chi2

    # ---- unified φ [+ r] marginal ----

    def _eval_phi_marginal(self, spot_groups, phys_args, phys_kw=None,
                           spot_batch=None):
        """Compute per-spot log-marginal.

        spot_groups : list of (type_key, idx, r_ang, log_w_r)
            type_key : "red", "blue", or "sys"
            idx      : absolute spot indices, shape (N_group,)
            r_ang    : (N_group,) for Mode 1, (N_group, n_r) for Mode 2
            log_w_r  : None (Mode 1) or (N_group, n_r) (Mode 2)

        The same φ sub-range structure stored in self._phi_subranges is
        used for every group, regardless of mode.

        spot_batch : optional int cap on the (N_batch, [n_r,] n_phi)
            intermediates, for running convergence tests on a small GPU.
        """
        if phys_kw is None:
            phys_kw = {}
        result = jnp.zeros(self.n_spots)

        for group in spot_groups:
            # Accept legacy 4-tuples (no shared_r flag, default False).
            if len(group) == 5:
                type_key, idx, r_ang, log_w_r, shared_r = group
            else:
                type_key, idx, r_ang, log_w_r = group
                shared_r = False
            n_idx = int(idx.shape[0])
            if n_idx == 0:
                continue
            pc = self._phi_concat[type_key]
            batch = (n_idx if spot_batch is None
                     else min(int(spot_batch), n_idx))

            # Resolve per-group accel flag. Mode 1 sys merges cons+uncons
            # into idx_sys so use the broader "sys" key; Mode 2 splits
            # them into shared_r (sys_uncons, never has accel) vs
            # per-spot (sys_cons, always has accel).
            if type_key == "sys":
                accel_key = ("sys_uncons" if shared_r
                             else "sys_cons" if self._n_sys_uncons == 0
                             else "sys")
            else:
                accel_key = type_key
            has_any_accel = self._group_has_accel[accel_key]

            ps_parts = []
            for s in range(0, n_idx, batch):
                sl = slice(s, s + batch)
                b_idx = idx[sl]

                if shared_r:
                    # r_ang shape (n_r,); weights shape (n_r,).
                    r_b = r_ang
                    lwr_b = log_w_r
                    r_pre = self._r_precompute(
                        r_b, b_idx, *phys_args, **phys_kw,
                        has_any_accel=has_any_accel)
                    # _phi_eval_shared_r returns -0.5*chi2; lnorm added after
                    # logsumexp so float32 max-subtraction operates on chi2
                    # differences only (bounded near 0), not on lnorm+chi2.
                    neg_half_chi2 = self._phi_eval_shared_r(
                        r_pre, pc["sin_phi"], pc["cos_phi"])
                    w2d = (lwr_b[None, :, None]
                           + pc["log_w_phi"][None, None, :])
                    lnorm_b = r_pre["lnorm"] + r_pre["lnorm_a"]
                    ps_b = lnorm_b + logsumexp(
                        neg_half_chi2 + w2d, axis=(-2, -1))
                else:
                    r_b = r_ang[sl]
                    lwr_b = None if log_w_r is None else log_w_r[sl]
                    r_pre = self._r_precompute(
                        r_b, b_idx, *phys_args, **phys_kw,
                        has_any_accel=has_any_accel)
                    # Same separation: _phi_eval returns -0.5*chi2 only.
                    neg_half_chi2 = self._phi_eval(
                        r_pre, pc["sin_phi"], pc["cos_phi"])
                    lnorm_b = r_pre["lnorm"] + r_pre["lnorm_a"]
                    if lwr_b is None:
                        ps_b = lnorm_b + logsumexp(
                            neg_half_chi2 + pc["log_w_phi"], axis=-1)
                    else:
                        w2d = (lwr_b[:, :, None]
                               + pc["log_w_phi"][None, None, :])
                        ps_b = lnorm_b + logsumexp(
                            neg_half_chi2 + w2d, axis=(-2, -1))
                ps_parts.append(ps_b)

            ps = (ps_parts[0] if len(ps_parts) == 1
                  else jnp.concatenate(ps_parts, axis=0))
            result = result.at[idx].set(ps)

        return result

    # ---- numpyro sample + log-likelihood ----

    def _sample_galaxy(self, shared_params, h):
        """Sample galaxy-local parameters and add ll_disk factor."""
        D_c = rsample("D_c", self.priors["D"], shared_params)
        if self._D_c_volume:
            factor("D_c_volume", 2 * jnp.log(D_c))

        z_cosmo = self.distance2redshift(
            jnp.atleast_1d(D_c), h=h).squeeze()
        D_A = D_c / (1 + z_cosmo)

        if "sigma_pec" in shared_params:
            sigma_pec = shared_params["sigma_pec"]
            cz_cosmo = SPEED_OF_LIGHT * z_cosmo
            factor("ll_redshift",
                   normal_logpdf_var(
                       cz_cosmo, self.v_sys_obs, sigma_pec**2))

        eta = rsample("eta", self.priors["eta"], shared_params)
        log_MBH = deterministic("log_MBH", eta + jnp.log10(D_A))
        M_BH = 10.0**(log_MBH - 7.0)
        x0 = rsample("x0", self.priors["x0"], shared_params)
        y0 = rsample("y0", self.priors["y0"], shared_params)

        i0_deg = rsample("i0", self.priors["i0"], shared_params)
        Omega0_deg = rsample(
            "Omega0", self.priors["Omega0"], shared_params)
        dOmega_dr_deg = rsample(
            "dOmega_dr", self.priors["dOmega_dr"], shared_params)
        di_dr_deg = rsample(
            "di_dr", self.priors["di_dr"], shared_params)

        i0 = jnp.deg2rad(i0_deg)
        Omega0 = jnp.deg2rad(Omega0_deg)
        dOmega_dr = jnp.deg2rad(dOmega_dr_deg)
        di_dr = jnp.deg2rad(di_dr_deg)

        sigma_x_floor2 = rsample(
            "sigma_x_floor", self.priors["sigma_x_floor"],
            shared_params)**2
        sigma_y_floor2 = rsample(
            "sigma_y_floor", self.priors["sigma_y_floor"],
            shared_params)**2
        var_v_sys = rsample(
            "sigma_v_sys", self.priors["sigma_v_sys"], shared_params)**2
        var_v_hv = rsample(
            "sigma_v_hv", self.priors["sigma_v_hv"], shared_params)**2
        sigma_a_floor2 = rsample(
            "sigma_a_floor", self.priors["sigma_a_floor"],
            shared_params)**2

        dv_sys = rsample("dv_sys", self.priors["dv_sys"], shared_params)
        v_sys = self.v_sys_obs + dv_sys

        # Eccentricity.
        ecc_kw = {}
        if self.use_ecc:
            if self.ecc_cartesian:
                e_x = rsample("e_x", self.priors["e_x"], shared_params)
                e_y = rsample("e_y", self.priors["e_y"], shared_params)
                r2 = e_x**2 + e_y**2
                factor("ecc_cartesian_jac",
                       jnp.log(4.0 / jnp.pi) - 0.5 * jnp.log(r2 + 1e-6))
                ecc = deterministic("ecc", jnp.sqrt(r2))
                periapsis_deg = deterministic(
                    "periapsis",
                    jnp.rad2deg(jnp.arctan2(e_y, e_x)) % 360.0)
            else:
                ecc = rsample("ecc", self.priors["ecc"], shared_params)
                periapsis_rad = rsample(
                    "periapsis_rad", VonMises(0.0, 0.0), shared_params)
                periapsis_deg = deterministic(
                    "periapsis", jnp.rad2deg(periapsis_rad) % 360.0)
            periapsis0 = jnp.deg2rad(periapsis_deg)
            dperiapsis_dr = jnp.deg2rad(rsample(
                "dperiapsis_dr", self.priors["dperiapsis_dr"],
                shared_params))
            ecc_kw = dict(ecc=ecc, periapsis0=periapsis0,
                          dperiapsis_dr=dperiapsis_dr)

        quad_kw = {}
        if self.use_quadratic_warp:
            d2i_dr2_deg = rsample(
                "d2i_dr2", self.priors["d2i_dr2"], shared_params)
            d2Omega_dr2_deg = rsample(
                "d2Omega_dr2", self.priors["d2Omega_dr2"], shared_params)
            quad_kw = dict(d2i_dr2=jnp.deg2rad(d2i_dr2_deg),
                           d2Omega_dr2=jnp.deg2rad(d2Omega_dr2_deg))

        phys_args = (x0, y0, D_A, M_BH, v_sys,
                     self._r_ang_ref_i, self._r_ang_ref_Omega,
                     self._r_ang_ref_periapsis,
                     i0, di_dr, Omega0, dOmega_dr,
                     sigma_x_floor2, sigma_y_floor2, var_v_sys, var_v_hv,
                     sigma_a_floor2)
        phys_kw = {**ecc_kw, **quad_kw}

        if self.mode == "mode0":
            with plate("spots", self.n_spots):
                r_spots = sample(
                    "r_ang",
                    ImproperUniform(dist_constraints.positive, (), ()))
                phi_u = sample("phi_u", Uniform(0.0, 1.0))
            phi_spots = self._phi_from_u(phi_u)
            idx_all = jnp.arange(self.n_spots)
            sin_phi = jnp.sin(phi_spots)[:, None]
            cos_phi = jnp.cos(phi_spots)[:, None]
            log_f = self._phi_integrand(
                r_spots, sin_phi, cos_phi, idx_all,
                *phys_args, **phys_kw)
            ll_per_spot = log_f[:, 0]
            factor("ll_disk", jnp.sum(ll_per_spot))
            return D_c

        if self.marginalise_r:
            spot_groups = self._build_r_grids_mode2(
                D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
        else:
            with plate("spots", self.n_spots):
                r_spots = sample(
                    "r_ang",
                    ImproperUniform(dist_constraints.positive, (), ()))
            spot_groups = []
            if self._n_sys > 0:
                spot_groups.append(
                    ("sys", self._idx_sys,
                     r_spots[self._idx_sys], None))
            if self._n_red > 0:
                spot_groups.append(
                    ("red", self._idx_red,
                     r_spots[self._idx_red], None))
            if self._n_blue > 0:
                spot_groups.append(
                    ("blue", self._idx_blue,
                     r_spots[self._idx_blue], None))

        ll_per_spot = self._eval_phi_marginal(
            spot_groups, phys_args, phys_kw)
        factor("ll_disk", jnp.sum(ll_per_spot))

        return D_c

    def phys_from_sample(self, sample):
        """Reconstruct (phys_args, phys_kw, diag) from a single posterior draw.

        `sample` is a dict mapping param name -> scalar or small array.
        For Mode 1, `sample["r_ang"]` has shape (n_spots,) and should be
        used by the caller directly; this helper does NOT consume it.
        """
        def g(key, default=None):
            if key in sample:
                return float(_np.asarray(sample[key]))
            if default is not None:
                return default
            raise KeyError(f"missing '{key}' in posterior sample")

        H0_ref = float(get_nested(self.config, "model/H0_ref", 73.0))
        h = g("H0", H0_ref) / 100.0

        D_c = g("D_c")
        eta = g("eta")
        z_cosmo = float(self.distance2redshift(
            jnp.atleast_1d(D_c), h=h).squeeze())
        D_A = D_c / (1.0 + z_cosmo)
        M_BH = 10.0 ** (eta + _np.log10(D_A) - 7.0)

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

        phys_kw = {}
        if self.use_quadratic_warp:
            phys_kw["d2i_dr2"] = _np.deg2rad(g("d2i_dr2"))
            phys_kw["d2Omega_dr2"] = _np.deg2rad(g("d2Omega_dr2"))
        if self.use_ecc:
            if "ecc" in sample and "periapsis" in sample:
                phys_kw["ecc"] = g("ecc")
                phys_kw["periapsis0"] = _np.deg2rad(g("periapsis"))
            elif "e_x" in sample and "e_y" in sample:
                e_x = g("e_x")
                e_y = g("e_y")
                phys_kw["ecc"] = float(_np.sqrt(e_x * e_x + e_y * e_y))
                phys_kw["periapsis0"] = float(_np.arctan2(e_y, e_x))
            else:
                raise KeyError(
                    "use_ecc=True but neither 'ecc'/'periapsis' nor "
                    "'e_x'/'e_y' present in sample")
            phys_kw["dperiapsis_dr"] = _np.deg2rad(g("dperiapsis_dr", 0.0))

        diag = dict(D_A=D_A, M_BH=M_BH, v_sys=v_sys)
        return phys_args, phys_kw, diag

    def __call__(self):
        if self.use_selection:
            raise RuntimeError(
                "Selection function must be applied in JointMaserModel, "
                "not MaserDiskModel.")
        H0_ref = float(get_nested(
            self.config, "model/H0_ref", 73.0))
        self._sample_galaxy({}, H0_ref / 100.0)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def remap_warp_to_r0(samples, r_ang_ref_i, r_ang_ref_Omega):
    """Re-express warp parameters as if the pivot were r = 0.

    Parameters
    ----------
    samples : dict
        Posterior samples dict with keys ``i0``, ``di_dr``, ``Omega0``,
        ``dOmega_dr`` (all in degrees / degrees per mas).  Optionally also
        ``d2i_dr2`` and ``d2Omega_dr2`` (degrees per mas^2) for quadratic
        warps.
    r_ang_ref_i, r_ang_ref_Omega : float
        Pivot radii in mas used during sampling (``model._r_ang_ref_i`` and
        ``model._r_ang_ref_Omega``).

    Returns
    -------
    dict with keys ``i0_r0``, ``Omega0_r0``, ``di_dr_r0``, ``dOmega_dr_r0``
    (same units as input).
    """
    i0 = _np.asarray(samples["i0"])
    di_dr = _np.asarray(samples["di_dr"])
    Om0 = _np.asarray(samples["Omega0"])
    dOm_dr = _np.asarray(samples["dOmega_dr"])

    d2i = (_np.asarray(samples["d2i_dr2"])
           if "d2i_dr2" in samples else None)
    d2Om = (_np.asarray(samples["d2Omega_dr2"])
            if "d2Omega_dr2" in samples else None)

    ri, rO = float(r_ang_ref_i), float(r_ang_ref_Omega)

    if d2i is not None:
        i0_r0 = i0 - di_dr * ri + d2i * ri**2
        di_dr_r0 = di_dr - 2.0 * d2i * ri
    else:
        i0_r0 = i0 - di_dr * ri
        di_dr_r0 = di_dr

    if d2Om is not None:
        Om0_r0 = Om0 - dOm_dr * rO + d2Om * rO**2
        dOm_dr_r0 = dOm_dr - 2.0 * d2Om * rO
    else:
        Om0_r0 = Om0 - dOm_dr * rO
        dOm_dr_r0 = dOm_dr

    return dict(i0_r0=i0_r0, Omega0_r0=Om0_r0,
                di_dr_r0=di_dr_r0, dOmega_dr_r0=dOm_dr_r0)


def _trapz_log_w_per_spot(r):
    """Per-spot trapezoidal log-weights for non-uniform (N, n_r) r-grids."""
    h = jnp.diff(r, axis=-1)
    N = r.shape[0]
    h_left = jnp.concatenate([jnp.zeros((N, 1)), h], axis=-1)
    h_right = jnp.concatenate([h, jnp.zeros((N, 1))], axis=-1)
    w = (h_left + h_right) / 2
    return jnp.log(jnp.maximum(w, 1e-30))


class JointMaserModel(ModelBase):
    """Joint multi-galaxy megamaser H0 model.

    Shares H0, sigma_pec, and selection parameters across all galaxies;
    all other parameters are per-galaxy and scoped via handlers.scope.
    """

    def __init__(self, config_path, data_list):
        super().__init__(config_path)
        fsection("Joint Maser Disk Model")
        self._load_and_set_priors()

        self.models = [MaserDiskModel(config_path, data) for data in data_list]
        self.galaxy_names = [d["galaxy_name"] for d in data_list]
        self.use_selection = any(m.use_selection for m in self.models)
        fprint(f"loaded {len(self.models)} galaxies: "
               f"{', '.join(self.galaxy_names)}")

    def __call__(self):
        H0 = rsample("H0", self.priors["H0"])
        sigma_pec = rsample("sigma_pec", self.priors["sigma_pec"])
        h = H0 / 100.0

        shared = {"H0": H0, "sigma_pec": sigma_pec}

        if self.use_selection:
            D_lim = rsample("D_lim", self.priors["D_lim"])
            D_width = rsample("D_width", self.priors["D_width"])
            shared["D_lim"] = D_lim
            shared["D_width"] = D_width

            m0 = self.models[0]
            log_sel_grid = jax_norm.logcdf(
                (D_lim - m0._sel_D_grid) / D_width)
            log_integrand = log_sel_grid + m0._sel_lp_vol
            log_Z_sel = ln_trapz_precomputed(
                log_integrand, m0._sel_log_w, axis=-1)

        for model, gname in zip(self.models, self.galaxy_names):
            with handlers.scope(prefix=gname):
                D_c = model._sample_galaxy(shared, h)

            if self.use_selection:
                log_sel_this = jax_norm.logcdf((D_lim - D_c) / D_width)
                factor(f"ll_sel_{gname}", log_sel_this - log_Z_sel)
