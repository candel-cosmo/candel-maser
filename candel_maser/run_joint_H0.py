#!/usr/bin/env python
"""Toy joint megamaser H0 from saved single-galaxy D_A distance chains."""
import argparse
import os
import sys
import tempfile
import time

import numpy as np
import tomli

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_maser.toml")
MCP_GALAXIES = ("CGCG074-064", "NGC5765b", "NGC6264",
                "NGC6323", "UGC3789")
PHI_INIT_JITTER_DEG = 5.0
TOY_DISTANCE_GRID_SIZE = 1024
TOY_DISTANCE_MAX_SAMPLES = 100_000
TOY_DISTANCE_KDE_CHUNK = 4096

with open(CONFIG_PATH, "rb") as f:
    MASTER_CFG = tomli.load(f)


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "candel-mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from jax import config as _jax_config  # noqa: E402
_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from h5py import File as H5File  # noqa: E402
from jax import random  # noqa: E402
from joint_H0_helpers import healpix_los_vectors  # noqa: E402
from joint_H0_helpers import (DEFAULT_FIELD_CONFIG, _attach_icrs_rhat,  # noqa
                              _attach_velocity_data, _interp_los_velocity,
                              _load_or_build_vlos_cache,
                              _load_volume_selection_data, _logmeanexp,
                              _predict_cz_exact, _resolve_velocity_beta,
                              _resolve_volume_subsample_fraction,
                              _volume_log_Z_distance, _volume_log_Z_redshift,
                              rotate_vext_to_frame)
from maser_config import (add_dataset_arg, apply_dataset,  # noqa: E402
                          check_chain_dataset)
from numpyro.diagnostics import print_summary  # noqa: E402
from numpyro.infer import MCMC, NUTS, init_to_value  # noqa: E402

from candel.cosmo.cosmography import (  # noqa: E402
    AngularDiameterDistance2Redshift, Distance2Redshift)
from candel.model.integration import ln_simpson  # noqa: E402
from candel.model.maser_blackjax import MaserBlackJaxResult  # noqa: E402
from candel.model.utils import Maxwell, log_prob_integrand_sel  # noqa: E402
from candel.util import (fprint, fsection, radec_cartesian_to_galactic,  # noqa
                         results_path)


def _pfx(galaxy, site):
    return f"{galaxy}__{site}"


def _split_galaxies(value):
    if value == "all":
        return list(MCP_GALAXIES)
    return [g.strip() for g in value.split(",") if g.strip()]


def _resolve_distance_prior(selection, choice):
    """Resolve --distance-prior to the flat_dist boolean.

    With no explicit choice the prior is uniform-in-distance when selection is
    off and uniform-in-volume when selection is modelled.  A selected
    population is volume-distributed, so selection requires the volume prior;
    an explicit 'distance' choice with selection on is rejected.
    """
    prior = choice or ("distance" if selection == "none" else "volume")
    if selection != "none" and prior == "distance":
        raise ValueError(
            "selection modelling requires the uniform-in-volume distance "
            "prior; --distance-prior distance is incompatible with "
            f"--selection {selection}.")
    return prior == "distance"


def _required_inference(cfg, key):
    if key not in cfg:
        raise KeyError(f"Missing [inference].{key} in {CONFIG_PATH}")
    return cfg[key]


def _support_interval(prior):
    support = prior.support
    lower = getattr(support, "lower_bound", np.nan)
    upper = getattr(support, "upper_bound", np.nan)
    lower = float(np.asarray(lower)) if np.asarray(lower).shape == () else None
    upper = float(np.asarray(upper)) if np.asarray(upper).shape == () else None
    return lower, upper


def _toy_distance_overrides(values):
    overrides = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(
                "--toy-distance-file entries must be GALAXY=PATH")
        galaxy, path = value.split("=", 1)
        galaxy = galaxy.strip()
        path = path.strip()
        if not galaxy or not path:
            raise ValueError(
                "--toy-distance-file entries must be GALAXY=PATH")
        overrides[galaxy] = path
    return overrides


def _toy_default_distance_file(galaxy, args):
    root = MASTER_CFG.get("io", {}).get("root_output", "results/Megamaser")
    init = str(MASTER_CFG.get("inference", {}).get(
        "init_strategy", "config"))
    parts = []
    if args.add_ecc:
        parts.append("ecc")
    if args.add_quadratic_warp:
        parts.append("qw")
    parts.append(f"init{init}")
    suffix = "blackjax_mcmc_rphi_" + "_".join(parts)
    return results_path(root, galaxy, f"{galaxy}_{suffix}.hdf5")


def _toy_h_ref():
    return float(MASTER_CFG["model"].get("H0_ref", 73.0)) / 100.0


_TOY_D2Z = None


def _toy_distance2redshift():
    """Cached comoving-distance -> redshift interpolator at the config Om."""
    global _TOY_D2Z
    if _TOY_D2Z is None:
        om = float(MASTER_CFG["model"].get(
            "Om", MASTER_CFG["model"].get("Om0", 0.3)))
        _TOY_D2Z = Distance2Redshift(Om0=om)
    return _TOY_D2Z


def _toy_D_A_bounds(D_lo, D_hi):
    """Config comoving bounds -> D_A bounds at the fiducial cosmology
    (H0_ref, Om), matching stage-1's uniform-D_A bound convention."""
    z = np.asarray(_toy_distance2redshift()(
        jnp.asarray([float(D_lo), float(D_hi)]), h=_toy_h_ref()))
    return float(D_lo / (1.0 + z[0])), float(D_hi / (1.0 + z[1]))


_TOY_AD2Z = None


def _toy_ad2redshift():
    """Cached angular-diameter-distance -> redshift interpolator at
    config Om."""
    global _TOY_AD2Z
    if _TOY_AD2Z is None:
        om = float(MASTER_CFG["model"].get(
            "Om", MASTER_CFG["model"].get("Om0", 0.3)))
        _TOY_AD2Z = AngularDiameterDistance2Redshift(Om0=om)
    return _TOY_AD2Z


def _load_toy_distance_samples(path, dataset=None):
    """Return stage-1 D_A samples (samples/D_A from a uniform_D_A chain).  The
    toy reuses the stage-1 distance posterior as a likelihood, which is only
    valid under stage-1's uniform-D_A prior; anything else would double-count
    the stage-1 prior on top of the stage-2 distance prior."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with H5File(path, "r") as f:
        prior = f.attrs.get("D_c_prior", None)
        prior_str = None if prior is None else str(prior)
        if prior_str != "uniform_D_A":
            raise ValueError(
                f"{path} has D_c_prior={prior_str!r}; the toy distance "
                f"likelihood requires a uniform_D_A stage-1 chain.")
        if dataset is not None:
            check_chain_dataset(f.attrs, dataset, path)
        if "samples/D_A" not in f:
            raise KeyError(f"{path} has no samples/D_A dataset")
        samples = np.asarray(f["samples/D_A"][...], dtype=np.float64)
    samples = samples.reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size < 20:
        raise ValueError(
            f"{path} has too few finite D_A samples ({samples.size})")
    return samples


def _build_log_distance_likelihood(samples, D_lo, D_hi, n_grid,
                                   max_samples):
    from scipy.special import logsumexp

    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if (
        max_samples is not None and max_samples > 0
        and samples.size > max_samples
    ):
        idx = np.linspace(0, samples.size - 1, int(max_samples), dtype=int)
        samples = samples[idx]

    std = float(np.std(samples, ddof=1))
    q25, q75 = np.percentile(samples, [25, 75])
    iqr_sigma = float((q75 - q25) / 1.349) if q75 > q25 else std
    scale = min(std, iqr_sigma) if iqr_sigma > 0 and std > 0 else std
    bw = 0.9 * scale * samples.size ** (-0.2)
    bw_floor = max(1e-3, 1e-4 * float(D_hi - D_lo))
    if not np.isfinite(bw) or bw <= bw_floor:
        bw = max(bw_floor, 0.01 * float(D_hi - D_lo))

    D_grid = np.linspace(float(D_lo), float(D_hi), int(n_grid))
    total = np.full(D_grid.shape, -np.inf, dtype=np.float64)
    for start in range(0, samples.size, TOY_DISTANCE_KDE_CHUNK):
        x = samples[start:start + TOY_DISTANCE_KDE_CHUNK]
        z = (D_grid[None, :] - x[:, None]) / bw
        total = np.logaddexp(total, logsumexp(-0.5 * z * z, axis=0))
    log_L = (
        total - np.log(samples.size) - np.log(bw)
        - 0.5 * np.log(2 * np.pi)
    )
    log_L = log_L - np.nanmax(log_L)
    return (
        jnp.asarray(D_grid), jnp.asarray(log_L), float(bw), int(samples.size)
    )


def _build_toy_items(galaxies, args, velocity_data=None):
    overrides = _toy_distance_overrides(args.toy_distance_file)
    items = []
    for galaxy in galaxies:
        gcfg = MASTER_CFG["model"]["galaxies"][galaxy]
        path = overrides.get(galaxy) or _toy_default_distance_file(
            galaxy, args)
        samples = _load_toy_distance_samples(         # D_A samples
            path, MASTER_CFG.get("io", {}).get("dataset"))
        D_q16, D_med, D_q84 = np.percentile(samples, [16, 50, 84])
        # D_c init for the sampled comoving distance: D_c = D_A (1 + z)
        # at H0_ref.
        z_med = float(np.asarray(_toy_ad2redshift()(
            jnp.asarray([float(D_med)]), h=_toy_h_ref()))[0])
        D_c_init = float(D_med * (1.0 + z_med))
        D_lo, D_hi = float(gcfg["D_lo"]), float(gcfg["D_hi"])
        DA_lo, DA_hi = _toy_D_A_bounds(D_lo, D_hi)
        outside = (samples < DA_lo) | (samples > DA_hi)
        D_grid, log_L_grid, bw, n_used = _build_log_distance_likelihood(
            samples, DA_lo, DA_hi,
            args.toy_distance_n_grid, args.toy_distance_max_samples)
        fprint(
            f"{galaxy}: toy D_A KDE from {path} "
            f"(median={D_med:.2f} -{D_med - D_q16:.2f} "
            f"+{D_q84 - D_med:.2f} Mpc; bw={bw:.3g}; "
            f"samples={n_used}/{samples.size})")
        if np.any(outside):
            fprint(
                f"  WARNING: {int(np.sum(outside))}/{samples.size} "
                f"source D_A samples are outside the D_A support "
                f"[{DA_lo:g}, {DA_hi:g}] Mpc.")
        items.append({
            "name": galaxy,
            "RA": float(gcfg["ra"]),
            "dec": float(gcfg["dec"]),
            # CMB-frame recession velocity for H0
            "v_sys_obs": float(gcfg.get("v_cmb_kms", gcfg["v_sys_obs"])),
            "D_lo": D_lo,            # comoving bounds (selection grid)
            "D_hi": D_hi,
            "DA_lo": DA_lo,          # D_A KDE-likelihood bounds
            "DA_hi": DA_hi,
            "D_init": D_c_init,      # sampled coordinate is D_c
            "D_grid": D_grid,
            "log_L_grid": log_L_grid,
            "toy_distance_file": os.path.abspath(path),
            "toy_distance_kde_bandwidth": bw,
            "toy_distance_kde_samples": n_used,
            "toy_distance_total_samples": int(samples.size),
            "toy_distance_source_outside_support": int(np.sum(outside)),
        })
    if velocity_data is None:
        _attach_icrs_rhat(items)
    else:
        _attach_velocity_data(items, velocity_data)
    return items


class ToyDistanceTarget:
    def __init__(self, items, shared_priors, selection, volume_data,
                 flat_dist, velocity_beta, sample_velocity_beta,
                 sample_vext, los_nside, vext_prior=None):
        self.items = tuple(items)
        self.selection = selection
        self.volume_data = volume_data
        self.flat_dist = bool(flat_dist)
        self.velocity_beta = float(velocity_beta)
        self.sample_velocity_beta = bool(sample_velocity_beta)
        self.sample_vext = bool(sample_vext)
        self.vext_prior = vext_prior
        self.shared_priors = dict(shared_priors)
        self.names = tuple(item["name"] for item in self.items)
        om = float(MASTER_CFG["model"].get(
            "Om", MASTER_CFG["model"].get("Om0", 0.3)))
        self.distance2redshift = Distance2Redshift(Om0=om)
        self.d_sel = jnp.linspace(
            1.0, max(item["D_hi"] for item in self.items), 1001)
        self.log_d2_sel = 2.0 * jnp.log(self.d_sel)
        self.sel_rhat = healpix_los_vectors(los_nside)
        # Frame of `item["rhat"]`: the field frame when the LOS velocity cache
        # supplied it, ICRS when built straight from RA/dec.
        self.los_frame = self.items[0]["rhat_frame"]

        specs = [("H0", self.shared_priors["H0"]),
                 ("sigma_pec", self.shared_priors["sigma_pec"])]
        if self.sample_velocity_beta:
            specs.append(("beta",
                          dist.Normal(self.velocity_beta, 0.02)))
        if self.sample_vext:
            if self.vext_prior is not None:
                specs.append(("Vext", dist.MultivariateNormal(
                    jnp.asarray(self.vext_prior["mean"]),
                    covariance_matrix=jnp.asarray(self.vext_prior["cov"]))))
            else:
                specs.extend([
                    ("Vext_phi", dist.Uniform(0.0, 2.0 * jnp.pi)),
                    ("Vext_cos_theta", dist.Uniform(-1.0, 1.0)),
                    ("Vext_mag", self.shared_priors["Vext_mag"]),
                ])
        if selection == "distance":
            specs.extend([
                ("D_lim", self.shared_priors["D_lim"]),
                ("D_width", self.shared_priors["D_width"]),
            ])
        elif selection == "redshift":
            specs.extend([
                ("cz_lim_selection",
                 self.shared_priors["cz_lim_selection"]),
                ("cz_lim_selection_width",
                 self.shared_priors["cz_lim_selection_width"]),
            ])
        for item in self.items:
            specs.append((_pfx(item["name"], "D_c"),
                          dist.Uniform(item["D_lo"], item["D_hi"])))
        self.specs = tuple(specs)
        self.theta_sites = tuple(name for name, _ in self.specs)

    def _nudge_init(self, init):
        out = dict(init)
        for name, prior in self.specs:
            if name not in out:
                raise KeyError(f"missing initial value {name!r}")
            if np.asarray(out[name]).ndim > 0:
                continue  # vector site (informative Vext): no scalar nudging
            lower, upper = _support_interval(prior)
            value = float(np.asarray(out[name]))
            if lower is not None and np.isfinite(lower):
                hi = upper if upper is not None and np.isfinite(upper) else (
                    lower + 1.0)
                margin = max((hi - lower) * 1e-3, 1e-6)
                if value <= lower:
                    value = lower + margin
            if upper is not None and np.isfinite(upper):
                lo = lower if lower is not None and np.isfinite(lower) else (
                    upper - 1.0)
                margin = max((upper - lo) * 1e-3, 1e-6)
                if value >= upper:
                    value = upper - margin
            out[name] = jnp.asarray(value)
        return out

    def _h(self, params):
        return params["H0"] / 100.0

    def _vext(self, params):
        if not self.sample_vext:
            return jnp.zeros(3)
        if self.vext_prior is not None:
            return params["Vext"]
        sin_theta = jnp.sqrt(1.0 - params["Vext_cos_theta"] ** 2)
        return params["Vext_mag"] * jnp.array([
            sin_theta * jnp.cos(params["Vext_phi"]),
            sin_theta * jnp.sin(params["Vext_phi"]),
            params["Vext_cos_theta"],
        ])

    def _beta(self, params):
        if self.sample_velocity_beta:
            return params["beta"]
        return self.velocity_beta

    def _D_c(self, params, item):
        return params[_pfx(item["name"], "D_c")]

    def _D_A(self, params, item):
        """Convert the sampled comoving distance to angular-diameter
        distance at the current H0 so the stage-1 KDE (built in D_A)
        can be evaluated."""
        D_c = self._D_c(params, item)
        z = self.distance2redshift(
            jnp.atleast_1d(D_c), h=self._h(params)).squeeze()
        return D_c / (1.0 + z)

    def _distance_loglike(self, params, item):
        return jnp.interp(
            self._D_A(params, item), item["D_grid"], item["log_L_grid"],
            left=-1e30, right=-1e30)

    def _redshift_and_selection(self, params):
        H0 = params["H0"]
        h = H0 / 100.0
        sigma_pec = params["sigma_pec"]
        Vext = self._vext(params)
        beta = self._beta(params)
        log_Z = None
        log_Z_fields = None
        if self.selection == "distance":
            D_lim = params["D_lim"]
            D_width = params["D_width"]
            if self.volume_data is None:
                log_sel_grid = jax.scipy.stats.norm.logcdf(
                    (D_lim - self.d_sel) / D_width)
                log_vol = 0.0 if self.flat_dist else self.log_d2_sel
                log_Z = ln_simpson(log_sel_grid + log_vol, self.d_sel)
            else:
                log_Z_fields = _volume_log_Z_distance(
                    self.volume_data, H0, D_lim, D_width, self.flat_dist)
        elif self.selection == "redshift":
            cz_lim = params["cz_lim_selection"]
            cz_width = params["cz_lim_selection_width"]
            if self.volume_data is None:
                z_sel = self.distance2redshift(self.d_sel, h=h)
                # Equal-weight full-sky average, so it needs no frame rotation.
                Vext_los = self.sel_rhat @ Vext
                cz_pred = _predict_cz_exact(
                    z_sel[None, :], Vext_los[:, None])
                log_P = log_prob_integrand_sel(
                    cz_pred, sigma_pec, cz_lim, cz_width)
                log_P_ang = jax.scipy.special.logsumexp(
                    log_P, axis=0) - jnp.log(self.sel_rhat.shape[0])
                log_vol = 0.0 if self.flat_dist else self.log_d2_sel
                log_Z = ln_simpson(log_P_ang + log_vol, self.d_sel)
            else:
                log_Z_fields = _volume_log_Z_redshift(
                    self.volume_data, H0, sigma_pec, beta,
                    Vext, cz_lim, cz_width, self.flat_dist)

        ll_fields_total = None
        Vext_los_frame = rotate_vext_to_frame(Vext, self.los_frame)
        for item in self.items:
            D_c = self._D_c(params, item)
            z_cosmo = self.distance2redshift(
                jnp.atleast_1d(D_c), h=h).squeeze()
            Vext_rad = jnp.dot(item["rhat"], Vext_los_frame)
            if "los_velocity" in item:
                vlos = _interp_los_velocity(
                    D_c * h, item["los_r"], item["los_velocity"])
                Vrad = beta * vlos + Vext_rad
            else:
                Vrad = jnp.atleast_1d(Vext_rad)
            cz = _predict_cz_exact(z_cosmo, Vrad)
            ll = dist.Normal(cz, sigma_pec).log_prob(item["v_sys_obs"])
            if self.selection == "distance":
                log_sel = jax.scipy.stats.norm.logcdf(
                    (params["D_lim"] - D_c) / params["D_width"])
                ll = ll + log_sel - (
                    log_Z if log_Z_fields is None else log_Z_fields)
            elif self.selection == "redshift":
                log_sel = jax.scipy.stats.norm.logcdf(
                    (params["cz_lim_selection"] - item["v_sys_obs"])
                    / params["cz_lim_selection_width"])
                ll = ll + log_sel - (
                    log_Z if log_Z_fields is None else log_Z_fields)
            ll_fields_total = ll if ll_fields_total is None else (
                ll_fields_total + ll)
        return _logmeanexp(ll_fields_total)

    def extra_logdensity(self, params):
        """Log-density terms beyond the parameter priors: the uniform-in-volume
        distance reweighting, the per-galaxy KDE distance likelihoods, and the
        joint redshift/selection term.  Prior densities are contributed by the
        numpyro sample sites, so they are excluded here."""
        total = jnp.asarray(0.0, dtype=params["H0"].dtype)
        for item in self.items:
            if not self.flat_dist:
                # D_c is sampled directly, so uniform-in-volume is exactly
                # p(D_c) ~ D_c^2 with no change-of-variables Jacobian.
                total = total + 2.0 * jnp.log(self._D_c(params, item))
            total = total + self._distance_loglike(params, item)
        return total + self._redshift_and_selection(params)

    def numpyro_model(self):
        params = {name: numpyro.sample(name, prior)
                  for name, prior in self.specs}
        for item in self.items:
            numpyro.deterministic(_pfx(item["name"], "D_A"),
                                  self._D_A(params, item))
        numpyro.factor("extra", self.extra_logdensity(params))


def _toy_init(target):
    init = {
        "H0": jnp.asarray(70.0),
        "sigma_pec": jnp.asarray(250.0),
    }
    if target.sample_velocity_beta:
        init["beta"] = jnp.asarray(float(target.velocity_beta))
    if target.sample_vext:
        if target.vext_prior is not None:
            init["Vext"] = jnp.asarray(
                np.asarray(target.vext_prior["mean"], dtype=float))
        else:
            init["Vext_phi"] = jnp.asarray(jnp.pi)
            init["Vext_cos_theta"] = jnp.asarray(0.0)
            init["Vext_mag"] = jnp.asarray(1.0)
    if target.selection == "distance":
        init["D_lim"] = jnp.asarray(max(item["D_hi"] for item in target.items))
        init["D_width"] = jnp.asarray(100.0)
    elif target.selection == "redshift":
        init["cz_lim_selection"] = jnp.asarray(10000.0)
        init["cz_lim_selection_width"] = jnp.asarray(500.0)
    for item in target.items:
        init[_pfx(item["name"], "D_c")] = jnp.asarray(item["D_init"])
    return init


def run_toy_mcmc(target, init, rng_key, *, num_warmup, num_samples,
                 num_chains, target_accept_theta, theta_step_init,
                 max_num_doublings):
    """Sample the toy joint-H0 target with numpyro NUTS.  Chains run vectorised
    (a single vmapped scan) so multiple chains share one device dispatch."""
    num_chains = int(num_chains)
    init = target._nudge_init(init)
    kernel = NUTS(
        target.numpyro_model,
        step_size=theta_step_init,
        target_accept_prob=target_accept_theta,
        max_tree_depth=max_num_doublings,
        dense_mass=True,
        init_strategy=init_to_value(values=init))
    mcmc = MCMC(
        kernel, num_warmup=int(num_warmup), num_samples=int(num_samples),
        num_chains=num_chains, chain_method="vectorized", progress_bar=True)

    t0 = time.time()
    mcmc.run(rng_key, extra_fields=(
        "diverging", "accept_prob", "num_steps", "potential_energy"))
    runtime = time.time() - t0

    grouped = num_chains > 1
    samples = {key: np.asarray(value) for key, value in
               mcmc.get_samples(group_by_chain=grouped).items()}
    extra = mcmc.get_extra_fields(group_by_chain=grouped)
    div = np.asarray(extra["diverging"])
    info = {
        "theta_acceptance_rate": np.asarray(extra["accept_prob"]),
        "theta_is_accepted": ~div,
        "theta_is_divergent": div,
        "theta_num_integration_steps": np.asarray(extra["num_steps"]),
    }
    log_density = -np.asarray(extra["potential_energy"])
    try:
        adapt = mcmc.last_state.adapt_state
        imm = adapt.inverse_mass_matrix
        # dense mass: one block over all sites
        if isinstance(imm, dict):
            imm = next(iter(imm.values()))
        parameters = {
            "step_size": np.asarray(jax.device_get(adapt.step_size)),
            "inverse_mass_matrix": np.asarray(jax.device_get(imm)),
        }
    except Exception:
        parameters = {}

    return MaserBlackJaxResult(
        samples=samples,
        log_density=log_density,
        info=info,
        warmup_info={},
        parameters=parameters,
        theta_sites=target.theta_sites,
        runtime_seconds=runtime)


def _joint_selection_nside():
    cfg = MASTER_CFG.get("joint", {}).get("selection", {})
    if "los_nside" not in cfg:
        raise KeyError(
            "Missing [joint.selection].los_nside in config_maser.toml")
    return int(cfg["los_nside"])


def _joint_volume_subsample_defaults():
    return MASTER_CFG.get("joint", {}).get("selection", {}).get(
        "volume_subsample_fraction", {})


def _joint_sigma_pec_prior():
    cfg = MASTER_CFG.get("joint", {}).get("priors", {}).get("sigma_pec")
    if cfg is None:
        raise KeyError("Missing [joint.priors.sigma_pec] in config_maser.toml")
    if str(cfg.get("dist", "")).lower() != "maxwell":
        raise ValueError("joint.priors.sigma_pec.dist must be 'maxwell'")
    return Maxwell(float(cfg["scale"]))


def _joint_vext_mag_bounds():
    cfg = MASTER_CFG.get("joint", {}).get("priors", {}).get("Vext_mag")
    if cfg is None:
        raise KeyError("Missing [joint.priors.Vext_mag] in config_maser.toml")
    if str(cfg.get("dist", "")).lower() != "uniform":
        raise ValueError("joint.priors.Vext_mag.dist must be 'uniform'")
    return float(cfg.get("lower", 0.0)), float(cfg["upper"])


def _joint_vext_prior(reconstruction):
    """Informative Vext prior (Cartesian mean + cov, with a magnitude/direction
    summary) from [joint.priors.Vext_informative.<reconstruction>], or None to
    keep the uniform prior.  The numbers are produced offline by
    scripts/megamaser/extract_vext_prior.py; no posterior files are read here.
    """
    cfg = (MASTER_CFG.get("joint", {}).get("priors", {})
           .get("Vext_informative", {}))
    entry = cfg.get(reconstruction)
    if entry is None:
        return None
    mean = np.asarray(entry["mean"], dtype=float)
    cov = np.asarray(entry["cov"], dtype=float)
    if mean.shape != (3,) or cov.shape != (3, 3):
        raise ValueError(
            f"Vext_informative['{reconstruction}'] needs mean (3,) and "
            f"cov (3, 3); got {mean.shape} and {cov.shape}")
    mag = float(np.linalg.norm(mean))
    mag_sigma = float(np.sqrt(mean @ cov @ mean) / mag)
    _, ell, b = radec_cartesian_to_galactic(*mean)
    return {"mean": mean, "cov": cov, "mag": mag, "mag_sigma": mag_sigma,
            "ell": float(ell) % 360.0, "b": float(b)}


def _save_hdf5(path, result, metadata):
    with H5File(path, "w") as f:
        grp = f.create_group("samples", track_order=True)
        for key in sorted(result.samples):
            grp.create_dataset(key, data=np.asarray(result.samples[key]))
        f.create_dataset("log_density", data=np.asarray(result.log_density))
        info = f.create_group("info", track_order=True)
        for key in sorted(result.info):
            info.create_dataset(key, data=np.asarray(result.info[key]))
        params = f.create_group("sampler_parameters", track_order=True)
        for key in sorted(result.parameters):
            params.create_dataset(key, data=np.asarray(result.parameters[key]))
        f.attrs["theta_sites"] = ",".join(result.theta_sites)
        for key, value in metadata.items():
            f.attrs[key] = value


def _result_path(galaxies, selection, reconstruction, flat_dist,
                 variant="", toy=False):
    gal_tag = "all" if tuple(galaxies) == MCP_GALAXIES else (
        "_".join(g.replace("-", "") for g in galaxies))
    stem = "joint_H0_toy" if toy else "joint_H0"
    prior_tag = "flat" if flat_dist else "r2"
    # root_output is dataset-namespaced by apply_dataset; keep stage-2 products
    # separate from the per-galaxy distance chains consumed by this runner.
    root = MASTER_CFG.get("io", {}).get("root_output", "results/Megamaser")
    return results_path(
        root, "H0",
        f"{stem}_{gal_tag}_{selection}"
        f"_{reconstruction}_{prior_tag}{variant}.hdf5")


def _vext_cartesian(samples):
    """Cartesian Vext (..., 3) from a chain, or None if Vext was not sampled.

    The informative prior samples the vector directly; the uniform prior stores
    mag/phi/cos_theta, which are rebuilt into the same ICRS-Cartesian vector.
    """
    if "Vext" in samples:
        return np.asarray(samples["Vext"], float)
    if all(k in samples for k in ("Vext_mag", "Vext_phi", "Vext_cos_theta")):
        mag = np.asarray(samples["Vext_mag"], float)
        phi = np.asarray(samples["Vext_phi"], float)
        ct = np.asarray(samples["Vext_cos_theta"], float)
        st = np.sqrt(np.clip(1.0 - ct ** 2, 0.0, None))
        return mag[..., None] * np.stack(
            [st * np.cos(phi), st * np.sin(phi), ct], axis=-1)
    return None


def _print_summary(result):
    """MCMC diagnostics: divergences, acceptance, and the numpyro parameter
    table for the scalar sites.  Per-spot ``r_ang``/``phi`` arrays and the
    derived ``D_A`` are excluded; per-galaxy sites are shown as
    ``galaxy/site``.
    """
    info, samples = result.info, result.samples
    H0 = np.asarray(samples["H0"], dtype=float)
    grouped = H0.ndim == 2                      # (num_chains, num_samples)
    max_ndim = 2 if grouped else 1
    fsection("MCMC summary")
    fprint(f"H0 = {H0.mean():.2f} +/- {H0.std(ddof=1):.2f} km/s/Mpc")
    div = np.asarray(info.get("theta_is_divergent", []))
    if div.size:
        nd = int(np.sum(np.asarray(div) != 0))
        fprint(f"NUTS divergences: {nd} / {div.size} "
               f"({100.0 * nd / div.size:.2f}%)")
    for key, label in (("theta_acceptance_rate", "theta acceptance"),
                       ("latent_accept_mean", "latent acceptance"),
                       ("reflect_accept_mean", "reflect acceptance")):
        a = np.asarray(info.get(key, []), dtype=float)
        if a.size:
            fprint(f"{label}: {float(np.mean(a)):.3f}")
    if H0.size < 20:
        fprint(f"parameter summary skipped: only {H0.size} samples.")
        return
    # Under the informative prior Vext is sampled as a 3-vector and the
    # chain tracks nothing spherical; print_summary drops non-scalar
    # sites, so report the sampled Cartesian components and a magnitude
    # derived here.  The uniform branch keeps its sampled
    # Vext_mag/phi/cos_theta sites untouched.
    tsamples = dict(samples)
    if "Vext" in tsamples:
        Vc = np.asarray(tsamples.pop("Vext"))
        for i, ax in enumerate("xyz"):
            tsamples[f"Vext_{ax}"] = Vc[..., i]
        tsamples["Vext_mag"] = np.sqrt(np.sum(Vc ** 2, axis=-1))
    order = [k for k in ("H0", "sigma_pec", "Vext_x", "Vext_y", "Vext_z",
                         "Vext_mag", "Vext_phi", "Vext_cos_theta")
             if k in tsamples]
    order += sorted(k for k in tsamples if k not in order)
    table = {}
    for k in order:
        a = np.asarray(tsamples[k])
        if a.ndim <= max_ndim and not k.endswith("__D_A"):
            table[k.replace("__", "/")] = a
    print_summary(table, prob=0.9, group_by_chain=grouped)

    # Report the posterior bulk flow as magnitude + Galactic direction,
    # derived after the run from the sampled Vext (Cartesian under the
    # informative prior, else rebuilt from the mag/phi/cos_theta sites).
    Vc = _vext_cartesian(samples)
    if Vc is not None:
        V = Vc.reshape(-1, 3)
        m = np.linalg.norm(V, axis=1)
        lo, med, hi = np.percentile(m, [5, 50, 95])
        _, ell, b = radec_cartesian_to_galactic(*V.mean(axis=0))
        fprint(f"Vext posterior: |Vext|={med:.0f} (+{hi - med:.0f}/"
               f"-{med - lo:.0f}) km/s, (l,b)=({float(ell) % 360:.0f},"
               f"{float(b):.0f}) deg")


def _print_toy_support_diagnostics(result, items):
    fsection("Toy distance support diagnostics")
    for item in items:
        key = _pfx(item["name"], "D_A")
        if key not in result.samples:
            continue
        D = np.asarray(result.samples[key], dtype=float).reshape(-1)
        D_lo = float(item["DA_lo"])
        D_hi = float(item["DA_hi"])
        span = D_hi - D_lo
        edge = max(0.01 * span, float(item["toy_distance_kde_bandwidth"]))
        lo_frac = float(np.mean(D <= D_lo + edge))
        hi_frac = float(np.mean(D >= D_hi - edge))
        q01, q50, q99 = np.percentile(D, [1, 50, 99])
        source_out = int(item.get("toy_distance_source_outside_support", 0))
        fprint(
            f"{item['name']}: D_A q01/50/99 = "
            f"{q01:.2f}/{q50:.2f}/{q99:.2f} Mpc; "
            f"edge window={edge:.2f} Mpc; "
            f"near lower/upper = {100 * lo_frac:.2f}%/"
            f"{100 * hi_frac:.2f}%; source outside={source_out}")
        if lo_frac > 0.01 or hi_frac > 0.01:
            fprint("  WARNING: posterior mass is near a configured distance "
                   "support edge; widen D_lo/D_hi or inspect the distance "
                   "chain if this is unexpected.")


def _toy_distance_kde_plots(items, outpath):
    """Overlay each toy distance KDE on a fine histogram of its raw D_A
    samples so the rule-of-thumb bandwidth can be eyeballed for
    over/under-smoothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = outpath[:-5] if outpath.endswith(".hdf5") else outpath
    n = len(items)
    fig, axs = plt.subplots(1, n, figsize=(4.5 * n, 3.6), squeeze=False)
    for ax, item in zip(axs[0], items):
        samples = _load_toy_distance_samples(
            item["toy_distance_file"],
            MASTER_CFG.get("io", {}).get("dataset"))
        D = np.asarray(item["D_grid"])
        L = np.exp(np.asarray(item["log_L_grid"]))
        norm = np.trapezoid(L, D)
        if norm > 0:
            L = L / norm
        ax.hist(samples, bins=120, range=(float(D[0]), float(D[-1])),
                density=True, histtype="stepfilled", color="0.8")
        ax.plot(D, L, "C3", lw=1.5,
                label=f"KDE bw={item['toy_distance_kde_bandwidth']:.3g}, "
                      f"n={item['toy_distance_kde_samples']}")
        ax.axvline(item["DA_lo"], color="0.4", ls=":", lw=1)
        ax.axvline(item["DA_hi"], color="0.4", ls=":", lw=1)
        ax.set_xlabel(r"$D_A\ [\mathrm{Mpc}]$")
        ax.set_ylabel(r"$p(D_A)$")
        ax.text(0.03, 0.95, item["name"], transform=ax.transAxes,
                ha="left", va="top")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fpath = os.path.abspath(f"{base}_toy_distance_kde.png")
    fig.savefig(fpath, dpi=300)
    plt.close(fig)
    fprint(f"saved toy distance KDE plot to {fpath}")


def _corner_plots(result, galaxies, outpath, selection, sample_vext):
    """Per-galaxy disk corners plus a joint H0/sigma_v/selection/distance
    corner.  Output names carry the joint_H0 stem so they are distinct from
    the single-galaxy runs."""
    import matplotlib
    matplotlib.use("Agg")
    from candel.plotting.corner import plot_corner

    samples = dict(result.samples)
    Vc = _vext_cartesian(samples)
    if Vc is not None:  # corner shows the magnitude and Galactic direction
        samples["Vext_mag"] = np.sqrt(np.sum(Vc ** 2, axis=-1))
        flat = Vc.reshape(-1, 3)
        _, ell, b = radec_cartesian_to_galactic(flat[:, 0], flat[:, 1],
                                                flat[:, 2])
        samples["Vext_ell"] = np.asarray(ell).reshape(Vc.shape[:-1])
        samples["Vext_b"] = np.asarray(b).reshape(Vc.shape[:-1])
    max_ndim = np.asarray(samples["H0"]).ndim    # 1 single / 2 multi-chain
    n_total = np.asarray(samples["H0"]).size
    if n_total < 50:
        fprint(f"corner plots skipped: only {n_total} samples.")
        return
    base = outpath[:-5] if outpath.endswith(".hdf5") else outpath

    jobs = []

    def _add(keys, path_smooth, path_unsmoothed):
        plot_samples = {}
        for k in keys:
            x = np.asarray(samples[k]).reshape(-1)
            if np.all(np.isfinite(x)) and np.ptp(x) > 0:
                plot_samples[k.replace("__", "/")] = x
        if len(plot_samples) < 2:
            fprint(f"corner skipped ({os.path.basename(path_smooth)}): "
                   "<2 varying params")
            return
        jobs.append((
            plot_samples,
            os.path.abspath(path_smooth),
            os.path.abspath(path_unsmoothed),
        ))

    for gal in galaxies:
        pfx = _pfx(gal, "")
        keys = [k for k in sorted(samples)
                if k.startswith(pfx)
                and not k.endswith(("__r_ang", "__phi", "__D_c"))
                and np.asarray(samples[k]).ndim <= max_ndim]
        gal_tag = gal.replace("-", "")
        _add(
            keys, f"{base}_corner_{gal_tag}.png",
            f"{base}_corner_unsmoothed_{gal_tag}.png")

    jkeys = ["H0", "sigma_pec"]
    if sample_vext and "Vext_mag" in samples:
        jkeys.append("Vext_mag")
        if "Vext_ell" in samples:
            jkeys += ["Vext_ell", "Vext_b"]
    if "beta" in samples:
        jkeys.append("beta")
    sel_keys = {"distance": ("D_lim", "D_width"),
                "redshift": ("cz_lim_selection",
                             "cz_lim_selection_width")}.get(selection, ())
    for k in sel_keys:
        if k in samples:
            jkeys.append(k)
    for gal in galaxies:
        k = _pfx(gal, "D_c")
        if k in samples:
            jkeys.append(k)
    _add(
        jkeys, f"{base}_corner_jointH0.png",
        f"{base}_corner_unsmoothed_jointH0.png")
    if not jobs:
        return
    fprint(f"Saving smoothed corner plots ({len(jobs)} files):")
    for _, path_smooth, _ in jobs:
        fprint(f"  {path_smooth}")
    for plot_samples, path_smooth, _ in jobs:
        plot_corner(
            plot_samples, show_fig=False, filename=path_smooth, smooth=1,
            keys=list(plot_samples), log_save=False)
    fprint(f"Saving non-smoothed corner plots ({len(jobs)} files):")
    for _, _, path_unsmoothed in jobs:
        fprint(f"  {path_unsmoothed}")
    for plot_samples, _, path_unsmoothed in jobs:
        plot_corner(
            plot_samples, show_fig=False, filename=path_unsmoothed, smooth=0,
            keys=list(plot_samples), log_save=False)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Toy joint megamaser H0: sample a shared H0 with per-"
                    "galaxy D_A against KDE distance likelihoods built from "
                    "the saved single-galaxy chains.")
    parser.add_argument("--galaxy", default="NGC6264,NGC6323",
                        help="Comma-separated galaxies, or all for five MCP.")
    add_dataset_arg(parser)
    parser.add_argument("--selection", choices=("none", "distance",
                                                "redshift"),
                        default="redshift")
    parser.add_argument("--reconstruction",
                        choices=("none", "Carrick2015",
                                 "ManticoreLocalCOLA"),
                        default="none")
    parser.add_argument("--field-config", default=DEFAULT_FIELD_CONFIG)
    parser.add_argument("--field-indices", type=int, nargs="*",
                        default=None)
    parser.add_argument("--velocity-beta", type=float, default=None)
    parser.add_argument("--velocity-smoothing-scale", type=float, default=0.0)
    parser.add_argument("--field-smoothing-scale", type=float, default=0.0)
    parser.add_argument("--selection-integral-radius", type=float,
                        default=250.0)
    parser.add_argument("--selection-integral-geometry",
                        choices=("sphere", "cube"), default="sphere")
    parser.add_argument("--volume-subsample-fraction", type=float,
                        default=None)
    parser.add_argument("--volume-subsample-seed", type=int, default=42)
    parser.add_argument("--volume-supersample-factor", type=int, default=1)
    parser.add_argument("--volume-supersample-radius", type=float,
                        default=0.0)
    parser.add_argument("--volume-supersample-target-dx", type=float,
                        default=0.0)
    parser.add_argument("--vlos-rmin", type=float, default=0.1)
    parser.add_argument("--vlos-rmax", type=float, default=250.0)
    parser.add_argument("--vlos-dr", type=float, default=0.5)
    parser.add_argument("--overwrite-vlos-cache", action="store_true")
    parser.add_argument("--Vext", action="store_true",
                        help="Sample a coherent external bulk flow Vext "
                             "(off by default).")
    parser.add_argument("--vext-prior-scale", type=float, default=None,
                        help="Upper bound of the Vext magnitude prior. "
                             "Default: [joint.priors.Vext_mag].upper in "
                             "config_maser.toml.")
    parser.add_argument("--distance-prior", choices=("distance", "volume"),
                        default=None,
                        help="Distance prior. Default: uniform-in-distance "
                             "when --selection none, uniform-in-volume "
                             "otherwise. Selection forbids 'distance'.")
    parser.add_argument("--toy-distance-file", action="append", default=[],
                        metavar="GALAXY=PATH",
                        help="Override the per-galaxy chain used for the KDE "
                             "distance likelihood. Repeat per galaxy.")
    parser.add_argument("--toy-distance-n-grid", type=int,
                        default=TOY_DISTANCE_GRID_SIZE,
                        help="Number of tabulation points for each toy KDE.")
    parser.add_argument("--toy-distance-max-samples", type=int,
                        default=TOY_DISTANCE_MAX_SAMPLES,
                        help="Maximum D_A samples used per KDE; <=0 uses all.")
    parser.add_argument("--num-warmup", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-accept-theta", type=float, default=None)
    parser.add_argument("--initial-step-size", type=float, default=None)
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--f64", action="store_true",
                        help="Accepted for compatibility; MCMC always uses "
                             "float64.")
    parser.add_argument("--add-ecc", action="store_true",
                        help="Select the eccentric stage-1 chains.")
    parser.add_argument("--add-quadratic-warp", action="store_true",
                        help="Select the quadratic-warp stage-1 chains.")
    parser.add_argument("--leave-one-out-dropped", default=None,
                        metavar="GAL",
                        help="Galaxy dropped in this leave-one-out fit; "
                             "logged for provenance (not analysed).")
    args = parser.parse_args(argv)
    apply_dataset(MASTER_CFG, args.dataset)

    inf_cfg = MASTER_CFG["inference"]
    joint_inf = MASTER_CFG.get("joint", {}).get("inference", {})
    args.num_warmup = int(args.num_warmup if args.num_warmup is not None
                          else joint_inf.get(
                              "num_warmup", inf_cfg["num_warmup"]))
    args.num_samples = int(args.num_samples if args.num_samples is not None
                           else joint_inf.get(
                               "num_samples", inf_cfg["num_samples"]))
    args.num_chains = int(args.num_chains if args.num_chains is not None
                          else _required_inference(inf_cfg, "num_chains"))
    args.seed = int(args.seed if args.seed is not None
                    else _required_inference(inf_cfg, "seed"))
    args.target_accept_theta = float(
        args.target_accept_theta if args.target_accept_theta is not None
        else joint_inf.get("target_accept_theta", 0.9))
    args.initial_step_size = float(
        args.initial_step_size if args.initial_step_size is not None
        else _required_inference(inf_cfg, "blackjax_initial_step_size"))
    args.max_tree_depth = int(
        args.max_tree_depth if args.max_tree_depth is not None
        else joint_inf.get("max_tree_depth",
                           _required_inference(inf_cfg, "max_tree_depth")))
    galaxies = _split_galaxies(args.galaxy)
    valid = set(MASTER_CFG["model"]["galaxies"])
    bad = [g for g in galaxies if g not in valid]
    if bad:
        parser.error(f"unknown galaxy: {', '.join(bad)}")
    if args.leave_one_out_dropped is not None:
        if args.leave_one_out_dropped not in valid:
            parser.error(f"unknown --leave-one-out-dropped galaxy: "
                         f"{args.leave_one_out_dropped}")
        if args.leave_one_out_dropped in galaxies:
            parser.error(
                f"--leave-one-out-dropped {args.leave_one_out_dropped} "
                "is also being analysed; exclude it from --galaxy.")
    if args.num_warmup < 0 or args.num_samples < 1:
        parser.error("--num-warmup must be >= 0 and --num-samples > 0.")
    if args.toy_distance_n_grid < 16:
        parser.error("--toy-distance-n-grid must be >= 16.")
    if "NGC4258" in galaxies:
        parser.error("joint H0 inference always excludes NGC4258; "
                     "remove it from --galaxy.")
    try:
        flat_dist = _resolve_distance_prior(
            args.selection, args.distance_prior)
    except ValueError as exc:
        parser.error(str(exc))

    velocity_beta = _resolve_velocity_beta(
        args.reconstruction, args.velocity_beta)
    # Sample beta only for the Carrick/2M++ field (linear-theory f/b
    # degeneracy); an explicit --velocity-beta pins it instead.
    sample_velocity_beta = (args.reconstruction == "Carrick2015"
                            and args.velocity_beta is None)
    volume_fraction = _resolve_volume_subsample_fraction(
        args.reconstruction, args.volume_subsample_fraction,
        _joint_volume_subsample_defaults())
    velocity_data = None
    volume_data = None
    if args.reconstruction != "none":
        r_vlos = np.arange(args.vlos_rmin, args.vlos_rmax + 0.5 * args.vlos_dr,
                           args.vlos_dr, dtype=np.float32)
        stub = []
        for galaxy in galaxies:
            gcfg = MASTER_CFG["model"]["galaxies"][galaxy]
            stub.append({
                "name": galaxy,
                "RA": float(gcfg["ra"]),
                "dec": float(gcfg["dec"]),
            })
        velocity_data = _load_or_build_vlos_cache(
            args.reconstruction, args.field_config, args.field_indices,
            r_vlos, stub,
            velocity_smoothing_scale=args.velocity_smoothing_scale,
            overwrite=args.overwrite_vlos_cache)
        if args.selection != "none":
            volume_data = _load_volume_selection_data(
                args.reconstruction, args.field_config,
                velocity_data["field_indices"],
                args.selection_integral_radius,
                args.selection_integral_geometry,
                volume_fraction,
                args.volume_subsample_seed,
                args.field_smoothing_scale,
                args.velocity_smoothing_scale,
                args.volume_supersample_factor,
                args.volume_supersample_radius,
                (None if np.isclose(args.volume_supersample_target_dx, 0.0)
                 else args.volume_supersample_target_dx))

    sample_vext = args.Vext
    vext_lo, vext_hi = _joint_vext_mag_bounds()
    if args.vext_prior_scale is not None:
        vext_hi = args.vext_prior_scale
    joint_priors = {
        "H0": dist.Uniform(10.0, 200.0),
        "sigma_pec": _joint_sigma_pec_prior(),
        "D_lim": dist.Uniform(15.0, 1000.0),
        "D_width": dist.Uniform(15.0, 500.0),
        "cz_lim_selection": dist.Uniform(500.0, 20000.0),
        "cz_lim_selection_width": dist.Uniform(50.0, 10000.0),
        "Vext_mag": dist.Uniform(vext_lo, vext_hi),
    }
    los_nside = _joint_selection_nside()
    vext_prior = (_joint_vext_prior(args.reconstruction)
                  if sample_vext else None)

    fsection("Loading toy distance likelihoods")
    items = _build_toy_items(galaxies, args, velocity_data=velocity_data)
    target = ToyDistanceTarget(
        items, joint_priors, args.selection, volume_data, flat_dist,
        velocity_beta, sample_velocity_beta, sample_vext, los_nside,
        vext_prior=vext_prior)
    init = _toy_init(target)

    fsection("Toy joint megamaser H0")
    fprint(f"galaxies: {', '.join(galaxies)}")
    if args.leave_one_out_dropped is not None:
        fprint(f"leave-one-out: dropped {args.leave_one_out_dropped}")
    prior_name = "uniform in distance" if flat_dist else "uniform in volume"
    beta_state = (f"N({velocity_beta:g}, 0.02) sampled"
                  if sample_velocity_beta else f"{velocity_beta:g} fixed")
    fprint(f"selection model: {args.selection}; reconstruction: "
           f"{args.reconstruction}; beta={beta_state}")
    fprint(f"distance prior: {prior_name}; Vext sampled: {sample_vext}")
    if vext_prior is not None:
        fprint(f"Vext prior: informative ({args.reconstruction}); "
               f"|Vext|={vext_prior['mag']:.0f}+/-"
               f"{vext_prior['mag_sigma']:.0f} km/s, (l,b)=("
               f"{vext_prior['ell']:.0f},{vext_prior['b']:.0f}) deg")
        if args.vext_prior_scale is not None:
            fprint("note: --vext-prior-scale ignored under informative "
                   "Vext prior")
    elif sample_vext:
        fprint("Vext prior: uniform (no informative prior configured)")
    fprint(f"distance likelihood: KDE over saved single-galaxy D_A "
           f"samples; grid={args.toy_distance_n_grid}; "
           f"max_samples={args.toy_distance_max_samples}")
    fprint(f"JAX backend: {jax.default_backend()}; precision: "
           f"{'float64' if jax.config.jax_enable_x64 else 'float32'}")
    fprint(f"warmup={args.num_warmup}; samples={args.num_samples}; "
           f"chains={args.num_chains}")
    fprint(f"target_accept_theta={args.target_accept_theta}; "
           f"initial_step_size={args.initial_step_size}")
    dist_sites = [site for site in target.theta_sites
                  if site.endswith("__D_c")]
    h0_dist_block = ", ".join(("H0", *dist_sites))
    fprint(f"theta mass matrix: dense over {len(target.theta_sites)} "
           f"sites; H0-distance block: {h0_dist_block}")

    result = run_toy_mcmc(
        target, init, random.PRNGKey(args.seed),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        target_accept_theta=args.target_accept_theta,
        theta_step_init=args.initial_step_size,
        max_num_doublings=args.max_tree_depth)

    variant = ("_ecc" if args.add_ecc else "") + (
        "_qw" if args.add_quadratic_warp else "")
    outpath = args.output or _result_path(
        galaxies, args.selection, args.reconstruction, flat_dist, variant,
        toy=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    _save_hdf5(outpath, result, {
        "sampler": "numpyro_joint_toy_distance_mcmc",
        "chain_method": "vectorized",
        "toy_distances": True,
        "galaxies": ",".join(galaxies),
        "dataset": str(MASTER_CFG["io"]["dataset"]),
        "selection": args.selection,
        "reconstruction": args.reconstruction,
        "use_ecc": bool(args.add_ecc),
        "use_quadratic_warp": bool(args.add_quadratic_warp),
        "velocity_beta": velocity_beta,
        "sample_velocity_beta": sample_velocity_beta,
        "sample_vext": sample_vext,
        "flat_dist": bool(flat_dist),
        "distance_prior": prior_name,
        "selection_los_nside": int(los_nside),
        "toy_distance_files": ",".join(
            item["toy_distance_file"] for item in items),
        "toy_distance_kde_bandwidths": ",".join(
            f"{item['toy_distance_kde_bandwidth']:.12g}"
            for item in items),
        "toy_distance_kde_samples": ",".join(
            str(item["toy_distance_kde_samples"]) for item in items),
        "toy_distance_source_outside_support": ",".join(
            str(item["toy_distance_source_outside_support"])
            for item in items),
        "toy_distance_n_grid": int(args.toy_distance_n_grid),
        "toy_distance_max_samples": int(args.toy_distance_max_samples),
        "runtime_seconds": float(result.runtime_seconds),
        "num_warmup": int(args.num_warmup),
        "num_samples": int(args.num_samples),
        "num_chains": int(args.num_chains),
        "seed": int(args.seed),
        "target_accept_theta": float(args.target_accept_theta),
        "initial_step_size": float(args.initial_step_size),
        "max_tree_depth": int(args.max_tree_depth),
        "theta_mass_matrix": "dense",
        "theta_distance_sites": ",".join(dist_sites),
    })
    fprint(f"saved samples to {outpath}")
    _print_summary(result)
    _print_toy_support_diagnostics(result, items)
    try:
        _toy_distance_kde_plots(items, outpath)
    except Exception as exc:                       # plots are non-critical
        fprint(f"toy distance KDE plotting failed: {exc}")
    try:
        _corner_plots(result, galaxies, outpath, args.selection,
                      sample_vext)
    except Exception as exc:                       # plots are non-critical
        fprint(f"corner plotting failed: {exc}")


if __name__ == "__main__":
    main()
