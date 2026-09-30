"""Fixed-globals latent-MAP / chi^2 diagnostic for the megamaser disk fit.

Fix the disk globals (to the DE/config solution, or to Reid/Pesce published
values) and optimise the per-spot latents ``(r_ang, phi)`` only.  With the
globals fixed the spots are conditionally independent, so a single pass of
per-spot 2D optimisations (Nelder-Mead, seeded in *both* phi halves) is the
exact joint latent optimum -- no outer iteration.

``lnorm`` (the Gaussian normalisation) is latent-independent, so the same
optimal latents serve the log-posterior (MAP) and the pure chi^2; both are read
off at that point.  Evaluating at Reid's globals reproduces their profile chi^2
for a like-for-like comparison with the DE solution.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .maser_blackjax import _initial_phi, _phi_support_arrays


def _perspot_neg_half_chi2(model, theta_complete, r_ang, phi, h):
    """Per-spot ``-0.5*chi^2`` (Gaussian norm removed), length n_spots.

    ``_fixed_phi_per_spot`` returns ``lnorm - 0.5*chi^2``; ``lnorm`` from
    ``_r_precompute`` is subtracted so only the data-fit term survives.
    """
    phys_args, phys_kw = model.phys_from_params_jax(theta_complete, h)
    out = jnp.zeros(model.n_spots, dtype=jnp.asarray(r_ang).dtype)
    for type_key, idx, r_g, _ in model._spot_groups_from_r(r_ang):
        if int(idx.shape[0]) == 0:
            continue
        has = model._group_has_any_accel(type_key)
        per = model._fixed_phi_per_spot(idx, r_g, phi[idx], has,
                                        phys_args, phys_kw)
        rpre = model._r_precompute(r_g, idx, *phys_args, **phys_kw,
                                   has_any_accel=has)
        out = out.at[idx].set(per - (rpre["lnorm"] + rpre["lnorm_a"]))
    return out


def marginal_loglik(target, theta, spot_batch=8):
    """Latent-marginalised data log-likelihood at fixed globals.

    Mirrors ``run_de_map._logp_2d_terms``: build the conditional r-grids and
    integrate (r, phi) jointly per spot.  The magic ``phys_args`` indices match
    that function.
    """
    model = target.model
    theta = target.complete_params(theta)
    phys_args, phys_kw = model.phys_from_params_jax(theta, target.h)
    groups = model._build_conditional_r_grids(
        phys_args[2], phys_args[3], phys_args[4], phys_args[16],
        phys_args[8], phys_args[15], phys_args, phys_kw)
    ll = model._sum_phi_marginal(
        groups, phys_args, phys_kw, spot_batch=spot_batch, remat=False)
    return float(ll)


def _make_perspot_eval(target):
    """JITed per-spot fixed-phi log-likelihood vector (length n_spots).

    Takes the constrained-globals dict directly (no unconstrain round-trip), so
    it works even for published globals outside CANDEL's prior support (e.g.
    Reid's tight error floors); only the prior/marginal are then undefined.
    """
    model = target.model

    @jax.jit
    def perspot(r_vec, phi_vec, theta):
        th = target.complete_params(theta)
        phys_args, phys_kw = model.phys_from_params_jax(th, target.h)
        groups = model._spot_groups_from_r(r_vec)
        return model._eval_phi_fixed(groups, phi_vec, phys_args, phys_kw)

    return perspot


def _optimise_latents(perspot, theta, r_ang, phi, lo, hi, ctr, nm_opts,
                      n_restarts=5, rng=None):
    """One pass of per-spot 2D optimisation (globals fixed via ``u``).

    Each spot's (r, phi) is optimised independently, from the current value
    (baseline, so the step never worsens a spot), each phi half's midpoint, and
    ``n_restarts`` random starts (phi uniform in the box, log r jittered).  The
    best over all starts is kept.  Globals fixed + spots independent => one
    pass is essentially exact; ``rng`` advances across passes for fresh starts.
    """
    # ponytail: each per-spot eval recomputes all n_spots (jitted, cheap on
    # CPU); if this dominates on large galaxies, gather spot i's data once and
    # jit a scalar objective instead.
    if rng is None:
        rng = np.random.default_rng(0)
    r_vec = np.asarray(r_ang).copy()
    phi_vec = np.asarray(phi).copy()
    for i in range(r_vec.shape[0]):
        a_lo, a_hi, a_ctr = float(lo[i]), float(hi[i]), float(ctr[i])
        logr_i = np.log(max(r_vec[i], 1e-6))

        def neg(v, i=i, a_lo=a_lo, a_hi=a_hi):
            if not (a_lo <= v[1] <= a_hi):
                return 1e18
            rr = r_vec.copy()
            pp = phi_vec.copy()
            rr[i] = np.exp(v[0])
            pp[i] = v[1]
            return -float(perspot(jnp.asarray(rr), jnp.asarray(pp), theta)[i])

        starts = [[logr_i, 0.5 * (a_lo + a_ctr)],   # lower phi half
                  [logr_i, 0.5 * (a_ctr + a_hi)]]   # upper phi half
        for _ in range(int(n_restarts)):
            starts.append([logr_i + rng.normal(0.0, 0.3),
                           rng.uniform(a_lo, a_hi)])

        best_x = np.array([logr_i, phi_vec[i]])
        best_f = neg(best_x)
        for x0 in starts:
            res = minimize(neg, np.asarray(x0), method="Nelder-Mead",
                           options=nm_opts)
            if res.fun < best_f:
                best_x, best_f = res.x, res.fun
        r_vec[i] = np.exp(best_x[0])
        phi_vec[i] = best_x[1]
    return r_vec, phi_vec


def _point_dict(theta):
    return {k: float(np.asarray(v)) for k, v in theta.items()
            if np.asarray(v).ndim == 0}


def _log_prior(target, theta):
    """Global log-prior (mirrors run_de_map._logp_2d_terms)."""
    lp = 0.0
    for site, _, prior in target.sites:
        if site == "eta":
            continue
        lp = lp + float(prior.log_prob(theta[site]))
    if target.mass_parameterization == "eta":
        lp = lp + float(target.model.priors["log_MBH"].log_prob(
            theta["log_MBH"]))
    return lp


def _dof_bookkeeping(target):
    model = target.model
    n_spots = int(model.n_spots)
    n_accel = int(np.asarray(model._all_has_accel).sum())
    n_global = len(target.names)
    n_used = 3 * n_spots + n_accel
    # Reid counts globals + per-spot (r, phi) as free params even when we hold
    # the globals fixed, so chi^2/dof is comparable to their reduced chi^2.
    n_params = n_global + 2 * n_spots
    return dict(n_spots=n_spots, n_accel=n_accel, n_global=n_global,
                n_used=n_used, n_params=n_params, dof=n_used - n_params)


def evaluate_at_globals(target, globals_point, *, init_r_ang=None,
                        nm_maxiter=400, n_restarts=5, seed=0,
                        marginal=True, verbose=True):
    """Hold globals fixed; optimise per-spot (r_ang, phi); report chi^2 & MAP.

    ``globals_point`` is a dict over ``target.names``.  ``init_r_ang`` (length
    n_spots) seeds the radii; without it the conditional r-MAP at these globals
    is used.  Returns a dict with the globals ``point``, optimised ``r_ang`` /
    ``phi``, total ``chi2`` and ``chi2_per_dof``, the log-posterior ``logpost``
    at that point, the marginal ``ln_marg`` at these globals, and dof
    bookkeeping.
    """
    model = target.model
    # Build theta directly from the fixed globals -- no unconstrain round-trip,
    # so out-of-prior published globals still yield a well-defined chi^2.
    use_x64 = jax.config.jax_enable_x64
    dtype = jnp.float64 if use_x64 else jnp.float32
    np_dtype = np.float64 if use_x64 else np.float32
    gp = {n: jnp.asarray(globals_point[n], dtype=dtype) for n in target.names}
    theta = target.complete_params(gp)
    phys_args, phys_kw = model.phys_from_params_jax(theta, target.h)

    if init_r_ang is not None:
        r_ang = np.asarray(init_r_ang, dtype=np_dtype)
    else:
        r_ang = np.asarray(model.conditional_r_ang_map(phys_args, phys_kw),
                           dtype=np_dtype)
    phi = np.asarray(_initial_phi(model, dtype), dtype=np_dtype)
    lo, hi, ctr = (np.asarray(a, dtype=np_dtype)
                   for a in _phi_support_arrays(model, dtype))

    perspot = _make_perspot_eval(target)
    nm_opts = dict(maxiter=nm_maxiter, xatol=1e-7, fatol=1e-7)
    rng = np.random.default_rng(seed)

    def _chi2(r, p):
        return float(-2.0 * np.asarray(_perspot_neg_half_chi2(
            model, theta, jnp.asarray(r), jnp.asarray(p), target.h)).sum())

    # Spots are independent given globals, so this converges in a couple of
    # passes; extra passes only mop up local Nelder-Mead under-convergence.
    chi2 = np.inf
    for k in range(6):
        r_ang, phi = _optimise_latents(perspot, theta, r_ang, phi, lo, hi, ctr,
                                       nm_opts, n_restarts=n_restarts, rng=rng)
        cur = _chi2(r_ang, phi)
        if verbose:
            print(f"  pass {k}: chi2={cur:.3f}", flush=True)
        if chi2 - cur < 1e-2:
            chi2 = cur
            break
        chi2 = cur
    point = _point_dict(theta)

    # The globals comparison uses the *marginal* log-posterior (log-prior +
    # sum_i 2D (r,phi) marginal) -- the same object as run_de_map's Pesce/Reid
    # baseline.  NOT the profile fixed-phi value, which mixes chi^2 with the
    # error-floor lnorm normalisation and can rank a worse-fitting disc higher.
    logP_marg = None
    if marginal:
        try:
            lp = _log_prior(target, theta) + marginal_loglik(target, theta)
            logP_marg = float(lp) if np.isfinite(lp) else None
        except Exception as exc:                       # noqa: BLE001
            if verbose:
                print(f"  marginal logP unavailable: {exc}", flush=True)

    info = _dof_bookkeeping(target)
    dof = info["dof"]
    return dict(
        point=point, r_ang=np.asarray(r_ang), phi=np.asarray(phi),
        D_A=float(np.asarray(phys_args[2])),
        chi2=chi2, chi2_per_dof=(chi2 / dof if dof > 0 else float("nan")),
        logP_marg=logP_marg, **info)
