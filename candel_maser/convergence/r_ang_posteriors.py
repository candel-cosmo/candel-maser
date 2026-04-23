"""
Per-spot 1D posterior on r_ang after marginalising phi, with all global
parameters pinned to the `init` block in config_maser.toml. No sampling;
this is a diagnostic pass through the Mode 1 phi-marginal at a single
point in global-parameter space.

For each galaxy: tile a shared r_ang grid across all spots, run
_r_precompute + _phi_eval, logsumexp over phi to get
log p(d_i | r_i, globals), normalise each spot's curve by trapezoidal
integration on r, and overlay the posteriors by spot class.

Usage:
    python scripts/megamaser/convergence/r_ang_posteriors.py \\
        [--galaxies UGC3789 NGC6323] [--n-r 501] [--out PATH]
"""
import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tomli
from jax.scipy.special import logsumexp

from candel.model.integration import trapz_log_weights
from candel.model.maser_convergence import build_model

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
DEFAULT_GALAXIES = ["UGC3789", "NGC6323"]
TYPES = ("sys", "red", "blue")
TYPE_COLORS = {"sys": "black", "red": "crimson", "blue": "royalblue"}
PHI_KEYS = ("n_phi_hv_high", "n_phi_hv_low", "n_phi_sys")


def _phys_from_init(model, galaxies_cfg, galaxy):
    init = galaxies_cfg[galaxy]["init"]
    sample = {k: np.asarray(v) for k, v in init.items()}
    return model.phys_from_sample(sample)


def _log_posterior_on_grid(model, type_key, idx, r_grid,
                           phys_args, phys_kw, r_batch):
    """log p(d_i | r, globals) on (N_group, n_r), unnormalised.

    r is chunked in slices of ``r_batch`` so the (N, n_r, n_phi)
    intermediate stays inside GPU memory; the sys channel's
    concatenated φ grid (2 × n_phi_sys) makes this necessary for larger
    galaxies.
    """
    n = int(idx.shape[0])
    if n == 0:
        return None
    n_r = int(r_grid.shape[0])
    has_any_accel = model._group_has_any_accel(type_key, shared_r=False)
    pc = model._phi_concat[type_key]
    parts = []
    for s in range(0, n_r, r_batch):
        r_chunk = r_grid[s:s + r_batch]
        r_ang_2d = jnp.broadcast_to(
            r_chunk[None, :], (n, r_chunk.shape[0]))
        r_pre = model._r_precompute(
            r_ang_2d, idx, *phys_args, **phys_kw,
            has_any_accel=has_any_accel)
        nhc = model._phi_eval(r_pre, pc["sin_phi"], pc["cos_phi"])
        chunk = logsumexp(nhc + pc["log_w_phi"], axis=-1)  # (N, r_chunk)
        # lnorm / lnorm_a are r-independent but _r_precompute returns
        # them once per call; take from the first chunk only.
        if not parts:
            lnorm_total = r_pre["lnorm"] + r_pre["lnorm_a"]
        parts.append(chunk)
    log_L = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)
    return log_L + lnorm_total[:, None]


def _normalise(log_L, r_grid):
    """Per-row log p(r | d) summing to 1 under trapezoidal integration."""
    log_w_r = trapz_log_weights(r_grid)
    log_Z = logsumexp(log_L + log_w_r[None, :], axis=-1, keepdims=True)
    return log_L - log_Z


def _phi_overrides(galaxy, master_cfg, f_grid):
    """Base phi grid size comes from the per-galaxy block if set, else
    from the global [model]; scaled by f_grid and rounded to the
    nearest odd int >= 3. Passed to build_model so the per-galaxy
    override is honoured (build_model strips it otherwise)."""
    m = master_cfg["model"]
    gblk = master_cfg["model"]["galaxies"][galaxy]
    out = {}
    for k in PHI_KEYS:
        base = int(gblk[k]) if k in gblk else int(m[k])
        n = int(round(base * f_grid))
        n = max(3, n if n % 2 else n + 1)
        out[k] = n
    return out


def compute_posteriors(galaxy, master_cfg, n_r, r_batch, f_grid):
    galaxies_cfg = master_cfg["model"]["galaxies"]
    overrides = _phi_overrides(galaxy, master_cfg, f_grid)
    print(f"  φ grid: {overrides}", flush=True)
    model = build_model(galaxy, master_cfg, mode="mode1", **overrides)
    phys_args, phys_kw, diag = _phys_from_init(
        model, galaxies_cfg, galaxy)
    D_A, M_BH, v_sys = diag["D_A"], diag["M_BH"], diag["v_sys"]

    r_lo, r_hi = model.r_ang_range(D_A)
    r_grid = jnp.exp(jnp.linspace(jnp.log(r_lo), jnp.log(r_hi), n_r))

    # r_est from the model's centering scheme (Mode 2 per-spot sinh grid
    # centre); aligns with model._idx_<type>.
    r_est, _, _, _ = model._estimate_adaptive_r(
        D_A, M_BH, v_sys, phys_args[16], phys_args[8], phys_args[15])
    r_est = np.asarray(r_est)

    out = dict(r_grid=np.asarray(r_grid),
               D_A=float(D_A), n_spots=model.n_spots)
    for type_key in TYPES:
        idx = getattr(model, f"_idx_{type_key}")
        log_L = _log_posterior_on_grid(
            model, type_key, idx, r_grid, phys_args, phys_kw, r_batch)
        if log_L is None:
            out[type_key] = None
            out[f"{type_key}_rest"] = None
            continue
        log_post = _normalise(log_L, r_grid)
        out[type_key] = np.asarray(jnp.exp(log_post))
        out[f"{type_key}_rest"] = r_est[np.asarray(idx)]
    return out


def _posterior_moments(r, p):
    """Posterior mean and std on a 1D r grid via trapezoidal integration."""
    mean = float(np.trapezoid(r * p, r))
    var = float(np.trapezoid((r - mean) ** 2 * p, r))
    return mean, np.sqrt(max(var, 0.0))


def plot(results, out_path, n_show, seed, n_sigma=5.0):
    n_gal = len(results)
    fig, axes = plt.subplots(
        len(TYPES), n_gal, figsize=(5.0 * n_gal, 8.5),
        squeeze=False)
    rng = np.random.default_rng(seed)
    cmap = plt.get_cmap("tab10")
    for j, (galaxy, res) in enumerate(results.items()):
        r = res["r_grid"]
        for i, tkey in enumerate(TYPES):
            ax = axes[i, j]
            post = res[tkey]
            rest = res.get(f"{tkey}_rest")
            if post is None:
                ax.text(0.5, 0.5, "no spots",
                        transform=ax.transAxes, ha="center", va="center")
            else:
                n = post.shape[0]
                k = min(n_show, n)
                sel = rng.choice(n, size=k, replace=False)
                panel_lo, panel_hi = [], []
                for c_idx, row_idx in enumerate(sel):
                    colour = cmap(c_idx % 10)
                    p_row = post[row_idx]
                    ax.plot(r, p_row, color=colour, alpha=0.9, lw=1.2)
                    if rest is not None:
                        ax.axvline(float(rest[row_idx]), color=colour,
                                   linestyle="--", alpha=0.7, lw=0.9)
                    m, s = _posterior_moments(r, p_row)
                    panel_lo.append(m - n_sigma * s)
                    panel_hi.append(m + n_sigma * s)
                ax.text(0.97, 0.95, f"N={n} ({k} shown)",
                        transform=ax.transAxes, ha="right", va="top")
                xlo = max(float(r[0]), min(panel_lo))
                xhi = min(float(r[-1]), max(panel_hi))
                if galaxy != "NGC4258":
                    xhi = min(xhi, 1.0)
                if xhi > xlo:
                    ax.set_xlim(xlo, xhi)
            ax.set_xlabel(r"$r_\mathrm{ang}$ [mas]")
            if i == 0:
                ax.set_title(
                    f"{galaxy}  "
                    r"($D_\mathrm{A}=$" + f"{res['D_A']:.1f}" + " Mpc)")
            if j == 0:
                ax.set_ylabel(
                    f"{tkey}: "
                    r"$p(r_\mathrm{ang}\,|\,d_i)$ "
                    r"[$\mathrm{mas}^{-1}$]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxies", nargs="+", default=None,
                        help=f"Galaxies to process (default: {DEFAULT_GALAXIES}).")
    parser.add_argument("--include-ngc4258", action="store_true",
                        help="Append NGC4258 to the default list. Ignored "
                             "if --galaxies is given explicitly.")
    parser.add_argument("--n-r", type=int, default=2001)
    parser.add_argument(
        "--r-batch", type=int, default=64,
        help="Chunk size on the r axis. Lower if the sys channel OOMs "
             "on a small GPU (default: 64).")
    parser.add_argument(
        "--f-grid", type=float, default=1.0,
        help="Scale factor for the phi grid sizes (n_phi_hv_high, "
             "n_phi_hv_low, n_phi_sys). Rounded to the nearest odd "
             "integer >= 3. Default 1.0 (config values).")
    parser.add_argument(
        "--n-show", type=int, default=8,
        help="Number of spots per type to plot (random sample).")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the spot selection.")
    parser.add_argument(
        "--out", type=str,
        default=("/mnt/users/rstiskalek/CANDEL/results/Megamaser/"
                 "convergence/r_ang_posteriors.png"))
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    print(f"JAX platform: {jax.default_backend()}, precision: float64",
          flush=True)

    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)

    if args.f_grid != 1.0:
        print(f"f-grid={args.f_grid:g}", flush=True)

    if args.galaxies is None:
        galaxies = list(DEFAULT_GALAXIES)
        if args.include_ngc4258:
            galaxies.append("NGC4258")
    else:
        galaxies = args.galaxies

    results = {}
    for galaxy in galaxies:
        print(f"\n── {galaxy} ──", flush=True)
        res = compute_posteriors(
            galaxy, master_cfg, args.n_r, args.r_batch, args.f_grid)
        print(f"  D_A = {res['D_A']:.2f} Mpc, n_spots = {res['n_spots']}, "
              f"r_ang grid: [{res['r_grid'][0]:.4f}, "
              f"{res['r_grid'][-1]:.4f}] mas, n_r = {args.n_r}",
              flush=True)
        results[galaxy] = res

    plot(results, args.out, args.n_show, args.seed)


if __name__ == "__main__":
    main()
