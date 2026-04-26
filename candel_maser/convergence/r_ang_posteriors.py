"""
Per-spot 1D posterior on r_ang after marginalising phi, with all global
parameters pinned to the `init` block in config_maser.toml. No sampling
— a diagnostic pass through the Mode 1 phi-marginal at a single point
in global-parameter space.

For each galaxy: tile a shared dense r_ang grid across all spots, run
_r_precompute + _phi_eval, logsumexp over phi to get
log p(d_i | r_i, globals), normalise each spot's curve by trapezoidal
integration on r, and overlay the posteriors by spot class.

Usage:
    python scripts/megamaser/convergence/r_ang_posteriors.py \\
        [--galaxies UGC3789 NGC6323] [--n-r 501] [--out PATH]
"""
import argparse
import logging
import os

# Silence the JAX CUDA-plugin init warning on CPU-only nodes.
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tomli  # noqa: E402
from jax.scipy.special import logsumexp  # noqa: E402

from candel.model.integration import trapz_log_weights  # noqa: E402
from candel.model.maser_convergence import build_model  # noqa: E402

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
DEFAULT_GALAXIES = ["UGC3789", "NGC6323"]
TYPES = ("sys", "red", "blue")
PHI_KEYS = ("n_phi_hv_high", "n_phi_hv_low", "n_phi_sys")


def _phys_from_init(model, galaxies_cfg, galaxy):
    init = galaxies_cfg[galaxy]["init"]
    sample = {k: np.asarray(v) for k, v in init.items()}
    return model.phys_from_sample(sample)


def _log_posterior_on_grid(model, type_key, idx, r_grid,
                           phys_args, phys_kw, r_batch):
    """log p(d_i | r, globals) on (N_group, n_r), unnormalised.

    r is chunked in slices of ``r_batch`` so the (N, n_r, n_phi)
    intermediate stays inside GPU memory. The chunk evaluator is
    jit-compiled and the r grid is padded up to a multiple of
    ``r_batch`` so every call hits the same compiled trace.
    """
    n = int(idx.shape[0])
    if n == 0:
        return None
    n_r = int(r_grid.shape[0])
    has_any_accel = model._group_has_any_accel(type_key)
    pc = model._phi_concat[type_key]

    @jax.jit
    def _one_chunk(r_chunk):
        r_ang_2d = jnp.broadcast_to(
            r_chunk[None, :], (n, r_chunk.shape[0]))
        r_pre = model._r_precompute(
            r_ang_2d, idx, *phys_args, **phys_kw,
            has_any_accel=has_any_accel)
        nhc = model._phi_eval(r_pre, pc["sin_phi"], pc["cos_phi"])
        chunk = logsumexp(nhc + pc["log_w_phi"], axis=-1)
        lnorm_total = r_pre["lnorm"] + r_pre["lnorm_a"]
        return chunk, lnorm_total

    pad = (-n_r) % r_batch
    if pad:
        r_padded = jnp.concatenate(
            [r_grid, jnp.full((pad,), r_grid[-1])])
    else:
        r_padded = r_grid

    parts = []
    lnorm_total = None
    n_total = int(r_padded.shape[0])
    for s in range(0, n_total, r_batch):
        chunk, lnt = _one_chunk(r_padded[s:s + r_batch])
        if lnorm_total is None:
            lnorm_total = lnt
        parts.append(chunk)
    log_L = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)
    if pad:
        log_L = log_L[..., :n_r]
    return log_L + lnorm_total[:, None]


def _normalise(log_L, r_grid):
    """Per-row log p(r | d) summing to 1 under trapezoidal integration."""
    log_w_r = trapz_log_weights(r_grid)
    log_Z = logsumexp(log_L + log_w_r[None, :], axis=-1, keepdims=True)
    return log_L - log_Z


def _phi_overrides(galaxy, master_cfg, f_grid):
    """Base phi grid size from the per-galaxy block if set, else from
    global [model]; scaled by f_grid and rounded to the nearest odd int
    ≥ 3. Passed to build_model so the per-galaxy override is honoured
    (build_model strips it otherwise)."""
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
    """Evaluate per-spot r posteriors + Mode 2 centring lines."""
    galaxies_cfg = master_cfg["model"]["galaxies"]
    overrides = _phi_overrides(galaxy, master_cfg, f_grid)
    print(f"  φ grid: {overrides}", flush=True)
    model = build_model(galaxy, master_cfg, mode="mode1", **overrides)
    if galaxies_cfg[galaxy].get("forbid_marginalise_r", False):
        centres = None
    else:
        model_m2 = build_model(
            galaxy, master_cfg, mode="mode2", **overrides)
        phys_args_m2, phys_kw_m2, _ = _phys_from_init(
            model_m2, galaxies_cfg, galaxy)
        centres = model_m2.get_mode2_centres(phys_args_m2, phys_kw_m2)

    phys_args, phys_kw, diag = _phys_from_init(
        model, galaxies_cfg, galaxy)
    D_A = diag["D_A"]

    r_lo, r_hi = model.r_ang_range(D_A)
    r_grid = jnp.exp(jnp.linspace(jnp.log(r_lo), jnp.log(r_hi), n_r))

    out = dict(r_grid=np.asarray(r_grid),
               D_A=float(D_A), n_spots=model.n_spots)

    for type_key in TYPES:
        idx = getattr(model, f"_idx_{type_key}")
        log_L = _log_posterior_on_grid(
            model, type_key, idx, r_grid, phys_args, phys_kw, r_batch)
        if log_L is None:
            out[type_key] = None
            out[f"{type_key}_rc"] = None
            continue
        log_post = _normalise(log_L, r_grid)
        out[type_key] = np.asarray(jnp.exp(log_post))
        if centres is not None and centres[type_key] is not None:
            out[f"{type_key}_rc"] = np.asarray(centres[type_key]["r_c"])
        else:
            out[f"{type_key}_rc"] = None
    return out


def _posterior_moments(r, p):
    """Posterior mean and std on a 1D r grid via trapezoidal integration."""
    mean = float(np.trapezoid(r * p, r))
    var = float(np.trapezoid((r - mean) ** 2 * p, r))
    return mean, np.sqrt(max(var, 0.0))


def select_spots(results, n_show, seed):
    """Pick up to ``n_show`` row indices per (galaxy, type).

    Returns ``{galaxy: {type_key: np.ndarray[int]}}`` where the integers
    index into the per-type group (i.e. into ``model._idx_<type>``).
    """
    rng = np.random.default_rng(seed)
    sel = {}
    for galaxy, res in results.items():
        sel[galaxy] = {}
        for tkey in TYPES:
            post = res[tkey]
            if post is None:
                sel[galaxy][tkey] = np.empty(0, dtype=int)
                continue
            n = int(post.shape[0])
            k = min(n_show, n)
            sel[galaxy][tkey] = np.sort(
                rng.choice(n, size=k, replace=False))
    return sel


def plot(results, selection, out_path, n_sigma=5.0):
    n_gal = len(results)
    fig, axes = plt.subplots(
        len(TYPES), n_gal, figsize=(5.0 * n_gal, 8.5),
        squeeze=False)
    cmap = plt.get_cmap("tab10")
    for j, (galaxy, res) in enumerate(results.items()):
        r = res["r_grid"]
        for i, tkey in enumerate(TYPES):
            ax = axes[i, j]
            post = res[tkey]
            rc = res.get(f"{tkey}_rc")
            if post is None:
                ax.text(0.5, 0.5, "no spots",
                        transform=ax.transAxes, ha="center", va="center")
            else:
                n = post.shape[0]
                sel = selection[galaxy][tkey]
                k = int(sel.shape[0])
                panel_lo, panel_hi = [], []
                for c_idx, row_idx in enumerate(sel):
                    colour = cmap(c_idx % 10)
                    p_row = post[row_idx]
                    ax.plot(r, p_row, color=colour, alpha=0.9, lw=1.2)
                    if rc is not None:
                        rc_i = float(rc[row_idx])
                        if np.isfinite(rc_i):
                            ax.axvline(rc_i, color=colour,
                                       linestyle=":", alpha=0.9, lw=1.1)
                    m, sigma = _posterior_moments(r, p_row)
                    panel_lo.append(m - n_sigma * sigma)
                    panel_hi.append(m + n_sigma * sigma)
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
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="k", linestyle=":",
               lw=1.1, alpha=0.9, label=r"$r_c$ (Mode 2 centre)"),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="upper left",
                      frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--galaxies", nargs="+", default=None,
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

    selection = select_spots(results, args.n_show, args.seed)
    plot(results, selection, args.out)


if __name__ == "__main__":
    main()
