"""Per-spot (r_ang, phi) likelihood maps for NGC5765b, illustrating the
front/back azimuthal bimodality that the production inference marginalises.

Globals are pinned at the fiducial linear-warp posterior median. For each
maser spot the data likelihood p(d_i | r_ang, phi, theta) is evaluated on a
dense uniform (r_ang, phi) grid via the model's own _phi_integrand (the same
kernel the sampler uses), then normalised to a 2D PDF under flat latent priors.

Two modes:
  survey (default): evaluate every spot, score its phi bimodality, print a
    ranked table, and save a contact sheet of the top candidates.
  final (--spots i j k): clean three-panel figure (PNG + PDF) for the paper.

Run from the CANDEL repo root with the project venv:
    venv_candel/bin/python papers/MMH0/plot_rphi_bimodality.py \
        [--spots ...]
"""
import argparse
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))

import h5py  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402,F401  (registers the "science" style)
import tomli  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

from candel_maser.convergence.convergence_utils import (build_model)  # noqa: E402
from candel_maser.maser_config import (apply_dataset, check_chain_dataset)  # noqa: E402

from candel_maser.paths import CANDEL_ROOT, CONFIG_PATH, RESULTS_ROOT  # noqa: E402
GALAXY = "NGC5765b"
DATASET = "original_published"
HDF5 = os.path.join(
    RESULTS_ROOT, f"results/Megamaser/{DATASET}/{GALAXY}/"
    f"{GALAXY}_blackjax_mcmc_rphi_initconfig.hdf5")
# Global keys whose posterior median pins the disc; eta drives the mass under
# mass_parameterization == "eta".
GLOBAL_KEYS = ("D_A", "D_c", "eta", "log_MBH", "i0", "di_dr", "Omega0",
               "dOmega_dr",
               "x0", "y0", "dv_sys", "sigma_x_floor", "sigma_y_floor",
               "sigma_v_sys", "sigma_v_hv", "sigma_a_floor",
               "sigma_x_floor_clump2", "sigma_y_floor_clump2",
               "sigma_v_floor_clump2", "sigma_a_floor_clump2")


def median_globals(path):
    with h5py.File(path, "r") as f:
        check_chain_dataset(f.attrs, DATASET, path)
        s = f["samples"]
        return {k: float(np.median(np.asarray(s[k]).ravel()))
                for k in GLOBAL_KEYS if k in s}


def spot_class(model):
    """Per-spot class label array of length n_spots."""
    lab = np.empty(model.n_spots, dtype=object)
    lab[np.asarray(model._idx_sys)] = "systemic"
    lab[np.asarray(model._idx_blue)] = "blue"
    lab[np.asarray(model._idx_red)] = "red"
    return lab


def build_grids(model, D_A, n_r, n_phi):
    r_lo, r_hi = model.r_ang_range(D_A)
    r_grid = jnp.exp(jnp.linspace(jnp.log(r_lo), jnp.log(r_hi), n_r))
    phi = jnp.linspace(-jnp.pi, jnp.pi, n_phi)
    return r_grid, phi


def _eval_fn(model, phys_args, phys_kw, r_grid, phi):
    """jit-compiled log p(d_i | r, phi) on the (r_grid, phi) grid for
    spot i."""
    sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)
    n_r = int(r_grid.shape[0])

    @jax.jit
    def one(i):
        idx = jnp.full((n_r,), i, dtype=jnp.int32)
        return model._phi_integrand(r_grid, sin_phi, cos_phi, idx,
                                    *phys_args, **phys_kw)

    return one


def maps_all_spots(model, phys_args, phys_kw, r_grid, phi):
    """Return log p(d_i | r, phi) as (n_spots, n_r, n_phi)."""
    one = _eval_fn(model, phys_args, phys_kw, r_grid, phi)
    return jnp.stack([one(i) for i in range(model.n_spots)], axis=0)


def maps_subset(model, phys_args, phys_kw, r_grid, phi, spots):
    """Return {i: log-map} for a subset of spots (high-res final panels)."""
    one = _eval_fn(model, phys_args, phys_kw, r_grid, phi)
    return {int(i): one(int(i)) for i in spots}


def to_pdf(logL):
    """Normalise a (n_r, n_phi) log-likelihood map to a 2D PDF (sum=1)."""
    p = np.exp(np.asarray(logL, dtype=np.float64) - np.max(logL))
    s = p.sum()
    return p / s if s > 0 else p


def phi_marginal(pdf):
    return pdf.sum(axis=0)


def find_modes(pm, phi_deg, min_frac=0.03, min_sep_deg=25.0):
    """Circular local maxima of a phi marginal, sorted by height.

    Returns list of (phi_deg, height) for peaks above ``min_frac`` of the
    global max and separated from taller peaks by ``min_sep_deg``.
    """
    n = pm.size
    hi = pm.max()
    cand = []
    for i in range(n):
        l, r = pm[(i - 1) % n], pm[(i + 1) % n]
        if pm[i] >= l and pm[i] >= r and pm[i] >= min_frac * hi:
            cand.append((phi_deg[i], pm[i]))
    cand.sort(key=lambda t: -t[1])
    kept = []
    for ph, h in cand:
        if all(min(abs(ph - q), 360 - abs(ph - q)) >= min_sep_deg
               for q, _ in kept):
            kept.append((ph, h))
    return kept


def bimodality(pm, phi_deg):
    """Score = secondary/primary peak-height ratio; 0 if unimodal."""
    modes = find_modes(pm, phi_deg)
    if len(modes) < 2:
        return 0.0, modes
    return modes[1][1] / modes[0][1], modes


def _window(marg, grid, pad=0.15, lo_q=2e-3, hi_q=1 - 2e-3):
    c = np.cumsum(marg) / marg.sum()
    lo = float(np.interp(lo_q, c, grid))
    hi = float(np.interp(hi_q, c, grid))
    span = max(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span


def panel(ax, pdf, r_grid, phi_deg, label, wrap_positive=False):
    r = np.asarray(r_grid)
    phi_deg = np.asarray(phi_deg)
    if wrap_positive:
        # Map [-180, 180] -> [0, 360] so approaching-spot modes sit around
        # phi = 270 deg (3*pi/2), matching the caption.
        phi_deg = phi_deg % 360.0
        order = np.argsort(phi_deg)
        phi_deg, pdf = phi_deg[order], pdf[:, order]
    phi_lo, phi_clip = (0.0, 360.0) if wrap_positive else (-180.0, 180.0)
    rlo, rhi = _window(pdf.sum(axis=1), r)
    plo, phi_hi = _window(pdf.sum(axis=0), phi_deg, pad=0.30)
    pcm = ax.pcolormesh(phi_deg, r, pdf / pdf.max(), cmap="afmhot_r",
                        norm=PowerNorm(0.45, vmin=0.0, vmax=1.0),
                        shading="gouraud", rasterized=True)
    ax.set_xlim(max(phi_lo, plo), min(phi_clip, phi_hi))
    ax.set_ylim(max(float(r[0]), rlo), min(float(r[-1]), rhi))
    ax.set_xlabel(r"$\phi\ [\mathrm{deg}]$")
    ax.set_title(label, fontsize=9)
    return pcm


def make_final(pdfs, r_grid, phi_deg, spots, labels, out_base, wrap):
    n = len(spots)
    with plt.style.context(["science"]):
        fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.0), squeeze=False,
                                 constrained_layout=True)
        for k, i in enumerate(spots):
            ax = axes[0, k]
            pcm = panel(ax, pdfs[i], r_grid, phi_deg, labels[k],
                        wrap_positive=wrap[k])
            ax.set_ylabel(r"$r\ [\mathrm{mas}]$")
            ax.tick_params(labelsize=8)
        cb = fig.colorbar(pcm, ax=axes.ravel().tolist(), pad=0.01,
                          fraction=0.046)
        cb.set_label("Normalised likelihood", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        for ext in ("png", "pdf"):
            p = f"{out_base}.{ext}"
            fig.savefig(p, dpi=300)
            print("wrote", p)
        plt.close(fig)


def make_contact(pdfs, modes_all, r_grid, phi_deg, ranked, labels, out_png,
                 ncol=3, nrow=4):
    sel = ranked[:ncol * nrow]
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for slot, i in enumerate(sel):
        ax = axes[slot // ncol, slot % ncol]
        ax.axis("on")
        panel(ax, pdfs[i], r_grid, phi_deg, labels[i])
        ax.set_ylabel(r"$r_\mathrm{ang}$ [mas]", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("wrote", out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", type=int, nargs="+", default=None,
                    help="Global spot indices for the final 3-panel figure.")
    ap.add_argument("--n-r", type=int, default=420)
    ap.add_argument("--n-phi", type=int, default=721)
    ap.add_argument("--n-r-final", type=int, default=1400)
    ap.add_argument("--n-phi-final", type=int, default=2800)
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(CANDEL_ROOT), "plots", "megamaser"))
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)
    apply_dataset(master_cfg, DATASET)
    sample = median_globals(HDF5)
    print("median globals:", {k: round(v, 3) for k, v in sample.items()})

    model = build_model(GALAXY, master_cfg)
    phys_args, phys_kw, diag = model.phys_from_sample(sample)
    phys_kw = {k: v for k, v in phys_kw.items() if k != "dv_sys"}
    print(f"D_A = {diag['D_A']:.2f} Mpc, M_BH = {diag['M_BH']:.3e} (1e7 Msun),"
          f" n_spots = {model.n_spots}")

    r_grid, phi = build_grids(model, diag["D_A"], args.n_r, args.n_phi)
    phi_deg = np.degrees(np.asarray(phi))
    logmaps = maps_all_spots(model, phys_args, phys_kw, r_grid, phi)

    labels_cls = spot_class(model)
    pdfs, modes_all, scores = {}, {}, np.zeros(model.n_spots)
    for i in range(model.n_spots):
        pdfs[i] = to_pdf(logmaps[i])
        sc, modes = bimodality(phi_marginal(pdfs[i]), phi_deg)
        modes_all[i] = modes
        scores[i] = sc
    panel_labels = {i: f"#{i} ({labels_cls[i]})" for i in range(model.n_spots)}

    ranked = list(np.argsort(-scores))
    print("\nTop bimodal spots (idx, class, ratio, mode_phi_deg):")
    for i in ranked[:12]:
        ph = ", ".join(f"{m[0]:+.0f}" for m in modes_all[i][:2])
        print(f"  #{i:3d}  {labels_cls[i]:8s}  ratio={scores[i]:.2f}  "
              f"phi=[{ph}]")

    if args.spots:
        rg_f, phi_f = build_grids(model, diag["D_A"], args.n_r_final,
                                  args.n_phi_final)
        phi_f_deg = np.degrees(np.asarray(phi_f))
        sub = maps_subset(model, phys_args, phys_kw, rg_f, phi_f, args.spots)
        pdfs_f = {i: to_pdf(sub[i]) for i in args.spots}
        # Paper figure (NGC5765b): --spots 70 98 191
        #   (a) #70  receding, connected curved ridge
        #   (b) #98  receding, two cleanly separated modes
        #   (c) #191 approaching, connected curved ridge
        word = {"red": "Receding", "blue": "Approaching",
                "systemic": "Systemic"}
        labs = [rf"$\mathrm{{({chr(97 + k)})\ {word[labels_cls[i]]}\ spot}}$"
                for k, i in enumerate(args.spots)]
        wrap = [labels_cls[i] == "blue" for i in args.spots]
        make_final(pdfs_f, rg_f, phi_f_deg, args.spots, labs,
                   os.path.join(args.out_dir, "rphi_bimodality_NGC5765b"),
                   wrap)
    else:
        make_contact(pdfs, modes_all, r_grid, phi_deg, ranked, panel_labels,
                     os.path.join(args.out_dir,
                                  "rphi_bimodality_NGC5765b_contact.png"))


if __name__ == "__main__":
    main()
