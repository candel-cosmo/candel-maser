#!/usr/bin/env python3
"""Validation outputs for the MMH0 appendix, from the chains in
results/Megamaser.

  agreement : a single-panel scatter of the per-spot data-fit term -0.5*chi^2
              from CANDEL vs Reid's fit_disk (f2py reidlik), pooled over
              posterior-seeded draws and all five galaxies, on a 1:1 line.
              Shows the two codes implement the same likelihood.
              -> figs/reid_agreement.pdf

  evidence  : prints (no figure) the per-galaxy latent-marginalised data
              log-evidence logZ_2d at our posterior-median disc parameters
              minus that at the published Pesce/Reid fit, for both warp
              models.  These are NOT recomputed: the legacy comparison runs
              already score and print them, and we read the stored
              values from the run logs (via analyse_sweep).  They are
              tabulated in the paper (tab:evidence) alongside the distance
              estimates.

  ev_vs_dist: scatter of the latent-marginalised likelihood gain
              (Delta ln L_2D, ours - P20) against the distance shift
              (Delta D_A), one point per galaxy, linear warp.  Every galaxy
              lies above zero: the data favour our disc whatever the distance
              shift.  -> figs/evidence_vs_distance.pdf

Run from the repo root with venv_candel:
    python papers/MMH0/plot_validation_figs.py both
"""
import argparse
import os
import sys

import numpy as np

# f64 + path setup happen in check_reid_agreement at import.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_reid_agreement as cra  # noqa: E402
import candel_maser.run_maser as rm  # noqa: E402
from palette import PALETTE  # noqa: E402

GALAXIES = cra.GALAXIES
GAL_LABEL = {
    "CGCG074-064": "CGCG 074-064", "NGC5765b": "NGC 5765b",
    "UGC3789": "UGC 3789", "NGC6264": "NGC 6264", "NGC6323": "NGC 6323",
}
OUTDIR = "/Users/rstiskalek/Papers/MMH0/figs"


def _setup(galaxy):
    """(model, target, samples, ctx) for one galaxy's fiducial chain."""
    model, _ = cra.build_model_target(galaxy)
    samples = cra.load_global_samples(galaxy)
    init = rm._complete_mass_point(
        model, {k: np.asarray(v[0]) for k, v in samples.items()})
    target = rm.MaserBlackJaxTarget(model, rm._h_ref(model), init)
    ctx = rm._reid_loglik_context(galaxy, model.n_spots, cra.DATASET)
    return model, target, samples, ctx


# --------------------------------------------------------------------------
# Figure: per-spot CANDEL vs Reid -0.5*chi^2
# --------------------------------------------------------------------------
def map_latents(model, target, point, r_ang, n_phi=181):
    """Per-spot (r_hat, phi*) where phi* maximises the CANDEL per-spot fit.

    The class-centre phi seed leaves spots from posterior-tail draws on the
    steep chi^2 wall; argmax over phi at the conditional r_hat puts every spot
    near its best fit, the regime the posterior actually occupies, so the two
    codes are compared where it matters rather than in the tails.
    """
    grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    vals = np.empty((n_phi, model.n_spots))
    for k, ph in enumerate(grid):
        phi_k = np.full(model.n_spots, ph)
        vals[k] = np.asarray(
            rm._candel_neg_half_chi2(model, target, point, r_ang, phi_k))
    return grid[np.nanargmax(vals, axis=0)]


def collect_agreement(n_draws, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for g in GALAXIES:
        model, target, samples, ctx = _setup(g)
        if ctx is None:
            continue
        n = len(next(iter(samples.values())))
        c_all, r_all = [], []
        for s in rng.choice(n, size=min(n_draws, n), replace=False):
            point = rm._complete_mass_point(
                model, {k: np.asarray(v[s]) for k, v in samples.items()})
            r_ang, _ = cra.seed_latents(model, target, point)
            phi = map_latents(model, target, point, r_ang)
            D_A = rm._point_D_A(model, target, point)
            c = np.asarray(rm._candel_neg_half_chi2(
                model, target, point, r_ang, phi))
            r = np.asarray(rm._reid_neg_half_chi2(
                ctx, g, point, r_ang, phi, D_A=D_A))
            ok = np.isfinite(c) & np.isfinite(r)
            c_all.append(c[ok])
            r_all.append(r[ok])
        out[g] = (np.concatenate(c_all), np.concatenate(r_all))
    return out


def plot_agreement(data, out):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    allc = np.concatenate([c for c, _ in data.values()])
    allr = np.concatenate([r for _, r in data.values()])
    resid = allc - allr
    aresid = np.abs(resid)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    med = float(np.median(aresid))
    mx = float(np.max(aresid))
    corr = float(np.corrcoef(allc, allr)[0, 1])

    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(3.5, 3.4))
        colors = PALETTE
        lo = min(allr.min(), allc.min())
        hi = max(allr.max(), allc.max())
        ax.plot([lo, hi], [lo, hi], ls=":", color="0.5", lw=0.9, zorder=0)
        for (g, (c, r)), col in zip(data.items(), colors):
            ax.scatter(r, c, s=4, alpha=0.35, edgecolors="none",
                       color=col, label=GAL_LABEL[g], rasterized=True)
        ax.set_xlabel(r"Reid per-spot $-\tfrac{1}{2}\chi^2$")
        ax.set_ylabel(r"CANDEL per-spot $-\tfrac{1}{2}\chi^2$")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", fontsize=5, markerscale=2,
                  handletextpad=0.1, borderpad=0.3)
        ax.text(0.04, 0.96,
                rf"$r={corr:.6f}$" + "\n"
                rf"median$\,|\Delta|={med:.1e}$" + "\n"
                rf"max$\,|\Delta|={mx:.1e}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=6)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"wrote {out}  (r={corr:.4f}, median|d|={med:.3f}, RMS={rms:.3f}, "
          f"n={resid.size})")


# --------------------------------------------------------------------------
# Figure: conditioned-evidence posterior vs the Pesce/Reid point
# --------------------------------------------------------------------------
def _stored_logZ(gdir, galaxy, qw):
    """(logZ_2d at our point, at the Pesce point) from the run logs.

    The legacy comparison runs already score and print these; we read them
    rather than recompute.  Prefer the config-initialised chain, fall back to
    reid-init (same posterior) where the config row was not logged.
    """
    import analyse_sweep as asw
    suffix = "_qw" if qw else ""
    for init in ("config", "reid"):
        fp = gdir / f"{galaxy}_blackjax_mcmc_rphi{suffix}_init{init}.hdf5"
        rows, _ = asw.parse_log(asw.find_log(gdir, str(fp)))
        mine = rows.get("MCMC median", {}).get("logZ_2d")
        pesce = rows.get("Pesce/Reid", {}).get("logZ_2d")
        if mine is not None and pesce is not None:
            return mine, pesce, init
    return None, None, None


def collect_evidence_from_logs():
    """Per-galaxy stored Delta logZ_2d = logZ_2d(ours) - logZ_2d(P20)."""
    import analyse_sweep as asw
    out_root = asw.root_output()
    fid, qw = {}, {}
    for g in GALAXIES:
        gdir = out_root / g
        m, p, init = _stored_logZ(gdir, g, qw=False)
        if m is not None:
            fid[g] = (m - p, init)
        m2, p2, _ = _stored_logZ(gdir, g, qw=True)
        if m2 is not None:
            qw[g] = m2 - p2
    return fid, qw


def print_evidence(fid, qw):
    """Print the stored Delta logZ_2d per galaxy for both warp models.

    These are the values tabulated in the paper (tab:evidence); the figure was
    replaced by a table that also carries the distance estimates.
    """
    print(f"{'Galaxy':14} {'linear':>10} {'quadratic':>10}  (Delta logZ_2D)")
    for g in (g for g in GALAXIES if g in fid):
        d, init = fid[g]
        q = qw.get(g)
        print(f"{GAL_LABEL[g]:14} {d:>+10.1f} "
              f"{('%+.1f' % q) if q is not None else '--':>10}"
              f"  ({init}-init)")


# --------------------------------------------------------------------------
# Figure: likelihood gain vs distance shift, ours - P20
# --------------------------------------------------------------------------
# Published Pesce+2020 angular-diameter-distance uncertainties: the symmetrised
# reported 1sigma in Mpc, from their table 1 and, for the four galaxies they
# remodelled, their table 2 (external literature, not a pipeline output).
# CGCG 074-064 is 87.6 +7.9 -7.2 (P20 table 1, from Pesce+2020a).
PESCE_DA_SIG = {"CGCG074-064": 0.5 * (7.9 + 7.2),
                "NGC5765b": 0.5 * (5.4 + 5.1),
                "UGC3789": 0.5 * (4.5 + 4.0),
                "NGC6264": 0.5 * (21.2 + 17.3),
                "NGC6323": 0.5 * (34.2 + 23.4)}


def collect_evidence_vs_distance():
    """Per-galaxy (Delta D_A, Delta logP_2d, sigma_dDA) = ours - P20, linear.

    Delta D_A and Delta logP_2d come from the same legacy comparison log row so
    they share the scored point; logP_2d carries the global-parameter prior of
    the paper's L_2D (eq:logP2d), i.e. the lnprob the chains report.  sigma_dDA
    propagates the uncertainty on the distance difference: our 68% half-width
    from the chain combined in quadrature with the symmetrised P20 uncertainty.
    """
    import analyse_sweep as asw
    import h5py
    out_root = asw.root_output()
    out = {}
    for g in GALAXIES:
        gdir = out_root / g
        for init in ("config", "reid"):
            fp = gdir / f"{g}_blackjax_mcmc_rphi_init{init}.hdf5"
            try:
                rows, _ = asw.parse_log(asw.find_log(gdir, str(fp)))
            except Exception:
                continue
            m = rows.get("MCMC median", {})
            p = rows.get("Pesce/Reid", {})
            if "logP_2d" not in m or "logP_2d" not in p:
                continue
            with h5py.File(fp, "r") as f:
                s = f["samples"]
                da = np.asarray(s["D_A"] if "D_A" in s else s["D_c"],
                                float).ravel()
            q16, q84 = np.percentile(da, [16, 84])
            sig = float(np.hypot(0.5 * (q84 - q16), PESCE_DA_SIG[g]))
            out[g] = (m["D_A"] - p["D_A"], m["logP_2d"] - p["logP_2d"], sig)
            break
    return out


def plot_evidence_vs_distance(data, out):
    """Latent-marginalised likelihood gain against the distance shift, ours
    minus P20, one point per galaxy.  Every galaxy sits above zero: the data
    favour our disc parameters whatever the distance shift."""
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    preamble = (r"\usepackage{amsmath}\usepackage{amssymb}"
                r"\usepackage{lmodern}")
    with plt.style.context(["science", {"text.latex.preamble": preamble}]):
        fig, ax = plt.subplots(figsize=(3.4, 3.3))
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=0)
        ax.axvline(0.0, color="0.7", lw=0.8, ls=":", zorder=0)
        for (g, (dDA, dlnL, sig)), c in zip(data.items(), PALETTE):
            ax.errorbar(dDA, dlnL, xerr=sig, fmt="o", ms=6, color=c,
                        mec="0.2", mew=0.5, ecolor=c, elinewidth=1.0,
                        capsize=2.5, zorder=3, label=GAL_LABEL[g])
        ax.set_xlabel(r"$\Delta D_\mathrm{A}$ (ours $-$ P20) $[\mathrm{Mpc}]$")
        ax.set_ylabel(r"$\Delta \ln \mathcal{L}_\mathrm{2D}$ (ours $-$ P20)")
        ax.set_ylim(bottom=-3.0)
        ax.legend(fontsize="small", frameon=True, loc="upper right",
                  handletextpad=0.2, labelspacing=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# Figure: latent-marginalised evidence gain vs the profile chi^2 difference
# --------------------------------------------------------------------------
def collect_chi2_vs_evidence():
    """Per-galaxy (Delta chi^2, Delta lnP_2D) = ours - P20.

    Read from chi2_evidence_table.json (written by
    papers/MMH0/chi2_evidence_table.py):
    the total chi^2 is computed at the median / P20 globals with the latents
    optimised, and lnP_2D is the value already stored in the legacy comparison
    logs, i.e. the same quantity as evidence_vs_distance.pdf.
    """
    import json

    import analyse_sweep as asw
    with open(asw.root_output() / "chi2_evidence_table.json") as f:
        by_gal = {r["galaxy"]: r for r in json.load(f)}
    return {g: (by_gal[g]["dchi2"], by_gal[g]["dlnP_2D"])
            for g in GALAXIES
            if g in by_gal and by_gal[g].get("dlnP_2D") is not None}


def plot_chi2_vs_evidence(data, out):
    """Latent-marginalised evidence gain against the profile chi^2 difference,
    ours minus P20, one per galaxy.  Every galaxy has Delta lnP_2D > 0, so
    the evidence favours our disc throughout; four also have Delta chi^2 < 0, a
    better point fit, while CGCG 074-064, whose distance nearly coincides with
    P20, is favoured by the evidence despite a marginally larger chi^2."""
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    preamble = (r"\usepackage{amsmath}\usepackage{amssymb}"
                r"\usepackage{lmodern}")
    with plt.style.context(["science", {"text.latex.preamble": preamble}]):
        fig, ax = plt.subplots(figsize=(3.4, 3.3))
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=0)
        ax.axvline(0.0, color="0.7", lw=0.8, ls=":", zorder=0)
        for (g, (dchi2, dlnP)), c in zip(data.items(), PALETTE):
            ax.scatter(-dchi2, dlnP, s=42, color=c, edgecolors="0.2",
                       linewidths=0.5, zorder=3, label=GAL_LABEL[g])
        ax.set_xlabel(r"$-\Delta \chi^2$ (P20 $-$ ours)")
        ax.set_ylabel(r"$\Delta \ln \mathcal{P}_\mathrm{2D}$ (ours $-$ P20)")
        ax.legend(fontsize="small", frameon=True, loc="lower right",
                  handletextpad=0.2, labelspacing=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fig",
                    choices=("agreement", "evidence", "ev_vs_dist",
                             "chi2_ev", "both"),
                    default="both", nargs="?")
    ap.add_argument("--n-draws-scatter", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=OUTDIR)
    ap.add_argument("--no-match-reid", dest="match_reid", action="store_false",
                    help="do NOT swap in Reid's physics constants/gamma")
    args = ap.parse_args(argv)

    if args.match_reid:
        # Reid constants + circular-speed SR gamma, before any model build,
        # so both the per-spot scatter and the logZ_2d scores are on Reid's
        # own likelihood (D_A is matched per draw inside the comparison).
        rm._apply_reid_physics_constants()

    if args.fig in ("agreement", "both"):
        data = collect_agreement(args.n_draws_scatter, args.seed)
        plot_agreement(data, os.path.join(args.out_dir, "reid_agreement.pdf"))

    if args.fig in ("evidence", "both"):
        fid, qw = collect_evidence_from_logs()
        print_evidence(fid, qw)

    if args.fig in ("ev_vs_dist", "both"):
        evd = collect_evidence_vs_distance()
        plot_evidence_vs_distance(
            evd, os.path.join(args.out_dir, "evidence_vs_distance.pdf"))

    if args.fig in ("chi2_ev", "both"):
        cev = collect_chi2_vs_evidence()
        plot_chi2_vs_evidence(
            cev, os.path.join(args.out_dir, "chi2_vs_evidence.pdf"))


if __name__ == "__main__":
    main()
