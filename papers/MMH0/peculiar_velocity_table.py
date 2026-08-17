#!/usr/bin/env python
"""Per-galaxy line-of-sight peculiar velocities from the joint-H0 chains.

For each reconstruction the peculiar velocity of galaxy `g` is
``V_pec = beta * v_rad(D_g) + Vext . nhat_g``, so it depends on the sampled
distance.  We therefore evaluate it at every posterior sample of the joint-H0
chain, which marginalises the line-of-sight distance uncertainty (and, for
Manticore, `beta` and `Vext`), and pool over the constrained field
realisations.  Reported are the posterior median and central 68% interval of
the reconstruction term alone (Vext excluded) and of the full peculiar
velocity (Vext included).
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import tomli

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
CONFIG_PATH = os.path.join(_HERE, "config_maser.toml")

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from jax import config as _jax_config  # noqa: E402
_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from h5py import File as H5File  # noqa: E402
from joint_H0_helpers import (DEFAULT_FIELD_CONFIG,  # noqa: E402
                              _interp_los_velocity,
                              _load_or_build_vlos_cache,
                              rotate_vext_to_frame)
from maser_config import add_dataset_arg, apply_dataset  # noqa: E402

from candel.util import fprint, fsection, results_path  # noqa: E402

# Cache order must match `run_joint_H0.MCP_GALAXIES`, else the LOS velocity
# cache digest differs and the (80-realisation) fields are re-interpolated.
GALAXIES = ("CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789")
ROW_ORDER = ("CGCG074-064", "NGC5765b", "UGC3789", "NGC6264", "NGC6323")
GAL_LABELS = {"CGCG074-064": r"CGCG~074-064", "NGC5765b": r"NGC~5765b",
              "UGC3789": r"UGC~3789", "NGC6264": r"NGC~6264",
              "NGC6323": r"NGC~6323"}
RECONSTRUCTIONS = ("Carrick2015", "ManticoreLocalCOLA")
RECON_LABELS = {"Carrick2015": r"\citetalias{Carrick2015}",
                "ManticoreLocalCOLA": r"\Manticore"}


def _field_config_with_MAS(field_config, which_MAS, workdir):
    """Overlay config that overrides Manticore's `which_MAS`.

    `config_paths.toml` pins it to CIC and a tracked config outranks
    `local_config.toml`, so a machine holding only another field product needs
    an overlay rather than a local-config entry.
    """
    if which_MAS is None:
        return field_config
    field_config = os.path.abspath(field_config)
    with open(field_config, "rb") as f:
        parent = tomli.load(f).get("base", [])
    # `load_config` reads each base file raw, so the parent's own bases must be
    # spliced in ahead of it rather than resolved recursively.
    base = [os.path.join(os.path.dirname(field_config), b)
            for b in ([parent] if isinstance(parent, str) else parent)]
    base.append(field_config)
    path = os.path.join(workdir, "field_config_MAS.toml")
    with open(path, "w") as f:
        f.write(f"base = {base!r}\n".replace("'", '"')
                + "[io.reconstruction_main.ManticoreLocalCOLA]\n"
                + f'which_MAS = "{which_MAS}"\n')
    return path


def _rotation_to_frame(frame):
    """3x3 rotation taking an ICRS-Cartesian vector into `frame`."""
    R = np.stack([np.asarray(rotate_vext_to_frame(jnp.asarray(e), frame))
                  for e in np.eye(3)], axis=1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10), "R is not orthogonal"
    return R


def _load_chain(path):
    with H5File(path, "r") as f:
        samples = {k: np.asarray(v, dtype=float) for k, v in
                   f["samples"].items()}
        attrs = dict(f.attrs)
    return samples, attrs


def _interval(x):
    q16, med, q84 = np.percentile(np.asarray(x).reshape(-1), [16, 50, 84])
    return med, q84 - med, med - q16


def _fmt(x, latex):
    med, plus, minus = _interval(x)
    if latex:
        return f"${med:.0f}^{{+{plus:.0f}}}_{{-{minus:.0f}}}$"
    return f"{med:7.1f} (+{plus:.1f}/-{minus:.1f})"


def peculiar_velocities(cfg, recon, chain_path, field_config, r_grid):
    """Posterior samples of V_pec per galaxy, with and without Vext.

    Returns ``{galaxy: (V_recon, V_ext_rad, V_total)}`` with `V_recon` and
    `V_total` of shape (n_samples, n_fields) and `V_ext_rad` (n_samples,).
    """
    samples, attrs = _load_chain(chain_path)
    if str(attrs.get("reconstruction")) != recon:
        raise ValueError(f"{chain_path} was run with reconstruction "
                         f"{attrs.get('reconstruction')!r}, not {recon!r}.")
    h = samples["H0"].reshape(-1) / 100.0
    beta = (samples["beta"].reshape(-1) if "beta" in samples
            else np.full(h.shape, float(attrs["velocity_beta"])))
    if "Vext" not in samples:
        raise ValueError(f"{chain_path} has no sampled Vext.")
    Vext = samples["Vext"].reshape(-1, 3)

    stub = [{"name": g, "RA": float(cfg["model"]["galaxies"][g]["ra"]),
             "dec": float(cfg["model"]["galaxies"][g]["dec"])}
            for g in GALAXIES]
    vd = _load_or_build_vlos_cache(recon, field_config, None, r_grid, stub)
    los_r = jnp.asarray(vd["r"])
    Vext_frame = Vext @ _rotation_to_frame(vd["coordinate_frame"]).T

    interp = jax.jit(jax.vmap(_interp_los_velocity, in_axes=(0, None, None)))
    out = {}
    for i, galaxy in enumerate(GALAXIES):
        D_c = samples[f"{galaxy}__D_c"].reshape(-1)
        vlos = np.asarray(interp(jnp.asarray(D_c * h), los_r,
                                 jnp.asarray(vd["los_velocity"][:, i, :])))
        V_recon = beta[:, None] * vlos
        V_ext_rad = Vext_frame @ np.asarray(vd["rhat"][i], dtype=float)
        out[galaxy] = (V_recon, V_ext_rad, V_recon + V_ext_rad[:, None])
    return out, vd["los_velocity"].shape[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arg(parser)
    parser.add_argument("--selection", choices=("none", "distance",
                                                "redshift"),
                        default="redshift")
    parser.add_argument("--field-config", default=DEFAULT_FIELD_CONFIG)
    parser.add_argument("--which-MAS", default=None,
                        choices=("CIC", "PCS", "SPH", "CIC_BORG"),
                        help="Override the Manticore field product. Default: "
                             "the configured one (CIC), which is what the "
                             "joint-H0 chains use.")
    parser.add_argument("--vlos-rmin", type=float, default=0.1)
    parser.add_argument("--vlos-rmax", type=float, default=250.0)
    parser.add_argument("--vlos-dr", type=float, default=0.5)
    parser.add_argument("--output", default=None,
                        help="Write the LaTeX table body here.")
    args = parser.parse_args(argv)

    with open(CONFIG_PATH, "rb") as f:
        cfg = tomli.load(f)
    apply_dataset(cfg, args.dataset)
    root = cfg.get("io", {}).get("root_output", "results/Megamaser")
    r_grid = np.arange(args.vlos_rmin,
                       args.vlos_rmax + 0.5 * args.vlos_dr,
                       args.vlos_dr, dtype=np.float32)

    with tempfile.TemporaryDirectory() as workdir:
        field_config = _field_config_with_MAS(
            args.field_config, args.which_MAS, workdir)
        results = {}
        for recon in RECONSTRUCTIONS:
            path = results_path(
                root, "H0",
                f"joint_H0_toy_all_{args.selection}_{recon}_r2.hdf5")
            fsection(f"{recon} ({os.path.basename(path)})")
            results[recon], n_fields = peculiar_velocities(
                cfg, recon, path, field_config, r_grid)
            fprint(f"field realisations: {n_fields}")
            for galaxy in ROW_ORDER:
                V_recon, V_ext_rad, V_total = results[recon][galaxy]
                fprint(f"{galaxy:<12s} excl. Vext {_fmt(V_recon, False)}  "
                       f"Vext.n {_fmt(V_ext_rad, False)}  "
                       f"incl. Vext {_fmt(V_total, False)} km/s")

    lines = []
    for galaxy in ROW_ORDER:
        cells = []
        for recon in RECONSTRUCTIONS:
            V_recon, V_ext_rad, V_total = results[recon][galaxy]
            cells += [_fmt(V_recon, True), _fmt(V_ext_rad, True),
                      _fmt(V_total, True)]
        lines.append(f"{GAL_LABELS[galaxy]} & " + " & ".join(cells) + r" \\")
    table = "\n".join(lines)
    fsection("LaTeX table body")
    print(table)
    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        fprint(f"wrote {args.output}")


if __name__ == "__main__":
    main()
