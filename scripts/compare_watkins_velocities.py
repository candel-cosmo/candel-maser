#!/usr/bin/env python
"""Compare the Manticore velocities of Watkins & Feldman (2026) with our own.

Watkins & Feldman (arXiv:2608.06247v4, hereafter WF26) tabulate Manticore
line-of-sight peculiar velocities for the six megamaser hosts and report
`607 +/- 53 km/s` for NGC 5765b, where our own Manticore query gives roughly
a third of that.  That galaxy carries 81 per cent of their `-2.80 km/s/Mpc`
Carrick-to-Manticore shift, so the discrepancy is the whole disagreement.

Two candidate mechanisms are separable with the products on disk:

1. **Which field.**  The same Manticore generation is distributed as
   `forward_fields/PCS`, the BORG forward-model grid with a mass-weighted
   velocity, and as `SWIFT_velocity_fields`, density and momentum deposited
   from the SWIFT particle realisation.  The SPH momentum ratio `p / rho`
   resolves collapsed structure that the forward grid smooths over, so it can
   differ by hundreds of km/s in a cell containing a halo.
2. **Where along the line of sight.**  WF26 evaluate at the P20 point distance;
   we marginalise over the stage-1 distance posterior.  For a host whose
   distance is uncertain at the 4.5 per cent level in a field varying on
   ~3 Mpc/h scales, the two need not agree.

The script evaluates both fields on the same line-of-sight grid over the same
80 realisations and reports the ensemble mean and scatter at the P20 distance,
the profile along the line of sight, and the sensitivity to distance.

Requires `io.reconstruction_main.ManticoreLocalSWIFT` in `local_config.toml`,
pointing at the directory of `mcmc_*.hdf5` momentum products, with `ngrid` and
`boxsize` set for the downloaded resolution.

Run from the repository root with::

    venv_candel/bin/python packages/candel-maser/scripts/compare_watkins_velocities.py
"""
import os
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")


import numpy as np  # noqa: E402
from candel_maser.joint_H0_helpers import DEFAULT_FIELD_CONFIG  # noqa: E402
from candel_maser.joint_H0_helpers import (  # noqa: E402
    interpolate_los_velocity)

from candel.field import name2field_loader  # noqa: E402
from candel.field.loader import available_mcmc_field_indices  # noqa: E402
from candel.util import load_config  # noqa: E402

# WF26 table 1: sky position, P20 distance, and their tabulated Manticore
# velocity.  v1-v3 used a single realisation; v4 the 80-realisation mean.
GALAXIES = (
    # name, RA, dec, D_P20 [Mpc], sigma_D, WF26 v4 mean, sd, WF26 v1-v3
    ("NGC4258", 184.7401, 47.3037, 7.58, 0.11, 285.0, 94.0, 162.0),
    ("UGC3789", 109.8787, 59.3551, 51.5, 4.2, 45.0, 48.0, 54.0),
    ("CGCG074-064", 210.7685, 8.9476, 87.6, 7.5, 334.0, 61.0, 149.0),
    ("NGC6323", 258.3252, 43.7824, 109.4, 30.0, 694.0, 88.0, 671.0),
    ("NGC5765b", 222.7146, 5.1145, 112.2, 5.0, 607.0, 53.0, 546.0),
    ("NGC6264", 254.3172, 27.8496, 132.1, 20.0, 439.0, 75.0, 434.0),
)
FIELDS = ("ManticoreLocalCOLA", "ManticoreLocalSWIFT")
# Comoving radius grid in Mpc/h.  Manticore is an h = 0.681 box, so the P20
# distances in Mpc are converted before the lookup.
LOS_R = np.arange(0.5, 250.0 + 0.25, 0.5, dtype=np.float32)
LITTLE_H = 0.681


def _field_kwargs(reconstruction):
    """Loader kwargs, with the COLA mass assignment set to what is on disk.

    The runs config pins `which_MAS = "CIC"`, but the downloaded generation
    ships the PCS forward fields, so the directory is resolved here rather than
    taken from the config.
    """
    config = load_config(DEFAULT_FIELD_CONFIG)
    kwargs = dict(config["io"]["reconstruction_main"][reconstruction])
    root = kwargs["fpath_root"]
    if reconstruction == "ManticoreLocalCOLA":
        available = [d for d in ("PCS", "CIC", "NGP", "TSC")
                     if os.path.isdir(os.path.join(root, d))]
        if not available:
            raise FileNotFoundError(f"No mass-assignment directory in {root}")
        kwargs["which_MAS"] = available[0]
        root = os.path.join(root, available[0])
    return kwargs, root


def _load(reconstruction, n_max=None):
    """Line-of-sight velocity, shape (n_realisation, n_galaxy, n_r)."""
    kwargs, index_root = _field_kwargs(reconstruction)
    indices = available_mcmc_field_indices(index_root)
    if n_max is not None:
        indices = indices[:int(n_max)]
    # Disposable scratch cache: rebuilding it means re-reading ~30 GB.
    cache = os.path.join(tempfile.gettempdir(),
                         f"vlos_watkins_{reconstruction}_"
                         f"{len(indices)}.npz")
    if os.path.exists(cache):
        print(f"{reconstruction}: reusing {cache}", flush=True)
        with np.load(cache) as f:
            return f["vlos"]
    loader_cls = name2field_loader(reconstruction)
    RA = np.array([g[1] for g in GALAXIES])
    dec = np.array([g[2] for g in GALAXIES])
    print(f"{reconstruction}: {len(indices)} realisations from "
          f"{kwargs['fpath_root']}", flush=True)
    out = []
    for nsim in indices:
        loader = loader_cls(nsim=int(nsim), **kwargs)
        vlos, _ = interpolate_los_velocity(loader, LOS_R, RA, dec,
                                           verbose=False)
        out.append(np.asarray(vlos, dtype=np.float64))
    vlos = np.stack(out)
    np.savez(cache, vlos=vlos)
    return vlos


def _at_distance(vlos, distance_mpc):
    """Ensemble of v_los at one distance, shape (n_realisation,)."""
    return np.array([np.interp(distance_mpc * LITTLE_H, LOS_R, v)
                     for v in vlos])


def main():
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else None
    fields = {name: _load(name, n_max) for name in FIELDS}
    n_real = {k: v.shape[0] for k, v in fields.items()}
    print(f"\nrealisations loaded: {n_real}")

    print("\n1. Line-of-sight velocity at the P20 distance [km/s], "
          "mean +/- sd over realisations")
    header = f"  {'galaxy':14s} {'D [Mpc]':>8s}"
    header += "".join(f"{k.replace('ManticoreLocal', ''):>18s}"
                      for k in FIELDS)
    header += f"{'WF26 v4':>18s}{'WF26 v1-v3':>12s}"
    print(header)
    for i, (name, _, _, dist, _, wf_mean, wf_sd, wf_v3) in enumerate(GALAXIES):
        row = f"  {name:14s} {dist:8.2f}"
        for key in FIELDS:
            ens = _at_distance(fields[key][:, i, :], dist)
            row += f"{ens.mean():11.0f} +/-{ens.std():4.0f}"
        row += f"{wf_mean:11.0f} +/-{wf_sd:4.0f}{wf_v3:12.0f}"
        print(row)

    print("\n2. NGC 5765b along the line of sight [km/s], "
          "mean +/- sd over realisations")
    idx = [g[0] for g in GALAXIES].index("NGC5765b")
    dist, sigma = GALAXIES[idx][3], GALAXIES[idx][4]
    print(f"  {'D [Mpc]':>9s}" + "".join(
        f"{k.replace('ManticoreLocal', ''):>18s}" for k in FIELDS))
    for d in (dist - 3 * sigma, dist - 2 * sigma, dist - sigma, dist,
              dist + sigma, dist + 2 * sigma, dist + 3 * sigma):
        row = f"  {d:9.1f}"
        for key in FIELDS:
            ens = _at_distance(fields[key][:, idx, :], d)
            row += f"{ens.mean():11.0f} +/-{ens.std():4.0f}"
        print(row)

    print("\n3. Distance marginalisation, Gaussian over the P20 distance")
    print("   (WF26 evaluate at the point distance; the shift is the "
          "difference)")
    rng = np.random.default_rng(42)
    print(f"  {'galaxy':14s}" + "".join(
        f"{k.replace('ManticoreLocal', '') + ' point':>20s}"
        f"{k.replace('ManticoreLocal', '') + ' marg':>20s}"
        for k in FIELDS))
    for i, (name, _, _, dist, sigma_d, *_) in enumerate(GALAXIES):
        draws = rng.normal(dist, sigma_d, 2000)
        row = f"  {name:14s}"
        for key in FIELDS:
            point = _at_distance(fields[key][:, i, :], dist)
            marg = np.concatenate(
                [np.interp(draws * LITTLE_H, LOS_R, v)
                 for v in fields[key][:, i, :]])
            row += f"{point.mean():13.0f} +/-{point.std():4.0f}"
            row += f"{marg.mean():13.0f} +/-{marg.std():4.0f}"
        print(row)

    print("\n4. Field-to-field difference at the P20 distance [km/s]")
    print(f"  {'galaxy':14s} {'SWIFT - COLA':>14s} {'WF26 - COLA':>14s} "
          f"{'WF26 - SWIFT':>14s}")
    for i, (name, _, _, dist, _, wf_mean, *_) in enumerate(GALAXIES):
        cola = _at_distance(fields["ManticoreLocalCOLA"][:, i, :],
                            dist).mean()
        swift = _at_distance(fields["ManticoreLocalSWIFT"][:, i, :],
                             dist).mean()
        print(f"  {name:14s} {swift - cola:14.0f} {wf_mean - cola:14.0f} "
              f"{wf_mean - swift:14.0f}")

    print("\n5. NGC 5765b marginalised over a Gaussian distance posterior "
          "of width sigma_D")
    print("   (P20 quote 5 Mpc; our stage-1 posterior is about twice that, "
          "so the\n    same field gives a much lower velocity once the "
          "distance is sampled)")
    D_5765 = GALAXIES[idx][3]
    print(f"  {'sigma_D [Mpc]':>13s}" + "".join(
        f"{k.replace('ManticoreLocal', ''):>20s}" for k in FIELDS))
    for sigma_d in (0.0, 2.0, 5.0, 8.0, 11.0, 15.0):
        row = f"  {sigma_d:13.1f}"
        for key in FIELDS:
            if sigma_d == 0.0:
                ens = _at_distance(fields[key][:, idx, :], D_5765)
            else:
                draws = rng.normal(D_5765, sigma_d, 4000)
                ens = np.concatenate(
                    [np.interp(draws * LITTLE_H, LOS_R, v)
                     for v in fields[key][:, idx, :]])
            median = np.median(ens)
            lo, hi = np.percentile(ens, [16, 84])
            row += f"{median:9.0f} +{hi - median:4.0f}/-{median - lo:4.0f}"
        print(row)


if __name__ == "__main__":
    main()
