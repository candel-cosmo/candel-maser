"""Extract the informative Vext prior for the joint megamaser H0 inference.

Offline tool: pools a reconstruction's own Vext posterior (samples/Vext,
N x 3, ICRS-equatorial Cartesian km/s), fits a Gaussian (mean + 3x3
covariance), runs a frame check, and prints a config block to paste under
[joint.priors.Vext_informative.<name>] in config_maser.toml.  The joint
model reads those static numbers from the config and never runs this
script; the frame already matches the model's, so no rotation is applied.

    python packages/candel-maser/scripts/extract_vext_prior.py --name ManticoreLocalCOLA \\
        --files 'results/VFO/single_fields/...field*_single.hdf5'
"""
import argparse
import glob

import numpy as np

from candel.util import radec_cartesian_to_galactic


def load_vext_samples(paths):
    """Pool samples/Vext (N, 3) across HDF5 files (ICRS-Cartesian, km/s)."""
    import h5py
    out = []
    for p in paths:
        with h5py.File(p, "r") as f:
            out.append(np.asarray(f["samples/Vext"], dtype=np.float64))
    return np.concatenate(out, axis=0)


def fit_vext_prior(V):
    """(mean (3,), cov (3, 3)) Gaussian fit to pooled Cartesian samples."""
    return V.mean(axis=0), np.cov(V, rowvar=False)


def _frame_check(path):
    """Max |delta| (deg) between Cartesian-derived and stored Galactic
    ell/b."""
    import h5py
    with h5py.File(path, "r") as f:
        V = np.asarray(f["samples/Vext"], float)
        ell = np.asarray(f["samples/Vext_ell"], float) % 360.0
        b = np.asarray(f["samples/Vext_b"], float)
    _, gell, gb = radec_cartesian_to_galactic(V[:, 0], V[:, 1], V[:, 2])
    dell = np.abs(((gell % 360.0) - ell + 180.0) % 360.0 - 180.0)
    return float(dell.max()), float(np.abs(gb - b).max())


def _toml_block(name, mean, cov):
    mrow = ", ".join(f"{x:.4f}" for x in mean)
    crows = ",\n       ".join(
        "[" + ", ".join(f"{x:.4f}" for x in row) + "]" for row in cov)
    return (f"[joint.priors.Vext_informative.{name}]\n"
            f"mean = [{mrow}]\n"
            f"cov = [{crows}]\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True,
                    help="HDF5 file(s)/glob(s) with samples/Vext to fit")
    ap.add_argument("--name", default="RECON",
                    help="reconstruction name for the emitted config block")
    args = ap.parse_args()

    paths = sorted(p for g in args.files for p in glob.glob(g))
    if not paths:
        ap.error("no files matched --files")
    V = load_vext_samples(paths)
    mean, cov = fit_vext_prior(V)
    mag = float(np.linalg.norm(mean))
    mag_sigma = float(np.sqrt(mean @ cov @ mean) / mag)
    _, ell, b = radec_cartesian_to_galactic(*mean)
    dmax_ell, dmax_b = _frame_check(paths[0])
    print(f"# pooled {len(paths)} file(s), {V.shape[0]} samples")
    print(f"# |Vext| = {mag:.1f} +/- {mag_sigma:.1f} km/s, "
          f"(l, b) = ({float(ell) % 360:.1f}, {float(b):.1f}) deg")
    print(f"# frame check: max|d l|={dmax_ell:.1e}, max|d b|={dmax_b:.1e} "
          f"deg (Cartesian == stored Galactic)")
    print(_toml_block(args.name, mean, cov))


if __name__ == "__main__":
    main()
