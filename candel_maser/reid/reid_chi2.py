"""Original Reid fit_disk chi^2 via the f2py-wrapped Fortran (reidlik).

Thin extraction of the Reid-likelihood helpers so diagnostics can compute the
per-spot -0.5*chi^2 with the ORIGINAL Reid code (``calc_warped_model`` +
``add_error_floors`` from ``fit_disk_v24d_unblinded.f``), not a
reimplementation.  Mirrors the corresponding helpers in run_maser.py.

Per-spot values are returned in the Reid loader order, which matches CANDEL's
spot order (both read the same data file), so summing gives a globals-order-
independent total chi^2 directly comparable to CANDEL's.
"""
import os
import sys
import tempfile

import numpy as np
import tomli_w

CLIGHT = 2.997925e5

GLOBAL_INIT_KEYS = (
    "D_c", "log_MBH", "dv_sys", "x0", "y0",
    "i0", "di_dr", "d2i_dr2", "Omega0", "dOmega_dr", "d2Omega_dr2",
    "e_x", "e_y", "dperiapsis_dr",
    "sigma_x_floor", "sigma_y_floor", "sigma_v_sys", "sigma_v_hv",
    "sigma_a_floor")


def loglik_context(galaxy, n_spots, dataset):
    """Import the f2py Reid likelihood and build `dataset`'s data once.

    Returns ``(rp, rr, d, dataset)`` or None if reidlik is not built in this
    environment or the spot count differs from CANDEL.
    """
    helper_dir = os.path.dirname(__file__)
    for p in (helper_dir, os.path.join(helper_dir, "reidlik_build")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import prepare_reid_data
        import reid_profile as rp
        import run_reid_mcmc as rr
    except ImportError as exc:
        print(f"reid_chi2: reidlik unavailable ({exc})")
        return None
    rp.setup_numbers()
    tmp = tempfile.NamedTemporaryFile(
        suffix="_reid_chi2.inp", delete=False)
    inp = tmp.name
    tmp.close()
    try:
        prepare_reid_data.main([galaxy, "--out", inp, "--dataset", dataset])
        d = rp.build_data(inp)
    finally:
        os.unlink(inp)
    if d["N"] != int(n_spots):
        print(f"reid_chi2: spot count mismatch (Reid {d['N']} vs {n_spots}).")
        return None
    return rp, rr, d, dataset


def _h0_for_D_A(rp, g, D_A):
    """H0 that makes Reid calc_warped_model use the requested D_A exactly."""
    v = float(g["Vsys_km_s"] + g["Vcor_km_s"])
    n_v = int(v + 0.5)
    eq14int, _ = rp.reidlik.dampc(float(n_v), 100.0)
    return CLIGHT * float(eq14int) / (float(D_A) * (1.0 + v / CLIGHT))


def neg_half_chi2(ctx, galaxy, point, r_ang, phi, D_A=None):
    """Reid fit_disk per-spot data-fit term -0.5*chi^2 at fixed (r_ang, phi).

    Uses Reid's own predictions (``calc_warped_model``) and error floors
    (``add_error_floors``).  chi^2 is dimensionless so the result carries no
    normalisation constants.  ``D_A`` (if given) resets Reid's H0 so its
    internal angular-diameter distance equals CANDEL's.
    """
    rp, rr, d, dataset = ctx
    r_ang = np.asarray(r_ang, dtype=float)
    init_block = {k: float(point[k]) for k in GLOBAL_INIT_KEYS
                  if k in point and np.asarray(point[k]).ndim == 0}
    if "D_c" not in init_block and "D_A" in point:
        init_block["D_A"] = float(point["D_A"])
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump({"model": {"galaxies": {galaxy: {"init": init_block}}}}, tmp)
    tmp.close()
    try:
        # dataset: the r_ang_ref_* warp pivots live in init_<dataset>.toml,
        # not in the one-point fragment written above.
        reid_init = rr.load_toml_init(rr.Path(tmp.name), galaxy, 0.0,
                                      dataset=dataset)
    finally:
        os.unlink(tmp.name)
    g = rp.with_derived(rr.shift_warp_pivots(
        reid_init.values,
        rp.reid_r_ref(d, reid_init.values["x0_mas"],
                      reid_init.values["y0_mas"])))
    if D_A is not None:
        g["H0"] = _h0_for_D_A(rp, g, D_A)
        g = rp.with_derived(g)
    params = rp.globals_to_params(g)
    rp.fill_ez(g["H0"], g["Vsys_km_s"], g["Vcor_km_s"])
    r_ref = rp.reid_r_ref(d, g["x0_mas"], g["y0_mas"])
    res_err = rp.reidlik.add_error_floors(
        params, d["N"], d["vlsr"], d["Vmin"], d["Vmax"],
        d["x_err"], d["y_err"], d["vlsr_err"], d["acc_err"])

    phi_deg = np.degrees(np.asarray(phi, dtype=float))
    data = d["data"]
    measured = np.asarray(d["acc_err"][:d["N"]]) > 0.0
    p = params.copy()
    out = np.zeros(d["N"])
    for i in range(d["N"]):
        base = rp.NUM_GLOBAL + 2 * i
        p[base] = r_ang[i]
        p[base + 1] = phi_deg[i]
        cx, cy, cv, ca = rp.reidlik.calc_warped_model(
            p, rp.NUM_GLOBAL, i + 1, r_ref)
        d4 = data[4 * i:4 * i + 4]
        chi2 = (((d4[0] - cx) / res_err[4 * i]) ** 2
                + ((d4[1] - cy) / res_err[4 * i + 1]) ** 2
                + ((d4[2] - cv) / res_err[4 * i + 2]) ** 2)
        if measured[i]:
            chi2 += ((d4[3] - ca) / res_err[4 * i + 3]) ** 2
        out[i] = -0.5 * chi2
    return out
