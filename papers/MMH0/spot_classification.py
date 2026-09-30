# Copyright (C) 2026 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Position-velocity diagram of maser spots coloured by spectral class."""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  registers the "science" matplotlib style
from palette import BLUESHIFTED, REDSHIFTED

# (legend label, mask key, colour, marker)
_CLASSES = [
    ("Systemic", "sys", "0.30", "o"),
    ("Approaching", "blue", BLUESHIFTED, "v"),
    ("Receding", "red", REDSHIFTED, "^"),
]


def _mathrm(s):
    """Wrap a label in upright math font: spaces -> thin space, hyphen kept."""
    return (
        r"$\mathrm{" + s.replace("-", r"\mbox{-}").replace(" ", r"\,") + "}$"
    )


def major_axis_offset(x, y, dv):
    """Spot position projected onto the disc major axis (input length units).

    The major axis is recovered model-free as the maximum-variance direction
    of the spot positions: the disc is edge-on, so the positions spread by
    the disc diameter along the line of nodes and only by the disc thickness
    across it,
    making the first principal component the major axis. This projection equals
    the impact parameter ``r*sin(phi)`` of the warped-disc model. The sign is
    fixed so the offset correlates with velocity (rotation-curve orientation).
    """
    P = np.column_stack([x, y]).astype(float)
    P = P - P.mean(0)
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    s = P @ vt[0]
    if np.corrcoef(s, dv)[0, 1] < 0:
        s = -s
    return s


def plot_spot_classification(galaxies, sort_by_count=True, style=("science",)):
    """Position-velocity diagram for a set of megamaser galaxies.

    Parameters
    ----------
    galaxies : list of (str, dict)
        Display name and the spot-data dict from
        ``candel_maser.megamaser_data.load_megamaser_spots``; needs keys
        ``velocity``, ``x``, ``y`` (microarcsec), ``is_highvel``, ``is_blue``,
        and ``n_spots``.
    sort_by_count : bool
        Order panels by descending spot count.
    style : sequence of str
        Matplotlib styles applied within a context manager.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = []
    for disp, d in galaxies:
        sys_mask = ~d["is_highvel"]
        # Systemic velocity = median of the systemic-class spots; frame-free
        # and matches the disc systemic velocity the systemic masers trace.
        dv = d["velocity"] - np.median(d["velocity"][sys_mask])
        s = major_axis_offset(d["x"], d["y"], dv) / 1e3  # uas -> mas
        panels.append((disp, dv, s, sys_mask, d["is_blue"],
                       d["is_highvel"] & ~d["is_blue"], int(d["n_spots"])))
    if sort_by_count:
        panels.sort(key=lambda p: p[-1], reverse=True)

    with plt.style.context(list(style)):
        fig, axes = plt.subplots(1, len(panels), figsize=(7.1, 2.5),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, (disp, dv, s, sys_mask, blue_mask, red_mask, n) in zip(
                axes, panels):
            masks = {"sys": sys_mask, "blue": blue_mask, "red": red_mask}
            for label, mkey, colour, marker in _CLASSES:
                m = masks[mkey]
                ax.scatter(s[m], dv[m] / 1e3, s=9, c=colour, marker=marker,
                           edgecolors="none", label=_mathrm(label))
            ax.set_title(_mathrm(f"{disp}, {n} spots"), fontsize=8)
            ax.axhline(0.0, color="0.7", lw=0.6, ls="--", zorder=0)
            ax.set_xlabel(_mathrm("Major-axis offset [mas]"))
        axes[0].set_ylabel(
            r"$v_{\rm los} - v_{\rm sys}\ [10^3\,\mathrm{km\,s^{-1}}]$")
        axes[0].legend(loc="upper left", fontsize="small", markerscale=1.8,
                       frameon=False, handletextpad=0.1, borderaxespad=0.2)
    return fig
