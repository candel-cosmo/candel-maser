# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Shared bound/selection helpers for global parameter searches.

Pure NumPy utilities used by the megamaser differential-evolution MAP
driver (``scripts/megamaser/run_de_map.py``).
"""
import numpy as np

# -----------------------------------------------------------------------
# Bounds extraction
# -----------------------------------------------------------------------


def _prior_bounds(dist, sobol_n_sigma=5):
    """Derive finite optimizer bounds from a numpyro distribution.

    For Uniform priors, uses the support directly. For all other
    distributions (Normal, TruncatedNormal, HalfNormal, ...),
    uses mean +/- sobol_n_sigma * std clipped to any finite support bounds.
    This avoids wasting Sobol points in low-probability tails
    (e.g. TruncatedNormal(2, 5, 0, 100) -> [0, 27] not [0, 100]).

    Delta priors return (None, None) — caller should skip.
    """
    dist_name = dist.__class__.__name__
    if dist_name == "Delta":
        return None, None

    support = dist.support
    lb = getattr(support, "lower_bound", None)
    ub = getattr(support, "upper_bound", None)
    lb = float(lb) if lb is not None else -np.inf
    ub = float(ub) if ub is not None else np.inf

    # Uniform or no tightening requested: use support directly
    if dist_name == "Uniform" or sobol_n_sigma is None:
        return lb, ub

    # Try to tighten with mean +/- sobol_n_sigma * std.
    # Some distributions (e.g. TruncatedNormal) don't implement variance
    # but have a base_dist that does.
    try:
        mu = float(dist.mean)
        sigma = float(dist.variance ** 0.5)
    except (NotImplementedError, AttributeError, TypeError):
        base = getattr(dist, "base_dist", None)
        if base is not None:
            try:
                mu = float(base.mean)
                sigma = float(base.variance ** 0.5)
            except (NotImplementedError, AttributeError, TypeError):
                return lb, ub
        else:
            return lb, ub

    tight_lo = mu - sobol_n_sigma * sigma
    tight_hi = mu + sobol_n_sigma * sigma

    # Clip to support (if finite)
    lb = max(lb, tight_lo) if np.isfinite(lb) else tight_lo
    ub = min(ub, tight_hi) if np.isfinite(ub) else tight_hi
    return lb, ub


def _select_distinct(points, logp_vals, M, min_dist_frac=0.01):
    """Select top-M points that are sufficiently distinct.

    Greedily picks the best point, then skips any candidate within
    min_dist_frac (L-inf normalised by range) of already selected points.
    Falls back to top-M if fewer than M distinct points found.
    """
    order = np.argsort(-logp_vals)
    selected = [order[0]]
    spread = np.max(points, axis=0) - np.min(points, axis=0)
    spread = np.where(spread > 0, spread, 1.0)

    for idx in order[1:]:
        if len(selected) >= M:
            break
        pt = points[idx]
        too_close = False
        for s in selected:
            if np.all(np.abs(pt - points[s]) / spread < min_dist_frac):
                too_close = True
                break
        if not too_close:
            selected.append(idx)

    if len(selected) < M:
        for idx in order:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= M:
                break

    return np.array(selected[:M])
