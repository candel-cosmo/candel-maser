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
"""Scalar 1-D optimisation primitives built on top of JAX.

The functions here are deliberately generic — no problem-specific
knowledge. Batching is done by the caller via ``jax.vmap``.
"""
import jax
import jax.numpy as jnp


def damped_newton_1d(f, x0, n_steps, step_max, hess_floor,
                     lo=-jnp.inf, hi=jnp.inf):
    """Fixed-iteration damped-Newton minimiser for a scalar function.

    Iterates ``x ← clip(x - g/max(h, hess_floor), ±step_max)`` for
    ``n_steps`` steps with ``g = f'(x)``, ``h = f''(x)``, both from
    ``jax.grad``. Each step's resulting ``x`` is also clipped to the
    bounds ``[lo, hi]``. Uses ``jax.lax.fori_loop`` so the loop stays
    inside the traced graph.

    Safeguards — ``hess_floor`` prevents division by zero or negative
    curvature; ``step_max`` prevents overshooting when the Hessian is
    small; ``lo``/``hi`` keep ``x`` in a valid domain.

    Not differentiable through ``x0`` or ``f``. Intended for use inside
    a larger JIT graph with the output consumed after
    ``jax.lax.stop_gradient``.

    Parameters
    ----------
    f         callable(scalar) -> scalar, twice-differentiable under JAX.
    x0        scalar starting point.
    n_steps   number of Newton iterations (compile-time constant).
    step_max  maximum |Δx| per iteration.
    hess_floor lower bound on the Hessian used in the divisor.
    lo, hi    optional scalar bounds on ``x``.

    Returns ``(x_opt, h_opt)`` — the optimised ``x`` and the
    floored Hessian at ``x_opt`` (useful as a per-spot local
    curvature estimate for setting downstream quadrature widths).
    """
    df = jax.grad(f)
    d2f = jax.grad(df)

    def body(_, x):
        g = df(x)
        h = jnp.maximum(d2f(x), hess_floor)
        step = jnp.clip(-g / h, -step_max, step_max)
        return jnp.clip(x + step, lo, hi)

    x_opt = jax.lax.fori_loop(0, n_steps, body, x0)
    h_opt = jnp.maximum(d2f(x_opt), hess_floor)
    return x_opt, h_opt
