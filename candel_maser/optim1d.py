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

GR = 0.5 * (jnp.sqrt(5.0) - 1.0)
_SQRT_EPS = 1.4901161193847656e-08


def brent_1d(f, a, b, n_steps):
    """Fixed-iteration Brent minimiser on [a, b].

    Follows the Numerical Recipes algorithm: parabolic interpolation
    when acceptable, golden-section fallback otherwise. No gradients.
    All branching via ``jnp.where`` for ``fori_loop`` compatibility.

    Typically reaches machine precision in 15-20 steps for smooth
    unimodal functions.

    Returns ``x_opt`` — the best point found.
    """
    # Initialise at the midpoint.
    x = 0.5 * (a + b)
    fx = f(x)
    w = x;  fw = fx
    v = x;  fv = fx
    e = b - a       # distance moved two steps ago
    d = 0.5 * e     # distance moved last step

    def body(_, state):
        a, b, x, fx, w, fw, v, fv, e, d = state
        mid = 0.5 * (a + b)
        tol1 = _SQRT_EPS * jnp.abs(x) + 1e-10
        tol2 = 2.0 * tol1

        # ── Parabolic interpolation ──
        r = (x - w) * (fx - fv)
        q = (x - v) * (fx - fw)
        p = (x - v) * q - (x - w) * r
        q = 2.0 * (q - r)
        # Make q positive, adjust p sign accordingly.
        p = jnp.where(q > 0, -p, p)
        q = jnp.abs(q)
        old_e = e

        para_d = p / jnp.where(q == 0, 1e-30, q)
        u_para = x + para_d
        para_ok = (
            (jnp.abs(p) < jnp.abs(0.5 * q * old_e))
            & (u_para - a > tol2)
            & (b - u_para > tol2)
        )

        # ── Golden-section fallback ──
        golden_e = jnp.where(x >= mid, a - x, b - x)
        golden_d = GR * golden_e

        # Select step.
        new_e = jnp.where(para_ok, d, golden_e)
        new_d = jnp.where(para_ok, para_d, golden_d)

        # Enforce minimum step size.
        u = x + jnp.where(jnp.abs(new_d) >= tol1,
                           new_d,
                           jnp.copysign(tol1, new_d))
        fu = f(u)

        # ── Update bracket ──
        # If fu <= fx, u is new best: shrink bracket toward u.
        # If fu > fx, x stays best: shrink bracket from the u side.
        a_new = jnp.where(
            fu <= fx,
            jnp.where(u < x, a, x),    # u is best: keep a (u left) or raise a to x (u right)
            jnp.where(u < x, u, a))    # x is best: raise a to u (u left) or keep a
        b_new = jnp.where(
            fu <= fx,
            jnp.where(u < x, x, b),    # u is best: lower b to x (u left) or keep b
            jnp.where(u < x, b, u))    # x is best: keep b (u left) or lower b to u

        # ── Update best / second / third ──
        # Case: u is new best.
        x1 = u;  fx1 = fu
        w1 = x;  fw1 = fx
        v1 = w;  fv1 = fw
        # Case: u improves on w (or w == x).
        w2 = u;  fw2 = fu
        v2 = v;  fv2 = fv
        update_v2 = (fu <= fv) | (v == x) | (v == w)
        v2 = jnp.where(update_v2, u, v)
        fv2 = jnp.where(update_v2, fu, fv)
        # Case: u only improves on v.
        v3 = v;  fv3 = fv
        update_v3 = (fu <= fv) | (v == x) | (v == w)
        v3 = jnp.where(update_v3, u, v)
        fv3 = jnp.where(update_v3, fu, fv)

        better = fu <= fx
        second = (~better) & ((fu <= fw) | (w == x))

        x_out = jnp.where(better, x1, x)
        fx_out = jnp.where(better, fx1, fx)
        w_out = jnp.where(better, w1, jnp.where(second, w2, w))
        fw_out = jnp.where(better, fw1, jnp.where(second, fw2, fw))
        v_out = jnp.where(better, v1, jnp.where(second, v2, v3))
        fv_out = jnp.where(better, fv1, jnp.where(second, fv2, fv3))

        return (a_new, b_new, x_out, fx_out, w_out, fw_out,
                v_out, fv_out, new_e, new_d)

    state = (a, b, x, fx, w, fw, v, fv, e, d)
    state = jax.lax.fori_loop(0, n_steps, body, state)
    return state[2]  # x
