"""Regressions for megamaser MCMC summary performance."""

import numpy as np


import candel_maser.run_maser as rm


def _direct_ess(x):
    y = x - x.mean()
    acov = np.correlate(y, y, mode="full")[x.size - 1:] / x.size
    rho = acov / acov[0]
    positive = rho[1:][rho[1:] > 0]
    first_nonpositive = np.flatnonzero(rho[1:] <= 0)
    if first_nonpositive.size:
        positive = rho[1:first_nonpositive[0] + 1]
    tau = 1.0 + 2.0 * positive.sum()
    return max(1.0, min(x.size, x.size / tau))


def test_fft_ess_matches_direct_autocorrelation():
    rng = np.random.default_rng(4)
    x = np.empty(1024)
    x[0] = rng.normal()
    for i in range(1, x.size):
        x[i] = 0.85 * x[i - 1] + rng.normal()

    np.testing.assert_allclose(rm._ess_1d(x), _direct_ess(x), rtol=1e-12)
