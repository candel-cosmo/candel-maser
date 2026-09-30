"""Regressions for the sole, Pesce-unseeded megamaser DE path."""

import numpy as np
import pytest
from scipy.stats import ks_2samp


import candel_maser.run_de_map as de


def _reference_lshade_trials(population, fitness, mutation_archive, m_f, m_cr,
                             rng, pbest_fraction=0.11):
    """Pre-vectorisation reference implementation of ``de._lshade_trials``.

    Verbatim copy of the original per-member Python loop, kept private to this
    test file so the vectorised version can be checked distributionally.
    """
    pop = np.asarray(population)
    n, dimension = pop.shape
    if n < 4:
        raise ValueError("L-SHADE requires at least four population members.")
    archive = np.asarray(mutation_archive).reshape(-1, dimension)
    union = np.vstack([pop, archive]) if archive.size else pop
    order = np.argsort(fitness)
    n_pbest = max(2, min(n, int(np.ceil(pbest_fraction * n))))
    memory_slots = rng.integers(len(m_f), size=n)
    f = np.empty(n)
    cr = np.empty(n)
    mutants = np.empty_like(pop)

    def draw_index(limit, forbidden):
        while True:
            value = int(rng.integers(limit))
            if value not in forbidden:
                return value

    for i, slot in enumerate(memory_slots):
        value = -1.0
        while value <= 0.0:
            value = m_f[slot] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
        f[i] = min(value, 1.0)
        cr[i] = (0.0 if m_cr[slot] < 0.0 else
                 np.clip(rng.normal(m_cr[slot], 0.1), 0.0, 1.0))
        pbest_pool = order[:n_pbest]
        pbest_pool = pbest_pool[pbest_pool != i]
        pbest = int(rng.choice(pbest_pool))
        r1 = draw_index(n, {i, pbest})
        r2 = draw_index(len(union), {i, pbest, r1})
        mutants[i] = (pop[i] + f[i] * (pop[pbest] - pop[i])
                      + f[i] * (pop[r1] - union[r2]))

    mutants = np.abs(mutants)
    cycle = np.floor(mutants).astype(np.int32)
    frac = mutants - np.floor(mutants)
    mutants = np.where(cycle % 2 == 0, frac, 1.0 - frac)
    cross = rng.random((n, dimension)) < cr[:, None]
    cross[np.arange(n), rng.integers(dimension, size=n)] = True
    return np.where(cross, mutants, pop), f, cr


def test_distance_gaussian_uses_local_logp_curvature():
    lo = np.array([80.0, -1.0])
    hi = np.array([120.0, 1.0])
    best = np.array([0.5, 0.5])
    sigma = 5.0

    def exact_eval(points):
        distance = lo[0] + np.asarray(points)[:, 0] * (hi[0] - lo[0])
        return 0.5 * ((distance - 100.0) / sigma)**2

    result = de._estimate_distance_gaussian(
        exact_eval, best, 0.0, 0, lo, hi)

    assert result["sigma"] == pytest.approx(sigma)
    assert result["gradient"] == pytest.approx(0.0, abs=1e-12)
    assert result["precision"] == pytest.approx(1.0 / sigma**2)


def test_lshade_trials_are_reproducible_and_bounded():
    population = np.linspace(0.05, 0.95, 24).reshape(8, 3)
    fitness = np.arange(8.0)
    archive = np.empty((0, 3))
    m_f = np.full(6, 0.5)
    m_cr = np.full(6, 0.5)

    out1 = de._lshade_trials(
        population, fitness, archive, m_f, m_cr,
        np.random.default_rng(123))
    out2 = de._lshade_trials(
        population, fitness, archive, m_f, m_cr,
        np.random.default_rng(123))

    for first, second in zip(out1, out2):
        np.testing.assert_allclose(first, second)
    trials, mutation, crossover = out1
    assert trials.shape == population.shape
    assert np.all((trials >= 0.0) & (trials <= 1.0))
    assert np.all((mutation > 0.0) & (mutation <= 1.0))
    assert np.all((crossover >= 0.0) & (crossover <= 1.0))


def test_lshade_trials_match_reference_distribution():
    rng0 = np.random.default_rng(7)
    n, dimension = 64, 6
    pop = rng0.random((n, dimension))
    fitness = rng0.random(n)
    archive = rng0.random((32, dimension))
    m_f = np.array([0.3, 0.5, 0.7, 0.5, 0.9, 0.4])
    m_cr = np.array([0.5, 0.2, -1.0, 0.8, 0.6, 0.3])
    n_batches = 400

    def collect(fn):
        fs, crs, disp, nocross = [], [], [], []
        for b in range(n_batches):
            rng = np.random.default_rng(1000 + b)
            trials, f, cr = fn(pop, fitness, archive, m_f, m_cr, rng)
            fs.append(f)
            crs.append(cr)
            disp.append(trials - pop)
            nocross.append(trials == pop)
        return (np.concatenate(fs), np.concatenate(crs),
                np.concatenate(disp, axis=0), np.concatenate(nocross, axis=0))

    f_new, cr_new, disp_new, nc_new = collect(de._lshade_trials)
    f_ref, cr_ref, disp_ref, nc_ref = collect(_reference_lshade_trials)

    ks_f = ks_2samp(f_new, f_ref)
    assert ks_f.pvalue > 1e-3, ks_f
    ks_cr = ks_2samp(cr_new[cr_new > 0], cr_ref[cr_ref > 0])
    assert ks_cr.pvalue > 1e-3, ks_cr

    np.testing.assert_allclose(
        disp_new.mean(axis=0), disp_ref.mean(axis=0), atol=0.02)
    np.testing.assert_allclose(
        np.cov(disp_new, rowvar=False), np.cov(disp_ref, rowvar=False),
        atol=0.02)
    np.testing.assert_allclose(nc_new.mean(), nc_ref.mean(), rtol=0.05)


def test_lshade_draw_indices_respect_exclusions():
    rng = np.random.default_rng(2024)
    for n, n_extra in [(4, 0), (8, 8), (17, 5), (32, 32)]:
        n_union = n + n_extra
        order = rng.permutation(n)
        n_pbest = max(2, int(np.ceil(0.11 * n)))
        pool = order[:n_pbest]
        idx = np.arange(n)
        for _ in range(400):
            pbest, r1, r2 = de._lshade_draw_indices(
                order, n_pbest, n, n_union, rng)
            assert np.all(np.isin(pbest, pool))
            assert np.all(pbest != idx)
            assert np.all(r1 != idx) and np.all(r1 != pbest)
            assert np.all((r2 != idx) & (r2 != pbest) & (r2 != r1))
            assert np.all(r1 < n) and np.all(r2 < n_union)


def test_fixed_device_block_evaluator_accepts_arbitrary_population_sizes():
    evaluate = de._make_batched_fitness(
        lambda row: de.jnp.sum(row ** 2), 1, ())
    first = np.arange(9.0).reshape(3, 3)
    second = np.arange(51.0).reshape(17, 3)

    np.testing.assert_allclose(
        evaluate(first), np.sum(first ** 2, axis=1))
    np.testing.assert_allclose(
        evaluate(second), np.sum(second ** 2, axis=1))
    profile = evaluate.device_profile()
    np.testing.assert_array_equal(profile["last_real_candidates"], [17])
    np.testing.assert_array_equal(profile["last_candidates"], [24])
    assert profile["block_size"] == 8
    assert profile["candidates_per_wave"] == 1
    assert profile["rebalances"] == 0
