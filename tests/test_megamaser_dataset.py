"""Regressions for the megamaser spot-table dataset switch.

The failure this guards against is silent: a wrong-dataset init block runs to
completion because ``maser_blackjax`` falls back to ``zeros_like(r_hat)`` when
``r_ang``'s length does not match ``n_spots``, and ``r_ang`` is not among the
validated scalar sites.
"""
import os

import numpy as np
from numpyro.distributions import DoublyTruncatedPowerLaw
import pytest
import tomli
import tomli_w

from candel_maser.model_H0_maser import MaserDiskModel
from candel_maser.megamaser_data import (load_megamaser_spots,
                                         maser_data_root)
from candel_maser.maser_config import (apply_dataset, check_init_block,
                                       dataset_init_path)

from candel_maser.paths import CONFIG_PATH, DATA_ROOT

# Spot counts of each dataset, from the tables themselves. The fiducial counts
# match the per-galaxy provenance the MCP gave for their vetting (see the
# checked-in P20 clipping audit).
EXPECTED_N_SPOTS = {
    "original_published": {"CGCG074-064": 165, "NGC4258": 358,
                           "NGC5765b": 212, "NGC6264": 66,
                           "NGC6323": 68, "UGC3789": 156},
    "fiducial": {"CGCG074-064": 165, "NGC4258": 358,
                 "NGC5765b": 169, "NGC6264": 61,
                 "NGC6323": 87, "UGC3789": 153},
    "unpruned": {"CGCG074-064": 165, "NGC4258": 358,
                 "NGC5765b": 212, "NGC6264": 66,
                 "NGC6323": 87, "UGC3789": 156},
}
SOURCE_DATASETS = tuple(EXPECTED_N_SPOTS)


def _config():
    with open(CONFIG_PATH, "rb") as f:
        return tomli.load(f)


def _load(dataset, galaxy):
    gcfg = _config()["model"]["galaxies"][galaxy]
    root = maser_data_root(dataset)
    if not os.path.isdir(root):
        pytest.skip(f"external megamaser dataset is not provisioned: {root}")
    return load_megamaser_spots(root, galaxy,
                                v_sys_obs=gcfg.get("v_sys_obs"))


def test_da2_prior_is_applied_to_sampled_da(tmp_path):
    cfg = _config()
    apply_dataset(cfg, "fiducial")
    cfg["model"]["D_c_prior"] = "volume_D_A"
    path = tmp_path / "config.toml"
    with open(path, "wb") as f:
        tomli_w.dump(cfg, f)

    data = _load("fiducial", "NGC6323")
    gcfg = cfg["model"]["galaxies"]["NGC6323"]
    data["D_lo"], data["D_hi"] = gcfg["D_lo"], gcfg["D_hi"]
    model = MaserDiskModel(path, data)

    assert model.D_A_prior == "volume_D_A"
    assert isinstance(model.priors["D"], DoublyTruncatedPowerLaw)
    lo, hi = model.priors["D"].low, model.priors["D"].high
    d1, d2 = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
    assert float(model.priors["D"].log_prob(d2)
                 - model.priors["D"].log_prob(d1)) == pytest.approx(
                     2 * np.log(float(d2 / d1)))


@pytest.mark.parametrize("dataset", SOURCE_DATASETS)
def test_spot_counts_and_provenance(dataset):
    for galaxy, n in EXPECTED_N_SPOTS[dataset].items():
        data = _load(dataset, galaxy)
        assert data["n_spots"] == n, (dataset, galaxy)
        assert data["dataset"] == dataset
        is_ngc5765b = galaxy == "NGC5765b"
        assert ("clump2_floor_mask" in data) == is_ngc5765b
        assert "sigma_a_likelihood" not in data
        assert "error_floor_policy" not in data
        # Every spot is classified into exactly one of the three supports.
        n_sys = int((~data["is_highvel"]).sum())
        n_blue = int(data["is_blue"].sum())
        assert 0 < n_sys < n and 0 < n_blue < n


def test_systemics_without_acceleration_are_retained():
    expected = {"CGCG074-064": 0, "NGC4258": 92, "NGC5765b": 20,
                "NGC6264": 0, "NGC6323": 0, "UGC3789": 0}
    for galaxy, n_missing in expected.items():
        published = _load("original_published", galaxy)
        missing = (~published["is_highvel"]) & (~published["accel_measured"])
        assert int(missing.sum()) == n_missing, galaxy

    published = _load("original_published", "NGC5765b")
    missing = (~published["is_highvel"]) & (~published["accel_measured"])
    assert np.all(published["a"][missing] == 1.0)
    assert np.all(published["sigma_a"][missing] == 1.0)

    printed = np.loadtxt(os.path.join(
        DATA_ROOT, "data", "Megamaser", "NGC5765b_Gao2016_table6_tex.dat"))
    printed = printed[(printed[:, 5] == 1.0) & (printed[:, 6] == 1.0)]
    by_velocity = {velocity: i for i, velocity in
                   enumerate(published["velocity"])}
    idx = np.array([by_velocity[velocity] for velocity in printed[:, 0]])
    np.testing.assert_allclose(published["x"][idx],
                               1000.0 * (printed[:, 1] + 0.002), atol=1e-12)
    np.testing.assert_allclose(published["y"][idx],
                               1000.0 * (printed[:, 3] - 0.013), atol=1e-12)
    np.testing.assert_allclose(published["sigma_x"][idx],
                               1000.0 * printed[:, 2], atol=1e-12)
    np.testing.assert_allclose(published["sigma_y"][idx],
                               1000.0 * printed[:, 4], atol=1e-12)


@pytest.mark.parametrize("dataset", SOURCE_DATASETS)
def test_p20_thresholds_agree_with_kmeans(dataset):
    """The fiducial tables state their own blue/red split; it must not
    repartition the spots relative to the k-means classifier the published
    tables use, or the phi supports change silently."""
    from scipy.cluster.vq import kmeans2
    for galaxy in EXPECTED_N_SPOTS[dataset]:
        data = _load(dataset, galaxy)
        if "spot_type" not in data:
            continue
        v = data["velocity"].astype(np.float64)
        centroids, lab = kmeans2(v, 3, minit="++", seed=42)
        order = np.argsort(centroids)
        remap = np.empty(3, dtype=int)
        remap[order] = np.arange(3)
        km = remap[lab]
        stated = np.array([{"b": 0, "s": 1, "r": 2}[t]
                           for t in data["spot_type"]])
        if galaxy in ("NGC4258", "CGCG074-064"):
            continue          # explicit spot_type from the source table
        assert np.array_equal(stated, km), (dataset, galaxy)


# ---- config split ----


@pytest.mark.parametrize("dataset", SOURCE_DATASETS)
def test_init_r_ang_lengths_match_spot_counts(dataset):
    """Every init block present must belong to its own dataset."""
    with open(dataset_init_path(dataset), "rb") as f:
        init_cfg = tomli.load(f)
    for galaxy, blk in init_cfg["model"]["galaxies"].items():
        for key, sub in blk.items():
            if not key.startswith("init"):
                continue
            assert len(sub["r_ang"]) == EXPECTED_N_SPOTS[dataset][galaxy], (
                dataset, galaxy, key)


# ---- apply_dataset ----


# ---- chain provenance ----


# ---- init guardrails ----

class _FakeModel:
    def __init__(self, n_spots=358, galaxy="NGC4258", dataset="fiducial"):
        self.n_spots = n_spots
        self.galaxy_name = galaxy
        self.dataset = dataset


def test_check_init_block_rejects_wrong_spot_count():
    """The direct guard on the silent zeros fallback."""
    init = {"r_ang": [1.0] * 357}
    with pytest.raises(SystemExit, match="357 r_ang values but dataset"):
        check_init_block(init, _FakeModel(n_spots=358))
