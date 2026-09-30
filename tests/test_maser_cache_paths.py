"""The maser velocity-LOS cache key ignores machine-local paths."""
from candel_maser.joint_H0_helpers import _toy_vlos_cache_path


def test_maser_vlos_cache_path_ignores_machine_paths(tmp_path):
    config = {"io": {"field_cache_dir": str(tmp_path)}}
    payload = {
        "reconstruction": "ManticoreLocalCOLA",
        "field_kwargs": {
            "fpath_root": "/machine-a/fields",
            "which_MAS": "CIC",
        },
    }
    other_machine = {
        **payload,
        "field_kwargs": {
            **payload["field_kwargs"],
            "fpath_root": "/machine-b/fields",
        },
    }
    other_mas = {
        **payload,
        "field_kwargs": {**payload["field_kwargs"], "which_MAS": "PCS"},
    }

    assert (_toy_vlos_cache_path(payload, config)
            == _toy_vlos_cache_path(other_machine, config))
    assert (_toy_vlos_cache_path(payload, config)
            != _toy_vlos_cache_path(other_mas, config))
