import pytest

from ud_genre_bootstrap.utils.config import Config


def test_parse_fit_sample_size_null_string():
    cfg = Config.from_dict({"clustering": {"fit_sample_size": "null"}})
    assert cfg.clustering.fit_sample_size is None


def test_parse_fit_sample_size_numeric_string():
    cfg = Config.from_dict({"clustering": {"fit_sample_size": "50000"}})
    assert cfg.clustering.fit_sample_size == 50000


def test_parse_fit_sample_size_rejects_invalid_string():
    with pytest.raises(ValueError, match="clustering.fit_sample_size"):
        Config.from_dict({"clustering": {"fit_sample_size": "not-a-number"}})


def test_parse_metadata_anchor_mode_parity():
    cfg = Config.from_dict(
        {"evaluation": {"metadata_validation": {"anchor_mode": "parity"}}}
    )
    assert cfg.evaluation.metadata_validation.anchor_mode == "parity"


def test_parse_metadata_anchor_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="evaluation.metadata_validation.anchor_mode"):
        Config.from_dict(
            {"evaluation": {"metadata_validation": {"anchor_mode": "invalid"}}}
        )


def test_parse_reference_weighting_uniform():
    cfg = Config.from_dict({"bootstrapping": {"reference_weighting": "uniform"}})
    assert cfg.bootstrapping.reference_weighting == "uniform"


def test_parse_reference_weighting_rejects_invalid_value():
    with pytest.raises(ValueError, match="bootstrapping.reference_weighting"):
        Config.from_dict({"bootstrapping": {"reference_weighting": "invalid"}})
