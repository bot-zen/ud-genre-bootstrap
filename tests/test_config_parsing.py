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
