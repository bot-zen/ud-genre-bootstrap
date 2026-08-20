"""Tests for CLI helper functions."""

import json

import pytest
from ud_genre_bootstrap.cli import (
    apply_treebank_exclusions,
    build_fixed_partition_protocol_report,
    build_paper_treebank_key,
    build_progressive_treebank_sets,
    normalize_paper_treebank_key,
    check_evaluation_fold_feasibility,
    normalize_anchor_mode,
    normalize_anchor_pool_policy,
    normalize_evaluation_protocol,
    parse_inline_treebank_sets,
    resolve_paper_evaluation_treebank_genres,
    resolve_paper_evaluation_treebank_ids,
)
from ud_genre_bootstrap.utils.config import Config


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self, treebank_codes, metadata=None):
        self._treebank_codes = treebank_codes
        self.metadata = metadata or {}

    def get_treebank_codes(self):
        """Return mock treebank codes."""
        return self._treebank_codes


class TestPaperEvaluationScope:
    """Test helpers for paper-parity evaluation scope."""

    def test_normalize_paper_treebank_key_handles_language_aliases(self):
        """Paper key normalization should collapse known spelling variants."""
        assert normalize_paper_treebank_key("Portugese/PUD") == normalize_paper_treebank_key(
            "Portuguese/PUD"
        )

    def test_build_paper_treebank_key_uses_dirname(self):
        """Paper-style keys should be reconstructed from UD metadata."""
        assert build_paper_treebank_key(
            "en_ewt", {"language": "English", "dirname": "UD_English-EWT"}
        ) == "English/EWT"

    def test_resolve_paper_evaluation_treebank_genres_matches_vendored_scope(self, tmp_path):
        """Current treebank IDs should resolve against vendored paper mapping keys."""
        mapping_path = tmp_path / "paper-scope.json"
        mapping_path.write_text(
            json.dumps({
                "English/EWT": {"blog": "blog", "email": "email"},
                "Portugese/PUD": {"n": "news", "w": "wiki"},
            }),
            encoding="utf-8",
        )
        data_loader = MockDataLoader(
            ["en_ewt", "en_gum", "pt_pud"],
            metadata={
                "en_ewt": {"language": "English", "dirname": "UD_English-EWT"},
                "en_gum": {"language": "English", "dirname": "UD_English-GUM"},
                "pt_pud": {"language": "Portuguese", "dirname": "UD_Portuguese-PUD"},
            },
        )

        result = resolve_paper_evaluation_treebank_genres(data_loader, mapping_path)

        assert result == {
            "en_ewt": ["blog", "email"],
            "pt_pud": ["news", "wiki"],
        }

    def test_resolve_paper_evaluation_treebank_ids_matches_vendored_scope(self, tmp_path):
        """Current treebank IDs should resolve against vendored paper mapping keys."""
        mapping_path = tmp_path / "paper-scope.json"
        mapping_path.write_text(
            json.dumps({
                "English/EWT": {"blog": "blog"},
                "Portugese/PUD": {"n": "news", "w": "wiki"},
            }),
            encoding="utf-8",
        )
        data_loader = MockDataLoader(
            ["en_ewt", "en_gum", "pt_pud"],
            metadata={
                "en_ewt": {"language": "English", "dirname": "UD_English-EWT"},
                "en_gum": {"language": "English", "dirname": "UD_English-GUM"},
                "pt_pud": {"language": "Portuguese", "dirname": "UD_Portuguese-PUD"},
            },
        )

        result = resolve_paper_evaluation_treebank_ids(data_loader, mapping_path)

        assert result == {"en_ewt", "pt_pud"}


class TestApplyTreebankExclusions:
    """Test apply_treebank_exclusions helper function."""

    def test_no_exclusions_returns_original_filter(self):
        """Test that with no exclusions, original filter is returned."""
        config = Config()
        config.exclude_treebanks = None

        data_loader = MockDataLoader(["en_ewt", "de_gsd", "fr_gsd"])
        treebank_filter = ["en_ewt", "de_gsd"]

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        assert result == treebank_filter
        assert result == ["en_ewt", "de_gsd"]

    def test_no_exclusions_empty_list_returns_original_filter(self):
        """Test that empty exclusion list returns original filter."""
        config = Config()
        config.exclude_treebanks = []

        data_loader = MockDataLoader(["en_ewt", "de_gsd", "fr_gsd"])
        treebank_filter = ["en_ewt", "de_gsd"]

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        assert result == treebank_filter

    def test_exclude_from_explicit_filter(self):
        """Test excluding treebanks from an explicit filter list."""
        config = Config()
        config.exclude_treebanks = ["de_gsd", "ar_nyuad"]

        data_loader = MockDataLoader(["en_ewt", "de_gsd", "fr_gsd", "ar_nyuad"])
        treebank_filter = ["en_ewt", "de_gsd", "fr_gsd"]

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        assert result == ["en_ewt", "fr_gsd"]
        assert "de_gsd" not in result
        assert "ar_nyuad" not in result

    def test_exclude_when_no_filter_provided(self):
        """Test that exclusions work when no initial filter is provided."""
        config = Config()
        config.exclude_treebanks = ["en_lines", "ar_nyuad"]

        data_loader = MockDataLoader(["en_ewt", "en_lines", "de_gsd", "ar_nyuad", "fr_gsd"])
        treebank_filter = None

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        # Should return all treebanks except excluded ones
        assert "en_ewt" in result
        assert "de_gsd" in result
        assert "fr_gsd" in result
        assert "en_lines" not in result
        assert "ar_nyuad" not in result

    def test_exclude_all_from_filter(self):
        """Test excluding all treebanks from filter."""
        config = Config()
        config.exclude_treebanks = ["en_ewt", "de_gsd"]

        data_loader = MockDataLoader(["en_ewt", "de_gsd", "fr_gsd"])
        treebank_filter = ["en_ewt", "de_gsd"]

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        assert result == []

    def test_exclude_nonexistent_treebank(self):
        """Test that excluding non-existent treebank doesn't cause errors."""
        config = Config()
        config.exclude_treebanks = ["nonexistent_tb"]

        data_loader = MockDataLoader(["en_ewt", "de_gsd"])
        treebank_filter = ["en_ewt", "de_gsd"]

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        # Should return original filter since excluded treebank doesn't exist
        assert result == ["en_ewt", "de_gsd"]

    def test_exclude_preserves_order(self):
        """Test that exclusion preserves the original order of treebanks."""
        config = Config()
        config.exclude_treebanks = ["de_gsd"]

        data_loader = MockDataLoader(["en_ewt", "de_gsd", "fr_gsd", "es_ancora"])
        treebank_filter = ["en_ewt", "de_gsd", "fr_gsd", "es_ancora"]

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        assert result == ["en_ewt", "fr_gsd", "es_ancora"]
        # Order should be preserved

    def test_exclude_with_duplicates_in_filter(self):
        """Test excluding when filter contains duplicates."""
        config = Config()
        config.exclude_treebanks = ["de_gsd"]

        data_loader = MockDataLoader(["en_ewt", "de_gsd", "fr_gsd"])
        treebank_filter = ["en_ewt", "de_gsd", "en_ewt"]  # Duplicate

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        # Should remove de_gsd but keep both en_ewt
        assert "de_gsd" not in result
        assert result.count("en_ewt") == 2

    def test_multiple_exclusions(self):
        """Test excluding multiple treebanks."""
        config = Config()
        config.exclude_treebanks = ["en_lines", "de_lit", "ar_nyuad"]

        data_loader = MockDataLoader([
            "en_ewt", "en_lines", "de_gsd", "de_lit", "ar_nyuad", "fr_gsd"
        ])
        treebank_filter = None

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        assert "en_ewt" in result
        assert "de_gsd" in result
        assert "fr_gsd" in result
        assert "en_lines" not in result
        assert "de_lit" not in result
        assert "ar_nyuad" not in result

    def test_case_sensitive_exclusion(self):
        """Test that exclusion is case-sensitive."""
        config = Config()
        config.exclude_treebanks = ["EN_EWT"]  # Wrong case

        data_loader = MockDataLoader(["en_ewt", "de_gsd"])
        treebank_filter = ["en_ewt", "de_gsd"]

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        # Should not exclude en_ewt (case doesn't match)
        assert result == ["en_ewt", "de_gsd"]


class TestExclusionConfiguration:
    """Test exclude_treebanks configuration field."""

    def test_config_has_exclude_treebanks_field(self):
        """Test that Config has exclude_treebanks field."""
        config = Config()

        # Should have the field (even if None by default)
        assert hasattr(config, "exclude_treebanks")

    def test_config_accepts_exclusion_list(self):
        """Test that Config accepts a list of treebanks to exclude."""
        config = Config()
        config.exclude_treebanks = ["en_lines", "ar_nyuad"]

        assert config.exclude_treebanks == ["en_lines", "ar_nyuad"]
        assert isinstance(config.exclude_treebanks, list)

    def test_config_handles_empty_exclusion_list(self):
        """Test that Config handles empty exclusion list."""
        config = Config()
        config.exclude_treebanks = []

        assert config.exclude_treebanks == []
        assert isinstance(config.exclude_treebanks, list)

    def test_config_handles_none_exclusion(self):
        """Test that Config handles None for exclusions."""
        config = Config()
        config.exclude_treebanks = None

        assert config.exclude_treebanks is None

    def test_config_accepts_named_evaluation_sets(self):
        """Evaluation config should store named treebank sets for reproducible runs."""
        config = Config()
        config.evaluation.treebank_sets = {
            "lit_small": ["de_pud", "cs_pdtc"],
            "lit_full": ["de_pud", "cs_pdtc", "en_pud"],
        }

        assert "lit_small" in config.evaluation.treebank_sets
        assert config.evaluation.treebank_sets["lit_full"] == [
            "de_pud",
            "cs_pdtc",
            "en_pud",
        ]


class TestExclusionIntegration:
    """Integration tests for exclusion functionality."""

    def test_exclusion_works_with_single_treebank_filter(self):
        """Test exclusion when filtering to a single treebank."""
        config = Config()
        config.exclude_treebanks = ["en_ewt"]

        data_loader = MockDataLoader(["en_ewt", "de_gsd", "fr_gsd"])
        treebank_filter = ["en_ewt"]  # User wants only en_ewt

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        # Config excludes en_ewt, so result should be empty
        assert result == []

    def test_exclusion_warning_message(self, capsys):
        """Test that exclusion produces a warning message."""
        config = Config()
        config.exclude_treebanks = ["en_lines", "ar_nyuad"]

        data_loader = MockDataLoader(["en_ewt", "en_lines", "de_gsd", "ar_nyuad"])
        treebank_filter = ["en_ewt", "en_lines", "ar_nyuad"]

        # This would normally print to console, but we're just testing the logic
        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        # Should have filtered out excluded treebanks
        assert result == ["en_ewt"]
        assert len(result) == 1

    def test_no_warning_when_exclusions_dont_match(self, capsys):
        """Test that no warning when exclusions don't match any treebanks."""
        config = Config()
        config.exclude_treebanks = ["nonexistent"]

        data_loader = MockDataLoader(["en_ewt", "de_gsd"])
        treebank_filter = ["en_ewt", "de_gsd"]

        result = apply_treebank_exclusions(config, data_loader, treebank_filter)

        # Should return original filter
        assert result == ["en_ewt", "de_gsd"]


class TestEvaluationSetHelpers:
    """Tests for evaluation-set helper logic."""

    def test_parse_inline_treebank_sets(self):
        """Inline set definitions should parse into named treebank lists."""
        parsed = parse_inline_treebank_sets(
            [
                "lit_small=de_pud,cs_pdtc",
                "lit_full:de_pud,cs_pdtc,en_pud",
            ]
        )

        assert parsed == {
            "lit_small": ["de_pud", "cs_pdtc"],
            "lit_full": ["de_pud", "cs_pdtc", "en_pud"],
        }

    def test_parse_inline_treebank_sets_rejects_invalid_definition(self):
        """Invalid set definitions should fail fast."""
        with pytest.raises(ValueError, match="Invalid --treebank-set"):
            parse_inline_treebank_sets(["invalid_format"])

        with pytest.raises(ValueError, match="Duplicate evaluation set name"):
            parse_inline_treebank_sets(["a=de_pud", "a=en_pud"])

    def test_build_progressive_treebank_sets_orders_by_virtual_split_potential(self):
        """Progressive sets should prioritize treebanks contributing more virtual splits."""
        multi_genre_treebanks = [
            {
                "treebank": "tb_alpha",
                "split": "train",
                "genres": ["news", "wiki", "social"],
                "language": "en",
                "sentence_count": 100,
            },
            {
                "treebank": "tb_beta",
                "split": "train",
                "genres": ["news", "wiki"],
                "language": "de",
                "sentence_count": 80,
            },
            {
                "treebank": "tb_gamma",
                "split": "train",
                "genres": ["news", "fiction"],
                "language": "fr",
                "sentence_count": 60,
            },
        ]

        progressive_sets = build_progressive_treebank_sets(
            multi_genre_treebanks,
            min_size=2,
            step=1,
        )

        assert progressive_sets["progressive_top_2"] == ["tb_alpha", "tb_beta"]
        assert progressive_sets["progressive_top_3"] == ["tb_alpha", "tb_beta", "tb_gamma"]


class TestAnchorModeHelpers:
    """Tests for evaluation anchor-mode parsing."""

    def test_normalize_anchor_mode_prefers_cli_value(self):
        assert normalize_anchor_mode("parity", "strict") == "parity"

    def test_normalize_anchor_mode_uses_config_default(self):
        assert normalize_anchor_mode(None, "parity") == "parity"

    def test_normalize_anchor_mode_rejects_invalid_values(self):
        with pytest.raises(ValueError, match="Invalid anchor mode"):
            normalize_anchor_mode("unknown", "strict")

    def test_check_evaluation_fold_feasibility_respects_grouping_constraints(self):
        """Fold feasibility should enforce minimum groups for grouped CV."""
        multi_genre_treebanks = [
            {"treebank": "tb1", "split": "train", "language": "en"},
            {"treebank": "tb2", "split": "train", "language": "en"},
            {"treebank": "tb3", "split": "train", "language": "de"},
        ]

        ok, reason = check_evaluation_fold_feasibility(
            multi_genre_treebanks,
            n_folds=3,
            group_by="language",
        )
        assert not ok
        assert "requires at least 3 languages" in reason

        ok, reason = check_evaluation_fold_feasibility(
            multi_genre_treebanks,
            n_folds=3,
            group_by="treebank",
        )
        assert ok
        assert reason == ""

    def test_normalize_anchor_pool_policy_auto_resolves_from_mode(self):
        assert (
            normalize_anchor_pool_policy(None, "auto", anchor_mode="strict")
            == "train_virtual"
        )
        assert (
            normalize_anchor_pool_policy(None, "auto", anchor_mode="parity")
            == "combined"
        )

    def test_normalize_anchor_pool_policy_accepts_aliases(self):
        assert (
            normalize_anchor_pool_policy(
                "single_genre_only",
                "auto",
                anchor_mode="strict",
            )
            == "single_genre"
        )

    def test_normalize_anchor_pool_policy_rejects_invalid_values(self):
        with pytest.raises(ValueError, match="Invalid anchor pool policy"):
            normalize_anchor_pool_policy("unknown", "auto", anchor_mode="strict")


    def test_normalize_evaluation_protocol_prefers_cli_value(self):
        assert normalize_evaluation_protocol("paper_parity", "generalization") == "paper_parity"

    def test_normalize_evaluation_protocol_uses_config_default(self):
        assert normalize_evaluation_protocol(None, "paper_parity") == "paper_parity"

    def test_normalize_evaluation_protocol_accepts_aliases(self):
        assert normalize_evaluation_protocol("paper", "generalization") == "paper_parity"

    def test_normalize_evaluation_protocol_rejects_invalid_values(self):
        with pytest.raises(ValueError, match="Invalid evaluation protocol"):
            normalize_evaluation_protocol("unknown", "generalization")


def test_build_fixed_partition_protocol_report_clean_generalization_case():
    report = build_fixed_partition_protocol_report(
        protocol="generalization",
        anchor_partitions=["train", "dev"],
        test_partition="test",
        requested_anchor_split_keys=[("de_pud", "test")],
        requested_test_split_keys=[("cs_pdtc", "test")],
    )

    assert report["protocol_scope"]["requested_anchor_split_keys"] == 1
    assert report["protocol_scope"]["requested_test_split_keys"] == 1
    assert report["protocol_notes"] == [
        "Generalization fixed holdout: anchors=train, dev -> test=test."
    ]
    assert report["protocol_deviations"] == []


def test_build_fixed_partition_protocol_report_flags_parity_deviations():
    report = build_fixed_partition_protocol_report(
        protocol="paper_parity",
        anchor_partitions=["test"],
        test_partition="test",
        requested_anchor_split_keys=[("de_pud", "test"), ("ar_nyuad", "test")],
        requested_test_split_keys=[("de_pud", "test"), ("ja_bccwj", "test")],
        source_missing_anchor_split_keys=[("ar_nyuad", "test")],
        source_missing_test_split_keys=[("ja_bccwj", "test")],
        load_error_split_keys=[("cs_cac", "test")],
        no_metadata_test_split_keys=[("pl_pud", "test")],
        dropped_single_anchor_split_keys=[("de_pud", "test")],
        missing_anchor_genres=["email", "government"],
        overlap_sentences=12,
    )

    assert report["protocol_notes"] == [
        "Paper-parity anchor policy: same-partition single-genre anchors from 'test'.",
        "Anchor/test overlap: 12 sentence(s) shared by design.",
    ]
    deviations = "\n".join(report["protocol_deviations"])
    assert "Split-map anchor keys missing from UD source: 1 split(s)" in deviations
    assert "Split-map test keys missing from UD source: 1 split(s)" in deviations
    assert "Load errors while reading split-map-selected data: 1 split(s)" in deviations
    assert "Split-map-selected test splits without usable genre metadata: 1 split(s)" in deviations
    assert "Single-genre anchor candidates removed after embedding filtering: 1 split(s)" in deviations
    assert "Expected test genres without anchor support: email, government" in deviations


class TestLabelCommandFlow:
    """Test that label command reuses shared bootstrapper logic."""

    def test_label_uses_shared_execute_bootstrap_labeling(self, monkeypatch, tmp_path):
        """label() should call execute_bootstrap_labeling with computed schedule."""
        import ud_genre_bootstrap.cli as cli_module

        cfg = Config()
        cfg.output.genres_path = str(tmp_path)

        class DummyBootstrapper:
            last_instance = None

            def __init__(self, _cfg):
                DummyBootstrapper.last_instance = self
                self.received_schedule = None

            def _generate_embeddings(self):
                return {("xx_demo", "train"): {"sent_id": [], "embedding": []}}

            def _cluster_treebanks(self, _embeddings_by_tb):
                return None

            def _compute_cluster_embeddings(self, _embeddings_by_tb):
                return None

            def _create_schedule(self):
                return [{"known": ["news"], "predict": [], "disjunct": []}]

            def execute_bootstrap_labeling(self, schedule=None):
                self.received_schedule = schedule
                return schedule

            def _export_results(self):
                return {"labeled_sentences": 0}

        monkeypatch.setattr(cli_module, "load_config_from_path", lambda *_, **__: cfg)
        monkeypatch.setattr(cli_module, "GenreBootstrapper", DummyBootstrapper)
        monkeypatch.setattr(cli_module, "_display_schedule_summary", lambda _: None)

        cli_module.label(config=None, clusters=None)

        instance = DummyBootstrapper.last_instance
        assert instance is not None
        assert instance.received_schedule == [
            {"known": ["news"], "predict": [], "disjunct": []}
        ]
