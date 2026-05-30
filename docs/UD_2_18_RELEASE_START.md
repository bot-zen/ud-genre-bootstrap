# UD 2.18 Release Start

The UD 2.18 cycle starts from `configs/2.18-community-release.yaml`.

The current release-state document is `docs/UD_2_18_RELEASE.md`.

This is a candidate release profile, not a promoted artifact yet. Keep
`configs/releases/genre_artifacts.yaml` limited to promoted artifacts and add
`ud2.18-full-ud-v1` there only after the full run, evaluation, source tag, and
HF publish are complete.

## Identity

- UD source revision: `2.18`
- Artifact ID: `ud2.18-full-ud-v1`
- HF branch: `2.18`
- HF tag: `artifact/ud2.18-full-ud-v1`
- Source branch: `release/v1`
- Source tag: `source/ud2.18-full-ud-v1`
- Output directory: `output/2.18-community-release/genres`
- HF Git checkout: `../ud_genre-hf/` mapping to `commul/ud_genre`

## First Checks

```bash
uv run ud-genre-bootstrap coverage \
  --config configs/2.18-community-release.yaml \
  --export output/2.18-community-release/coverage.json
```

Run focused metadata checks for the known high-signal treebanks:

```bash
uv run ud-genre-bootstrap test-genres \
  --config configs/2.18-community-release.yaml \
  --treebank ru_taiga \
  --split train \
  --limit 0 \
  --no-examples

uv run ud-genre-bootstrap test-genres \
  --config configs/2.18-community-release.yaml \
  --treebank be_hse \
  --split train \
  --limit 0 \
  --no-examples

uv run ud-genre-bootstrap test-genres \
  --config configs/2.18-community-release.yaml \
  --treebank en_ewt \
  --split train \
  --limit 0 \
  --no-examples
```

If coverage or metadata extraction changed in UD 2.18, update
`configs/metadata_patterns.json`, `configs/pud-patterns.json`, or
`configs/genre_mappings.json` before running the full embedding and labeling
pipeline.

## Full Candidate Run

```bash
uv run ud-genre-bootstrap embed   --config configs/2.18-community-release.yaml
uv run ud-genre-bootstrap cluster --config configs/2.18-community-release.yaml
uv run ud-genre-bootstrap label \
  --config configs/2.18-community-release.yaml \
  --clusters output/2.18-community-release/genres/clusters
uv run ud-genre-bootstrap evaluate --config configs/2.18-community-release.yaml
```

After validation, create `source/ud2.18-full-ud-v1` at the clean source commit,
publish through the HF dataset checkout `../ud_genre-hf/`, and then add the
promoted artifact to the release registry.
