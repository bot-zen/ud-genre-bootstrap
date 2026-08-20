# Release Process

This project publishes synchronized release trains for derived UD genre labels.
Do not create one release note or one committed full config per UD version. The
release matrix, registry, generated manifests, and HF Git tags are the release
record.

## Source Of Truth

- shared profile: `configs/release_profiles/full-ud.yaml`
- release matrix: `configs/releases/full-ud-v1.0.0.yaml`
- train registry: `configs/releases/genre_artifacts.yaml`
- generated local release directory: `output/<ud-version>-community-release/genres`
- HF dataset Git checkout: `../ud_genre-hf/`
- HF dataset repo: `commul/ud_genre`
- upstream UD HF dataset repo: `universal-dependencies/universal_dependencies`

The current configured train is `full-ud-v1.0.0`. It covers UD `2.7` through
`2.18`; UD `2.18` is the default target for HF `main`.

## Identity Model

The public release train has this shape:

```text
<scope>-<label-schema>-vMAJOR.MINOR.PATCH
```

Example: `full-ud-v1.0.0`.

Each UD branch is one projection of that train:

- artifact key: `full-ud-v1.0.0-ud2.18`
- moving HF branch: `2.18`
- immutable HF tag: `artifact/full-ud-v1.0.0/ud2.18`
- source branch: `release/full-ud-v1`
- source tag: `source/full-ud-v1.0.0`

Algorithm settings are not part of the train name. Embedding model, pooling,
clustering method, thresholds, reference weighting, and seed are recorded as
`algorithm_recipe` in `run_metadata.json` and `release_manifest.json`.

## Versioning Policy

- `MAJOR.MINOR.PATCH` belongs to the train, not to one UD version.
- Normal public changes are synchronized: bump the train version and rebuild all
  supported UD versions.
- Latest/default hotfixes are allowed. Publish the new train first for the
  default UD version and HF `main`, mark the registry status `default_hotfix`,
  then rebuild the rest of the inventory. Mark it `complete` only when all
  supported UD branches have the same train version.
- Do not create independent older-UD patch streams. A case like UD `2.17` on
  `v1.0.2` while UD `2.18` is on `v1.0.1` is intentionally avoided.
- Historical source/HF tags and branches using `ud2.X-full-ud-v...`,
  `source/ud2.*`, or `release/v1` were transitional scratch state. They have
  been deleted from this source repository and must not be recreated for
  promoted releases.

## Config Resolution

Use the matrix for release work:

```bash
uv run ud-genre-bootstrap upload \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version 2.18 \
  --dry-run
```

The matrix resolver combines:

1. shared profile settings
2. train identity and source/HF naming templates
3. per-UD overrides, such as evaluation sets or baseline summaries

The resolved config is written to `config.snapshot.yaml` in the generated output
directory. Existing `--config` commands remain available for experiments and
debugging, but promoted release work should use `--release-matrix`.

## Preflight

Run coverage and focused metadata checks before expensive generation:

```bash
uv run ud-genre-bootstrap coverage \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version <UD_VERSION> \
  --export output/<UD_VERSION>-community-release/coverage.json

uv run ud-genre-bootstrap test-genres \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version <UD_VERSION> \
  --treebank ru_taiga \
  --split train \
  --limit 0 \
  --no-examples
```

If coverage or extraction changes, update the shared mapping or pattern files
before full generation.

## Full Generation

Run one UD version end to end before starting the next:

```bash
export HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/"
export HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/"

uv run ud-genre-bootstrap embed \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version <UD_VERSION>

uv run ud-genre-bootstrap cluster \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version <UD_VERSION>

uv run ud-genre-bootstrap label \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version <UD_VERSION> \
  --clusters output/<UD_VERSION>-community-release/genres/clusters

uv run ud-genre-bootstrap evaluate \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version <UD_VERSION>
```

`label` must reuse the saved cluster directory for the promoted run. Running
`label` without `--clusters` can start a new clustering pass.

## Required Local Artifacts

The local release directory should contain:

- `all_genres.parquet`
- `README.md`
- `release_manifest.json`
- `run_metadata.json`
- `config.snapshot.yaml`
- copied mapping files under `mappings/`
- `clusters/cluster_assignments.parquet`
- `clusters/cluster_statistics.json`
- `clusters/cluster_state.pkl`
- `evaluation/baseline_summary.json`, when configured

`cluster_state.pkl` is local resume/debug state and must not be uploaded.

The public HF Git payload is intentionally minimal:

- `README.md`
- `all_genres.parquet`
- `release_manifest.json`

## Validation

Inspect counts and metadata:

```bash
uv run python -c "import json, pathlib, pandas as pd; base=pathlib.Path('output/<UD_VERSION>-community-release/genres'); df=pd.read_parquet(base/'all_genres.parquet'); manifest=json.loads((base/'release_manifest.json').read_text()); print(len(df)); print(manifest['train_id']); print(manifest['artifact_key']); print(manifest['ud_source_revision'])"
```

Run release tests after source changes:

```bash
uv run pytest tests/test_release_identity.py tests/test_release_artifacts.py tests/test_release_matrix.py -q
```

Inspect the upload plan:

```bash
uv run ud-genre-bootstrap upload \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version <UD_VERSION> \
  --dry-run
```

## Source Tagging

Commit the exact source state responsible for a new train version, then create
the release branch and immutable source tag:

```bash
git branch -f release/full-ud-v1 HEAD
git tag source/full-ud-v1.0.0 HEAD
```

Do this before non-dry-run Git-backed publishing. The publish command validates
that `source/full-ud-v1.0.0` points at the current clean source commit. Once the
source tag has been pushed, do not move it; bump the train version instead.
For an already-tagged train, publish from the tagged source commit rather than
from later docs-only commits on `main`.

## Git-Backed HF Publish

Check the HF checkout first:

```bash
git -C ../ud_genre-hf status --short --branch
git -C ../ud_genre-hf remote -v
```

Inspect the publish plan:

```bash
uv run ud-genre-bootstrap publish \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version <UD_VERSION> \
  --hf-repo-dir ../ud_genre-hf \
  --dry-run
```

Publish locally into `../ud_genre-hf/`:

```bash
uv run ud-genre-bootstrap publish \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version <UD_VERSION> \
  --hf-repo-dir ../ud_genre-hf
```

Use `--include-main` only for the default UD version, currently `2.18`:

```bash
uv run ud-genre-bootstrap publish \
  --release-matrix configs/releases/full-ud-v1.0.0.yaml \
  --ud-version 2.18 \
  --hf-repo-dir ../ud_genre-hf \
  --include-main
```

After reviewing the HF checkout, push:

```bash
git -C ../ud_genre-hf push origin <UD_VERSION>
git -C ../ud_genre-hf push origin artifact/full-ud-v1.0.0/ud<UD_VERSION>
```

If publishing the default UD version, also push `main`.

Also push the source branch and source tag:

```bash
git push origin main release/full-ud-v1
git push origin source/full-ud-v1.0.0
```

## Inventory Completion

For the initial `full-ud-v1.0.0` train:

1. Generate, validate, and publish UD `2.18` with `--include-main`.
2. Generate, validate, and publish UD `2.17`.
3. Generate, validate, and publish UD `2.7` through `2.16` without
   `--include-main`.
4. Update `configs/releases/genre_artifacts.yaml` from `partial` to `complete`
   only after all supported branches and immutable HF tags are present.

Do not create per-UD release-note files. Per-version provenance belongs in the
generated manifests and the HF tags.
