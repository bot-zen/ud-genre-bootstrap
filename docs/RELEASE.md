# Release Process

This is the single procedural release document for UD genre artifacts. Do not
create one release note per UD version. Per-version identity belongs in release
configs, `configs/releases/genre_artifacts.yaml`, and generated release
metadata.

## Source Of Truth

- release config: `configs/<ud-version>-community-release.yaml`
- promoted artifact registry: `configs/releases/genre_artifacts.yaml`
- generated local release directory: `output/<release-name>/genres`
- HF dataset Git checkout: `../ud_genre-hf/`
- HF dataset repo: `commul/ud_genre`
- upstream UD HF dataset repo: `universal-dependencies/universal_dependencies`

For UD 2.7 through UD 2.16 backfills, add or update a release config and
registry entry, then use this document. Create a version-specific document only
for exceptional narrative context that cannot be represented in config,
registry, generated metadata, or evaluation documentation.

## Artifact Identity

Artifact IDs have the shape:

```text
ud<UD version>-<scope>-<label schema>-<artifact version>
```

Examples:

- `ud2.17-full-ud-v1.0.1`
- `ud2.18-full-ud-v1.0.1`

Field meanings:

- `ud<UD version>`: source Universal Dependencies release
- `scope`: data scope, normally `full`
- `label_schema`: genre inventory/profile, currently `ud`
- `artifact_version`: explicit semver-like artifact revision, for example
  `v1.0.0` or `v1.0.1`

Algorithm settings are not part of the public artifact ID. Embedding model,
pooling, clustering method, thresholds, reference weighting, and seed are
recorded as `algorithm_recipe` in `run_metadata.json` and
`release_manifest.json`.

## Versioning Policy

- The first public artifact for a UD version, scope, and label schema uses
  `artifact_version: v1.0.0`, even for backfilled older UD releases.
- Major or minor artifact changes are source-wide release-train milestones and
  should rebuild every active promoted artifact in the registry.
- Patch changes are artifact-specific fixes, for example a dataset-card refresh
  or metadata extraction fix for one UD line.
- Every public artifact version change gets a new immutable source tag and a new
  immutable HF artifact tag.
- Existing immutable tags must not be moved.

If generated data do not change and only the README/provenance metadata changes,
bump the artifact patch version, for example from `v1.0.0` to `v1.0.1`.

Legacy shorthand artifact IDs such as `ud2.18-full-ud-v1` may exist from early
publication work, but promoted configs should use explicit patch versions and
new shorthand artifact tags should not be created.

## Branches And Tags

Source repository:

- `main`: ongoing development
- `release/v1`: current source release train for all `v1.x.y` artifacts
- `source/<artifact-id>`: immutable source tag for the exact producing source
  state

HF dataset repository:

- `<UD version>`, for example `2.18`: moving branch for end-user loading
- `main`: moving default branch rendered in the HF web UI and used when callers
  omit `revision`
- `artifact/<artifact-id>`: immutable HF artifact tag

HF branches are published dataset views, not development branches. Do not merge
HF `2.17` into `2.18` or the reverse. Regenerate from the source repository and
publish each target branch independently.

## Current Public State

The current intended default is UD 2.18:

- default HF branch: `main`
- UD branch: `2.18`
- artifact ID: `ud2.18-full-ud-v1.0.1`
- HF tag: `artifact/ud2.18-full-ud-v1.0.1`
- source tag: `source/ud2.18-full-ud-v1.0.1`

Default status is represented by the HF `main` branch, not by a separate
registry status. Registry status values are `active`, `superseded`, and
`deprecated`.

UD 2.17 remains available as a patched metadata refresh:

- UD branch: `2.17`
- artifact ID: `ud2.17-full-ud-v1.0.1`
- HF tag: `artifact/ud2.17-full-ud-v1.0.1`
- source tag: `source/ud2.17-full-ud-v1.0.1`
- data status: label data unchanged from `ud2.17-full-ud-v1`

## Release Config Lifecycle

Start each UD release from a committed config, not from an ad hoc sweep config.
The config should pin:

- `ud_version` and `ud_source`
- release identity, HF branch/tag, and source tag
- explicit treebank exclusions
- embedding model, pooling, layer, batch size, and cache location
- clustering method, seed, and fit settings
- bootstrapping thresholds and reference weighting
- mapping and metadata-pattern files
- output directory, config name, run ID, and UD source revision

Promote an artifact by adding it to `configs/releases/genre_artifacts.yaml`.
The registry should record artifact status, change scope, HF branches/tag,
source branch/tag, source config, baseline summary, mapping files, and notes.
`source_config` is the path as it exists at the immutable `source_tag`; for
superseded artifacts, do not read it as the current worktree version of that
file.

## Release Modes

### New Current UD Release

Use this path when a new UD version should become the public default, as with
UD 2.18:

1. Create or update `configs/<ud-version>-community-release.yaml`.
2. Use `artifact_version: v1.0.0` unless an artifact for that exact
   UD/scope/schema has already been published.
3. Set `hf_branches` to the UD version branch, for example `["2.18"]`.
4. Set `hf_tag` and `source_tag` from the artifact ID.
5. Generate, validate, and commit the source state.
6. Create `release/v1` and `source/<artifact-id>` at that source commit.
7. Publish with `--include-main` so HF `main` renders the new default.

### Initial Backfills For Older UD Releases

Use this path for first-time public artifacts for UD 2.7 through UD 2.16:

1. Create one release config per UD version, for example
   `configs/2.7-community-release.yaml`.
2. Set `ud_version`, `output.genres_path`, `output.config_name`,
   `output.run_id`, and `output.ud_source_revision` to that UD version.
3. Use an initial artifact identity such as `ud2.7-full-ud-v1.0.0`.
4. Set `hf_branches` to the matching UD branch, for example `["2.7"]`.
5. Set `hf_tag: artifact/ud2.7-full-ud-v1.0.0` and
   `source_tag: source/ud2.7-full-ud-v1.0.0`.
6. Add a registry entry with `status: active` and
   `change_scope: source_milestone`.
7. Generate and validate each artifact with its own config.
8. Commit the code, config, mapping, and registry state. Generated output stays
   local.
9. Create one source tag per artifact. Multiple backfill tags may point to the
   same source commit if that commit contains the code and configs used for all
   of them.
10. Publish each UD branch without `--include-main`; the HF default should
    remain on the current default artifact unless intentionally changed.

Backfills should not get per-version release-note files. The config, registry
entry, generated `run_metadata.json`, generated `release_manifest.json`, and HF
tag provide the per-version record.

## Preflight Checks

Run coverage and focused metadata checks before expensive embedding generation:

```bash
uv run ud-genre-bootstrap coverage \
  --config configs/<ud-version>-community-release.yaml \
  --export output/<release-name>/coverage.json

uv run ud-genre-bootstrap test-genres \
  --config configs/<ud-version>-community-release.yaml \
  --treebank ru_taiga \
  --split train \
  --limit 0 \
  --no-examples

uv run ud-genre-bootstrap test-genres \
  --config configs/<ud-version>-community-release.yaml \
  --treebank be_hse \
  --split train \
  --limit 0 \
  --no-examples

uv run ud-genre-bootstrap test-genres \
  --config configs/<ud-version>-community-release.yaml \
  --treebank en_ewt \
  --split train \
  --limit 0 \
  --no-examples
```

If coverage or extraction changes, update `configs/metadata_patterns.json`,
`configs/pud-patterns.json`, or `configs/genre_mappings.json` before the full
run.

## Full Generation

Run the pipeline in resumable stages. Use shared scratch caches for large HF
inputs when available:

```bash
HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/" \
HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/" \
uv run ud-genre-bootstrap embed \
  --config configs/<ud-version>-community-release.yaml

HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/" \
HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/" \
uv run ud-genre-bootstrap cluster \
  --config configs/<ud-version>-community-release.yaml

HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/" \
HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/" \
uv run ud-genre-bootstrap label \
  --config configs/<ud-version>-community-release.yaml \
  --clusters output/<release-name>/genres/clusters
```

`label` must reuse the saved cluster directory for the promoted run. Running
`label` without `--clusters` can start a new clustering pass.

Run evaluation when the config enables it:

```bash
uv run ud-genre-bootstrap evaluate \
  --config configs/<ud-version>-community-release.yaml
```

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

## Output Schema

The community-facing Parquet file should include:

- `treebank`
- `split`
- `sent_id`
- `genre`
- `confidence`
- `method`
- `ud_version`
- `model`
- `pooling`
- `clustering_method`
- `config_name`
- `run_id`

Artifact-level details such as `ud_source_revision`, source commit, config hash,
mapping hashes, HF branch/tag names, and algorithm recipe belong in
`run_metadata.json` and `release_manifest.json`.

## Dataset Card

The generated `README.md` is the HF dataset card. It should state:

- that labels are derived and not authoritative gold annotations
- the loading command for the moving UD branch
- the immutable artifact tag
- source dataset and join key back to `universal-dependencies/universal_dependencies`
- artifact identity, source commit, source branch/tag, and HF branch/tag
- config hash, mapping-file hashes, and algorithm recipe provenance
- output column meanings
- known evaluation framing and limitations
- project repository, paper, and point of contact

For metadata-only card changes after publication, create a patch artifact and
republish the same data with a new artifact ID and HF tag.

## Validation

Before publishing, inspect counts and metadata:

```bash
uv run python -c "import json, pathlib, pandas as pd; base=pathlib.Path('output/<release-name>/genres'); df=pd.read_parquet(base/'all_genres.parquet'); manifest=json.loads((base/'release_manifest.json').read_text()); print(len(df)); print(manifest['artifact_id']); print(manifest['ud_source_revision'])"
```

Run focused release tests after source changes:

```bash
uv run pytest tests/test_release_identity.py tests/test_release_artifacts.py -q
uv run ruff check src/ud_genre_bootstrap/utils/release_artifacts.py tests/test_release_artifacts.py tests/test_release_identity.py
```

Use the compatibility upload dry-run to inspect the full API-upload file set:

```bash
uv run ud-genre-bootstrap upload \
  --config configs/<ud-version>-community-release.yaml \
  --dry-run
```

## Source Tagging

Commit the exact source state responsible for the artifact, then create or update
the release-train branch and immutable source tag:

```bash
git branch -f release/v1 HEAD
git tag source/<artifact-id> HEAD
```

Do this before non-dry-run Git-backed publishing. The publish command validates
that the configured `release.source_tag` points at the current clean source
commit.

## Git-Backed HF Publish

Check the HF checkout first:

```bash
git -C ../ud_genre-hf status --short --branch
git -C ../ud_genre-hf remote -v
```

Inspect the publish plan:

```bash
uv run ud-genre-bootstrap publish \
  --config configs/<ud-version>-community-release.yaml \
  --hf-repo-dir ../ud_genre-hf \
  --dry-run
```

Publish locally into `../ud_genre-hf/`:

```bash
uv run ud-genre-bootstrap publish \
  --config configs/<ud-version>-community-release.yaml \
  --hf-repo-dir ../ud_genre-hf
```

Use `--include-main` only for the artifact that should become the HF default:

```bash
uv run ud-genre-bootstrap publish \
  --config configs/<ud-version>-community-release.yaml \
  --hf-repo-dir ../ud_genre-hf \
  --include-main
```

After reviewing the HF checkout, push:

```bash
git -C ../ud_genre-hf push origin <ud-version> main
git -C ../ud_genre-hf push origin artifact/<artifact-id>
```

If `main` is not moving for this artifact, omit `main` from the branch push.

Also push the source branch and source tag:

```bash
git push origin main release/v1
git push origin source/<artifact-id>
```

## Promotion Checklist

- Source repo is clean and committed.
- Release config has the intended artifact ID and tags.
- Local release directory exists and contains `all_genres.parquet`.
- `README.md`, `release_manifest.json`, and `run_metadata.json` were regenerated.
- Counts, join key, and output schema were inspected.
- Focused release tests pass.
- Source tag points at the current clean source commit.
- HF checkout is clean before publish.
- HF moving branch and immutable artifact tag were created locally.
- HF `main` moved only if the artifact is the default.
- Registry entry records the promoted artifact.
