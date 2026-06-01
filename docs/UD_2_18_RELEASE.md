# UD 2.18 Release

This document records the current UD 2.18 genre artifact and the publish path to
the Hugging Face dataset repository. UD 2.18 is the intended default artifact for
the HF `main` branch.

## Identity

- UD source revision: `2.18`
- Artifact ID: `ud2.18-full-ud-v1`
- Scope: `full`
- Label schema: `ud`
- Artifact version: `v1`
- Source config: `configs/2.18-community-release.yaml`
- Local release directory: `output/2.18-community-release/genres`
- HF dataset repo: `commul/ud_genre`
- Local HF Git checkout: `../ud_genre-hf/`
- HF moving branch: `2.18`
- HF default branch: `main` (points to this artifact)
- HF artifact tag: `artifact/ud2.18-full-ud-v1`
- Source branch: `release/v1`
- Source tag: `source/ud2.18-full-ud-v1`

`../ud_genre-hf/` is the local Git checkout that maps directly to the Hugging
Face repository:

```bash
git -C ../ud_genre-hf remote -v
# origin  git@hf.co:datasets/commul/ud_genre (fetch)
# origin  git@hf.co:datasets/commul/ud_genre (push)
```

Use this checkout for Git-backed publishing. Do not treat it as a scratch output
directory; it is the working copy whose branches and tags are pushed to the HF
dataset repository.

## Generated State

The local release artifacts have been generated in
`output/2.18-community-release/genres`.

Observed release output:

- `all_genres.parquet`: `2,221,815` labeled rows
- `README.md`: generated dataset card with YAML metadata
- `release_manifest.json`: compact artifact identity and provenance
- `run_metadata.json`: expanded provenance, config hashes, mapping hashes, and
  algorithm recipe
- `config.snapshot.yaml`: frozen config copy
- `mappings/`: copied mapping and metadata-pattern files
- `clusters/cluster_assignments.parquet`: public cluster assignments
- `clusters/cluster_statistics.json`: public cluster summary
- `clusters/cluster_state.pkl`: local resume/debug state only

`cluster_state.pkl` is intentionally not part of the public HF payload.

The generated parquet columns are:

```text
treebank, split, sent_id, genre, confidence, method, ud_version, model,
pooling, clustering_method, config_name, run_id
```

Artifact-level source details such as `ud_source_revision`, config hashes, mapping
hashes, HF branch/tag names, and source tag names live in `run_metadata.json` and
`release_manifest.json`.

## Generation Commands

The embeddings were generated with the shared scratch/cache locations:

```bash
HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/" \
HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/" \
uv run ud-genre-bootstrap embed --config configs/2.18-community-release.yaml
```

Clustering used the same cache locations:

```bash
HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/" \
HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/" \
uv run ud-genre-bootstrap cluster --config configs/2.18-community-release.yaml
```

Labeling must reuse the saved cluster state. Running `label` without `--clusters`
starts a new clustering pass.

```bash
HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/" \
HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/" \
uv run ud-genre-bootstrap label \
  --config configs/2.18-community-release.yaml \
  --clusters output/2.18-community-release/genres/clusters
```

## Pre-Publish Checks

Inspect the generated release metadata:

```bash
uv run python -c "import json, pathlib, pandas as pd; base=pathlib.Path('output/2.18-community-release/genres'); df=pd.read_parquet(base/'all_genres.parquet'); manifest=json.loads((base/'release_manifest.json').read_text()); print(len(df)); print(manifest['artifact_id']); print(manifest['ud_source_revision'])"
```

Run the compatibility upload dry-run to inspect the public file set:

```bash
uv run ud-genre-bootstrap upload \
  --config configs/2.18-community-release.yaml \
  --dry-run
```

The dry-run should include public release files and cluster summaries, but not
`cluster_state.pkl`.

## Git-Backed Publish

Publishing through Git is the preferred path for this release because it updates
the HF dataset repository as an ordinary Git repository, creates the immutable HF
artifact tag, and can also move the default `main` branch for the web UI.

Check the HF checkout first:

```bash
git -C ../ud_genre-hf status --short --branch
git -C ../ud_genre-hf remote -v
```

Inspect the publish plan:

```bash
uv run ud-genre-bootstrap publish \
  --config configs/2.18-community-release.yaml \
  --hf-repo-dir ../ud_genre-hf \
  --include-main \
  --dry-run
```

Publish locally into `../ud_genre-hf/`:

```bash
uv run ud-genre-bootstrap publish \
  --config configs/2.18-community-release.yaml \
  --hf-repo-dir ../ud_genre-hf \
  --include-main
```

Push after reviewing the HF checkout:

```bash
uv run ud-genre-bootstrap publish \
  --config configs/2.18-community-release.yaml \
  --hf-repo-dir ../ud_genre-hf \
  --include-main \
  --push
```

`--include-main` is required when this release should become the default shown in
the HF web UI and the default used by clients that do not pass a revision.

## Promotion Checklist

- Confirm this source repository has the intended code, config, and release docs.
- Create `source/ud2.18-full-ud-v1` at the source state responsible for the
  artifact.
- Publish through `../ud_genre-hf/`.
- Verify HF branch `2.18`, HF tag `artifact/ud2.18-full-ud-v1`, and default
  branch `main` when `--include-main` is used.
- Add `ud2.18-full-ud-v1` to `configs/releases/genre_artifacts.yaml` only after
  the artifact has been promoted.
