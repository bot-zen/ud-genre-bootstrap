# UD Genre Backfills (UD 2.7 to 2.16)

This runbook is for generating initial public artifacts for UD versions 2.7
through 2.16 using Git-backed publishing (minimal payload). The general release
process is in `docs/RELEASE.md`.

Backfills should not move the HF default branch (`main`). Only the current
default UD version should move `main`.

## 10-Point Backfill Checklist (Tracked)

1. Done: Push immutable `artifact/<artifact-id>` tags for the already-promoted artifacts.
2. Done: Commit backfill configs + this runbook to `ud-genre-bootstrap` (this repo).
3. Pending: For each UD version (2.7..2.16), create + push one immutable source tag in this repo:
   - `source/ud<ud_version>-full-ud-v1.0.0`
4. Pending: For each UD version (2.7..2.16), run `embed` locally.
5. Pending: For each UD version (2.7..2.16), run `cluster` locally.
6. Pending: For each UD version (2.7..2.16), run `label --clusters ...` locally.
7. Pending: For each UD version (2.7..2.16), run `evaluate` locally (store outputs in the version output dir).
8. Pending: For each UD version (2.7..2.16), run a `upload --dry-run` sanity check (no network) to verify the HF payload files are present.
9. Pending: For each UD version (2.7..2.16), publish to the HF Git checkout:
   - Move only the UD version branch (e.g. `2.7`) and create HF tag `artifact/<artifact-id>`.
   - Do not move `main`.
10. Pending: Update the promoted artifact registry after each publish.

## Prerequisites

- A working `uv` environment that can run `ud-genre-bootstrap`.
- Sufficient disk for embeddings and intermediate artifacts.
- Optional but recommended: shared HF caches:
  - `HF_DATASETS_CACHE`
  - `HF_HUB_CACHE`

## Configs

Backfill configs live at:

- `configs/2.7-community-release.yaml`
- `configs/2.8-community-release.yaml`
- `configs/2.9-community-release.yaml`
- `configs/2.10-community-release.yaml`
- `configs/2.11-community-release.yaml`
- `configs/2.12-community-release.yaml`
- `configs/2.13-community-release.yaml`
- `configs/2.14-community-release.yaml`
- `configs/2.15-community-release.yaml`
- `configs/2.16-community-release.yaml`

Each config writes to `output/<ud-version>-community-release/genres`.

## Per-Version Pipeline Commands

Run the pipeline in stages and reuse the saved cluster directory for labeling.
The examples below use UD 2.7; substitute the config path for other versions.

```bash
export HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/"
export HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/"

uv run ud-genre-bootstrap embed --config configs/2.7-community-release.yaml
uv run ud-genre-bootstrap cluster --config configs/2.7-community-release.yaml
uv run ud-genre-bootstrap label --config configs/2.7-community-release.yaml --clusters output/2.7-community-release/genres/clusters
uv run ud-genre-bootstrap evaluate --config configs/2.7-community-release.yaml
```

## Quick Validation (Per Version)

After `label`, verify the release directory contains `all_genres.parquet` and
regenerate release metadata:

```bash
uv run ud-genre-bootstrap upload --config configs/2.7-community-release.yaml --dry-run
```

Also sanity check the manifest:

```bash
uv run python -c "import json, pathlib; p=pathlib.Path('output/2.7-community-release/genres/release_manifest.json'); print(json.loads(p.read_text())['artifact_id'])"
```

## Batch Execution

To run all versions sequentially, execute the four commands above for each config
from 2.7 to 2.16. Prefer running one version end-to-end before starting the
next so failures are isolated.

## Publishing Notes

Publishing (Git-backed) should be done later after validation:

- Publish each backfill UD branch without `--include-main`.
- Create one immutable `source/<artifact-id>` tag per backfill artifact at the
  source commit used for that run.
- Create one immutable HF `artifact/<artifact-id>` tag per backfill artifact.
