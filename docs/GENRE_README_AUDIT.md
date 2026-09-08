# Genre README Audit

This audit is a preflight step before regenerating release-train artifacts. It
does not replace `coverage`; it explains where README text and sentence comments
suggest that extraction patterns or genre mappings may still be missing.

## UD 2.18 README Scale

The current local UD 2.18 README scan found 781 treebank-level genre mentions
across the 18 canonical `ud` schema labels:

| Genre | README mentions |
| --- | ---: |
| `academic` | 15 |
| `bible` | 29 |
| `blog` | 24 |
| `email` | 4 |
| `fiction` | 102 |
| `government` | 14 |
| `grammar-examples` | 76 |
| `learner-essays` | 6 |
| `legal` | 42 |
| `medical` | 10 |
| `news` | 146 |
| `nonfiction` | 122 |
| `poetry` | 18 |
| `reviews` | 10 |
| `social` | 23 |
| `spoken` | 58 |
| `web` | 18 |
| `wiki` | 64 |

These are treebank README declarations, not sentence counts. Multi-genre
treebanks contribute more than one mention.

When the release matrix is used without `--include-excluded`, the audit follows
`exclude_treebanks` from `configs/release_profiles/full-ud.yaml`; for UD 2.18
that means `ar_nyuad`, `ja_bccwj`, and `pt_cintil` are skipped. The
release-scoped UD 2.18 audit covers 350 treebanks and 771 README genre
mentions.

## Extracted Information

The JSON report is intentionally broader than the console table. It records:

- README genre lines and normalized README genres.
- Generated UD metadata genres when a local metadata JSON file is available.
- Current sentence-level extraction coverage from local CoNLL-U comments.
- Raw direct `# genre = ...` and `# newdoc genre = ...` values.
- Alias mappings already integrated by the configured mapping files.
- Raw labels that still do not normalize to a canonical `ud` schema label.
- Non-canonical labels emitted by existing extraction patterns.
- Comment-key counts, `sent_id` token counts, and scanned CoNLL-U file names.

Use the Markdown report for triage and the JSON report when checking the exact
evidence behind a candidate.

## Command

Run the audit against a local UD treebank checkout:

```bash
uv run ud-genre-bootstrap audit-readme-genres \
  --release-matrix configs/releases/full-ud-v1.0.2.yaml \
  --ud-version 2.18 \
  --ud-root ../huggingface/universal_dependencies/tools/ud-treebanks-v2.18 \
  --sort-by uncovered-sentences \
  --export output/2.18-community-release/readme_genre_audit.json \
  --markdown output/2.18-community-release/readme_genre_audit.md \
  --only-candidates
```

Add `--include-excluded` to reproduce the full README scale above instead of
the release-scoped audit.

For a quick pass, limit the sentence-comment scan:

```bash
uv run ud-genre-bootstrap audit-readme-genres \
  --release-matrix configs/releases/full-ud-v1.0.2.yaml \
  --ud-version 2.18 \
  --ud-root ../huggingface/universal_dependencies/tools/ud-treebanks-v2.18 \
  --max-sentences-per-treebank 500
```

Use `--sort-by` to choose the triage order:

- `priority`: existing severity-first order.
- `priority-sentences`: severity/action first, then larger treebanks first.
- `sentences`: larger scanned treebanks first.
- `uncovered-sentences`: larger currently-uncovered sentence counts first.

For full regeneration preflight, `uncovered-sentences` is usually the most
useful view because it brings high-impact extraction gaps to the top. With
`--max-sentences-per-treebank`, these counts refer to the scanned sample, not
the full treebank.

## Priority Meanings

- `high` / `add_mapping`: sentence comments contain direct genre labels that do
  not normalize to the configured canonical schema.
- `high` / `fix_pattern_or_mapping`: current extraction emits a non-canonical
  label.
- `high` / `review_existing_pattern`: a configured pattern exists but does not
  reach the coverage threshold.
- `medium` / `inspect_readme_hint_for_pattern`: a multi-genre treebank lacks
  enough current sentence-level coverage, but its README mentions `sent_id`,
  document IDs, file names, or source conventions that may encode genre.
- `integrated` / `covered_by_alias_mapping`: an alias is already mapped, such as
  `examples -> grammar-examples`.
- `covered`: current extraction or treebank-level single-genre metadata is
  sufficient for the release workflow.
- `low` / `manual_review`: the treebank is multi-genre but has no obvious
  machine-readable clue.

## Current Integrated Cases

The first UD 2.18 audit pass identified two low-risk fixes that are integrated
before regeneration:

- `pl_pud` now uses `sent_id` prefixes in current UD metadata and maps `n` to
  `news`, `w` to `nonfiction`.
- `ru_syntagrus` now maps direct `journalism` sentence metadata to `news`.

After these changes, `pl_pud` is no longer a candidate. `ru_syntagrus` remains
`high` / `review_existing_pattern`, because only part of the treebank carries
direct sentence-level genre comments; the fixed mapping recovers the
`journalism` portion but does not invent labels for sentences without metadata.

## Suggested Steps

1. Run the audit before the expensive `embed`, `cluster`, `label`, and
   `evaluate` sequence.
2. Integrate `high` priority mapping fixes first, because they are usually
   low-risk and easy to test.
3. Review broken existing patterns before adding new ones.
4. Add `medium` priority README-derived patterns only when the README statement,
   sentence-comment samples, and `test-genres` output agree unambiguously.
5. Re-run `coverage` and `audit-readme-genres` after every mapping or pattern
   change.
6. Defer unclear `low` priority items until after the first regenerated artifact
   and treat them as evaluation-improvement candidates rather than release
   blockers.

The audit is deliberately conservative. It surfaces possible evidence but does
not automatically create patterns from prose.
