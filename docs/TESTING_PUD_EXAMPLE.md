# Genre Extraction for PUD Treebanks

This guide shows how to extract genres from PUD (Parallel Universal Dependencies) treebanks, which encode genre information in the `sent_id` field.

## Background

PUD treebanks are multilingual parallel test sets created for the CoNLL 2017 Shared Task. They contain:
- ~1000 sentences per language
- Only **test split** (no train/dev)
- Two genres encoded in sent_id:
  - `n` = news
  - `w` = wiki

## Step 1: Examine the Data

First, let's look at how genre is encoded in PUD treebanks:

```bash
# Look at first sentences in German PUD
head -30 ../huggingface/universal_dependencies/tools/UD_repos/UD_German-PUD/de_pud-ud-test.conllu
```

**Output**:
```
# newdoc id = n01001
# sent_id = n01001011
# parallel_id = pud/n01001011
# text = „Ein Großteil des digitalen Übergangs..."

...

# newdoc id = w01001
# sent_id = w01001049
# parallel_id = pud/w01001049
# text = Admiral Erich Raeder...
```

**Observation**:
- News sentences: `# sent_id = n...`
- Wiki sentences: `# sent_id = w...`
- The first character after `sent_id = ` indicates the genre

## Step 2: Create Pattern Configuration

Create `configs/pud-patterns.json`:

```json
{
  "de_pud": [
    {
      "pattern": "# sent_id = ([nw])",
      "genre": "$1"
    }
  ]
}
```

**Pattern explanation**:
- `# sent_id = ([nw])` - Matches `# sent_id = n` or `# sent_id = w`
- `([nw])` - Capture group extracts the letter
- `"genre": "$1"` - Use captured letter as genre
- `$1` gets replaced with `n` or `w`

## Step 3: Create Genre Mapping

Create or update `configs/genre_mappings.json`:

```json
{
  "n": "news",
  "w": "wiki"
}
```

This maps the single-letter codes to canonical UD genres.

## Step 4: Create Test Configuration

Create `configs/test-pud.yaml`:

```yaml
ud_version: "2.17"
ud_source: "local://../huggingface/universal_dependencies/tools/UD_repos/"

genre_extraction:
  mapping_path: "configs/genre_mappings.json"
  patterns_path: "configs/pud-patterns.json"

# ... rest of config ...
```

## Step 5: Test Single Treebank

Test with German PUD (remember to use `--split test`!):

```bash
ud-genre-bootstrap test-genres \
    --treebank de_pud \
    --split test \
    --config configs/test-pud.yaml \
    --limit 100
```

**Expected Output**:
```
Testing: de_pud (test)

       Extraction Statistics
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric                  ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Sentences         │ 100   │
│ Sentences with Genre    │ 100   │
│ Sentences without Genre │ 0     │
│ Coverage                │ 100%  │
└─────────────────────────┴───────┘

Extracted Genres:
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Genre ┃ Count ┃ Percentage ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ news  │ 60    │ 60.0%      │
│ wiki  │ 40    │ 40.0%      │
└───────┴───────┴────────────┘

Extraction Methods:
  • pattern_match: 100

Example Matches:

Genre: news
  sent_id: n01001011
  text: „Ein Großteil des digitalen Übergangs ist für die Vereinigten...
  comments: # sent_id = n01001011

Genre: wiki
  sent_id: w01001049
  text: Admiral Erich Raeder, der während der...
  comments: # sent_id = w01001049
```

## Step 6: Apply to All PUD Treebanks

PUD exists for 19 languages. Add all of them to your patterns file:

```json
{
  "_comment": "PUD treebanks encode genre in sent_id: 'n' = news, 'w' = wiki",

  "ar_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "cs_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "de_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "en_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "es_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "fi_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "fr_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "hi_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "id_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "it_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "ja_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "ko_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "pl_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "pt_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "ru_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "sv_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "th_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "tr_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}],
  "zh_pud": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}]
}
```

## Step 7: Batch Test Multiple PUD Treebanks

Test several PUD treebanks at once:

```bash
for tb in de_pud en_pud fr_pud es_pud; do
    echo "Testing $tb..."
    ud-genre-bootstrap test-genres \
        --treebank $tb \
        --split test \
        --config configs/test-pud.yaml \
        --limit 50 \
        --no-examples
done
```

## Important Notes

### 1. Split Selection
PUD treebanks **only have test split**. Always use `--split test`:
```bash
# ✓ Correct
--split test

# ✗ Wrong (will fail)
--split train
```

### 2. Coverage Should Be 100%
If coverage is less than 100%, check:
- Pattern is correct: `"# sent_id = ([nw])"`
- Genre mapping includes both: `"n": "news", "w": "wiki"`
- Config files are loaded correctly

### 3. Genre Distribution
Typical PUD distribution:
- **News**: ~60% (600 sentences)
- **Wiki**: ~40% (400 sentences)

If you see different ratios, the pattern may be extracting incorrectly.

## Alternative Pattern: Using genre_mapping

Instead of capture groups, you can use inline genre_mapping:

```json
{
  "de_pud": [
    {
      "pattern": "# sent_id = ([nw])",
      "genre_mapping": {
        "n": "news",
        "w": "wiki"
      }
    }
  ]
}
```

This approach doesn't require a separate genre_mappings.json file for these specific patterns.

## Troubleshooting

### Problem: 0% Coverage

**Check 1**: Verify pattern matches comments
```bash
grep "^# sent_id" ../path/to/de_pud-ud-test.conllu | head -5
```

**Check 2**: Test pattern with Python
```python
import re
comment = "# sent_id = n01001011"
match = re.search(r"# sent_id = ([nw])", comment)
print(match.group(1))  # Should print: n
```

**Check 3**: Verify config files are loaded
Add debug logging to see what patterns are being used.

### Problem: Genre is "$1" instead of "news"

This means capture group substitution isn't working. Verify:
- Using latest version of ud-genre-bootstrap
- Pattern has parentheses: `([nw])` not `[nw]`
- Genre template has dollar sign: `"$1"` not `"1"`

### Problem: "Split train not found"

PUD treebanks only have test split. Use:
```bash
--split test
```

## Next Steps

1. **Test all PUD treebanks**: Verify 100% coverage across all 19 languages
2. **Add to main patterns**: Include PUD patterns in your production config
3. **Combine with other patterns**: Merge with patterns for other treebanks
4. **Run full pipeline**: Use extracted genres for bootstrapping

## See Also

- [Genre Pattern Configuration](GENRE_PATTERNS.md) - Full pattern documentation
- [Testing Example](TESTING_EXAMPLE.md) - General testing workflow
- [configs/pud-patterns.json](../configs/pud-patterns.json) - Ready-to-use PUD patterns
