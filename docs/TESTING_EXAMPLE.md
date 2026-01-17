# Genre Extraction Testing Example

This guide walks through testing and debugging genre extraction patterns for the English EWT treebank.

## Step 1: Test Without Patterns

First, let's see what the default behavior is without any custom patterns:

```bash
ud-genre-bootstrap test-genres --treebank en_ewt --limit 20
```

**Output**:
```
Testing: en_ewt (train)
Expected genres from metadata: blog, social, reviews, email, web

       Extraction Statistics
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric                  ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Sentences         │ 20    │
│ Sentences with Genre    │ 0     │
│ Sentences without Genre │ 20    │
│ Coverage                │ 0.0%  │
└─────────────────────────┴───────┘

Examples Without Genre Match:
  sent_id: weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0001
  comments: ['newdoc id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000']
```

**Observation**:
- 0% coverage means no genres are being extracted
- The comments show `newdoc id = weblog-...` pattern
- We need to create a pattern to extract "weblog" from this

## Step 2: Create Pattern Configuration

Based on the comment structure, create `configs/metadata_patterns.json`:

```json
{
  "en_ewt": [
    {
      "pattern": "newdoc id = (weblog|answers|email|reviews)-",
      "genre": "$1"
    }
  ]
}
```

And `configs/genre_mappings.json` to normalize "weblog" to "blog":

```json
{
  "weblog": "blog",
  "answers": "social"
}
```

## Step 3: Create Custom Config

Create `configs/test-ewt.yaml`:

```yaml
ud_version: "2.17"
ud_source: "hf://commul/universal_dependencies"

genre_extraction:
  mapping_path: "configs/genre_mappings.json"
  patterns_path: "configs/metadata_patterns.json"

embeddings:
  model: "xlm-roberta-base"
  device: "auto"
```

## Step 4: Test With Patterns

```bash
ud-genre-bootstrap test-genres \
    --treebank en_ewt \
    --config configs/test-ewt.yaml \
    --limit 100
```

**Expected Output**:
```
Testing: en_ewt (train)
Expected genres from metadata: blog, social, reviews, email, web

       Extraction Statistics
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric                  ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Sentences         │ 100   │
│ Sentences with Genre    │ 85    │
│ Sentences without Genre │ 15    │
│ Coverage                │ 85.0% │
└─────────────────────────┴───────┘

Extracted Genres:
┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Genre   ┃ Count ┃ Percentage ┃
┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ blog    │ 60    │ 70.6%      │
│ social  │ 20    │ 23.5%      │
│ email   │ 5     │ 5.9%       │
└─────────┴───────┴────────────┘

Extraction Methods:
  • pattern_match: 85

Example Matches:

Genre: blog
  sent_id: weblog-juancole.com_juancole_20051126063000-0001
  text: Al-Zaman : American forces killed Shaikh Abdullah al-Ani...
  comments: newdoc id = weblog-juancole.com_juancole_20051126063000

Genre: social
  sent_id: answers-20111108105724AAhFy9Q_ans-0001
  text: How many pounds did you lose on your first week of weight watchers?
  comments: newdoc id = answers-20111108105724AAhFy9Q_ans

Examples Without Genre Match:
  sent_id: train-s1234
  text: This is a sentence without newdoc comment.
  comments: [none]
```

## Step 5: Debug Unmatched Sentences

If you see sentences without matches, check their comments:

```bash
# Show more examples
ud-genre-bootstrap test-genres \
    --treebank en_ewt \
    --config configs/test-ewt.yaml \
    --limit 500 \
    --examples
```

Look at the "Examples Without Genre Match" section to understand:
- What comment format is used?
- Is it a different document type?
- Does the pattern need refinement?

## Step 6: Refine Patterns

If you find new patterns, update your configuration:

```json
{
  "en_ewt": [
    {
      "pattern": "newdoc id = (weblog|answers|email|reviews)-",
      "genre": "$1"
    },
    {
      "_description": "Handle newsgroup format",
      "pattern": "newdoc id = (newsgroup)-",
      "genre": "news"
    }
  ]
}
```

Test again:

```bash
ud-genre-bootstrap test-genres \
    --treebank en_ewt \
    --config configs/test-ewt.yaml \
    --limit 500
```

## Step 7: Test All Splits

Once your patterns work well on train, test on other splits:

```bash
# Test dev split
ud-genre-bootstrap test-genres \
    --treebank en_ewt \
    --split dev \
    --config configs/test-ewt.yaml

# Test test split
ud-genre-bootstrap test-genres \
    --treebank en_ewt \
    --split test \
    --config configs/test-ewt.yaml
```

## Debugging Tips

### Low Coverage (<50%)

**Problem**: Only half or fewer sentences get genres

**Solutions**:
1. Check unmatched examples for patterns
2. Look for variations in comment format
3. Add more pattern alternatives with `|` operator
4. Consider if some documents legitimately have no genre

### Wrong Genre Extracted

**Problem**: Pattern extracts wrong text

**Solutions**:
1. Make pattern more specific (add context)
2. Check capture group `$1` captures the right part
3. Use genre_mapping to normalize extracted values
4. Test pattern with Python regex first

### Pattern Not Matching

**Problem**: Pattern should match but doesn't

**Solutions**:
1. Copy exact comment text and test with Python `re.search()`
2. Check for hidden characters or extra spaces
3. Ensure pattern is under correct treebank code
4. Try simpler pattern first, then add specificity

### Multiple Genres Per Sentence

**Problem**: Some sentences get multiple genres

**Solutions**:
1. This is normal if different patterns match
2. System automatically deduplicates
3. Order patterns by specificity (most specific first)
4. Consider if the sentence truly has multiple genres

## Example: Testing Multiple Treebanks

Test several treebanks at once:

```bash
# Test first 10 treebanks
ud-genre-bootstrap test-genres --config configs/test-ewt.yaml

# Focus on specific treebanks
for tb in en_ewt en_gum de_gsd fr_gsd; do
    echo "Testing $tb..."
    ud-genre-bootstrap test-genres \
        --treebank $tb \
        --config configs/test-ewt.yaml \
        --limit 50 \
        --no-examples
done
```

## Best Workflow

1. **Start simple**: Test 20-50 sentences first
2. **Examine comments**: Look at unmatched examples
3. **Create pattern**: Write regex based on comments
4. **Test pattern**: Use `test-genres` to verify
5. **Refine**: Adjust pattern based on results
6. **Scale up**: Test on more sentences (100-500)
7. **Validate**: Test on all splits
8. **Document**: Add comments explaining patterns

## Common Patterns Library

### English EWT
```json
{
  "pattern": "newdoc id = (weblog|answers|email|reviews)-",
  "genre": "$1"
}
```

### English GUM
```json
{
  "pattern": "newdoc id = GUM_(blog|news|fiction|academic|interview|voyage|whow)",
  "genre": "$1"
}
```

### Finnish TDT
```json
{
  "pattern": "doc_name = (blog|news|wiki)_",
  "genre": "$1"
}
```

### Any treebank with explicit genre
```json
{
  "pattern": "genre = (.+)",
  "genre": "$1"
}
```

## See Also

- [Genre Pattern Configuration](GENRE_PATTERNS.md) - Full documentation
- [Genre Mappings Example](../configs/genre_mappings.example.json)
- [Pattern Configuration Example](../configs/metadata_patterns.example.json)
