# Genre Pattern Configuration

This document describes how to configure pattern-based genre extraction for treebanks where sentence-level genre information is not explicitly available in standard metadata fields.

## Overview

The genre extraction system supports three methods for identifying sentence genres:

1. **Direct metadata field**: If a sentence has a `genre` field in its metadata (automatic)
2. **Standard CoNLL-U comments**: (automatic, no patterns needed)
   - `# newdoc genre = news`
   - `# genre = news`
3. **Pattern-based extraction**: Custom patterns for treebank-specific metadata formats

### What's Automatic?

The following formats are **automatically extracted** without any pattern configuration:

```python
# Direct genre field in sentence
sentence = {"genre": "news"}  # ✓ Automatic

# Standard UD format
"# newdoc genre = news"        # ✓ Automatic

# Alternative format (without newdoc)
"# genre = blog"               # ✓ Automatic
```

### When Do You Need Patterns?

Use pattern-based extraction when genre information is encoded in:
- Document IDs: `# newdoc id = weblog-example-001`
- Sentence IDs: `# sent_id = n01001011` (PUD treebanks)
- File paths: `# sent_id = news/article001`
- Source fields: `# source = wikipedia`
- Other non-standard locations

### When Do You Need Mappings?

Use genre mappings when extracted values differ from canonical UD genres:
- **Global mappings**: `"weblog" → "blog"`, `"newspaper" → "news"`
- **Treebank-specific mappings**: Override behavior for specific treebanks

### Treebank-Specific Mappings Without Patterns

You can use treebank-specific mappings **without defining any patterns**. This is useful when a treebank uses standard genre metadata (`# genre = ...`) but interprets genre values differently.

**Example**: The `web` genre means different things in different treebanks:

```json
{
  "weblog": "blog",         // Global: normalize non-standard terms

  "de_gsd:web": "blog",     // de_gsd: 'web' means 'blog'
  "fr_gsd:web": "web",      // fr_gsd: 'web' means generic web (keep as-is)
  "en_gum:web": "nonfiction" // en_gum: 'web' means 'nonfiction'
}
```

**Usage**: Just add to `genre_mappings.json`, no patterns needed!

```yaml
# config.yaml
genre_extraction:
  mapping_path: "configs/genre_mappings.json"
  # patterns_path not needed for this!
```

**Behavior**:
```python
# All treebanks have: # genre = web

de_gsd: "web" → "blog"        # Override canonical genre
fr_gsd: "web" → "web"         # Keep canonical genre
en_ewt: "web" → "web"         # No override, use canonical
```

**Priority order**:
1. **Treebank-specific mapping** (highest priority, can override canonical)
2. **Global mapping**
3. **Canonical genre** (no mapping needed)

This document focuses on pattern-based extraction (method 3).

## When to Use Pattern-Based Extraction

Use pattern-based extraction when:
- A treebank has genre information in non-standard metadata fields
- Genre information is encoded in document IDs or file paths
- Genre needs to be inferred from specific comment patterns
- Multiple treebanks use different metadata conventions

## Configuration Files

### Genre Mapping File

Maps non-standard genre labels to canonical UD genres.

**Location**: Specified via `genre_mapping_path` in config or passed to `GenreMapper`

**Format**: JSON dictionary

```json
{
  "newspaper": "news",
  "blog-post": "blog",
  "scientific": "academic",
  "en_ewt:weblog": "blog",
  "de_gsd:wikipedia": "wiki"
}
```

**Keys**:
- Simple string: Global mapping (e.g., `"newspaper" -> "news"`)
- Treebank-prefixed: Treebank-specific mapping (e.g., `"en_ewt:weblog" -> "blog"`)

### Metadata Patterns File

Defines extraction patterns for sentence-level metadata.

**Location**: Specified via `metadata_patterns_path` in config or passed to `GenreMapper`

**Format**: JSON dictionary with treebank codes as keys

```json
{
  "en_ewt": [
    {
      "pattern": "# newdoc id = (weblog|answers|email|reviews)-",
      "genre": "$1"
    },
    {
      "pattern": "# source = (.+)",
      "genre_mapping": {
        "web": "blog",
        "email": "email",
        "answers": "social"
      }
    }
  ],
  "de_gsd": [
    {
      "pattern": "# genre = (.+)",
      "genre": "$1"
    }
  ],
  "fi_tdt": [
    {
      "pattern": "# doc_name = (.+?)_",
      "genre": "$1"
    }
  ]
}
```

## Pattern Specification

### Basic Pattern Object

```json
{
  "pattern": "regex pattern",
  "genre": "genre_label or $1 for capture group"
}
```

**Fields**:
- `pattern` (required): Regular expression to match against sentence comments
- `genre` (optional): Static genre label or `$1`, `$2`, etc. to use regex capture groups

### Advanced Pattern Object

```json
{
  "pattern": "regex pattern",
  "genre_mapping": {
    "captured_value_1": "canonical_genre_1",
    "captured_value_2": "canonical_genre_2"
  }
}
```

**Fields**:
- `pattern` (required): Regular expression with capture group
- `genre_mapping` (optional): Dictionary mapping captured values to genres

**Inline vs Global Mappings**:

The inline `genre_mapping` can be partial. Values not in the inline mapping will fall back to global genre mappings:

```json
// Pattern file
{
  "cs_pdt": [
    {
      "pattern": "# source = (.+)",
      "genre_mapping": {
        "news": "news",
        "magazine": "news"
        // Other values fall back to global mappings
      }
    }
  ]
}

// Global mappings file
{
  "blog": "blog",
  "weblog": "blog"
}
```

**Behavior**:
- `# source = news` → inline mapping → `news` ✓
- `# source = blog` → not in inline, uses global → `blog` ✓
- `# source = weblog` → not in inline, global normalizes → `blog` ✓

This allows you to:
1. Use inline mappings for treebank-specific overrides
2. Use global mappings for common normalizations
3. Combine both approaches flexibly

### Combining Multiple Capture Groups

You can combine multiple capture groups to extract non-contiguous parts:

```json
{
  "pattern": "# newdoc id = ([a-z])\\d+([a-z])",
  "genre": "$1$2"
}
```

**Example**: Czech CAC treebank uses document IDs like `a01w`, `b12s`
- First letter indicates corpus section
- Last letter indicates genre
- Combined: `aw` → needs mapping to canonical genre

```json
{
  "cs_cac": [
    {
      "pattern": "# newdoc id = ([a-z])\\d+([a-z])",
      "genre": "$1$2"
    }
  ]
}
```

With mapping:
```json
{
  "aw": "news",
  "as": "news",
  "bw": "nonfiction",
  "bs": "nonfiction"
}
```

Result:
- `# newdoc id = a01w` → extracts `aw` → maps to `news` ✓
- `# newdoc id = b12s` → extracts `bs` → maps to `nonfiction` ✓

## Examples

### Example 1: English EWT Treebank

The English EWT treebank encodes genre in document IDs:

```
# newdoc id = weblog-blogspot.com_escapethecity_20040916151919
# newdoc id = answers-20111108101712AAVqhJj_ans
# newdoc id = email-enronsent38_02-0080
```

**Pattern configuration**:

```json
{
  "en_ewt": [
    {
      "pattern": "# newdoc id = (weblog|answers|email|reviews)-",
      "genre": "$1"
    }
  ]
}
```

**Genre mapping** (if needed):

```json
{
  "weblog": "blog",
  "answers": "social",
  "reviews": "reviews"
}
```

### Example 2: Finnish TDT Treebank

Finnish TDT uses document names with genre prefixes:

```
# doc_name = blog_2015_01_01
# doc_name = news_2015_02_15
# doc_name = wiki_article_123
```

**Pattern configuration**:

```json
{
  "fi_tdt": [
    {
      "pattern": "# doc_name = (blog|news|wiki)_",
      "genre": "$1"
    }
  ]
}
```

### Example 3: Multiple Patterns with Mapping

For treebanks with complex genre indicators:

```json
{
  "xx_example": [
    {
      "pattern": "# source_type = (.+)",
      "genre_mapping": {
        "web_blog": "blog",
        "web_wiki": "wiki",
        "written_news": "news",
        "written_academic": "academic",
        "spoken_conversation": "spoken"
      }
    },
    {
      "pattern": "# genre_explicit = (.+)",
      "genre": "$1"
    }
  ]
}
```

**Processing order**:
1. First pattern is tried
2. If no match, second pattern is tried
3. First match wins

### Example 4: Fallback to Comment Fields

For treebanks with direct genre comments:

```json
{
  "de_gsd": [
    {
      "pattern": "# genre = (.+)",
      "genre": "$1"
    }
  ]
}
```

This extracts the genre directly from comments like:
```
# genre = news
# genre = wiki
```

## Configuration Integration

### In YAML Config

**Single pattern file:**
```yaml
genre_extraction:
  mapping_path: "configs/genre_mappings.json"
  patterns_path: "configs/metadata_patterns.json"
```

**Multiple pattern files (patterns are merged):**
```yaml
genre_extraction:
  mapping_path: "configs/genre_mappings.json"
  patterns_path:
    - "configs/metadata_patterns.json"  # Base patterns
    - "configs/pud-patterns.json"       # PUD-specific patterns
```

When using multiple files:
- Patterns are loaded in order and merged
- For the same treebank code, patterns from all files are combined
- This allows modular pattern organization (e.g., separate files for PUD, GUM, etc.)

### In Python Code

**Single pattern file:**
```python
from ud_genre_bootstrap.utils.genre_mapping import GenreMapper
from pathlib import Path

# Initialize with single pattern file
mapper = GenreMapper(
    genre_mapping_path=Path("configs/genre_mappings.json"),
    metadata_patterns_path=Path("configs/metadata_patterns.json")
)
```

**Multiple pattern files:**
```python
from ud_genre_bootstrap.utils.genre_mapping import GenreMapper
from pathlib import Path

# Initialize with multiple pattern files
mapper = GenreMapper(
    genre_mapping_path=Path("configs/genre_mappings.json"),
    metadata_patterns_path=[
        Path("configs/metadata_patterns.json"),
        Path("configs/pud-patterns.json")
    ]
)
```

**Extract genres from sentence:**
```python
sentence = {
    'sent_id': 'en_ewt-ud-train#1',
    'comments': [
        '# newdoc id = weblog-blogspot.com_example',
        '# sent_id = weblog-blogspot.com_example-0001'
    ]
}

genres = mapper.extract_genres_from_metadata(sentence, 'en_ewt')
print(genres)  # ['blog']
```

## Canonical UD Genres

All extracted genres should map to these canonical UD genre labels:

- `academic` - Academic writing, scientific papers
- `blog` - Blog posts, web logs
- `email` - Email messages
- `fiction` - Fiction literature
- `government` - Government documents
- `grammar-examples` - Constructed grammar examples
- `legal` - Legal documents
- `medical` - Medical texts
- `news` - News articles
- `nonfiction` - Non-fiction prose
- `reviews` - Product/service reviews
- `social` - Social media
- `spoken` - Transcribed speech
- `web` - General web content
- `wiki` - Wikipedia articles

## Testing Patterns

### CLI Testing Tool

The easiest way to test patterns is using the built-in CLI command:

```bash
# Test a specific treebank
ud-genre-bootstrap test-genres --treebank en_ewt --config configs/default.yaml

# Test with custom patterns
ud-genre-bootstrap test-genres \
    --treebank en_ewt \
    --config configs/custom.yaml \
    --limit 100 \
    --split train

# Test multiple treebanks (first 10)
ud-genre-bootstrap test-genres --config configs/default.yaml

# Test without showing examples
ud-genre-bootstrap test-genres --treebank en_gum --no-examples
```

**Output includes**:
- Coverage statistics (% of sentences with genres)
- Genre distribution
- Extraction methods used (direct field, standard comment, pattern match)
- Example matched sentences
- Example unmatched sentences

**Example output**:
```
Testing: en_ewt (train)
Expected genres from metadata: weblog, answers, email, reviews

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                    ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Total Sentences           │ 100    │
│ Sentences with Genre      │ 85     │
│ Sentences without Genre   │ 15     │
│ Coverage                  │ 85.0%  │
└───────────────────────────┴────────┘

Extracted Genres:
┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Genre   ┃ Count ┃ Percentage ┃
┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ blog    │ 45    │ 52.9%      │
│ social  │ 25    │ 29.4%      │
│ email   │ 10    │ 11.8%      │
│ reviews │ 5     │ 5.9%       │
└─────────┴───────┴────────────┘

Extraction Methods:
  • pattern_match: 85
```

### Test Pattern Matching (Python)

```python
import re

pattern = r"# newdoc id = (weblog|answers|email)-"
test_comments = [
    "# newdoc id = weblog-example",
    "# newdoc id = answers-test",
    "# newdoc id = fiction-novel",  # No match
]

for comment in test_comments:
    match = re.search(pattern, comment)
    if match:
        print(f"Matched: {comment} -> genre: {match.group(1)}")
    else:
        print(f"No match: {comment}")
```

**Output**:
```
Matched: # newdoc id = weblog-example -> genre: weblog
Matched: # newdoc id = answers-test -> genre: answers
No match: # newdoc id = fiction-novel
```

### Test Full Extraction (Python)

```python
from pathlib import Path
from ud_genre_bootstrap.utils.genre_mapping import GenreMapper

mapper = GenreMapper(
    genre_mapping_path=Path("genre_mappings.json"),
    metadata_patterns_path=Path("metadata_patterns.json")
)

# Test sentence
test_sentence = {
    'sent_id': 'test-001',
    'comments': [
        '# newdoc id = weblog-example-001',
        '# sent_id = weblog-example-001-0001'
    ]
}

genres = mapper.extract_genres_from_metadata(test_sentence, 'en_ewt')
assert 'blog' in genres, f"Expected 'blog', got {genres}"
```

## Best Practices

1. **Start with existing patterns**: Check if your treebank uses standard UD genre comments
2. **Use specific patterns**: Match exact formats to avoid false positives
3. **Test thoroughly**: Verify patterns on sample data before full runs
4. **Document conventions**: Add comments explaining treebank-specific patterns
5. **Use capture groups**: Extract dynamic values with `$1`, `$2` instead of hardcoding
6. **Normalize consistently**: Always map to canonical UD genres
7. **Handle ambiguity**: Define priority order when multiple patterns could match

## Troubleshooting

### Pattern Not Matching

**Problem**: Pattern doesn't extract genres from sentences

**Solutions**:
1. Check regex syntax: Test pattern with sample comments
2. Verify comment format: Print actual sentence comments
3. Check treebank code: Ensure pattern file uses correct treebank ID
4. Enable debug logging: Set log level to DEBUG to see extraction attempts

```python
import logging
logging.getLogger('ud_genre_bootstrap.utils.genre_mapping').setLevel(logging.DEBUG)
```

### Wrong Genre Extracted

**Problem**: Extracted genre doesn't match expected value

**Solutions**:
1. Check capture groups: Verify `$1` captures correct text
2. Update genre mapping: Add mapping from extracted value to canonical genre
3. Refine pattern: Make regex more specific to target correct text

### Multiple Genres Extracted

**Problem**: Sentence assigned to multiple genres

**Solutions**:
1. Order patterns by specificity: More specific patterns first
2. Use single capture group: Extract one value per pattern
3. Post-process duplicates: The system automatically deduplicates

## Reference: Complete Example Configuration

### genre_mappings.json

```json
{
  "weblog": "blog",
  "weblogs": "blog",
  "blog-post": "blog",
  "answers": "social",
  "qa": "social",
  "reviews": "reviews",
  "review": "reviews",
  "newspaper": "news",
  "newsgroup": "news",
  "wikipedia": "wiki",
  "wiki-article": "wiki",
  "scientific": "academic",
  "academic-paper": "academic",
  "legal-doc": "legal",
  "spoken-conv": "spoken",
  "conversation": "spoken",
  "fiction-novel": "fiction",
  "government-doc": "government"
}
```

### metadata_patterns.json

```json
{
  "en_ewt": [
    {
      "pattern": "# newdoc id = (weblog|answers|email|reviews)-",
      "genre": "$1"
    }
  ],
  "en_gum": [
    {
      "pattern": "# newdoc id = GUM_(blog|news|fiction|academic|interview|voyage|whow)",
      "genre": "$1"
    }
  ],
  "fi_tdt": [
    {
      "pattern": "# doc_name = (blog|news|wiki)_",
      "genre": "$1"
    }
  ],
  "de_gsd": [
    {
      "pattern": "# genre = (.+)",
      "genre": "$1"
    }
  ],
  "cs_pdt": [
    {
      "pattern": "# source = (.+)",
      "genre_mapping": {
        "news": "news",
        "magazine": "news",
        "business": "news",
        "culture": "news",
        "sport": "news"
      }
    }
  ]
}
```

## See Also

- [GenreMapper API Reference](../src/ud_genre_bootstrap/utils/genre_mapping.py)
- [Configuration Guide](../README.md#configuration)
- [UD Genre Documentation](https://universaldependencies.org/format.html#genre)
