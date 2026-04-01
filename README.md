# Categorization Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Generic framework for classifying conversations and messages using LLMs (stream or batch mode).

## Overview

4-step pipeline: extract from BigQuery → preprocess → label with one or more classifiers → `combined_labeled.csv`.

**Key design principles:**
- **Extraction and classification are decoupled.** `CONVERSATIONS` is an extraction context, not a classifier. Run it once, stack multiple classifiers on the same dataset without re-extracting.
- **Steps autodiscover each other** via `--context_name` + `--workflow_names`. No `--experiment_id` needed.
- **Two labeling modes.** `stream` (immediate, concurrent API calls) or `batch` (50% cheaper, 2-24h wait via OpenAI Batch API).
- **Adding a classifier = one dict entry** in `classifiers.py`.

---

## Available Classifiers

**Conversation-level** (classify full user+agent transcripts from the `CONVERSATIONS` context):

| Classifier | Output column | Labels |
|---|---|---|
| `CONVERSATION_SENTIMENT` | `sentiment` | positive, negative, neutral, mixed |
| `CONVERSATION_ESCALATION` | `escalated_to_human` | escalated, not_escalated |

**Message-level** (classify individual user messages):

| Classifier | Output column | Labels |
|---|---|---|
| `SENTIMENT` | `sentiment` | positive, negative, neutral, mixed |
| `ESCALATION` | `escalation_priority` | urgent, high, medium, low |
| `FEEDBACK` | `feedback_type` | bug_report, feature_request, improvement, praise, question |

**Conversation-level (generic — groups raw messages with Python timeout logic):**

| Classifier | Output column | Labels |
|---|---|---|
| `CONVERSATION_RESOLUTION` | `resolution_status` | resolved, partially_resolved, unresolved, unclear |
| `CONVERSATION_SATISFACTION` | `satisfaction_level` | satisfied, neutral, dissatisfied, escalated |
| `CONVERSATION_INTENT` | `conversation_intent` | technical_support, billing_and_payments, account_management, product_information, complaint, general_inquiry |

---

## Installation

```bash
uv pip install -e .
# or: pip install -e .

cp env.template .env
# fill in OPENAI_API_KEY, GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS
```

---

## Quick Start

Full end-to-end example: 50 conversations per workflow, two classifiers stacked on the same dataset.

```bash
# Step 1 — extract conversations from BigQuery (50 per workflow)
categorization make_context \
  --context_name CONVERSATIONS \
  --workflow_names "azteca,palaciodehierro-ng,wa-al2318-alkosto-qa,wa-cc1661-ccu-cl-b2b" \
  --n_samples 50 \
  --start_date 2026-01-01 \
  --end_date 2026-12-31

# Step 2 — preprocess (dedup, length filter)
categorization make_preprocess \
  --context_name CONVERSATIONS \
  --workflow_names "azteca,palaciodehierro-ng,wa-al2318-alkosto-qa,wa-cc1661-ccu-cl-b2b"

# Step 3 — label with sentiment classifier (immediate, stream mode)
categorization make_label \
  --context_name CONVERSATIONS \
  --classifier CONVERSATION_SENTIMENT \
  --workflow_names "azteca,palaciodehierro-ng,wa-al2318-alkosto-qa,wa-cc1661-ccu-cl-b2b"

# Step 3 — stack escalation classifier on the same dataset (no re-extraction)
categorization make_label \
  --context_name CONVERSATIONS \
  --classifier CONVERSATION_ESCALATION \
  --workflow_names "azteca,palaciodehierro-ng,wa-al2318-alkosto-qa,wa-cc1661-ccu-cl-b2b"
```

Output: `experiments/conversations_{timestamp}/combined_labeled.csv` with both `sentiment` and `escalated_to_human` columns.

### Batch mode (50% cheaper, 2-24h wait)

```bash
categorization make_label \
  --context_name CONVERSATIONS \
  --classifier CONVERSATION_SENTIMENT \
  --mode batch \
  --workflow_names "azteca,palaciodehierro-ng"

# Check status and download when ready
categorization make_process_batches \
  --context_name CONVERSATIONS \
  --workflow_names "azteca,palaciodehierro-ng"
```

---

## CLI Reference

### `make_context` — Step 1: Extract from BigQuery

```bash
categorization make_context \
  --context_name CONVERSATIONS \       # extraction context (CONVERSATIONS or classifier name)
  --workflow_names "bot-a,bot-b" \     # comma-separated; omit for all workflows
  --n_samples 1000 \                   # max rows per workflow (default: 1000)
  --min_length 6 \                     # min character length (default: 6)
  --start_date 2026-01-01 \
  --end_date 2026-12-31
```

Creates: `experiments/conversations_{timestamp}/`

---

### `make_preprocess` — Step 2: Clean and deduplicate

```bash
categorization make_preprocess \
  --context_name CONVERSATIONS \
  --workflow_names "bot-a,bot-b"       # must match what was used in make_context
```

Removes nulls, filters by `min_length`, deduplicates within each workflow. For conversation-unit contexts, groups raw messages into conversations using a timeout-based inactivity window.

---

### `make_label` — Step 3: Label with a classifier

```bash
categorization make_label \
  --context_name CONVERSATIONS \
  --classifier CONVERSATION_SENTIMENT \
  --workflow_names "bot-a,bot-b" \
  --mode stream \                      # stream (default) or batch
  --max_workers 20 \                   # concurrent threads for stream mode (default: 20)
  --dry_run                            # print cost estimate and exit
```

- `--mode stream` — concurrent Chat Completions, results immediately
- `--mode batch` — OpenAI Batch API, 50% cheaper, 2-24h wait
- Runs can be stacked: each invocation adds one label column to `combined_labeled.csv`

Cost is always logged before execution. `--dry_run` exits after printing the estimate.

---

### `make_process_batches` — Step 4: Download batch results (batch mode only)

```bash
categorization make_process_batches \
  --context_name CONVERSATIONS \
  --workflow_names "bot-a,bot-b"
```

Polls the batch status. Prints progress if still running, downloads and processes when complete.

---

### Utility commands

```bash
categorization status --context_name CONVERSATIONS
categorization list --context_name CONVERSATIONS
categorization version
```

---

## Output

Each experiment lives in `experiments/{context}_{timestamp}/`:

```
experiments/conversations_20260330_170524/
├── metadata.json                           # experiment state + parameters
├── conversations_context.parquet           # Step 1: raw extracted data
├── conversations_preprocessed.parquet      # Step 2: cleaned data
├── conversation_sentiment_labeled.parquet  # Step 3: sentiment labels
├── conversation_escalation_labeled.parquet # Step 3: escalation labels
└── combined_labeled.csv                    # merged output — one row per conversation
```

The `combined_labeled.csv` always has all columns from preprocessing + one column per classifier run.

**Schema (CONVERSATIONS context):**

| Column | Description |
|---|---|
| `message_id` | conversation ID (from BigQuery `conversation_id`) |
| `message_text` | full conversation transcript (`user:` / `flow:` lines) |
| `workflow_id` | workflow identifier |
| `workflow_name` | bot/workflow display name |
| `conversation_start` | timestamp of first message |
| `sentiment` | from CONVERSATION_SENTIMENT |
| `escalated_to_human` | from CONVERSATION_ESCALATION |

---

## Adding a Classifier

Add one entry to `CLASSIFIERS` in [src/categorization/pipelines/classifiers.py](src/categorization/pipelines/classifiers.py):

```python
"MY_CLASSIFIER": {
    "context_name": "MY_CLASSIFIER",
    "system_prompt": "You are an expert classifier. Respond ONLY with the category name.",
    "user_prompt_template": """Classify the following into one of these categories:
- category_a: description
- category_b: description

Respond with ONLY the category name, nothing else.

{message}""",
    "output_column": "my_label",
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_tokens": 15,
},
```

Then run:

```bash
categorization make_label \
  --context_name CONVERSATIONS \
  --classifier MY_CLASSIFIER \
  --workflow_names "bot-a,bot-b"
```

The system auto-generates file names, handles retries, logs cost, and merges into `combined_labeled.csv`.

**Optional fields:**

| Field | Default | Description |
|---|---|---|
| `model` | `gpt-4o-mini` | OpenAI model |
| `temperature` | `0` | sampling temperature |
| `max_tokens` | `10` | max response tokens |
| `task_description` | classifier name | OpenAI batch job metadata |

---

## Adding an Extraction Context

To add a new data source (not a classifier), add an entry to `EXTRACTION_CONTEXTS`:

```python
EXTRACTION_CONTEXTS = {
    "CONVERSATIONS": {
        "sql_query_path": "sql/context/conversation_context.sql",
        "unit": "message",
    },
    "MY_SOURCE": {
        "sql_query_path": "sql/context/my_source.sql",
        "unit": "message",
    },
}
```

Then use `--context_name MY_SOURCE` in `make_context`.

---

## Development

```bash
# Install dev deps
uv sync --group dev

# Lint + format
uv run ruff check src/
uv run ruff format src/

# Type check
uv run python -m mypy src/categorization

# Tests (151 tests)
uv run python -m pytest tests/ -q
```

### Project structure

```
src/categorization/
├── clients/          # BigQuery + OpenAI API clients
├── management/       # CLI + make_* pipeline steps
├── pipelines/        # classifiers.py — CLASSIFIERS + EXTRACTION_CONTEXTS registry
├── settings/         # credentials, logging, pricing
├── sql/context/      # SQL query templates
└── utils/            # metadata, deduplication, conversation grouping
tests/
├── unit/             # classifiers, metadata, deduplication, pricing, classifier_utils
└── integration/      # full stream pipeline (mocked BigQuery + OpenAI)
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key with Batch API access |
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON |

---

## License

Internal use only.
