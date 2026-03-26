# Categorization Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Generic framework for categorizing messages using OpenAI Batch API** - Get 50% cost savings on LLM-based classification tasks.

## Overview

The Categorization Pipeline is a production-ready framework for batch classification of messages using Large Language Models. It extracts data from BigQuery, processes it through OpenAI's cost-effective Batch API, and delivers labeled datasets ready for analysis or model training.

Built on the proven CIE (Conversational Insights Engine) architecture, this pipeline provides automatic experiment tracking, robust error handling, and an extensible design for multiple categorization tasks.

### Available Classifiers

**Message-level classifiers** (classify individual user messages):
- **SENTIMENT** - positive, negative, neutral, mixed
- **ESCALATION** - urgent, high, medium, low
- **FEEDBACK** - bug_report, feature_request, improvement, praise, question

**Conversation-level classifiers** (classify entire user+agent conversations):
- **CONVERSATION_RESOLUTION** - resolved, partially_resolved, unresolved, unclear
- **CONVERSATION_SATISFACTION** - satisfied, neutral, dissatisfied, escalated
- **CONVERSATION_INTENT** - technical_support, billing_and_payments, account_management, product_information, complaint, general_inquiry

**Adding new classifiers is trivial** - just add one configuration entry (~30 lines) to define your prompt!

### Key Features

- **50% Cost Savings**: Uses OpenAI Batch API instead of real-time API
- **Automatic Experiment Tracking**: Each run creates timestamped experiment folders
- **Auto-Discovery**: Automatically finds latest experiment - no config files needed
- **Async Processing**: Submit batch, wait 2-24 hours, then download results
- **BigQuery Integration**: Direct extraction from `mart_conversations`
- **Quality Control**: Deduplication, length filtering, validation, and retry logic
- **Extensible Architecture**: Registry pattern ready for multiple categorization contexts
- **Context-Aware**: Supports working with different categorization types via `--context_name`
- **Conversation-Level Tagging**: Classify entire user+agent conversations, not just individual messages. Uses the same timeout-based conversation definition as CIE.

## Requirements

- **Python**: 3.11 or 3.12
- **Package Manager**: `uv` (recommended) or `pip`
- **Cloud Access**: Google Cloud credentials with BigQuery read access
- **API Keys**: OpenAI API key with Batch API access

## Installation

### Step 1: Install uv (recommended)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Clone and Install Package

```bash
cd /path/to/ml-sentiment-analysis
uv pip install -e .
```

Or with pip:
```bash
pip install -e .
```

### Step 3: Configure Environment Variables

```bash
# Copy template
cp env.template .env

# Edit with your credentials
nano .env
```

Required variables in `.env`:
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
```

### Step 4: Verify Installation

```bash
categorization version
# Output: Sentiment Analysis Pipeline v1.0.0

categorization --help
# Shows available commands
```

## Quick Start

Get started with any classifier in 4 simple steps. Here's an example using SENTIMENT:

```bash
# Step 1: Extract 10 sample messages from BigQuery
categorization make_context \
  --context_name SENTIMENT \
  --start_date 2025-10-01 \
  --end_date 2025-10-31 \
  --n_samples 10

# Step 2: Clean and deduplicate
categorization make_preprocess --context_name SENTIMENT

# Step 3a: Submit to OpenAI Batch API
categorization make_labeling --context_name SENTIMENT

# [Wait 2-24 hours for batch processing]

# Step 3b: Download results
categorization make_process_batches --context_name SENTIMENT
```

**Want to try a different classifier?** Just change `SENTIMENT` to `ESCALATION` or `FEEDBACK` - same commands work for all classifiers!

## Pipeline Architecture

The pipeline follows a 4-step sequential workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CATEGORIZATION PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

 Step 1: make_context
 ├─ Extract messages from BigQuery
 ├─ Apply date filters and sampling
 └─ Output: sentiment_context.parquet
          ↓
 Step 2: make_preprocess
 ├─ Remove null/empty messages
 ├─ Filter by length
 ├─ Deduplicate within workflows
 └─ Output: sentiment_preprocessed.parquet
          ↓
 Step 3a: make_labeling
 ├─ Create batch request file (JSONL)
 ├─ Upload to OpenAI
 ├─ Submit batch job
 └─ Output: batch_requests.jsonl, batch_info.json
          ↓
 [Wait 2-24 hours - Batch API processing]
          ↓
 Step 3b: make_process_batches
 ├─ Check batch status
 ├─ Download results when ready
 ├─ Parse sentiment labels
 └─ Output: sentiment_labeled.parquet
```

### Context-Based Design

The pipeline uses a **context** system for different categorization tasks. Simply change `--context_name` to run any classifier:

**Message-level classifiers** (one label per user message):
- **SENTIMENT**: positive, negative, neutral, mixed
- **ESCALATION**: urgent, high, medium, low
- **FEEDBACK**: bug_report, feature_request, improvement, praise, question

**Conversation-level classifiers** (one label per full user+agent conversation):
- **CONVERSATION_RESOLUTION**: resolved, partially_resolved, unresolved, unclear
- **CONVERSATION_SATISFACTION**: satisfied, neutral, dissatisfied, escalated
- **CONVERSATION_INTENT**: technical_support, billing_and_payments, account_management, product_information, complaint, general_inquiry

**Same commands, different contexts:**
```bash
# Message-level
categorization make_context --context_name SENTIMENT --start_date 2025-10-01 --end_date 2025-10-31
categorization make_context --context_name ESCALATION --start_date 2025-10-01 --end_date 2025-10-31

# Conversation-level (add --workflow_name to speed up the BQ query)
categorization make_context --context_name CONVERSATION_SATISFACTION \
  --start_date 2025-10-01 --end_date 2025-10-31 --n_samples 50 --workflow_name azteca
```

All commands accept `--context_name` (defaults to SENTIMENT if not specified).

## CLI Commands Reference

### Pipeline Commands

#### 1. Extract Messages (`make_context`)

Extract user messages from BigQuery and create a new experiment.

```bash
categorization make_context \
  --start_date 2025-10-01 \
  --end_date 2025-10-31 \
  --n_samples 1000 \
  --min_length 6 \
  --context_name SENTIMENT
```

**Parameters:**
- `--start_date` (required): Start date in YYYY-MM-DD format
- `--end_date` (required): End date in YYYY-MM-DD format
- `--n_samples` (optional, default: 1000): Number of messages (or conversations) to sample
- `--min_length` (optional, default: 6): Minimum message/conversation length in characters
- `--context_name` (optional, default: SENTIMENT): Categorization context
- `--workflow_name` (optional): Filter by bot name (e.g. `azteca`). Speeds up conversation-mode queries significantly.

**Output:**
```
Extracting messages from BigQuery...
   Context: SENTIMENT
   Date range: 2025-10-01 to 2025-10-31
   Samples: 1,000, Min length: 6

Context creation completed!
Experiment ID: sentiment_20251105_143022

Next step: categorization make_preprocess --context_name SENTIMENT
```

**Creates:**
- `experiments/sentiment_20251105_143022/sentiment_context.parquet`
- `experiments/sentiment_20251105_143022/metadata.json`

---

#### 2. Clean and Deduplicate (`make_preprocess`)

Preprocess messages by removing nulls, filtering by length, and deduplicating.

```bash
# Auto-detect latest SENTIMENT experiment
categorization make_preprocess --context_name SENTIMENT

# Or specify experiment ID explicitly
categorization make_preprocess --experiment_id sentiment_20251105_143022
```

**Parameters:**
- `--experiment_id` (optional): Specific experiment ID (defaults to latest)
- `--context_name` (optional): Context for auto-detection

**What it does:**
1. Removes null and empty messages
2. Filters messages shorter than `min_length` (from Step 1)
3. Deduplicates messages within each workflow (case-insensitive)
4. Validates message quality

**Output:**
```
Preprocessing latest experiment (SENTIMENT)...

Preprocessing completed!
Experiment ID: sentiment_20251105_143022

Next step: categorization make_labeling --context_name SENTIMENT
```

**Creates:**
- `experiments/sentiment_20251105_143022/sentiment_preprocessed.parquet`

---

#### 3a. Submit Batch for Labeling (`make_labeling`)

Submit messages to OpenAI Batch API for sentiment classification.

```bash
# Auto-detect latest experiment
categorization make_labeling --context_name SENTIMENT

# Or specify experiment and model
categorization make_labeling \
  --experiment_id sentiment_20251105_143022 \
  --model gpt-4o-mini \
  --context_name SENTIMENT
```

**Parameters:**
- `--experiment_id` (optional): Specific experiment ID (defaults to latest)
- `--model` (optional, default: gpt-4o-mini): OpenAI model to use
- `--context_name` (optional, default: SENTIMENT): Categorization context

**Available Models:**
- `gpt-4o-mini` - Fast, cost-effective (recommended)
- `gpt-4o` - Higher quality, more expensive
- `gpt-3.5-turbo` - Legacy option

**What it does:**
1. Creates JSONL batch request file with all messages
2. Uploads file to OpenAI
3. Submits batch job
4. Saves batch ID and metadata

**Output:**
```
Submitting batch for latest experiment (SENTIMENT)...

Batch submitted successfully!
Experiment ID: sentiment_20251105_143022
Batch ID: batch_690b866d825c819093a737fb417c529d

Batch processing will complete in 2-24 hours
Check status: categorization make_process_batches --context_name SENTIMENT
```

**Creates:**
- `experiments/sentiment_20251105_143022/batch_requests.jsonl`
- `experiments/sentiment_20251105_143022/batch_info.json`

**Cost Estimate (gpt-4o-mini with 50% Batch discount):**
- 1,000 messages ≈ $0.01
- 10,000 messages ≈ $0.10
- 100,000 messages ≈ $1.00

---

#### 3b. Download Results (`make_process_batches`)

Check batch status and download results when ready.

```bash
# Auto-detect latest experiment
categorization make_process_batches --context_name SENTIMENT

# Or specify experiment
categorization make_process_batches --experiment_id sentiment_20251105_143022
```

**Parameters:**
- `--experiment_id` (optional): Specific experiment ID (defaults to latest)
- `--context_name` (optional, default: SENTIMENT): Categorization context

**What it does:**
1. Checks batch status with OpenAI API
2. If in progress: Shows progress and exits (run again later)
3. If completed: Downloads results, parses labels, creates labeled dataset

**Output (in progress):**
```
Processing batch for latest experiment (SENTIMENT)...

Batch still processing...
Try again later: categorization make_process_batches --context_name SENTIMENT
```

**Output (completed):**
```
Processing batch for latest experiment (SENTIMENT)...

Batch processing completed!
Experiment ID: sentiment_20251105_143022

Labeled data ready! Check: experiments/sentiment_20251105_143022/sentiment_labeled.parquet
```

**Creates:**
- `experiments/{context}_20251105_143022/batch_results.jsonl`
- `experiments/{context}_20251105_143022/{context}_labeled.parquet` **FINAL OUTPUT**

**Output Categories (vary by classifier):**
- SENTIMENT: `positive`, `negative`, `neutral`, `mixed`
- ESCALATION: `urgent`, `high`, `medium`, `low`
- FEEDBACK: `bug_report`, `feature_request`, `improvement`, `praise`, `question`
- All classifiers may also return `error` for failed classifications

---

### Utility Commands

#### Check Experiment Status

View detailed status of any experiment:

```bash
# Latest experiment (any context)
categorization status

# Latest SENTIMENT experiment
categorization status --context_name SENTIMENT

# Specific experiment
categorization status --experiment_id sentiment_20251105_143022
```

**Example Output:**
```
============================================================
Experiment: sentiment_20251105_143022
============================================================

Created: 2025-11-05T14:15:25.284427

Parameters:
  start_date: 2025-10-01
  end_date: 2025-10-31
  n_samples: 1000
  min_length: 6

Steps Completed:
  - context
  - preprocess
  - labeling
  - process_batches

Batch Status: completed
  Completed: 950/950
  Cost: $0.0095

Statistics:
  sentiment_extracted: 1000
  sentiment_after_preprocess: 950
  sentiment_labeled: 942

Directory: experiments/sentiment_20251105_143022
============================================================
```

---

#### List All Experiments

View all experiments with their status:

```bash
# List all experiments
categorization list

# Filter by context
categorization list --context_name SENTIMENT
```

**Example Output:**
```
================================================================================
All Experiments - SENTIMENT (3 total)
================================================================================

sentiment_20251105_143022
   Created: 2025-11-05T14:15:25.284427
   Progress: 4/4 steps
   Status: [DONE] context -> [DONE] preprocess -> [DONE] labeling -> [DONE] process_batches
   Batch: completed
   Parameters: 2025-10-01 to 2025-10-31, 1,000 samples

sentiment_20251104_091530
   Created: 2025-11-04T09:15:30.123456
   Progress: 3/4 steps
   Status: [DONE] context -> [DONE] preprocess -> [DONE] labeling -> [PENDING] process_batches
   Batch: in_progress
   Parameters: 2025-09-01 to 2025-09-30, 500 samples

================================================================================
```

---

#### Show Version

Display pipeline version:

```bash
categorization version
```

**Output:**
```
Sentiment Analysis Pipeline v1.0.0
```

---

## Complete Workflow Example

Here's a complete workflow for labeling 1,000 messages with sentiment:

### Day 1 - Morning (9:00 AM)

```bash
# Step 1: Extract 1,000 messages from October 2025
categorization make_context \
  --start_date 2025-10-01 \
  --end_date 2025-10-31 \
  --n_samples 1000 \
  --context_name SENTIMENT
```
**Output:**
```
Context creation completed!
Experiment ID: sentiment_20251105_090015
```

```bash
# Step 2: Clean and deduplicate
categorization make_preprocess --context_name SENTIMENT
```
**Output:**
```
Removed 25 duplicates (2.5%)
Removed 25 null/empty messages
Preprocessing completed! 950 messages ready
```

```bash
# Step 3a: Submit batch to OpenAI
categorization make_labeling --context_name SENTIMENT
```
**Output:**
```
Batch submitted successfully!
Batch ID: batch_abc123xyz789
Batch processing will complete in 2-24 hours
```

---

### Day 1 - Evening (6:00 PM) - Optional Status Check

```bash
# Check if batch is ready
categorization make_process_batches --context_name SENTIMENT
```
**Output:**
```
Batch still processing...
Progress: 45% (430/950 completed)
Try again later
```

---

### Day 2 - Morning (9:00 AM)

```bash
# Download completed results
categorization make_process_batches --context_name SENTIMENT
```
**Output:**
```
Batch processing completed!
Cost: $0.0095

Label Distribution:
  positive: 420 (44.4%)
  neutral: 310 (32.8%)
  negative: 180 (19.0%)
  mixed: 32 (3.4%)
  error: 8 (0.8%)

Labeled data ready! Check: experiments/sentiment_20251105_090015/sentiment_labeled.parquet
```

```bash
# View final status
categorization status --context_name SENTIMENT
```
**Output:**
```
Experiment: sentiment_20251105_090015
Steps Completed: [DONE] context -> [DONE] preprocess -> [DONE] labeling -> [DONE] process_batches
Batch Status: completed
Statistics: 1000 extracted -> 950 preprocessed -> 942 labeled
```

---

## Output Structure

### Experiment Directory

Each experiment creates a timestamped folder with all artifacts. The file names are automatically generated based on the classifier context:

**Example: SENTIMENT classifier**
```
experiments/
└── sentiment_20251105_090015/
    ├── metadata.json                   # Experiment state and tracking
    ├── batch_info.json                 # OpenAI batch details and costs
    ├── sentiment_context.parquet       # Step 1: Raw extracted messages
    ├── sentiment_preprocessed.parquet  # Step 2: Cleaned messages
    ├── batch_requests.jsonl            # Step 3a: Batch API requests
    ├── batch_results.jsonl             # Step 3b: Raw API responses
    └── sentiment_labeled.parquet       # Step 3b: FINAL LABELED DATA
```

**Example: ESCALATION classifier**
```
experiments/
└── escalation_20251106_120030/
    ├── metadata.json
    ├── batch_info.json
    ├── escalation_context.parquet
    ├── escalation_preprocessed.parquet
    ├── batch_requests.jsonl
    ├── batch_results.jsonl
    └── escalation_labeled.parquet      # FINAL OUTPUT
```

**Pattern:** `experiments/{context_name}_{timestamp}/{context_name}_*.parquet`

### Final Output Schema

The final labeled parquet file (`{context}_labeled.parquet`) contains all original columns plus your classifier's output column:

**Message-level classifiers — common columns:**

| Column | Type | Description |
|--------|------|-------------|
| `workflow_id` | string | Workflow identifier |
| `workflow_name` | string | Bot/workflow display name |
| `message_id` | string | Unique message ID |
| `message_text` | string | User message content |
| `event_timestamp` | timestamp | When message was sent |
| `step_name` | string | Conversation step |

**Conversation-level classifiers — common columns:**

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | `{first_message_id}_{last_message_id}` — same as CIE's `convo_id_composite` |
| `workflow_id` | string | Workflow identifier |
| `workflow_name` | string | Bot/workflow display name |
| `user_id` | string | User phone number or identifier |
| `conversation_start_time` | timestamp | Timestamp of the first message |
| `conversation_end_time` | timestamp | Timestamp of the last message |
| `message_count` | int | Total messages (user + agent) in the conversation |
| `conversation_text` | string | Full turn-by-turn transcript: `User: ...\nAssistant: ...` |

**Classifier-specific output column:**

| Classifier | Output Column | Example Values |
|------------|---------------|----------------|
| SENTIMENT | `sentiment` | positive, negative, neutral, mixed |
| ESCALATION | `urgency_level` | urgent, high, medium, low |
| FEEDBACK | `feedback_type` | bug_report, feature_request, improvement, praise, question |
| CONVERSATION_RESOLUTION | `resolution_status` | resolved, partially_resolved, unresolved, unclear |
| CONVERSATION_SATISFACTION | `satisfaction_level` | satisfied, neutral, dissatisfied, escalated |
| CONVERSATION_INTENT | `conversation_intent` | technical_support, billing_and_payments, account_management, product_information, complaint, general_inquiry |

> **`conversation_id` explained:** This is a composite key built from exactly two BigQuery `message_id` values — the first and last message of the conversation. You can always join it back to the raw `_context.parquet` file, which contains one row per message with a matching `conversation_id` column.

**Example: SENTIMENT labeled output**
```python
>>> import pandas as pd
>>> df = pd.read_parquet("experiments/sentiment_20251105_090015/sentiment_labeled.parquet")
>>> df[['message_text', 'sentiment']].head(3)
                    message_text  sentiment
0  Thank you for your help!      positive
1  This is not working          negative
2  What are your hours?         neutral
```

**Example: ESCALATION labeled output**
```python
>>> df = pd.read_parquet("experiments/escalation_20251106_120030/escalation_labeled.parquet")
>>> df[['message_text', 'urgency_level']].head(3)
                    message_text  urgency_level
0  My account is locked!         urgent
1  Can you help with this?       medium
2  Just checking in              low
```

### Metadata Structure

`metadata.json` tracks experiment state:

```json
{
  "experiment_id": "sentiment_20251105_090015",
  "context_name": "SENTIMENT",
  "created_at": "2025-11-05T09:00:15.123456",
  "parameters": {
    "start_date": "2025-10-01",
    "end_date": "2025-10-31",
    "n_samples": 1000,
    "min_length": 6
  },
  "steps_completed": ["context", "preprocess", "labeling", "process_batches"],
  "batch_info": {
    "batch_id": "batch_abc123xyz789",
    "status": "completed",
    "submitted_at": "2025-11-05T09:05:30.123456"
  },
  "stats": {
    "sentiment_extracted": 1000,
    "sentiment_after_preprocess": 950,
    "sentiment_labeled": 942
  }
}
```

---

## Adding a New Classifier

The pipeline uses a **prompt-driven architecture** - adding a new classifier is incredibly simple!

### How Simple Is It?

**Old approach:** ~150 lines across 8 files (constants, SQL queries, configs, registries)
**New approach:** ~30 lines in 1 file (just the prompt configuration!)

**80% reduction in boilerplate code**

### Step-by-Step Guide

#### 1. Edit the Classifier Registry

Open [`src/categorization/pipelines/classifiers.py`](src/categorization/pipelines/classifiers.py) and add your classifier to the `CLASSIFIERS` dictionary:

```python
CLASSIFIERS = {
    # ... existing classifiers (SENTIMENT, ESCALATION, FEEDBACK) ...

    "YOUR_CLASSIFIER": {
        "context_name": "YOUR_CLASSIFIER",
        "system_prompt": "You are an expert at classifying messages. Respond ONLY with the category name.",
        "user_prompt_template": """Classify the following message into one of these categories:

- category1: Description of category 1
- category2: Description of category 2
- category3: Description of category 3

Respond with ONLY the category name (category1, category2, or category3), nothing else.

User message: {message}""",
        "output_column": "your_label_column",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 20,
        "use_generic_query": True,
    },
}
```

#### 2. That's It!

No other files to create or modify. The system automatically:
- Generates file names (`your_classifier_context.parquet`, `your_classifier_preprocessed.parquet`, etc.)
- Uses the generic SQL query for data extraction
- Handles all preprocessing and batch submission
- Creates labeled output with your specified column name

#### 3. Run Your Classifier

```bash
# Extract messages
categorization make_context --context_name YOUR_CLASSIFIER \
  --start_date 2025-10-01 --end_date 2025-10-31 --n_samples 1000

# Preprocess
categorization make_preprocess --context_name YOUR_CLASSIFIER

# Submit to OpenAI
categorization make_labeling --context_name YOUR_CLASSIFIER

# Download results (after 2-24 hours)
categorization make_process_batches --context_name YOUR_CLASSIFIER
```

### Real Examples

The pipeline includes 3 production-ready classifiers:

**SENTIMENT Classifier:**
```python
"SENTIMENT": {
    "output_column": "sentiment",
    "system_prompt": "You are a sentiment analysis expert...",
    "user_prompt_template": "Analyze sentiment: positive, negative, neutral, or mixed...",
}
```
- Output: `sentiment_labeled.parquet` with `sentiment` column
- Categories: positive, negative, neutral, mixed

**ESCALATION Classifier:**
```python
"ESCALATION": {
    "output_column": "urgency_level",
    "system_prompt": "You are an urgency classification expert...",
    "user_prompt_template": "Classify urgency: urgent, high, medium, or low...",
}
```
- Output: `escalation_labeled.parquet` with `urgency_level` column
- Categories: urgent, high, medium, low

**FEEDBACK Classifier:**
```python
"FEEDBACK": {
    "output_column": "feedback_type",
    "system_prompt": "You are a feedback classification expert...",
    "user_prompt_template": "Classify feedback type: bug_report, feature_request, etc...",
}
```
- Output: `feedback_labeled.parquet` with `feedback_type` column
- Categories: bug_report, feature_request, improvement, praise, question

### Configuration Options

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `context_name` | Yes | Uppercase identifier for your classifier | `"SENTIMENT"` |
| `system_prompt` | Yes | System instruction for the LLM | `"You are an expert..."` |
| `user_prompt_template` | Yes | Prompt template with `{message}` placeholder | `"Classify: {message}"` |
| `output_column` | Yes | Column name for labels in output parquet | `"sentiment"` |
| `unit` | No | `"message"` (default) or `"conversation"` | `"conversation"` |
| `conversation_timeout` | No | Inactivity gap defining conversation boundary (default: `"30m"`) | `"30m"`, `"1h"`, `"2d"` |
| `model` | No | OpenAI model (default: `"gpt-4o-mini"`) | `"gpt-4o"` |
| `temperature` | No | Sampling temperature (default: `0`) | `0` |
| `max_tokens` | No | Max tokens in response (default: `10`) | `20` |
| `use_generic_query` | No | Use generic SQL query (default: `True`) | `True` |

### Adding a Conversation-Level Classifier

Set `unit: "conversation"` and use `{message}` in your prompt template (it will contain the full conversation transcript):

```python
"MY_CONV_CLASSIFIER": {
    "context_name": "MY_CONV_CLASSIFIER",
    "unit": "conversation",
    "conversation_timeout": "30m",
    "system_prompt": "You are an expert at classifying customer service conversations.",
    "user_prompt_template": """Classify the following conversation into one of these categories:
- category1: Description
- category2: Description

Respond with ONLY the category name.

Conversation:
{message}""",
    "output_column": "my_label",
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_tokens": 20,
    "use_generic_query": True,
},
```

The pipeline will automatically extract all messages (user + agent turns), group them into conversations using the timeout-based definition (matching CIE), and submit one classification request per conversation.

### Prompt Engineering Tips

1. **Be specific about categories**: List all possible categories with descriptions
2. **Request exact format**: "Respond with ONLY the category name, nothing else"
3. **Use low temperature**: Set `temperature: 0` for consistent classification
4. **Limit max_tokens**: Short responses are faster and cheaper
5. **Test your prompt**: Try a few examples manually before running full batch

---

## Development

### Install Dev Dependencies

```bash
# Install all dev tools
uv pip install ruff mypy pytest pytest-mock

# Or sync from dependency groups
uv sync --group dev
```

### Code Quality

```bash
# Format code
uv run ruff format src/

# Lint code
uv run ruff check src/

# Auto-fix linting issues
uv run ruff check src/ --fix

# Type checking
uv run mypy src/categorization
```

### Running Tests

```bash
# Run all tests
uv run pytest -vv

# Run specific test file
uv run pytest tests/test_make_preprocess.py -vv

# Run with coverage
uv run pytest --cov=src/categorization --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Project Structure

```
ml-sentiment-analysis/
├── src/categorization/
│   ├── clients/              # API clients (OpenAI, BigQuery)
│   ├── management/           # CLI and orchestration (make_* functions)
│   ├── pipelines/            # Context configs (sentiment, future: escalation, etc.)
│   ├── settings/             # Configuration, logging, credentials
│   ├── sql/                  # SQL queries by context
│   └── utils/                # Utilities (metadata, deduplication, etc.)
├── experiments/              # Generated experiment folders
├── tests/                    # Test suite
├── pyproject.toml           # Package configuration
├── env.template             # Environment variable template
└── README.md                # This file
```

---

## Troubleshooting

WIP

---


## Next Steps

After labeling your data, you can:

### 1. Export to BigQuery


### 2. Train a Classifier
Use labeled data to train a scikit-learn model:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load labeled data
df = pd.read_parquet("experiments/sentiment_20251105_090015/sentiment_labeled.parquet")

# Prepare features
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['message_text'])
y = df['sentiment']

# Train model
model = MultinomialNB()
model.fit(X, y)
```

### 3. Analyze Label Distribution

**For SENTIMENT classifier:**
```python
import pandas as pd

df = pd.read_parquet("experiments/sentiment_20251105_090015/sentiment_labeled.parquet")

# Overall distribution
print(df['sentiment'].value_counts(normalize=True))

# By workflow
print(df.groupby('workflow_name')['sentiment'].value_counts(normalize=True))

# By time period
df['month'] = pd.to_datetime(df['event_timestamp']).dt.month
print(df.groupby('month')['sentiment'].value_counts(normalize=True))
```

**For ESCALATION classifier:**
```python
df = pd.read_parquet("experiments/escalation_20251106_120030/escalation_labeled.parquet")

# Urgency distribution
print(df['urgency_level'].value_counts(normalize=True))

# Urgent messages by workflow
urgent = df[df['urgency_level'] == 'urgent']
print(urgent['workflow_name'].value_counts())
```

**For FEEDBACK classifier:**
```python
df = pd.read_parquet("experiments/feedback_20251107_150045/feedback_labeled.parquet")

# Feedback type distribution
print(df['feedback_type'].value_counts(normalize=True))

# Bug reports vs feature requests
bugs_vs_features = df[df['feedback_type'].isin(['bug_report', 'feature_request'])]
print(bugs_vs_features['feedback_type'].value_counts())
```

### 4. Quality Assurance
Spot-check a sample of labels:
```python
import pandas as pd

df = pd.read_parquet("experiments/sentiment_20251105_090015/sentiment_labeled.parquet")

# Random sample
sample = df.sample(20)
print(sample[['message_text', 'sentiment']])

# Check errors
errors = df[df['sentiment'] == 'error']
print(f"Error rate: {len(errors) / len(df) * 100:.1f}%")
print(errors[['message_text']].head())
```

---

## Reference

### Architecture Notes

- **BigQueryClient**: Robust BigQuery integration with retry logic
- **Logging System**: Structured JSON logging with emoji indicators
- **Pipeline Pattern**: `make_*` orchestration functions with CLI
- **Error Handling**: Exponential backoff for API failures

**Key Design Decisions:**
1. **Registry Pattern**: Supports multiple contexts via centralized config registry
2. **No YAML Configs**: CLI arguments only - simpler and more flexible
3. **Experiment Folders**: Timestamped folders prevent state confusion
4. **Auto-Discovery**: Latest experiment detection for seamless workflow
5. **Split Labeling**: Async submit/download accommodates 24h Batch API wait
6. **Metadata Tracking**: JSON-based state management between steps

### File Naming Conventions

Files are automatically prefixed with the lowercase context name. This pattern enables running multiple classifiers in parallel without file conflicts:

**SENTIMENT classifier:**
```
sentiment_context.parquet       # Step 1: Extracted messages
sentiment_preprocessed.parquet  # Step 2: Cleaned messages
sentiment_labeled.parquet       # Step 3b: Final output with 'sentiment' column
```

**ESCALATION classifier:**
```
escalation_context.parquet      # Step 1: Extracted messages
escalation_preprocessed.parquet # Step 2: Cleaned messages
escalation_labeled.parquet      # Step 3b: Final output with 'urgency_level' column
```

**FEEDBACK classifier:**
```
feedback_context.parquet        # Step 1: Extracted messages
feedback_preprocessed.parquet   # Step 2: Cleaned messages
feedback_labeled.parquet        # Step 3b: Final output with 'feedback_type' column
```

**Pattern:** `{context_name.lower()}_{step}.parquet`

The system auto-generates these filenames from your context name - no manual configuration needed!

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key with Batch API access | `sk-proj-abc123...` |
| `GCP_PROJECT_ID` | Yes | Google Cloud project ID | `my-project-12345` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | Path to service account JSON | `/path/to/creds.json` |

---

## Version History & Roadmap

### Current Release (v1.1.0) — Conversation-Level Tagging

**New in v1.1.0:**
- Conversation-level classification: classify entire user+agent conversations as one unit
- 3 new conversation classifiers: CONVERSATION_RESOLUTION, CONVERSATION_SATISFACTION, CONVERSATION_INTENT
- `--workflow_name` filter on `make_context` for faster BQ queries
- Conversation splitting done directly in SQL (BigQuery window functions), matching the CIE `convo_id_composite` definition exactly
- `unit` and `conversation_timeout` fields in classifier config

### v1.0.0 — Initial Release

**Architecture:**
- Prompt-driven classifier system (80% reduction in boilerplate)
- Unified CLASSIFIERS registry in single file
- Auto-generated file names and step names
- Generic SQL query for all classifiers
- Context-based experiment isolation

**Supported Classifiers:**
- SENTIMENT (positive, negative, neutral, mixed)
- ESCALATION (urgent, high, medium, low)
- FEEDBACK (bug_report, feature_request, improvement, praise, question)

### Future Roadmap

**Planned Features:**
- BigQuery upload step (write labeled data back to BQ)
- Parallel batch processing (run multiple classifiers simultaneously)
- Batch status monitoring dashboard
- Cost tracking and optimization tools

---


## License

Internal use only. Not for external distribution.

---

**Built using the CIE architecture pattern**
