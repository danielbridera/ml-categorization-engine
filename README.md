# Categorization Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Generic framework for categorizing messages using OpenAI Batch API** - Get 50% cost savings on LLM-based classification tasks.

## Overview

The Categorization Pipeline is a production-ready framework for batch classification of messages using Large Language Models. It extracts data from BigQuery, processes it through OpenAI's cost-effective Batch API, and delivers labeled datasets ready for analysis or model training.

Built on the proven CIE (Conversational Insights Engine) architecture, this pipeline provides automatic experiment tracking, robust error handling, and an extensible design for multiple categorization tasks.

### Current Support

**✅ SENTIMENT** - Classify messages as positive, negative, neutral, or mixed
**🔄 Future** - Architecture supports additional contexts (escalation detection, feedback classification, etc.)

### Key Features

- **💰 50% Cost Savings**: Uses OpenAI Batch API instead of real-time API
- **🔄 Automatic Experiment Tracking**: Each run creates timestamped experiment folders
- **🎯 Auto-Discovery**: Automatically finds latest experiment - no config files needed
- **⏱️ Async Processing**: Submit batch, wait 2-24 hours, then download results
- **📊 BigQuery Integration**: Direct extraction from `mart_conversations`
- **✨ Quality Control**: Deduplication, length filtering, validation, and retry logic
- **🏗️ Extensible Architecture**: Registry pattern ready for multiple categorization contexts
- **📈 Context-Aware**: Supports working with different categorization types via `--context_name`

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

Get started with sentiment analysis in 4 simple steps:

```bash
# Step 1: Extract 10 sample messages from BigQuery
categorization make_context \
  --start_date 2025-10-01 \
  --end_date 2025-10-31 \
  --n_samples 10

# Step 2: Clean and deduplicate
categorization make_preprocess

# Step 3a: Submit to OpenAI Batch API
categorization make_labeling

# [Wait 2-24 hours for batch processing]

# Step 3b: Download results
categorization make_process_batches
```

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
 └─ Output: sentiment_labeled.parquet ✨
```

### Context-Based Design

The pipeline uses a **context** system for different categorization tasks:

- **SENTIMENT** (current): Classify message sentiment
- **Future contexts**: ESCALATION, FEEDBACK, INTENT, etc.

All commands accept `--context_name SENTIMENT` (default) to specify which categorization task to run.

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
- `--n_samples` (optional, default: 1000): Number of messages to sample
- `--min_length` (optional, default: 6): Minimum message length in characters
- `--context_name` (optional, default: SENTIMENT): Categorization context

**Output:**
```
🔄 Extracting messages from BigQuery...
   Context: SENTIMENT
   Date range: 2025-10-01 to 2025-10-31
   Samples: 1,000, Min length: 6

✅ Context creation completed!
📁 Experiment ID: sentiment_20251105_143022

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
🔄 Preprocessing latest experiment (SENTIMENT)...

✅ Preprocessing completed!
📁 Experiment ID: sentiment_20251105_143022

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
🔄 Submitting batch for latest experiment (SENTIMENT)...

✅ Batch submitted successfully!
📁 Experiment ID: sentiment_20251105_143022
🆔 Batch ID: batch_690b866d825c819093a737fb417c529d

⏳ Batch processing will complete in 2-24 hours
💡 Check status: categorization make_process_batches --context_name SENTIMENT
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
🔄 Processing batch for latest experiment (SENTIMENT)...

⏳ Batch still processing...
💡 Try again later: categorization make_process_batches --context_name SENTIMENT
```

**Output (completed):**
```
🔄 Processing batch for latest experiment (SENTIMENT)...

✅ Batch processing completed!
📁 Experiment ID: sentiment_20251105_143022

Labeled data ready! Check: experiments/sentiment_20251105_143022/sentiment_labeled.parquet
```

**Creates:**
- `experiments/sentiment_20251105_143022/batch_results.jsonl`
- `experiments/sentiment_20251105_143022/sentiment_labeled.parquet` ✨ **FINAL OUTPUT**

**Sentiment Categories:**
- `positive` - Satisfaction, gratitude, happiness
- `negative` - Frustration, anger, disappointment
- `neutral` - Factual, informational
- `mixed` - Contains both positive and negative sentiment
- `error` - Failed to categorize

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
  ✅ context
  ✅ preprocess
  ✅ labeling
  ✅ process_batches

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

📁 sentiment_20251105_143022
   Created: 2025-11-05T14:15:25.284427
   Progress: 4/4 steps
   Status: ✅ context → ✅ preprocess → ✅ labeling → ✅ process_batches
   Batch: completed
   Parameters: 2025-10-01 to 2025-10-31, 1,000 samples

📁 sentiment_20251104_091530
   Created: 2025-11-04T09:15:30.123456
   Progress: 3/4 steps
   Status: ✅ context → ✅ preprocess → ✅ labeling → ⏸️ process_batches
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
✅ Context creation completed!
📁 Experiment ID: sentiment_20251105_090015
```

```bash
# Step 2: Clean and deduplicate
categorization make_preprocess --context_name SENTIMENT
```
**Output:**
```
Removed 25 duplicates (2.5%)
Removed 25 null/empty messages
✅ Preprocessing completed! 950 messages ready
```

```bash
# Step 3a: Submit batch to OpenAI
categorization make_labeling --context_name SENTIMENT
```
**Output:**
```
✅ Batch submitted successfully!
🆔 Batch ID: batch_abc123xyz789
⏳ Batch processing will complete in 2-24 hours
```

---

### Day 1 - Evening (6:00 PM) - Optional Status Check

```bash
# Check if batch is ready
categorization make_process_batches --context_name SENTIMENT
```
**Output:**
```
⏳ Batch still processing...
Progress: 45% (430/950 completed)
💡 Try again later
```

---

### Day 2 - Morning (9:00 AM)

```bash
# Download completed results
categorization make_process_batches --context_name SENTIMENT
```
**Output:**
```
✅ Batch processing completed!
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
Steps Completed: ✅ context → ✅ preprocess → ✅ labeling → ✅ process_batches
Batch Status: completed
Statistics: 1000 extracted → 950 preprocessed → 942 labeled
```

---

## Output Structure

### Experiment Directory

Each experiment creates a timestamped folder with all artifacts:

```
experiments/
└── sentiment_20251105_090015/
    ├── metadata.json                   # Experiment state and tracking
    ├── batch_info.json                 # OpenAI batch details and costs
    ├── sentiment_context.parquet       # Step 1: Raw extracted messages
    ├── sentiment_preprocessed.parquet  # Step 2: Cleaned messages
    ├── batch_requests.jsonl            # Step 3a: Batch API requests
    ├── batch_results.jsonl             # Step 3b: Raw API responses
    └── sentiment_labeled.parquet       # Step 3b: FINAL LABELED DATA ✨
```

### Final Output Schema

`sentiment_labeled.parquet` contains:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `workflow_id` | string | Workflow identifier | "wa-company-bot-prd" |
| `workflow_name` | string | Workflow display name | "Company Support Bot" |
| `message_id` | string | Unique message ID | "msg_abc123" |
| `message_text` | string | User message content | "Thank you for your help!" |
| `event_timestamp` | timestamp | When message was sent | 2025-10-15 14:30:22 |
| `step_name` | string | Conversation step | "initial_contact" |
| `sentiment` | string | **Labeled sentiment** | "positive" |

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

## Extensibility

### Adding New Categorization Contexts

The pipeline is designed to support multiple categorization tasks beyond SENTIMENT:

**Planned Contexts:**
- `ESCALATION` - Detect if message requires human escalation
- `FEEDBACK` - Classify as feature request, bug report, or praise
- `INTENT` - Determine user's primary intent
- `URGENCY` - Classify message urgency level

**Current Status:**
✅ Architecture supports multiple contexts via registry pattern
⚠️ Only SENTIMENT is currently implemented

**To add a new context:**
1. Create context constants in `src/categorization/pipelines/your_context/constants.py`
2. Define SQL query in `src/categorization/sql/context/your_context/`
3. Configure prompts in `src/categorization/pipelines/your_context/labeling.py`
4. Register in `src/categorization/pipelines/__init__.py`

Example future usage:
```bash
# Escalation detection (future)
categorization make_context --context_name ESCALATION --n_samples 500
categorization make_labeling --context_name ESCALATION
```

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

Files are prefixed with context name for multi-context support:

```
sentiment_context.parquet       # SENTIMENT context extraction
sentiment_preprocessed.parquet  # SENTIMENT preprocessing
sentiment_labeled.parquet       # SENTIMENT final output

# Future:
escalation_context.parquet     # ESCALATION context extraction
escalation_labeled.parquet     # ESCALATION final output
```

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key with Batch API access | `sk-proj-abc123...` |
| `GCP_PROJECT_ID` | Yes | Google Cloud project ID | `my-project-12345` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | Path to service account JSON | `/path/to/creds.json` |

---

## Version History

**Supported Contexts:**
- SENTIMENT (positive, negative, neutral, mixed)

**Future Roadmap:**
- ESCALATION context
- FEEDBACK context
- INTENT context
- BigQuery upload step
- Parallel batch processing

---


## License

Internal use only. Not for external distribution.

---

**Built with ❤️ using the CIE architecture pattern**
