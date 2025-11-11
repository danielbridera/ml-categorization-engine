# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready ML categorization pipeline that processes messages through OpenAI's Batch API for cost-effective classification (50% cheaper than real-time API). The pipeline extracts messages from BigQuery, preprocesses them, submits batch jobs to OpenAI, and produces labeled datasets.

**Architecture:** Prompt-driven classifier system - add new classifiers by adding a single configuration entry

**Available Classifiers:**
- SENTIMENT: Classify sentiment (positive, negative, neutral, mixed)
- ESCALATION: Classify urgency (urgent, high, medium, low)
- FEEDBACK: Classify feedback type (bug_report, feature_request, improvement, praise, question)

**Key Technologies:**
- Python 3.11/3.12 (strict requirement)
- `uv` package manager (strongly recommended over pip)
- Google Cloud BigQuery for data extraction
- OpenAI Batch API for classification (2-24 hour async processing)
- Parquet files for data storage

## Development Commands

### Installation
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install package in editable mode
uv pip install -e .

# Install with dev dependencies
uv sync --group dev
```

### Linting & Formatting
```bash
# Check and auto-fix linting issues, then format
uv run ruff check src/ --fix && uv run ruff format src/

# Just check (no changes)
uv run ruff check src/

# Just format
uv run ruff format src/
```

### Type Checking
```bash
uv run mypy src/categorization
```

### Testing
```bash
# Run all tests
uv run pytest -vv

# Run specific test file
uv run pytest tests/test_make_preprocess.py -vv

# Run with coverage
uv run pytest --cov=src/categorization --cov-report=html
```

### Pipeline Execution
```bash
# Complete workflow (SENTIMENT example)
categorization make_context --context_name SENTIMENT --start_date 2025-10-01 --end_date 2025-10-31 --n_samples 1000
categorization make_preprocess --context_name SENTIMENT
categorization make_labeling --context_name SENTIMENT
# [Wait 2-24 hours for OpenAI batch processing]
categorization make_process_batches --context_name SENTIMENT

# Works with any classifier (ESCALATION example)
categorization make_context --context_name ESCALATION --start_date 2025-10-01 --end_date 2025-10-31
categorization make_preprocess --context_name ESCALATION
categorization make_labeling --context_name ESCALATION
categorization make_process_batches --context_name ESCALATION

# Utility commands
categorization status              # Show current experiment status
categorization list                # List all experiments
categorization version             # Show version and available classifiers
```

## Architecture

### Pipeline Flow (4 Sequential Steps)

```
Step 1: make_context
└─> Extract messages from BigQuery → sentiment_context.parquet

Step 2: make_preprocess
└─> Clean, deduplicate, validate → sentiment_preprocessed.parquet

Step 3a: make_labeling
└─> Submit to OpenAI Batch API → batch_requests.jsonl, batch_info.json
    [Wait 2-24 hours]

Step 3b: make_process_batches
└─> Download results, merge labels → sentiment_labeled.parquet ✨
```

Each step validates prerequisites through metadata-driven state management in `experiments/{context}_{timestamp}/metadata.json`.

### Prompt-Driven Classifier Registry

All classifiers are configured in a single unified registry at [src/categorization/pipelines/classifiers.py](src/categorization/pipelines/classifiers.py):

```python
CLASSIFIERS = {
    "SENTIMENT": {
        "context_name": "SENTIMENT",
        "system_prompt": "You are a sentiment analysis expert...",
        "user_prompt_template": "Analyze the sentiment of...\n\nUser message: {message}",
        "output_column": "sentiment",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 10,
        "use_generic_query": True,
    },
    # Add more classifiers here...
}
```

**To add a new classifier:** Simply add a new entry to the CLASSIFIERS dictionary. No separate config files, SQL queries, or orchestration changes needed!

### Key Components

**Orchestration Layer:**
- [src/categorization/management/cli.py](src/categorization/management/cli.py) - Click-based CLI (9 commands)
- [src/categorization/management/make_context.py](src/categorization/management/make_context.py) - Step 1: BigQuery extraction
- [src/categorization/management/make_preprocess.py](src/categorization/management/make_preprocess.py) - Step 2: Cleaning/deduplication
- [src/categorization/management/make_labeling.py](src/categorization/management/make_labeling.py) - Step 3a: Batch submission
- [src/categorization/management/make_process_batches.py](src/categorization/management/make_process_batches.py) - Step 3b: Results processing

**Infrastructure:**
- [src/categorization/clients/bigquery.py](src/categorization/clients/bigquery.py) - BigQuery client with retry logic
- [src/categorization/clients/openai_batch.py](src/categorization/clients/openai_batch.py) - OpenAI Batch API wrapper
- [src/categorization/utils/metadata.py](src/categorization/utils/metadata.py) - Experiment tracking, auto-discovery, state management
- [src/categorization/utils/classifier_utils.py](src/categorization/utils/classifier_utils.py) - Auto-generate file names and step names from context
- [src/categorization/settings/log.py](src/categorization/settings/log.py) - Structured JSON logging with file persistence

**Classifier Configuration:**
- [src/categorization/pipelines/classifiers.py](src/categorization/pipelines/classifiers.py) - **Unified classifier registry** - all classifiers defined here
- [src/categorization/sql/context/generic_text_classification.sql](src/categorization/sql/context/generic_text_classification.sql) - Generic SQL query used by all classifiers

### Experiment Isolation

Each pipeline run creates an isolated experiment directory:
```
experiments/sentiment_20251105_143022/
├── metadata.json                      # State tracking, parameters, stats
├── batch_info.json                    # OpenAI batch details, cost
├── sentiment_context.parquet          # Step 1 output
├── sentiment_preprocessed.parquet     # Step 2 output
├── batch_requests.jsonl               # Step 3a input to OpenAI
├── batch_results.jsonl                # Step 3b downloaded results
└── sentiment_labeled.parquet          # Final labeled dataset
```

**Experiment naming:** `{context_name_lower}_{YYYYMMDD_HHMMSS}`

**Auto-discovery:** Commands automatically use the latest experiment or accept `--experiment_id` for specific runs.

## Important Patterns & Conventions

### 1. File Naming with Context Prefix
All data files are prefixed with the context name (e.g., `sentiment_context.parquet`, `sentiment_labeled.parquet`). This supports future multi-context experiments in the same directory.

### 2. Metadata-Driven State Management
Every step checks prerequisites and updates state:
```python
# Check prerequisite
if not metadata.is_step_completed("preprocess"):
    raise ValueError("Run make_preprocess first")

# Update on completion
metadata.update_step("context", stats={"messages_extracted": 1000})
```

### 3. Validation at Every Step
Data is validated before and after each operation using [src/categorization/utils/utils.py](src/categorization/utils/utils.py):
- `load_parquet_with_validation()` - Validates file exists, has required columns
- `validate_message_quality()` - Checks for nulls, quality issues
- Empty DataFrame checks after filtering/deduplication

### 4. Error Handling with Exponential Backoff
API calls use retry logic in [src/categorization/clients/bigquery.py](src/categorization/clients/bigquery.py):
```python
def retry_with_backoff(func, max_retries=3, initial_delay=1.0, backoff_factor=2.0)
```
Handles: `RateLimitError`, `APIConnectionError`, `APITimeoutError`

### 5. Structured Logging with Emoji Prefixes
[src/categorization/settings/log.py](src/categorization/settings/log.py) provides visual clarity:
- `logger.phase_start()` - 🚀
- `logger.success()` - ✅
- `logger.info()` - ℹ️
- `logger.warning()` - ⚠️
- `logger.error()` - ❌
- `logger.debug()` - 🔍

**Dual output:** Console (INFO level) + File (`logs/`, DEBUG level, JSON format)

### 6. Prompt-Driven Configuration
All classifier logic is driven by prompts in the unified registry:
```python
# pipelines/classifiers.py - Single source of truth for all classifiers
CLASSIFIERS = {
    "SENTIMENT": {
        "system_prompt": "You are a sentiment analysis expert...",
        "user_prompt_template": "Analyze the sentiment...\n\nUser message: {message}",
        "output_column": "sentiment",
        "model": "gpt-4o-mini",
    },
}
```
File names, SQL queries, and processing logic are all auto-generated from this configuration.

### 7. Deduplication Strategy
[src/categorization/utils/deduplication_utils.py](src/categorization/utils/deduplication_utils.py) implements:
- Case-insensitive text comparison
- Within-workflow deduplication (keeps first occurrence)
- Whitespace normalization
- Message quality validation

## Environment Setup

### Required Credentials

Create `.env` from `env.template`:
```bash
cp env.template .env
```

**Required variables:**
- `OPENAI_API_KEY` - OpenAI API key with Batch API access
- `GCP_PROJECT_ID` - Google Cloud project ID
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account JSON file

**GCP Service Account Requirements:**
- Read access to `arched-photon-194421.DWH2.mart_conversations` table in BigQuery

### OpenAI Batch API Notes

- **Async Processing:** Submit batch → wait 2-24 hours → download results
- **Cost Savings:** 50% cheaper than real-time API
- **Pricing estimate:** ~$0.01 per 1,000 messages (varies by model/length)
- **Completion window:** 24 hours (configurable in API call)

## Configuration Files

### pyproject.toml
Central configuration for:
- Package metadata and dependencies
- Ruff linting rules (line length 88, select E/F/I/N/W)
- Pytest configuration (pythonpath, testpaths)
- Mypy type checking settings
- CLI entry point: `categorization = "categorization.management.cli:main"`

### .gitignore
Key exclusions:
- `.env` (keeps `env.template`)
- `experiments/` (all experiment data)
- `logs/` (log files)
- `*.parquet`, `*.jsonl`, `*.json` (except configs)

## Adding a New Classifier

Adding a new classifier is incredibly simple - just add one configuration entry!

**Edit [src/categorization/pipelines/classifiers.py](src/categorization/pipelines/classifiers.py)** and add your new classifier:

```python
CLASSIFIERS = {
    # ... existing classifiers ...

    "YOUR_CLASSIFIER": {
        "context_name": "YOUR_CLASSIFIER",
        "system_prompt": "Your system instruction here...",
        "user_prompt_template": """Your classification prompt:
- category1: Description
- category2: Description

Respond with ONLY the category name, nothing else.

User message: {message}""",
        "output_column": "your_label_column",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 10,
        "use_generic_query": True,
    },
}
```

**That's it!** No other files to create or modify. The system automatically:
- Generates file names (`your_classifier_context.parquet`, etc.)
- Uses the generic SQL query for data extraction
- Handles all preprocessing and batch submission
- Creates the labeled output with your specified column name

**Run the pipeline:**
```bash
categorization make_context --context_name YOUR_CLASSIFIER --start_date 2025-10-01 --end_date 2025-10-31
categorization make_preprocess --context_name YOUR_CLASSIFIER
categorization make_labeling --context_name YOUR_CLASSIFIER
categorization make_process_batches --context_name YOUR_CLASSIFIER
```
