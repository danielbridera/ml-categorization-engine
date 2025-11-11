# Ruff Setup and Configuration

## Overview

Ruff is a fast Python linter and formatter written in Rust. It combines the functionality of multiple tools (flake8, isort, black, etc.) into a single fast tool.

## Installation

Ruff is already configured in `pyproject.toml` as a dev dependency:

```toml
[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-mock>=3.11.0",
    "ruff>=0.8.0",
    "mypy>=1.7.0",
]
```

Install with:
```bash
uv pip install ruff
```

## Configuration

Ruff is configured in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]  # Line too long (handled by formatter)
```

**Selected Rules:**
- `E`: pycodestyle errors
- `F`: Pyflakes (logical errors)
- `I`: isort (import sorting)
- `N`: pep8-naming
- `W`: pycodestyle warnings

## Usage

### Check for issues:
```bash
uv run ruff check src/
```

### Auto-fix issues:
```bash
uv run ruff check src/ --fix
```

### Format code:
```bash
uv run ruff format src/
```

### Check and format (recommended workflow):
```bash
uv run ruff check src/ --fix && uv run ruff format src/
```

## Initial Run Results

### Issues Found (18 total):

1. **Import Sorting (I001)** - 1 issue
   - Un-sorted imports in `openai_batch.py`
   - Auto-fixed ✅

2. **Unused Variables (F841)** - 3 issues
   - Unused exception variables in error handling
   - Unused `quality_stats` variable
   - Auto-fixed ✅

3. **F-strings Without Placeholders (F541)** - 13 issues
   - F-strings that don't use any variables
   - Files: `cli.py`, `metadata.py`, `utils.py`, `openai_batch.py`
   - Auto-fixed ✅

4. **Unused Imports (F401)** - 1 issue
   - Unused `pandas` import in `make_process_batches.py`
   - Auto-fixed ✅

### Files Modified:
- `src/categorization/clients/openai_batch.py`
- `src/categorization/management/cli.py`
- `src/categorization/management/make_preprocess.py`
- `src/categorization/management/make_process_batches.py`
- `src/categorization/utils/metadata.py`
- `src/categorization/utils/utils.py`

### Formatting:
- 11 files reformatted for consistent style
- 15 files already compliant

## Final Status

✅ **All checks passed!**
- 0 linting errors
- 0 formatting issues
- Code quality improved

## Recommended Workflow

Add ruff to your development workflow:

1. **Before committing:**
   ```bash
   uv run ruff check src/ --fix
   uv run ruff format src/
   ```

2. **In CI/CD:**
   ```bash
   uv run ruff check src/  # Fail if issues found
   uv run ruff format src/ --check  # Fail if not formatted
   ```

3. **Pre-commit hook (optional):**
   Create `.pre-commit-config.yaml`:
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.8.0
       hooks:
         - id: ruff
           args: [--fix]
         - id: ruff-format
   ```

## Benefits

- **Fast**: 10-100x faster than traditional tools
- **All-in-one**: Replaces multiple tools (flake8, isort, black)
- **Auto-fix**: Most issues can be fixed automatically
- **Consistent**: Enforces coding standards across the project
- **Modern**: Supports Python 3.11+ features and type hints

## Next Steps

1. ✅ Ruff installed and configured
2. ✅ All code linted and formatted
3. ✅ All checks passing
4. 🔄 Consider adding to CI/CD pipeline
5. 🔄 Consider adding pre-commit hooks
