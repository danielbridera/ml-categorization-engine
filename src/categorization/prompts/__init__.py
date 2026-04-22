"""
Prompts package — classifier specs live one-per-file in family subfolders.

Layout::

    prompts/
    ├── _base.py                 # ClassifierSpec dataclass
    ├── monitoring/
    │   ├── __init__.py
    │   └── <eval_name>.py       # exports CLASSIFIER = ClassifierSpec(...)
    └── conversations/
        ├── __init__.py
        └── <classifier_name>.py

Adding a new classifier is a one-file change: drop a ``*.py`` in the right
family folder, define ``CLASSIFIER = ClassifierSpec(...)``, and the auto-loader
picks it up. No edits to ``classifiers.py`` required.

Files starting with ``_`` are skipped (internal helpers).
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from categorization.prompts._base import ClassifierSpec
from categorization.settings.log import logger


def _discover_specs() -> dict[str, ClassifierSpec]:
    """Walk the package, import every non-underscore module, collect CLASSIFIERs.

    Duplicate ``name`` across specs is an error — fail loudly so naming
    conflicts don't silently overwrite each other.
    """
    from categorization import prompts as pkg

    specs: dict[str, ClassifierSpec] = {}
    for module_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
        name = module_info.name
        leaf = name.rsplit(".", 1)[-1]
        if leaf.startswith("_") or module_info.ispkg:
            continue
        module = importlib.import_module(name)
        spec = getattr(module, "CLASSIFIER", None)
        if spec is None:
            continue
        if not isinstance(spec, ClassifierSpec):
            logger.warning(
                f"Module {name} exports CLASSIFIER but it isn't a ClassifierSpec — skipping"
            )
            continue
        if spec.name in specs:
            raise RuntimeError(
                f"Duplicate ClassifierSpec name {spec.name!r}: "
                f"already registered from {specs[spec.name].__class__.__module__}, "
                f"conflicts with {name}"
            )
        specs[spec.name] = spec
    return specs


def discover_classifier_configs() -> dict[str, dict[str, Any]]:
    """Return discovered classifiers as ``{name: ClassifierConfig-shaped dict}``.

    Called at import time by ``pipelines.classifiers`` to merge into the
    legacy ``CLASSIFIERS`` registry so downstream code doesn't need to change.
    """
    return {name: spec.to_config() for name, spec in _discover_specs().items()}


__all__ = ["ClassifierSpec", "discover_classifier_configs"]
