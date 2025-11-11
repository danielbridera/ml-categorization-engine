"""
Pipeline Configuration Registry

Unified prompt-driven classifier registry. All text classification tasks are
configured through a single CLASSIFIERS dictionary in classifiers.py.

To add a new classifier, simply add an entry to CLASSIFIERS in classifiers.py.
No need to create separate config files or update multiple registries.
"""

from categorization.pipelines.classifiers import (
    CLASSIFIERS,
    ClassifierConfig,
    get_classifier_config,
    get_classifier_info,
    list_classifiers,
)

__all__ = [
    "CLASSIFIERS",
    "ClassifierConfig",
    "get_classifier_config",
    "get_classifier_info",
    "list_classifiers",
]
