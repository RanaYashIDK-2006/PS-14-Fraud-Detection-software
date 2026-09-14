"""Anti-tuning safeguards — prevents accidental use of external data for tuning.

The external evaluation dataset must be treated as a final holdout. If a user
attempts to run tuning against it, produce a clear error or warning.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable


class ExternalDataProtectionError(Exception):
    """Raised when external data is used inappropriately."""
    pass


class TuningAttemptDetected(Exception):
    """Raised when a tuning operation is attempted with external data."""
    pass


# Functions that should NEVER be called with external data
TUNING_FUNCTIONS = {
    "fit",
    "fit_transform",
    "partial_fit",
    "update",
    "train",
    "retrain",
    "finetune",
    "tune",
    "calibrate",
    "optimize",
    "grid_search",
    "random_search",
    "bayesian_search",
}


# Classes that represent tuning operations
TUNING_CLASSES = {
    "GridSearchCV",
    "RandomizedSearchCV",
    "BayesSearchCV",
    "OptunaSearch",
}


def protect_external_data(func: Callable) -> Callable:
    """Decorator that prevents tuning functions from being called with external data.

    Use this to wrap any function that should not modify the model when
    external data is the input.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Check if any argument looks like external data
        for arg in args:
            if isinstance(arg, str) and "external" in arg.lower():
                raise TuningAttemptDetected(
                    f"Function '{func.__name__}' cannot be called with external data. "
                    f"External datasets must be used for evaluation only, not training."
                )
        for key, val in kwargs.items():
            if isinstance(val, str) and "external" in val.lower():
                raise TuningAttemptDetected(
                    f"Function '{func.__name__}' cannot be called with external data. "
                    f"External datasets must be used for evaluation only, not training."
                )
        return func(*args, **kwargs)
    return wrapper


def validate_no_tuning(X: Any, y: Any = None) -> tuple[bool, str]:
    """Validate that data is not being used for tuning.

    Args:
        X: Feature matrix.
        y: Labels (optional).

    Returns:
        (is_safe, message)
    """
    # Check if data has markers indicating it's external
    if hasattr(X, "attrs") and X.attrs.get("is_external", False):
        return False, "Data is marked as external — cannot be used for tuning"

    return True, "No tuning violations detected"


def create_heldout_guard(dataset_name: str) -> Callable:
    """Create a guard that prevents tuning on a specific heldout dataset.

    Args:
        dataset_name: Name of the heldout dataset to protect.

    Returns:
        A decorator that raises TuningAttemptDetected if the dataset is used for tuning.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if dataset_name appears in any argument
            all_args = list(args) + list(kwargs.values())
            for arg in all_args:
                if isinstance(arg, str) and dataset_name in arg:
                    raise TuningAttemptDetected(
                        f"Dataset '{dataset_name}' is a heldout evaluation set. "
                        f"Function '{func.__name__}' cannot modify models using this data."
                    )
                if hasattr(arg, "attrs") and arg.attrs.get("dataset_name") == dataset_name:
                    raise TuningAttemptDetected(
                        f"Dataset '{dataset_name}' is a heldout evaluation set. "
                        f"Function '{func.__name__}' cannot modify models using this data."
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator
