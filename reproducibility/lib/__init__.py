"""Private Python helpers for QueryGym reproducibility tooling.

The contract that crosses repo boundaries is `reproducibility/schema.json`.
This module is internal to this repository and used by the example pipeline,
the aggregator, the submission tool, and the tests. External consumers
(the dashboard product, third-party tools) should read schema.json directly.
"""

from .emit import (
    SCHEMA_VERSION,
    build_run_summary,
    compute_params_hash,
    compute_run_id,
)
from .validate import validate, ValidationError

__all__ = [
    "SCHEMA_VERSION",
    "build_run_summary",
    "compute_params_hash",
    "compute_run_id",
    "validate",
    "ValidationError",
]
