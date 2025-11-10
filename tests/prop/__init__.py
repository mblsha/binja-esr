"""
Property-based testing utilities for SC62015 SCIL vs legacy equivalence.

This package hosts Hypothesis strategies, harness helpers, and the test
entrypoints for both the fast CI lane and the nightly fuzz job.
"""

# Apply LLIL evaluator patches before any harness code runs.
from . import llil_patches  # noqa: F401
