"""Format-specific loaders for queryGym.

This module provides convenience functions for loading data from popular IR dataset
formats like BEIR and MS MARCO. These loaders assume you've already downloaded the
data using the respective dataset libraries.

For general file loading, use queryGym.data.DataLoader instead.
"""

from . import beir, msmarco, bright

__all__ = ["beir", "msmarco", "bright"]
