"""Filter Registry for Dataset Subset Generation.

Provides common filter implementations for selecting records from datasets.
Filters declare whether they need denormalized data (requires_data=True)
or can work from just a list of member RIDs (requires_data=False).

- requires_data=False: The script uses list_dataset_members() directly.
  Fast, no bag download. Use for random sampling, all_records, etc.
- requires_data=True: The script downloads a metadata-only bag and
  denormalizes into DataFrames. Use for value-based filtering.

Custom filters can be registered with @register_filter("name") and
referenced by name in hydra configs via filter_name parameter.

Built-in filters:
  - random_sample: Random sample of n records (requires_data=False)
  - all_records: All records from the element table (requires_data=False)
  - has_feature: Records with a non-null feature value (requires_data=True)
  - feature_equals: Records matching a specific value (requires_data=True)
  - feature_in: Records matching any of several values (requires_data=True)
  - numeric_range: Records within numeric bounds (requires_data=True)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


class DataFilterFunction(Protocol):
    """Protocol for filters that need denormalized DataFrames."""

    def __call__(
        self,
        dataframes: dict[str, pd.DataFrame],
        *,
        element_table: str,
        **kwargs,
    ) -> tuple[list[str], str]: ...


class MemberFilterFunction(Protocol):
    """Protocol for filters that only need a list of RIDs."""

    def __call__(
        self,
        member_rids: list[str],
        **kwargs,
    ) -> tuple[list[str], str]: ...


@dataclass
class FilterEntry:
    """A registered filter with metadata."""
    fn: DataFilterFunction | MemberFilterFunction
    requires_data: bool
    name: str = ""


FILTER_REGISTRY: dict[str, FilterEntry] = {}


def register_filter(name: str, *, requires_data: bool = True):
    """Decorator to register a filter function by name.

    Args:
        name: Lookup name for the filter.
        requires_data: If False, the filter only needs member RIDs (fast path).
            If True, the filter needs denormalized DataFrames (bag download).
    """
    def decorator(fn):
        FILTER_REGISTRY[name] = FilterEntry(fn=fn, requires_data=requires_data, name=name)
        return fn
    return decorator


def get_filter(name: str) -> FilterEntry:
    """Look up a registered filter by name with a helpful error message."""
    if name not in FILTER_REGISTRY:
        available = ", ".join(sorted(FILTER_REGISTRY.keys()))
        raise ValueError(f"Unknown filter '{name}'. Available filters: {available}")
    return FILTER_REGISTRY[name]


# =============================================================================
# Shared helpers (for data-requiring filters)
# =============================================================================


def _merge_dataframes(
    dataframes: dict[str, pd.DataFrame],
    element_table: str,
) -> pd.DataFrame:
    """Concatenate source DataFrames and validate the RID column exists."""
    if not dataframes:
        raise ValueError("No source DataFrames provided — source_dataset_rids may be empty")
    df = pd.concat(dataframes.values(), ignore_index=True)
    rid_col = f"{element_table}.RID"
    if rid_col not in df.columns:
        available = sorted(c for c in df.columns if c.endswith(".RID"))
        raise KeyError(
            f"Column '{rid_col}' not found in DataFrame. "
            f"RID columns available: {available}"
        )
    return df


def _extract_rids(df: pd.DataFrame, element_table: str) -> list[str]:
    """Extract unique RIDs from a DataFrame."""
    return df[f"{element_table}.RID"].unique().tolist()


def _validate_column(df: pd.DataFrame, column: str) -> None:
    """Validate that a column exists in the DataFrame."""
    if column not in df.columns:
        available = sorted(c for c in df.columns if not c.startswith("_"))
        raise KeyError(
            f"Column '{column}' not found in DataFrame. "
            f"Available columns: {available}"
        )


# =============================================================================
# Member-only filters (requires_data=False — fast path, no bag download)
# =============================================================================


@register_filter("random_sample", requires_data=False)
def random_sample(
    member_rids: list[str],
    *,
    n: int = 100,
    seed: int = 42,
    **kwargs,
) -> tuple[list[str], str]:
    """Select a random sample of n records.

    Use this for creating small development/iteration datasets.
    The seed ensures reproducibility.
    """
    if n >= len(member_rids):
        desc = f"Requested {n} but only {len(member_rids)} available — returning all"
        return member_rids, desc

    rng = random.Random(seed)
    sampled = rng.sample(member_rids, n)

    desc = f"Randomly sampled {len(sampled)} of {len(member_rids)} records (seed={seed})"
    return sampled, desc


@register_filter("all_records", requires_data=False)
def all_records(
    member_rids: list[str],
    **kwargs,
) -> tuple[list[str], str]:
    """Select all records — no filtering.

    Use this to create a dataset containing every member of the source.
    """
    desc = f"Selected all {len(member_rids)} records"
    return member_rids, desc


# =============================================================================
# Data-requiring filters (requires_data=True — bag download path)
# =============================================================================


@register_filter("has_feature", requires_data=True)
def has_feature(
    dataframes: dict[str, pd.DataFrame],
    *,
    element_table: str,
    column: str,
    **kwargs,
) -> tuple[list[str], str]:
    """Select records that have a non-null value for a feature column."""
    df = _merge_dataframes(dataframes, element_table)
    _validate_column(df, column)
    selected = df[df[column].notna()]
    rids = _extract_rids(selected, element_table)

    total = df[f"{element_table}.RID"].nunique()
    desc = f"Selected {len(rids)} of {total} records that have a value for {column}"
    return rids, desc


@register_filter("feature_equals", requires_data=True)
def feature_equals(
    dataframes: dict[str, pd.DataFrame],
    *,
    element_table: str,
    column: str,
    value: str,
    **kwargs,
) -> tuple[list[str], str]:
    """Select records where a feature column matches a specific value."""
    df = _merge_dataframes(dataframes, element_table)
    _validate_column(df, column)
    selected = df[df[column] == value]
    rids = _extract_rids(selected, element_table)

    desc = f"Selected {len(rids)} records where {column} = '{value}'"
    return rids, desc


@register_filter("feature_in", requires_data=True)
def feature_in(
    dataframes: dict[str, pd.DataFrame],
    *,
    element_table: str,
    column: str,
    values: list[str],
    **kwargs,
) -> tuple[list[str], str]:
    """Select records where a feature column is in a list of values."""
    df = _merge_dataframes(dataframes, element_table)
    _validate_column(df, column)
    selected = df[df[column].isin(values)]
    rids = _extract_rids(selected, element_table)

    if len(values) > 10:
        values_str = ", ".join(values[:10]) + f", ... ({len(values)} total)"
    else:
        values_str = ", ".join(values)
    desc = f"Selected {len(rids)} records where {column} in: {values_str}"
    return rids, desc


@register_filter("numeric_range", requires_data=True)
def numeric_range(
    dataframes: dict[str, pd.DataFrame],
    *,
    element_table: str,
    column: str,
    min_val: float | None = None,
    max_val: float | None = None,
    **kwargs,
) -> tuple[list[str], str]:
    """Select records where a numeric column is within bounds."""
    if min_val is None and max_val is None:
        raise ValueError(
            "numeric_range requires at least one of min_val or max_val. "
            "Use 'all_records' or 'has_feature' if no bounds are needed."
        )
    df = _merge_dataframes(dataframes, element_table)
    _validate_column(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(
            f"Column '{column}' has dtype '{df[column].dtype}', expected a numeric type"
        )
    mask = df[column].notna()
    if min_val is not None:
        mask = mask & (df[column] >= min_val)
    if max_val is not None:
        mask = mask & (df[column] <= max_val)
    selected = df[mask]
    rids = _extract_rids(selected, element_table)

    bounds = []
    if min_val is not None:
        bounds.append(f">= {min_val}")
    if max_val is not None:
        bounds.append(f"<= {max_val}")
    desc = f"Selected {len(rids)} records where {column} {' and '.join(bounds)}"
    return rids, desc
