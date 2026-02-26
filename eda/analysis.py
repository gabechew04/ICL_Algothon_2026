"""Functions for identifying and describing column types in a DataFrame."""

from __future__ import annotations

import pandas as pd


def get_numerical_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of numerical (numeric dtype) column names.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    list[str]
        Column names with numeric dtypes.
    """
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(
    df: pd.DataFrame,
    max_unique: int | None = None,
) -> list[str]:
    """Return a list of categorical column names.

    A column is considered categorical if its dtype is ``object``,
    ``category``, or ``bool``.  Optionally, numerical columns with at most
    *max_unique* distinct values are also included (useful for low-cardinality
    integer codes).

    Parameters
    ----------
    df:
        Input DataFrame.
    max_unique:
        When set, numerical columns with ``nunique() <= max_unique`` are also
        returned as categorical.  ``None`` (default) disables this behaviour.

    Returns
    -------
    list[str]
        Categorical column names.
    """
    cat_cols = df.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()

    if max_unique is not None:
        for col in df.select_dtypes(include="number").columns:
            if df[col].nunique() <= max_unique and col not in cat_cols:
                cat_cols.append(col)

    return cat_cols


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table categorising each column.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Table with columns ``dtype``, ``kind`` (``"numerical"`` or
        ``"categorical"``), ``unique_values``, ``missing_count``, and
        ``missing_pct``, indexed by column name.
    """
    num_cols = set(get_numerical_columns(df))
    rows = []
    for col in df.columns:
        missing = df[col].isnull().sum()
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "kind": "numerical" if col in num_cols else "categorical",
                "unique_values": df[col].nunique(),
                "missing_count": missing,
                "missing_pct": round(missing / len(df) * 100, 2),
            }
        )
    return pd.DataFrame(rows).set_index("column")
