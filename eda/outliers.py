"""Functions for outlier detection and Winsorisation of financial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import zscore

from .analysis import get_numerical_columns


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> dict[str, pd.Series]:
    """Detect outliers using the Inter-Quartile Range (IQR) method.

    A value is flagged as an outlier if it falls below
    ``Q1 - factor * IQR`` or above ``Q3 + factor * IQR``.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numerical columns to check.  Defaults to all numerical columns.
    factor:
        Multiplier applied to the IQR.  Use ``1.5`` (default) for a standard
        box-plot fence; ``3.0`` for extreme outliers only.

    Returns
    -------
    dict[str, pd.Series]
        Mapping of column name → boolean Series (``True`` marks outliers).
    """
    cols = columns if columns is not None else get_numerical_columns(df)
    result: dict[str, pd.Series] = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        result[col] = (df[col] < lower) | (df[col] > upper)
    return result


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> dict[str, pd.Series]:
    """Detect outliers using Z-score.

    A value is flagged as an outlier if ``|z-score| > threshold``.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numerical columns to check.  Defaults to all numerical columns.
    threshold:
        Absolute Z-score threshold.  Defaults to ``3.0``.

    Returns
    -------
    dict[str, pd.Series]
        Mapping of column name → boolean Series (``True`` marks outliers).
    """
    cols = columns if columns is not None else get_numerical_columns(df)
    result: dict[str, pd.Series] = {}
    for col in cols:
        col_data = df[col].dropna()
        z = np.abs(zscore(col_data))
        outlier_mask = pd.Series(False, index=df.index)
        outlier_mask.loc[col_data.index] = z > threshold
        result[col] = outlier_mask
    return result


def winsorise(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> pd.DataFrame:
    """Winsorise (clip) numerical columns to specified percentile bounds.

    Values below the *lower_pct* percentile are set to the *lower_pct*
    percentile value; values above the *upper_pct* percentile are set to the
    *upper_pct* percentile value.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to winsorise.  Defaults to all numerical columns.
    lower_pct:
        Lower percentile bound (0–1).  Defaults to ``0.05``.
    upper_pct:
        Upper percentile bound (0–1).  Defaults to ``0.95``.

    Returns
    -------
    pd.DataFrame
        DataFrame with winsorised values (other columns unchanged).

    Raises
    ------
    ValueError
        If ``lower_pct >= upper_pct`` or bounds are outside [0, 1].
    """
    if not (0 <= lower_pct < upper_pct <= 1):
        raise ValueError(
            "Percentile bounds must satisfy 0 <= lower_pct < upper_pct <= 1."
        )

    cols = columns if columns is not None else get_numerical_columns(df)
    df_out = df.copy()
    for col in cols:
        lower = df_out[col].quantile(lower_pct)
        upper = df_out[col].quantile(upper_pct)
        df_out[col] = df_out[col].clip(lower=lower, upper=upper)
    return df_out


def outlier_summary(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr",
    factor: float = 1.5,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Return a summary of outlier counts and percentages per column.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numerical columns to check.  Defaults to all numerical columns.
    method:
        Detection method: ``"iqr"`` (default) or ``"zscore"``.
    factor:
        IQR multiplier (used when *method* is ``"iqr"``).
    threshold:
        Z-score threshold (used when *method* is ``"zscore"``).

    Returns
    -------
    pd.DataFrame
        Summary with columns ``outlier_count`` and ``outlier_pct``, indexed by
        column name.

    Raises
    ------
    ValueError
        If *method* is not ``"iqr"`` or ``"zscore"``.
    """
    if method == "iqr":
        masks = detect_outliers_iqr(df, columns=columns, factor=factor)
    elif method == "zscore":
        masks = detect_outliers_zscore(df, columns=columns, threshold=threshold)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'iqr' or 'zscore'.")

    rows = []
    for col, mask in masks.items():
        count = int(mask.sum())
        rows.append(
            {
                "column": col,
                "outlier_count": count,
                "outlier_pct": round(count / len(df) * 100, 2),
            }
        )
    return pd.DataFrame(rows).set_index("column")
