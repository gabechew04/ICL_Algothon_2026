"""Functions for loading financial datasets and handling data quality issues."""

from __future__ import annotations

import pandas as pd


def load_dataset(path: str, **kwargs) -> pd.DataFrame:
    """Load a dataset from a file path.

    Supports CSV, Excel (.xlsx/.xls), JSON, and Parquet formats, determined by
    file extension.

    Parameters
    ----------
    path:
        Path to the data file.
    **kwargs:
        Additional keyword arguments forwarded to the underlying pandas reader.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path, **kwargs)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, **kwargs)
    if lower.endswith(".json"):
        return pd.read_json(path, **kwargs)
    if lower.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)
    raise ValueError(
        f"Unsupported file format for '{path}'. "
        "Supported formats: csv, xlsx, xls, json, parquet."
    )


def summarise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``missing_count`` and ``missing_pct``, indexed
        by column name, sorted by ``missing_pct`` descending.
    """
    missing_count = df.isnull().sum()
    missing_pct = missing_count / len(df) * 100
    summary = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    )
    return summary[summary["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )


def drop_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns and rows with excessive missing values.

    Columns where the fraction of missing values exceeds *threshold* are
    dropped first.  Remaining rows that contain any missing values are then
    dropped.

    Parameters
    ----------
    df:
        Input DataFrame.
    threshold:
        Maximum allowed fraction of missing values in a column (0–1).
        Columns above this threshold are removed.  Defaults to ``0.5``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a reset index.
    """
    col_threshold = int(threshold * len(df))
    df_clean = df.dropna(axis=1, thresh=col_threshold)
    df_clean = df_clean.dropna(axis=0)
    return df_clean.reset_index(drop=True)


def fill_missing(
    df: pd.DataFrame,
    strategy: str = "mean",
    fill_value=None,
) -> pd.DataFrame:
    """Fill missing values in numerical columns using the given strategy.

    Categorical / object columns are filled with their mode when *strategy* is
    ``"mean"`` or ``"median"``, or with *fill_value* when *strategy* is
    ``"constant"``.

    Parameters
    ----------
    df:
        Input DataFrame.
    strategy:
        One of ``"mean"``, ``"median"``, ``"mode"``, or ``"constant"``.
    fill_value:
        Value used when *strategy* is ``"constant"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled.

    Raises
    ------
    ValueError
        If an unknown strategy is provided.
    """
    valid = {"mean", "median", "mode", "constant"}
    if strategy not in valid:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {sorted(valid)}."
        )

    df_filled = df.copy()
    num_cols = df_filled.select_dtypes(include="number").columns
    cat_cols = df_filled.select_dtypes(exclude="number").columns

    if strategy == "mean":
        df_filled[num_cols] = df_filled[num_cols].fillna(
            df_filled[num_cols].mean()
        )
        for col in cat_cols:
            mode = df_filled[col].mode()
            if not mode.empty:
                df_filled[col] = df_filled[col].fillna(mode.iloc[0])
    elif strategy == "median":
        df_filled[num_cols] = df_filled[num_cols].fillna(
            df_filled[num_cols].median()
        )
        for col in cat_cols:
            mode = df_filled[col].mode()
            if not mode.empty:
                df_filled[col] = df_filled[col].fillna(mode.iloc[0])
    elif strategy == "mode":
        for col in df_filled.columns:
            mode = df_filled[col].mode()
            if not mode.empty:
                df_filled[col] = df_filled[col].fillna(mode.iloc[0])
    elif strategy == "constant":
        df_filled = df_filled.fillna(fill_value)

    return df_filled


def drop_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame.
    subset:
        Column labels to consider for identifying duplicates.  ``None`` uses
        all columns.
    keep:
        Which duplicate to keep.  ``"first"`` (default), ``"last"``, or
        ``False`` to drop all duplicates.

    Returns
    -------
    pd.DataFrame
        DataFrame without duplicate rows, with a reset index.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep).reset_index(
        drop=True
    )
    n_dropped = n_before - len(df_clean)
    if n_dropped:
        print(f"Dropped {n_dropped} duplicate row(s).")
    return df_clean


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a comprehensive summary of each column.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary table with dtype, missing count/percentage, number of unique
        values, and basic descriptive statistics for numerical columns.
    """
    rows = []
    for col in df.columns:
        missing = df[col].isnull().sum()
        row = {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": missing,
            "missing_pct": round(missing / len(df) * 100, 2),
            "unique_values": df[col].nunique(),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            row.update(
                {
                    "mean": round(df[col].mean(), 4),
                    "std": round(df[col].std(), 4),
                    "min": df[col].min(),
                    "max": df[col].max(),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")
