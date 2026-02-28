"""Functions for plotting correlations between columns."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .analysis import get_numerical_columns


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    figsize: tuple[int, int] | None = None,
    annot: bool = True,
    cmap: str = "coolwarm",
    save_path: str | None = None,
) -> pd.DataFrame:
    """Plot a correlation heatmap for numerical columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to include.  Defaults to all numerical columns.
    method:
        Correlation method: ``"pearson"`` (default), ``"spearman"``, or
        ``"kendall"``.
    figsize:
        Figure size ``(width, height)`` in inches.  Auto-computed when
        ``None``.
    annot:
        Whether to annotate each cell with the correlation value.
    cmap:
        Colormap for the heatmap.
    save_path:
        If provided, the figure is saved to this path instead of displayed.

    Returns
    -------
    pd.DataFrame
        The correlation matrix.
    """
    cols = columns if columns is not None else get_numerical_columns(df)
    if not cols:
        warnings.warn("No numerical columns found for correlation heatmap.")
        return pd.DataFrame()

    corr = df[cols].corr(method=method)
    n = len(cols)
    width, height = figsize or (max(8, n), max(6, n - 1))

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        corr,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Correlation Matrix ({method.capitalize()})")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

    return corr


def plot_pairplot(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    hue: str | None = None,
    diag_kind: str = "hist",
    save_path: str | None = None,
) -> None:
    """Plot a pairplot (scatter matrix) for numerical columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to include.  Defaults to all numerical columns (max 10 to
        keep the plot readable).
    hue:
        Categorical column to use for colour-coding points.
    diag_kind:
        Kind of plot on the diagonal: ``"hist"`` (default) or ``"kde"``.
    save_path:
        If provided, the figure is saved to this path instead of displayed.
    """
    cols = columns if columns is not None else get_numerical_columns(df)[:10]
    if not cols:
        warnings.warn("No numerical columns found for pairplot.")
        return

    plot_cols = list(cols)
    if hue and hue not in plot_cols:
        plot_cols_with_hue = plot_cols + [hue]
    else:
        plot_cols_with_hue = plot_cols

    pair_grid = sns.pairplot(
        df[plot_cols_with_hue],
        hue=hue,
        diag_kind=diag_kind,
        plot_kws={"alpha": 0.5, "s": 15},
    )
    pair_grid.fig.suptitle("Pairplot", y=1.02)

    if save_path:
        pair_grid.fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(pair_grid.fig)


def get_top_correlations(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    n: int = 10,
    absolute: bool = True,
) -> pd.DataFrame:
    """Return the top correlated column pairs.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to include.  Defaults to all numerical columns.
    method:
        Correlation method: ``"pearson"`` (default), ``"spearman"``, or
        ``"kendall"``.
    n:
        Number of top pairs to return.
    absolute:
        If ``True`` (default), rank by absolute correlation value so that
        strong negative correlations are also surfaced.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``col_a``, ``col_b``, and ``correlation``,
        sorted by absolute (or raw) correlation descending.
    """
    cols = columns if columns is not None else get_numerical_columns(df)
    if not cols:
        warnings.warn("No numerical columns found for correlation analysis.")
        return pd.DataFrame(columns=["col_a", "col_b", "correlation"])

    corr = df[cols].corr(method=method)

    # Extract upper triangle (exclude diagonal)
    pairs = []
    col_list = corr.columns.tolist()
    for i in range(len(col_list)):
        for j in range(i + 1, len(col_list)):
            val = corr.iloc[i, j]
            pairs.append(
                {"col_a": col_list[i], "col_b": col_list[j], "correlation": val}
            )

    result = pd.DataFrame(pairs)
    if result.empty:
        return result

    sort_key = result["correlation"].abs() if absolute else result["correlation"]
    return result.iloc[sort_key.argsort()[::-1]].head(n).reset_index(drop=True)
