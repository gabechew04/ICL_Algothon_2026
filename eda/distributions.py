"""Functions for plotting column distributions."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

from .analysis import get_numerical_columns


def plot_histogram(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 30,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot histograms for numerical columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to plot.  Defaults to all numerical columns.
    bins:
        Number of histogram bins.
    figsize:
        Figure size ``(width, height)`` in inches.  Auto-computed when
        ``None``.
    save_path:
        If provided, the figure is saved to this path instead of displayed.
    """
    cols = columns if columns is not None else get_numerical_columns(df)
    if not cols:
        warnings.warn("No numerical columns found for histogram.")
        return

    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    width, height = figsize or (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    axes = [axes] if n == 1 else list(_flatten(axes))

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.hist(data, bins=bins, edgecolor="black", color="steelblue", alpha=0.7)
        ax.set_title(f"Histogram – {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_boxplot(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot boxplots for numerical columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to plot.  Defaults to all numerical columns.
    figsize:
        Figure size ``(width, height)`` in inches.  Auto-computed when
        ``None``.
    save_path:
        If provided, the figure is saved to this path instead of displayed.
    """
    cols = columns if columns is not None else get_numerical_columns(df)
    if not cols:
        warnings.warn("No numerical columns found for boxplot.")
        return

    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    width, height = figsize or (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    axes = [axes] if n == 1 else list(_flatten(axes))

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.boxplot(data, patch_artist=True, orientation="vertical",
                   boxprops=dict(facecolor="steelblue", alpha=0.7))
        ax.set_title(f"Boxplot – {col}")
        ax.set_ylabel(col)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_qqplot(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot Q-Q plots (against normal distribution) for numerical columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to plot.  Defaults to all numerical columns.
    figsize:
        Figure size ``(width, height)`` in inches.  Auto-computed when
        ``None``.
    save_path:
        If provided, the figure is saved to this path instead of displayed.
    """
    cols = columns if columns is not None else get_numerical_columns(df)
    if not cols:
        warnings.warn("No numerical columns found for Q-Q plot.")
        return

    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    width, height = figsize or (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    axes = [axes] if n == 1 else list(_flatten(axes))

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        (theoretical_quantiles, sample_quantiles), (slope, intercept, _) = stats.probplot(data, dist="norm")
        ax.scatter(theoretical_quantiles, sample_quantiles, color="steelblue", alpha=0.6, s=10)
        ax.plot(
            [min(theoretical_quantiles), max(theoretical_quantiles)],
            [slope * min(theoretical_quantiles) + intercept, slope * max(theoretical_quantiles) + intercept],
            color="red",
            linewidth=1.5,
        )
        ax.set_title(f"Q-Q Plot – {col}")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_all_distributions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 30,
    figsize_per_col: tuple[int, int] = (5, 4),
    save_path: str | None = None,
) -> None:
    """Plot histogram, boxplot, and Q-Q plot side-by-side for each column.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to plot.  Defaults to all numerical columns.
    bins:
        Number of histogram bins.
    figsize_per_col:
        ``(width, height)`` allocated per sub-plot in inches.
    save_path:
        If provided, the figure is saved to this path instead of displayed.
    """
    cols = columns if columns is not None else get_numerical_columns(df)
    if not cols:
        warnings.warn("No numerical columns found for distribution plots.")
        return

    n = len(cols)
    fig, axes = plt.subplots(
        n, 3, figsize=(3 * figsize_per_col[0], n * figsize_per_col[1])
    )
    # Ensure axes is always 2-D
    if n == 1:
        axes = [axes]

    for row_axes, col in zip(axes, cols):
        data = df[col].dropna()

        # Histogram
        row_axes[0].hist(
            data, bins=bins, edgecolor="black", color="steelblue", alpha=0.7
        )
        row_axes[0].set_title(f"Histogram – {col}")
        row_axes[0].set_xlabel(col)
        row_axes[0].set_ylabel("Frequency")

        # Boxplot
        row_axes[1].boxplot(
            data, patch_artist=True, orientation="vertical",
            boxprops=dict(facecolor="steelblue", alpha=0.7),
        )
        row_axes[1].set_title(f"Boxplot – {col}")
        row_axes[1].set_ylabel(col)

        # Q-Q plot
        (theoretical_quantiles, sample_quantiles), (slope, intercept, _) = stats.probplot(data, dist="norm")
        row_axes[2].scatter(theoretical_quantiles, sample_quantiles, color="steelblue", alpha=0.6, s=10)
        row_axes[2].plot(
            [min(theoretical_quantiles), max(theoretical_quantiles)],
            [slope * min(theoretical_quantiles) + intercept, slope * max(theoretical_quantiles) + intercept],
            color="red",
            linewidth=1.5,
        )
        row_axes[2].set_title(f"Q-Q Plot – {col}")
        row_axes[2].set_xlabel("Theoretical quantiles")
        row_axes[2].set_ylabel("Sample quantiles")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten(axes):
    """Flatten a potentially nested numpy array of Axes objects."""
    try:
        for ax in axes:
            yield from _flatten(ax)
    except TypeError:
        yield axes
