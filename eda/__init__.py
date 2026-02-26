"""EDA Financial - Exploratory Data Analysis library for financial data."""

from .loader import (
    load_dataset,
    summarise_missing,
    drop_missing,
    fill_missing,
    drop_duplicates,
    data_summary,
)
from .analysis import (
    get_numerical_columns,
    get_categorical_columns,
    describe_columns,
)
from .distributions import (
    plot_histogram,
    plot_boxplot,
    plot_qqplot,
    plot_all_distributions,
)
from .outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    winsorise,
    outlier_summary,
)
from .correlations import (
    plot_correlation_heatmap,
    plot_pairplot,
    get_top_correlations,
)

__all__ = [
    # loader
    "load_dataset",
    "summarise_missing",
    "drop_missing",
    "fill_missing",
    "drop_duplicates",
    "data_summary",
    # analysis
    "get_numerical_columns",
    "get_categorical_columns",
    "describe_columns",
    # distributions
    "plot_histogram",
    "plot_boxplot",
    "plot_qqplot",
    "plot_all_distributions",
    # outliers
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "winsorise",
    "outlier_summary",
    # correlations
    "plot_correlation_heatmap",
    "plot_pairplot",
    "get_top_correlations",
]
