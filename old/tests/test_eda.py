"""Unit tests for the eda financial library."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Use a non-interactive matplotlib backend so tests run in CI without a display
import matplotlib
matplotlib.use("Agg")

from eda import (
    load_dataset,
    summarise_missing,
    drop_missing,
    fill_missing,
    drop_duplicates,
    data_summary,
    get_numerical_columns,
    get_categorical_columns,
    describe_columns,
    plot_histogram,
    plot_boxplot,
    plot_qqplot,
    plot_all_distributions,
    detect_outliers_iqr,
    detect_outliers_zscore,
    winsorise,
    outlier_summary,
    plot_correlation_heatmap,
    plot_pairplot,
    get_top_correlations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small DataFrame with mixed dtypes and known properties."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "price": rng.normal(100, 15, n),
            "volume": rng.exponential(1000, n),
            "returns": rng.normal(0, 0.02, n),
            "sector": rng.choice(["Tech", "Finance", "Energy"], n),
            "flag": rng.choice([True, False], n),
        }
    )


@pytest.fixture()
def df_with_missing(sample_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_df.copy()
    df.loc[0:5, "price"] = np.nan
    df.loc[10:12, "sector"] = np.nan
    return df


@pytest.fixture()
def df_with_outliers() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 200).tolist()
    data += [50.0, -50.0]  # extreme outliers
    return pd.DataFrame({"value": data})


# ---------------------------------------------------------------------------
# loader tests
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_load_csv(self, sample_df: pd.DataFrame) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            sample_df.to_csv(path, index=False)
            loaded = load_dataset(path)
            assert loaded.shape == sample_df.shape
        finally:
            os.unlink(path)

    def test_load_json(self, sample_df: pd.DataFrame) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sample_df.to_json(path)
            loaded = load_dataset(path)
            assert set(loaded.columns) == set(sample_df.columns)
        finally:
            os.unlink(path)

    def test_unsupported_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset("data.txt")


class TestSummariseMissing:
    def test_no_missing(self, sample_df: pd.DataFrame) -> None:
        result = summarise_missing(sample_df)
        assert result.empty

    def test_with_missing(self, df_with_missing: pd.DataFrame) -> None:
        result = summarise_missing(df_with_missing)
        assert "price" in result.index
        assert result.loc["price", "missing_count"] == 6


class TestDropMissing:
    def test_drops_rows(self, df_with_missing: pd.DataFrame) -> None:
        cleaned = drop_missing(df_with_missing)
        assert cleaned.isnull().sum().sum() == 0

    def test_threshold(self) -> None:
        df = pd.DataFrame({"a": [np.nan] * 90 + [1.0] * 10, "b": range(100)})
        cleaned = drop_missing(df, threshold=0.5)
        assert "a" not in cleaned.columns


class TestFillMissing:
    def test_mean_strategy(self, df_with_missing: pd.DataFrame) -> None:
        filled = fill_missing(df_with_missing, strategy="mean")
        assert filled["price"].isnull().sum() == 0

    def test_median_strategy(self, df_with_missing: pd.DataFrame) -> None:
        filled = fill_missing(df_with_missing, strategy="median")
        assert filled["price"].isnull().sum() == 0

    def test_constant_strategy(self, df_with_missing: pd.DataFrame) -> None:
        filled = fill_missing(df_with_missing, strategy="constant", fill_value=0)
        assert filled.isnull().sum().sum() == 0
        assert (filled["price"].iloc[0:6] == 0).all()

    def test_invalid_strategy(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            fill_missing(sample_df, strategy="unknown")


class TestDropDuplicates:
    def test_removes_duplicates(self, sample_df: pd.DataFrame) -> None:
        df_dup = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        cleaned = drop_duplicates(df_dup)
        assert len(cleaned) == len(sample_df)

    def test_no_duplicates(self, sample_df: pd.DataFrame) -> None:
        cleaned = drop_duplicates(sample_df)
        assert len(cleaned) == len(sample_df)


class TestDataSummary:
    def test_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        summary = data_summary(sample_df)
        assert isinstance(summary, pd.DataFrame)
        assert set(summary.index) == set(sample_df.columns)

    def test_has_expected_columns(self, sample_df: pd.DataFrame) -> None:
        summary = data_summary(sample_df)
        for col in ("dtype", "missing_count", "missing_pct", "unique_values"):
            assert col in summary.columns


# ---------------------------------------------------------------------------
# analysis tests
# ---------------------------------------------------------------------------

class TestGetNumericalColumns:
    def test_returns_numeric(self, sample_df: pd.DataFrame) -> None:
        num_cols = get_numerical_columns(sample_df)
        assert set(num_cols) == {"price", "volume", "returns"}

    def test_empty_df(self) -> None:
        assert get_numerical_columns(pd.DataFrame()) == []


class TestGetCategoricalColumns:
    def test_returns_categorical(self, sample_df: pd.DataFrame) -> None:
        cat_cols = get_categorical_columns(sample_df)
        assert "sector" in cat_cols
        assert "flag" in cat_cols

    def test_max_unique(self) -> None:
        df = pd.DataFrame({"code": [1, 2, 3, 1, 2], "val": range(5)})
        cat_cols = get_categorical_columns(df, max_unique=3)
        assert "code" in cat_cols
        assert "val" not in cat_cols


class TestDescribeColumns:
    def test_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        desc = describe_columns(sample_df)
        assert isinstance(desc, pd.DataFrame)
        assert "kind" in desc.columns

    def test_correct_kinds(self, sample_df: pd.DataFrame) -> None:
        desc = describe_columns(sample_df)
        assert desc.loc["price", "kind"] == "numerical"
        assert desc.loc["sector", "kind"] == "categorical"


# ---------------------------------------------------------------------------
# distributions tests
# ---------------------------------------------------------------------------

class TestPlotHistogram:
    def test_runs_without_error(self, sample_df: pd.DataFrame, tmp_path) -> None:
        save = str(tmp_path / "hist.png")
        plot_histogram(sample_df, save_path=save)
        assert os.path.exists(save)

    def test_subset_columns(self, sample_df: pd.DataFrame, tmp_path) -> None:
        save = str(tmp_path / "hist_sub.png")
        plot_histogram(sample_df, columns=["price"], save_path=save)
        assert os.path.exists(save)


class TestPlotBoxplot:
    def test_runs_without_error(self, sample_df: pd.DataFrame, tmp_path) -> None:
        save = str(tmp_path / "box.png")
        plot_boxplot(sample_df, save_path=save)
        assert os.path.exists(save)


class TestPlotQQPlot:
    def test_runs_without_error(self, sample_df: pd.DataFrame, tmp_path) -> None:
        save = str(tmp_path / "qq.png")
        plot_qqplot(sample_df, save_path=save)
        assert os.path.exists(save)


class TestPlotAllDistributions:
    def test_runs_without_error(self, sample_df: pd.DataFrame, tmp_path) -> None:
        save = str(tmp_path / "all_dist.png")
        plot_all_distributions(sample_df, save_path=save)
        assert os.path.exists(save)

    def test_single_column(self, sample_df: pd.DataFrame, tmp_path) -> None:
        save = str(tmp_path / "all_dist_single.png")
        plot_all_distributions(sample_df, columns=["price"], save_path=save)
        assert os.path.exists(save)


# ---------------------------------------------------------------------------
# outliers tests
# ---------------------------------------------------------------------------

class TestDetectOutliersIQR:
    def test_flags_known_outliers(self, df_with_outliers: pd.DataFrame) -> None:
        masks = detect_outliers_iqr(df_with_outliers)
        outlier_values = df_with_outliers.loc[masks["value"], "value"]
        assert 50.0 in outlier_values.values
        assert -50.0 in outlier_values.values

    def test_returns_boolean_series(self, sample_df: pd.DataFrame) -> None:
        masks = detect_outliers_iqr(sample_df, columns=["price"])
        assert masks["price"].dtype == bool


class TestDetectOutliersZscore:
    def test_flags_known_outliers(self, df_with_outliers: pd.DataFrame) -> None:
        masks = detect_outliers_zscore(df_with_outliers, threshold=3.0)
        outlier_values = df_with_outliers.loc[masks["value"], "value"]
        assert 50.0 in outlier_values.values


class TestWinsorise:
    def test_clips_values(self, df_with_outliers: pd.DataFrame) -> None:
        winsorised = winsorise(df_with_outliers, lower_pct=0.05, upper_pct=0.95)
        q05 = df_with_outliers["value"].quantile(0.05)
        q95 = df_with_outliers["value"].quantile(0.95)
        assert winsorised["value"].max() <= q95
        assert winsorised["value"].min() >= q05

    def test_invalid_bounds(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Percentile bounds"):
            winsorise(sample_df, lower_pct=0.9, upper_pct=0.1)

    def test_original_unchanged(self, df_with_outliers: pd.DataFrame) -> None:
        original_max = df_with_outliers["value"].max()
        winsorise(df_with_outliers)
        assert df_with_outliers["value"].max() == original_max


class TestOutlierSummary:
    def test_iqr_method(self, df_with_outliers: pd.DataFrame) -> None:
        summary = outlier_summary(df_with_outliers, method="iqr")
        assert "outlier_count" in summary.columns
        assert summary.loc["value", "outlier_count"] >= 2

    def test_zscore_method(self, df_with_outliers: pd.DataFrame) -> None:
        summary = outlier_summary(df_with_outliers, method="zscore")
        assert summary.loc["value", "outlier_count"] >= 2

    def test_invalid_method(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            outlier_summary(sample_df, method="unknown")


# ---------------------------------------------------------------------------
# correlations tests
# ---------------------------------------------------------------------------

class TestPlotCorrelationHeatmap:
    def test_returns_corr_matrix(self, sample_df: pd.DataFrame, tmp_path) -> None:
        save = str(tmp_path / "heatmap.png")
        corr = plot_correlation_heatmap(sample_df, save_path=save)
        assert isinstance(corr, pd.DataFrame)
        assert os.path.exists(save)

    def test_spearman(self, sample_df: pd.DataFrame, tmp_path) -> None:
        save = str(tmp_path / "heatmap_sp.png")
        corr = plot_correlation_heatmap(sample_df, method="spearman", save_path=save)
        assert corr.shape == (3, 3)  # 3 numerical columns


class TestPlotPairplot:
    def test_runs_without_error(self, sample_df: pd.DataFrame, tmp_path) -> None:
        save = str(tmp_path / "pairplot.png")
        plot_pairplot(sample_df, save_path=save)
        assert os.path.exists(save)


class TestGetTopCorrelations:
    def test_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        result = get_top_correlations(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert "col_a" in result.columns
        assert "col_b" in result.columns
        assert "correlation" in result.columns

    def test_n_parameter(self, sample_df: pd.DataFrame) -> None:
        result = get_top_correlations(sample_df, n=2)
        assert len(result) <= 2

    def test_all_correlations_within_bounds(self, sample_df: pd.DataFrame) -> None:
        result = get_top_correlations(sample_df)
        assert (result["correlation"].abs() <= 1.0).all()
