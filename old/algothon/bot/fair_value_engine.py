"""
fair_value_engine.py — Computes theoretical fair values for all 8 products.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_engine import DataEngine

log = logging.getLogger("algothon.fv")


class FairValueEngine:
    """Computes settlement estimates from DataEngine state."""

    def __init__(self, data: "DataEngine"):
        self.data = data
        self.fair_values: dict[str, float] = {}
        self.confidence: dict[str, float] = {}

    def update_all(self, session_start: datetime, session_end: datetime):
        """Recompute all fair values. Call after data refresh."""
        self._update_tide_spot(session_end)
        self._update_tide_swing(session_start, session_end)
        self._update_wx_spot(session_end)
        self._update_wx_sum(session_start, session_end)
        self._update_lhr_count(session_start, session_end)
        self._update_lhr_index(session_start, session_end)
        self._update_lon_etf()
        self._update_lon_fly()
        return self.fair_values

    # ── TIDE_SPOT ───────────────────────────────────────────────────────────

    def _update_tide_spot(self, session_end: datetime):
        df = self.data.thames
        if df is None or df.empty:
            self.confidence["TIDE_SPOT"] = 0.0
            return
        predicted_level = self._extrapolate_tide(df, session_end)
        self.fair_values["TIDE_SPOT"] = abs(predicted_level) * 1000
        self.confidence["TIDE_SPOT"] = self._tide_confidence(df, session_end)

    def _extrapolate_tide(self, df: pd.DataFrame, target: datetime) -> float:
        """Sinusoidal extrapolation (semi-diurnal, T=12.42h)."""
        if len(df) < 10:
            return df["level"].iloc[-1]

        t0 = df["time"].iloc[0]
        hours = (df["time"] - t0).dt.total_seconds() / 3600.0
        levels = df["level"].values
        T = 12.42
        omega = 2 * np.pi / T

        A = np.column_stack([
            np.ones(len(hours)),
            np.sin(omega * hours),
            np.cos(omega * hours),
        ])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, levels, rcond=None)
        except Exception:
            return df["level"].iloc[-1]

        target_hours = (target - t0).total_seconds() / 3600.0
        return float(
            coeffs[0]
            + coeffs[1] * np.sin(omega * target_hours)
            + coeffs[2] * np.cos(omega * target_hours)
        )

    def _tide_confidence(self, df: pd.DataFrame, target: datetime) -> float:
        if df.empty:
            return 0.0
        hours_to_target = abs((target - df["time"].iloc[-1]).total_seconds()) / 3600
        return max(0.2, 0.9 - 0.03 * hours_to_target)

    # ── TIDE_SWING ──────────────────────────────────────────────────────────

    def _update_tide_swing(self, session_start: datetime, session_end: datetime):
        df = self.data.thames
        if df is None or len(df) < 2:
            self.confidence["TIDE_SWING"] = 0.0
            return

        mask = (df["time"] >= session_start) & (df["time"] <= session_end)
        session_df = df[mask].copy()

        if len(session_df) < 2:
            self._estimate_tide_swing_from_history(df, session_start, session_end)
            return

        diffs_cm = session_df["level"].diff().abs() * 100
        diffs_cm = diffs_cm.dropna()

        observed_payoff = sum(self._strangle_payoff(d) for d in diffs_cm)
        total_intervals = 96
        observed_intervals = len(diffs_cm)
        remaining = max(0, total_intervals - observed_intervals)

        avg_payoff = (
            observed_payoff / observed_intervals if observed_intervals > 0
            else self._historical_avg_tide_swing_payoff(df)
        )

        self.fair_values["TIDE_SWING"] = observed_payoff + avg_payoff * remaining
        self.confidence["TIDE_SWING"] = min(0.9, 0.3 + 0.6 * (observed_intervals / total_intervals))

    def _strangle_payoff(self, diff_cm: float) -> float:
        return max(0, 20 - diff_cm) + max(0, diff_cm - 25)

    def _estimate_tide_swing_from_history(self, df, session_start, session_end):
        diffs_cm = df["level"].diff().abs() * 100
        diffs_cm = diffs_cm.dropna()
        if len(diffs_cm) == 0:
            self.confidence["TIDE_SWING"] = 0.0
            return
        avg = np.mean([self._strangle_payoff(d) for d in diffs_cm])
        self.fair_values["TIDE_SWING"] = avg * 96
        self.confidence["TIDE_SWING"] = 0.3

    def _historical_avg_tide_swing_payoff(self, df) -> float:
        diffs_cm = df["level"].diff().abs() * 100
        diffs_cm = diffs_cm.dropna()
        return np.mean([self._strangle_payoff(d) for d in diffs_cm]) if len(diffs_cm) > 0 else 5.0

    # ── WX_SPOT ─────────────────────────────────────────────────────────────

    def _update_wx_spot(self, session_end: datetime):
        df = self.data.weather
        if df is None or df.empty:
            self.confidence["WX_SPOT"] = 0.0
            return

        target_ts = pd.Timestamp(session_end)
        if target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize("Europe/London")

        idx = (df["time"] - target_ts).abs().idxmin()
        row = df.iloc[idx]

        self.fair_values["WX_SPOT"] = row["wx_metric"]
        hours_away = abs((row["time"] - target_ts).total_seconds()) / 3600
        self.confidence["WX_SPOT"] = max(0.3, 0.95 - 0.04 * hours_away)

    # ── WX_SUM ──────────────────────────────────────────────────────────────

    def _update_wx_sum(self, session_start: datetime, session_end: datetime):
        df = self.data.weather
        if df is None or df.empty:
            self.confidence["WX_SUM"] = 0.0
            return

        start_ts = pd.Timestamp(session_start)
        end_ts = pd.Timestamp(session_end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("Europe/London")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("Europe/London")

        now = pd.Timestamp.now("Europe/London")
        mask_observed = (df["time"] >= start_ts) & (df["time"] <= now)
        mask_forecast = (df["time"] > now) & (df["time"] <= end_ts)

        observed_sum = df.loc[mask_observed, "wx_metric"].sum()
        forecast_sum = df.loc[mask_forecast, "wx_metric"].sum()

        self.fair_values["WX_SUM"] = (observed_sum + forecast_sum) / 100
        n_obs = mask_observed.sum()
        self.confidence["WX_SUM"] = min(0.9, 0.4 + 0.5 * (n_obs / 96))

    # ── LHR_COUNT ───────────────────────────────────────────────────────────

    def _update_lhr_count(self, session_start: datetime, session_end: datetime):
        flights = self.data.flights
        if flights is None:
            self.confidence["LHR_COUNT"] = 0.0
            return
        n_arr = len(flights.get("arrivals", []))
        n_dep = len(flights.get("departures", []))
        self.fair_values["LHR_COUNT"] = n_arr + n_dep
        self.confidence["LHR_COUNT"] = 0.5

    # ── LHR_INDEX ───────────────────────────────────────────────────────────

    def _update_lhr_index(self, session_start: datetime, session_end: datetime):
        flights = self.data.flights
        if flights is None:
            self.confidence["LHR_INDEX"] = 0.0
            return
        try:
            arr = len(flights.get("arrivals", []))
            dep = len(flights.get("departures", []))
            if arr + dep > 0:
                imbalance = abs(arr - dep) / max(arr + dep, 1)
                self.fair_values["LHR_INDEX"] = imbalance * 48 * 100
            else:
                self.fair_values["LHR_INDEX"] = 0
            self.confidence["LHR_INDEX"] = 0.3
        except Exception:
            self.confidence["LHR_INDEX"] = 0.0

    # ── LON_ETF ─────────────────────────────────────────────────────────────

    def _update_lon_etf(self):
        components = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        if all(c in self.fair_values for c in components):
            self.fair_values["LON_ETF"] = sum(self.fair_values[c] for c in components)
            self.confidence["LON_ETF"] = min(self.confidence.get(c, 0) for c in components)
        else:
            self.confidence["LON_ETF"] = 0.0

    # ── LON_FLY ─────────────────────────────────────────────────────────────

    def _update_lon_fly(self):
        if "LON_ETF" not in self.fair_values:
            self.confidence["LON_FLY"] = 0.0
            return

        etf = self.fair_values["LON_ETF"]
        etf_conf = self.confidence.get("LON_ETF", 0)

        if etf_conf > 0.7:
            self.fair_values["LON_FLY"] = self._fly_intrinsic(etf)
            self.confidence["LON_FLY"] = etf_conf * 0.9
        else:
            self.fair_values["LON_FLY"] = self._fly_expected_value(etf)
            self.confidence["LON_FLY"] = etf_conf * 0.6

    def _fly_intrinsic(self, etf: float) -> float:
        put = lambda k, s: max(0, k - s)
        call = lambda k, s: max(0, s - k)
        return (
            2 * put(6200, etf)
            + 1 * call(6200, etf)
            - 2 * call(6600, etf)
            + 3 * call(7000, etf)
        )

    def _fly_expected_value(self, etf_mean: float, etf_std: float = 300) -> float:
        """Monte Carlo expected value of the options package."""
        n_sims = 10_000
        samples = np.random.normal(etf_mean, etf_std, n_sims)
        payoffs = np.array([self._fly_intrinsic(s) for s in samples])
        return float(np.mean(payoffs))
