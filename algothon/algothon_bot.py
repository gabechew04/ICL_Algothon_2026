"""
IMCity Algothon — Full Trading System
======================================

Architecture:
  1. DataEngine       — fetches & caches weather, tides, flights
  2. FairValueEngine  — computes theo for all 8 products
  3. RiskEngine       — tracks positions, PnL, exposure limits
  4. MarketMaker      — quotes around fair value with inventory skew
  5. ArbEngine        — ETF vs components arbitrage
  6. OptionsEngine    — prices LON_FLY from ETF distribution
  7. AlgothonBot      — main orchestrator (extends BaseBot)

Usage:
  bot = AlgothonBot(EXCHANGE_URL, USERNAME, PASSWORD, AERODATABOX_KEY)
  bot.run()
"""

import json
import math
import time
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from bot_template import (
    BaseBot, OrderBook, Order, OrderRequest, OrderResponse,
    Trade, Side, Product,
)

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("algothon")


# =============================================================================
# 1. DATA ENGINE — Fetch & cache real-world data
# =============================================================================

LONDON_LAT, LONDON_LON = 51.5074, -0.1278
THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"


class DataEngine:
    """Fetches and caches weather, tidal, and flight data with rate-limiting."""

    def __init__(self, aero_key: str = "", min_interval: float = 60.0):
        self.aero_key = aero_key
        self.min_interval = min_interval  # seconds between fetches per source

        # Cached DataFrames
        self.weather: Optional[pd.DataFrame] = None
        self.thames: Optional[pd.DataFrame] = None
        self.flights: Optional[dict] = None

        # Timestamps of last successful fetch
        self._last_fetch = {"weather": 0.0, "thames": 0.0, "flights": 0.0}

    def _should_fetch(self, source: str) -> bool:
        return time.time() - self._last_fetch[source] > self.min_interval

    # ── Weather (Open-Meteo, free) ──────────────────────────────────────────

    def fetch_weather(self, force: bool = False) -> Optional[pd.DataFrame]:
        if not force and not self._should_fetch("weather"):
            return self.weather
        try:
            variables = (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "precipitation,wind_speed_10m,cloud_cover,visibility"
            )
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": LONDON_LAT, "longitude": LONDON_LON,
                    "minutely_15": variables,
                    "past_minutely_15": 96,   # 24h back
                    "forecast_minutely_15": 96,  # 24h forward
                    "timezone": "Europe/London",
                },
                timeout=10,
            )
            resp.raise_for_status()
            m = resp.json()["minutely_15"]
            self.weather = pd.DataFrame({
                "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
                "temperature_c": m["temperature_2m"],
                "humidity": m["relative_humidity_2m"],
                "precipitation": m["precipitation"],
                "wind_speed": m["wind_speed_10m"],
                "cloud_cover": m["cloud_cover"],
                "visibility": m["visibility"],
                "apparent_temperature": m["apparent_temperature"],
            })
            # Pre-compute Fahrenheit & spot metric
            self.weather["temperature_f"] = self.weather["temperature_c"] * 9 / 5 + 32
            self.weather["wx_metric"] = (
                self.weather["temperature_f"] * self.weather["humidity"]
            )
            self._last_fetch["weather"] = time.time()
            log.info(f"Weather: {len(self.weather)} rows fetched")
        except Exception as e:
            log.warning(f"Weather fetch failed: {e}")
        return self.weather

    # ── Thames Tidal (EA Flood Monitoring, free) ────────────────────────────

    def fetch_thames(self, limit: int = 200, force: bool = False) -> Optional[pd.DataFrame]:
        if not force and not self._should_fetch("thames"):
            return self.thames
        try:
            resp = requests.get(
                f"https://environment.data.gov.uk/flood-monitoring/id/measures/"
                f"{THAMES_MEASURE}/readings",
                params={"_sorted": "", "_limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            df = pd.DataFrame(items)[["dateTime", "value"]].rename(
                columns={"dateTime": "time", "value": "level"}
            )
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
            self.thames = df.sort_values("time").reset_index(drop=True)
            self._last_fetch["thames"] = time.time()
            log.info(f"Thames: {len(self.thames)} readings fetched")
        except Exception as e:
            log.warning(f"Thames fetch failed: {e}")
        return self.thames

    # ── Flights (AeroDataBox via RapidAPI) ──────────────────────────────────

    def fetch_flights(self, offset_minutes: int = -720, duration_minutes: int = 720,
                      force: bool = False) -> Optional[dict]:
        """Fetch flights. Use sparingly — free tier is ~150 req/month."""
        if not force and not self._should_fetch("flights"):
            return self.flights
        if not self.aero_key:
            log.warning("No AeroDataBox API key set — skipping flights")
            return self.flights
        try:
            host = "aerodatabox.p.rapidapi.com"
            url = (
                f"https://{host}/flights/airports/iata/LHR"
                f"?offsetMinutes={offset_minutes}"
                f"&durationMinutes={duration_minutes}&direction=Both"
            )
            resp = requests.get(url, headers={
                "x-rapidapi-host": host, "x-rapidapi-key": self.aero_key,
            }, timeout=15)
            resp.raise_for_status()
            self.flights = resp.json()
            n_arr = len(self.flights.get("arrivals", []))
            n_dep = len(self.flights.get("departures", []))
            self._last_fetch["flights"] = time.time()
            log.info(f"Flights: {n_arr} arrivals, {n_dep} departures")
        except Exception as e:
            log.warning(f"Flights fetch failed: {e}")
        return self.flights

    def fetch_all(self, force: bool = False):
        self.fetch_weather(force=force)
        self.fetch_thames(force=force)
        self.fetch_flights(force=force)


# =============================================================================
# 2. FAIR VALUE ENGINE — Compute theoretical values for all 8 products
# =============================================================================

class FairValueEngine:
    """Computes settlement estimates from DataEngine state."""

    def __init__(self, data: DataEngine):
        self.data = data
        self.fair_values: dict[str, float] = {}
        self.confidence: dict[str, float] = {}  # 0=no idea, 1=high confidence

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

    # ── TIDE_SPOT: abs(level at settlement) × 1000 ─────────────────────────

    def _update_tide_spot(self, session_end: datetime):
        df = self.data.thames
        if df is None or df.empty:
            self.confidence["TIDE_SPOT"] = 0.0
            return

        # Use latest reading as naive forecast (tidal patterns are predictable)
        # Better: fit sinusoidal model to predict level at session_end
        latest_level = df["level"].iloc[-1]
        latest_time = df["time"].iloc[-1]

        # Simple approach: use tidal harmonic extrapolation
        predicted_level = self._extrapolate_tide(df, session_end)
        self.fair_values["TIDE_SPOT"] = abs(predicted_level) * 1000
        self.confidence["TIDE_SPOT"] = self._tide_confidence(df, session_end)

    def _extrapolate_tide(self, df: pd.DataFrame, target: datetime) -> float:
        """Sinusoidal extrapolation of tidal level.

        Thames tides are predominantly semi-diurnal (~12.42h period).
        We fit: level = a + b*sin(2π/T * t) + c*cos(2π/T * t)
        """
        if len(df) < 10:
            return df["level"].iloc[-1]

        # Convert to hours from first reading
        t0 = df["time"].iloc[0]
        hours = (df["time"] - t0).dt.total_seconds() / 3600.0
        levels = df["level"].values

        T = 12.42  # semi-diurnal tidal period in hours
        omega = 2 * np.pi / T

        # Design matrix: [1, sin(ωt), cos(ωt)]
        A = np.column_stack([
            np.ones(len(hours)),
            np.sin(omega * hours),
            np.cos(omega * hours),
        ])

        # Least squares fit
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, levels, rcond=None)
        except Exception:
            return df["level"].iloc[-1]

        # Predict at target time
        target_hours = (target - t0).total_seconds() / 3600.0
        predicted = (
            coeffs[0]
            + coeffs[1] * np.sin(omega * target_hours)
            + coeffs[2] * np.cos(omega * target_hours)
        )
        return float(predicted)

    def _tide_confidence(self, df: pd.DataFrame, target: datetime) -> float:
        """Higher confidence when we have recent data and target is closer."""
        if df.empty:
            return 0.0
        hours_to_target = abs((target - df["time"].iloc[-1]).total_seconds()) / 3600
        # Decays from 0.9 (0h away) to 0.2 (24h away)
        return max(0.2, 0.9 - 0.03 * hours_to_target)

    # ── TIDE_SWING: sum of strangle payoffs on 15-min diffs (cm) ───────────

    def _update_tide_swing(self, session_start: datetime, session_end: datetime):
        df = self.data.thames
        if df is None or len(df) < 2:
            self.confidence["TIDE_SWING"] = 0.0
            return

        # Filter to session window
        mask = (df["time"] >= session_start) & (df["time"] <= session_end)
        session_df = df[mask].copy()

        if len(session_df) < 2:
            # Not enough session data — estimate from historical patterns
            self._estimate_tide_swing_from_history(df, session_start, session_end)
            return

        # Compute observed payoffs so far
        diffs_cm = session_df["level"].diff().abs() * 100  # mAOD -> cm
        diffs_cm = diffs_cm.dropna()

        observed_payoff = sum(self._strangle_payoff(d) for d in diffs_cm)

        # Estimate remaining intervals
        total_intervals = 96  # 24h * 4 per hour
        observed_intervals = len(diffs_cm)
        remaining = max(0, total_intervals - observed_intervals)

        # Average payoff per interval from observed data
        if observed_intervals > 0:
            avg_payoff = observed_payoff / observed_intervals
        else:
            avg_payoff = self._historical_avg_tide_swing_payoff(df)

        estimated_total = observed_payoff + avg_payoff * remaining
        self.fair_values["TIDE_SWING"] = estimated_total
        self.confidence["TIDE_SWING"] = min(0.9, 0.3 + 0.6 * (observed_intervals / total_intervals))

    def _strangle_payoff(self, diff_cm: float) -> float:
        """Strangle with strikes 20 and 25."""
        put = max(0, 20 - diff_cm)
        call = max(0, diff_cm - 25)
        return put + call

    def _estimate_tide_swing_from_history(self, df, session_start, session_end):
        """Use historical diffs to estimate swing settlement."""
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
        if len(diffs_cm) == 0:
            return 5.0  # fallback
        return np.mean([self._strangle_payoff(d) for d in diffs_cm])

    # ── WX_SPOT: temperature_F × humidity at settlement ────────────────────

    def _update_wx_spot(self, session_end: datetime):
        df = self.data.weather
        if df is None or df.empty:
            self.confidence["WX_SPOT"] = 0.0
            return

        # Find forecast closest to session_end
        target_ts = pd.Timestamp(session_end)
        if target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize("Europe/London")

        idx = (df["time"] - target_ts).abs().idxmin()
        row = df.iloc[idx]

        self.fair_values["WX_SPOT"] = row["wx_metric"]

        # Confidence based on distance to target
        hours_away = abs((row["time"] - target_ts).total_seconds()) / 3600
        self.confidence["WX_SPOT"] = max(0.3, 0.95 - 0.04 * hours_away)

    # ── WX_SUM: sum(temp_F × humidity) / 100 over session ──────────────────

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

        # Split into observed and forecast
        mask_observed = (df["time"] >= start_ts) & (df["time"] <= pd.Timestamp.now("Europe/London"))
        mask_forecast = (df["time"] > pd.Timestamp.now("Europe/London")) & (df["time"] <= end_ts)

        observed_sum = df.loc[mask_observed, "wx_metric"].sum()
        forecast_sum = df.loc[mask_forecast, "wx_metric"].sum()

        total = (observed_sum + forecast_sum) / 100
        self.fair_values["WX_SUM"] = total

        n_obs = mask_observed.sum()
        n_total = 96
        self.confidence["WX_SUM"] = min(0.9, 0.4 + 0.5 * (n_obs / max(n_total, 1)))

    # ── LHR_COUNT: total arrivals + departures ─────────────────────────────

    def _update_lhr_count(self, session_start: datetime, session_end: datetime):
        flights = self.data.flights
        if flights is None:
            self.confidence["LHR_COUNT"] = 0.0
            return

        n_arr = len(flights.get("arrivals", []))
        n_dep = len(flights.get("departures", []))
        observed_total = n_arr + n_dep

        # Estimate: scale up if we only fetched a partial window
        # For a full 24h, typical Heathrow count is ~1200-1400 flights
        # If our API call covered 12h, we extrapolate
        # Better: use flight schedule data
        self.fair_values["LHR_COUNT"] = observed_total
        self.confidence["LHR_COUNT"] = 0.5  # medium, depends on coverage

    # ── LHR_INDEX: abs(sum of imbalance metrics) ──────────────────────────

    def _update_lhr_index(self, session_start: datetime, session_end: datetime):
        flights = self.data.flights
        if flights is None:
            self.confidence["LHR_INDEX"] = 0.0
            return

        # Parse flight times and compute per-30min imbalance
        # This requires binning arrivals/departures by time bucket
        try:
            arrivals = flights.get("arrivals", [])
            departures = flights.get("departures", [])

            # Count by 30-min buckets (simplified)
            arr_count = len(arrivals)
            dep_count = len(departures)

            if arr_count + dep_count > 0:
                # Rough estimate: overall imbalance scaled
                overall_imbalance = abs(arr_count - dep_count) / max(arr_count + dep_count, 1)
                # The actual metric sums per 30-min bucket ratios
                # Rough approximation: N_buckets × average_imbalance × 100
                n_buckets = 48  # 24h / 30min
                self.fair_values["LHR_INDEX"] = overall_imbalance * n_buckets * 100
            else:
                self.fair_values["LHR_INDEX"] = 0

            self.confidence["LHR_INDEX"] = 0.3
        except Exception:
            self.confidence["LHR_INDEX"] = 0.0

    # ── LON_ETF: TIDE_SPOT + WX_SPOT + LHR_COUNT ──────────────────────────

    def _update_lon_etf(self):
        components = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        if all(c in self.fair_values for c in components):
            self.fair_values["LON_ETF"] = sum(self.fair_values[c] for c in components)
            self.confidence["LON_ETF"] = min(self.confidence.get(c, 0) for c in components)
        else:
            self.confidence["LON_ETF"] = 0.0

    # ── LON_FLY: options structure on ETF ──────────────────────────────────

    def _update_lon_fly(self):
        if "LON_ETF" not in self.fair_values:
            self.confidence["LON_FLY"] = 0.0
            return

        etf = self.fair_values["LON_ETF"]
        etf_conf = self.confidence.get("LON_ETF", 0)

        # At high confidence (near settlement), use intrinsic value
        # At low confidence, we need to model ETF distribution
        if etf_conf > 0.7:
            # Near-settlement: use intrinsic value
            self.fair_values["LON_FLY"] = self._fly_intrinsic(etf)
            self.confidence["LON_FLY"] = etf_conf * 0.9
        else:
            # Earlier in session: Monte Carlo or analytical with ETF volatility
            self.fair_values["LON_FLY"] = self._fly_expected_value(etf)
            self.confidence["LON_FLY"] = etf_conf * 0.6

    def _fly_intrinsic(self, etf: float) -> float:
        """2×Put(6200) + Call(6200) − 2×Call(6600) + 3×Call(7000)"""
        put = lambda k, s: max(0, k - s)
        call = lambda k, s: max(0, s - k)
        return (
            2 * put(6200, etf)
            + 1 * call(6200, etf)
            - 2 * call(6600, etf)
            + 3 * call(7000, etf)
        )

    def _fly_expected_value(self, etf_mean: float, etf_std: float = 300) -> float:
        """Monte Carlo expected value of the options package.

        Uses a normal distribution around the ETF fair value estimate.
        The std should decrease as we approach settlement.
        """
        n_sims = 10000
        etf_samples = np.random.normal(etf_mean, etf_std, n_sims)
        payoffs = np.array([self._fly_intrinsic(s) for s in etf_samples])
        return float(np.mean(payoffs))


# =============================================================================
# 3. RISK ENGINE — Position tracking, limits, and exposure management
# =============================================================================

@dataclass
class RiskState:
    positions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    max_position: int = 100
    # Per-product risk parameters
    max_notional_per_product: float = float("inf")
    # Track fills for PnL estimation
    fill_history: list[Trade] = field(default_factory=list)
    realized_pnl: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    avg_entry: dict[str, float] = field(default_factory=lambda: defaultdict(float))


class RiskEngine:
    """Tracks positions, enforces limits, computes inventory skew."""

    def __init__(self, max_position: int = 100):
        self.state = RiskState(max_position=max_position)
        self._lock = threading.Lock()

    def update_positions(self, positions: dict[str, int]):
        """Sync positions from exchange."""
        with self._lock:
            self.state.positions = defaultdict(int, positions)

    def record_fill(self, trade: Trade, our_side: Side):
        """Record a fill and update average entry price."""
        with self._lock:
            self.state.fill_history.append(trade)
            product = trade.product
            pos = self.state.positions[product]
            qty = trade.volume if our_side == Side.BUY else -trade.volume

            old_pos = pos
            new_pos = pos + qty
            self.state.positions[product] = new_pos

            # Update average entry for open position
            if abs(new_pos) > abs(old_pos):
                # Adding to position
                old_cost = self.state.avg_entry[product] * abs(old_pos)
                new_cost = trade.price * abs(qty)
                if new_pos != 0:
                    self.state.avg_entry[product] = (old_cost + new_cost) / abs(new_pos)

    def get_position(self, product: str) -> int:
        return self.state.positions.get(product, 0)

    def can_buy(self, product: str, volume: int) -> int:
        """Returns max volume we can buy (capped by position limit)."""
        pos = self.get_position(product)
        return max(0, min(volume, self.state.max_position - pos))

    def can_sell(self, product: str, volume: int) -> int:
        """Returns max volume we can sell (capped by position limit)."""
        pos = self.get_position(product)
        return max(0, min(volume, self.state.max_position + pos))

    def inventory_skew(self, product: str) -> float:
        """Returns skew factor [-1, 1]. Positive = long, should sell more aggressively.

        Used to bias quotes: when long, lower asks; when short, raise bids.
        """
        pos = self.get_position(product)
        return pos / self.state.max_position

    def quote_size(self, product: str, base_size: int) -> tuple[int, int]:
        """Returns (bid_size, ask_size) adjusted for inventory.

        When heavily long: reduce bid size, increase ask size.
        When heavily short: increase bid size, reduce ask size.
        """
        skew = self.inventory_skew(product)

        # Linear scaling: at max position, quote 0 on one side
        bid_size = max(1, int(base_size * (1 - max(0, skew))))
        ask_size = max(1, int(base_size * (1 + min(0, skew))))

        bid_size = self.can_buy(product, bid_size)
        ask_size = self.can_sell(product, ask_size)

        return bid_size, ask_size


# =============================================================================
# 4. MARKET MAKER — Quotes around fair value with inventory management
# =============================================================================

class MarketMaker:
    """Generates two-sided quotes around fair value with adaptive width."""

    def __init__(self, risk: RiskEngine, fv_engine: FairValueEngine):
        self.risk = risk
        self.fv = fv_engine

        # Configuration per product (can be tuned)
        self.config = {
            # product: (base_width, base_size, min_edge, max_levels)
            "TIDE_SPOT":  (15, 5, 3, 2),
            "TIDE_SWING": (10, 3, 2, 2),
            "WX_SPOT":    (15, 5, 3, 2),
            "WX_SUM":     (10, 3, 2, 2),
            "LHR_COUNT":  (20, 5, 5, 2),
            "LHR_INDEX":  (15, 3, 3, 2),
            "LON_ETF":    (25, 5, 5, 3),
            "LON_FLY":    (20, 3, 5, 2),
        }

    def generate_quotes(self, product: str, tick_size: float,
                        market_mid: Optional[float] = None) -> list[OrderRequest]:
        """Generate bid/ask orders for a product."""
        fv = self.fv.fair_values.get(product)
        conf = self.fv.confidence.get(product, 0)

        if fv is None or conf < 0.1:
            # No fair value — fall back to market mid if available
            if market_mid is not None:
                fv = market_mid
                conf = 0.2
            else:
                return []

        cfg = self.config.get(product, (15, 5, 3, 2))
        base_width, base_size, min_edge, max_levels = cfg

        # Adaptive width: wider when less confident
        width = base_width / max(conf, 0.15)
        width = max(width, min_edge * 2)

        # Inventory skew: shift quotes toward reducing position
        skew = self.risk.inventory_skew(product)
        skew_offset = skew * width * 0.3  # shift mid by up to 30% of width

        adjusted_fv = fv - skew_offset  # when long, lower effective FV → cheaper asks

        # Generate levels
        orders = []
        bid_size, ask_size = self.risk.quote_size(product, base_size)

        for level in range(max_levels):
            level_offset = width * (0.5 + level * 0.5)

            bid_price = math.floor((adjusted_fv - level_offset) / tick_size) * tick_size
            ask_price = math.ceil((adjusted_fv + level_offset) / tick_size) * tick_size

            # Size decreases at outer levels
            lvl_bid_size = max(1, bid_size // (level + 1))
            lvl_ask_size = max(1, ask_size // (level + 1))

            if bid_price > 0 and lvl_bid_size > 0:
                orders.append(OrderRequest(product, bid_price, Side.BUY, lvl_bid_size))
            if ask_price > bid_price and lvl_ask_size > 0:
                orders.append(OrderRequest(product, ask_price, Side.SELL, lvl_ask_size))

        return orders


# =============================================================================
# 5. ETF ARBITRAGE ENGINE — LON_ETF vs components
# =============================================================================

class ArbEngine:
    """Detects and trades ETF vs component mispricings.

    LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT

    If ETF trades cheap vs sum of components → buy ETF, sell components
    If ETF trades rich vs sum of components → sell ETF, buy components
    """

    def __init__(self, risk: RiskEngine, min_edge: float = 10.0):
        self.risk = risk
        self.min_edge = min_edge  # minimum mispricing to trade

    def check_arb(self, orderbooks: dict[str, OrderBook]) -> list[OrderRequest]:
        """Check for ETF arbitrage opportunities."""
        components = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        etf_sym = "LON_ETF"

        # Need all orderbooks
        if etf_sym not in orderbooks:
            return []
        for c in components:
            if c not in orderbooks:
                return []

        etf_ob = orderbooks[etf_sym]
        comp_obs = {c: orderbooks[c] for c in components}

        orders = []

        # Check: can we buy components cheap and sell ETF rich?
        comp_ask_total = 0
        comp_asks_valid = True
        for c in components:
            ob = comp_obs[c]
            if ob.sell_orders:
                comp_ask_total += ob.sell_orders[0].price
            else:
                comp_asks_valid = False

        if comp_asks_valid and etf_ob.buy_orders:
            etf_bid = etf_ob.buy_orders[0].price
            edge = etf_bid - comp_ask_total  # positive = ETF is expensive

            if edge > self.min_edge:
                # Sell ETF, buy components
                vol = min(3,
                          self.risk.can_sell(etf_sym, 3),
                          *[self.risk.can_buy(c, 3) for c in components])
                if vol > 0:
                    orders.append(OrderRequest(etf_sym, etf_bid, Side.SELL, vol))
                    for c in components:
                        orders.append(OrderRequest(c, comp_obs[c].sell_orders[0].price, Side.BUY, vol))
                    log.info(f"ARB: Sell ETF @ {etf_bid}, Buy components. Edge={edge:.0f}")

        # Check: can we buy ETF cheap and sell components rich?
        comp_bid_total = 0
        comp_bids_valid = True
        for c in components:
            ob = comp_obs[c]
            if ob.buy_orders:
                comp_bid_total += ob.buy_orders[0].price
            else:
                comp_bids_valid = False

        if comp_bids_valid and etf_ob.sell_orders:
            etf_ask = etf_ob.sell_orders[0].price
            edge = comp_bid_total - etf_ask  # positive = components expensive

            if edge > self.min_edge:
                # Buy ETF, sell components
                vol = min(3,
                          self.risk.can_buy(etf_sym, 3),
                          *[self.risk.can_sell(c, 3) for c in components])
                if vol > 0:
                    orders.append(OrderRequest(etf_sym, etf_ask, Side.BUY, vol))
                    for c in components:
                        orders.append(OrderRequest(c, comp_obs[c].buy_orders[0].price, Side.SELL, vol))
                    log.info(f"ARB: Buy ETF @ {etf_ask}, Sell components. Edge={edge:.0f}")

        return orders


# =============================================================================
# 6. OPTIONS ENGINE — Price LON_FLY from ETF estimates
# =============================================================================

class OptionsEngine:
    """Prices LON_FLY options structure and detects mispricing vs market.

    Structure: 2×Put(6200) + Call(6200) − 2×Call(6600) + 3×Call(7000)
    """

    STRIKES = [6200, 6600, 7000]

    def __init__(self, fv_engine: FairValueEngine, risk: RiskEngine):
        self.fv = fv_engine
        self.risk = risk

    def price_fly(self, etf_mean: float, etf_std: float = 300,
                  n_sims: int = 20000) -> float:
        """Monte Carlo price of the options package."""
        samples = np.random.normal(etf_mean, etf_std, n_sims)
        # Ensure non-negative (all settlements are non-negative)
        samples = np.maximum(samples, 0)
        payoffs = np.array([self._payoff(s) for s in samples])
        return float(np.mean(payoffs))

    def _payoff(self, etf: float) -> float:
        put = lambda k: max(0, k - etf)
        call = lambda k: max(0, etf - k)
        return (
            2 * put(6200)
            + 1 * call(6200)
            - 2 * call(6600)
            + 3 * call(7000)
        )

    def check_mispricing(self, fly_ob: OrderBook) -> list[OrderRequest]:
        """Check if LON_FLY market price diverges from our model."""
        etf_fv = self.fv.fair_values.get("LON_ETF")
        if etf_fv is None:
            return []

        etf_conf = self.fv.confidence.get("LON_ETF", 0)
        # ETF std decreases as session progresses
        etf_std = 300 * (1 - etf_conf)
        etf_std = max(etf_std, 50)

        model_price = self.price_fly(etf_fv, etf_std)
        orders = []

        min_edge = 15

        if fly_ob.sell_orders:
            best_ask = fly_ob.sell_orders[0].price
            if model_price - best_ask > min_edge:
                vol = self.risk.can_buy("LON_FLY", 3)
                if vol > 0:
                    orders.append(OrderRequest("LON_FLY", best_ask, Side.BUY, vol))
                    log.info(f"OPT: Buy FLY @ {best_ask}, model={model_price:.0f}")

        if fly_ob.buy_orders:
            best_bid = fly_ob.buy_orders[0].price
            if best_bid - model_price > min_edge:
                vol = self.risk.can_sell("LON_FLY", 3)
                if vol > 0:
                    orders.append(OrderRequest("LON_FLY", best_bid, Side.SELL, vol))
                    log.info(f"OPT: Sell FLY @ {best_bid}, model={model_price:.0f}")

        return orders


# =============================================================================
# 7. MAIN BOT — Orchestrates everything
# =============================================================================

class AlgothonBot(BaseBot):
    """Full trading bot for the IMCity Algothon.

    Main loop (every N seconds):
      1. Fetch data
      2. Compute fair values
      3. Cancel stale orders
      4. Generate market-making quotes
      5. Check ETF arbitrage
      6. Check options mispricing
      7. Send orders (rate-limited)
    """

    def __init__(self, cmi_url: str, username: str, password: str,
                 aero_key: str = "",
                 session_start: Optional[datetime] = None,
                 session_end: Optional[datetime] = None,
                 loop_interval: float = 5.0,
                 data_interval: float = 120.0):  # fetch data every 2 min
        super().__init__(cmi_url, username, password)

        # Session window (defaults to standard Sat 12pm → Sun 12pm London)
        london_tz = timezone(timedelta(hours=0))  # Adjust for BST if needed
        if session_start is None:
            # Default: this Saturday 12pm London
            now = datetime.now(tz=london_tz)
            session_start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if session_end is None:
            session_end = session_start + timedelta(hours=24)

        self.session_start = session_start
        self.session_end = session_end
        self.loop_interval = max(loop_interval, 2.0)  # minimum 2s to avoid rate limits
        self.data_interval = data_interval

        # Sub-engines
        self.data_engine = DataEngine(aero_key=aero_key, min_interval=data_interval)
        self.fv_engine = FairValueEngine(self.data_engine)
        self.risk = RiskEngine(max_position=100)
        self.mm = MarketMaker(self.risk, self.fv_engine)
        self.arb = ArbEngine(self.risk, min_edge=10)
        self.options = OptionsEngine(self.fv_engine, self.risk)

        # State
        self._orderbooks: dict[str, OrderBook] = {}
        self._products: dict[str, Product] = {}
        self._last_data_fetch: float = 0
        self._running = False
        self._order_count = 0

    # ── SSE Callbacks ───────────────────────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook):
        """Called on every orderbook update via SSE."""
        self._orderbooks[orderbook.product] = orderbook

    def on_trades(self, trade: Trade):
        """Called when WE trade."""
        if trade.buyer == self.username:
            side = Side.BUY
            log.info(f"FILL: BOUGHT {trade.volume}x {trade.product} @ {trade.price}")
        else:
            side = Side.SELL
            log.info(f"FILL: SOLD {trade.volume}x {trade.product} @ {trade.price}")
        self.risk.record_fill(trade, side)

    # ── Main Loop ───────────────────────────────────────────────────────────

    def run(self):
        """Start the bot and run the main trading loop."""
        log.info("═" * 60)
        log.info("  ALGOTHON BOT STARTING")
        log.info(f"  Session: {self.session_start} → {self.session_end}")
        log.info("═" * 60)

        # Discover products
        self._products = {p.symbol: p for p in self.get_products()}
        log.info(f"Products: {list(self._products.keys())}")

        # Start SSE stream
        self.start()
        self._running = True

        # Initial data fetch
        self.data_engine.fetch_all(force=True)
        self.fv_engine.update_all(self.session_start, self.session_end)
        self._log_fair_values()

        try:
            while self._running:
                t0 = time.time()
                self._trading_tick()
                elapsed = time.time() - t0
                sleep_time = max(0, self.loop_interval - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
        finally:
            self._shutdown()

    def _trading_tick(self):
        """Single iteration of the trading loop."""
        # 1. Refresh data periodically
        if time.time() - self._last_data_fetch > self.data_interval:
            self.data_engine.fetch_all()
            self.fv_engine.update_all(self.session_start, self.session_end)
            self._last_data_fetch = time.time()
            self._log_fair_values()

        # 2. Sync positions from exchange
        try:
            positions = self.get_positions()
            self.risk.update_positions(positions)
        except Exception as e:
            log.warning(f"Position sync failed: {e}")

        # 3. Cancel all existing orders (cancel-and-replace strategy)
        # NOTE: In production, be smarter — only cancel if FV moved materially
        try:
            self.cancel_all_orders()
        except Exception as e:
            log.warning(f"Cancel failed: {e}")

        time.sleep(0.5)  # rate limit buffer

        # 4. Generate and send market-making quotes
        all_orders: list[OrderRequest] = []

        for symbol, product in self._products.items():
            ob = self._orderbooks.get(symbol)
            market_mid = self._get_market_mid(ob)
            quotes = self.mm.generate_quotes(symbol, product.tickSize, market_mid)
            all_orders.extend(quotes)

        # 5. Check ETF arbitrage
        arb_orders = self.arb.check_arb(self._orderbooks)
        all_orders.extend(arb_orders)

        # 6. Check options mispricing
        fly_ob = self._orderbooks.get("LON_FLY")
        if fly_ob:
            opt_orders = self.options.check_mispricing(fly_ob)
            all_orders.extend(opt_orders)

        # 7. Send orders (batched, respecting rate limits)
        if all_orders:
            # Batch send using threaded helper
            self._send_rate_limited(all_orders)

        # 8. Log status
        self._log_status()

    def _get_market_mid(self, ob: Optional[OrderBook]) -> Optional[float]:
        if ob is None:
            return None
        bids = [o.price for o in ob.buy_orders if o.volume - o.own_volume > 0]
        asks = [o.price for o in ob.sell_orders if o.volume - o.own_volume > 0]
        if bids and asks:
            return (max(bids) + min(asks)) / 2
        return None

    def _send_rate_limited(self, orders: list[OrderRequest]):
        """Send orders respecting 1 req/sec rate limit.

        Uses send_orders (threaded) for batches, but keeps total under limit.
        """
        # send_orders uses threading internally — send in small batches
        batch_size = 4  # 4 parallel requests
        for i in range(0, len(orders), batch_size):
            batch = orders[i:i + batch_size]
            self.send_orders(batch)
            self._order_count += len(batch)
            if i + batch_size < len(orders):
                time.sleep(1.0)  # rate limit pause between batches

    def _shutdown(self):
        log.info("Shutting down...")
        try:
            self.cancel_all_orders()
        except Exception:
            pass
        self.stop()
        log.info("Bot stopped.")

    # ── Logging ─────────────────────────────────────────────────────────────

    def _log_fair_values(self):
        log.info("─── Fair Values ───")
        for sym in sorted(self.fv_engine.fair_values.keys()):
            fv = self.fv_engine.fair_values[sym]
            conf = self.fv_engine.confidence.get(sym, 0)
            log.info(f"  {sym:<12} FV={fv:>8.0f}  conf={conf:.2f}")

    def _log_status(self):
        positions = {k: v for k, v in self.risk.state.positions.items() if v != 0}
        if positions:
            pos_str = "  ".join(f"{k}={v:+d}" for k, v in positions.items())
            log.info(f"Positions: {pos_str}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone

    EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
    USERNAME = "stopLossIOQ"
    PASSWORD = "stopLOSSimccmi"
    AERODATABOX_KEY = "0a7ff9f16fmsh54fb7da32af7310p12b89fjsn800a543a8628"

    # London is UTC+0 in winter (GMT), UTC+1 in summer (BST)
    # Late February = still GMT, so UTC+0
    london_offset = timezone(timedelta(hours=0))
    SESSION_START = datetime(2026, 2, 28, 12, 0, 0, tzinfo=london_offset)
    SESSION_END = datetime(2026, 3, 1, 12, 0, 0, tzinfo=london_offset)

    bot = AlgothonBot(
        cmi_url=EXCHANGE_URL,
        username=USERNAME,
        password=PASSWORD,
        aero_key=AERODATABOX_KEY,
        session_start=SESSION_START,
        session_end=SESSION_END,
        loop_interval=5.0,
        data_interval=120.0,
    )

    print("Starting bot (Ctrl+C to stop)...")
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nStopped.")
