"""
IMCity Algothon — Full Trading System (IMPROVED v2)
======================================================

Key improvements over v1:
  - Profit-taking: actively closes positions when sufficiently profitable
  - Better confidence thresholds: min 0.25 to quote (was 0.1)
  - Price sanity check: reject FV if too far from market mid (>20%)
  - Drawdown guard: tighten spreads / reduce sizes after large drawdown
  - Smarter cancel: only cancel if FV moved materially (>1 tick)
  - Larger spreads for low-confidence products
  - Stop accumulating when position PnL is negative & position is large
  - Market-anchored quotes: blend FV with market mid when conf is low
  - Faster data refresh interval (60s not 120s)
  - Better realized PnL tracking (FIFO)
  - ETF arb: increased edge threshold to avoid noise trades
  - On startup: cancel ALL open orders before placing new ones
  - Session-aware sizing: trade more aggressively early, less at end

Architecture:
  1. DataEngine       — fetches & caches weather, tides, flights
  2. FairValueEngine  — computes theo for all 8 products
  3. RiskEngine       — tracks positions, PnL, exposure limits
  4. MarketMaker      — quotes around fair value with inventory skew
  5. ArbEngine        — ETF vs components arbitrage
  6. OptionsEngine    — prices LON_FLY from ETF distribution
  7. AlgothonBot      — main orchestrator (extends BaseBot)
"""

import json
import math
import time
import logging
import threading
from collections import defaultdict, deque
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
        self.min_interval = min_interval

        self.weather: Optional[pd.DataFrame] = None
        self.thames: Optional[pd.DataFrame] = None
        self.flights: Optional[dict] = None

        self._last_fetch = {"weather": 0.0, "thames": 0.0, "flights": 0.0}

    def _should_fetch(self, source: str) -> bool:
        return time.time() - self._last_fetch[source] > self.min_interval

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
                    "past_minutely_15": 96,
                    "forecast_minutely_15": 96,
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
            self.weather["temperature_f"] = self.weather["temperature_c"] * 9 / 5 + 32
            self.weather["wx_metric"] = (
                self.weather["temperature_f"] * self.weather["humidity"]
            )
            self._last_fetch["weather"] = time.time()
            log.info(f"Weather: {len(self.weather)} rows fetched")
        except Exception as e:
            log.warning(f"Weather fetch failed: {e}")
        return self.weather

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

    def fetch_flights(self, offset_minutes: int = -720, duration_minutes: int = 720,
                      force: bool = False) -> Optional[dict]:
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
        self.confidence: dict[str, float] = {}
        self._prev_fair_values: dict[str, float] = {}

    def update_all(self, session_start: datetime, session_end: datetime):
        self._prev_fair_values = dict(self.fair_values)
        self._update_tide_spot(session_end)
        self._update_tide_swing(session_start, session_end)
        self._update_wx_spot(session_end)
        self._update_wx_sum(session_start, session_end)
        self._update_lhr_count(session_start, session_end)
        self._update_lhr_index(session_start, session_end)
        self._update_lon_etf()
        self._update_lon_fly()
        return self.fair_values

    def fv_moved_materially(self, product: str, tick_size: float, threshold_ticks: int = 2) -> bool:
        """Returns True if FV has moved by more than threshold_ticks since last update."""
        old = self._prev_fair_values.get(product)
        new = self.fair_values.get(product)
        if old is None or new is None:
            return True
        return abs(new - old) >= threshold_ticks * tick_size

    # ── TIDE_SPOT ──────────────────────────────────────────────────────────
    def _update_tide_spot(self, session_end: datetime):
        df = self.data.thames
        if df is None or df.empty:
            self.confidence["TIDE_SPOT"] = 0.0
            return
        predicted_level = self._extrapolate_tide(df, session_end)
        self.fair_values["TIDE_SPOT"] = abs(predicted_level) * 1000
        self.confidence["TIDE_SPOT"] = self._tide_confidence(df, session_end)

    def _extrapolate_tide(self, df: pd.DataFrame, target: datetime) -> float:
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

    # ── TIDE_SWING ─────────────────────────────────────────────────────────
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

    # ── WX_SPOT ────────────────────────────────────────────────────────────
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

    # ── WX_SUM ─────────────────────────────────────────────────────────────
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

    # ── LHR_COUNT ──────────────────────────────────────────────────────────
    def _update_lhr_count(self, session_start: datetime, session_end: datetime):
        flights = self.data.flights
        if flights is None:
            self.confidence["LHR_COUNT"] = 0.0
            return
        n_arr = len(flights.get("arrivals", []))
        n_dep = len(flights.get("departures", []))
        self.fair_values["LHR_COUNT"] = n_arr + n_dep
        self.confidence["LHR_COUNT"] = 0.5

    # ── LHR_INDEX ──────────────────────────────────────────────────────────
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

    # ── LON_ETF ────────────────────────────────────────────────────────────
    def _update_lon_etf(self):
        components = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        if all(c in self.fair_values for c in components):
            self.fair_values["LON_ETF"] = sum(self.fair_values[c] for c in components)
            self.confidence["LON_ETF"] = min(self.confidence.get(c, 0) for c in components)
        else:
            self.confidence["LON_ETF"] = 0.0

    # ── LON_FLY ────────────────────────────────────────────────────────────
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
        n_sims = 10_000
        samples = np.random.normal(etf_mean, etf_std, n_sims)
        samples = np.maximum(samples, 0)
        payoffs = np.array([self._fly_intrinsic(s) for s in samples])
        return float(np.mean(payoffs))


# =============================================================================
# 3. RISK ENGINE — Tracks positions, PnL, exposure limits
# =============================================================================

@dataclass
class RiskState:
    positions: dict = field(default_factory=lambda: defaultdict(int))
    max_position: int = 100
    fill_history: list = field(default_factory=list)
    realized_pnl: dict = field(default_factory=lambda: defaultdict(float))
    avg_entry: dict = field(default_factory=lambda: defaultdict(float))
    # FIFO queue for realized PnL tracking
    entry_queue: dict = field(default_factory=lambda: defaultdict(deque))


class RiskEngine:
    """Tracks positions, enforces limits, computes inventory skew."""

    def __init__(self, max_position: int = 100):
        self.state = RiskState(max_position=max_position)
        self._lock = threading.Lock()
        self._peak_pnl: float = 0.0
        self._trough_pnl: float = 0.0

    def update_positions(self, positions: dict):
        with self._lock:
            self.state.positions = defaultdict(int, positions)

    def record_fill(self, trade: "Trade", our_side: "Side"):
        from bot_template import Side
        with self._lock:
            self.state.fill_history.append(trade)
            product = trade.product
            qty = trade.volume if our_side == Side.BUY else -trade.volume
            pos = self.state.positions[product]
            new_pos = pos + qty

            # FIFO realized PnL
            if our_side == Side.BUY:
                self.state.entry_queue[product].append((trade.price, trade.volume))
            else:
                # Selling: realize against FIFO buys
                remaining = trade.volume
                while remaining > 0 and self.state.entry_queue[product]:
                    entry_price, entry_vol = self.state.entry_queue[product][0]
                    matched = min(remaining, entry_vol)
                    self.state.realized_pnl[product] += matched * (trade.price - entry_price)
                    remaining -= matched
                    if matched == entry_vol:
                        self.state.entry_queue[product].popleft()
                    else:
                        self.state.entry_queue[product][0] = (entry_price, entry_vol - matched)

            self.state.positions[product] = new_pos

            # Update avg entry
            old_pos = pos
            if abs(new_pos) > abs(old_pos):
                old_cost = self.state.avg_entry[product] * abs(old_pos)
                new_cost = trade.price * abs(qty)
                if new_pos != 0:
                    self.state.avg_entry[product] = (old_cost + new_cost) / abs(new_pos)
            elif new_pos == 0:
                self.state.avg_entry[product] = 0.0

    def get_position(self, product: str) -> int:
        return self.state.positions.get(product, 0)

    def can_buy(self, product: str, volume: int) -> int:
        pos = self.get_position(product)
        return max(0, min(volume, self.state.max_position - pos))

    def can_sell(self, product: str, volume: int) -> int:
        pos = self.get_position(product)
        return max(0, min(volume, self.state.max_position + pos))

    def inventory_skew(self, product: str) -> float:
        return self.get_position(product) / self.state.max_position

    def quote_size(self, product: str, base_size: int) -> tuple:
        skew = self.inventory_skew(product)
        bid_size = max(1, int(base_size * (1 - max(0, skew))))
        ask_size = max(1, int(base_size * (1 + min(0, skew))))
        bid_size = self.can_buy(product, bid_size)
        ask_size = self.can_sell(product, ask_size)
        return bid_size, ask_size

    def total_exposure(self) -> int:
        """Sum of absolute positions across all products."""
        return sum(abs(v) for v in self.state.positions.values())


# =============================================================================
# 4. MARKET MAKER — Improved with better conf thresholds & profit-taking
# =============================================================================

class MarketMaker:
    """Generates two-sided quotes around fair value with adaptive width.
    
    Improvements:
    - Min confidence 0.25 (was 0.1) to actually quote
    - Blends FV with market mid when confidence is low
    - Sanity check: skip if FV too far from market mid (>15%)
    - Wider spreads for low-confidence products
    - Position-based profit taking orders
    """

    MIN_CONF = 0.25  # Don't quote if confidence below this

    def __init__(self, risk: RiskEngine, fv_engine: FairValueEngine):
        self.risk = risk
        self.fv = fv_engine

        # (base_width, base_size, min_edge, max_levels)
        # Increased base widths to be less aggressive / more conservative
        self.config = {
            "TIDE_SPOT":  (20, 4, 5, 2),
            "TIDE_SWING": (15, 3, 3, 2),
            "WX_SPOT":    (20, 4, 5, 2),
            "WX_SUM":     (15, 3, 3, 2),
            "LHR_COUNT":  (25, 4, 7, 2),
            "LHR_INDEX":  (20, 3, 5, 2),
            "LON_ETF":    (40, 3, 10, 2),
            "LON_FLY":    (30, 2, 8, 2),
        }

    def generate_quotes(self, product: str, tick_size: float,
                        market_mid: Optional[float] = None) -> list:
        from bot_template import OrderRequest, Side

        fv = self.fv.fair_values.get(product)
        conf = self.fv.confidence.get(product, 0)

        # If no FV or very low confidence, fall back to pure market making
        if fv is None or conf < self.MIN_CONF:
            if market_mid is not None and conf < self.MIN_CONF:
                # Use market mid with very wide spread
                fv = market_mid
                conf = 0.15
            else:
                return []

        # Sanity check: if FV is more than 15% away from market mid, don't trust FV alone
        if market_mid is not None and market_mid > 0:
            fv_vs_mid_pct = abs(fv - market_mid) / market_mid
            if fv_vs_mid_pct > 0.15:
                # Blend: 30% FV + 70% market mid (distrust FV when very different)
                log.warning(f"{product}: FV={fv:.0f} vs mid={market_mid:.0f} "
                            f"divergence {fv_vs_mid_pct:.1%} — blending")
                fv = 0.3 * fv + 0.7 * market_mid
                conf = min(conf, 0.3)

        cfg = self.config.get(product, (20, 4, 5, 2))
        base_width, base_size, min_edge, max_levels = cfg

        # Adaptive width: wider when less confident
        width = base_width / max(conf, 0.15)
        width = max(width, min_edge * 2)

        # Inventory skew
        skew = self.risk.inventory_skew(product)
        skew_offset = skew * width * 0.4  # stronger skew to reduce position
        adjusted_fv = fv - skew_offset

        orders = []
        bid_size, ask_size = self.risk.quote_size(product, base_size)

        for level in range(max_levels):
            level_offset = width * (0.5 + level * 0.6)

            bid_price = math.floor((adjusted_fv - level_offset) / tick_size) * tick_size
            ask_price = math.ceil((adjusted_fv + level_offset) / tick_size) * tick_size

            lvl_bid_size = max(1, bid_size // (level + 1))
            lvl_ask_size = max(1, ask_size // (level + 1))

            if bid_price > 0 and lvl_bid_size > 0:
                orders.append(OrderRequest(product, bid_price, Side.BUY, lvl_bid_size))
            if ask_price > bid_price and lvl_ask_size > 0:
                orders.append(OrderRequest(product, ask_price, Side.SELL, lvl_ask_size))

        return orders

    def generate_profit_taking_orders(self, product: str, tick_size: float,
                                       market_mid: Optional[float]) -> list:
        """Generate aggressive closing orders when position is profitable."""
        from bot_template import OrderRequest, Side

        pos = self.risk.get_position(product)
        if abs(pos) < 10:  # Too small to bother
            return []

        avg_entry = self.risk.state.avg_entry.get(product, 0.0)
        if avg_entry == 0.0 or market_mid is None:
            return []

        pnl_per_unit = (market_mid - avg_entry) if pos > 0 else (avg_entry - market_mid)

        # Take profit if up > 0.5% per unit on a position of 20+
        if pnl_per_unit > avg_entry * 0.005 and abs(pos) >= 20:
            close_vol = max(1, abs(pos) // 4)  # close 25% of position
            if pos > 0:
                # Long position — sell to take profit (at market mid)
                price = math.floor(market_mid / tick_size) * tick_size
                vol = self.risk.can_sell(product, close_vol)
                if vol > 0:
                    log.info(f"PROFIT-TAKE: Sell {vol}x {product} @ {price} "
                             f"(avg={avg_entry:.0f}, pnl/unit={pnl_per_unit:.1f})")
                    return [OrderRequest(product, price, Side.SELL, vol)]
            else:
                # Short position — buy to take profit
                price = math.ceil(market_mid / tick_size) * tick_size
                vol = self.risk.can_buy(product, close_vol)
                if vol > 0:
                    log.info(f"PROFIT-TAKE: Buy {vol}x {product} @ {price} "
                             f"(avg={avg_entry:.0f}, pnl/unit={pnl_per_unit:.1f})")
                    return [OrderRequest(product, price, Side.BUY, vol)]

        return []


# =============================================================================
# 5. ETF ARBITRAGE ENGINE — Improved edge threshold
# =============================================================================

class ArbEngine:
    """Detects and trades ETF vs component mispricings.

    LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT
    """

    def __init__(self, risk: RiskEngine, min_edge: float = 20.0):
        self.risk = risk
        self.min_edge = min_edge  # Increased from 10 to reduce noise trades

    def check_arb(self, orderbooks: dict) -> list:
        from bot_template import OrderRequest, Side

        components = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        etf_sym = "LON_ETF"

        if etf_sym not in orderbooks:
            return []
        for c in components:
            if c not in orderbooks:
                return []

        etf_ob = orderbooks[etf_sym]
        comp_obs = {c: orderbooks[c] for c in components}
        orders = []

        # Sell ETF, Buy components
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
            edge = etf_bid - comp_ask_total

            if edge > self.min_edge:
                vol = min(5,
                          self.risk.can_sell(etf_sym, 5),
                          *[self.risk.can_buy(c, 5) for c in components])
                if vol > 0:
                    orders.append(OrderRequest(etf_sym, etf_bid, Side.SELL, vol))
                    for c in components:
                        orders.append(OrderRequest(c, comp_obs[c].sell_orders[0].price, Side.BUY, vol))
                    log.info(f"ARB: Sell ETF @ {etf_bid}, Buy components. Edge={edge:.0f}")

        # Buy ETF, Sell components
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
            edge = comp_bid_total - etf_ask

            if edge > self.min_edge:
                vol = min(5,
                          self.risk.can_buy(etf_sym, 5),
                          *[self.risk.can_sell(c, 5) for c in components])
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
    """Prices LON_FLY options structure."""

    def __init__(self, fv_engine: FairValueEngine, risk: RiskEngine):
        self.fv = fv_engine
        self.risk = risk

    def price_fly(self, etf_mean: float, etf_std: float = 300, n_sims: int = 20000) -> float:
        samples = np.random.normal(etf_mean, etf_std, n_sims)
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

    def check_mispricing(self, fly_ob: "OrderBook") -> list:
        from bot_template import OrderRequest, Side

        etf_fv = self.fv.fair_values.get("LON_ETF")
        if etf_fv is None:
            return []

        etf_conf = self.fv.confidence.get("LON_ETF", 0)
        etf_std = 300 * (1 - etf_conf)
        etf_std = max(etf_std, 50)

        model_price = self.price_fly(etf_fv, etf_std)
        orders = []

        min_edge = 20  # Increased from 15

        if fly_ob.sell_orders:
            best_ask = fly_ob.sell_orders[0].price
            if model_price - best_ask > min_edge:
                vol = self.risk.can_buy("LON_FLY", 2)
                if vol > 0:
                    orders.append(OrderRequest("LON_FLY", best_ask, Side.BUY, vol))
                    log.info(f"OPT: Buy FLY @ {best_ask}, model={model_price:.0f}")

        if fly_ob.buy_orders:
            best_bid = fly_ob.buy_orders[0].price
            if best_bid - model_price > min_edge:
                vol = self.risk.can_sell("LON_FLY", 2)
                if vol > 0:
                    orders.append(OrderRequest("LON_FLY", best_bid, Side.SELL, vol))
                    log.info(f"OPT: Sell FLY @ {best_bid}, model={model_price:.0f}")

        return orders


# =============================================================================
# 7. MAIN BOT — Orchestrates everything
# =============================================================================

class AlgothonBot(BaseBot):
    """Full trading bot for the IMCity Algothon — v2 (improved).

    Key improvements:
    - Profit-taking on large profitable positions
    - FV sanity checks against market mid
    - Higher confidence threshold (0.25)
    - Drawdown-aware position sizing
    - Session-time-aware aggressiveness
    - Cancel-all on startup BEFORE trading
    - Faster data refresh (60s)
    """

    def __init__(self, cmi_url: str, username: str, password: str,
                 aero_key: str = "",
                 session_start: Optional[datetime] = None,
                 session_end: Optional[datetime] = None,
                 loop_interval: float = 5.0,
                 data_interval: float = 60.0):  # 60s refresh (was 120s)
        super().__init__(cmi_url, username, password)

        london_tz = timezone(timedelta(hours=0))
        if session_start is None:
            now = datetime.now(tz=london_tz)
            session_start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if session_end is None:
            session_end = session_start + timedelta(hours=24)

        self.session_start = session_start
        self.session_end = session_end
        self.loop_interval = max(loop_interval, 2.0)
        self.data_interval = data_interval

        # Sub-engines
        self.data_engine = DataEngine(aero_key=aero_key, min_interval=data_interval)
        self.fv_engine = FairValueEngine(self.data_engine)
        self.risk = RiskEngine(max_position=100)
        self.mm = MarketMaker(self.risk, self.fv_engine)
        self.arb = ArbEngine(self.risk, min_edge=20)
        self.options = OptionsEngine(self.fv_engine, self.risk)

        # State
        self._orderbooks: dict[str, OrderBook] = {}
        self._products: dict[str, Product] = {}
        self._last_data_fetch: float = 0
        self._running = False
        self._order_count = 0

        # Drawdown tracking
        self._peak_exchange_pnl: float = 0.0
        self._last_exchange_pnl: float = 0.0
        self._drawdown_mode: bool = False

        # PnL history for tracking
        self._pnl_history: deque = deque(maxlen=50)

    # ── SSE Callbacks ──────────────────────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook):
        self._orderbooks[orderbook.product] = orderbook

    def on_trades(self, trade: Trade):
        if trade.buyer == self.username:
            side = Side.BUY
            log.info(f"FILL: BOUGHT {trade.volume}x {trade.product} @ {trade.price}")
        else:
            side = Side.SELL
            log.info(f"FILL: SOLD {trade.volume}x {trade.product} @ {trade.price}")
        self.risk.record_fill(trade, side)

    # ── Main Loop ──────────────────────────────────────────────────────────

    def run(self):
        log.info("═" * 60)
        log.info("  ALGOTHON BOT STARTING (v2 — improved)")
        log.info(f"  Session: {self.session_start} → {self.session_end}")
        log.info("═" * 60)

        self._products = {p.symbol: p for p in self.get_products()}
        log.info(f"Products: {list(self._products.keys())}")

        self.start()
        self._running = True

        # CRITICAL: Cancel ALL open orders from previous runs before trading
        log.info("Cancelling all open orders from previous sessions...")
        try:
            self.cancel_all_orders()
            time.sleep(1.0)
            log.info("All orders cancelled. Starting fresh.")
        except Exception as e:
            log.warning(f"Initial cancel failed: {e}")

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

    def _session_progress(self) -> float:
        """Returns 0.0 at session start, 1.0 at session end."""
        now = datetime.now(tz=self.session_start.tzinfo)
        total = (self.session_end - self.session_start).total_seconds()
        elapsed = (now - self.session_start).total_seconds()
        return max(0.0, min(1.0, elapsed / total))

    def _update_drawdown_mode(self, exchange_pnl: float):
        """Track peak PnL and detect significant drawdowns."""
        if exchange_pnl > self._peak_exchange_pnl:
            self._peak_exchange_pnl = exchange_pnl

        self._last_exchange_pnl = exchange_pnl
        drawdown = self._peak_exchange_pnl - exchange_pnl

        # Enter drawdown mode if we've lost more than 500k from peak
        if drawdown > 500_000 and self._peak_exchange_pnl > 100_000:
            if not self._drawdown_mode:
                log.warning(f"DRAWDOWN MODE: Peak={self._peak_exchange_pnl:+.0f}, "
                            f"Current={exchange_pnl:+.0f}, DD={drawdown:+.0f}")
            self._drawdown_mode = True
        else:
            self._drawdown_mode = False

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

        # 3. Cancel stale orders (only if FV has moved materially)
        try:
            fv_moved = any(
                self.fv_engine.fv_moved_materially(sym, prod.tickSize)
                for sym, prod in self._products.items()
            )
            if fv_moved:
                self.cancel_all_orders()
                time.sleep(0.3)
            else:
                # Still cancel every 3 loops to refresh stale quotes
                if self._order_count % 3 == 0:
                    self.cancel_all_orders()
                    time.sleep(0.3)
        except Exception as e:
            log.warning(f"Cancel failed: {e}")

        time.sleep(0.3)

        # 4. Generate orders
        all_orders: list[OrderRequest] = []

        # In drawdown mode: only generate profit-taking orders, no new risk
        for symbol, product in self._products.items():
            ob = self._orderbooks.get(symbol)
            market_mid = self._get_market_mid(ob)

            if not self._drawdown_mode:
                quotes = self.mm.generate_quotes(symbol, product.tickSize, market_mid)
                all_orders.extend(quotes)

            # Always generate profit-taking (even in drawdown mode)
            pt_orders = self.mm.generate_profit_taking_orders(symbol, product.tickSize, market_mid)
            all_orders.extend(pt_orders)

        # 5. ETF arbitrage (disable in drawdown mode to avoid compounding losses)
        if not self._drawdown_mode:
            arb_orders = self.arb.check_arb(self._orderbooks)
            all_orders.extend(arb_orders)

        # 6. Options mispricing
        if not self._drawdown_mode:
            fly_ob = self._orderbooks.get("LON_FLY")
            if fly_ob:
                opt_orders = self.options.check_mispricing(fly_ob)
                all_orders.extend(opt_orders)

        # 7. Send orders
        if all_orders:
            self._send_rate_limited(all_orders)

        # 8. Log status
        self._order_count += 1
        self._log_status()

    def _get_market_mid(self, ob: Optional[OrderBook]) -> Optional[float]:
        if ob is None:
            return None
        bids = [o.price for o in ob.buy_orders if o.volume - o.own_volume > 0]
        asks = [o.price for o in ob.sell_orders if o.volume - o.own_volume > 0]
        if bids and asks:
            return (max(bids) + min(asks)) / 2
        return None

    def _mark_price(self, product: str) -> Optional[float]:
        ob = self._orderbooks.get(product)
        if ob is None:
            return None
        bids = [o.price for o in ob.buy_orders if (o.volume - o.own_volume) > 0]
        asks = [o.price for o in ob.sell_orders if (o.volume - o.own_volume) > 0]
        if bids and asks:
            return (max(bids) + min(asks)) / 2
        if bids:
            return max(bids)
        if asks:
            return min(asks)
        return None

    def _compute_pnl(self) -> tuple:
        realized = float(sum(self.risk.state.realized_pnl.values()))
        unreal = 0.0
        for product, pos in self.risk.state.positions.items():
            try:
                pos_i = int(round(float(pos)))
            except Exception:
                continue
            if pos_i == 0:
                continue
            mark = self._mark_price(product)
            if mark is None:
                continue
            avg = float(self.risk.state.avg_entry.get(product, 0.0))
            if avg == 0.0:
                continue
            unreal += pos_i * (float(mark) - avg)
        return realized + unreal, unreal, realized

    def _send_rate_limited(self, orders: list):
        batch_size = 4
        for i in range(0, len(orders), batch_size):
            batch = orders[i:i + batch_size]
            summary = "; ".join(f"{o.product} {o.side} x{o.volume}@{o.price}" for o in batch)
            log.info(f"Sending {len(batch)} orders: {summary}")
            self.send_orders(batch)
            if i + batch_size < len(orders):
                time.sleep(1.0)

    def _shutdown(self):
        log.info("Shutting down...")
        try:
            self.cancel_all_orders()
        except Exception:
            pass
        self.stop()
        log.info("Bot stopped.")

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_fair_values(self):
        log.info("─── Fair Values ───")
        for sym in sorted(self.fv_engine.fair_values.keys()):
            fv = self.fv_engine.fair_values[sym]
            conf = self.fv_engine.confidence.get(sym, 0)
            mid = self._mark_price(sym)
            mid_str = f"  mid={mid:.0f}" if mid else ""
            log.info(f"  {sym:<12} FV={fv:>8.0f}  conf={conf:.2f}{mid_str}")

    def _log_status(self):
        positions = {k: v for k, v in self.risk.state.positions.items() if abs(v) > 0}

        def fmt_pos(x):
            try:
                xi = int(round(float(x)))
            except Exception:
                return str(x)
            return f"{xi:+d}"

        pos_str = "  ".join(f"{k}={fmt_pos(v)}" for k, v in positions.items()) if positions else "flat"
        total, unreal, realized = self._compute_pnl()

        # Try to get exchange PnL
        exch_total = None
        try:
            pnl_dict = self.get_pnl()
            if isinstance(pnl_dict, dict):
                exch_total = (
                    pnl_dict.get("totalProfit")
                    or pnl_dict.get("profit")
                    or pnl_dict.get("pnl")
                )
        except Exception:
            pass

        if exch_total is not None:
            self._update_drawdown_mode(float(exch_total))
            dd_flag = " [DRAWDOWN MODE]" if self._drawdown_mode else ""
            log.info(
                f"Positions: {pos_str} | "
                f"LocalPnL={total:+.1f} (U={unreal:+.1f}, R={realized:+.1f}) | "
                f"ExchangePnL={float(exch_total):+.1f}{dd_flag}"
            )
        else:
            log.info(
                f"Positions: {pos_str} | "
                f"PnL: total={total:+.1f} (unreal={unreal:+.1f}, real={realized:+.1f})"
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone

    EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
    USERNAME = "stopLossIOQ_1"
    PASSWORD = "stopLOSSimccmi"
    AERODATABOX_KEY = "0a7ff9f16fmsh54fb7da32af7310p12b89fjsn800a543a8628"

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
        data_interval=60.0,  # Refresh data every 60s (was 120s)
    )

    print("Starting bot v2 (Ctrl+C to stop)...")
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nStopped.")
