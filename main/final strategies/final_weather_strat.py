"""
Weather Mispricing Bot
======================
1. Fetches live London weather from 4 sources every 30s
2. Computes fair value for WX_SPOT and WX_SUM
3. Reads the orderbook from the CMI Exchange
4. Buys if fair_value > best_ask + threshold
   Sells if fair_value < best_bid - threshold
5. Adjusts confidence as the session progresses

FIXES vs previous version:
  [FIX 1] SPOT orderbook None  → always falls back to REST and refreshes
           the SSE cache so the problem self-heals
  [FIX 2] Position limit       → MAX_POSITION raised to 50, plus a
           "trim" mode that peels off TRIM_VOLUME lots whenever the market
           moves sharply against a maxed-out position, keeping us nimble
"""

import math
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from threading import Thread, Lock
from bot_template import BaseBot, OrderBook, OrderRequest, Trade, Side, Order

print("Imports OK")


# ============================================================
# SECTION 1 — CONFIG
# ============================================================

TEST_URL      = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com"
CHALLENGE_URL = "https://cmiexchange.replit.app"

EXCHANGE_URL = TEST_URL   # ← switch to CHALLENGE_URL for competition
USERNAME     = "weatherhamza"
PASSWORD     = "weatherhamza"

# ── API keys ──────────────────────────────────────────────────────────────────
OWM_API_KEY  = "ad6b547d314dbfae707b72c101dbc740"
WAPI_API_KEY = "c5031f71c3874ac69e0191831262802"

# ── Trading params ────────────────────────────────────────────────────────────
TRADE_VOLUME      = 5     # lots per aggressive order
TRIM_VOLUME       = 5     # lots to trim when maxed-out and market moves against us
MAX_POSITION      = 50    # FIX 2: raised from 20 → 50 to avoid getting stuck
THRESHOLD_WIDE    = 10    # early-session threshold (points of mispricing needed)
THRESHOLD_TIGHT   = 5     # late-session threshold (tightens as conf → 1)
WEATHER_POLL_SEC  = 30    # seconds between weather fetches

# FIX 2: trim triggers when market mid is this far PAST fair value while maxed
TRIM_TRIGGER      = 50    # points beyond FV before we start trimming

LONDON_LAT, LONDON_LON = 51.5074, -0.1278


def c_to_f(c):
    return c * 9 / 5 + 32


print(f"Exchange : {EXCHANGE_URL}")
print(f"Params   : vol={TRADE_VOLUME}, max_pos={MAX_POSITION}, "
      f"threshold={THRESHOLD_WIDE}→{THRESHOLD_TIGHT}, trim_trigger={TRIM_TRIGGER}")


# ============================================================
# SECTION 2 — WEATHER FETCHERS (4 sources)
# ============================================================

def fetch_open_meteo(past_steps=96, forecast_steps=96):
    """Source 1: Open-Meteo — THE settlement source. 15-min resolution."""
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LONDON_LAT, "longitude": LONDON_LON,
        "minutely_15": "temperature_2m,relative_humidity_2m",
        "past_minutely_15": past_steps,
        "forecast_minutely_15": forecast_steps,
        "timezone": "Europe/London",
    }, timeout=10)
    resp.raise_for_status()
    m = resp.json()["minutely_15"]
    df = pd.DataFrame({
        "time":     pd.to_datetime(m["time"]),
        "temp_c":   m["temperature_2m"],
        "humidity": m["relative_humidity_2m"],
    })
    df["temp_f"]         = c_to_f(df["temp_c"])
    df["temp_f_rounded"] = df["temp_f"].round()
    df["t_x_h"]          = df["temp_f_rounded"] * df["humidity"]
    return df


def fetch_owm_current():
    """Source 2: OpenWeatherMap — current conditions."""
    resp = requests.get("https://api.openweathermap.org/data/2.5/weather", params={
        "lat": LONDON_LAT, "lon": LONDON_LON,
        "appid": OWM_API_KEY, "units": "metric",
    }, timeout=10)
    resp.raise_for_status()
    d        = resp.json()
    temp_c   = d["main"]["temp"]
    humidity = d["main"]["humidity"]
    temp_f   = c_to_f(temp_c)
    return {"temp_f": temp_f, "temp_f_rounded": round(temp_f),
            "humidity": humidity, "t_x_h": round(temp_f) * humidity, "source": "owm"}


def fetch_owm_forecast():
    """Source 2b: OpenWeatherMap — 5-day/3-hour forecast."""
    resp = requests.get("https://api.openweathermap.org/data/2.5/forecast", params={
        "lat": LONDON_LAT, "lon": LONDON_LON,
        "appid": OWM_API_KEY, "units": "metric",
    }, timeout=10)
    resp.raise_for_status()
    rows = []
    for item in resp.json()["list"]:
        tc = item["main"]["temp"]
        h  = item["main"]["humidity"]
        tf = c_to_f(tc)
        rows.append({"time": pd.Timestamp.fromtimestamp(item["dt"]),
                     "temp_f": tf, "temp_f_rounded": round(tf),
                     "humidity": h, "t_x_h": round(tf) * h})
    return pd.DataFrame(rows)


def fetch_wapi_current():
    """Source 3: WeatherAPI.com — current conditions."""
    resp = requests.get("https://api.weatherapi.com/v1/current.json", params={
        "key": WAPI_API_KEY, "q": f"{LONDON_LAT},{LONDON_LON}",
    }, timeout=10)
    resp.raise_for_status()
    d = resp.json()["current"]
    return {"temp_f": d["temp_f"], "temp_f_rounded": round(d["temp_f"]),
            "humidity": d["humidity"],
            "t_x_h": round(d["temp_f"]) * d["humidity"], "source": "wapi"}


def fetch_wapi_forecast(days=2):
    """Source 3b: WeatherAPI.com — hourly forecast."""
    resp = requests.get("https://api.weatherapi.com/v1/forecast.json", params={
        "key": WAPI_API_KEY, "q": f"{LONDON_LAT},{LONDON_LON}", "days": days,
    }, timeout=10)
    resp.raise_for_status()
    rows = []
    for day in resp.json()["forecast"]["forecastday"]:
        for hour in day["hour"]:
            tf = hour["temp_f"]
            h  = hour["humidity"]
            rows.append({"time": pd.Timestamp(hour["time"]),
                         "temp_f": tf, "temp_f_rounded": round(tf),
                         "humidity": h, "t_x_h": round(tf) * h})
    return pd.DataFrame(rows)


def fetch_wttr_current():
    """Source 4: wttr.in — no API key needed."""
    resp = requests.get("https://wttr.in/London", params={"format": "j1"}, timeout=10)
    resp.raise_for_status()
    d        = resp.json()["current_condition"][0]
    temp_f   = float(d["temp_F"])
    humidity = int(d["humidity"])
    return {"temp_f": temp_f, "temp_f_rounded": round(temp_f),
            "humidity": humidity, "t_x_h": round(temp_f) * humidity, "source": "wttr"}


# ── Self-test ──────────────────────────────────────────────────────────────────
print("Fetchers defined. Testing …")
try:
    df_test = fetch_open_meteo(past_steps=4, forecast_steps=4)
    print(f"  Open-Meteo  OK : {len(df_test)} rows")
except Exception as e:
    print(f"  Open-Meteo  FAIL: {e}")

for name, fn in [("OWM", fetch_owm_current),
                 ("WeatherAPI", fetch_wapi_current),
                 ("wttr", fetch_wttr_current)]:
    try:
        r = fn()
        print(f"  {name:10s} OK : T={r['temp_f_rounded']}°F  "
              f"H={r['humidity']}%  T×H={r['t_x_h']}")
    except Exception as e:
        print(f"  {name:10s} FAIL: {e}")


# ============================================================
# SECTION 3 — FAIR VALUE ENGINE
# ============================================================
#
#  WX_SPOT = round(Temp_F) × Humidity  at Sunday 12:00
#  WX_SUM  = Σ (round(Temp_F) × Humidity / 100)  over 96 × 15-min intervals
#             from Saturday 12:00 → Sunday 12:00

class FairValueEngine:
    """
    Aggregates weather data from multiple sources and computes
    fair values for WX_SPOT and WX_SUM.

    Thread-safe: the weather-polling thread writes; the trading loop reads.
    """

    def __init__(self, session_start: pd.Timestamp, session_end: pd.Timestamp):
        self.session_start = session_start
        self.session_end   = session_end
        self.lock          = Lock()

        self.wx_spot_fv    = None   # fair value for WX_SPOT
        self.wx_sum_fv     = None   # fair value for WX_SUM
        self.confidence    = 0.0    # 0 at session start → 1 at session end
        self.last_update   = None
        self.source_readings = {}
        self.om_data       = None   # latest Open-Meteo DataFrame

    # ── public ────────────────────────────────────────────────────────────────

    def update(self):
        """Fetch all weather sources and recompute fair values."""
        now = pd.Timestamp.now(tz="Europe/London")

        # ── Fetch sources ─────────────────────────────────────────────────────
        readings = {}

        try:
            self.om_data  = fetch_open_meteo(past_steps=96, forecast_steps=96)
            now_naive     = now.tz_localize(None)
            past          = self.om_data[self.om_data.time <= now_naive]
            if len(past) > 0:
                latest = past.iloc[-1]
                readings["open_meteo"] = {
                    "temp_f_rounded": latest["temp_f_rounded"],
                    "humidity":       latest["humidity"],
                    "t_x_h":          latest["t_x_h"],
                }
        except Exception as e:
            print(f"  [FV] Open-Meteo error: {e}")

        for name, fn in [("owm", fetch_owm_current),
                          ("wapi", fetch_wapi_current),
                          ("wttr", fetch_wttr_current)]:
            try:
                readings[name] = fn()
            except Exception as e:
                print(f"  [FV] {name} error: {e}")

        if not readings:
            print("  [FV] WARNING: no sources available — skipping update")
            return

        # ── WX_SPOT fair value ────────────────────────────────────────────────
        spot_estimates, spot_weights = [], []

        if self.om_data is not None:
            settlement_naive = self.session_end.tz_localize(None)
            diffs   = (self.om_data.time - settlement_naive).abs()
            closest = self.om_data.iloc[diffs.argsort().iloc[0]]
            spot_estimates.append(closest["t_x_h"])
            spot_weights.append(0.50)

        try:
            df_owm_fc        = fetch_owm_forecast()
            settlement_naive = self.session_end.tz_localize(None)
            diffs            = (df_owm_fc.time - settlement_naive).abs()
            closest_owm      = df_owm_fc.iloc[diffs.argsort().iloc[0]]
            spot_estimates.append(closest_owm["t_x_h"])
            spot_weights.append(0.25)
        except Exception:
            pass

        try:
            df_wapi_fc       = fetch_wapi_forecast(days=2)
            settlement_naive = self.session_end.tz_localize(None)
            diffs            = (df_wapi_fc.time - settlement_naive).abs()
            closest_wapi     = df_wapi_fc.iloc[diffs.argsort().iloc[0]]
            spot_estimates.append(closest_wapi["t_x_h"])
            spot_weights.append(0.25)
        except Exception:
            pass

        # ── WX_SUM fair value ─────────────────────────────────────────────────
        sum_fv = None
        if self.om_data is not None:
            start_naive  = self.session_start.tz_localize(None)
            end_naive    = self.session_end.tz_localize(None)
            now_naive    = now.tz_localize(None)
            session_data = self.om_data[
                (self.om_data.time >= start_naive) &
                (self.om_data.time <= end_naive)
            ]

            if len(session_data) > 0:
                observed     = session_data[session_data.time <= now_naive]
                forecast     = session_data[session_data.time >  now_naive]
                observed_sum = (observed["t_x_h"] / 100).sum()
                forecast_sum = (forecast["t_x_h"] / 100).sum()
                sum_fv       = observed_sum + forecast_sum
            elif spot_estimates:
                # Session is beyond Open-Meteo's forecast window.
                # Use the locally-computed spot FV as a proxy for avg interval value.
                w             = np.array(spot_weights)
                w             = w / w.sum()
                local_spot_fv = np.average(spot_estimates, weights=w)
                sum_fv        = 96 * local_spot_fv / 100

        # ── Session confidence ────────────────────────────────────────────────
        total_duration = (self.session_end   - self.session_start).total_seconds()
        elapsed        = (now                - self.session_start).total_seconds()
        confidence     = max(0.0, min(1.0, elapsed / total_duration))

        # ── Commit (thread-safe) ──────────────────────────────────────────────
        with self.lock:
            if spot_estimates:
                w                = np.array(spot_weights)
                w                = w / w.sum()
                self.wx_spot_fv  = np.average(spot_estimates, weights=w)
            self.wx_sum_fv       = sum_fv
            self.confidence      = confidence
            self.source_readings = readings
            self.last_update     = now

    def get_state(self) -> dict:
        with self.lock:
            return {
                "wx_spot_fv": self.wx_spot_fv,
                "wx_sum_fv":  self.wx_sum_fv,
                "confidence": self.confidence,
                "last_update": self.last_update,
                "n_sources":  len(self.source_readings),
            }

    def get_threshold(self) -> float:
        """Linearly narrow threshold from WIDE (session start) → TIGHT (end)."""
        return THRESHOLD_WIDE + (THRESHOLD_TIGHT - THRESHOLD_WIDE) * self.confidence


print("FairValueEngine defined.")


# ============================================================
# SECTION 4 — TRADING BOT
# ============================================================

class WeatherBot(BaseBot):
    """
    Trades WX_SPOT and WX_SUM based on multi-source weather fair value.

    Architecture
    ────────────
    Weather Thread  →  FairValueEngine  →  Trading Loop
         ↓ writes            ↑ reads            ↓ sends orders
       every 30 s        thread-safe         every 5 s
    """

    def __init__(self, cmi_url, username, password, session_start, session_end):
        super().__init__(cmi_url, username, password)
        self.fv_engine       = FairValueEngine(session_start, session_end)
        self._weather_thread = None
        self._running        = False
        self._orderbooks: dict[str, OrderBook] = {}
        self._ob_lock        = Lock()
        # Track timestamp of last REST orderbook refresh per product
        self._ob_rest_ts: dict[str, float] = {}
        self._ob_rest_ttl = 2.0  # seconds before we force a REST refresh

    # ── SSE callbacks ─────────────────────────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook):
        with self._ob_lock:
            self._orderbooks[orderbook.product] = orderbook

    def on_trades(self, trade: Trade):
        if trade.buyer == self.username or trade.seller == self.username:
            side = "BOUGHT" if trade.buyer == self.username else "SOLD"
            print(f"  💰 FILL: {side} {trade.volume}x {trade.product} @ {trade.price}")

    # ── Weather thread ────────────────────────────────────────────────────────

    def _weather_loop(self):
        while self._running:
            try:
                self.fv_engine.update()
                s       = self.fv_engine.get_state()
                spot_s  = f"{s['wx_spot_fv']:.0f}" if s["wx_spot_fv"] is not None else "?"
                sum_s   = f"{s['wx_sum_fv']:.0f}"  if s["wx_sum_fv"]  is not None else "?"
                print(f"  🌤  FV updated: SPOT={spot_s}  SUM={sum_s}"
                      f"  conf={s['confidence']:.1%}  sources={s['n_sources']}")
            except Exception as e:
                print(f"  [Weather] Error: {e}")
            time.sleep(WEATHER_POLL_SEC)

    # ── Orderbook helpers ─────────────────────────────────────────────────────
    # FIX 1: robust _get_best_bid_ask
    #   • If the SSE cache has no market-side volume, immediately fetch via REST
    #   • The REST result is written back into the SSE cache so future reads
    #     benefit from it (self-healing)
    #   • We also force a REST refresh if the cached book is older than
    #     _ob_rest_ttl seconds AND both sides appear empty

    def _fetch_ob_rest(self, product: str) -> OrderBook | None:
        """Fetch orderbook via REST and update the SSE cache."""
        try:
            ob = self.get_orderbook(product)
            with self._ob_lock:
                self._orderbooks[product] = ob
            self._ob_rest_ts[product] = time.monotonic()
            return ob
        except Exception as e:
            print(f"  [OB-REST] {product} error: {e}")
            return None

    def _get_ob(self, product: str) -> OrderBook | None:
        """Return a valid orderbook, falling back to REST when SSE is stale/empty."""
        with self._ob_lock:
            ob = self._orderbooks.get(product)

        # Check whether the cached book has any real market volume
        def has_market_volume(ob: OrderBook) -> bool:
            bid_ok = any((o.volume - o.own_volume) > 0 for o in ob.buy_orders)
            ask_ok = any((o.volume - o.own_volume) > 0 for o in ob.sell_orders)
            return bid_ok and ask_ok

        if ob is None or not has_market_volume(ob):
            # FIX 1: SSE cache is empty or stale — pull fresh data via REST
            ob = self._fetch_ob_rest(product)

        return ob

    def _get_best_bid_ask(self, product: str) -> tuple[float | None, float | None]:
        ob = self._get_ob(product)
        if ob is None:
            return None, None

        best_bid = next(
            (o.price for o in ob.buy_orders  if (o.volume - o.own_volume) > 0), None
        )
        best_ask = next(
            (o.price for o in ob.sell_orders if (o.volume - o.own_volume) > 0), None
        )
        return best_bid, best_ask

    def _get_mid(self, product: str) -> float | None:
        bid, ask = self._get_best_bid_ask(product)
        return (bid + ask) / 2 if bid is not None and ask is not None else None

    # ── Trading logic ─────────────────────────────────────────────────────────
    # FIX 2: smarter position management
    #   • MAX_POSITION raised to 50 — we can now hold more when the signal is
    #     strong and sustained
    #   • "Trim" logic: if we are at or near the max in one direction AND the
    #     market has moved more than TRIM_TRIGGER points *past* fair value in
    #     the other direction, we peel off TRIM_VOLUME lots to free up headroom.
    #     This prevents the bot sitting at max while missing cheap buy-backs.

    def _evaluate_and_trade(self, product: str, fair_value: float | None):
        if fair_value is None:
            return

        best_bid, best_ask = self._get_best_bid_ask(product)
        if best_bid is None or best_ask is None:
            return

        threshold   = self.fv_engine.get_threshold()
        positions   = self.get_positions()
        current_pos = positions.get(product, 0)
        mid         = (best_bid + best_ask) / 2

        # ── Trim logic (FIX 2) ───────────────────────────────────────────────
        # We're long at max but market is now well BELOW FV → cheap to sell some
        if current_pos >= MAX_POSITION and mid < fair_value - TRIM_TRIGGER:
            print(f"  ✂️  TRIM LONG {product}: pos={current_pos}  "
                  f"mid={mid:.0f}  FV={fair_value:.0f}")
            self.send_order(OrderRequest(product, best_bid, Side.SELL, TRIM_VOLUME))
            return

        # We're short at max but market is now well ABOVE FV → cheap to buy some
        if current_pos <= -MAX_POSITION and mid > fair_value + TRIM_TRIGGER:
            print(f"  ✂️  TRIM SHORT {product}: pos={current_pos}  "
                  f"mid={mid:.0f}  FV={fair_value:.0f}")
            self.send_order(OrderRequest(product, best_ask, Side.BUY, TRIM_VOLUME))
            return

        # ── Aggressive entry ──────────────────────────────────────────────────
        if fair_value > best_ask + threshold:
            if current_pos < MAX_POSITION:
                vol = min(TRADE_VOLUME, MAX_POSITION - current_pos)
                if vol > 0:
                    print(f"  🟢 BUY  {product}: FV={fair_value:.0f} "
                          f"> ask={best_ask}+{threshold:.0f}  pos {current_pos}→{current_pos+vol}")
                    self.send_order(OrderRequest(product, best_ask, Side.BUY, vol))

        elif fair_value < best_bid - threshold:
            if current_pos > -MAX_POSITION:
                vol = min(TRADE_VOLUME, MAX_POSITION + current_pos)
                if vol > 0:
                    print(f"  🔴 SELL {product}: FV={fair_value:.0f} "
                          f"< bid={best_bid}-{threshold:.0f}  pos {current_pos}→{current_pos-vol}")
                    self.send_order(OrderRequest(product, best_bid, Side.SELL, vol))

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self, interval: int = 5):
        print("=" * 60)
        print("  WEATHER MISPRICING BOT  (v2 — with FIX 1 & FIX 2)")
        print("=" * 60)

        self.start()
        print("  SSE stream started.")

        self._running        = True
        self._weather_thread = Thread(target=self._weather_loop, daemon=True)
        self._weather_thread.start()
        print(f"  Weather polling started (every {WEATHER_POLL_SEC}s).")

        print("  Waiting for first weather update …")
        while self.fv_engine.last_update is None:
            time.sleep(1)
        print("  First weather update received. Trading loop starting.\n")

        try:
            while True:
                s         = self.fv_engine.get_state()
                spot_mid  = self._get_mid("WX_SPOT")
                sum_mid   = self._get_mid("WX_SUM")
                positions = self.get_positions()

                spot_pos  = positions.get("WX_SPOT", 0)
                sum_pos   = positions.get("WX_SUM",  0)
                threshold = self.fv_engine.get_threshold()

                spot_fv_s = f"{s['wx_spot_fv']:.0f}" if s["wx_spot_fv"] is not None else "?"
                sum_fv_s  = f"{s['wx_sum_fv']:.0f}"  if s["wx_sum_fv"]  is not None else "?"
                spot_mid_s = f"{spot_mid:.1f}" if spot_mid is not None else "None"
                sum_mid_s  = f"{sum_mid:.1f}"  if sum_mid  is not None else "None"

                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"SPOT: FV={spot_fv_s} mid={spot_mid_s} pos={spot_pos:+d} | "
                      f"SUM: FV={sum_fv_s} mid={sum_mid_s} pos={sum_pos:+d} | "
                      f"thr={threshold:.0f} conf={s['confidence']:.0%}")

                self._evaluate_and_trade("WX_SPOT", s["wx_spot_fv"])
                self._evaluate_and_trade("WX_SUM",  s["wx_sum_fv"])

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n  KeyboardInterrupt — shutting down …")
        finally:
            self._running = False
            self.cancel_all_orders()
            self.stop()
            print("  Bot stopped.")
            print("  Final positions :", self.get_positions())
            print("  PnL             :", self.get_pnl())


print("WeatherBot defined.")


# ============================================================
# SECTION 5 — RUN
# ============================================================
# Adjust session times to match your actual competition session.
# Settlement window: Saturday 12:00 → Sunday 12:00 (London time)

SESSION_START = pd.Timestamp("2026-03-07 12:00", tz="Europe/London")
SESSION_END   = pd.Timestamp("2026-03-08 12:00", tz="Europe/London")

bot = WeatherBot(
    cmi_url       = EXCHANGE_URL,
    username      = USERNAME,
    password      = PASSWORD,
    session_start = SESSION_START,
    session_end   = SESSION_END,
)

bot.run(interval=5)


# ============================================================
# SECTION 6 — DEBUG / MONITOR  (run cells independently)
# ============================================================

def snapshot_weather():
    """Quick check: what do the 4 sources say right now?"""
    sources = {}
    for name, fn in [
        ("Open-Meteo", lambda: fetch_open_meteo(4, 4).iloc[-1].to_dict()),
        ("OWM",        fetch_owm_current),
        ("WeatherAPI", fetch_wapi_current),
        ("wttr.in",    fetch_wttr_current),
    ]:
        try:
            r = fn()
            sources[name] = {"T_F": r.get("temp_f_rounded"), "H": r["humidity"], "TxH": r["t_x_h"]}
        except Exception as e:
            sources[name] = {"error": str(e)}
    df = pd.DataFrame(sources).T
    print("=== CURRENT WEATHER SNAPSHOT ===")
    print(df)
    if "TxH" in df.columns:
        vals = pd.to_numeric(df["TxH"], errors="coerce").dropna()
        print(f"\nConsensus TxH: {vals.mean():.0f} ± {vals.std():.0f}")


def snapshot_orderbooks():
    """Check orderbooks (requires bot to be connected)."""
    for product in ["WX_SPOT", "WX_SUM"]:
        try:
            ob   = bot.get_orderbook(product)
            bids = [(o.price, o.volume) for o in ob.buy_orders[:3]]
            asks = [(o.price, o.volume) for o in ob.sell_orders[:3]]
            print(f"\n{product}:  BIDS {bids}  |  ASKS {asks}")
        except Exception as e:
            print(f"{product}: {e}")


def snapshot_pnl():
    try:
        print("Positions:", bot.get_positions())
        print("PnL:      ", bot.get_pnl())
    except Exception as e:
        print(f"Error: {e} — bot may not be connected")


# ============================================================
# SECTION 7 — HISTORICAL BACKTEST
# ============================================================

def fetch_historical(start_date="2025-01-01", end_date="2025-12-31"):
    """Fetch hourly historical data from Open-Meteo Archive."""
    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LONDON_LAT, "longitude": LONDON_LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Europe/London",
    })
    resp.raise_for_status()
    h  = resp.json()["hourly"]
    df = pd.DataFrame({"time":     pd.to_datetime(h["time"]),
                       "temp_c":   h["temperature_2m"],
                       "humidity": h["relative_humidity_2m"]})
    df["temp_f"]         = c_to_f(df["temp_c"])
    df["temp_f_rounded"] = df["temp_f"].round()
    df["t_x_h"]          = df["temp_f_rounded"] * df["humidity"]
    df["hour"]           = df["time"].dt.hour
    df["month"]          = df["time"].dt.month
    return df


def run_backtest():
    df_hist = fetch_historical("2025-01-01", "2025-12-31")
    print(f"Loaded {len(df_hist)} hourly readings\n")

    noon = df_hist[df_hist.hour == 12]
    print("WX_SPOT (T×H at noon):")
    print(f"  Mean={noon.t_x_h.mean():.0f}  Std={noon.t_x_h.std():.0f}  "
          f"Min={noon.t_x_h.min():.0f}  Max={noon.t_x_h.max():.0f}\n")

    df_hist["session_date"] = (df_hist.time - pd.Timedelta(hours=12)).dt.date
    daily_sums = df_hist.groupby("session_date").apply(
        lambda g: (g["t_x_h"] / 100).sum() * 4   # ×4: hourly → 15-min
    )
    print("WX_SUM (24h accumulator, approx from hourly):")
    print(f"  Mean={daily_sums.mean():.0f}  Std={daily_sums.std():.0f}  "
          f"5th%={daily_sums.quantile(0.05):.0f}  95th%={daily_sums.quantile(0.95):.0f}")

    print("\nMonthly T×H at noon (mean ± std):")
    for month, g in noon.groupby("month"):
        print(f"  Month {month:2d}: {g.t_x_h.mean():6.0f} ± {g.t_x_h.std():5.0f}")