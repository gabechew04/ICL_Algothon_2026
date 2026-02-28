"""
simple_bot.py — Bot simplifié et lisible.

STRATÉGIE EN 3 COUCHES (par ordre de priorité):
  1. ALPHA DIRECTIONNEL — On connaît les données réelles → on trade quand le marché se trompe
  2. ETF ARBITRAGE     — LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT → arb si écart
  3. OPTIONS PRICING   — LON_FLY est complexe → on le price mieux que les autres

Chaque trade est loggé avec sa RAISON et son EDGE attendu.
"""

import time
import math
import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from threading import Thread

from bot_template import BaseBot, OrderBook, OrderRequest, Side, Trade, Order

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

log = logging.getLogger("bot")

# Colored trade logger — on voit immédiatement ce qui se passe
trade_log = logging.getLogger("TRADES")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Tout est ici, facile à tweaker
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # Position limits
    MAX_POS = 100               # Limite exchange
    SOFT_POS = 70               # On ralentit à partir de là
    
    # Alpha directionnel — combien d'edge minimum pour trader
    ALPHA_MIN_EDGE = {
        "TIDE_SPOT": 20,        # En ticks (= prix)
        "WX_SPOT":   15,
        "LHR_COUNT": 30,
        "WX_SUM":    15,
        "TIDE_SWING": 15,
        "LHR_INDEX": 20,
    }
    ALPHA_SIZE = 5              # Volume par trade alpha
    
    # ETF Arb
    ARB_MIN_EDGE = 8            # Edge minimum pour arber l'ETF
    ARB_SIZE = 3
    
    # Options
    OPT_MIN_EDGE = 20           # Edge minimum sur LON_FLY
    OPT_SIZE = 3
    OPT_SIMULATIONS = 20_000    # Monte Carlo sims
    
    # Timing
    LOOP_INTERVAL = 3.0         # Secondes entre chaque cycle
    DATA_REFRESH = 120.0        # Secondes entre fetch data


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHER — Sources de données réelles
# ═══════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """Fetch weather, tides, flights. Chaque source = un edge potentiel."""
    
    def __init__(self, aero_key: str = ""):
        self.aero_key = aero_key
        self.weather: pd.DataFrame | None = None
        self.thames: pd.DataFrame | None = None
        self.flights: dict | None = None
        self._last = {"weather": 0.0, "thames": 0.0, "flights": 0.0}
    
    def fetch_all(self, min_interval: float = 120.0):
        now = time.time()
        if now - self._last["weather"] > min_interval:
            self._fetch_weather()
        if now - self._last["thames"] > min_interval:
            self._fetch_thames()
        if now - self._last["flights"] > max(min_interval, 300):  # Flights: limité
            self._fetch_flights()
    
    def _fetch_weather(self):
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": 51.5074, "longitude": -0.1278,
                    "minutely_15": "temperature_2m,relative_humidity_2m",
                    "past_minutely_15": 96, "forecast_minutely_15": 96,
                    "timezone": "Europe/London",
                }, timeout=10)
            resp.raise_for_status()
            m = resp.json()["minutely_15"]
            self.weather = pd.DataFrame({
                "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
                "temp_c": m["temperature_2m"],
                "humidity": m["relative_humidity_2m"],
            })
            self.weather["temp_f"] = self.weather["temp_c"] * 9/5 + 32
            self.weather["wx_metric"] = self.weather["temp_f"] * self.weather["humidity"]
            self._last["weather"] = time.time()
            log.info(f"📡 Weather: {len(self.weather)} points fetched")
        except Exception as e:
            log.warning(f"Weather fetch failed: {e}")
    
    def _fetch_thames(self):
        try:
            resp = requests.get(
                "https://environment.data.gov.uk/flood-monitoring/id/measures/"
                "0006-level-tidal_level-i-15_min-mAOD/readings",
                params={"_sorted": "", "_limit": 200}, timeout=10)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            df = pd.DataFrame(items)[["dateTime", "value"]].rename(
                columns={"dateTime": "time", "value": "level"})
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
            self.thames = df.sort_values("time").reset_index(drop=True)
            self._last["thames"] = time.time()
            log.info(f"📡 Thames: {len(self.thames)} readings fetched")
        except Exception as e:
            log.warning(f"Thames fetch failed: {e}")
    
    def _fetch_flights(self):
        if not self.aero_key:
            return
        try:
            host = "aerodatabox.p.rapidapi.com"
            resp = requests.get(
                f"https://{host}/flights/airports/iata/LHR"
                f"?offsetMinutes=-720&durationMinutes=720&direction=Both",
                headers={"x-rapidapi-host": host, "x-rapidapi-key": self.aero_key},
                timeout=15)
            resp.raise_for_status()
            self.flights = resp.json()
            n_arr = len(self.flights.get("arrivals", []))
            n_dep = len(self.flights.get("departures", []))
            self._last["flights"] = time.time()
            log.info(f"📡 Flights: {n_arr} arr + {n_dep} dep = {n_arr+n_dep} total")
        except Exception as e:
            log.warning(f"Flights fetch failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE — Estimation simple et transparente de chaque produit
# ═══════════════════════════════════════════════════════════════════════════════

class FairValue:
    """
    Calcule le fair value de chaque produit.
    Retourne (value, confidence) où confidence ∈ [0, 1].
    Plus la confidence est haute, plus on trade agressivement.
    """
    
    def __init__(self, data: DataFetcher, session_start: datetime, session_end: datetime):
        self.data = data
        self.session_start = session_start
        self.session_end = session_end
        self.values: dict[str, float] = {}
        self.confidence: dict[str, float] = {}
    
    def update(self):
        """Recalcule tout. Appelé après chaque data refresh."""
        self._calc_tide_spot()
        self._calc_tide_swing()
        self._calc_wx_spot()
        self._calc_wx_sum()
        self._calc_lhr_count()
        self._calc_lhr_index()
        self._calc_etf()
        self._calc_fly()
        
        # Log résumé
        summary = " | ".join(
            f"{k}={v:.0f}(c={self.confidence.get(k,0):.1f})"
            for k, v in self.values.items()
        )
        log.info(f"📊 FV: {summary}")
    
    # ── TIDE_SPOT: abs(water_level) × 1000 à 12h dimanche ──────────────────
    def _calc_tide_spot(self):
        df = self.data.thames
        if df is None or len(df) < 10:
            self.confidence["TIDE_SPOT"] = 0.0
            return
        
        # Extrapolation sinusoïdale (marée semi-diurne T=12.42h)
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
            target_h = (self.session_end - t0).total_seconds() / 3600.0
            predicted = coeffs[0] + coeffs[1]*np.sin(omega*target_h) + coeffs[2]*np.cos(omega*target_h)
        except:
            predicted = df["level"].iloc[-1]
        
        self.values["TIDE_SPOT"] = abs(predicted) * 1000
        
        # Confidence: plus on est proche de la fin, plus on est sûr
        hours_left = abs((self.session_end - df["time"].iloc[-1]).total_seconds()) / 3600
        self.confidence["TIDE_SPOT"] = max(0.2, min(0.95, 0.9 - 0.03 * hours_left))
    
    # ── TIDE_SWING: somme des payoffs strangle sur 15min changes ────────────
    def _calc_tide_swing(self):
        df = self.data.thames
        if df is None or len(df) < 2:
            self.confidence["TIDE_SWING"] = 0.0
            return
        
        start_ts = pd.Timestamp(self.session_start).tz_localize("Europe/London") \
            if pd.Timestamp(self.session_start).tzinfo is None \
            else pd.Timestamp(self.session_start)
        end_ts = pd.Timestamp(self.session_end).tz_localize("Europe/London") \
            if pd.Timestamp(self.session_end).tzinfo is None \
            else pd.Timestamp(self.session_end)
        
        session = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        
        TOTAL_INTERVALS = 96  # 24h / 15min
        
        if len(session) >= 2:
            diffs_cm = session["level"].diff().abs().dropna() * 100
            observed = sum(self._strangle(d) for d in diffs_cm)
            n_obs = len(diffs_cm)
            remaining = max(0, TOTAL_INTERVALS - n_obs)
            avg_payoff = observed / n_obs if n_obs > 0 else 5.0
            self.values["TIDE_SWING"] = observed + avg_payoff * remaining
            self.confidence["TIDE_SWING"] = min(0.9, 0.3 + 0.6 * (n_obs / TOTAL_INTERVALS))
        else:
            # Pas encore de données session → estimation historique
            diffs_cm = df["level"].diff().abs().dropna() * 100
            if len(diffs_cm) > 0:
                avg = np.mean([self._strangle(d) for d in diffs_cm])
                self.values["TIDE_SWING"] = avg * TOTAL_INTERVALS
                self.confidence["TIDE_SWING"] = 0.25
            else:
                self.confidence["TIDE_SWING"] = 0.0
    
    @staticmethod
    def _strangle(diff_cm: float) -> float:
        """Strangle payoff: strikes 20 et 25."""
        return max(0, 20 - diff_cm) + max(0, diff_cm - 25)
    
    # ── WX_SPOT: temp_F × humidity à 12h dimanche ──────────────────────────
    def _calc_wx_spot(self):
        df = self.data.weather
        if df is None or df.empty:
            self.confidence["WX_SPOT"] = 0.0
            return
        
        target = pd.Timestamp(self.session_end)
        if target.tzinfo is None:
            target = target.tz_localize("Europe/London")
        
        idx = (df["time"] - target).abs().idxmin()
        row = df.iloc[idx]
        self.values["WX_SPOT"] = row["wx_metric"]
        
        hours_away = abs((row["time"] - target).total_seconds()) / 3600
        self.confidence["WX_SPOT"] = max(0.3, min(0.95, 0.95 - 0.04 * hours_away))
    
    # ── WX_SUM: sum(temp_F × humidity) / 100 over 24h ──────────────────────
    def _calc_wx_sum(self):
        df = self.data.weather
        if df is None or df.empty:
            self.confidence["WX_SUM"] = 0.0
            return
        
        start = pd.Timestamp(self.session_start)
        end = pd.Timestamp(self.session_end)
        if start.tzinfo is None: start = start.tz_localize("Europe/London")
        if end.tzinfo is None: end = end.tz_localize("Europe/London")
        
        mask = (df["time"] >= start) & (df["time"] <= end)
        self.values["WX_SUM"] = df.loc[mask, "wx_metric"].sum() / 100
        
        now = pd.Timestamp.now("Europe/London")
        n_obs = ((df["time"] >= start) & (df["time"] <= now)).sum()
        self.confidence["WX_SUM"] = min(0.9, 0.4 + 0.5 * (n_obs / 96))
    
    # ── LHR_COUNT: arrivals + departures ────────────────────────────────────
    def _calc_lhr_count(self):
        f = self.data.flights
        if f is None:
            self.confidence["LHR_COUNT"] = 0.0
            return
        self.values["LHR_COUNT"] = len(f.get("arrivals", [])) + len(f.get("departures", []))
        self.confidence["LHR_COUNT"] = 0.5
    
    # ── LHR_INDEX: imbalance metric ─────────────────────────────────────────
    def _calc_lhr_index(self):
        f = self.data.flights
        if f is None:
            self.confidence["LHR_INDEX"] = 0.0
            return
        arr = len(f.get("arrivals", []))
        dep = len(f.get("departures", []))
        if arr + dep > 0:
            self.values["LHR_INDEX"] = abs(arr - dep) / max(arr + dep, 1) * 48 * 100
        else:
            self.values["LHR_INDEX"] = 0
        self.confidence["LHR_INDEX"] = 0.3
    
    # ── LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT ──────────────────────────
    def _calc_etf(self):
        parts = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        if all(p in self.values for p in parts):
            self.values["LON_ETF"] = sum(self.values[p] for p in parts)
            self.confidence["LON_ETF"] = min(self.confidence.get(p, 0) for p in parts)
        else:
            self.confidence["LON_ETF"] = 0.0
    
    # ── LON_FLY: options package on ETF ─────────────────────────────────────
    def _calc_fly(self):
        if "LON_ETF" not in self.values:
            self.confidence["LON_FLY"] = 0.0
            return
        
        etf = self.values["LON_ETF"]
        conf = self.confidence.get("LON_ETF", 0)
        
        if conf > 0.6:
            # Haute confiance → prix intrinsèque
            self.values["LON_FLY"] = self._fly_payoff(etf)
            self.confidence["LON_FLY"] = conf * 0.85
        else:
            # Basse confiance → Monte Carlo
            std = max(300 * (1 - conf), 100)
            samples = np.random.normal(etf, std, Config.OPT_SIMULATIONS)
            payoffs = np.array([self._fly_payoff(s) for s in samples])
            self.values["LON_FLY"] = float(np.mean(payoffs))
            self.confidence["LON_FLY"] = conf * 0.5
    
    @staticmethod
    def _fly_payoff(etf: float) -> float:
        """2×Put(6200) + Call(6200) − 2×Call(6600) + 3×Call(7000)"""
        put = lambda k: max(0.0, k - etf)
        call = lambda k: max(0.0, etf - k)
        return 2*put(6200) + call(6200) - 2*call(6600) + 3*call(7000)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGIES — Chacune est indépendante et loggée
# ═══════════════════════════════════════════════════════════════════════════════

class Strategies:
    """
    Chaque méthode retourne une liste d'OrderRequest.
    Chaque trade est loggé avec sa raison.
    """
    
    def __init__(self, fv: FairValue, positions: dict[str, int]):
        self.fv = fv
        self.pos = positions  # Mis à jour depuis l'exchange
    
    def _can_buy(self, product: str, qty: int) -> int:
        pos = self.pos.get(product, 0)
        return max(0, min(qty, Config.MAX_POS - pos))
    
    def _can_sell(self, product: str, qty: int) -> int:
        pos = self.pos.get(product, 0)
        return max(0, min(qty, Config.MAX_POS + pos))
    
    def _size_for_confidence(self, product: str, base_size: int) -> int:
        """Plus on est confiant, plus on trade gros."""
        conf = self.fv.confidence.get(product, 0)
        pos = abs(self.pos.get(product, 0))
        
        # Réduire la taille si on est déjà très positionné
        pos_factor = max(0.2, 1.0 - pos / Config.MAX_POS)
        
        return max(1, int(base_size * conf * pos_factor))
    
    # ── STRATÉGIE 1: Alpha Directionnel ─────────────────────────────────────
    def alpha_directional(self, orderbooks: dict[str, OrderBook]) -> list[OrderRequest]:
        """
        Si notre fair value est loin du marché → on prend position.
        C'est le cœur de notre edge: on a les VRAIES données.
        """
        orders = []
        
        for product in ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT", "WX_SUM", "TIDE_SWING", "LHR_INDEX"]:
            if product not in self.fv.values or product not in orderbooks:
                continue
            
            fv = self.fv.values[product]
            conf = self.fv.confidence.get(product, 0)
            ob = orderbooks[product]
            min_edge = Config.ALPHA_MIN_EDGE.get(product, 20)
            
            if conf < 0.2:
                continue  # Pas assez confiant, skip
            
            # Ajuster l'edge requis en fonction de la confidence
            required_edge = min_edge / max(conf, 0.3)
            
            size = self._size_for_confidence(product, Config.ALPHA_SIZE)
            
            # Si le marché vend en dessous de notre FV → acheter
            if ob.sell_orders:
                best_ask = ob.sell_orders[0].price
                edge = fv - best_ask
                if edge > required_edge:
                    vol = self._can_buy(product, size)
                    if vol > 0:
                        orders.append(OrderRequest(product, best_ask, Side.BUY, vol))
                        trade_log.info(
                            f"🟢 ALPHA BUY {product} × {vol} @ {best_ask} "
                            f"| FV={fv:.0f} edge=+{edge:.0f} conf={conf:.1f}"
                        )
            
            # Si le marché achète au-dessus de notre FV → vendre
            if ob.buy_orders:
                best_bid = ob.buy_orders[0].price
                edge = best_bid - fv
                if edge > required_edge:
                    vol = self._can_sell(product, size)
                    if vol > 0:
                        orders.append(OrderRequest(product, best_bid, Side.SELL, vol))
                        trade_log.info(
                            f"🔴 ALPHA SELL {product} × {vol} @ {best_bid} "
                            f"| FV={fv:.0f} edge=+{edge:.0f} conf={conf:.1f}"
                        )
        
        return orders
    
    # ── STRATÉGIE 2: ETF Arbitrage ──────────────────────────────────────────
    def etf_arb(self, orderbooks: dict[str, OrderBook]) -> list[OrderRequest]:
        """
        LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT
        Si le prix de l'ETF ≠ somme des composants → arbitrage.
        C'est du "free money" (si on arrive à exécuter les deux legs).
        """
        orders = []
        ETF = "LON_ETF"
        PARTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        
        if ETF not in orderbooks or not all(p in orderbooks for p in PARTS):
            return orders
        
        etf_ob = orderbooks[ETF]
        part_obs = {p: orderbooks[p] for p in PARTS}
        
        # CAS 1: ETF trop cher → vendre ETF, acheter composants
        all_asks = all(part_obs[p].sell_orders for p in PARTS)
        if all_asks and etf_ob.buy_orders:
            comp_cost = sum(part_obs[p].sell_orders[0].price for p in PARTS)
            etf_bid = etf_ob.buy_orders[0].price
            edge = etf_bid - comp_cost
            
            if edge > Config.ARB_MIN_EDGE:
                vol = min(Config.ARB_SIZE,
                          self._can_sell(ETF, Config.ARB_SIZE),
                          *[self._can_buy(p, Config.ARB_SIZE) for p in PARTS])
                if vol > 0:
                    orders.append(OrderRequest(ETF, etf_bid, Side.SELL, vol))
                    for p in PARTS:
                        orders.append(OrderRequest(p, part_obs[p].sell_orders[0].price, Side.BUY, vol))
                    trade_log.info(
                        f"⚡ ARB: Sell ETF @ {etf_bid}, Buy parts @ {comp_cost} "
                        f"| edge={edge:.0f}"
                    )
        
        # CAS 2: ETF trop pas cher → acheter ETF, vendre composants
        all_bids = all(part_obs[p].buy_orders for p in PARTS)
        if all_bids and etf_ob.sell_orders:
            comp_value = sum(part_obs[p].buy_orders[0].price for p in PARTS)
            etf_ask = etf_ob.sell_orders[0].price
            edge = comp_value - etf_ask
            
            if edge > Config.ARB_MIN_EDGE:
                vol = min(Config.ARB_SIZE,
                          self._can_buy(ETF, Config.ARB_SIZE),
                          *[self._can_sell(p, Config.ARB_SIZE) for p in PARTS])
                if vol > 0:
                    orders.append(OrderRequest(ETF, etf_ask, Side.BUY, vol))
                    for p in PARTS:
                        orders.append(OrderRequest(p, part_obs[p].buy_orders[0].price, Side.SELL, vol))
                    trade_log.info(
                        f"⚡ ARB: Buy ETF @ {etf_ask}, Sell parts @ {comp_value} "
                        f"| edge={edge:.0f}"
                    )
        
        return orders
    
    # ── STRATÉGIE 3: Options Pricing (LON_FLY) ─────────────────────────────
    def options_mispricing(self, orderbooks: dict[str, OrderBook]) -> list[OrderRequest]:
        """
        LON_FLY = 2×Put(6200) + Call(6200) − 2×Call(6600) + 3×Call(7000)
        
        On utilise notre estimation de LON_ETF pour pricer cette structure.
        Si le marché la price différemment → on trade.
        """
        orders = []
        
        if "LON_FLY" not in self.fv.values or "LON_FLY" not in orderbooks:
            return orders
        
        fly_fv = self.fv.values["LON_FLY"]
        fly_conf = self.fv.confidence.get("LON_FLY", 0)
        ob = orderbooks["LON_FLY"]
        
        if fly_conf < 0.15:
            return orders
        
        required_edge = Config.OPT_MIN_EDGE / max(fly_conf, 0.2)
        size = self._size_for_confidence("LON_FLY", Config.OPT_SIZE)
        
        # Marché vend en dessous de notre modèle → acheter
        if ob.sell_orders:
            best_ask = ob.sell_orders[0].price
            edge = fly_fv - best_ask
            if edge > required_edge:
                vol = self._can_buy("LON_FLY", size)
                if vol > 0:
                    orders.append(OrderRequest("LON_FLY", best_ask, Side.BUY, vol))
                    etf_fv = self.fv.values.get("LON_ETF", 0)
                    trade_log.info(
                        f"🎯 OPT BUY LON_FLY × {vol} @ {best_ask} "
                        f"| model={fly_fv:.0f} edge=+{edge:.0f} "
                        f"ETF_FV={etf_fv:.0f} conf={fly_conf:.1f}"
                    )
        
        # Marché achète au-dessus de notre modèle → vendre
        if ob.buy_orders:
            best_bid = ob.buy_orders[0].price
            edge = best_bid - fly_fv
            if edge > required_edge:
                vol = self._can_sell("LON_FLY", size)
                if vol > 0:
                    orders.append(OrderRequest("LON_FLY", best_bid, Side.SELL, vol))
                    etf_fv = self.fv.values.get("LON_ETF", 0)
                    trade_log.info(
                        f"🎯 OPT SELL LON_FLY × {vol} @ {best_bid} "
                        f"| model={fly_fv:.0f} edge=+{edge:.0f} "
                        f"ETF_FV={etf_fv:.0f} conf={fly_conf:.1f}"
                    )
        
        return orders
    
    # ── STRATÉGIE BONUS: Alpha sur LON_ETF directement ──────────────────────
    def etf_alpha(self, orderbooks: dict[str, OrderBook]) -> list[OrderRequest]:
        """
        On connaît les composants → on trade aussi l'ETF directement
        quand il est mispriced vs notre modèle.
        """
        orders = []
        if "LON_ETF" not in self.fv.values or "LON_ETF" not in orderbooks:
            return orders
        
        fv = self.fv.values["LON_ETF"]
        conf = self.fv.confidence.get("LON_ETF", 0)
        ob = orderbooks["LON_ETF"]
        
        if conf < 0.3:
            return orders
        
        min_edge = 25 / max(conf, 0.3)
        size = self._size_for_confidence("LON_ETF", 4)
        
        if ob.sell_orders:
            best_ask = ob.sell_orders[0].price
            edge = fv - best_ask
            if edge > min_edge:
                vol = self._can_buy("LON_ETF", size)
                if vol > 0:
                    orders.append(OrderRequest("LON_ETF", best_ask, Side.BUY, vol))
                    trade_log.info(
                        f"🟢 ALPHA BUY LON_ETF × {vol} @ {best_ask} "
                        f"| FV={fv:.0f} edge=+{edge:.0f} conf={conf:.1f}"
                    )
        
        if ob.buy_orders:
            best_bid = ob.buy_orders[0].price
            edge = best_bid - fv
            if edge > min_edge:
                vol = self._can_sell("LON_ETF", size)
                if vol > 0:
                    orders.append(OrderRequest("LON_ETF", best_bid, Side.SELL, vol))
                    trade_log.info(
                        f"🔴 ALPHA SELL LON_ETF × {vol} @ {best_bid} "
                        f"| FV={fv:.0f} edge=+{edge:.0f} conf={conf:.1f}"
                    )
        
        return orders


# ═══════════════════════════════════════════════════════════════════════════════
# BOT PRINCIPAL — Orchestre tout
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleBot(BaseBot):
    """
    Bot simple et lisible.
    Cycle: fetch data → update FV → check strategies → execute.
    """
    
    def __init__(self, cmi_url, username, password, aero_key,
                 session_start, session_end,
                 loop_interval=3.0, data_interval=120.0):
        super().__init__(cmi_url, username, password)
        
        self.session_start = session_start
        self.session_end = session_end
        self.loop_interval = loop_interval
        self.data_interval = data_interval
        
        self.data = DataFetcher(aero_key)
        self.fv = FairValue(self.data, session_start, session_end)
        
        self.orderbooks: dict[str, OrderBook] = {}
        self.positions: dict[str, int] = defaultdict(int)
        self.strategies: Strategies | None = None
        
        # Tracking PnL
        self._trade_count = 0
        self._last_data_fetch = 0.0
    
    # ── SSE Callbacks ───────────────────────────────────────────────────────
    def on_orderbook(self, orderbook: OrderBook):
        self.orderbooks[orderbook.product] = orderbook
    
    def on_trades(self, trade: Trade):
        self._trade_count += 1
        side = "BOUGHT" if trade.buyer == self.username else "SOLD"
        trade_log.info(
            f"💰 FILL: {side} {trade.product} × {trade.volume} @ {trade.price}"
        )
    
    # ── Main Loop ───────────────────────────────────────────────────────────
    def run(self):
        self.start()  # Start SSE stream
        log.info("🚀 Bot started!")
        
        try:
            while True:
                self._cycle()
                time.sleep(self.loop_interval)
        except KeyboardInterrupt:
            log.info("Bot stopping...")
        finally:
            self.stop()
    
    def _cycle(self):
        """Un cycle complet du bot."""
        now = time.time()
        
        # 1. Refresh data si nécessaire
        if now - self._last_data_fetch > self.data_interval:
            self.data.fetch_all(self.data_interval)
            self.fv.update()
            self._last_data_fetch = now
        
        # 2. Sync positions depuis l'exchange
        try:
            self.positions = defaultdict(int, self.get_positions())
        except Exception as e:
            log.warning(f"Position fetch failed: {e}")
            return
        
        # 3. Cancel old orders avant de placer les nouveaux
        try:
            self.cancel_all_orders()
        except Exception as e:
            log.warning(f"Cancel failed: {e}")
        
        # 4. Créer les stratégies avec les positions actuelles
        self.strategies = Strategies(self.fv, dict(self.positions))
        
        # 5. Collecter tous les ordres des 4 stratégies
        all_orders = []
        
        # Stratégie 1: Alpha directionnel (la plus importante)
        all_orders.extend(self.strategies.alpha_directional(self.orderbooks))
        
        # Stratégie 2: ETF alpha
        all_orders.extend(self.strategies.etf_alpha(self.orderbooks))
        
        # Stratégie 3: ETF arbitrage
        all_orders.extend(self.strategies.etf_arb(self.orderbooks))
        
        # Stratégie 4: Options mispricing
        all_orders.extend(self.strategies.options_mispricing(self.orderbooks))
        
        # 6. Exécuter
        if all_orders:
            log.info(f"📤 Sending {len(all_orders)} orders")
            self.send_orders(all_orders)
        
        # 7. Status périodique
        if int(now) % 60 < self.loop_interval:
            self._log_status()
    
    def _log_status(self):
        """Log un résumé de l'état du bot."""
        pos_str = " | ".join(
            f"{k}={v:+d}" for k, v in self.positions.items() if v != 0
        )
        try:
            pnl = self.get_pnl()
            pnl_str = f"PnL={pnl}" if pnl else ""
        except:
            pnl_str = ""
        
        log.info(f"📈 Status: {pos_str or 'flat'} | trades={self._trade_count} {pnl_str}")
