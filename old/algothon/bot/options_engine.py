"""
options_engine.py — Prices LON_FLY and detects mispricing vs the market.

Structure: 2×Put(6200) + Call(6200) − 2×Call(6600) + 3×Call(7000)
"""

import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fair_value_engine import FairValueEngine
    from risk_engine import RiskEngine
    from bot_template import OrderBook, OrderRequest

log = logging.getLogger("algothon.options")


class OptionsEngine:
    """Monte Carlo pricer + mispricing detector for LON_FLY."""

    def __init__(self, fv_engine: "FairValueEngine", risk: "RiskEngine"):
        self.fv = fv_engine
        self.risk = risk

    # ── Pricing ─────────────────────────────────────────────────────────────

    def price_fly(self, etf_mean: float, etf_std: float = 300,
                  n_sims: int = 20_000) -> float:
        """Monte Carlo price of the options package."""
        samples = np.maximum(np.random.normal(etf_mean, etf_std, n_sims), 0)
        payoffs = np.array([self._payoff(s) for s in samples])
        return float(np.mean(payoffs))

    def _payoff(self, etf: float) -> float:
        put  = lambda k: max(0.0, k - etf)
        call = lambda k: max(0.0, etf - k)
        return (
            2 * put(6200)
            + 1 * call(6200)
            - 2 * call(6600)
            + 3 * call(7000)
        )

    # ── Mispricing check ─────────────────────────────────────────────────────

    def check_mispricing(self, fly_ob: "OrderBook") -> list:
        """Return orders if LON_FLY market price diverges from model."""
        from bot_template import OrderRequest, Side

        etf_fv = self.fv.fair_values.get("LON_ETF")
        if etf_fv is None:
            return []

        etf_conf = self.fv.confidence.get("LON_ETF", 0)
        etf_std = max(300 * (1 - etf_conf), 50)
        model_price = self.price_fly(etf_fv, etf_std)

        min_edge = 15
        orders = []

        if fly_ob.sell_orders:
            best_ask = fly_ob.sell_orders[0].price
            if model_price - best_ask > min_edge:
                vol = self.risk.can_buy("LON_FLY", 3)
                if vol > 0:
                    orders.append(OrderRequest("LON_FLY", best_ask, Side.BUY, vol))
                    log.info(f"OPT: Buy FLY @ {best_ask} (model={model_price:.0f})")

        if fly_ob.buy_orders:
            best_bid = fly_ob.buy_orders[0].price
            if best_bid - model_price > min_edge:
                vol = self.risk.can_sell("LON_FLY", 3)
                if vol > 0:
                    orders.append(OrderRequest("LON_FLY", best_bid, Side.SELL, vol))
                    log.info(f"OPT: Sell FLY @ {best_bid} (model={model_price:.0f})")

        return orders
