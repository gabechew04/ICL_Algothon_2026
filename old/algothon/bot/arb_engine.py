"""
arb_engine.py — Detects and trades ETF vs component mispricings.

LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT

If ETF trades cheap vs sum of components → buy ETF, sell components.
If ETF trades rich  vs sum of components → sell ETF, buy components.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from risk_engine import RiskEngine
    from bot_template import OrderBook, OrderRequest

log = logging.getLogger("algothon.arb")

ETF_SYM = "LON_ETF"
COMPONENTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]


class ArbEngine:
    """Detects and executes ETF vs component arbitrage."""

    def __init__(self, risk: "RiskEngine", min_edge: float = 10.0):
        self.risk = risk
        self.min_edge = min_edge

    def check_arb(self, orderbooks: dict) -> list:
        """Return a list of orders if an arb opportunity exists."""
        from bot_template import OrderRequest, Side

        if ETF_SYM not in orderbooks:
            return []
        if not all(c in orderbooks for c in COMPONENTS):
            return []

        etf_ob = orderbooks[ETF_SYM]
        comp_obs = {c: orderbooks[c] for c in COMPONENTS}
        orders = []

        # ── Leg 1: Sell ETF, Buy components ────────────────────────────────
        comp_asks_ok = all(comp_obs[c].sell_orders for c in COMPONENTS)
        if comp_asks_ok and etf_ob.buy_orders:
            comp_ask_total = sum(comp_obs[c].sell_orders[0].price for c in COMPONENTS)
            etf_bid = etf_ob.buy_orders[0].price
            edge = etf_bid - comp_ask_total

            if edge > self.min_edge:
                vol = min(3,
                          self.risk.can_sell(ETF_SYM, 3),
                          *[self.risk.can_buy(c, 3) for c in COMPONENTS])
                if vol > 0:
                    orders.append(OrderRequest(ETF_SYM, etf_bid, Side.SELL, vol))
                    for c in COMPONENTS:
                        orders.append(OrderRequest(c, comp_obs[c].sell_orders[0].price, Side.BUY, vol))
                    log.info(f"ARB: Sell ETF @ {etf_bid}, Buy components — edge={edge:.0f}")

        # ── Leg 2: Buy ETF, Sell components ────────────────────────────────
        comp_bids_ok = all(comp_obs[c].buy_orders for c in COMPONENTS)
        if comp_bids_ok and etf_ob.sell_orders:
            comp_bid_total = sum(comp_obs[c].buy_orders[0].price for c in COMPONENTS)
            etf_ask = etf_ob.sell_orders[0].price
            edge = comp_bid_total - etf_ask

            if edge > self.min_edge:
                vol = min(3,
                          self.risk.can_buy(ETF_SYM, 3),
                          *[self.risk.can_sell(c, 3) for c in COMPONENTS])
                if vol > 0:
                    orders.append(OrderRequest(ETF_SYM, etf_ask, Side.BUY, vol))
                    for c in COMPONENTS:
                        orders.append(OrderRequest(c, comp_obs[c].buy_orders[0].price, Side.SELL, vol))
                    log.info(f"ARB: Buy ETF @ {etf_ask}, Sell components — edge={edge:.0f}")

        return orders
