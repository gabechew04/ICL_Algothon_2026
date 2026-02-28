"""
market_maker.py — Generates two-sided quotes around fair value with inventory skew.
"""

import math
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from risk_engine import RiskEngine
    from fair_value_engine import FairValueEngine
    from bot_template import OrderRequest, Side

log = logging.getLogger("algothon.mm")

# (base_spread, base_size, min_edge, max_levels)
DEFAULT_CONFIG: dict[str, tuple[int, int, int, int]] = {
    "TIDE_SPOT":  (15, 5, 3, 2),
    "TIDE_SWING": (10, 3, 2, 2),
    "WX_SPOT":    (15, 5, 3, 2),
    "WX_SUM":     (10, 3, 2, 2),
    "LHR_COUNT":  (20, 5, 5, 2),
    "LHR_INDEX":  (15, 3, 3, 2),
    "LON_ETF":    (25, 5, 5, 3),
    "LON_FLY":    (20, 3, 5, 2),
}


class MarketMaker:
    """Generates two-sided quotes around fair value with adaptive width."""

    def __init__(self, risk: "RiskEngine", fv_engine: "FairValueEngine",
                 config: dict | None = None):
        self.risk = risk
        self.fv = fv_engine
        self.config = config or DEFAULT_CONFIG

    def generate_quotes(self, product: str, tick_size: float,
                        market_mid: Optional[float] = None) -> list:
        """Generate bid/ask orders for a product."""
        from bot_template import OrderRequest, Side

        fv = self.fv.fair_values.get(product)
        conf = self.fv.confidence.get(product, 0)

        if fv is None or conf < 0.1:
            if market_mid is not None:
                fv = market_mid
                conf = 0.2
            else:
                return []

        base_width, base_size, min_edge, max_levels = self.config.get(
            product, (15, 5, 3, 2)
        )

        # Wider spread when less confident
        width = base_width / max(conf, 0.15)
        width = max(width, min_edge * 2)

        # Inventory skew: shift mid to reduce position
        skew = self.risk.inventory_skew(product)
        skew_offset = skew * width * 0.3
        adjusted_fv = fv - skew_offset

        orders = []
        bid_size, ask_size = self.risk.quote_size(product, base_size)

        for level in range(max_levels):
            level_offset = width * (0.5 + level * 0.5)

            bid_price = math.floor((adjusted_fv - level_offset) / tick_size) * tick_size
            ask_price = math.ceil((adjusted_fv + level_offset) / tick_size) * tick_size

            lvl_bid = max(1, bid_size // (level + 1))
            lvl_ask = max(1, ask_size // (level + 1))

            if bid_price > 0 and lvl_bid > 0:
                orders.append(OrderRequest(product, bid_price, Side.BUY, lvl_bid))
            if ask_price > bid_price and lvl_ask > 0:
                orders.append(OrderRequest(product, ask_price, Side.SELL, lvl_ask))

        return orders
