"""
risk_engine.py — Tracks positions, enforces limits, computes inventory skew.
"""

import threading
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot_template import Trade, Side

log = logging.getLogger("algothon.risk")


@dataclass
class RiskState:
    positions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    max_position: int = 100
    fill_history: list = field(default_factory=list)
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

    def record_fill(self, trade: "Trade", our_side: "Side"):
        """Record a fill and update average entry price."""
        from bot_template import Side
        with self._lock:
            self.state.fill_history.append(trade)
            product = trade.product
            pos = self.state.positions[product]
            qty = trade.volume if our_side == Side.BUY else -trade.volume

            old_pos = pos
            new_pos = pos + qty
            self.state.positions[product] = new_pos

            if abs(new_pos) > abs(old_pos):
                old_cost = self.state.avg_entry[product] * abs(old_pos)
                new_cost = trade.price * abs(qty)
                if new_pos != 0:
                    self.state.avg_entry[product] = (old_cost + new_cost) / abs(new_pos)

    def get_position(self, product: str) -> int:
        return self.state.positions.get(product, 0)

    def can_buy(self, product: str, volume: int) -> int:
        pos = self.get_position(product)
        return max(0, min(volume, self.state.max_position - pos))

    def can_sell(self, product: str, volume: int) -> int:
        pos = self.get_position(product)
        return max(0, min(volume, self.state.max_position + pos))

    def inventory_skew(self, product: str) -> float:
        """Returns skew factor [-1, 1]. Positive = long → should sell."""
        return self.get_position(product) / self.state.max_position

    def quote_size(self, product: str, base_size: int) -> tuple[int, int]:
        """Returns (bid_size, ask_size) adjusted for inventory."""
        skew = self.inventory_skew(product)
        bid_size = max(1, int(base_size * (1 - max(0, skew))))
        ask_size = max(1, int(base_size * (1 + min(0, skew))))
        bid_size = self.can_buy(product, bid_size)
        ask_size = self.can_sell(product, ask_size)
        return bid_size, ask_size
