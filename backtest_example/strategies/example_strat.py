from typing import Optional, Dict, Any


def make_signal(
    symbol: str,
    action: str,
    quantity: float,
    price: float,
    confidence: Optional[float] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    sig: Dict[str, Any] = {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": price,
    }
    if confidence is not None:
        sig["confidence"] = float(confidence)
    if reason is not None:
        sig["reason"] = str(reason)
    return sig


class Strategy:
    def __init__(self, **kwargs):
        self.symbol = "NVDA"
        self.quantity = float(kwargs.get("quantity", 1))
        self._next_action = "buy"

    @staticmethod
    def _select_symbol(target: str, bars: dict) -> tuple[str, Optional[dict]]:
        data = bars.get(target)
        if data is not None:
            return target, data
        for key, value in bars.items():
            if "NVDA" in key.upper():
                return key, value
        return target, None

    def generate_signal(self, team: dict, bars: dict, current_prices: dict):
        symbol, data = self._select_symbol(self.symbol, bars)

        closes: list[float] = []
        if data:
            closes_raw = data.get("close") or []
            closes = [float(x) for x in closes_raw if x is not None]

        price_val = current_prices.get(symbol)
        if price_val is None and closes:
            price_val = closes[-1]
        if price_val is None:
            return None
        price = float(price_val)
        if price <= 0:
            return None

        action = self._next_action
        signal = make_signal(
            symbol,
            action,
            self.quantity,
            price,
            reason=f"Alternating {action} order",
        )
        self._next_action = "sell" if action == "buy" else "buy"
        return signal
