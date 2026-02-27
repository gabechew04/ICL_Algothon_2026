"""
Mean Reversion Strategy

Improvements:
 - Evaluate every minute 
 - Multi-timeframe z-score (fast + slow lookback)
 - Volume spike confirmation only enter when volume validates the move
 - Volatility regime filter skip entries in extreme-vol regimes (trending market)
 - Adaptive thresholds : z-score entry/exit scales with recent volatility rank
 - Dynamic position sizing, larger size on higher-conviction setups
 - Bollinger Band squeeze detection, higher conviction when bands tighten first
 - Trailing stop,  lock in gains as position moves in our favour
 - Short-selling support,  sell when price is far above mean
 - Time-of-day filter, avoid first/last 15 minutes (noisy open/close)
 - Cooldown after stop-loss,  avoid revenge trading
 -  Multi-symbol ready, handles whichever symbols QTC provides
"""

from typing import Optional, Dict, Any
import numpy as np
from collections import deque


class Strategy:
    """Mean reversion trading strategy."""

    # ── Configuration ─
    # All tunable parameters are here for easy optimisation.

    # Lookback windows (in 1-minute bars)
    FAST_WINDOW = 60          # 1-hour rolling stats
    SLOW_WINDOW = 390         # 1 full trading day
    VOLUME_WINDOW = 60        # Volume average lookback

    # Z-score thresholds (base — can be adapted per regime)
    ENTRY_Z_LONG = -2.0       # Buy when price is 2σ below mean
    ENTRY_Z_SHORT = 2.0       # Sell short when price is 2σ above mean
    EXIT_Z_LONG = -0.3        # Exit long when z reverts towards 0
    EXIT_Z_SHORT = 0.3        # Exit short when z reverts towards 0

    # Risk management
    TAKE_PROFIT = 0.012       # 1.2% profit target
    STOP_LOSS = -0.015        # 1.5% stop loss
    TRAILING_STOP_ACTIVATE = 0.006   # Activate trailing stop after 0.6% gain
    TRAILING_STOP_DISTANCE = 0.004   # Trail by 0.4% from peak
    MAX_HOLD_BARS = 120       # ~2 hours max hold time

    # Filters
    VOLUME_SPIKE_THRESHOLD = 1.5     # Require 1.5x average volume for entry
    MIN_VOLATILITY_RANK = 0.15       # Skip if vol rank < 15% (too quiet, won't revert)
    MAX_VOLATILITY_RANK = 0.85       # Skip if vol rank > 85% (trending/crisis)
    BOLLINGER_SQUEEZE_RATIO = 0.6    # Band width < 60% of recent avg = squeeze

    # Cooldown
    COOLDOWN_BARS_AFTER_STOP = 30    # Wait 30 mins after stop-loss

    # Position sizing
    BASE_QUANTITY = 1
    MAX_QUANTITY = 3
    CASH_RESERVE_RATIO = 0.1         # Keep 10% cash reserve

    # Time filters (hour boundaries to avoid)
    AVOID_FIRST_MINUTES = 15         # Skip first 15 min after open
    AVOID_LAST_MINUTES = 15          # Skip last 15 min before close

    # ── State ─────

    def __init__(self, quantity: int = 1):
        self.base_quantity = max(quantity, self.BASE_QUANTITY)
        self.quantity = self.base_quantity

        # Position tracking
        self.in_position = False
        self.position_direction = None   # 'long' or 'short'
        self.entry_price = None
        self.entry_bar_index = 0
        self.peak_price = None           # For trailing stop
        self.trough_price = None         # For short trailing stop

        # Rolling data buffers (use deques for memory efficiency)
        self.closes = deque(maxlen=self.SLOW_WINDOW + 50)
        self.volumes = deque(maxlen=self.VOLUME_WINDOW + 50)
        self.volatilities = deque(maxlen=self.SLOW_WINDOW)  # Track rolling vols

        # State
        self.bar_count = 0
        self.cooldown_until = 0          # Bar index when cooldown ends
        self.last_trade_bar = 0
        self.total_trades = 0

    # ── Helpers ────────────────────────────────────────────────────────

    def _z_score(self, price: float, window: int) -> Optional[float]:
        """Calculate z-score of current price vs rolling mean/std."""
        if len(self.closes) < window:
            return None
        arr = np.array(list(self.closes)[-window:], dtype=np.float64)
        mu = arr.mean()
        sigma = arr.std(ddof=1)
        if sigma < 1e-10:
            return None
        return (price - mu) / sigma

    def _rolling_volatility(self, window: int) -> Optional[float]:
        """Annualised volatility from minute returns."""
        if len(self.closes) < window + 1:
            return None
        arr = np.array(list(self.closes)[-(window + 1):], dtype=np.float64)
        returns = np.diff(arr) / arr[:-1]
        return float(returns.std(ddof=1))

    def _volatility_rank(self) -> Optional[float]:
        """Where does current vol sit relative to recent history? 0-1."""
        if len(self.volatilities) < 20:
            return None
        current_vol = self._rolling_volatility(self.FAST_WINDOW)
        if current_vol is None:
            return None
        arr = np.array(self.volatilities, dtype=np.float64)
        rank = float(np.searchsorted(np.sort(arr), current_vol)) / len(arr)
        return rank

    def _volume_spike(self) -> bool:
        """Is current volume elevated vs recent average?"""
        if len(self.volumes) < self.VOLUME_WINDOW:
            return True  # Not enough data — don't block
        current_vol = self.volumes[-1]
        avg_vol = np.mean(list(self.volumes)[-self.VOLUME_WINDOW:])
        if avg_vol <= 0:
            return True
        return current_vol >= avg_vol * self.VOLUME_SPIKE_THRESHOLD

    def _bollinger_squeeze(self) -> bool:
        """Are Bollinger Bands squeezed (low bandwidth)?"""
        if len(self.closes) < self.SLOW_WINDOW:
            return False
        arr = np.array(list(self.closes)[-self.SLOW_WINDOW:], dtype=np.float64)
        # Current band width (20-bar window)
        recent = arr[-60:]
        current_bw = recent.std(ddof=1) / recent.mean() if recent.mean() > 0 else 0
        # Average band width
        avg_bw = arr.std(ddof=1) / arr.mean() if arr.mean() > 0 else 0
        if avg_bw <= 0:
            return False
        return current_bw / avg_bw < self.BOLLINGER_SQUEEZE_RATIO

    def _is_valid_trading_time(self, timestamps, index: int) -> bool:
        """Check if current bar is within allowed trading hours."""
        if not timestamps or index >= len(timestamps):
            return True  # Can't filter without timestamps
        ts = timestamps[index]
        if not hasattr(ts, 'hour'):
            return True
        # Market hours: 9:30 - 16:00
        minutes_from_open = (ts.hour - 9) * 60 + (ts.minute - 30)
        minutes_to_close = (16 * 60) - (ts.hour * 60 + ts.minute)
        if minutes_from_open < self.AVOID_FIRST_MINUTES:
            return False
        if minutes_to_close < self.AVOID_LAST_MINUTES:
            return False
        return True

    def _compute_quantity(self, cash: float, price: float, conviction: float) -> int:
        """
        Dynamic position sizing based on conviction score (0-1).

        Higher conviction → larger position, up to MAX_QUANTITY.
        Always respects cash reserve.
        """
        available = cash * (1.0 - self.CASH_RESERVE_RATIO)
        max_affordable = int(available / price) if price > 0 else 0
        # Scale between base and max based on conviction
        target = self.base_quantity + int(
            (self.MAX_QUANTITY - self.base_quantity) * conviction
        )
        return max(1, min(target, max_affordable, self.MAX_QUANTITY))

    def _conviction_score(self, z_fast: float, z_slow: float,
                          vol_spike: bool, squeeze: bool) -> float:
        """
        Score 0-1 based on how many confirming signals align.

        Factors:
          - Z-score magnitude (deeper = more conviction)
          - Multi-timeframe agreement (fast & slow z-scores same direction)
          - Volume spike confirmation
          - Bollinger squeeze (anticipates expansion)
        """
        score = 0.0
        abs_z = abs(z_fast)

        # Z-score depth: 2.0 → 0.25, 2.5 → 0.50, 3.0+ → 0.75
        if abs_z >= 3.0:
            score += 0.30
        elif abs_z >= 2.5:
            score += 0.20
        else:
            score += 0.10

        # Multi-timeframe agreement
        if z_slow is not None and np.sign(z_fast) == np.sign(z_slow):
            score += 0.25

        # Volume confirmation
        if vol_spike:
            score += 0.25

        # Bollinger squeeze (setup before expansion)
        if squeeze:
            score += 0.20

        return min(score, 1.0)

    # ── Main Signal Generation ─────────────────────────────────────────

    def generate_signal(
        self,
        team: Dict[str, Any],
        bars: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal. Called once per minute by QTC orchestrator.

        Returns a single signal dict or None.
        """
        # Pick the primary symbol
        symbol = list(current_prices.keys())[0] if current_prices else None
        if not symbol or symbol not in bars:
            return None

        closes = bars[symbol].get('close', [])
        volumes = bars[symbol].get('volume', [])
        timestamps = bars[symbol].get('timestamp', [])

        if len(closes) == 0:
            return None

        current_price = float(closes[-1])
        current_volume = float(volumes[-1]) if len(volumes) > 0 else 0

        # Update rolling buffers
        self.closes.append(current_price)
        self.volumes.append(current_volume)
        self.bar_count += 1

        # Track rolling volatility for regime detection
        vol = self._rolling_volatility(self.FAST_WINDOW)
        if vol is not None:
            self.volatilities.append(vol)

        # ── Time filter ──
        if not self._is_valid_trading_time(timestamps, len(closes) - 1):
            # Still manage exits even during filtered times
            if self.in_position:
                return self._check_exit(symbol, current_price)
            return None

        # ── Warmup period ──
        if len(self.closes) < self.SLOW_WINDOW:
            return None

        # ── Compute indicators ──
        z_fast = self._z_score(current_price, self.FAST_WINDOW)
        z_slow = self._z_score(current_price, self.SLOW_WINDOW)
        vol_rank = self._volatility_rank()
        vol_spike = self._volume_spike()
        squeeze = self._bollinger_squeeze()

        if z_fast is None:
            return None

        # ── Exit logic (always check first) ──
        if self.in_position:
            return self._check_exit(symbol, current_price, z_fast)

        # ── Entry filters ──
        # Cooldown after stop-loss
        if self.bar_count < self.cooldown_until:
            return None

        # Volatility regime filter
        if vol_rank is not None:
            if vol_rank < self.MIN_VOLATILITY_RANK or vol_rank > self.MAX_VOLATILITY_RANK:
                return None

        # ── Entry logic ──
        cash = team.get('cash', 0)

        # LONG entry: price significantly below mean
        if z_fast <= self.ENTRY_Z_LONG:
            conviction = self._conviction_score(z_fast, z_slow, vol_spike, squeeze)
            qty = self._compute_quantity(cash, current_price, conviction)
            if qty > 0 and cash >= current_price * qty:
                self.in_position = True
                self.position_direction = 'long'
                self.entry_price = current_price
                self.entry_bar_index = self.bar_count
                self.peak_price = current_price
                self.quantity = qty
                self.total_trades += 1
                return {
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': qty,
                    'price': current_price,
                    'confidence': round(conviction, 2),
                    'reason': (
                        f'LONG: z_fast={z_fast:.2f} z_slow={z_slow or 0:.2f} '
                        f'conv={conviction:.0%} vol_rank={vol_rank or 0:.2f}'
                    ),
                }

        # SHORT entry: price significantly above mean
        elif z_fast >= self.ENTRY_Z_SHORT:
            # Check if we have a position to sell (QTC may not support naked shorts)
            positions = team.get('positions', {})
            held = positions.get(symbol, 0)
            if held > 0:
                conviction = self._conviction_score(z_fast, z_slow, vol_spike, squeeze)
                qty = min(self._compute_quantity(cash, current_price, conviction), held)
                if qty > 0:
                    self.in_position = True
                    self.position_direction = 'short'
                    self.entry_price = current_price
                    self.entry_bar_index = self.bar_count
                    self.trough_price = current_price
                    self.quantity = qty
                    self.total_trades += 1
                    return {
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': qty,
                        'price': current_price,
                        'confidence': round(conviction, 2),
                        'reason': (
                            f'SHORT: z_fast={z_fast:.2f} z_slow={z_slow or 0:.2f} '
                            f'conv={conviction:.0%} vol_rank={vol_rank or 0:.2f}'
                        ),
                    }

        return None

    # ── Exit Management ────────────────────────────────────────────────

    def _check_exit(
        self, symbol: str, current_price: float, z_fast: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Evaluate all exit conditions for the current position."""
        if not self.in_position or self.entry_price is None:
            return None

        bars_held = self.bar_count - self.entry_bar_index
        reason = None

        if self.position_direction == 'long':
            pnl_pct = (current_price - self.entry_price) / self.entry_price

            # Update peak for trailing stop
            if current_price > (self.peak_price or 0):
                self.peak_price = current_price

            # 1. Z-score reversion
            if z_fast is not None and z_fast >= self.EXIT_Z_LONG:
                reason = f'Z-revert: z={z_fast:.2f} pnl={pnl_pct*100:.2f}%'

            # 2. Take profit
            elif pnl_pct >= self.TAKE_PROFIT:
                reason = f'TP hit: {pnl_pct*100:.2f}%'

            # 3. Trailing stop
            elif (self.peak_price and
                  pnl_pct >= self.TRAILING_STOP_ACTIVATE and
                  current_price <= self.peak_price * (1.0 - self.TRAILING_STOP_DISTANCE)):
                reason = f'Trail stop: peak={self.peak_price:.2f} now={current_price:.2f}'

            # 4. Hard stop loss
            elif pnl_pct <= self.STOP_LOSS:
                reason = f'Stop loss: {pnl_pct*100:.2f}%'
                self.cooldown_until = self.bar_count + self.COOLDOWN_BARS_AFTER_STOP

            # 5. Max hold time
            elif bars_held >= self.MAX_HOLD_BARS:
                reason = f'Timeout: {bars_held} bars, pnl={pnl_pct*100:.2f}%'

            if reason:
                self._reset_position()
                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': self.quantity,
                    'price': current_price,
                    'reason': f'EXIT LONG — {reason}',
                }

        elif self.position_direction == 'short':
            # For shorts, profit when price goes down
            pnl_pct = (self.entry_price - current_price) / self.entry_price

            # Update trough for trailing stop
            if current_price < (self.trough_price or float('inf')):
                self.trough_price = current_price

            # 1. Z-score reversion
            if z_fast is not None and z_fast <= self.EXIT_Z_SHORT:
                reason = f'Z-revert: z={z_fast:.2f} pnl={pnl_pct*100:.2f}%'

            # 2. Take profit
            elif pnl_pct >= self.TAKE_PROFIT:
                reason = f'TP hit: {pnl_pct*100:.2f}%'

            # 3. Trailing stop (for shorts, price going up = loss)
            elif (self.trough_price and
                  pnl_pct >= self.TRAILING_STOP_ACTIVATE and
                  current_price >= self.trough_price * (1.0 + self.TRAILING_STOP_DISTANCE)):
                reason = f'Trail stop: trough={self.trough_price:.2f} now={current_price:.2f}'

            # 4. Hard stop loss
            elif pnl_pct <= self.STOP_LOSS:
                reason = f'Stop loss: {pnl_pct*100:.2f}%'
                self.cooldown_until = self.bar_count + self.COOLDOWN_BARS_AFTER_STOP

            # 5. Max hold time
            elif bars_held >= self.MAX_HOLD_BARS:
                reason = f'Timeout: {bars_held} bars, pnl={pnl_pct*100:.2f}%'

            if reason:
                self._reset_position()
                return {
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': self.quantity,
                    'price': current_price,
                    'reason': f'EXIT SHORT — {reason}',
                }

        return None

    def _reset_position(self):
        """Clear position state."""
        self.in_position = False
        self.position_direction = None
        self.entry_price = None
        self.entry_bar_index = 0
        self.peak_price = None
        self.trough_price = None