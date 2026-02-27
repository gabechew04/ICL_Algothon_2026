"""
Pairs Trading Strategy

Trades two correlated stocks (AAPL and SPY).
- Buy AAPL + Short SPY when AAPL outperforms
- Sell AAPL + Cover SPY when spread reverts

This strategy exploits mean reversion of the spread between two correlated assets.
"""

from typing import Optional, Dict, Any, List


class Strategy:
    """Pairs trading strategy with two correlated assets."""
    
    def __init__(self, quantity: int = 1):
        """
        Initialize strategy.
        
        Args:
            quantity: Number of shares per trade for each leg
        """
        self.quantity = quantity
        self.in_position = False
        self.entry_spread = None
    
    def generate_signal(self, team: Dict[str, Any], bars: Dict[str, Dict[str, Any]], 
                       current_prices: Dict[str, float]) -> Optional[List[Dict[str, Any]]]:
        """
        Generate trading signals based on pairs spread.
        
        Returns a list of signal dicts (one per leg).
        
        Args:
            team: Team dictionary with cash and positions
            bars: Dict of symbol -> {open, high, low, close, volume}
            current_prices: Dict of symbol -> current price
            
        Returns:
            List of signal dicts or None
        """
        # Need both symbols
        if 'AAPL' not in bars or 'SPY' not in bars:
            return None
        if 'AAPL' not in current_prices or 'SPY' not in current_prices:
            return None
        
        aapl_closes = bars['AAPL']['close']
        spy_closes = bars['SPY']['close']
        
        # Need at least 20 bars for moving average
        if len(aapl_closes) < 20 or len(spy_closes) < 20:
            return None
        
        aapl_price = current_prices['AAPL']
        spy_price = current_prices['SPY']
        
        # Calculate 20-bar moving averages
        aapl_ma = sum(aapl_closes[-20:]) / 20
        spy_ma = sum(spy_closes[-20:]) / 20
        
        # Normalize prices (remove absolute price differences)
        aapl_ratio = aapl_price / aapl_ma
        spy_ratio = spy_price / spy_ma
        
        # Spread: how much AAPL is outperforming SPY
        spread = aapl_ratio - spy_ratio
        
        # Mean reversion signals
        
        # BUY signal: AAPL is 2% above MA while SPY is 2% below MA (divergence)
        if spread > 0.02 and not self.in_position:
            # Check we have enough cash for both legs
            total_cost = (aapl_price + spy_price) * self.quantity * 1.001  # +commission
            if team.get('cash', 0) >= total_cost:
                self.in_position = True
                self.entry_spread = spread
                reason = (
                    f'Pairs divergence: AAPL at {aapl_ratio:.4f} (MA), '
                    f'SPY at {spy_ratio:.4f} (MA), spread={spread:.4f}'
                )
                return [
                    {
                        'symbol': 'AAPL',
                        'action': 'buy',
                        'quantity': self.quantity,
                        'price': aapl_price,
                        'reason': reason
                    },
                    {
                        'symbol': 'SPY',
                        'action': 'sell',
                        'quantity': self.quantity,
                        'price': spy_price,
                        'reason': reason
                    }
                ]
        
        # SELL signal: spread reverts back to 0 (mean reversion)
        elif self.in_position and self.entry_spread and spread < 0.005:
            self.in_position = False
            reason = f'Spread mean reversion: {spread:.4f} (was {self.entry_spread:.4f})'
            return [
                {
                    'symbol': 'AAPL',
                    'action': 'sell',
                    'quantity': self.quantity,
                    'price': aapl_price,
                    'reason': reason
                },
                {
                    'symbol': 'SPY',
                    'action': 'buy',
                    'quantity': self.quantity,
                    'price': spy_price,
                    'reason': reason
                }
            ]
        
        return None
