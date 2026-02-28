"""
Simple Momentum Strategy

Buy when price goes up 2% in last 5 bars, sell when it goes down 1%.
"""

from typing import Optional, Dict, Any


class Strategy:
    """Simple momentum-based strategy."""
    
    def __init__(self, quantity: int = 10):
        """
        Initialize strategy.
        
        Args:
            quantity: Number of shares per trade
        """
        self.quantity = quantity
        self.in_position = False
        self.entry_price = None
    
    def generate_signal(self, team: Dict[str, Any], bars: Dict[str, Dict[str, Any]], current_prices: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on momentum.
        
        Returns a single signal dict or None (called once per bar).
        
        Args:
            team: Team dictionary with cash and positions
            bars: Dict of symbol -> {open, high, low, close, volume}
            current_prices: Dict of symbol -> current price
            
        Returns:
            Single signal dict or None
        """
        # Focus on the first symbol available
        symbol = list(current_prices.keys())[0] if current_prices else None
        
        if not symbol or symbol not in bars:
            return None
        
        closes = bars[symbol]['close']
        
        # Need at least 6 bars of history to look back 5
        if len(closes) < 6:
            return None
        
        current_price = closes[-1]
        price_5_bars_ago = closes[-6]
        price_1_bar_ago = closes[-2]
        
        # Calculate momentum
        momentum_5 = (current_price - price_5_bars_ago) / price_5_bars_ago
        momentum_1 = (current_price - price_1_bar_ago) / price_1_bar_ago
        
        # Buy signal: up 2% in last 5 bars
        if momentum_5 > 0.02 and not self.in_position:
            if team.get('cash', 0) >= current_price * self.quantity:
                self.in_position = True
                self.entry_price = current_price
                return {
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': self.quantity,
                    'price': current_price,
                    'reason': f'Momentum up {momentum_5*100:.2f}% over 5 bars'
                }
        
        # Sell signal: down 1% from previous bar
        elif momentum_1 < -0.01 and self.in_position:
            self.in_position = False
            return {
                'symbol': symbol,
                'action': 'sell',
                'quantity': self.quantity,
                'price': current_price,
                'reason': f'Momentum down {momentum_1*100:.2f}% - exit position'
            }
        
        return None
