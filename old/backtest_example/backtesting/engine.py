"""
Backtesting Engine

Core engine for running strategy backtests on historical data.
Simulates the orchestrator interface for local testing.
"""

from typing import Dict, Any, Optional
import numpy as np
from signal_validator import SignalValidator


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    
    Simulates the orchestrator by calling strategy.generate_signal()
    once per minute with team, bars, and current_prices.
    
    Attributes:
        initial_capital (float): Starting capital for backtest
        commission (float): Commission rate per trade (e.g., 0.001 for 0.1%)
        slippage (float): Slippage rate per trade
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting capital in dollars
            commission: Commission rate per trade
            slippage: Estimated slippage per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = None
    
    def run_backtest(
        self,
        strategy,
        bars: Dict[str, Dict[str, Any]],
        team: Optional[Dict[str, Any]] = None,
        validate_signals: bool = True
    ) -> Dict[str, Any]:
        """
        Run a backtest using the orchestrator interface.
        
        Args:
            strategy: Strategy instance with generate_signal method
            bars: Dictionary of symbol -> OHLCV data with structure:
                  {symbol: {"timestamp": [...], "open": [...], "high": [...], 
                            "low": [...], "close": [...], "volume": [...]}}
            team: Team state dict (created if not provided). Should have:
                  {id, name, cash, positions, params, api}
            validate_signals: Whether to validate signals before processing
            
        Returns:
            Dictionary containing backtest results
        """
        if team is None:
            from api_client import create_team_dict
            team = create_team_dict("backtest_team")
        
        # Initialize team state
        team["cash"] = self.initial_capital
        team["positions"] = {}
        
        # Track signals, trades, and equity
        signals_history = []
        trades = []
        bar_prices_history = []
        bar_timestamps = []
        positions = {}
        entry_prices = {}
        trade_returns = []
        cash = self.initial_capital
        equity_history = [self.initial_capital]
        
        # Get the number of bars (assuming all symbols have same length)
        if not bars:
            return self._empty_results()
        
        num_bars = len(next(iter(bars.values()))['close'])
        
        # Iterate through each bar (simulating one minute at a time)
        for bar_idx in range(num_bars):
            # Build current_prices from the current bar data
            current_prices = {}
            current_bars = {}
            
            for symbol, ohlcv in bars.items():
                if bar_idx < len(ohlcv['close']):
                    current_prices[symbol] = ohlcv['close'][bar_idx]
                    # Build historical data up to current bar
                    current_bars[symbol] = {
                        'open': ohlcv['open'][:bar_idx + 1],
                        'high': ohlcv['high'][:bar_idx + 1],
                        'low': ohlcv['low'][:bar_idx + 1],
                        'close': ohlcv['close'][:bar_idx + 1],
                        'volume': ohlcv['volume'][:bar_idx + 1],
                    }
                    if 'timestamp' in ohlcv:
                        current_bars[symbol]['timestamp'] = ohlcv['timestamp'][:bar_idx + 1]
            
            # Capture bar timestamp if available
            bar_timestamp = None
            for symbol, ohlcv in bars.items():
                if 'timestamp' in ohlcv and bar_idx < len(ohlcv['timestamp']):
                    bar_timestamp = ohlcv['timestamp'][bar_idx]
                    break
            bar_timestamps.append(bar_timestamp)
            bar_prices_history.append(current_prices.copy())
            
            # Call strategy's generate_signal method (once per minute)
            try:
                signal = strategy.generate_signal(team, current_bars, current_prices)
            except Exception as e:
                print(f"Error in strategy at bar {bar_idx}: {e}")
                signal = None
            
            # Normalize to a list for multi-signal strategies
            if signal is None:
                signals = []
            elif isinstance(signal, list):
                signals = signal
            else:
                signals = [signal]
            
            # Validate signals if enabled
            if validate_signals and signals:
                valid_signals = []
                for sig in signals:
                    is_valid, error_msg = SignalValidator.validate(sig)
                    if not is_valid:
                        print(f"Invalid signal at bar {bar_idx}: {error_msg}")
                        continue
                    valid_signals.append(sig)
                signals = valid_signals
            
            for sig in signals:
                sig_with_meta = dict(sig)
                sig_with_meta["timestamp"] = bar_timestamp
                sig_with_meta["bar_idx"] = bar_idx
                signals_history.append(sig_with_meta)
                trades.append(sig_with_meta)
                
                action = sig_with_meta["action"]
                symbol = sig_with_meta["symbol"]
                quantity = float(sig_with_meta["quantity"])
                price = float(sig_with_meta["price"])
                
                if action == 'buy':
                    prev_qty = positions.get(symbol, 0.0)
                    if prev_qty >= 0:
                        new_qty = prev_qty + quantity
                        prev_entry = entry_prices.get(symbol, price)
                        entry_prices[symbol] = (prev_entry * prev_qty + price * quantity) / new_qty
                        positions[symbol] = new_qty
                        cash -= quantity * price + quantity * price * (self.commission + self.slippage)
                    else:
                        cover_qty = min(quantity, abs(prev_qty))
                        entry_price = entry_prices.get(symbol, price)
                        trade_returns.append((entry_price - price) / entry_price)
                        cash -= cover_qty * price + cover_qty * price * (self.commission + self.slippage)
                        remaining_qty = prev_qty + cover_qty
                        if remaining_qty == 0:
                            positions.pop(symbol, None)
                            entry_prices.pop(symbol, None)
                        else:
                            positions[symbol] = remaining_qty
                        
                        open_qty = quantity - cover_qty
                        if open_qty > 0:
                            positions[symbol] = open_qty
                            entry_prices[symbol] = price
                            cash -= open_qty * price + open_qty * price * (self.commission + self.slippage)
                elif action == 'sell':
                    prev_qty = positions.get(symbol, 0.0)
                    if prev_qty <= 0:
                        new_qty = prev_qty - quantity
                        prev_entry = entry_prices.get(symbol, price)
                        entry_prices[symbol] = (prev_entry * abs(prev_qty) + price * quantity) / abs(new_qty)
                        positions[symbol] = new_qty
                        cash += quantity * price - quantity * price * (self.commission + self.slippage)
                    else:
                        close_qty = min(quantity, prev_qty)
                        entry_price = entry_prices.get(symbol, price)
                        trade_returns.append((price - entry_price) / entry_price)
                        cash += close_qty * price - close_qty * price * (self.commission + self.slippage)
                        remaining_qty = prev_qty - close_qty
                        if remaining_qty == 0:
                            positions.pop(symbol, None)
                            entry_prices.pop(symbol, None)
                        else:
                            positions[symbol] = remaining_qty
                        
                        open_qty = quantity - close_qty
                        if open_qty > 0:
                            positions[symbol] = -open_qty
                            entry_prices[symbol] = price
                            cash += open_qty * price - open_qty * price * (self.commission + self.slippage)
            
            team["cash"] = cash
            team["positions"] = positions.copy()
            
            current_equity = cash
            for symbol, qty in positions.items():
                if symbol in current_prices:
                    current_equity += qty * current_prices[symbol]
            equity_history.append(max(current_equity, 0))
        
        # Close any remaining positions at final price (apply closing costs)
        if bar_prices_history:
            final_prices = bar_prices_history[-1]
            for symbol, qty in list(positions.items()):
                if symbol not in final_prices:
                    continue
                final_price = final_prices[symbol]
                entry_price = entry_prices.get(symbol, final_price)
                if qty > 0:
                    cash += qty * final_price - qty * final_price * (self.commission + self.slippage)
                    trade_returns.append((final_price - entry_price) / entry_price)
                elif qty < 0:
                    cash -= abs(qty) * final_price + abs(qty) * final_price * (self.commission + self.slippage)
                    trade_returns.append((entry_price - final_price) / entry_price)
            positions = {}
            entry_prices = {}
            equity_history[-1] = max(cash, 0)
        
        # Calculate returns from equity history
        periods_per_year = self._estimate_periods_per_year(bar_timestamps)
        results = self._calculate_returns(
            equity_history,
            trade_returns,
            signals_history,
            periods_per_year
        )
        results["trades"] = trades
        
        # Store results
        self.results = results
        
        return {
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'total_trades': results['total_trades'],
            'winning_trades': results['winning_trades'],
            'losing_trades': results['losing_trades'],
            'win_rate': results['win_rate'],
            'final_capital': results['final_capital'],
            'equity_curve': results['equity_curve'],
            'signals': results['signals'],
            'trades': results['trades']
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results when no data is available."""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'final_capital': self.initial_capital,
            'equity_curve': np.array([self.initial_capital]),
            'signals': [],
            'trades': []
        }
    
    def _calculate_returns(
        self,
        equity_history: list,
        trade_returns: list,
        signals_history: list,
        periods_per_year: int
    ) -> Dict[str, Any]:
        """
        Calculate returns and performance metrics from equity history.
        
        Args:
            equity_history: List of equity values per bar
            trade_returns: List of per-trade returns
            signals_history: List of signal dictionaries
            periods_per_year: Number of return periods per year
            
        Returns:
            Dictionary with performance metrics
        """
        if len(equity_history) < 2:
            return self._empty_results()
        
        # Calculate metrics
        final_capital = max(equity_history[-1], 0)
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (minute-by-minute returns)
        equity_array = np.array(equity_history, dtype=float)
        if len(equity_array) > 1 and np.all(equity_array[:-1] > 0):
            returns_array = equity_array[1:] / equity_array[:-1] - 1.0
        else:
            returns_array = np.array([])
        sharpe_ratio = self._calculate_sharpe_ratio(returns_array, periods_per_year)
        
        # Max drawdown (positive magnitude)
        max_drawdown = self._calculate_max_drawdown(equity_array)
        
        # Trade statistics
        total_trades = len(trade_returns)
        winning_trades = sum(1 for r in trade_returns if r > 0)
        losing_trades = sum(1 for r in trade_returns if r < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'final_capital': final_capital,
            'equity_curve': equity_array,
            'signals': signals_history
        }
    
    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        # Filter out invalid returns (NaN, inf, etc.)
        valid_returns = returns[np.isfinite(returns)]
        
        if len(valid_returns) == 0 or np.std(valid_returns) == 0:
            return 0.0
        
        per_period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = valid_returns - per_period_rf
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    
    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity: Array of portfolio equity values
            
        Returns:
            Maximum drawdown as a percentage
        """
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return float(-np.min(drawdown))
    
    def _estimate_periods_per_year(self, timestamps: list) -> int:
        if not timestamps:
            return 252 * 390
        dates = [ts.date() for ts in timestamps if ts is not None]
        if not dates:
            return 252 * 390
        unique_days = len(set(dates))
        if unique_days == 0:
            return 252 * 390
        bars_per_day = max(1, int(round(len(timestamps) / unique_days)))
        return 252 * bars_per_day

    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent backtest results.
        
        Returns:
            Dictionary of results or None if no backtest has been run
        """
        return self.results
