"""
Universal Backtester

Run any strategy on real QTC API data or synthetic data.

Usage:
    python backtest.py <strategy_file> [--capital 1000] [--quantity 1] [--start 2025-10-01] [--end 2025-10-10] [--symbols AAPL,SPY]

Examples:
    python backtest.py strategies/simple_strategy.py
    python backtest.py strategies/example_strat.py --capital 5000 --quantity 5
    python backtest.py strategies/pairs_trading_strategy.py --symbols AAPL,SPY
    python backtest.py my_strategy.py --start 2025-10-01 --end 2025-10-15
"""

# NOTE: Default tries QTC first (if QTC_API_KEY is set); on failure it falls back to local CSVs in data/.

import sys
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta, timezone
from alpaca_data_fetcher import QTCDataFetcher
from backtesting.engine import BacktestEngine
from api_client import create_team_dict
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from typing import Optional

DATA_DIR = Path("data")


def _parse_start_dt(start_date: str) -> datetime:
    if 'T' in start_date:
        cleaned = start_date.replace('Z', '')
        return datetime.fromisoformat(cleaned)
    return datetime.strptime(start_date, '%Y-%m-%d').replace(hour=9, minute=30)


def _parse_dt(date_str: str, default_hour: int, default_minute: int) -> datetime:
    if 'T' in date_str:
        cleaned = date_str.replace('Z', '')
        return datetime.fromisoformat(cleaned)
    return datetime.strptime(date_str, '%Y-%m-%d').replace(hour=default_hour, minute=default_minute)


def _to_iso_z(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def _merge_symbol_frames(frames: list) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if 'timestamp' in merged.columns:
        merged = merged.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    return merged.reset_index(drop=True)


def _fetch_alpaca_live(symbols: list, start_date: str, end_date: str, chunk_days: int = 10) -> Optional[dict]:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except Exception:
        print("    ✗ alpaca-py not available")
        return None

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("    ✗ Alpaca keys not found (ALPACA_API_KEY/ALPACA_API_SECRET)")
        return None

    feed = os.getenv("ALPACA_DATA_FEED", "iex")
    client = StockHistoricalDataClient(api_key, api_secret)

    start_dt = _parse_dt(start_date, 9, 30).replace(tzinfo=timezone.utc)
    end_dt = _parse_dt(end_date, 16, 0).replace(tzinfo=timezone.utc)
    if end_dt <= start_dt:
        print("    ✗ Alpaca: end_date must be after start_date")
        return None

    symbol_frames = {symbol: [] for symbol in symbols}
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=current_start,
                end=current_end,
                feed=feed
            )
            bars = client.get_stock_bars(req)
            df = bars.df
        except Exception as e:
            print(f"    ✗ Alpaca fetch failed: {e}")
            return None

        if df is None or df.empty:
            print("    ✗ Alpaca returned no data")
            break

        if isinstance(df.index, pd.MultiIndex):
            for symbol in symbols:
                if symbol in df.index.get_level_values(0):
                    sym_df = df.xs(symbol)
                    if not sym_df.empty:
                        sym_df = sym_df.reset_index()
                        if sym_df['timestamp'].dt.tz is not None:
                            sym_df['timestamp'] = sym_df['timestamp'].dt.tz_convert(None)
                        symbol_frames[symbol].append(
                            sym_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        )
        else:
            sym_df = df.reset_index()
            if 'timestamp' not in sym_df.columns:
                sym_df.rename(columns={'index': 'timestamp'}, inplace=True)
            if sym_df['timestamp'].dt.tz is not None:
                sym_df['timestamp'] = sym_df['timestamp'].dt.tz_convert(None)
            symbol_frames[symbols[0]].append(
                sym_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            )

        current_start = current_end

    result = {}
    for symbol in symbols:
        merged = _merge_symbol_frames(symbol_frames[symbol])
        if not merged.empty:
            result[symbol] = {
                'timestamp': merged['timestamp'].tolist(),
                'open': merged['open'].values,
                'high': merged['high'].values,
                'low': merged['low'].values,
                'close': merged['close'].values,
                'volume': merged['volume'].values
            }

    return result if result else None


def _find_local_file(symbol: str, data_dir: Path) -> Optional[Path]:
    exact = [data_dir / f"{symbol}.csv", data_dir / f"{symbol}_data.csv"]
    for path in exact:
        if path.exists():
            return path

    if not data_dir.exists():
        return None

    candidates = [path for path in data_dir.glob("*.csv") if path.stem.lower().startswith(symbol.lower())]
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_local_data(symbols: list, start_date: str, end_date: str, data_dir: Path) -> Optional[dict]:
    if not data_dir.exists():
        return None

    start_dt = _parse_dt(start_date, 9, 30)
    end_dt = _parse_dt(end_date, 16, 0)

    result = {}
    for symbol in symbols:
        path = _find_local_file(symbol, data_dir)
        if not path:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"    ✗ Failed to read {path}: {e}")
            continue

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        elif 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        df = df.dropna(subset=['timestamp'])
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert(None)

        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
        if df.empty:
            continue

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0
                else:
                    df[col] = df['close']

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
        result[symbol] = {
            'timestamp': df['timestamp'].tolist(),
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values
        }

    return result if result else None




def _next_trading_day_start(dt: datetime) -> datetime:
    current = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    while current.weekday() >= 5:
        current += timedelta(days=1)
        current = current.replace(hour=9, minute=30, second=0, microsecond=0)
    return current


def _trading_minutes_between(start_dt: datetime, end_dt: datetime, minutes_per_day: int) -> list:
    if end_dt <= start_dt:
        return []
    timestamps = []
    current_day = _next_trading_day_start(start_dt)
    while current_day.date() <= end_dt.date():
        for minute in range(minutes_per_day):
            ts = current_day + timedelta(minutes=minute)
            if ts < start_dt or ts > end_dt:
                continue
            timestamps.append(ts)
        current_day = _next_trading_day_start(current_day + timedelta(days=1))
    return timestamps


def generate_sample_bars(
    symbol: str = "AAPL",
    days: int = 30,
    minutes_per_day: int = 390,
    start_date: str = "2025-01-01",
    end_date: Optional[str] = None
):
    """
    Generate synthetic market bars for testing.
    
    Args:
        symbol: Stock symbol
        days: Number of days to generate (used if end_date is not provided)
        minutes_per_day: Minutes per trading day (typically 390 for 9:30-16:00)
        start_date: Start date for data
        end_date: End date for data (optional)
        
    Returns:
        Dict with symbol -> {timestamp, open, high, low, close, volume}
    """
    start_dt = _parse_dt(start_date, 9, 30)
    if end_date:
        end_dt = _parse_dt(end_date, 16, 0)
        timestamps = _trading_minutes_between(start_dt, end_dt, minutes_per_day)
    else:
        timestamps = []
        current_day = _next_trading_day_start(start_dt)
        trading_days = 0
        while trading_days < days:
            for minute in range(minutes_per_day):
                timestamps.append(current_day + timedelta(minutes=minute))
            trading_days += 1
            current_day = current_day + timedelta(days=1)
            current_day = _next_trading_day_start(current_day)

    num_bars = len(timestamps)
    
    # Generate realistic price movements
    returns = np.random.normal(0.0001, 0.005, num_bars)
    start_price = 100.0
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    opens = prices + np.random.uniform(-0.5, 0.5, num_bars)
    highs = np.maximum(prices, opens) + np.random.uniform(0, 2, num_bars)
    lows = np.minimum(prices, opens) - np.random.uniform(0, 2, num_bars)
    closes = prices
    volumes = np.random.randint(100000, 5000000, num_bars)
    
    return {
        symbol: {
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }
    }


def generate_sample_bars_multi(
    symbols: list,
    days: int = 30,
    minutes_per_day: int = 390,
    start_date: str = "2025-01-01",
    end_date: Optional[str] = None
):
    """
    Generate synthetic market bars for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        days: Number of days to generate (used if end_date is not provided)
        minutes_per_day: Minutes per trading day (typically 390 for 9:30-16:00)
        end_date: End date for data (optional)
        
    Returns:
        Dict with symbol -> {timestamp, open, high, low, close, volume}
    """
    data = {}
    for symbol in symbols:
        data.update(generate_sample_bars(
            symbol,
            days=days,
            minutes_per_day=minutes_per_day,
            start_date=start_date,
            end_date=end_date
        ))
    return data


def load_strategy_from_file(file_path: str):
    """
    Dynamically load a strategy class from a Python file.
    
    The file should have a Strategy class with generate_signal method.
    
    Args:
        file_path: Path to the strategy Python file
        
    Returns:
        Strategy class
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location("strategy_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'Strategy'):
        raise ValueError(f"Strategy file must define a 'Strategy' class: {file_path}")
    
    return module.Strategy


def fetch_real_data(api_key: Optional[str], start_date: str, end_date: str, symbols: list = None) -> Optional[dict]:
    """
    Fetch real data from QTC API.
    
    Args:
        api_key: QTC API key
        start_date: Start date (YYYY-MM-DD or ISO format)
        end_date: End date (YYYY-MM-DD or ISO format)
        symbols: List of symbols to fetch (default ['AAPL'])
        
    Returns:
        Dict with symbol -> bars, or None if failed
    """
    if symbols is None:
        symbols = ['AAPL']
    
    try:
        max_days_per_request = 20
        chunk_days = 10

        start_dt = _parse_dt(start_date, 9, 30)
        end_dt = _parse_dt(end_date, 16, 0)

        if end_dt <= start_dt:
            raise ValueError(f"end_date must be after start_date (got {start_date} -> {end_date})")

        if not api_key:
            print("    ✗ QTC_API_KEY not found; skipping QTC fetch")
            return None

        fetcher = QTCDataFetcher(api_key=api_key)
        if end_dt - start_dt <= timedelta(days=max_days_per_request):
            bars_dict = fetcher.getBars(
                symbols,
                _to_iso_z(start_dt),
                _to_iso_z(end_dt)
            )
        else:
            print(f"    ✓ Range exceeds {max_days_per_request} days; fetching in {chunk_days}-day chunks")
            symbol_frames = {symbol: [] for symbol in symbols}
            current_start = start_dt

            while current_start < end_dt:
                current_end = min(current_start + timedelta(days=chunk_days), end_dt)
                bars_dict = fetcher.getBars(
                    symbols,
                    _to_iso_z(current_start),
                    _to_iso_z(current_end)
                )
                if all((df is None or df.empty) for df in bars_dict.values()):
                    print("    ✗ No data returned for chunk; stopping further requests")
                    break
                for symbol in symbols:
                    df = bars_dict.get(symbol)
                    if df is not None and not df.empty:
                        symbol_frames[symbol].append(df)
                current_start = current_end

            bars_dict = {symbol: _merge_symbol_frames(frames) for symbol, frames in symbol_frames.items()}

        result = {}
        for symbol in symbols:
            if symbol in bars_dict and not bars_dict[symbol].empty:
                df = bars_dict[symbol]
                volume = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
                result[symbol] = {
                    'timestamp': df['timestamp'].tolist() if 'timestamp' in df.columns else [],
                    'open': df['open'].values,
                    'high': df['high'].values,
                    'low': df['low'].values,
                    'close': df['close'].values,
                    'volume': volume
                }

        return result if result else None
    except Exception as e:
        print(f"    ✗ Error fetching data: {e}")
        return None


def run_backtest(strategy_file: str, capital: float = 1000.0, quantity: float = 1.0, 
                 start_date: str = '2025-10-01', end_date: str = '2025-10-10',
                 use_real_data: bool = True, symbols: list = None,
                 alpaca_live: bool = False,
                 validate_signals: bool = True):
    """
    Run a backtest with the given strategy.
    
    Args:
        strategy_file: Path to strategy Python file
        capital: Initial capital in dollars
        quantity: Default quantity per trade
        start_date: Start date for backtest
        end_date: End date for backtest
        use_real_data: Try to use real data, fall back to synthetic if unavailable
        symbols: List of symbols to fetch (default ['AAPL'])
        validate_signals: Whether to validate signals before processing
    """
    
    # Get strategy file name
    strategy_name = Path(strategy_file).stem
    
    print("\n" + "="*70)
    print(f"BACKTEST: {strategy_name}".center(70))
    print("="*70)
    
    # Load strategy
    print(f"\n[1] Loading strategy from {strategy_file}...")
    try:
        StrategyClass = load_strategy_from_file(strategy_file)
        print(f"    ✓ Loaded Strategy class")
    except Exception as e:
        print(f"    ✗ Failed to load strategy: {e}")
        return None
    
    # Try to fetch real data
    bars = None
    data_type = "Real Data"
    if symbols is None or not symbols:
        symbols = ['AAPL']
    
    if use_real_data:
        print(f"\n[2] Fetching real market data...")
        load_dotenv()
        if alpaca_live:
            print("    ✓ Alpaca live mode enabled")
            print(f"    ✓ Date range: {start_date} to {end_date}")
            bars = _fetch_alpaca_live(symbols, start_date, end_date)
            if bars:
                symbols_loaded = list(bars.keys())
                sample_symbol = symbols_loaded[0]
                num_bars = len(bars[sample_symbol]['close'])
                print(f"    ✓ Retrieved {num_bars} bars for {', '.join(symbols_loaded)}")
            else:
                print("    ✗ Alpaca live failed; trying QTC...")

        api_key = os.getenv('QTC_API_KEY')
        if not bars:
            if api_key:
                print(f"    ✓ QTC API key found")
            else:
                print(f"    ✗ QTC_API_KEY not found in .env")
            print(f"    ✓ Date range: {start_date} to {end_date}")
            bars = fetch_real_data(api_key, start_date, end_date, symbols=symbols)

        if bars:
            symbols_loaded = list(bars.keys())
            sample_symbol = symbols_loaded[0]
            num_bars = len(bars[sample_symbol]['close'])
            print(f"    ✓ Retrieved {num_bars} bars for {', '.join(symbols_loaded)}")
        else:
            print(f"    ✗ No QTC data available; trying Alpaca...")
            bars = _fetch_alpaca_live(symbols, start_date, end_date)
            if bars:
                symbols_loaded = list(bars.keys())
                sample_symbol = symbols_loaded[0]
                num_bars = len(bars[sample_symbol]['close'])
                print(f"    ✓ Retrieved {num_bars} bars for {', '.join(symbols_loaded)}")
                data_type = "Alpaca Data"
            else:
                print(f"    ✗ No Alpaca data available; trying local data...")
                bars = _load_local_data(symbols, start_date, end_date, DATA_DIR)
                if bars:
                    symbols_loaded = list(bars.keys())
                    sample_symbol = symbols_loaded[0]
                    num_bars = len(bars[sample_symbol]['close'])
                    print(f"    ✓ Loaded {num_bars} bars for {', '.join(symbols_loaded)}")
                    data_type = "Local Data"
    
    # Fall back to synthetic data
    if not bars:
        print(f"\n[2] Generating synthetic data...")
        bars = generate_sample_bars_multi(
            symbols,
            days=10,
            minutes_per_day=390,
            start_date=start_date,
            end_date=end_date
        )
        sample_symbol = list(bars.keys())[0]
        num_bars = len(bars[sample_symbol]['close'])
        print(f"    ✓ Generated {num_bars} synthetic bars for {', '.join(symbols)}")
        data_type = "Synthetic Data"
    
    symbols_display = ", ".join(bars.keys())
    
    # Create team
    print(f"\n[3] Setting up backtest parameters...")
    team = create_team_dict("_team-h", capital, None)
    print(f"    ✓ Team: {team['id']}")
    print(f"    ✓ Initial capital: ${capital:,.2f}")
    print(f"    ✓ Data type: {data_type}")
    print(f"    ✓ Symbols: {symbols_display}")
    
    # Initialize strategy
    print(f"\n[4] Initializing strategy...")
    try:
        strategy = StrategyClass(quantity=quantity)
        print(f"    ✓ {strategy_name} ready")
        print(f"    ✓ Trade quantity: {quantity}")
    except Exception as e:
        print(f"    ✗ Failed to initialize strategy: {e}")
        return None
    
    # Run backtest
    print(f"\n[5] Running backtest...")
    try:
        engine = BacktestEngine(initial_capital=capital, commission=0.001)
        results = engine.run_backtest(strategy, bars, team=team, validate_signals=validate_signals)
        print(f"    ✓ Backtest complete")
    except Exception as e:
        print(f"    ✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS".center(70))
    print("="*70)
    print(f"Initial Capital:     ${capital:>10,.2f}")
    print(f"Final Capital:       ${results['final_capital']:>10,.2f}")
    print(f"Total Return:        {results['total_return']*100:>10.2f}%")
    print(f"Sharpe Ratio:        {results['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:        {results['max_drawdown']*100:>10.2f}%")
    print(f"Total Trades:        {results['total_trades']:>10}")
    print(f"Winning Trades:      {results['winning_trades']:>10}")
    print(f"Losing Trades:       {results['losing_trades']:>10}")
    print(f"Win Rate:            {results['win_rate']*100:>10.2f}%")
    print("="*70)
    
    # Show signals
    if results['signals']:
        print(f"\nGenerated {len(results['signals'])} trading signals:\n")
        for i, sig in enumerate(results['signals'][:20], 1):
            action = sig['action'].upper()
            reason = sig.get('reason', '')
            ts = sig.get('timestamp')
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts is not None else 'N/A'
            print(
                f"  {i:2}. {action:4} x{sig['quantity']:.0f} @ ${sig['price']:.2f} "
                f"[{ts_str}] - {reason}"
            )
        
        if len(results['signals']) > 20:
            print(f"\n... and {len(results['signals']) - 20} more signals")
    else:
        print("\nNo trading signals generated")
    
    print("\n" + "="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Universal backtester for trading strategies',
        epilog='Example: python backtest.py strategies/simple_strategy.py --capital 5000 --quantity 2'
    )
    
    parser.add_argument(
        'strategy',
        help='Path to strategy Python file (must have a Strategy class)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=1000.0,
        help='Initial capital in dollars (default: 1000)'
    )
    parser.add_argument(
        '--quantity',
        type=float,
        default=1.0,
        help='Default quantity per trade (default: 1)'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2025-10-01',
        help='Start date YYYY-MM-DD (default: 2025-10-01)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2025-10-10',
        help='End date YYYY-MM-DD (default: 2025-10-10)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default='AAPL',
        help='Comma-separated symbols to fetch (default: AAPL)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip signal validation (useful for synthetic or multi-leg signals)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use only synthetic data (skip API)'
    )
    parser.add_argument(
        '--alpaca-live',
        action='store_true',
        help='Fetch live data from Alpaca first (requires ALPACA_API_KEY/ALPACA_API_SECRET)'
    )
    
    args = parser.parse_args()
    
    # Run backtest
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    results = run_backtest(
        strategy_file=args.strategy,
        capital=args.capital,
        quantity=args.quantity,
        start_date=args.start,
        end_date=args.end,
        use_real_data=not args.synthetic,
        symbols=symbols,
        alpaca_live=args.alpaca_live,
        validate_signals=not args.no_validate
    )
    
    if results:
        print("\n✓ Backtest completed successfully!")
        return 0
    else:
        print("\n✗ Backtest failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
