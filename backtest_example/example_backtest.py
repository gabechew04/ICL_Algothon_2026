"""
Example: Running a Backtest

This example demonstrates how to use the backtesting engine
with a moving average strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.moving_average_strategy import MovingAverageStrategy
from backtesting.engine import BacktestEngine
from backtesting.performance import PerformanceMetrics
from data.data import fetch_data


def generate_sample_data(ticker, days: int = 365) -> pd.DataFrame:
    """
    Generate sample OHLCV data for demonstration.
    
    Args:
        days: Number of days of data to generate
        
    Returns:
        DataFrame with OHLCV data
    """

    start =  str(((datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d") + "T09:30:00"))
    end = str((datetime.now()).strftime("%Y-%m-%d") + "T16:00:00")

    # if difference between start and end is more than 3 days, then fetch data in chunks of 3 days
    delta = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S") - datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
    if delta.days > 10:
        all_data = []
        current_start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
        while current_start < datetime.strptime(end, "%Y-%m-%dT%H:%M:%S"):
            current_end = min(current_start + timedelta(days=3), datetime.strptime(end, "%Y-%m-%dT%H:%M:%S"))
            print(f"Fetching data from {current_start} to {current_end}...")
            chunk_data = fetch_data(ticker, start=current_start.strftime("%Y-%m-%dT%H:%M:%S"), end=current_end.strftime("%Y-%m-%dT%H:%M:%S"))
            all_data.append(chunk_data[ticker])
            current_start = current_end
        data = pd.concat(all_data)
        data.to_csv(f"{ticker}_data.csv")
        data = data[['timestamp', 'open', 'high', 'low', 'close']]

        # Change 'timestamp' to timestamp index
        data.index = pd.to_datetime(data['timestamp'])
        data = data.drop(columns=['timestamp'])

        return data
    
    print(f"Fetching data from {start} to {end}...")
    data = fetch_data(ticker, start=start, end=end)[ticker]
    data.to_csv(f"{ticker}_data.csv")
    data = data[['timestamp', 'open', 'high', 'low', 'close']]


    # Change 'timestamp' to timestamp index
    data.index = pd.to_datetime(data['timestamp'])
    data = data.drop(columns=['timestamp'])

    return data


def main():
    """Run the backtest example."""
    print("QTC Trading System - Backtest Example")
    print("=" * 50)
    
    # Generate sample data
    print("\n1. Generating sample market data...")
    data = generate_sample_data("NVDA", days=20)
    print(f"   Generated {len(data)} days of data")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Create strategy
    print("\n2. Initializing Moving Average Strategy...")
    strategy = MovingAverageStrategy(short_window=20, long_window=50)
    print(f"   Strategy: {strategy.name}")
    print(f"   Parameters: {strategy.get_parameters()}")
    
    # Create backtest engine
    print("\n3. Setting up Backtesting Engine...")
    engine = BacktestEngine(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005
    )
    print(f"   Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"   Commission: {engine.commission * 100}%")
    print(f"   Slippage: {engine.slippage * 100}%")
    
    # Run backtest
    print("\n4. Running backtest...")
    results = engine.run_backtest(strategy, data)
    
    # Display results
    print("\n5. Results:")
    PerformanceMetrics.print_summary(results)
    
    # Additional analysis
    print("6. Additional Metrics:")
    metrics_df = PerformanceMetrics.calculate_metrics(results)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
