# QTC-Team-H
QTC Trading System - Backtesting with Real QTC API Data

## Overview

Backtesting system for trading strategies using QTC data with local CSV fallback.

Features:
- QTC API integration with chunked fetching
- Local CSV fallback (`data/*.csv`) when QTC is down
- Minute-level backtesting engine with signal validation

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your QTC API key to .env
echo "QTC_API_KEY=your_api_key_here" > .env
```

### 2. Run Backtest

```bash
python backtest.py strategies/mean_reversion_strategy.py --start 2025-10-10 --end 2025-12-31
```

### Data Sources (Team Workflow)

- Backtests first attempt QTC using `QTC_API_KEY`.
- If QTC is down, they fall back to local CSVs in `data/` (e.g., `AAPL_data.csv`).
- Only maintainers with Alpaca keys should run `download_alpaca_data.py` to refresh CSVs and commit the data files.

## Project Structure

```
QTC-Team-H/
├── backtest.py             # Main backtest runner
├── download_alpaca_data.py # Optional: download Alpaca data into data/
├── alpaca_data_fetcher.py  # QTC API client for fetching market data
├── api_client.py           # APIClient interface and team management
├── signal_validator.py     # Signal validation against handbook schema
├── strategies/             # Trading strategy implementations
├── backtesting/            # Backtesting engine
│   └── engine.py           # Core backtesting engine
├── requirements.txt        # Python dependencies
└── .env                    # API configuration (add your key here)
```

Edit `.env` and add your actual API keys:

```env
QTC_API_KEY=your_actual_qtc_api_key
```

Optional (maintainers only, for downloading Alpaca data):

```env
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
ALPACA_DATA_FEED=iex
```

## Usage

### Running a Backtest

```bash
python backtest.py strategies/mean_reversion_strategy.py --start 2025-10-10 --end 2025-12-31 --symbols AAPL
```

### Refreshing Local Data (Maintainers)

```bash
python download_alpaca_data.py --symbols AAPL,SPY --start 2025-10-10 --end 2025-12-31
```

### Creating a Custom Strategy

Create a new strategy by defining a `Strategy` class with a `generate_signal` method:

```python
class Strategy:
    def __init__(self, quantity: int = 1):
        self.quantity = quantity

    def generate_signal(self, team, bars, current_prices):
        # Implement your signal generation logic
        return None
```

### Using the Backtesting Engine

```python
from backtesting.engine import BacktestEngine

engine = BacktestEngine(initial_capital=100000.0)
results = engine.run_backtest(strategy, bars, team=team)
```

## Features

### Strategies Module

- **BaseStrategy**: Abstract base class for all strategies
  - `generate_signals()`: Generate trading signals from historical data
  - `on_data()`: Process real-time market data
  - Parameter management

- **MovingAverageStrategy**: Example implementation
  - Simple moving average crossover
  - Configurable short and long windows

### Backtesting Engine

- Historical strategy simulation
- Transaction cost modeling (commission + slippage)
- Performance metrics calculation:
  - Total return
  - Sharpe ratio
  - Maximum drawdown
  - Win rate
  - Trade statistics

### Performance Metrics

- Comprehensive metric calculation
- Formatted result summaries
- Additional metrics:
  - Volatility
  - Calmar ratio
  - Sortino ratio

## API Configuration

The `.env.example` file provides a template for configuring trading API credentials:

- `API_KEY`: Your trading platform API key
- `API_SECRET`: Your API secret
- `API_BASE_URL`: API endpoint (paper or live trading)
- `WEBSOCKET_URL`: WebSocket endpoint for real-time data
- `TRADING_MODE`: 'paper' or 'live'

## Security

- Never commit your `.env` file to version control
- The `.gitignore` file already excludes `.env` files
- Use paper trading mode for testing strategies
- Always validate strategies with backtesting before live trading

## Contributing

Feel free to add new strategies, improve the backtesting engine, or enhance performance metrics.

## License

This project is for educational and research purposes.
