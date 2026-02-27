"""
Alpaca data downloader (do not commit keys; share only the CSVs in data/).

Usage:
  python download_alpaca_data.py --symbols AAPL,SPY --start 2025-10-10 --end 2025-12-31
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def _parse_dt(date_str: str, default_hour: int, default_minute: int) -> datetime:
    if "T" in date_str:
        cleaned = date_str.replace("Z", "")
        return datetime.fromisoformat(cleaned)
    return datetime.strptime(date_str, "%Y-%m-%d").replace(hour=default_hour, minute=default_minute)


def _merge_symbol_frames(frames: list) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "timestamp" in merged.columns:
        merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return merged.reset_index(drop=True)


def fetch_alpaca_bars(symbols: list, start_date: str, end_date: str, chunk_days: int = 10) -> dict:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except Exception as e:
        raise RuntimeError("alpaca-py is required. Install it via requirements.txt.") from e

    import os

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET in environment.")

    feed = os.getenv("ALPACA_DATA_FEED", "iex")
    client = StockHistoricalDataClient(api_key, api_secret)

    start_dt = _parse_dt(start_date, 9, 30).replace(tzinfo=timezone.utc)
    end_dt = _parse_dt(end_date, 16, 0).replace(tzinfo=timezone.utc)
    if end_dt <= start_dt:
        raise ValueError("end_date must be after start_date")

    symbol_frames = {symbol: [] for symbol in symbols}
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=current_start,
            end=current_end,
            feed=feed,
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            break

        if isinstance(df.index, pd.MultiIndex):
            for symbol in symbols:
                if symbol in df.index.get_level_values(0):
                    sym_df = df.xs(symbol)
                    if not sym_df.empty:
                        sym_df = sym_df.reset_index()
                        if sym_df["timestamp"].dt.tz is not None:
                            sym_df["timestamp"] = sym_df["timestamp"].dt.tz_convert(None)
                        symbol_frames[symbol].append(
                            sym_df[["timestamp", "open", "high", "low", "close", "volume"]]
                        )
        else:
            sym_df = df.reset_index()
            if "timestamp" not in sym_df.columns:
                sym_df.rename(columns={"index": "timestamp"}, inplace=True)
            if sym_df["timestamp"].dt.tz is not None:
                sym_df["timestamp"] = sym_df["timestamp"].dt.tz_convert(None)
            symbol_frames[symbols[0]].append(
                sym_df[["timestamp", "open", "high", "low", "close", "volume"]]
            )

        current_start = current_end

    result = {}
    for symbol in symbols:
        merged = _merge_symbol_frames(symbol_frames[symbol])
        if not merged.empty:
            result[symbol] = merged
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Alpaca minute bars to data/*.csv")
    parser.add_argument("--symbols", type=str, default="AAPL", help="Comma-separated symbols")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--chunk-days", type=int, default=10, help="Chunk size in days")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory for CSVs")
    args = parser.parse_args()

    load_dotenv()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    data = fetch_alpaca_bars(symbols, args.start, args.end, chunk_days=args.chunk_days)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol, df in data.items():
        df.to_csv(out_dir / f"{symbol}_data.csv", index=False)
        print(f"Saved {len(df)} bars to {out_dir}/{symbol}_data.csv")

    if not data:
        print("No data returned.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
