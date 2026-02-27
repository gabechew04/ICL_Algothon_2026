"""
API Client for QTC Orchestrator

Handles data fetching for local testing and development.
Mirrors the interface available via team["api"] during orchestrator execution.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date, timezone
import pandas as pd
import numpy as np


class DataPoint:
    """Represents a single market data point (minute bar)."""
    
    def __init__(self, timestamp: datetime, open_: float, high: float, low: float, 
                 close: float, volume: int = 0, trade_count: int = 0, vwap: float = 0.0):
        self.timestamp = timestamp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.trade_count = trade_count
        self.vwap = vwap


class APIClient:
    """
    API client for accessing historical market data.
    
    During local development/testing, this provides mock data.
    In production, this will be provided via team["api"].
    """
    
    def __init__(self, data: Optional[Dict[str, Dict[str, List]]] = None):
        """
        Initialize the API client.
        
        Args:
            data: Optional pre-loaded data in format:
                  {symbol: {"timestamp": [...], "open": [...], "close": [...], ...}}
        """
        self.data = data or {}
    
    def getLastN(self, symbol: str, count: int) -> pd.DataFrame:
        """
        Get the last N minutes of data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "NVDA")
            count: Number of minutes to retrieve
            
        Returns:
            DataFrame with columns: ticker, timestamp, open, high, low, close, volume
        """
        if symbol not in self.data:
            return pd.DataFrame()
        
        data = self.data[symbol]
        
        # Convert to DataFrame format
        rows = []
        timestamps = data.get('timestamp', [])
        opens = data.get('open', [])
        highs = data.get('high', [])
        lows = data.get('low', [])
        closes = data.get('close', [])
        volumes = data.get('volume', [])
        
        # Get last N records
        n_records = min(count, len(closes))
        start_idx = len(closes) - n_records
        
        for i in range(start_idx, len(closes)):
            rows.append({
                'ticker': symbol,
                'timestamp': timestamps[i] if i < len(timestamps) else None,
                'open': opens[i] if i < len(opens) else None,
                'high': highs[i] if i < len(highs) else None,
                'low': lows[i] if i < len(lows) else None,
                'close': closes[i],
                'volume': volumes[i] if i < len(volumes) else 0
            })
        
        return pd.DataFrame(rows)
    
    def getDay(self, symbol: str, target_date: date) -> pd.DataFrame:
        """
        Get all minute bars for a specific trading day.
        
        Args:
            symbol: Trading symbol
            target_date: Date to retrieve (as datetime.date)
            
        Returns:
            DataFrame with all bars for that day
        """
        if symbol not in self.data:
            return pd.DataFrame()
        
        data = self.data[symbol]
        
        # Convert to DataFrame and filter by date
        rows = []
        timestamps = data.get('timestamp', [])
        closes = data.get('close', [])
        
        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                bar_date = ts.date()
            else:
                bar_date = pd.Timestamp(ts).date()
            
            if bar_date == target_date and i < len(closes):
                rows.append({
                    'ticker': symbol,
                    'timestamp': ts,
                    'open': data.get('open', [None] * len(closes))[i],
                    'high': data.get('high', [None] * len(closes))[i],
                    'low': data.get('low', [None] * len(closes))[i],
                    'close': closes[i],
                    'volume': data.get('volume', [0] * len(closes))[i]
                })
        
        return pd.DataFrame(rows)
    
    def getRange(self, symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        Get market data for a specific time range.
        
        Args:
            symbol: Trading symbol
            start_dt: Start datetime (UTC)
            end_dt: End datetime (UTC)
            
        Returns:
            DataFrame with bars in the specified range
        """
        if symbol not in self.data:
            return pd.DataFrame()
        
        data = self.data[symbol]
        
        rows = []
        timestamps = data.get('timestamp', [])
        closes = data.get('close', [])
        
        for i, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                bar_ts = ts
            else:
                bar_ts = pd.Timestamp(ts).to_pydatetime()
            
            if start_dt <= bar_ts <= end_dt and i < len(closes):
                rows.append({
                    'ticker': symbol,
                    'timestamp': ts,
                    'open': data.get('open', [None] * len(closes))[i],
                    'high': data.get('high', [None] * len(closes))[i],
                    'low': data.get('low', [None] * len(closes))[i],
                    'close': closes[i],
                    'volume': data.get('volume', [0] * len(closes))[i]
                })
        
        return pd.DataFrame(rows)
    
    def getRangeMulti(self, symbols: List[str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        Get data for multiple symbols in a time range.
        
        Args:
            symbols: List of symbols
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            Combined DataFrame with all symbols
        """
        dfs = [self.getRange(symbol, start_dt, end_dt) for symbol in symbols]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def getDayMulti(self, symbols: List[str], target_date: date) -> pd.DataFrame:
        """
        Get data for multiple symbols on a specific day.
        
        Args:
            symbols: List of symbols
            target_date: Date to retrieve
            
        Returns:
            Combined DataFrame with all symbols
        """
        dfs = [self.getDay(symbol, target_date) for symbol in symbols]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def getLastNMulti(self, symbols: List[str], count: int) -> pd.DataFrame:
        """
        Get recent data for multiple symbols.
        
        Args:
            symbols: List of symbols
            count: Number of minutes to retrieve
            
        Returns:
            Combined DataFrame with all symbols
        """
        dfs = [self.getLastN(symbol, count) for symbol in symbols]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def create_team_dict(team_id: str, initial_capital: float = 100000.0, 
                     api_client: Optional[APIClient] = None) -> Dict[str, Any]:
    """
    Create a team state dictionary as would be passed by the orchestrator.
    
    Args:
        team_id: Team identifier/slug
        initial_capital: Starting cash amount
        api_client: APIClient instance for data fetching
        
    Returns:
        Team dict with structure: {id, name, cash, positions, params, api}
    """
    return {
        'id': team_id,
        'name': team_id,
        'cash': initial_capital,
        'positions': {},  # {symbol: quantity}
        'params': {},  # Team-specific parameter overrides
        'api': api_client or APIClient()
    }
