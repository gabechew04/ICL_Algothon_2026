"""
QTC Market Data Fetcher

Fetches real market data from QTC API (https://api.qtcq.xyz) and integrates with APIClient.
Implements the QTC Trader Handbook data access interface (Section 3.2).

Configuration:
- Create a .env file with QTC API credentials (see .env.example)
- Or set environment variables: QTC_API_KEY, QTC_API_BASE_URL
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import requests
from urllib.parse import urljoin

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


logger = logging.getLogger(__name__)


class QTCDataFetcher:
    """
    Fetches real market data from QTC API (https://api.qtcq.xyz).
    
    Provides methods matching the QTC handbook specification:
    - getLastN(symbol, count) - GET /api/v1/market-data/{symbol}/recent/{count}
    - getDay(symbol, date) - GET /api/v1/market-data/{symbol}/day/{date}
    - getRange(symbol, start_dt, end_dt) - GET /api/v1/market-data/{symbol}/range
    - getLastNMulti(symbols, count)
    - getDayMulti(symbols, date)
    - getRangeMulti(symbols, start_dt, end_dt)
    """
    
    # Default QTC API base URL
    DEFAULT_BASE_URL = "https://api.qtcq.xyz"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize QTC data fetcher.
        
        Args:
            api_key: QTC API key (or set QTC_API_KEY env var)
            base_url: QTC API base URL (or set QTC_API_BASE_URL env var)
        """
        # Get credentials from args or environment
        self.api_key = api_key or os.getenv('QTC_API_KEY')
        self.base_url = base_url or os.getenv('QTC_API_BASE_URL', self.DEFAULT_BASE_URL)
        
        # API key is optional for public endpoints
        self.headers = {}
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        logger.info(f"QTC data fetcher initialized ({self.base_url})")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make HTTP request to QTC API.
        
        Args:
            endpoint: API endpoint (e.g., '/api/v1/market-data/AAPL/recent/100')
            params: Query parameters
            
        Returns:
            JSON response or None on error
        """
        try:
            url = urljoin(self.base_url, endpoint)
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Try to get error details from response
            try:
                error_data = e.response.json()
                logger.error(f"API request failed ({e.response.status_code}): {error_data}")
            except:
                logger.error(f"API request failed: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def getLastN(self, symbol: str, count: int) -> pd.DataFrame:
        """
        Get last N minute bars for a symbol.
        
        Endpoint: GET /api/v1/market-data/{symbol}/recent/{count}
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            count: Number of bars to return
            
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume, trade_count, vwap]
        """
        try:
            endpoint = f"/api/v1/market-data/{symbol}/recent/{count}"
            data = self._make_request(endpoint)
            
            if not data or 'data' not in data:
                logger.warning(f"No data found for {symbol}")
                return self._empty_dataframe()
            
            df = pd.DataFrame(data['data'])
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching last {count} bars for {symbol}: {e}")
            return self._empty_dataframe()
    
    def getDay(self, symbol: str, date: str) -> pd.DataFrame:
        """
        Get all minute bars for a specific trading day.
        
        Endpoint: GET /api/v1/market-data/{symbol}/day/{date}
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            date: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume, trade_count, vwap]
        """
        try:
            endpoint = f"/api/v1/market-data/{symbol}/day/{date}"
            data = self._make_request(endpoint)
            
            if not data or 'data' not in data:
                logger.warning(f"No data found for {symbol} on {date}")
                return self._empty_dataframe()
            
            df = pd.DataFrame(data['data'])
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching day data for {symbol} on {date}: {e}")
            return self._empty_dataframe()
    
    def getRange(self, symbol: str, start_dt: str, end_dt: str) -> pd.DataFrame:
        """
        Get minute bars for a date/time range.
        
        Endpoint: GET /api/v1/market-data/{symbol}/range?start=...&end=...
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_dt: Start datetime (ISO format: 2025-01-15T09:30:00Z or YYYY-MM-DD)
            end_dt: End datetime (ISO format: 2025-01-15T16:00:00Z or YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume, trade_count, vwap]
        """
        try:
            # Convert to ISO format if needed
            start_dt = self._to_iso_format(start_dt)
            end_dt = self._to_iso_format(end_dt)
            
            endpoint = f"/api/v1/market-data/{symbol}/range"
            params = {'start': start_dt, 'end': end_dt}
            data = self._make_request(endpoint, params)
            
            if not data or 'data' not in data:
                logger.warning(f"No data found for {symbol} between {start_dt} and {end_dt}")
                return self._empty_dataframe()
            
            df = pd.DataFrame(data['data'])
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching range for {symbol}: {e}")
            return self._empty_dataframe()
    
    def getLastNMulti(self, symbols: List[str], count: int) -> pd.DataFrame:
        """
        Get last N minute bars for multiple symbols.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'NVDA', 'SPY'])
            count: Number of bars per symbol
            
        Returns:
            DataFrame with columns: [symbol, timestamp, open, high, low, close, volume, trade_count, vwap]
        """
        all_data = []
        
        for symbol in symbols:
            try:
                df = self.getLastN(symbol, count)
                if not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")
        
        if not all_data:
            return self._empty_dataframe_multi()
        
        result = pd.concat(all_data, ignore_index=True)
        cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        return result[[c for c in cols if c in result.columns]]
    
    def getDayMulti(self, symbols: List[str], date: str) -> pd.DataFrame:
        """
        Get all minute bars for multiple symbols on a specific day.
        
        Args:
            symbols: List of stock symbols
            date: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: [symbol, timestamp, open, high, low, close, volume, trade_count, vwap]
        """
        all_data = []
        
        for symbol in symbols:
            try:
                df = self.getDay(symbol, date)
                if not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Skipping {symbol} for {date}: {e}")
        
        if not all_data:
            return self._empty_dataframe_multi()
        
        result = pd.concat(all_data, ignore_index=True)
        cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        return result[[c for c in cols if c in result.columns]]
    
    def getRangeMulti(self, symbols: List[str], start_dt: str, end_dt: str) -> pd.DataFrame:
        """
        Get minute bars for multiple symbols over a date range.
        
        Args:
            symbols: List of stock symbols
            start_dt: Start datetime (ISO format or YYYY-MM-DD)
            end_dt: End datetime (ISO format or YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: [symbol, timestamp, open, high, low, close, volume, trade_count, vwap]
        """
        all_data = []
        
        for symbol in symbols:
            try:
                df = self.getRange(symbol, start_dt, end_dt)
                if not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")
        
        if not all_data:
            return self._empty_dataframe_multi()
        
        result = pd.concat(all_data, ignore_index=True)
        cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        return result[[c for c in cols if c in result.columns]]
    
    def getBars(self, symbols: List[str], start_dt: str, end_dt: str) -> Dict[str, pd.DataFrame]:
        """
        Get minute bars using the unified /api/v1/market-data/bars endpoint.
        
        This is the primary endpoint for fetching bars and supports both single and multi-symbol queries.
        
        Endpoint: GET /api/v1/market-data/bars?symbols=...&start=...&end=...&key=...
        
        Args:
            symbols: Single symbol (str) or list of symbols (e.g., 'AAPL' or ['AAPL', 'SPY', 'NVDA'])
            start_dt: Start datetime (ISO format: 2025-10-01T09:30:00 or YYYY-MM-DD)
            end_dt: End datetime (ISO format: 2025-10-02T16:00:00 or YYYY-MM-DD)
            
        Returns:
            Dict[symbol, DataFrame] with columns: [timestamp, open, high, low, close, volume]
            
        Example:
            fetcher = QTCDataFetcher(api_key='YOUR_KEY')
            
            # Single symbol
            bars = fetcher.getBars('AAPL', '2025-10-01', '2025-10-02')
            df = bars['AAPL']  # Returns DataFrame
            
            # Multiple symbols
            bars = fetcher.getBars(['AAPL', 'SPY'], '2025-10-01', '2025-10-02')
            aapl_df = bars['AAPL']
            spy_df = bars['SPY']
        """
        try:
            # Normalize input
            if isinstance(symbols, str):
                symbols_list = [symbols]
            else:
                symbols_list = symbols if isinstance(symbols, list) else list(symbols)
            
            # Convert to ISO format
            start_dt = self._to_iso_format(start_dt)
            end_dt = self._to_iso_format(end_dt)
            
            # Build request - key must be included as query parameter per API docs
            endpoint = "/api/v1/market-data/bars"
            params = {
                'symbols': ','.join(symbols_list),
                'start': start_dt,
                'end': end_dt,
                'key': self.api_key
            }
            
            if not self.api_key:
                logger.error("API key is required for getBars endpoint")
                return {sym: self._empty_dataframe() for sym in symbols_list}
            
            logger.info(f"Fetching bars for {','.join(symbols_list)} from {start_dt} to {end_dt}")
            data = self._make_request(endpoint, params)
            
            if not data:
                logger.warning(f"No data returned from getBars")
                return {sym: self._empty_dataframe() for sym in symbols_list}
            
            result = {}
            
            # Handle single symbol response
            if 'symbol' in data:
                symbol = data['symbol']
                bars = data.get('bars', [])
                df = pd.DataFrame(bars)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                result[symbol] = df
                logger.info(f"✓ {symbol}: {len(df)} bars")
            
            # Handle multi-symbol response
            elif 'data' in data:
                for symbol in symbols_list:
                    if symbol in data['data']:
                        bars = data['data'][symbol].get('bars', [])
                        df = pd.DataFrame(bars)
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        result[symbol] = df
                        logger.info(f"✓ {symbol}: {len(df)} bars")
                    else:
                        result[symbol] = self._empty_dataframe()
                        logger.warning(f"✗ {symbol}: No data")
            
            else:
                logger.warning(f"Unexpected response format: {data.keys()}")
                return {sym: self._empty_dataframe() for sym in symbols_list}
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching bars: {e}")
            return {sym: self._empty_dataframe() for sym in (symbols if isinstance(symbols, list) else [symbols])}
    
    @staticmethod
    def _to_iso_format(dt_string: str) -> str:
        """Convert datetime string to ISO format (YYYY-MM-DDTHH:MM:SSZ)."""
        try:
            # If it's already ISO format, return as-is
            if 'T' in dt_string:
                return dt_string
            
            # If it's just a date, convert to ISO start of day
            parsed = datetime.strptime(dt_string, '%Y-%m-%d')
            return parsed.isoformat() + 'Z'
        except:
            return dt_string
    
    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        """Return empty DataFrame with correct structure."""
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'])
    
    @staticmethod
    def _empty_dataframe_multi() -> pd.DataFrame:
        """Return empty DataFrame with correct structure for multi-symbol queries."""
        return pd.DataFrame(columns=['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'])


class QTCIntegration:
    """
    Helper to integrate QTC API data with the existing APIClient.
    
    Usage:
        # Initialize
        qtc = QTCIntegration()
        
        # Use with APIClient
        api = qtc.get_api_client()
        
        # Or use directly
        df = qtc.get_bars('AAPL', '2025-01-15')
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """Initialize QTC integration."""
        self.fetcher = QTCDataFetcher(api_key, base_url)
    
    def get_api_client(self):
        """
        Get an APIClient configured to use QTC API data.
        
        Returns:
            APIClient instance with QTC data backend
        """
        from api_client import APIClient
        
        # Create a custom APIClient that uses QTC data
        class QTCAPIClient(APIClient):
            def __init__(self, fetcher):
                super().__init__()
                self.fetcher = fetcher
            
            def getLastN(self, symbol: str, count: int) -> pd.DataFrame:
                return self.fetcher.getLastN(symbol, count)
            
            def getDay(self, symbol: str, date: str) -> pd.DataFrame:
                return self.fetcher.getDay(symbol, date)
            
            def getRange(self, symbol: str, start_dt: str, end_dt: str) -> pd.DataFrame:
                return self.fetcher.getRange(symbol, start_dt, end_dt)
            
            def getLastNMulti(self, symbols: list, count: int) -> pd.DataFrame:
                return self.fetcher.getLastNMulti(symbols, count)
            
            def getDayMulti(self, symbols: list, date: str) -> pd.DataFrame:
                return self.fetcher.getDayMulti(symbols, date)
            
            def getRangeMulti(self, symbols: list, start_dt: str, end_dt: str) -> pd.DataFrame:
                return self.fetcher.getRangeMulti(symbols, start_dt, end_dt)
        
        return QTCAPIClient(self.fetcher)
    
    def get_bars(self, symbol: str, date: str) -> pd.DataFrame:
        """Get bars for a symbol on a specific date."""
        return self.fetcher.getDay(symbol, date)


# Utility function for quick setup
def setup_qtc_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
):
    """
    Quick setup for QTC API client with team dict.
    
    Usage:
        team = setup_qtc_client()
        strategy = MyStrategy()
        engine = BacktestEngine()
        results = engine.run_backtest(strategy, bars, team=team)
    
    Args:
        api_key: QTC API key (or use env variable QTC_API_KEY)
        base_url: QTC API base URL (defaults to https://api.qtcq.xyz)
        
    Returns:
        Team dictionary ready for backtesting with QTC data
    """
    from api_client import create_team_dict
    
    qtc = QTCIntegration(api_key, base_url)
    api_client = qtc.get_api_client()
    
    team = create_team_dict(
        team_id="QTC",
        initial_capital=100000.0,
        api_client=api_client
    )
    
    return team


# Backward compatibility aliases
AlpacaDataFetcher = QTCDataFetcher
AlpacaIntegration = QTCIntegration
setup_alpaca_client = setup_qtc_client
