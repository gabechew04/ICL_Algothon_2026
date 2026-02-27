import httpx
from typing import Optional, Dict, Any

# load api base url from environment or use deault

from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

BASE_URL = os.getenv("QTC_API_BASE_URL", "https://api.qtcq.xyz")

API_KEY = os.getenv("QTC_API_KEY", "")
# print(API_KEY)

def fetch_data(symbols: str, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, str] = {"symbols": symbols}

    extension = "api/v1/market-data/bars"

    if start:
        params["start"] = start
    if end:
        params["end"] = end
    
    params["key"] = API_KEY
    

    resp = httpx.get(f"{BASE_URL}/{extension}", params=params)
    resp.raise_for_status()

    data = resp.json()

    main_df = {}
    if('data' in data):
        
        for symbol in symbols.split(","):
            df = pd.DataFrame(data["data"][symbol]['bars'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            main_df[symbol] = df
        return main_df
    else:
        df = pd.DataFrame(data['bars'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        main_df[symbols] = df
        return main_df

print(fetch_data("NVDA","2025-10-01T09:30:00","2025-10-02T16:00:00"))
