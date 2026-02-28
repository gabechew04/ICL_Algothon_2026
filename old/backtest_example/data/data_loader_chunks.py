import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .data import fetch_data


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