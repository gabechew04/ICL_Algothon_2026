"""
data_engine.py — Fetches & caches weather, tides, flights data.
"""

import time
import logging
import pandas as pd
import requests
from typing import Optional

log = logging.getLogger("algothon.data")

LONDON_LAT, LONDON_LON = 51.5074, -0.1278
THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"


class DataEngine:
    """Fetches and caches weather, tidal, and flight data with rate-limiting."""

    def __init__(self, aero_key: str = "", min_interval: float = 60.0):
        self.aero_key = aero_key
        self.min_interval = min_interval

        self.weather: Optional[pd.DataFrame] = None
        self.thames: Optional[pd.DataFrame] = None
        self.flights: Optional[dict] = None

        self._last_fetch = {"weather": 0.0, "thames": 0.0, "flights": 0.0}

    def _should_fetch(self, source: str) -> bool:
        return time.time() - self._last_fetch[source] > self.min_interval

    # ── Weather (Open-Meteo, free) ──────────────────────────────────────────

    def fetch_weather(self, force: bool = False) -> Optional[pd.DataFrame]:
        if not force and not self._should_fetch("weather"):
            return self.weather
        try:
            variables = (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "precipitation,wind_speed_10m,cloud_cover,visibility"
            )
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": LONDON_LAT, "longitude": LONDON_LON,
                    "minutely_15": variables,
                    "past_minutely_15": 96,
                    "forecast_minutely_15": 96,
                    "timezone": "Europe/London",
                },
                timeout=10,
            )
            resp.raise_for_status()
            m = resp.json()["minutely_15"]
            self.weather = pd.DataFrame({
                "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
                "temperature_c": m["temperature_2m"],
                "humidity": m["relative_humidity_2m"],
                "precipitation": m["precipitation"],
                "wind_speed": m["wind_speed_10m"],
                "cloud_cover": m["cloud_cover"],
                "visibility": m["visibility"],
                "apparent_temperature": m["apparent_temperature"],
            })
            self.weather["temperature_f"] = self.weather["temperature_c"] * 9 / 5 + 32
            self.weather["wx_metric"] = (
                self.weather["temperature_f"] * self.weather["humidity"]
            )
            self._last_fetch["weather"] = time.time()
            log.info(f"Weather: {len(self.weather)} rows fetched")
        except Exception as e:
            log.warning(f"Weather fetch failed: {e}")
        return self.weather

    # ── Thames Tidal (EA Flood Monitoring, free) ────────────────────────────

    def fetch_thames(self, limit: int = 200, force: bool = False) -> Optional[pd.DataFrame]:
        if not force and not self._should_fetch("thames"):
            return self.thames
        try:
            resp = requests.get(
                f"https://environment.data.gov.uk/flood-monitoring/id/measures/"
                f"{THAMES_MEASURE}/readings",
                params={"_sorted": "", "_limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            df = pd.DataFrame(items)[["dateTime", "value"]].rename(
                columns={"dateTime": "time", "value": "level"}
            )
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
            self.thames = df.sort_values("time").reset_index(drop=True)
            self._last_fetch["thames"] = time.time()
            log.info(f"Thames: {len(self.thames)} readings fetched")
        except Exception as e:
            log.warning(f"Thames fetch failed: {e}")
        return self.thames

    # ── Flights (AeroDataBox via RapidAPI) ──────────────────────────────────

    def fetch_flights(self, offset_minutes: int = -720, duration_minutes: int = 720,
                      force: bool = False) -> Optional[dict]:
        """Fetch flights. Use sparingly — free tier is ~150 req/month."""
        if not force and not self._should_fetch("flights"):
            return self.flights
        if not self.aero_key:
            log.warning("No AeroDataBox API key set — skipping flights")
            return self.flights
        try:
            host = "aerodatabox.p.rapidapi.com"
            url = (
                f"https://{host}/flights/airports/iata/LHR"
                f"?offsetMinutes={offset_minutes}"
                f"&durationMinutes={duration_minutes}&direction=Both"
            )
            resp = requests.get(url, headers={
                "x-rapidapi-host": host, "x-rapidapi-key": self.aero_key,
            }, timeout=15)
            resp.raise_for_status()
            self.flights = resp.json()
            n_arr = len(self.flights.get("arrivals", []))
            n_dep = len(self.flights.get("departures", []))
            self._last_fetch["flights"] = time.time()
            log.info(f"Flights: {n_arr} arrivals, {n_dep} departures")
        except Exception as e:
            log.warning(f"Flights fetch failed: {e}")
        return self.flights

    def fetch_all(self, force: bool = False):
        self.fetch_weather(force=force)
        self.fetch_thames(force=force)
        self.fetch_flights(force=force)
