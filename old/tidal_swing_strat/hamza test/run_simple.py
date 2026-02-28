"""
run_simple.py — Lance le bot simplifié.

Usage: python run_simple.py
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta, timezone

from simple_bot import SimpleBot

# ── Logging ───────────────────────────────────────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s [%(name)-8s] %(message)s", datefmt="%H:%M:%S")

# Console
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

# File
log_file = Path("bot.log")
fh = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
fh.setFormatter(fmt)
logger.addHandler(fh)

# ── Config ────────────────────────────────────────────────────────────────
EXCHANGE_URL    = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
USERNAME        = "zab"
PASSWORD        = "zab"
AERODATABOX_KEY = "0a7ff9f16fmsh54fb7da32af7310p12b89fjsn800a543a8628"

# Session: Saturday 12pm → Sunday 12pm (London = GMT in February)
LONDON_TZ = timezone(timedelta(hours=0))
SESSION_START = datetime(2026, 2, 28, 12, 0, 0, tzinfo=LONDON_TZ)
SESSION_END   = datetime(2026, 3,  1, 12, 0, 0, tzinfo=LONDON_TZ)

# ── Go ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = SimpleBot(
        cmi_url=EXCHANGE_URL,
        username=USERNAME,
        password=PASSWORD,
        aero_key=AERODATABOX_KEY,
        session_start=SESSION_START,
        session_end=SESSION_END,
        loop_interval=3.0,      # Cycle toutes les 3s
        data_interval=120.0,    # Refresh data toutes les 2min
    )

    print("=" * 60)
    print("  SIMPLE BOT — 3 strategies, clear logging")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nStopped.")
