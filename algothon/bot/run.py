"""run.py — Bot entrypoint.

Configures logging (console + file), session window and credentials,
then starts the bot.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta, timezone

from bot import AlgothonBot

# ── Logging ───────────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "bot_run.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

# Rotating file handler (~1 MB per file, keep a few backups)
fh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

# ── Config ───────────────────────────────────────────────────────────────────

EXCHANGE_URL    = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
USERNAME        = "testhamza1"
PASSWORD        = "testhamza1"
AERODATABOX_KEY = "0a7ff9f16fmsh54fb7da32af7310p12b89fjsn800a543a8628"

# London is UTC+0 in winter (GMT), UTC+1 in summer (BST)
# Late February = GMT → UTC+0
LONDON_TZ     = timezone(timedelta(hours=0))
SESSION_START = datetime(2026, 2, 28, 12, 0, 0, tzinfo=LONDON_TZ)
SESSION_END   = datetime(2026, 3,  1, 12, 0, 0, tzinfo=LONDON_TZ)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = AlgothonBot(
        cmi_url=EXCHANGE_URL,
        username=USERNAME,
        password=PASSWORD,
        aero_key=AERODATABOX_KEY,
        session_start=SESSION_START,
        session_end=SESSION_END,
        loop_interval=5.0,
        data_interval=120.0,
    )

    print("Starting bot — press Ctrl+C to stop.")
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nStopped.")
