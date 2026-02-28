"""
bot.py — AlgothonBot: main orchestrator that wires all engines together.

Main loop (every N seconds):
    1. Fetch external data (weather, tides, flights)
    2. Compute fair values
    3. Sync positions from exchange
    4. Cancel stale orders
    5. Generate market-making quotes
    6. Check ETF arbitrage
    7. Check options mispricing
    8. Send orders (rate-limited)
    9. Log PnL & positions
"""
import sys
from pathlib import Path

# Ensure the parent "algothon" directory (where bot_template.py lives)
# is on sys.path when running this as algothon/bot/run.py
ALGOTHON_DIR = Path(__file__).resolve().parents[1]
if str(ALGOTHON_DIR) not in sys.path:
        sys.path.append(str(ALGOTHON_DIR))


import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from bot_template import BaseBot, OrderBook, Trade, Side, Product, OrderRequest

from data_engine import DataEngine
from fair_value_engine import FairValueEngine
from risk_engine import RiskEngine
from market_maker import MarketMaker
from arb_engine import ArbEngine
from options_engine import OptionsEngine

log = logging.getLogger("algothon.bot")


class AlgothonBot(BaseBot):
    """Full trading bot for the IMCity Algothon."""

    def __init__(
        self,
        cmi_url: str,
        username: str,
        password: str,
        aero_key: str = "",
        session_start: Optional[datetime] = None,
        session_end: Optional[datetime] = None,
        loop_interval: float = 5.0,
        data_interval: float = 120.0,
    ):
        super().__init__(cmi_url, username, password)

        london_tz = timezone(timedelta(hours=0))
        if session_start is None:
            now = datetime.now(tz=london_tz)
            session_start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if session_end is None:
            session_end = session_start + timedelta(hours=24)

        self.session_start = session_start
        self.session_end = session_end
        self.loop_interval = max(loop_interval, 2.0)
        self.data_interval = data_interval

        # ── Sub-engines ──────────────────────────────────────────────────────
        self.data_engine   = DataEngine(aero_key=aero_key, min_interval=data_interval)
        self.fv_engine     = FairValueEngine(self.data_engine)
        self.risk          = RiskEngine(max_position=100)
        self.mm            = MarketMaker(self.risk, self.fv_engine)
        self.arb           = ArbEngine(self.risk, min_edge=10)
        self.options       = OptionsEngine(self.fv_engine, self.risk)

        # ── State ────────────────────────────────────────────────────────────
        self._orderbooks: dict[str, OrderBook] = {}
        self._products: dict[str, Product] = {}
        self._last_data_fetch: float = 0.0
        self._running = False

        # Set up separated trade logs & interactive HTML viewer
        self.logs_dir = Path(__file__).resolve().parents[2] / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.html_log_path = self.logs_dir / "interactive_trades.html"
        self._init_html_log()

    # ── SSE Callbacks ────────────────────────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook):
        self._orderbooks[orderbook.product] = orderbook

    def on_trades(self, trade: Trade):
        if trade.buyer == self.username:
            side = Side.BUY
            log.info(f"[INSTRUMENT: {trade.product}] FILL: BOUGHT {trade.volume}x @ {trade.price}")
        else:
            side = Side.SELL
            log.info(f"[INSTRUMENT: {trade.product}] FILL: SOLD  {trade.volume}x @ {trade.price}")
        
        self.risk.record_fill(trade, side)

        # 1. Append to separate text file per instrument
        ts_str = datetime.now().strftime("%H:%M:%S")
        txt_path = self.logs_dir / f"{trade.product}_trades.log"
        with txt_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts_str}] {side.value} {trade.volume}x @ {trade.price}\n")
            
        # 2. Append row to interactive HTML log
        row = f'        <tr class="trade-row" data-product="{trade.product}"><td>{ts_str}</td><td>{trade.product}</td><td class="{side.value}">{side.value}</td><td>{trade.volume}</td><td>{trade.price}</td></tr>\n'
        with self.html_log_path.open("a", encoding="utf-8") as f:
            f.write(row)

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self):
        log.info("═" * 60)
        log.info("  ALGOTHON BOT STARTING")
        log.info(f"  Session: {self.session_start} → {self.session_end}")
        log.info("═" * 60)

        self._products = {p.symbol: p for p in self.get_products()}
        log.info(f"Products discovered: {list(self._products.keys())}")

        self.start()  # SSE stream
        self._running = True

        # Initial data load
        self.data_engine.fetch_all(force=True)
        self.fv_engine.update_all(self.session_start, self.session_end)
        self._log_fair_values()

        try:
            while self._running:
                t0 = time.time()
                self._trading_tick()
                sleep_time = max(0.0, self.loop_interval - (time.time() - t0))
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
        finally:
            self._shutdown()

    # ── Trading tick ─────────────────────────────────────────────────────────

    def _trading_tick(self):
        # 1. Refresh data
        if time.time() - self._last_data_fetch > self.data_interval:
            self.data_engine.fetch_all()
            self.fv_engine.update_all(self.session_start, self.session_end)
            self._last_data_fetch = time.time()
            self._log_fair_values()

        # 2. Sync positions
        try:
            self.risk.update_positions(self.get_positions())
        except Exception as e:
            log.warning(f"Position sync failed: {e}")

        # 3. Cancel stale orders
        try:
            self.cancel_all_orders()
        except Exception as e:
            log.warning(f"Cancel failed: {e}")

        time.sleep(0.5)  # small pause before re-quoting

        # 4. Collect all orders
        orders: list[OrderRequest] = []

        # Market making
        for symbol, product in self._products.items():
            ob = self._orderbooks.get(symbol)
            mid = self._get_market_mid(ob)
            orders.extend(self.mm.generate_quotes(symbol, product.tickSize, mid))

        # ETF arbitrage
        orders.extend(self.arb.check_arb(self._orderbooks))

        # Options mispricing
        fly_ob = self._orderbooks.get("LON_FLY")
        if fly_ob:
            orders.extend(self.options.check_mispricing(fly_ob))

        # 5. Send orders
        if orders:
            from collections import defaultdict
            orders_by_product = defaultdict(list)
            for o in orders:
                orders_by_product[o.product].append(o)
            
            for prod, pr_orders in orders_by_product.items():
                summary = "; ".join(f"{o.side} x{o.volume} @ {o.price}" for o in pr_orders)
                log.info(f"[INSTRUMENT: {prod}] Prepared {len(pr_orders)} orders: {summary}")
                
            self._send_rate_limited(orders)
        else:
            log.info("No candidate orders generated this tick")

        # 6. Status log
        self._log_status()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_market_mid(self, ob: Optional[OrderBook]) -> Optional[float]:
        if ob is None:
            return None
        bids = [o.price for o in ob.buy_orders  if o.volume - o.own_volume > 0]
        asks = [o.price for o in ob.sell_orders if o.volume - o.own_volume > 0]
        if bids and asks:
            return (max(bids) + min(asks)) / 2
        return None

    def _mark_price(self, product: str) -> Optional[float]:
        ob = self._orderbooks.get(product)
        if ob is None:
            return None
        bids = [o.price for o in ob.buy_orders  if (o.volume - o.own_volume) > 0]
        asks = [o.price for o in ob.sell_orders if (o.volume - o.own_volume) > 0]
        if bids and asks:
            return (max(bids) + min(asks)) / 2
        return max(bids) if bids else (min(asks) if asks else None)

    def _send_rate_limited(self, orders: list[OrderRequest], batch_size: int = 4):
        total = len(orders)
        if total == 0:
            return

        for i in range(0, total, batch_size):
            batch = orders[i:i + batch_size]
            batch_no = i // batch_size + 1
            n_batches = (total + batch_size - 1) // batch_size

            log.info(
                "Sending batch %d/%d (%d orders): %s",
                batch_no,
                n_batches,
                len(batch),
                "; ".join(
                    f"{o.product} {o.side} x{o.volume} @ {o.price}" for o in batch
                ),
            )

            responses = self.send_orders(batch)
            log.info(
                "Exchange acknowledged %d/%d orders in batch %d/%d",
                len(responses),
                len(batch),
                batch_no,
                n_batches,
            )

            if i + batch_size < total:
                time.sleep(1.0)

    def _shutdown(self):
        log.info("Shutting down...")
        try:
            self.cancel_all_orders()
        except Exception:
            pass
        self.stop()
        log.info("Bot stopped.")

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_fair_values(self):
        log.info("─── Fair Values ───────────────────────────────")
        for sym in sorted(self.fv_engine.fair_values):
            fv   = self.fv_engine.fair_values[sym]
            conf = self.fv_engine.confidence.get(sym, 0)
            log.info(f"  {sym:<12} FV={fv:>8.0f}  conf={conf:.2f}")

    def _compute_pnl(self) -> tuple[float, float, float]:
        realized = float(sum(self.risk.state.realized_pnl.values()))
        unrealized = 0.0
        for product, pos in self.risk.state.positions.items():
            try:
                pos_i = int(round(float(pos)))
            except Exception:
                continue
            if pos_i == 0:
                continue
            mark = self._mark_price(product)
            avg  = float(self.risk.state.avg_entry.get(product, 0.0))
            if mark is None or avg == 0.0:
                continue
            unrealized += pos_i * (mark - avg)
        return realized + unrealized, unrealized, realized


    def _init_html_log(self):
        """Initialize an HTML file that acts as an interactive UI with a dropdown."""
        header = '''<!DOCTYPE html>
<html>
<head>
    <title>Interactive Trade Log</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1e1e1e; color: #fff; padding: 20px; }
        select { margin-bottom: 20px; padding: 10px; font-size: 16px; background: #333; color: white; border: 1px solid #555; }
        table { border-collapse: collapse; width: 100%; max-width: 800px; }
        th, td { padding: 10px; border-bottom: 1px solid #444; text-align: left; }
        th { background: #2d2d2d; }
        .BUY { color: #4cc38a; font-weight: bold; }
        .SELL { color: #f45b5b; font-weight: bold; }
    </style>
    <script>
        function filterTable() {
            const filter = document.getElementById('productDropdown').value;
            const rows = document.querySelectorAll('tr.trade-row');
            rows.forEach(row => {
                if (filter === 'ALL' || row.dataset.product === filter) row.style.display = '';
                else row.style.display = 'none';
            });
        }
    </script>
</head>
<body>
    <h2>Live Trade Tracker</h2>
    <!-- Dropdown for filtering -->
    <select id="productDropdown" onchange="filterTable()">
        <option value="ALL">Show All Instruments</option>
        <option value="TIDE_SPOT">TIDE_SPOT</option>
        <option value="TIDE_SWING">TIDE_SWING</option>
        <option value="WX_SPOT">WX_SPOT</option>
        <option value="WX_SUM">WX_SUM</option>
        <option value="LHR_COUNT">LHR_COUNT</option>
        <option value="LHR_INDEX">LHR_INDEX</option>
        <option value="LON_ETF">LON_ETF</option>
        <option value="LON_FLY">LON_FLY</option>
    </select>
    <table>
        <thead><tr><th>Time</th><th>Product</th><th>Side</th><th>Volume</th><th>Price</th></tr></thead>
        <tbody>
'''
        with self.html_log_path.open("w", encoding="utf-8") as f:
            f.write(header)
    def _log_status(self):
        positions = {k: v for k, v in self.risk.state.positions.items() if abs(v) > 0}
        pos_str = (
            "  ".join(f"{k}={int(round(float(v))):+d}" for k, v in positions.items())
            if positions else "flat"
        )
        total, unreal, real = self._compute_pnl()
        log.info(
            f"Positions: {pos_str} | "
            f"PnL: total={total:+.1f} (unreal={unreal:+.1f}, real={real:+.1f})"
        )
