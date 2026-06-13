"""
Main bot loop for the XAU/USD algorithm trader.

Cycle (every 30 seconds on M15 data):
  1. Refresh news — halt if high-impact event is live
  2. Check daily loss limit — halt if breached
  3. Skip if max open trades reached
  4. Pull OHLCV bars for M15 (signal) and H1 (trend)
  5. Run strategy.evaluate()
  6. If signal: size position → place order
"""

import logging
import signal
import sys
import time
from datetime import datetime

from . import mt5_connector as mt5c
from .config import strategy_cfg as scfg, risk_cfg as rcfg
from .news_feed import news_feed
from .strategy import evaluate

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 30   # check every 30 s; M15 bars close every 900 s


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _shutdown(signum, frame):
    logger.info("Shutdown signal received — disconnecting MT5.")
    mt5c.disconnect()
    sys.exit(0)


def run(log_level: str = "INFO") -> None:
    _setup_logging(log_level)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("=== XAU/USD Trader Bot starting ===")
    logger.info(
        "Strategy: trend on %s | signals on %s | TP=%d pts | SL=%d pts | max_lot=%.2f",
        scfg.trend_timeframe,
        scfg.timeframe,
        rcfg.take_profit_points,
        rcfg.stop_loss_points,
        rcfg.max_lot,
    )

    if not mt5c.connect():
        logger.critical("Cannot connect to MetaTrader 5 — exiting.")
        sys.exit(1)

    consecutive_errors = 0

    while True:
        try:
            _tick(scfg.symbol)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            logger.error("Unhandled error in tick (attempt %d): %s", consecutive_errors, exc, exc_info=True)
            if consecutive_errors >= 10:
                logger.critical("Too many consecutive errors — shutting down.")
                mt5c.disconnect()
                sys.exit(1)

        time.sleep(POLL_INTERVAL_SEC)


def _tick(symbol: str) -> None:
    # ── 1. News guard ────────────────────────────────────────────────────────
    high_impact = news_feed.high_impact_near_now(window_minutes=30)
    if high_impact:
        logger.info(
            "HIGH-IMPACT NEWS — pausing entry: '%s' (%s)",
            high_impact.title,
            high_impact.published_at.strftime("%H:%M UTC"),
        )
        _print_headlines()
        return

    # ── 2. Daily loss guard ──────────────────────────────────────────────────
    if mt5c.daily_loss_exceeded(symbol):
        logger.warning("Daily loss limit reached — no new trades today.")
        return

    # ── 3. Open trade count guard ────────────────────────────────────────────
    open_trades = mt5c.get_open_bot_trades(symbol)
    if len(open_trades) >= rcfg.max_open_trades:
        logger.debug("Max open trades (%d) reached — skipping.", rcfg.max_open_trades)
        return

    # ── 4. Fetch OHLCV bars ──────────────────────────────────────────────────
    m15 = mt5c.get_ohlcv(symbol, scfg.timeframe, count=300)
    h1 = mt5c.get_ohlcv(symbol, scfg.trend_timeframe, count=300)
    if m15 is None or h1 is None:
        logger.warning("Could not fetch bars — skipping tick.")
        return

    # ── 5. Evaluate strategy ─────────────────────────────────────────────────
    sig = evaluate(m15, h1)
    if sig is None:
        logger.debug("[%s] No signal.", datetime.utcnow().strftime("%H:%M:%S"))
        return

    logger.info(
        "*** SIGNAL *** %s %s | strength=%.2f | %s",
        sig.direction, sig.entry_type, sig.strength, sig.reason,
    )

    # ── 6. Size and place order ──────────────────────────────────────────────
    equity = mt5c.get_account_equity()
    lot = mt5c.calc_lot_size(rcfg.stop_loss_points, equity)

    comment = f"xau-bot-{sig.entry_type.lower()[:2]}-{sig.strength:.0%}"
    result = mt5c.place_order(
        symbol=symbol,
        direction=sig.direction,
        lot=lot,
        sl_points=rcfg.stop_loss_points,
        tp_points=rcfg.take_profit_points,
        comment=comment,
    )

    if result:
        logger.info(
            "Trade opened — ticket=%s | %s | lot=%.2f | TP=%d pts | SL=%d pts",
            result.order,
            sig.direction,
            lot,
            rcfg.take_profit_points,
            rcfg.stop_loss_points,
        )
    else:
        logger.error("Order placement failed.")


def _print_headlines() -> None:
    headlines = news_feed.latest_headlines(n=5)
    for h in headlines:
        flag = "⚠ " if h.is_high_impact else "  "
        logger.info("%s[%s] %s — %s", flag, h.published_at.strftime("%H:%M"), h.title, h.source)
