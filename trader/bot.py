"""
Main bot loop for the XAU/USD algorithm trader.

Cycle (every 30 seconds on M15 data):
  1. Refresh news — halt + notify if high-impact event is live
  2. Check daily loss limit — halt + notify if breached
  3. Check pause flag (set via iPhone dashboard)
  4. Skip if max open trades reached
  5. Pull OHLCV bars for M15 (signal) and H1 (trend)
  6. Run strategy.evaluate()
  7. If signal: size position → place order → push iPhone notification
"""

import logging
import signal
import sys
import time
from datetime import datetime

from . import mt5_connector as mt5c
from .config import strategy_cfg as scfg, risk_cfg as rcfg
from .news_feed import news_feed
from .notifications import (
    notify_trade_opened,
    notify_high_impact_news,
    notify_daily_limit_hit,
    notify_status,
)
from .strategy import evaluate

logger = logging.getLogger(__name__)

# Optional API server state — imported lazily so the bot works without Flask too
_api_state = None

POLL_INTERVAL_SEC = 30


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _shutdown(signum, frame):
    logger.info("Shutdown signal received — disconnecting MT5.")
    notify_status("Bot stopped.")
    mt5c.disconnect()
    sys.exit(0)


def _api_update(**kwargs):
    """Push state into the API server if it's running."""
    if _api_state is not None:
        _api_state.update_state(**kwargs)


def run(log_level: str = "INFO", with_api: bool = False,
        api_host: str = "0.0.0.0", api_port: int = 8080) -> None:
    global _api_state

    _setup_logging(log_level)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("=== XAU/USD Trader Bot starting ===")
    logger.info(
        "Strategy: trend on %s | signals on %s | TP=%d pts | SL=%d pts | max_lot=%.2f",
        scfg.trend_timeframe, scfg.timeframe,
        rcfg.take_profit_points, rcfg.stop_loss_points, rcfg.max_lot,
    )

    if with_api:
        try:
            from . import api_server
            _api_state = api_server
            api_server.start_api_server(host=api_host, port=api_port)
        except ImportError:
            logger.warning("Flask not installed — iPhone dashboard disabled. pip install flask")

    if not mt5c.connect():
        logger.critical("Cannot connect to MetaTrader 5 — exiting.")
        sys.exit(1)

    notify_status(
        f"Bot started — {scfg.symbol} | TP {rcfg.take_profit_points}pt | SL {rcfg.stop_loss_points}pt"
    )

    consecutive_errors = 0

    while True:
        try:
            _tick(scfg.symbol)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            logger.error("Unhandled error in tick (attempt %d): %s",
                         consecutive_errors, exc, exc_info=True)
            if consecutive_errors >= 10:
                logger.critical("Too many consecutive errors — shutting down.")
                notify_status("Bot crashed — too many errors. Check logs.")
                mt5c.disconnect()
                sys.exit(1)

        time.sleep(POLL_INTERVAL_SEC)


def _tick(symbol: str) -> None:
    # ── 1. News guard ────────────────────────────────────────────────────────
    news_feed.refresh_if_needed()
    high_impact = news_feed.high_impact_near_now(window_minutes=30)
    if high_impact:
        logger.info(
            "HIGH-IMPACT NEWS — pausing entry: '%s' (%s)",
            high_impact.title,
            high_impact.published_at.strftime("%H:%M UTC"),
        )
        notify_high_impact_news(high_impact.title)
        _print_headlines()
        return

    # ── 2. Daily loss guard ──────────────────────────────────────────────────
    if mt5c.daily_loss_exceeded(symbol):
        logger.warning("Daily loss limit reached — no new trades today.")
        notify_daily_limit_hit()
        return

    # ── 3. Pause guard (iPhone dashboard) ────────────────────────────────────
    if _api_state is not None and _api_state.is_paused():
        logger.debug("Bot is paused via iPhone dashboard — skipping tick.")
        return

    # ── 4. Open trade count guard ────────────────────────────────────────────
    open_trades = mt5c.get_open_bot_trades(symbol)
    equity = mt5c.get_account_equity()

    # Sync open trades to iPhone dashboard
    _api_update(
        open_trades=[
            {
                "type": "BUY" if p.type == 0 else "SELL",
                "lot": p.volume,
                "profit": p.profit,
                "ticket": p.ticket,
            }
            for p in open_trades
        ],
        equity=equity,
    )

    if len(open_trades) >= rcfg.max_open_trades:
        logger.debug("Max open trades (%d) reached — skipping.", rcfg.max_open_trades)
        return

    # ── 5. Fetch OHLCV bars ──────────────────────────────────────────────────
    m15 = mt5c.get_ohlcv(symbol, scfg.timeframe, count=300)
    h1 = mt5c.get_ohlcv(symbol, scfg.trend_timeframe, count=300)
    if m15 is None or h1 is None:
        logger.warning("Could not fetch bars — skipping tick.")
        return

    # ── 6. Evaluate strategy ─────────────────────────────────────────────────
    sig = evaluate(m15, h1)
    _api_update(last_signal=sig)

    if sig is None:
        logger.debug("[%s] No signal.", datetime.utcnow().strftime("%H:%M:%S"))
        return

    logger.info(
        "*** SIGNAL *** %s %s | strength=%.2f | %s",
        sig.direction, sig.entry_type, sig.strength, sig.reason,
    )

    # ── 7. Size and place order ──────────────────────────────────────────────
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
            result.order, sig.direction, lot,
            rcfg.take_profit_points, rcfg.stop_loss_points,
        )
        # Push iPhone notification
        try:
            import MetaTrader5 as _mt5
            tick = _mt5.symbol_info_tick(symbol)
            price = (tick.ask if sig.direction == "BUY" else tick.bid) if tick else 0.0
            point = _mt5.symbol_info(symbol).point if _mt5.symbol_info(symbol) else 0.01
        except ImportError:
            price = 0.0
            point = 0.01
        tp_price = price + rcfg.take_profit_points * point if sig.direction == "BUY" else price - rcfg.take_profit_points * point
        sl_price = price - rcfg.stop_loss_points * point if sig.direction == "BUY" else price + rcfg.stop_loss_points * point
        notify_trade_opened(
            direction=sig.direction,
            entry_type=sig.entry_type,
            lot=lot,
            price=price,
            tp=tp_price,
            sl=sl_price,
        )
        # Update win/trade counters
        with_state = getattr(_api_state, "_state", None)
        if with_state is not None:
            _api_state._state["total_trades_today"] += 1
    else:
        logger.error("Order placement failed.")


def _print_headlines() -> None:
    headlines = news_feed.latest_headlines(n=5)
    for h in headlines:
        flag = "⚠ " if h.is_high_impact else "  "
        logger.info("%s[%s] %s — %s",
                    flag, h.published_at.strftime("%H:%M"), h.title, h.source)
