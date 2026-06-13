"""
Main bot loop for the XAU/USD algorithm trader.

Cycle (every 30 seconds on M15 data):
  1. Daily manager check — stop if profit target or loss limit hit for today
  2. Refresh news — halt + notify if high-impact event is live
  3. Pause guard (set via iPhone dashboard)
  4. Skip if max open trades reached
  5. Pull OHLCV bars for M15 / H1 / H4
  6. Run strategy_v2.evaluate()
  7. If signal: size position dynamically → place order → push notification
"""

import logging
import signal
import sys
import time
from datetime import datetime

from . import mt5_connector as mt5c
from .config import strategy_cfg as scfg, risk_cfg as rcfg
from .daily_manager import DailyManager, DailyManagerConfig
from .position_sizing import calculate_lot, sl_to_points
from .news_feed import news_feed
from .notifications import (
    notify_trade_opened,
    notify_high_impact_news,
    notify_daily_limit_hit,
    notify_status,
)
from .strategy_v2 import StrategyParams, evaluate

logger = logging.getLogger(__name__)

_api_state = None
POLL_INTERVAL_SEC = 30

# Daily manager — $500 profit target, $150 loss limit
_daily = DailyManager(DailyManagerConfig(
    daily_profit_target=500.0,
    daily_loss_limit=150.0,
    max_trades_per_day=3,
))

# Strategy v3 parameters (multi-seed optimized)
_params = StrategyParams(
    adx_min=22.0,
    st_mult=3.0,
    breakout_atr_mult=0.8,
    min_momentum_score=1.0,
    min_body_ratio=0.38,
    atr_sl_mult=2.5,
    atr_tp_mult=5.5,
    atr_be_mult=1.5,
    signal_cooldown_bars=20,
    session_open=8,
    session_close=18,
)


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
    if _api_state is not None:
        _api_state.update_state(**kwargs)


def run(log_level: str = "INFO", with_api: bool = False,
        api_host: str = "0.0.0.0", api_port: int = 8080) -> None:
    global _api_state

    _setup_logging(log_level)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("=== XAU/USD Trader Bot v3 starting ===")
    logger.info("Daily target: +$%.0f | Loss limit: -$%.0f | Max trades/day: %d",
                _daily.cfg.daily_profit_target,
                _daily.cfg.daily_loss_limit,
                _daily.cfg.max_trades_per_day)
    logger.info("Strategy: ADX≥%.0f | ST=%.1f | SL=%.1f×ATR | TP=%.1f×ATR | session %d–%d UTC",
                _params.adx_min, _params.st_mult,
                _params.atr_sl_mult, _params.atr_tp_mult,
                _params.session_open, _params.session_close)

    if with_api:
        try:
            from . import api_server
            _api_state = api_server
            api_server.start_api_server(host=api_host, port=api_port)
        except ImportError:
            logger.warning("Flask not installed — iPhone dashboard disabled.")

    if not mt5c.connect():
        logger.critical("Cannot connect to MetaTrader 5 — exiting.")
        sys.exit(1)

    notify_status(
        f"Bot v3 started — {scfg.symbol} | "
        f"Target +${_daily.cfg.daily_profit_target:.0f}/day | "
        f"Limit -${_daily.cfg.daily_loss_limit:.0f}/day"
    )

    consecutive_errors = 0
    while True:
        try:
            _tick(scfg.symbol)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            logger.error("Unhandled error in tick (%d): %s", consecutive_errors, exc, exc_info=True)
            if consecutive_errors >= 10:
                logger.critical("Too many errors — shutting down.")
                notify_status("Bot crashed — check logs.")
                mt5c.disconnect()
                sys.exit(1)
        time.sleep(POLL_INTERVAL_SEC)


def _tick(symbol: str) -> None:
    now_str = datetime.utcnow().strftime("%H:%M:%S")

    # ── 1. Daily manager guard ────────────────────────────────────────────────
    if not _daily.should_trade():
        logger.debug("[%s] %s", now_str, _daily.progress)
        _api_update(daily_status=_daily.progress, daily_pnl=_daily.daily_pnl)
        return

    # ── 2. News guard ─────────────────────────────────────────────────────────
    news_feed.refresh_if_needed()
    high_impact = news_feed.high_impact_near_now(window_minutes=30)
    if high_impact:
        logger.info("HIGH-IMPACT NEWS — halting entry: '%s'", high_impact.title)
        notify_high_impact_news(high_impact.title)
        return

    # ── 3. Pause guard (iPhone dashboard) ─────────────────────────────────────
    if _api_state is not None and _api_state.is_paused():
        logger.debug("Paused via dashboard — skipping.")
        return

    # ── 4. Open trade count guard ─────────────────────────────────────────────
    open_trades = mt5c.get_open_bot_trades(symbol)
    equity = mt5c.get_account_equity()

    _api_update(
        open_trades=[
            {"type": "BUY" if p.type == 0 else "SELL",
             "lot": p.volume, "profit": p.profit, "ticket": p.ticket}
            for p in open_trades
        ],
        equity=equity,
        daily_pnl=_daily.daily_pnl,
        daily_status=_daily.progress,
    )

    if len(open_trades) >= rcfg.max_open_trades:
        logger.debug("Max open trades reached — skipping.")
        return

    # ── 5. Fetch OHLCV bars (M15 + H1 + H4) ──────────────────────────────────
    m15 = mt5c.get_ohlcv(symbol, "M15", count=400)
    h1  = mt5c.get_ohlcv(symbol, "H1",  count=300)
    h4  = mt5c.get_ohlcv(symbol, "H4",  count=200)

    if m15 is None or h1 is None or h4 is None:
        logger.warning("Could not fetch bars — skipping tick.")
        return

    # ── 6. Evaluate strategy v3 ───────────────────────────────────────────────
    sig = evaluate(m15, h1, h4, _params)
    _api_update(last_signal=sig)

    if sig is None:
        logger.debug("[%s] No signal.", now_str)
        return

    logger.info("*** SIGNAL *** %s %s | ATR=%.2f | SL=%dpt TP=%dpt | %s",
                sig.direction, sig.entry_type,
                sig.atr_value, sig.sl_pts, sig.tp_pts, sig.reason)

    # ── 7. Dynamic position sizing (ATR-based, 1% account risk) ──────────────
    sl_pts = sig.sl_pts
    lot = calculate_lot(
        equity=equity,
        risk_pct=1.0,           # risk 1% of account per trade
        sl_points=sl_pts,
        max_lot=rcfg.max_lot,   # hard cap at 0.50 lot
    )
    # Override with fixed lot if dynamic sizing gives less than fixed
    lot = max(lot, rcfg.fixed_lot)

    comment = f"xaubot-v3-{sig.entry_type[:2].lower()}"
    result = mt5c.place_order(
        symbol=symbol,
        direction=sig.direction,
        lot=lot,
        sl_points=sl_pts,
        tp_points=sig.tp_pts,
        comment=comment,
    )

    if result:
        logger.info("Trade opened — ticket=%s | %s %.2f lot | SL=%dpt TP=%dpt",
                    result.order, sig.direction, lot, sl_pts, sig.tp_pts)

        try:
            import MetaTrader5 as _mt5
            tick  = _mt5.symbol_info_tick(symbol)
            price = (tick.ask if sig.direction == "BUY" else tick.bid) if tick else 0.0
            pt    = _mt5.symbol_info(symbol).point if _mt5.symbol_info(symbol) else 0.01
        except ImportError:
            price = 0.0
            pt    = 0.01

        tp_price = (price + sig.tp_pts * pt) if sig.direction == "BUY" else (price - sig.tp_pts * pt)
        sl_price = (price - sl_pts * pt)       if sig.direction == "BUY" else (price + sl_pts * pt)

        notify_trade_opened(
            direction=sig.direction, entry_type=sig.entry_type,
            lot=lot, price=price, tp=tp_price, sl=sl_price,
        )

        # Record in daily manager (actual P&L recorded when trade closes)
        _api_update(total_trades_today=_daily.trades_today + 1)

    else:
        logger.error("Order placement failed.")


def record_closed_trade(pnl: float, direction: str, entry: float, result: str):
    """Call this from the MT5 connector when a trade closes."""
    _daily.record_trade(pnl=pnl, direction=direction, entry=entry, result=result)
    _api_update(daily_pnl=_daily.daily_pnl, daily_status=_daily.progress)
    if _daily.is_stopped:
        notify_status(f"Daily target reached — bot paused. P&L: ${_daily.daily_pnl:+.2f}")


def _print_headlines() -> None:
    for h in news_feed.latest_headlines(n=5):
        flag = "⚠ " if h.is_high_impact else "  "
        logger.info("%s[%s] %s — %s",
                    flag, h.published_at.strftime("%H:%M"), h.title, h.source)
