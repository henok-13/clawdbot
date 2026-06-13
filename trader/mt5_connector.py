"""MetaTrader 5 connection, data fetching, and order execution for XAU/USD."""

import logging
from datetime import datetime
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore[assignment]  # not available on Linux/macOS
import pandas as pd

from .config import mt5_cfg, strategy_cfg, risk_cfg

logger = logging.getLogger(__name__)

# Map string timeframe names to MT5 constants
_TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def connect() -> bool:
    """Initialise MT5 and log in to the Vantage account."""
    kwargs: dict = {}
    if mt5_cfg.path:
        kwargs["path"] = mt5_cfg.path

    if not mt5.initialize(**kwargs):
        logger.error("MT5 initialize failed: %s", mt5.last_error())
        return False

    if not mt5.login(mt5_cfg.login, password=mt5_cfg.password, server=mt5_cfg.server):
        logger.error("MT5 login failed: %s", mt5.last_error())
        mt5.shutdown()
        return False

    info = mt5.account_info()
    logger.info(
        "Connected — account %s | server %s | balance %.2f %s",
        info.login, info.server, info.balance, info.currency,
    )
    return True


def disconnect() -> None:
    mt5.shutdown()
    logger.info("MT5 disconnected.")


def get_ohlcv(symbol: str, timeframe: str, count: int = 500) -> Optional[pd.DataFrame]:
    """Return the last `count` closed bars as a DataFrame."""
    tf = _TF_MAP.get(timeframe)
    if tf is None:
        logger.error("Unknown timeframe: %s", timeframe)
        return None

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        logger.error("copy_rates_from_pos failed: %s", mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def get_tick(symbol: str) -> Optional[mt5.Tick]:
    """Return the latest tick for `symbol`."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("symbol_info_tick failed: %s", mt5.last_error())
    return tick


def get_account_equity() -> float:
    info = mt5.account_info()
    return info.equity if info else 0.0


def get_open_bot_trades(symbol: str) -> list:
    """Return all open positions tagged with the bot's magic number."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return [p for p in positions if p.magic == risk_cfg.magic_number]


def calc_lot_size(stop_loss_points: int, equity: float) -> float:
    """
    Size the position so that the dollar risk per trade ≤ max_risk_pct of equity.

    For XAU/USD on MT5:
      1 standard lot = 100 oz
      1 point = $0.01 price movement
      P&L per lot per point = 100 * 0.01 = $1.00
    """
    risk_dollars = equity * (risk_cfg.max_risk_pct / 100.0)
    pnl_per_lot_per_point = 100.0 * 0.01  # $1.00 per lot per point
    raw_lot = risk_dollars / (stop_loss_points * pnl_per_lot_per_point)

    # Clamp to configured min/max and round to 2 decimals
    lot = max(risk_cfg.min_lot, min(risk_cfg.max_lot, round(raw_lot, 2)))
    return lot


def _point_to_price(symbol: str, points: int) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return points * 0.01
    return points * info.point


def place_order(
    symbol: str,
    direction: str,   # "BUY" or "SELL"
    lot: float,
    sl_points: int,
    tp_points: int,
    comment: str = "xauusd-bot",
) -> Optional[mt5.OrderSendResult]:
    """Send a market order with SL and TP expressed in points."""
    tick = get_tick(symbol)
    if tick is None:
        return None

    sl_price_delta = _point_to_price(symbol, sl_points)
    tp_price_delta = _point_to_price(symbol, tp_points)

    if direction == "BUY":
        price = tick.ask
        sl = price - sl_price_delta
        tp = price + tp_price_delta
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + sl_price_delta
        tp = price - tp_price_delta
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           round(sl, 2),
        "tp":           round(tp, 2),
        "deviation":    10,
        "magic":        risk_cfg.magic_number,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            "order_send failed — direction %s retcode %s",
            direction,
            result.retcode if result else "None",
        )
        return None

    logger.info(
        "Order placed — %s %s lot=%.2f price=%.2f SL=%.2f TP=%.2f ticket=%s",
        direction, symbol, lot, price, sl, tp, result.order,
    )
    return result


def daily_loss_exceeded(symbol: str) -> bool:
    """Return True if today's realised + floating loss breaches the daily limit."""
    equity = get_account_equity()
    info = mt5.account_info()
    if info is None:
        return False
    balance = info.balance
    drawdown_pct = (balance - equity) / balance * 100.0 if balance else 0.0
    return drawdown_pct >= risk_cfg.daily_loss_limit_pct
