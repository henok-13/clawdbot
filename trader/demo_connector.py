"""
Paper-trading connector that mirrors the mt5_connector interface but
uses Yahoo Finance data instead of a live MT5 terminal.

Ticker used: GC=F  (COMEX gold futures — closest liquid proxy for XAU/USD)
yfinance returns prices in USD/oz, same denomination as MT5 XAU/USD.

A virtual account starts with DEMO_BALANCE and tracks open + closed trades.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from .config import risk_cfg

logger = logging.getLogger(__name__)

DEMO_BALANCE: float = 10_000.0   # starting virtual balance
SYMBOL_YF: str = "GC=F"          # Yahoo Finance ticker for Gold Futures
SPREAD_POINTS: int = 20          # simulate a 20-point bid/ask spread

# ── Virtual position ─────────────────────────────────────────────────────────

@dataclass
class VirtualPosition:
    ticket: str
    symbol: str
    type: int          # 0 = BUY, 1 = SELL (mirrors MT5 convention)
    volume: float
    price_open: float
    sl: float
    tp: float
    magic: int
    comment: str
    time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    profit: float = 0.0


@dataclass
class VirtualOrderResult:
    order: str
    retcode: int = 10009  # TRADE_RETCODE_DONE


# ── Demo account state ────────────────────────────────────────────────────────

class DemoAccount:
    def __init__(self, balance: float = DEMO_BALANCE):
        self.balance = balance
        self.equity = balance
        self.open_positions: list[VirtualPosition] = []
        self.closed_trades: list[dict] = []
        self._daily_open_balance = balance
        self.login = 99999999
        self.server = "Vantage-Demo (paper)"
        self.currency = "USD"

    def _pnl_per_lot_per_point(self) -> float:
        # XAU/USD: 1 std lot = 100 oz, 1 point = $0.01 → $1.00 P&L per lot per point
        return 1.0

    def _current_price(self) -> float:
        """Fetch latest gold price from Yahoo Finance (fallback to synthetic price)."""
        try:
            tick = yf.Ticker(SYMBOL_YF)
            hist = tick.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        # Fallback: return last known price from most recently fetched data
        return self._last_known_price if hasattr(self, "_last_known_price") and self._last_known_price else 2340.0

    def update_equity(self, current_price: float) -> None:
        """Mark-to-market all open positions."""
        floating = 0.0
        for pos in self.open_positions:
            point = 0.01
            if pos.type == 0:  # BUY
                pnl_points = (current_price - pos.price_open) / point
            else:              # SELL
                pnl_points = (pos.price_open - current_price) / point
            pnl = pnl_points * pos.volume * self._pnl_per_lot_per_point()
            pos.profit = round(pnl, 2)
            floating += pnl

            # Check TP/SL hit
            if pos.type == 0:
                if current_price >= pos.tp:
                    self._close_position(pos, pos.tp, "TP")
                elif current_price <= pos.sl:
                    self._close_position(pos, pos.sl, "SL")
            else:
                if current_price <= pos.tp:
                    self._close_position(pos, pos.tp, "TP")
                elif current_price >= pos.sl:
                    self._close_position(pos, pos.sl, "SL")

        self.equity = self.balance + floating

    def _close_position(self, pos: VirtualPosition, close_price: float, reason: str) -> None:
        if pos not in self.open_positions:
            return
        point = 0.01
        if pos.type == 0:
            pnl_points = (close_price - pos.price_open) / point
        else:
            pnl_points = (pos.price_open - close_price) / point
        pnl = pnl_points * pos.volume * self._pnl_per_lot_per_point()
        self.balance += pnl
        self.equity = self.balance
        self.open_positions.remove(pos)
        self.closed_trades.append({
            "ticket": pos.ticket,
            "direction": "BUY" if pos.type == 0 else "SELL",
            "lot": pos.volume,
            "open_price": pos.price_open,
            "close_price": close_price,
            "pnl": round(pnl, 2),
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            "Position CLOSED [%s] — %s lot=%.2f entry=%.2f exit=%.2f P&L=$%.2f",
            reason, "BUY" if pos.type == 0 else "SELL",
            pos.volume, pos.price_open, close_price, pnl,
        )

    def place_order(
        self,
        direction: str,
        lot: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> VirtualOrderResult:
        spread_price = price + SPREAD_POINTS * 0.01 if direction == "BUY" else price
        ticket = str(uuid.uuid4())[:8]
        pos = VirtualPosition(
            ticket=ticket,
            symbol="XAUUSD",
            type=0 if direction == "BUY" else 1,
            volume=lot,
            price_open=round(spread_price, 2),
            sl=round(sl, 2),
            tp=round(tp, 2),
            magic=risk_cfg.magic_number,
            comment=comment,
        )
        self.open_positions.append(pos)
        logger.info(
            "Paper trade — %s lot=%.2f @ %.2f  SL=%.2f  TP=%.2f  [%s]",
            direction, lot, spread_price, sl, tp, ticket,
        )
        return VirtualOrderResult(order=ticket)

    def daily_drawdown_pct(self) -> float:
        if self._daily_open_balance == 0:
            return 0.0
        return max(0.0, (self._daily_open_balance - self.equity) / self._daily_open_balance * 100.0)


# Module-level singleton
_account = DemoAccount()


# ── Public API (mirrors mt5_connector) ───────────────────────────────────────

def connect() -> bool:
    logger.info(
        "Demo account ready — login=%s | server=%s | balance=%.2f %s",
        _account.login, _account.server, _account.balance, _account.currency,
    )
    return True


def disconnect() -> None:
    logger.info("Demo connector disconnected.")


def get_ohlcv(symbol: str, timeframe: str, count: int = 300) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Yahoo Finance; fall back to synthetic data if unavailable."""
    tf_map = {
        "M1":  "1m",  "M5":  "5m",  "M15": "15m",
        "M30": "30m", "H1":  "1h",  "H4":  "4h",  "D1":  "1d",
    }
    period_map = {
        "1m": "7d", "5m": "60d", "15m": "60d",
        "30m": "60d", "1h": "730d", "4h": "730d", "1d": "5y",
    }
    interval = tf_map.get(timeframe, "15m")
    period = period_map.get(interval, "60d")

    try:
        df = yf.download(
            SYMBOL_YF, period=period, interval=interval,
            progress=False, auto_adjust=True,
        )
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            # cache last price
            _account._last_known_price = float(df["close"].iloc[-1])
            return df.iloc[-count:]
    except Exception:
        pass

    # Fallback: synthetic data (for demo/backtest in restricted environments)
    logger.info("Yahoo Finance unavailable — using synthetic XAU/USD data for %s", timeframe)
    from .synthetic_data import generate_xauusd, resample_to_h1
    tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
    minutes = tf_minutes.get(timeframe, 15)
    n = max(count * 2, 2000)
    m15 = generate_xauusd(n_bars=n, timeframe_minutes=15)
    if timeframe == "H1":
        df = resample_to_h1(m15)
    else:
        df = m15
    _account._last_known_price = float(df["close"].iloc[-1])
    return df.iloc[-count:]


def get_tick(symbol: str):
    """Return a simple namespace mimicking MT5 Tick with bid/ask."""
    price = _account._current_price()
    half_spread = SPREAD_POINTS * 0.01 / 2

    class _Tick:
        bid = round(price - half_spread, 2)
        ask = round(price + half_spread, 2)

    return _Tick()


def get_account_equity() -> float:
    price = _account._current_price()
    if price:
        _account.update_equity(price)
    return _account.equity


def get_open_bot_trades(symbol: str) -> list:
    return list(_account.open_positions)


def calc_lot_size(stop_loss_points: int, equity: float) -> float:
    risk_dollars = equity * (risk_cfg.max_risk_pct / 100.0)
    pnl_per_lot_per_point = 1.0
    raw_lot = risk_dollars / (stop_loss_points * pnl_per_lot_per_point)
    lot = max(risk_cfg.min_lot, min(risk_cfg.max_lot, round(raw_lot, 2)))
    return lot


def place_order(
    symbol: str,
    direction: str,
    lot: float,
    sl_points: int,
    tp_points: int,
    comment: str = "demo",
):
    tick = get_tick(symbol)
    point = 0.01
    if direction == "BUY":
        price = tick.ask
        sl = price - sl_points * point
        tp = price + tp_points * point
    else:
        price = tick.bid
        sl = price + sl_points * point
        tp = price - tp_points * point

    return _account.place_order(direction, lot, price, sl, tp, comment)


def daily_loss_exceeded(symbol: str) -> bool:
    return _account.daily_drawdown_pct() >= risk_cfg.daily_loss_limit_pct


def get_demo_account() -> DemoAccount:
    return _account
