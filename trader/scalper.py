"""
High-frequency scalping backtest engine for XAU/USD.

Strategy: Enter on M5 bars with H1 trend confirmation; exit at
fixed TP (+120 pts) or SL (-60 pts). Target 40+ trades per day.
Lot size: 0.50.

XAU/USD constants (matches rest of codebase):
  POINT = 0.01   (1 point = $0.01 price move)
  PPL   = 1.00   ($1.00 P&L per lot per point)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import ema, rsi, macd, candle_body_ratio

POINT = 0.01
PPL   = 100 * POINT   # $1.00 P&L per lot per point


@dataclass
class ScalpParams:
    tp_points: int       = 120   # take profit in points
    sl_points: int       = 60    # stop loss in points
    max_trades_day: int  = 60    # hard daily cap
    session_open: int    = 7     # UTC hour (inclusive)
    session_close: int   = 21    # UTC hour (exclusive)
    min_body_ratio: float = 0.25 # lower threshold suits M5 candles
    use_h1_trend: bool   = True  # require H1 EMA alignment
    spread_points: int   = 4     # spread cost deducted per trade (points)


def _build_h1_trend(m5: pd.DataFrame, h1: pd.DataFrame) -> np.ndarray:
    """
    Compute H1 trend direction (+1 BUY / -1 SELL / 0 neutral) using
    EMA20 > EMA50 > EMA100, then forward-fill to the M5 index.

    We use EMA100 instead of EMA200 so the indicator warms up faster
    on a few months of data.
    """
    h1_e20  = ema(h1["close"], 20)
    h1_e50  = ema(h1["close"], 50)
    h1_e100 = ema(h1["close"], 100)

    bull = (h1_e20 > h1_e50) & (h1_e50 > h1_e100)
    bear = (h1_e20 < h1_e50) & (h1_e50 < h1_e100)

    h1_dir = pd.Series(0, index=h1.index, dtype=np.int8)
    h1_dir[bull] = 1
    h1_dir[bear] = -1

    # Forward-fill H1 values onto M5 timestamps — no look-ahead bias
    m5_dir = h1_dir.reindex(m5.index, method="ffill").fillna(0).astype(np.int8)
    return m5_dir.values


def run_scalper(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    params: ScalpParams | None = None,
    lot: float = 0.50,
    start_balance: float = 10_000.0,
) -> dict:
    """
    Walk-forward scalping backtest.

    Parameters
    ----------
    m5 : M5 OHLCV DataFrame with DatetimeIndex (UTC).
    h1 : H1 OHLCV DataFrame with DatetimeIndex (UTC).
    params : ScalpParams (uses defaults if None).
    lot : fixed lot size.
    start_balance : starting account equity.

    Returns
    -------
    dict with keys: n, wins, losses, win_rate, net_pnl, roi,
                    avg_win, avg_loss, profit_factor,
                    trades_per_day, final_balance.
    """
    if params is None:
        params = ScalpParams()

    n = len(m5)

    # ── Pre-compute indicators (vectorised) ───────────────────────────────────
    close  = m5["close"].values
    high   = m5["high"].values
    low    = m5["low"].values
    open_  = m5["open"].values

    rsi_arr  = rsi(m5["close"], 14).values
    _, _, macd_hist = macd(m5["close"])
    macd_arr = macd_hist.values
    e20_arr  = ema(m5["close"], 20).values
    body_arr = candle_body_ratio(m5).values

    # H1 trend mapped to M5 bars
    if params.use_h1_trend:
        trend_arr = _build_h1_trend(m5, h1)
    else:
        # No trend filter: allow both directions based purely on M5 signals
        trend_arr = np.zeros(n, dtype=np.int8)

    # Session mask: in-hours bars
    hours = m5.index.hour
    in_session = ((hours >= params.session_open) & (hours < params.session_close))

    # TP / SL distances in price units
    tp_price = params.tp_points * POINT
    sl_price = params.sl_points * POINT
    # Spread cost in dollars per trade
    spread_cost = params.spread_points * lot * PPL

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    balance  = start_balance
    trades: list[dict] = []

    in_trade     = False
    t_dir        = 0        # +1 BUY, -1 SELL
    t_entry      = 0.0
    t_tp         = 0.0
    t_sl_price   = 0.0
    t_entry_date = None

    daily_counts: dict = {}  # date -> int

    # Warmup: skip first ~200 bars for indicator convergence
    warmup = 200

    for i in range(warmup, n):
        c = close[i]
        h = high[i]
        l = low[i]
        bar_date = m5.index[i].date()

        # ── Update open position ──────────────────────────────────────────────
        if in_trade:
            # Check TP / SL within this bar's range
            if t_dir == 1:
                tp_hit = h >= t_tp
                sl_hit = l <= t_sl_price
            else:
                tp_hit = l <= t_tp
                sl_hit = h >= t_sl_price

            # Force close at end-of-session (use close price)
            eod_close = (
                not in_session[i]
                and m5.index[i].hour >= params.session_close
                and t_entry_date == bar_date
            )

            if tp_hit:
                pnl_pts = params.tp_points
                pnl = pnl_pts * lot * PPL - spread_cost
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": "TP"})
                in_trade = False

            elif sl_hit:
                pnl_pts = -params.sl_points
                pnl = pnl_pts * lot * PPL - spread_cost
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": "SL"})
                in_trade = False

            elif eod_close:
                pnl_pts = (c - t_entry) / POINT if t_dir == 1 else (t_entry - c) / POINT
                pnl = pnl_pts * lot * PPL - spread_cost
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": "EOD"})
                in_trade = False

            if in_trade:
                continue  # still in the same trade

        # ── Check entry conditions ────────────────────────────────────────────
        if not in_session[i]:
            continue

        # Daily cap
        daily_count = daily_counts.get(bar_date, 0)
        if daily_count >= params.max_trades_day:
            continue

        # H1 trend direction
        d = trend_arr[i]
        if params.use_h1_trend and d == 0:
            continue

        # Indicators
        rsi_v  = rsi_arr[i]
        mh     = macd_arr[i]
        e20    = e20_arr[i]
        br     = body_arr[i]

        if np.isnan(rsi_v) or np.isnan(mh) or np.isnan(e20) or np.isnan(br):
            continue

        # Candle body quality
        if br < params.min_body_ratio:
            continue

        # Determine signal direction when use_h1_trend is False
        if not params.use_h1_trend:
            # Derive direction purely from M5 momentum
            if mh > 0 and c > e20:
                d = 1
            elif mh < 0 and c < e20:
                d = -1
            else:
                continue

        if d == 1:
            # BUY conditions
            rsi_ok   = 45 <= rsi_v <= 70
            macd_ok  = mh > 0
            ema_ok   = c > e20
        else:
            # SELL conditions
            rsi_ok   = 30 <= rsi_v <= 55
            macd_ok  = mh < 0
            ema_ok   = c < e20

        if not (rsi_ok and macd_ok and ema_ok):
            continue

        # ── Open trade at next bar open ───────────────────────────────────────
        if i + 1 >= n:
            break

        entry = open_[i + 1]
        if entry <= 0:
            continue

        if d == 1:
            t_tp       = entry + tp_price
            t_sl_price = entry - sl_price
        else:
            t_tp       = entry - tp_price
            t_sl_price = entry + sl_price

        in_trade     = True
        t_dir        = d
        t_entry      = entry
        t_entry_date = m5.index[i + 1].date()

        daily_counts[bar_date] = daily_count + 1

    # Close any leftover position at last close
    if in_trade:
        cp = close[-1]
        pnl_pts = (cp - t_entry) / POINT if t_dir == 1 else (t_entry - cp) / POINT
        pnl = pnl_pts * lot * PPL - spread_cost
        balance += pnl
        trades.append({"pnl": round(pnl, 2), "result": "END"})

    # ── Stats ─────────────────────────────────────────────────────────────────
    if not trades:
        return {
            "n": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "net_pnl": 0.0, "roi": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "trades_per_day": 0.0,
            "final_balance": round(start_balance, 2),
        }

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    net = balance - start_balance

    # Trading days: unique dates that had at least one trade
    trade_dates = set(daily_counts.keys())
    n_days = max(len(trade_dates), 1)

    return {
        "n":             len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins) / max(len(trades), 1) * 100, 1),
        "net_pnl":       round(net, 2),
        "roi":           round(net / start_balance * 100, 2),
        "avg_win":       round(gw / max(len(wins), 1), 2),
        "avg_loss":      round(-gl / max(len(losses), 1), 2),
        "profit_factor": round(gw / max(gl, 0.01), 2),
        "trades_per_day": round(len(trades) / n_days, 1),
        "final_balance": round(balance, 2),
    }
