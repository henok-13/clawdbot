"""
High-frequency scalping backtest engine for XAU/USD.

Strategy: Enter on M5 bars with full multi-timeframe confirmation; exit at
fixed TP (+200 pts) or SL (-135 pts). Target 15+ trades per day.
Lot size: 0.60.

Multi-timeframe confirmation stack (4 layers):
  H4  : Supertrend direction must agree
  H1  : EMA20>EMA50>EMA100 + RSI zone (>52 bull / <48 bear)
  M5  : ADX ≥ 22, RSI zone, MACD sign, body ≥ 0.38, price vs EMA20
  Risk: Breakeven stop at +50 pts (SL slides to entry)

XAU/USD constants:
  POINT = 0.01  |  PPL = $1.00 per lot per point
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import ema, rsi, macd, adx, supertrend, candle_body_ratio, volume_ratio

POINT = 0.01
PPL   = 100 * POINT


@dataclass
class ScalpParams:
    tp_points: int        = 200
    sl_points: int        = 135
    max_trades_day: int   = 60
    session_open: int     = 8    # London open (UTC)
    session_close: int    = 18   # NY mid-session (UTC) — peak trending hours
    min_body_ratio: float = 0.38   # strong candle required
    use_h1_trend: bool    = True   # H1 EMA alignment gate
    use_h1_rsi: bool      = True   # H1 RSI confirmation gate
    h1_rsi_bull_min: float = 56.0  # H1 RSI must be above this for buys
    h1_rsi_bear_max: float = 44.0  # H1 RSI must be below this for sells
    use_h4_trend: bool    = True   # H4 Supertrend gate
    spread_points: int    = 4
    adx_min: float        = 24.0   # M5 ADX — only trade strong trends
    rsi_buy_min: float    = 52.0   # tighter M5 RSI zones
    rsi_buy_max: float    = 68.0
    rsi_sell_min: float   = 32.0
    rsi_sell_max: float   = 48.0
    be_trigger_points: int = 50    # slide SL to entry once +50 pts in profit
    min_vol_ratio: float  = 0.90   # volume must be ≥ 90% of 20-bar average


def _build_h1_trend(m5: pd.DataFrame, h1: pd.DataFrame) -> np.ndarray:
    """H1 EMA alignment: +1 bull / -1 bear / 0 neutral, forward-filled to M5."""
    e20  = ema(h1["close"], 20)
    e50  = ema(h1["close"], 50)
    e100 = ema(h1["close"], 100)

    h1_dir = pd.Series(0, index=h1.index, dtype=np.int8)
    h1_dir[(e20 > e50) & (e50 > e100)] = 1
    h1_dir[(e20 < e50) & (e50 < e100)] = -1

    return h1_dir.reindex(m5.index, method="ffill").fillna(0).astype(np.int8).values


def _build_h1_rsi(m5: pd.DataFrame, h1: pd.DataFrame) -> np.ndarray:
    """H1 RSI (14) forward-filled to M5 index."""
    h1_rsi = rsi(h1["close"], 14)
    return h1_rsi.reindex(m5.index, method="ffill").fillna(50.0).values


def _build_h4_trend(m5: pd.DataFrame, h4: pd.DataFrame) -> np.ndarray:
    """H4 Supertrend direction (+1 bull / -1 bear) forward-filled to M5."""
    st = supertrend(h4, period=10, multiplier=3.0)
    return st.reindex(m5.index, method="ffill").fillna(1).astype(np.int8).values


def run_scalper(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame | None = None,
    params: ScalpParams | None = None,
    lot: float = 0.60,
    start_balance: float = 10_000.0,
) -> dict:
    """
    Walk-forward scalping backtest.

    Parameters
    ----------
    m5     : M5 OHLCV DataFrame with DatetimeIndex (UTC).
    h1     : H1 OHLCV DataFrame with DatetimeIndex (UTC).
    h4     : H4 OHLCV DataFrame (optional — built from m5 if None).
    params : ScalpParams (uses defaults if None).
    lot    : fixed lot size.
    start_balance : starting account equity.

    Returns
    -------
    dict with keys: n, wins, losses, breakevens, win_rate, net_pnl, roi,
                    avg_win, avg_loss, profit_factor, trades_per_day,
                    n_days, final_balance.
    """
    if params is None:
        params = ScalpParams()

    # Build H4 from M5 if not supplied
    if h4 is None:
        h4 = m5.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

    n = len(m5)

    # ── Pre-compute all indicators (vectorised) ───────────────────────────────
    close  = m5["close"].values
    high   = m5["high"].values
    low    = m5["low"].values
    open_  = m5["open"].values

    rsi_arr  = rsi(m5["close"], 14).values
    _, _, macd_hist = macd(m5["close"])
    macd_arr = macd_hist.values
    e20_arr  = ema(m5["close"], 20).values
    body_arr = candle_body_ratio(m5).values
    adx_arr  = adx(m5, 14).values
    vol_arr  = volume_ratio(m5, 20).values

    # H1 layers
    h1_trend_arr = _build_h1_trend(m5, h1) if params.use_h1_trend else np.ones(n, dtype=np.int8)
    h1_rsi_arr   = _build_h1_rsi(m5, h1)   if params.use_h1_rsi   else np.full(n, 50.0)

    # H4 layer
    h4_trend_arr = _build_h4_trend(m5, h4) if params.use_h4_trend else np.ones(n, dtype=np.int8)

    hours = m5.index.hour
    in_session = (hours >= params.session_open) & (hours < params.session_close)

    tp_price    = params.tp_points * POINT
    sl_price    = params.sl_points * POINT
    be_price    = params.be_trigger_points * POINT
    spread_cost = params.spread_points * lot * PPL

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    balance  = start_balance
    trades: list[dict] = []

    in_trade     = False
    t_dir        = 0
    t_entry      = 0.0
    t_tp         = 0.0
    t_sl_price   = 0.0
    t_be_set     = False
    t_entry_date = None

    daily_counts: dict = {}
    warmup = 300   # slightly longer for H1/H4 indicator warmup

    for i in range(warmup, n):
        c = close[i]
        h = high[i]
        l = low[i]
        bar_date = m5.index[i].date()

        # ── Update open position ──────────────────────────────────────────────
        if in_trade:
            # Slide SL to entry (breakeven) once profit target reached
            if not t_be_set:
                if t_dir == 1 and h >= t_entry + be_price:
                    t_sl_price = t_entry
                    t_be_set   = True
                elif t_dir == -1 and l <= t_entry - be_price:
                    t_sl_price = t_entry
                    t_be_set   = True

            if t_dir == 1:
                tp_hit = h >= t_tp
                sl_hit = l <= t_sl_price
            else:
                tp_hit = l <= t_tp
                sl_hit = h >= t_sl_price

            eod_close = (
                not in_session[i]
                and m5.index[i].hour >= params.session_close
                and t_entry_date == bar_date
            )

            if tp_hit:
                pnl = params.tp_points * lot * PPL - spread_cost
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": "TP"})
                in_trade = False

            elif sl_hit:
                if t_be_set:
                    pnl    = -spread_cost
                    result = "BE"
                else:
                    pnl    = -params.sl_points * lot * PPL - spread_cost
                    result = "SL"
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": result})
                in_trade = False

            elif eod_close:
                pnl_pts = (c - t_entry) / POINT if t_dir == 1 else (t_entry - c) / POINT
                pnl = pnl_pts * lot * PPL - spread_cost
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": "EOD"})
                in_trade = False

            if in_trade:
                continue

        # ── Entry conditions ──────────────────────────────────────────────────
        if not in_session[i]:
            continue

        daily_count = daily_counts.get(bar_date, 0)
        if daily_count >= params.max_trades_day:
            continue

        # Layer 1 — H1 EMA trend direction
        h1_d = h1_trend_arr[i]
        if params.use_h1_trend and h1_d == 0:
            continue

        # Layer 2 — H4 Supertrend must agree with H1 trend
        h4_d = h4_trend_arr[i]
        if params.use_h4_trend and h4_d != h1_d:
            continue

        # M5 indicators
        rsi_v = rsi_arr[i]
        mh    = macd_arr[i]
        e20   = e20_arr[i]
        br    = body_arr[i]
        adx_v = adx_arr[i]
        h1_r  = h1_rsi_arr[i]
        vol_v = vol_arr[i]

        if np.isnan(rsi_v) or np.isnan(mh) or np.isnan(e20) or np.isnan(br) or np.isnan(adx_v):
            continue

        # Layer 3 — M5 ADX (trending market)
        if adx_v < params.adx_min:
            continue

        # Layer 3 — candle body quality
        if br < params.min_body_ratio:
            continue

        # Layer 3 — volume confirmation (above-average participation)
        if not np.isnan(vol_v) and vol_v < params.min_vol_ratio:
            continue

        # Layer 2 — H1 RSI confirmation
        if params.use_h1_rsi:
            if h1_d == 1 and h1_r < params.h1_rsi_bull_min:
                continue
            if h1_d == -1 and h1_r > params.h1_rsi_bear_max:
                continue

        d = h1_d  # trade direction follows H1 (and H4-confirmed) trend

        if d == 1:
            rsi_ok  = params.rsi_buy_min  <= rsi_v <= params.rsi_buy_max
            macd_ok = mh > 0
            ema_ok  = c > e20
        else:
            rsi_ok  = params.rsi_sell_min <= rsi_v <= params.rsi_sell_max
            macd_ok = mh < 0
            ema_ok  = c < e20

        if not (rsi_ok and macd_ok and ema_ok):
            continue

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
        t_be_set     = False
        t_entry_date = m5.index[i + 1].date()

        daily_counts[bar_date] = daily_count + 1

    # Close leftover at last bar
    if in_trade:
        cp      = close[-1]
        pnl_pts = (cp - t_entry) / POINT if t_dir == 1 else (t_entry - cp) / POINT
        pnl     = pnl_pts * lot * PPL - spread_cost
        balance += pnl
        trades.append({"pnl": round(pnl, 2), "result": "END"})

    # ── Stats ─────────────────────────────────────────────────────────────────
    if not trades:
        return {
            "n": 0, "wins": 0, "losses": 0, "breakevens": 0, "win_rate": 0.0,
            "net_pnl": 0.0, "roi": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "trades_per_day": 0.0, "n_days": 1,
            "final_balance": round(start_balance, 2),
        }

    wins       = [t for t in trades if t["result"] == "TP"]
    breakevens = [t for t in trades if t["result"] == "BE"]
    losses     = [t for t in trades if t["result"] in ("SL", "EOD", "END") and t["pnl"] < 0]

    gw  = sum(t["pnl"] for t in wins)
    gl  = abs(sum(t["pnl"] for t in losses))
    net = balance - start_balance

    trade_dates = set(daily_counts.keys())
    n_days = max(len(trade_dates), 1)

    decisive = len(wins) + len(losses)

    return {
        "n":              len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "breakevens":     len(breakevens),
        "win_rate":       round(len(wins) / max(decisive, 1) * 100, 1),
        "net_pnl":        round(net, 2),
        "roi":            round(net / start_balance * 100, 2),
        "avg_win":        round(gw / max(len(wins), 1), 2),
        "avg_loss":       round(gl / max(len(losses), 1), 2),   # positive magnitude
        "profit_factor":  round(gw / max(gl, 0.01), 2),
        "trades_per_day": round(len(trades) / n_days, 1),
        "n_days":         n_days,
        "final_balance":  round(balance, 2),
    }
