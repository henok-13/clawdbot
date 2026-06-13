"""
Vectorised backtest engine — ALL ADVANTAGES edition.

Strategy v4 upgrades over v3:
  - THREE entry types: Breakout | Fibonacci Pullback | Momentum Continuation
  - Partial Take Profit: close 50% at TP1 (2×ATR), trail remainder to TP2 (5.5×ATR)
  - Tighter breakeven: SL moves to entry after partial TP hit
  - ATR volatility-regime filter (skip flat/explosive markets)
  - H1 RSI trend alignment
  - Extended session: 07:00–21:00 UTC (full London + NY + Asian crossover)
  - Compound lot scaling: lot grows automatically with account equity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import ema, rsi, macd, atr, adx, supertrend, candle_body_ratio

logger = logging.getLogger(__name__)

POINT = 0.01
PPL   = 100 * POINT   # $1.00 P&L per lot per point


# ── Trend mask ────────────────────────────────────────────────────────────────

def _precompute_trend_mask(
    m15, h1, h4,
    adx_min, st_mult,
    session_open, session_close,
    h1_rsi_bull_min=50.0, h1_rsi_bear_max=50.0,
    use_h4_macd=False,
):
    n = len(m15)

    h1_e20  = ema(h1["close"], 20)
    h1_e50  = ema(h1["close"], 50)
    h1_e200 = ema(h1["close"], 200)
    h1_adx  = adx(h1, 14)
    h1_rsi_v = rsi(h1["close"], 14)

    h1_bull = ((h1["close"] > h1_e20) & (h1_e20 > h1_e50) &
               (h1_e50 > h1_e200) & (h1_adx > adx_min) &
               (h1_rsi_v > h1_rsi_bull_min))
    h1_bear = ((h1["close"] < h1_e20) & (h1_e20 < h1_e50) &
               (h1_e50 < h1_e200) & (h1_adx > adx_min) &
               (h1_rsi_v < h1_rsi_bear_max))

    h4_e20 = ema(h4["close"], 20)
    h4_e50 = ema(h4["close"], 50)
    h4_st  = supertrend(h4, 10, st_mult)
    h4_bull = (h4["close"] > h4_e20) & (h4_e20 > h4_e50) & (h4_st == 1)
    h4_bear = (h4["close"] < h4_e20) & (h4_e20 < h4_e50) & (h4_st == -1)

    if use_h4_macd:
        _, _, h4_hist = macd(h4["close"])
        h4_bull = h4_bull & (h4_hist > 0)
        h4_bear = h4_bear & (h4_hist < 0)

    h1_bull_m15 = h1_bull.reindex(m15.index, method="ffill").fillna(False)
    h1_bear_m15 = h1_bear.reindex(m15.index, method="ffill").fillna(False)
    h4_bull_m15 = h4_bull.reindex(m15.index, method="ffill").fillna(False)
    h4_bear_m15 = h4_bear.reindex(m15.index, method="ffill").fillna(False)

    hours = m15.index.hour
    in_session = (hours >= session_open) & (hours < session_close)

    trend = np.zeros(n, dtype=np.int8)
    bull = h1_bull_m15.values & h4_bull_m15.values & in_session
    bear = h1_bear_m15.values & h4_bear_m15.values & in_session
    trend[bull] = 1
    trend[bear] = -1
    return trend


def _rolling_swing_high(highs, lookback):
    n = len(highs)
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        out[i] = np.max(highs[i - lookback: i])
    return out


def _rolling_swing_low(lows, lookback):
    n = len(lows)
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        out[i] = np.min(lows[i - lookback: i])
    return out


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    # Trend filters
    adx_min: float           = 22.0
    st_mult: float           = 3.0
    h1_rsi_bull_min: float   = 50.0
    h1_rsi_bear_max: float   = 50.0
    use_h4_macd: bool        = False

    # Entry types
    breakout_atr_mult: float  = 0.8
    min_momentum_score: float = 1.0
    min_body_ratio: float     = 0.38
    use_momentum_cont: bool   = True   # momentum continuation entries
    cont_lookback: int        = 12     # bars to look back for recent EMA touch
    cont_ema_tol: float       = 0.003  # price within 0.3% of EMA20 counts as touch

    # ATR quality filter
    atr_ratio_min: float     = 0.5
    atr_ratio_max: float     = 2.5

    # Exit — partial TP system
    atr_tp1_mult: float      = 2.0    # partial TP (50% of position)
    atr_tp2_mult: float      = 5.5    # full TP (remaining 50%)
    atr_sl_mult: float       = 2.5
    atr_be_mult: float       = 1.5    # move SL to entry at this profit
    trail_atr_mult: float    = 0.8    # trail after partial TP hit

    # Session & cooldown
    session_open: int        = 7      # extended: 07:00–21:00 UTC
    session_close: int       = 21
    cooldown_bars: int       = 16     # tighter cooldown = more signals

    # Swing levels
    swing_lookback: int      = 80
    swing_min_age: int       = 8
    fib_lo: float            = 0.382
    fib_hi: float            = 0.618
    fib_ema_tol: float       = 0.005


# ── Main backtest ─────────────────────────────────────────────────────────────

def run(
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    params: BacktestParams,
    lot: float = 0.30,
    warmup: int = 900,
    risk_pct: float = 0.0,     # if > 0, override lot with dynamic sizing
    start_balance: float = 10_000.0,
) -> dict:
    """
    Vectorised walk-forward backtest.
    If risk_pct > 0, lot size scales dynamically with account balance.
    """
    n = len(m15)

    close = m15["close"].values
    high  = m15["high"].values
    low   = m15["low"].values
    open_ = m15["open"].values

    atr_arr     = atr(m15, 14).values
    atr_avg_arr = pd.Series(atr_arr).rolling(50, min_periods=10).mean().values
    rsi_arr     = rsi(m15["close"], 14).values
    _, _, macd_hist = macd(m15["close"])
    macd_arr    = macd_hist.values
    e20_arr     = ema(m15["close"], 20).values
    body_arr    = candle_body_ratio(m15).values

    lb = params.swing_lookback + params.swing_min_age
    swing_h = _rolling_swing_high(high, lb)
    swing_l = _rolling_swing_low(low, lb)

    trend = _precompute_trend_mask(
        m15, h1, h4,
        adx_min=params.adx_min, st_mult=params.st_mult,
        session_open=params.session_open, session_close=params.session_close,
        h1_rsi_bull_min=params.h1_rsi_bull_min,
        h1_rsi_bear_max=params.h1_rsi_bear_max,
        use_h4_macd=params.use_h4_macd,
    )

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    balance    = start_balance
    start_bal  = start_balance
    trades: list[dict] = []

    in_trade     = False
    t_dir        = 0
    t_entry      = 0.0
    t_sl         = 0.0
    t_tp1        = 0.0   # first (partial) TP
    t_tp2        = 0.0   # full TP
    t_tp1_hit    = False
    t_be_moved   = False
    t_atr        = 0.0
    t_lot        = lot
    last_entry   = -9999

    for i in range(warmup, n):
        h = high[i]
        l = low[i]
        c = close[i]

        # ── Update open position ──────────────────────────────────────────────
        if in_trade:
            # Breakeven: move SL once profit ≥ be_trig
            if not t_be_moved:
                be_trig = t_entry + params.atr_be_mult * t_atr if t_dir == 1 else t_entry - params.atr_be_mult * t_atr
                if (t_dir == 1 and h >= be_trig) or (t_dir == -1 and l <= be_trig):
                    t_sl = t_entry + POINT * 5 if t_dir == 1 else t_entry - POINT * 5
                    t_be_moved = True

            # Partial TP: first exit at TP1 — book 50%, trail the rest
            if not t_tp1_hit:
                if (t_dir == 1 and h >= t_tp1) or (t_dir == -1 and l <= t_tp1):
                    cp1 = t_tp1
                    pnl_pts1 = (cp1 - t_entry) / POINT if t_dir == 1 else (t_entry - cp1) / POINT
                    pnl1 = pnl_pts1 * (t_lot * 0.5) * PPL
                    balance += pnl1
                    trades.append({"pnl": round(pnl1, 2), "result": "TP1"})
                    t_tp1_hit = True
                    # Move SL to entry (guaranteed no loss on remaining 50%)
                    t_sl = t_entry + POINT * 5 if t_dir == 1 else t_entry - POINT * 5
                    t_be_moved = True

            # Trailing stop after partial TP
            if t_tp1_hit and params.trail_atr_mult > 0:
                trail_dist = params.trail_atr_mult * t_atr
                if t_dir == 1:
                    new_sl = h - trail_dist
                    if new_sl > t_sl:
                        t_sl = new_sl
                else:
                    new_sl = l + trail_dist
                    if new_sl < t_sl:
                        t_sl = new_sl

            # Check full TP2 / SL for remaining position
            result = None
            cp = 0.0
            rem = 0.5 if t_tp1_hit else 1.0   # remaining position fraction
            if t_dir == 1:
                if h >= t_tp2:   result, cp = "TP2", t_tp2
                elif l <= t_sl:  result, cp = "SL" if not t_tp1_hit else "TP1+trail", t_sl
            else:
                if l <= t_tp2:   result, cp = "TP2", t_tp2
                elif h >= t_sl:  result, cp = "SL" if not t_tp1_hit else "TP1+trail", t_sl

            if result:
                pnl_pts = (cp - t_entry) / POINT if t_dir == 1 else (t_entry - cp) / POINT
                pnl = pnl_pts * (t_lot * rem) * PPL
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": result})
                in_trade = False

        if in_trade:
            continue
        if (i - last_entry) < params.cooldown_bars:
            continue
        if balance < start_bal * 0.80:
            break   # hard stop -20%

        d = trend[i]
        if d == 0:
            continue

        av = atr_arr[i]
        if np.isnan(av) or av <= 0:
            continue

        # ATR quality filter
        av_avg = atr_avg_arr[i]
        if not np.isnan(av_avg) and av_avg > 0:
            ratio = av / av_avg
            if ratio < params.atr_ratio_min or ratio > params.atr_ratio_max:
                continue

        # Candle body quality
        br = body_arr[i]
        if np.isnan(br) or br < params.min_body_ratio:
            continue

        # M15 momentum score
        rsi_v = rsi_arr[i]
        mh    = macd_arr[i]
        mh_p  = macd_arr[i - 1] if i > 0 else 0.0
        score = 0.0
        if d == 1:
            if 42 <= rsi_v <= 70:  score += 0.5
            if mh > 0 and mh > mh_p: score += 0.5
        else:
            if 30 <= rsi_v <= 58:  score += 0.5
            if mh < 0 and mh < mh_p: score += 0.5
        if score < params.min_momentum_score:
            continue

        sh   = swing_h[i]
        sl_v = swing_l[i]
        e20  = e20_arr[i]

        is_breakout    = False
        is_pullback    = False
        is_continuation = False

        if not (np.isnan(sh) or np.isnan(sl_v)):
            # 1. Breakout
            if d == 1 and c > sh + params.breakout_atr_mult * av:
                is_breakout = True
            elif d == -1 and c < sl_v - params.breakout_atr_mult * av:
                is_breakout = True

            # 2. Fibonacci pullback
            swing_range = sh - sl_v
            if swing_range > 0 and not np.isnan(e20):
                near_ema = abs(c - e20) / e20 < params.fib_ema_tol
                o = open_[i]
                if d == 1:
                    fib_hi_v = sh - params.fib_lo * swing_range
                    fib_lo_v = sh - params.fib_hi * swing_range
                    if fib_lo_v <= c <= fib_hi_v and near_ema and c > o:
                        is_pullback = True
                else:
                    fib_lo_v = sl_v + params.fib_lo * swing_range
                    fib_hi_v = sl_v + params.fib_hi * swing_range
                    if fib_lo_v <= c <= fib_hi_v and near_ema and c < o:
                        is_pullback = True

        # 3. Momentum continuation (new entry type)
        if params.use_momentum_cont and not np.isnan(e20) and not is_breakout and not is_pullback:
            lb_cont = min(i, params.cont_lookback)
            if lb_cont > 0:
                if d == 1:
                    # Price recently touched EMA20 from above (pullback), now bouncing
                    recent_lows = low[i - lb_cont: i]
                    ema_touched = np.any(recent_lows <= e20 * (1 + params.cont_ema_tol))
                    price_above_ema = c > e20
                    rsi_bullish = 48 <= rsi_v <= 72
                    if ema_touched and price_above_ema and rsi_bullish and mh > 0:
                        is_continuation = True
                else:
                    recent_highs = high[i - lb_cont: i]
                    ema_touched = np.any(recent_highs >= e20 * (1 - params.cont_ema_tol))
                    price_below_ema = c < e20
                    rsi_bearish = 28 <= rsi_v <= 52
                    if ema_touched and price_below_ema and rsi_bearish and mh < 0:
                        is_continuation = True

        if not (is_breakout or is_pullback or is_continuation):
            continue

        # ── Open trade ────────────────────────────────────────────────────────
        if i + 1 >= n:
            break
        entry = open_[i + 1]

        # Dynamic lot sizing: risk_pct% of current balance
        if risk_pct > 0:
            sl_dist_price = params.atr_sl_mult * av
            sl_pts = sl_dist_price / POINT
            t_lot = round(max(0.01, (balance * risk_pct / 100) / (sl_pts * PPL)), 2)
        else:
            t_lot = lot

        sl_dist = params.atr_sl_mult * av
        tp1_dist = params.atr_tp1_mult * av
        tp2_dist = params.atr_tp2_mult * av

        if d == 1:
            t_sl  = entry - sl_dist
            t_tp1 = entry + tp1_dist
            t_tp2 = entry + tp2_dist
        else:
            t_sl  = entry + sl_dist
            t_tp1 = entry - tp1_dist
            t_tp2 = entry - tp2_dist

        in_trade     = True
        t_dir        = d
        t_entry      = entry
        t_be_moved   = False
        t_tp1_hit    = False
        t_atr        = av
        last_entry   = i

    # Close remaining open position
    if in_trade:
        cp = close[-1]
        rem = 0.5 if t_tp1_hit else 1.0
        pnl_pts = (cp - t_entry) / POINT if t_dir == 1 else (t_entry - cp) / POINT
        pnl = pnl_pts * (t_lot * rem) * PPL
        balance += pnl
        trades.append({"pnl": round(pnl, 2), "result": "END"})

    # ── Stats ─────────────────────────────────────────────────────────────────
    if not trades:
        return {"n": 0, "wins": 0, "win_rate": 0.0,
                "net_pnl": 0.0, "roi": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0}

    # Count signals (not partial exits) as trades
    signal_trades = [t for t in trades if t["result"] in ("SL", "TP2", "END", "TP1+trail")]
    partial_wins  = [t for t in trades if t["result"] == "TP1"]
    all_closed    = trades

    wins   = [t for t in all_closed if t["pnl"] > 0]
    losses = [t for t in all_closed if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    net = balance - start_bal

    return {
        "n":              len(signal_trades) + len(partial_wins),
        "signals":        len(signal_trades),
        "partial_wins":   len(partial_wins),
        "wins":           len(wins),
        "win_rate":       len(wins) / max(len(all_closed), 1) * 100,
        "net_pnl":        round(net, 2),
        "roi":            round(net / start_bal * 100, 2),
        "avg_win":        round(gw / max(len(wins), 1), 2),
        "avg_loss":       round(-gl / max(len(losses), 1), 2),
        "profit_factor":  round(gw / max(gl, 0.01), 2),
        "final_balance":  round(balance, 2),
    }
