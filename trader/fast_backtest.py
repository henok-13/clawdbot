"""
Vectorised backtest engine — precomputes all indicators once,
then walks bars using numpy arrays for speed.

v3 enhancements over v2:
  - H1 RSI alignment filter (RSI must confirm trend direction)
  - H4 MACD histogram filter (macro momentum must agree)
  - ATR quality filter (skip flat/explosive market regimes)
  - Trailing stop after breakeven (locks in profits dynamically)
  - Tighter candle quality + momentum thresholds
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import ema, rsi, macd, atr, adx, supertrend, candle_body_ratio

logger = logging.getLogger(__name__)

POINT = 0.01          # 1 point = $0.01
PPL   = 100 * POINT   # $1.00 P&L per lot per point (XAU/USD 100 oz)


# ── Trend mask ────────────────────────────────────────────────────────────────

def _precompute_trend_mask(
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    adx_min: float,
    st_mult: float,
    session_open: int,
    session_close: int,
    h1_rsi_bull_min: float = 50.0,
    h1_rsi_bear_max: float = 50.0,
    use_h4_macd: bool = False,
) -> np.ndarray:
    """
    Returns int8 array aligned to m15 index:
      +1 = bullish confluence  -1 = bearish  0 = no signal
    """
    n = len(m15)

    # H1 trend: EMA stack + ADX + optional RSI
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

    # H4 trend: EMA20>EMA50 + Supertrend + optional MACD histogram
    h4_e20 = ema(h4["close"], 20)
    h4_e50 = ema(h4["close"], 50)
    h4_st  = supertrend(h4, 10, st_mult)
    h4_bull = (h4["close"] > h4_e20) & (h4_e20 > h4_e50) & (h4_st == 1)
    h4_bear = (h4["close"] < h4_e20) & (h4_e20 < h4_e50) & (h4_st == -1)

    if use_h4_macd:
        _, _, h4_hist = macd(h4["close"])
        h4_bull = h4_bull & (h4_hist > 0)
        h4_bear = h4_bear & (h4_hist < 0)

    # Map H1 and H4 onto M15 (forward-fill, no look-ahead)
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


def _rolling_swing_high(highs: np.ndarray, lookback: int) -> np.ndarray:
    n = len(highs)
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        out[i] = np.max(highs[i - lookback : i])
    return out


def _rolling_swing_low(lows: np.ndarray, lookback: int) -> np.ndarray:
    n = len(lows)
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        out[i] = np.min(lows[i - lookback : i])
    return out


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    # Trend filters
    adx_min: float           = 25.0
    st_mult: float           = 3.0
    h1_rsi_bull_min: float   = 52.0   # H1 RSI must be above this for buys
    h1_rsi_bear_max: float   = 48.0   # H1 RSI must be below this for sells
    use_h4_macd: bool        = True   # require H4 MACD hist to agree

    # Setup / entry
    breakout_atr_mult: float  = 0.8
    min_momentum_score: float = 1.0
    min_body_ratio: float     = 0.40

    # ATR volatility regime filter
    atr_ratio_min: float     = 0.6    # skip if ATR < 0.6× its 50-bar avg (too flat)
    atr_ratio_max: float     = 2.5    # skip if ATR > 2.5× avg (too explosive)

    # Risk / exit
    atr_sl_mult: float       = 2.0
    atr_tp_mult: float       = 5.5
    atr_be_mult: float       = 1.2    # move SL to entry when profit ≥ this × ATR
    trail_atr_mult: float    = 0.8    # after BE, trail SL this × ATR behind price (0=off)

    # Session & cooldown
    session_open: int        = 8      # tighter: 08:00–18:00 UTC peak liquidity
    session_close: int       = 18
    cooldown_bars: int       = 20

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
) -> dict:
    """
    Vectorised walk-forward backtest. Returns stats dict.
    All indicator arrays are computed once up front.
    """
    n = len(m15)

    # ── Pre-compute indicator arrays ─────────────────────────────────────────
    close = m15["close"].values
    high  = m15["high"].values
    low   = m15["low"].values
    open_ = m15["open"].values

    atr_arr   = atr(m15, 14).values
    # 50-bar rolling mean of ATR for volatility-regime filter
    atr_avg_arr = pd.Series(atr_arr).rolling(50, min_periods=10).mean().values

    rsi_arr   = rsi(m15["close"], 14).values
    _, _, macd_hist = macd(m15["close"])
    macd_arr  = macd_hist.values
    e20_arr   = ema(m15["close"], 20).values
    body_arr  = candle_body_ratio(m15).values

    lb = params.swing_lookback + params.swing_min_age
    swing_h = _rolling_swing_high(high, lb)
    swing_l = _rolling_swing_low(low, lb)

    trend = _precompute_trend_mask(
        m15, h1, h4,
        adx_min=params.adx_min,
        st_mult=params.st_mult,
        session_open=params.session_open,
        session_close=params.session_close,
        h1_rsi_bull_min=params.h1_rsi_bull_min,
        h1_rsi_bear_max=params.h1_rsi_bear_max,
        use_h4_macd=params.use_h4_macd,
    )

    # ── Walk-forward simulation ───────────────────────────────────────────────
    balance   = 10_000.0
    start_bal = balance
    trades: list[dict] = []

    in_trade   = False
    t_dir      = 0
    t_entry    = 0.0
    t_sl       = 0.0
    t_tp       = 0.0
    t_be_trig  = 0.0
    t_be_moved = False
    t_atr      = 0.0    # ATR at entry (used for trailing)
    last_entry = -9999

    for i in range(warmup, n):
        h = high[i]
        l = low[i]
        c = close[i]
        o = open_[i]

        # ── Update open position ──────────────────────────────────────────────
        if in_trade:
            # Breakeven: move SL to entry when profit ≥ be_trig
            if not t_be_moved:
                if t_dir == 1 and h >= t_be_trig:
                    t_sl = t_entry + POINT * 5
                    t_be_moved = True
                elif t_dir == -1 and l <= t_be_trig:
                    t_sl = t_entry - POINT * 5
                    t_be_moved = True

            # Trailing stop: after BE, trail SL by trail_atr_mult × ATR
            if t_be_moved and params.trail_atr_mult > 0:
                trail_dist = params.trail_atr_mult * t_atr
                if t_dir == 1:
                    new_sl = h - trail_dist
                    if new_sl > t_sl:
                        t_sl = new_sl
                else:
                    new_sl = l + trail_dist
                    if new_sl < t_sl:
                        t_sl = new_sl

            # Check TP / SL
            result = None
            cp = 0.0
            if t_dir == 1:
                if h >= t_tp:   result, cp = "TP", t_tp
                elif l <= t_sl: result, cp = "SL", t_sl
            else:
                if l <= t_tp:   result, cp = "TP", t_tp
                elif h >= t_sl: result, cp = "SL", t_sl

            if result:
                pnl_pts = (cp - t_entry) / POINT if t_dir == 1 else (t_entry - cp) / POINT
                pnl = pnl_pts * lot * PPL
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": result})
                in_trade = False

        # ── Guards ────────────────────────────────────────────────────────────
        if in_trade:
            continue
        if (i - last_entry) < params.cooldown_bars:
            continue
        if balance < start_bal * 0.85:
            break

        d = trend[i]
        if d == 0:
            continue

        # ── Signal checks ─────────────────────────────────────────────────────
        av = atr_arr[i]
        if np.isnan(av) or av <= 0:
            continue

        # ATR volatility-regime filter
        av_avg = atr_avg_arr[i]
        if not np.isnan(av_avg) and av_avg > 0:
            ratio = av / av_avg
            if ratio < params.atr_ratio_min or ratio > params.atr_ratio_max:
                continue

        # Candle body quality
        br = body_arr[i]
        if np.isnan(br) or br < params.min_body_ratio:
            continue

        # M15 momentum score (RSI zone + MACD histogram acceleration)
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

        # Setup: breakout OR Fibonacci pullback
        sh   = swing_h[i]
        sl_v = swing_l[i]
        e20  = e20_arr[i]

        is_breakout = False
        is_pullback = False

        if not (np.isnan(sh) or np.isnan(sl_v)):
            if d == 1 and c > sh + params.breakout_atr_mult * av:
                is_breakout = True
            elif d == -1 and c < sl_v - params.breakout_atr_mult * av:
                is_breakout = True

            swing_range = sh - sl_v
            if swing_range > 0 and not np.isnan(e20):
                near_ema = abs(c - e20) / e20 < params.fib_ema_tol
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

        if not (is_breakout or is_pullback):
            continue

        # ── Open trade on next bar open ───────────────────────────────────────
        if i + 1 >= n:
            break
        entry = open_[i + 1]

        sl_dist = params.atr_sl_mult * av
        tp_dist = params.atr_tp_mult * av
        be_dist = params.atr_be_mult * av

        if d == 1:
            t_sl = entry - sl_dist
            t_tp = entry + tp_dist
            t_be_trig = entry + be_dist
        else:
            t_sl = entry + sl_dist
            t_tp = entry - tp_dist
            t_be_trig = entry - be_dist

        in_trade   = True
        t_dir      = d
        t_entry    = entry
        t_be_moved = False
        t_atr      = av
        last_entry = i

    # Close any remaining open position at last bar
    if in_trade:
        cp = close[-1]
        pnl_pts = (cp - t_entry) / POINT if t_dir == 1 else (t_entry - cp) / POINT
        pnl = pnl_pts * lot * PPL
        balance += pnl
        trades.append({"pnl": round(pnl, 2), "result": "END"})

    # ── Stats ─────────────────────────────────────────────────────────────────
    if not trades:
        return {"n": 0, "wins": 0, "win_rate": 0.0,
                "net_pnl": 0.0, "roi": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    net = balance - start_bal

    return {
        "n":             len(trades),
        "wins":          len(wins),
        "win_rate":      len(wins) / len(trades) * 100,
        "net_pnl":       round(net, 2),
        "roi":           round(net / start_bal * 100, 2),
        "avg_win":       round(gw / max(len(wins), 1), 2),
        "avg_loss":      round(-gl / max(len(losses), 1), 2),
        "profit_factor": round(gw / max(gl, 0.01), 2),
    }
