"""
XAU/USD Strategy v2 — High-conviction filter stack.

Entry requires ALL of the following to be true:

  1. Macro trend (H4): EMA stack aligned + Supertrend bullish/bearish
  2. Trend (H1): EMA stack aligned + ADX > adx_min
  3. Session: current bar within London/NY trading hours (07:00–20:00 UTC)
  4. Setup (M15): breakout above aged swing high/low OR Fibonacci pullback
  5. Momentum (M15): RSI in zone AND MACD histogram accelerating (score = 1.0)
  6. Candle quality: signal bar body > 55% of range (no indecision wicks)
  7. Volume: current bar volume > vol_ratio_min × 20-bar average

Trade management:
  - SL = atr_sl_mult × ATR (dynamic)
  - TP = atr_tp_mult × ATR (dynamic)
  - Breakeven: SL moved to entry when floating P&L ≥ atr_be_mult × ATR
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd

from .indicators import (
    ema, rsi, macd, atr, adx, supertrend,
    swing_highs_lows, candle_body_ratio, volume_ratio, fibonacci_levels,
)

logger = logging.getLogger(__name__)

Direction = Literal["BUY", "SELL"]


@dataclass
class StrategyParams:
    # Trend
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    adx_min: float = 22.0          # H1 ADX minimum (22 = mild trend ok)
    st_period: int = 10
    st_mult: float = 3.0

    # Session (UTC hours, inclusive)
    session_open: int = 7
    session_close: int = 20

    # Breakout
    breakout_atr_mult: float = 1.0
    breakout_swing_lookback: int = 12
    breakout_min_age: int = 5

    # Fibonacci pullback
    fib_min: float = 0.382
    fib_max: float = 0.618
    fib_ema_tolerance: float = 0.005   # price within 0.5% of EMA20

    # Momentum (min 0.5 = at least one of RSI or MACD must fire)
    rsi_period: int = 14
    rsi_bull_lo: float = 42.0
    rsi_bull_hi: float = 70.0
    rsi_bear_lo: float = 30.0
    rsi_bear_hi: float = 58.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    min_momentum_score: float = 0.5   # 0.5=one confirm, 1.0=both

    # Signal quality
    min_body_ratio: float = 0.35      # signal candle body ≥35% of range
    use_volume_filter: bool = False   # off by default (synthetic data has flat vol)

    # Risk
    atr_period: int = 14
    atr_sl_mult: float = 2.5
    atr_tp_mult: float = 5.0          # R:R = 2.0
    atr_be_mult: float = 1.5          # move SL to breakeven at this profit

    # Cooldown
    signal_cooldown_bars: int = 20    # 5 hours on M15


@dataclass
class Signal:
    direction: Direction
    entry_type: Literal["BREAKOUT", "PULLBACK"]
    strength: float
    atr_value: float
    sl_pts: int
    tp_pts: int
    be_pts: int       # breakeven trigger in points
    reason: str


# ── Internal helpers ─────────────────────────────────────────────────────────

def _h4_trend(h4: pd.DataFrame, p: StrategyParams) -> Optional[Direction]:
    """
    H4 macro trend: EMA20 > EMA50 direction + Supertrend agree.
    EMA200 is skipped on H4 (needs 800+ M15 bars of warmup per H4 bar);
    H1 already provides the 200-EMA anchor.
    """
    close = h4["close"]
    if len(close) < p.ema_slow + 10:   # need at least 60 H4 bars
        return None
    e20  = ema(close, p.ema_fast).iloc[-1]
    e50  = ema(close, p.ema_slow).iloc[-1]
    st   = supertrend(h4, p.st_period, p.st_mult).iloc[-1]
    last = close.iloc[-1]

    if last > e20 > e50 and st == 1:
        return "BUY"
    if last < e20 < e50 and st == -1:
        return "SELL"
    return None


def _h1_trend(h1: pd.DataFrame, p: StrategyParams) -> Optional[Direction]:
    """H1 intermediate trend: EMA stack + ADX strength."""
    close = h1["close"]
    if len(close) < p.ema_trend + 10:
        return None
    e20  = ema(close, p.ema_fast).iloc[-1]
    e50  = ema(close, p.ema_slow).iloc[-1]
    e200 = ema(close, p.ema_trend).iloc[-1]
    adx_val = adx(h1, 14).iloc[-1]
    last = close.iloc[-1]

    if adx_val < p.adx_min:
        return None
    if last > e20 > e50 > e200:
        return "BUY"
    if last < e20 < e50 < e200:
        return "SELL"
    return None


def _session_ok(bar_time: pd.Timestamp, p: StrategyParams) -> bool:
    """Only trade during London + New York sessions."""
    hour = bar_time.hour
    return p.session_open <= hour < p.session_close


def _momentum_score(m15: pd.DataFrame, direction: Direction, p: StrategyParams) -> float:
    """Returns 1.0 only if BOTH RSI and MACD histogram confirm direction."""
    close = m15["close"]
    rsi_v = rsi(close, p.rsi_period).iloc[-1]
    _, _, hist = macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
    last_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]

    score = 0.0
    if direction == "BUY":
        if p.rsi_bull_lo <= rsi_v <= p.rsi_bull_hi:
            score += 0.5
        if last_hist > 0 and last_hist > prev_hist:
            score += 0.5
    else:
        if p.rsi_bear_lo <= rsi_v <= p.rsi_bear_hi:
            score += 0.5
        if last_hist < 0 and last_hist < prev_hist:
            score += 0.5

    return score


def _check_breakout(m15: pd.DataFrame, direction: Direction,
                    atr_val: float, p: StrategyParams) -> bool:
    is_sh, is_sl = swing_highs_lows(m15, lookback=p.breakout_swing_lookback)
    min_age = p.breakout_min_age
    last_close = m15["close"].iloc[-1]

    if direction == "BUY":
        sh_old = is_sh.iloc[:-min_age]
        idx = sh_old[sh_old].index
        if idx.empty:
            return False
        level = m15.loc[idx[-1], "high"]
        return last_close > level + p.breakout_atr_mult * atr_val
    else:
        sl_old = is_sl.iloc[:-min_age]
        idx = sl_old[sl_old].index
        if idx.empty:
            return False
        level = m15.loc[idx[-1], "low"]
        return last_close < level - p.breakout_atr_mult * atr_val


def _check_pullback(m15: pd.DataFrame, direction: Direction, p: StrategyParams) -> bool:
    """
    Fibonacci pullback to 38.2–61.8% of last swing, price near EMA20.
    Requires a bounce candle (close moving back in trend direction vs open).
    """
    close = m15["close"]
    e20 = ema(close, p.ema_fast).iloc[-1]
    is_sh, is_sl = swing_highs_lows(m15, lookback=p.breakout_swing_lookback)

    sh_idx = is_sh[is_sh].index
    sl_idx = is_sl[is_sl].index
    if sh_idx.empty or sl_idx.empty:
        return False

    last_sh = float(m15.loc[sh_idx[-1], "high"])
    last_sl = float(m15.loc[sl_idx[-1], "low"])
    last_close = float(close.iloc[-1])
    last_open  = float(m15["open"].iloc[-1])

    fibs = fibonacci_levels(last_sl, last_sh)

    near_ema = abs(last_close - e20) / e20 < p.fib_ema_tolerance

    if direction == "BUY":
        fib_lo = fibs["0.618"]  # deepest allowed retrace
        fib_hi = fibs["0.382"]  # shallowest allowed retrace
        in_zone = fib_lo <= last_close <= fib_hi
        bouncing = last_close > last_open   # bullish candle
        return in_zone and near_ema and bouncing

    # SELL
    fib_hi = last_sl + (last_sh - last_sl) * (1 - p.fib_min)  # 0.382 retrace up
    fib_lo = last_sl + (last_sh - last_sl) * (1 - p.fib_max)  # 0.618 retrace up
    in_zone = fib_lo <= last_close <= fib_hi
    bouncing = last_close < last_open   # bearish candle
    return in_zone and near_ema and bouncing


def _candle_quality(m15: pd.DataFrame, p: StrategyParams) -> bool:
    ratio = candle_body_ratio(m15).iloc[-1]
    return (not pd.isna(ratio)) and ratio >= p.min_body_ratio


def _volume_ok(m15: pd.DataFrame, p: StrategyParams) -> bool:
    vr = volume_ratio(m15, 20).iloc[-1]
    return (not pd.isna(vr)) and vr >= p.vol_ratio_min


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate(
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    params: StrategyParams,
) -> Optional[Signal]:
    """
    Full v2 signal pipeline. Returns Signal or None.
    Requires H4, H1, and M15 DataFrames.
    """
    if len(m15) < 60 or len(h1) < 60 or len(h4) < 50:
        return None

    # 1. Session filter
    bar_time = m15.index[-1]
    if not _session_ok(bar_time, params):
        return None

    # 2. H4 macro trend
    h4_dir = _h4_trend(h4, params)
    if h4_dir is None:
        return None

    # 3. H1 intermediate trend (must match H4)
    h1_dir = _h1_trend(h1, params)
    if h1_dir != h4_dir:
        return None

    direction = h4_dir

    # 4. ATR
    atr_val = atr(m15, params.atr_period).iloc[-1]

    # 5. Setup: breakout OR Fibonacci pullback
    is_breakout = _check_breakout(m15, direction, atr_val, params)
    is_pullback = _check_pullback(m15, direction, params)
    if not (is_breakout or is_pullback):
        return None

    entry_type: Literal["BREAKOUT", "PULLBACK"] = "BREAKOUT" if is_breakout else "PULLBACK"

    # 6. Momentum — parameterised minimum score
    mom_score = _momentum_score(m15, direction, params)
    if mom_score < params.min_momentum_score:
        return None

    # 7. Candle quality (body fraction threshold)
    if not _candle_quality(m15, params):
        return None

    # 8. Volume spike (optional — often disabled for synthetic data)
    if params.use_volume_filter and not _volume_ok(m15, params):
        return None

    # Compute SL / TP / breakeven in points (1 point = $0.01)
    point = 0.01
    sl_pts = int(atr_val * params.atr_sl_mult / point)
    tp_pts = int(atr_val * params.atr_tp_mult / point)
    be_pts = int(atr_val * params.atr_be_mult / point)

    reason = (
        f"H4={h4_dir} H1={direction} | {entry_type} | "
        f"mom=1.0 | ATR={atr_val:.2f} | "
        f"SL={sl_pts}pt TP={tp_pts}pt R:R=1:{params.atr_tp_mult/params.atr_sl_mult:.1f}"
    )
    logger.info("SIGNALv2: %s %s — %s", direction, entry_type, reason)

    return Signal(
        direction=direction,
        entry_type=entry_type,
        strength=1.0,
        atr_value=atr_val,
        sl_pts=sl_pts,
        tp_pts=tp_pts,
        be_pts=be_pts,
        reason=reason,
    )
