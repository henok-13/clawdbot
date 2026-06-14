"""
XAU/USD trading strategy:
  1. Identify trend (EMA stack + ADX on H1)
  2. Wait for breakout OR pullback on M15
  3. Confirm momentum (MACD cross + RSI in correct zone)
  4. Return a signal dict or None
"""

import logging
from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd

from .config import strategy_cfg as cfg
from .indicators import ema, rsi, macd, atr, adx, swing_highs_lows

logger = logging.getLogger(__name__)

Direction = Literal["BUY", "SELL"]


@dataclass
class Signal:
    direction: Direction
    entry_type: Literal["BREAKOUT", "PULLBACK"]
    strength: float          # 0.0 – 1.0 composite score
    atr_value: float         # current ATR for SL sizing reference
    reason: str


def _trend_direction(h1: pd.DataFrame) -> Optional[Direction]:
    """
    Determine trend from H1 chart.
    Bullish : close > EMA20 > EMA50 > EMA200 AND ADX > threshold
    Bearish : close < EMA20 < EMA50 < EMA200 AND ADX > threshold
    """
    close = h1["close"]
    e20 = ema(close, cfg.ema_fast)
    e50 = ema(close, cfg.ema_slow)
    e200 = ema(close, cfg.ema_trend)
    adx_val = adx(h1, 14)

    last_close = close.iloc[-1]
    last_e20 = e20.iloc[-1]
    last_e50 = e50.iloc[-1]
    last_e200 = e200.iloc[-1]
    last_adx = adx_val.iloc[-1]

    if last_adx < cfg.adx_threshold:
        logger.debug("ADX %.1f below threshold — no clear trend", last_adx)
        return None

    if last_close > last_e20 > last_e50 > last_e200:
        logger.debug("H1 trend: BULLISH (ADX=%.1f)", last_adx)
        return "BUY"

    if last_close < last_e20 < last_e50 < last_e200:
        logger.debug("H1 trend: BEARISH (ADX=%.1f)", last_adx)
        return "SELL"

    return None


def _momentum_ok(m15: pd.DataFrame, direction: Direction) -> tuple[bool, float]:
    """
    Returns (confirmed, score 0-1).
    Checks RSI position and MACD histogram direction.
    """
    close = m15["close"]
    rsi_val = rsi(close, cfg.rsi_period).iloc[-1]
    _, _, hist = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    prev_hist = hist.iloc[-2]
    last_hist = hist.iloc[-1]

    score = 0.0
    reasons = []

    if direction == "BUY":
        # RSI between 45-65 — strong but not overbought
        if cfg.rsi_bull_min <= rsi_val < cfg.rsi_overbought:
            score += 0.5
            reasons.append(f"RSI={rsi_val:.1f}")
        # MACD histogram turning positive
        if last_hist > 0 and last_hist > prev_hist:
            score += 0.5
            reasons.append("MACD↑")
    else:
        if cfg.rsi_oversold < rsi_val <= cfg.rsi_bear_max:
            score += 0.5
            reasons.append(f"RSI={rsi_val:.1f}")
        if last_hist < 0 and last_hist < prev_hist:
            score += 0.5
            reasons.append("MACD↓")

    confirmed = score >= 0.5   # at least one momentum factor must agree
    logger.debug("Momentum %s score=%.1f (%s)", direction, score, ", ".join(reasons))
    return confirmed, score


def _check_breakout(m15: pd.DataFrame, direction: Direction, atr_val: float) -> bool:
    """
    Price has just closed beyond the most recent swing high (BUY) or
    swing low (SELL) by at least breakout_atr_mult * ATR.
    The reference swing must be at least breakout_min_swing_age_bars old
    to avoid triggering on the same bar the swing formed.
    """
    is_sh, is_sl = swing_highs_lows(m15, lookback=12)
    min_age = cfg.breakout_min_swing_age_bars

    if direction == "BUY":
        # Only consider swings formed ≥ min_age bars ago
        sh_series = is_sh.iloc[:-min_age] if len(is_sh) > min_age else is_sh
        swing_highs_idx = sh_series[sh_series].index
        if swing_highs_idx.empty:
            return False
        last_swing_idx = swing_highs_idx[-1]
        last_swing_high = m15.loc[last_swing_idx, "high"]
        breakout_level = last_swing_high + cfg.breakout_atr_mult * atr_val
        result = m15["close"].iloc[-1] > breakout_level
        logger.debug("Breakout BUY: close=%.2f vs level=%.2f (swing@%.2f + %.1f*ATR) → %s",
                     m15["close"].iloc[-1], breakout_level, last_swing_high,
                     cfg.breakout_atr_mult, result)
        return result

    # SELL
    sl_series = is_sl.iloc[:-min_age] if len(is_sl) > min_age else is_sl
    swing_lows_idx = sl_series[sl_series].index
    if swing_lows_idx.empty:
        return False
    last_swing_idx = swing_lows_idx[-1]
    last_swing_low = m15.loc[last_swing_idx, "low"]
    breakout_level = last_swing_low - cfg.breakout_atr_mult * atr_val
    result = m15["close"].iloc[-1] < breakout_level
    logger.debug("Breakout SELL: close=%.2f vs level=%.2f → %s",
                 m15["close"].iloc[-1], breakout_level, result)
    return result


def _check_pullback(m15: pd.DataFrame, direction: Direction) -> bool:
    """
    Price has pulled back to the EMA20 zone after a strong move,
    and the current candle is starting to reject back in trend direction.
    Retrace must be between pullback_min_retrace and pullback_max_retrace of last swing.
    """
    close = m15["close"]
    e20 = ema(close, cfg.ema_fast)
    is_sh, is_sl = swing_highs_lows(m15, lookback=8)

    last_close = close.iloc[-1]
    last_ema20 = e20.iloc[-1]

    if direction == "BUY":
        swing_highs = m15["high"][is_sh]
        swing_lows = m15["low"][is_sl]
        if swing_highs.empty or swing_lows.empty:
            return False
        swing_range = swing_highs.iloc[-1] - swing_lows.iloc[-1]
        if swing_range <= 0:
            return False
        retrace = (swing_highs.iloc[-1] - last_close) / swing_range
        near_ema = abs(last_close - last_ema20) / last_ema20 < 0.003  # within 0.3%
        result = (
            cfg.pullback_min_retrace <= retrace <= cfg.pullback_max_retrace
            and near_ema
            and last_close > last_ema20 * 0.997   # not broken below ema20
        )
        logger.debug("Pullback BUY: retrace=%.2f near_ema=%s → %s", retrace, near_ema, result)
        return result

    swing_highs = m15["high"][is_sh]
    swing_lows = m15["low"][is_sl]
    if swing_highs.empty or swing_lows.empty:
        return False
    swing_range = swing_highs.iloc[-1] - swing_lows.iloc[-1]
    if swing_range <= 0:
        return False
    retrace = (last_close - swing_lows.iloc[-1]) / swing_range
    near_ema = abs(last_close - last_ema20) / last_ema20 < 0.003
    result = (
        cfg.pullback_min_retrace <= retrace <= cfg.pullback_max_retrace
        and near_ema
        and last_close < last_ema20 * 1.003
    )
    logger.debug("Pullback SELL: retrace=%.2f near_ema=%s → %s", retrace, near_ema, result)
    return result


def evaluate(m15: pd.DataFrame, h1: pd.DataFrame) -> Optional[Signal]:
    """
    Main entry point. Returns a Signal if all three conditions are met,
    otherwise None.

    Pipeline:
      1. Trend direction from H1
      2. Breakout OR Pullback setup on M15
      3. Momentum confirmation on M15
    """
    if len(m15) < 60 or len(h1) < 60:
        logger.warning("Insufficient bars for analysis")
        return None

    direction = _trend_direction(h1)
    if direction is None:
        return None

    atr_val = atr(m15, cfg.atr_period).iloc[-1]

    is_breakout = _check_breakout(m15, direction, atr_val)
    is_pullback = _check_pullback(m15, direction)

    if not (is_breakout or is_pullback):
        logger.debug("No breakout or pullback setup for %s", direction)
        return None

    entry_type: Literal["BREAKOUT", "PULLBACK"] = "BREAKOUT" if is_breakout else "PULLBACK"

    momentum_ok, mom_score = _momentum_ok(m15, direction)
    if not momentum_ok:
        logger.debug("Momentum not confirmed for %s %s", direction, entry_type)
        return None

    # Composite strength: breakout/pullback contributes 0.5, momentum the rest
    strength = round(0.5 + mom_score * 0.5, 2)

    # Reject weak signals — only trade high-conviction setups
    if strength < cfg.min_signal_strength:
        logger.debug("Signal strength %.2f below threshold %.2f — skipped",
                     strength, cfg.min_signal_strength)
        return None

    reason = (
        f"H1 trend={direction} | {entry_type} on M15 | "
        f"momentum score={mom_score:.1f} | ATR={atr_val:.2f}"
    )
    logger.info("SIGNAL: %s %s strength=%.2f — %s", direction, entry_type, strength, reason)

    return Signal(
        direction=direction,
        entry_type=entry_type,
        strength=strength,
        atr_value=atr_val,
        reason=reason,
    )
