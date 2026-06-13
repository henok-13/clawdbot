"""Technical indicator calculations (pure pandas/numpy — no TA-lib dependency)."""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_vals = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr_s = tr_vals.ewm(com=period - 1, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(com=period - 1, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(com=period - 1, adjust=False).mean() / atr_s

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(com=period - 1, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Supertrend indicator. Returns a Series of +1 (bullish) / -1 (bearish).
    Classic implementation: midpoint ± multiplier * ATR bands.
    """
    high, low, close = df["high"], df["low"], df["close"]
    atr_s = atr(df, period)

    mid = (high + low) / 2
    upper = mid + multiplier * atr_s
    lower = mid - multiplier * atr_s

    trend = pd.Series(1, index=df.index, dtype=int)
    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, len(df)):
        # Upper band
        if upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Lower band
        if lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Trend direction
        if trend.iloc[i - 1] == -1 and close.iloc[i] > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif trend.iloc[i - 1] == 1 and close.iloc[i] < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    return trend


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current bar volume relative to the rolling mean — > 1.5 = volume spike."""
    vol = df["volume"].astype(float)
    return vol / vol.rolling(period).mean()


def swing_highs_lows(df: pd.DataFrame, lookback: int = 10) -> tuple[pd.Series, pd.Series]:
    """
    Return boolean Series marking swing high and swing low bars.
    A swing high is a bar whose high is the highest in [i-lookback, i+lookback].
    """
    highs = df["high"]
    lows = df["low"]
    n = len(df)

    is_swing_high = pd.Series(False, index=df.index)
    is_swing_low = pd.Series(False, index=df.index)

    for i in range(lookback, n - lookback):
        window_highs = highs.iloc[i - lookback : i + lookback + 1]
        window_lows = lows.iloc[i - lookback : i + lookback + 1]
        if highs.iloc[i] == window_highs.max():
            is_swing_high.iloc[i] = True
        if lows.iloc[i] == window_lows.min():
            is_swing_low.iloc[i] = True

    return is_swing_high, is_swing_low


def candle_body_ratio(df: pd.DataFrame) -> pd.Series:
    """Body size as fraction of total candle range (0–1). Strong candles > 0.55."""
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    return body / rng.replace(0, np.nan)


def fibonacci_levels(swing_low: float, swing_high: float) -> dict:
    """Return key Fibonacci retracement levels for a given swing."""
    diff = swing_high - swing_low
    return {
        "0.0":   swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5":   swing_high - 0.5   * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1.0":   swing_low,
    }
