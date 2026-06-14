"""
Generates realistic synthetic XAU/USD OHLCV data for demo/backtesting.

Uses a regime-switching model:
  - Randomly alternates between trending (bullish/bearish) and ranging regimes
  - Volatility calibrated to real gold: ATR ~$12-20 per day on 15-min bars
  - Adds realistic wicks, gaps, and volume profile
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


def generate_xauusd(
    n_bars: int = 2000,
    timeframe_minutes: int = 15,
    start_price: float = 2340.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Regime parameters (switch every 60-300 bars)
    regime = "bull"
    regime_bars_left = int(rng.integers(80, 200))

    prices = []
    close = start_price

    for i in range(n_bars):
        # Regime switch logic
        regime_bars_left -= 1
        if regime_bars_left <= 0:
            regime = rng.choice(["bull", "bear", "range", "range"])
            regime_bars_left = int(rng.integers(60, 250))

        # Base drift and volatility per 15-min bar
        # Real gold: ~0.5% daily vol → ~0.10% per 15-min bar
        bar_vol = close * 0.0010   # ~$2.35 sigma per bar at $2350

        if regime == "bull":
            drift = close * 0.00008    # slight upward drift
        elif regime == "bear":
            drift = -close * 0.00008
        else:
            drift = 0.0

        # Add occasional momentum bursts (news events)
        if rng.random() < 0.02:
            burst = rng.choice([-1, 1]) * bar_vol * rng.uniform(2.0, 4.0)
        else:
            burst = 0.0

        move = drift + burst + rng.normal(0, bar_vol)
        open_price = close
        close = max(open_price + move, 1000.0)   # floor at $1000

        # Generate realistic high/low around open-close
        body_hi = max(open_price, close)
        body_lo = min(open_price, close)
        body_size = abs(close - open_price)
        wick_range = bar_vol * rng.uniform(0.3, 1.0)

        high  = body_hi + wick_range * rng.uniform(0.1, 0.6)
        low   = body_lo - wick_range * rng.uniform(0.1, 0.6)
        low   = max(low, 1000.0)

        # Volume: higher during directional moves
        base_volume = 1000
        volume = int(base_volume * (1.0 + abs(move) / bar_vol) * rng.uniform(0.7, 1.4))

        prices.append({
            "open":   round(open_price, 2),
            "high":   round(high, 2),
            "low":    round(low, 2),
            "close":  round(close, 2),
            "volume": volume,
        })

    # Build timestamp index (skip weekends — markets closed)
    start_dt = datetime(2025, 12, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = []
    t = start_dt
    added = 0
    while added < n_bars:
        if t.weekday() < 5:   # Mon-Fri only
            timestamps.append(t)
            added += 1
        t += timedelta(minutes=timeframe_minutes)

    df = pd.DataFrame(prices, index=pd.DatetimeIndex(timestamps))
    return df


def resample_to_h1(m15: pd.DataFrame) -> pd.DataFrame:
    """Resample M15 bars to H1 for trend analysis."""
    return m15.resample("1h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
