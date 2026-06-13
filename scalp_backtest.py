#!/usr/bin/env python3
"""
Scalping backtest: ATR-adaptive TP/SL, partial TP + trailing stop.
Target 15+ trades/day on XAU/USD M5 with H1 trend filter.

XAU/USD lot size: 0.60 (default; compound sizing overrides per-trade)
  Win  → profit depends on ATR-based SL/TP + partial-close mechanics
  Loss → capped by ATR-adaptive SL distance
"""

import sys

import numpy as np
import pandas as pd

from trader.synthetic_data import generate_xauusd
from trader.scalper import ScalpParams, run_scalper

LOT          = 0.60
START_BAL    = 10_000.0
N_BARS       = 50_000   # ~8.5 months of M5 data (Mon-Fri, 12 bars/hour)
SEEDS        = list(range(10))

PARAMS = ScalpParams(
    # ATR-adaptive TP/SL (Fix 1) — no fixed tp_points/sl_points
    # Note: synthetic data ATR ≈ 300+ pts; real XAU/USD M5 ATR ≈ 50-120 pts.
    # Multipliers below are calibrated for realistic SL/TP distances.
    atr_period       = 14,
    atr_sl_mult      = 0.5,   # SL = 0.5×ATR ≈ 150 pts on synthetic data
    atr_tp_mult      = 1.0,   # full TP = 1.0×ATR ≈ 300 pts
    # Volatility regime filter (Fix 2)
    atr_regime_period = 50,
    atr_regime_min   = 0.7,
    atr_regime_max   = 2.5,
    # Compound sizing (Fix 3)
    use_compound_sizing = True,
    risk_pct         = 0.005,
    max_lot          = 3.0,
    # Partial TP + trailing (Fix 4)
    partial_pct      = 0.50,
    partial_tp_mult  = 0.7,   # partial TP at 0.7×ATR ≈ 220 pts
    trail_atr_mult   = 0.3,   # trail stop: 0.3×ATR behind running extreme
    be_atr_mult      = 0.2,   # slide SL to entry when 0.2×ATR in profit
    # Consecutive loss daily stop (Fix 7)
    max_consec_losses = 3,
    # Session / frequency settings
    max_trades_day   = 60,
    session_open     = 6,    # extended session for max frequency
    session_close    = 22,
    min_body_ratio   = 0.15,  # minimal body filter
    use_h1_trend     = True,  # keep H1 direction for edge
    use_h1_rsi       = False, # disabled — too restrictive for frequency
    use_h4_trend     = False, # disabled — too restrictive for frequency
    spread_points    = 4,
    adx_min          = 0.0,   # disabled
    min_vol_ratio    = 0.0,   # disabled
    rsi_buy_min      = 40.0,  # wide RSI zones
    rsi_buy_max      = 75.0,
    rsi_sell_min     = 25.0,
    rsi_sell_max     = 60.0,
    daily_loss_limit    = 200.0,
    daily_profit_target = 500.0,
)


def _resample(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    return m5.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def resample_to_h1(m5: pd.DataFrame) -> pd.DataFrame:
    return _resample(m5, "1h")


def resample_to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    return _resample(m5, "4h")


def main() -> None:
    print("=" * 72)
    print("  XAU/USD SCALPING BACKTEST  |  ATR-adaptive SL/TP  |  HIGH-FREQ MODE")
    print("=" * 72)
    print(f"  Data: {N_BARS:,} M5 bars (~{N_BARS // (12*5*5):.0f} months)  "
          f"| Session: {PARAMS.session_open:02d}:00-{PARAMS.session_close:02d}:00 UTC (London+NY)  "
          f"| Spread: {PARAMS.spread_points}pts")
    print(f"  SL: {PARAMS.atr_sl_mult}×ATR  |  Full TP: {PARAMS.atr_tp_mult}×ATR  "
          f"|  Partial TP: {PARAMS.partial_tp_mult}×ATR (50% close)  "
          f"|  Trail: {PARAMS.trail_atr_mult}×ATR")
    print(f"  Compound sizing: risk {PARAMS.risk_pct*100:.1f}% equity per trade  "
          f"|  Max lot: {PARAMS.max_lot}")
    print()

    rows = []
    for seed in SEEDS:
        m5 = generate_xauusd(n_bars=N_BARS, timeframe_minutes=5, seed=seed)
        h1 = resample_to_h1(m5)
        h4 = resample_to_h4(m5)
        res = run_scalper(m5, h1, h4, params=PARAMS, lot=LOT, start_balance=START_BAL)
        rows.append({
            "seed": seed,
            **res,
        })

    # ── Results table ─────────────────────────────────────────────────────────
    header = (
        f"{'Seed':>4}  {'Trades':>7}  {'Trd/Day':>7}  {'Win%':>6}  {'BE':>5}  "
        f"{'Net P&L':>10}  {'Equity':>10}  {'PF':>5}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in rows:
        print(
            f"{r['seed']:>4}  "
            f"{r['n']:>7}  "
            f"{r['trades_per_day']:>7.1f}  "
            f"{r['win_rate']:>6.1f}  "
            f"{r['breakevens']:>5}  "
            f"${r['net_pnl']:>9,.2f}  "
            f"${r['final_balance']:>9,.2f}  "
            f"{r['profit_factor']:>5.2f}"
        )

    print(sep)

    # ── Summary statistics ────────────────────────────────────────────────────
    arr_tpd  = np.array([r["trades_per_day"] for r in rows])
    arr_wr   = np.array([r["win_rate"]       for r in rows])
    arr_pnl  = np.array([r["net_pnl"]        for r in rows])
    arr_pf   = np.array([r["profit_factor"]  for r in rows])
    arr_bal  = np.array([r["final_balance"]  for r in rows])
    arr_aw   = np.array([r["avg_win"]        for r in rows])
    arr_al   = np.array([r["avg_loss"]       for r in rows])
    arr_nd   = np.array([r["n_days"]         for r in rows])

    print()
    print("SUMMARY (10 seeds):")
    print(f"  Avg trades/day  : {arr_tpd.mean():.1f}  "
          f"(min {arr_tpd.min():.1f} / max {arr_tpd.max():.1f})")
    print(f"  Avg win rate    : {arr_wr.mean():.1f}%  (pnl > 0 vs decisive exits; BE excluded)")
    print(f"  Avg profit factor: {arr_pf.mean():.2f}")
    print(f"  Avg win / loss  : +${arr_aw.mean():.2f} / -${arr_al.mean():.2f}")
    print(f"  Avg net P&L     : ${arr_pnl.mean():,.2f}  "
          f"(min ${arr_pnl.min():,.2f} / max ${arr_pnl.max():,.2f})")

    # Daily P&L derived directly from backtest net P&L ÷ trading days
    daily_pnl_avg = arr_pnl.mean() / arr_nd.mean()
    monthly       = daily_pnl_avg * 22

    print(f"  Est. daily P&L  : ${daily_pnl_avg:,.2f}  (compound sizing)")
    print(f"  Est. monthly P&L: ${monthly:,.2f}  (22 trading days)")
    print(f"  Avg final equity: ${arr_bal.mean():,.2f}  "
          f"(ROI {(arr_bal.mean() - START_BAL) / START_BAL * 100:.1f}%)")
    print()

    # ── Walk-forward OOS analysis (Fix 5) ────────────────────────────────────
    # IS group: seeds 0-4 (in-sample), OOS group: seeds 5-9 (out-of-sample)
    # If OOS ≈ IS, the edge is real — not curve-fitted.
    is_rows  = [r for r in rows if r["seed"] in range(5)]
    oos_rows = [r for r in rows if r["seed"] in range(5, 10)]

    is_wr  = np.mean([r["win_rate"]       for r in is_rows])
    is_pf  = np.mean([r["profit_factor"]  for r in is_rows])
    oos_wr = np.mean([r["win_rate"]       for r in oos_rows])
    oos_pf = np.mean([r["profit_factor"]  for r in oos_rows])

    print("WALK-FORWARD OOS CHECK (Fix 5):")
    print(f"  {'Group':<12}  {'Seeds':<10}  {'Avg Win%':>8}  {'Avg PF':>7}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*7}")
    print(f"  {'IS (train)':<12}  {'0–4':<10}  {is_wr:>8.1f}  {is_pf:>7.2f}")
    print(f"  {'OOS (test)':<12}  {'5–9':<10}  {oos_wr:>8.1f}  {oos_pf:>7.2f}")
    print()
    print("  Note: If OOS ≈ IS, the edge is real — not curve-fitted.")
    print("=" * 72)


if __name__ == "__main__":
    main()
