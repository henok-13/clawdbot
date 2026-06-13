#!/usr/bin/env python3
"""
Scalping backtest: TP=200pts, SL=135pts, target 40+ trades/day.
Uses M5 data with H1 trend filter.

XAU/USD lot size: 0.60
  Win  → +200 pts × 0.60 lot × $1.00/pt − spread = +$120 − $2.40 = +$117.60 net
  Loss → −135 pts × 0.60 lot × $1.00/pt − spread = −$81 − $2.40 = −$83.40 net
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
    tp_points      = 200,
    sl_points      = 135,
    max_trades_day = 60,
    session_open   = 6,    # extended session for max frequency
    session_close  = 22,
    min_body_ratio = 0.15,  # minimal body filter
    use_h1_trend   = True,  # keep H1 direction for edge
    use_h1_rsi     = False, # disabled — too restrictive for frequency
    use_h4_trend   = False, # disabled — too restrictive for frequency
    spread_points  = 4,
    adx_min        = 0.0,   # disabled
    min_vol_ratio  = 0.0,   # disabled
    rsi_buy_min    = 40.0,  # wide RSI zones
    rsi_buy_max    = 75.0,
    rsi_sell_min   = 25.0,
    rsi_sell_max   = 60.0,
    be_trigger_points = 50,
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
    print("  XAU/USD SCALPING BACKTEST  |  TP=200pts  SL=135pts  Lot=0.60  | HIGH-FREQ MODE")
    print("=" * 72)
    print(f"  Data: {N_BARS:,} M5 bars (~{N_BARS // (12*5*5):.0f} months)  "
          f"| Session: {PARAMS.session_open:02d}:00-{PARAMS.session_close:02d}:00 UTC (London+NY)  "
          f"| Spread: {PARAMS.spread_points}pts")
    print(f"  Win  → +${PARAMS.tp_points * LOT:.0f} − ${PARAMS.spread_points * LOT:.0f} = "
          f"+${PARAMS.tp_points * LOT - PARAMS.spread_points * LOT:.0f} net per trade")
    print(f"  Loss → −${PARAMS.sl_points * LOT:.0f} − ${PARAMS.spread_points * LOT:.0f} = "
          f"−${PARAMS.sl_points * LOT + PARAMS.spread_points * LOT:.0f} net per trade")
    print()

    rows = []
    for seed in SEEDS:
        m5 = generate_xauusd(n_bars=N_BARS, timeframe_minutes=5, seed=seed)
        h1 = resample_to_h1(m5)
        h4 = resample_to_h4(m5)
        res = run_scalper(m5, h1, h4, params=PARAMS, lot=LOT, start_balance=START_BAL)
        rows.append({
            "seed":       seed,
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
    print(f"  Avg win rate    : {arr_wr.mean():.1f}%  (TP hits vs decisive exits; BE excluded)")
    print(f"  Avg profit factor: {arr_pf.mean():.2f}")
    print(f"  Avg win / loss  : +${arr_aw.mean():.2f} / -${arr_al.mean():.2f}")
    print(f"  Avg net P&L     : ${arr_pnl.mean():,.2f}  "
          f"(min ${arr_pnl.min():,.2f} / max ${arr_pnl.max():,.2f})")

    # Daily P&L derived directly from backtest net P&L ÷ trading days
    daily_pnl = arr_pnl.mean() / arr_nd.mean()
    monthly   = daily_pnl * 22

    print(f"  Est. daily P&L  : ${daily_pnl:,.2f}  (at {LOT} lot)")
    print(f"  Est. monthly P&L: ${monthly:,.2f}  (22 trading days)")
    print(f"  Avg final equity: ${arr_bal.mean():,.2f}  "
          f"(ROI {(arr_bal.mean() - START_BAL) / START_BAL * 100:.1f}%)")
    print()

    win_net  = (PARAMS.tp_points - PARAMS.spread_points) * LOT
    loss_net = (PARAMS.sl_points + PARAMS.spread_points) * LOT
    be_wr = loss_net / (win_net + loss_net) * 100
    print(f"  Break-even win rate: {be_wr:.1f}%  "
          f"(TP={PARAMS.tp_points}pts, SL={PARAMS.sl_points}pts, spread={PARAMS.spread_points}pts, lot={LOT})")
    print("=" * 72)


if __name__ == "__main__":
    main()
