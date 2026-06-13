#!/usr/bin/env python3
"""
Grid-search optimizer for the XAU/USD strategy.
Uses the vectorised fast_backtest engine — runs 108 configs in < 2 minutes.

Usage:
    python optimize.py              # optimize + print best final backtest
    python optimize.py --seed 42   # different data seed
    python optimize.py --bars 15000
"""

import argparse
import itertools
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from trader.synthetic_data import generate_xauusd, resample_to_h1
from trader.fast_backtest import BacktestParams, run as fast_run

FIXED_LOT = 0.30

DIV = "─" * 74


def resample_h4(m15):
    return m15.resample("4h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna()


def optimize(m15, h1, h4, min_trades: int = 20):
    print(f"\n{'═'*74}")
    print("  STRATEGY OPTIMIZER  —  XAU/USD  —  0.30 lot")
    print(f"  Dataset: {len(m15)} M15 bars | {len(h1)} H1 | {len(h4)} H4")
    print(f"{'─'*74}")
    print(f"  {'ADX':>3} {'ST':>4} {'BO':>4} {'MOM':>4} {'SL':>4} {'TP':>4} "
          f"{'N':>4} {'WIN%':>5} {'P/F':>5} {'AVG_W':>7} {'AVG_L':>7} {'NET P&L':>9} {'ROI':>5}")
    print(DIV)

    grid = {
        "adx_min":            [20, 25, 28],
        "st_mult":            [2.5, 3.0],
        "breakout_atr_mult":  [0.8, 1.2],
        "min_momentum_score": [0.5, 1.0],
        "atr_sl_mult":        [2.0, 2.5, 3.0],
        "atr_tp_mult":        [3.5, 4.5, 5.5],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    total = len(combos)
    results = []

    t0 = time.time()
    for idx, vals in enumerate(combos):
        p = BacktestParams(**dict(zip(keys, vals)), atr_be_mult=vals[4] * 0.6)
        s = fast_run(m15, h1, h4, p, lot=FIXED_LOT)

        if s["n"] < min_trades:
            continue

        score = s["win_rate"] * s["profit_factor"]
        results.append((score, p, s))

        adx_v, st_m, bo_m, mom_s, sl_m, tp_m = vals
        print(f"  {adx_v:>3.0f} {st_m:>4.1f} {bo_m:>4.1f} {mom_s:>4.1f} "
              f"{sl_m:>4.1f} {tp_m:>4.1f} "
              f"{s['n']:>4} {s['win_rate']:>5.1f} {s['profit_factor']:>5.2f} "
              f"${s['avg_win']:>6.0f} ${s['avg_loss']:>6.0f} "
              f"${s['net_pnl']:>8.2f} {s['roi']:>4.1f}%")

    elapsed = time.time() - t0
    print(f"\n  [{total} configs tested in {elapsed:.1f}s | {len(results)} with ≥{min_trades} trades]")

    if not results:
        print("  No valid configs found.")
        return BacktestParams()

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'═'*74}")
    print("  TOP 5 CONFIGURATIONS  (ranked by win_rate × profit_factor)")
    print(f"{'─'*74}")
    print(f"  {'#':>2} {'ADX':>3} {'ST':>4} {'BO':>4} {'MOM':>4} {'SL':>4} {'TP':>4} "
          f"{'N':>4} {'WIN%':>5} {'P/F':>5} {'NET P&L':>9} {'ROI':>5}")
    for rank, (score, p, s) in enumerate(results[:5], 1):
        print(f"  {rank:>2} {p.adx_min:>3.0f} {p.st_mult:>4.1f} {p.breakout_atr_mult:>4.1f} "
              f"{p.min_momentum_score:>4.1f} {p.atr_sl_mult:>4.1f} {p.atr_tp_mult:>4.1f} "
              f"{s['n']:>4} {s['win_rate']:>5.1f} {s['profit_factor']:>5.2f} "
              f"${s['net_pnl']:>8.2f} {s['roi']:>4.1f}%")

    best_score, best_p, best_s = results[0]
    print(f"\n{'═'*74}")
    print("  BEST PARAMETER SET")
    print(f"{'─'*74}")
    print(f"  ADX min              : {best_p.adx_min}")
    print(f"  Supertrend mult      : {best_p.st_mult}")
    print(f"  Breakout ATR mult    : {best_p.breakout_atr_mult}")
    print(f"  Min momentum score   : {best_p.min_momentum_score}")
    print(f"  SL ATR mult          : {best_p.atr_sl_mult}")
    print(f"  TP ATR mult          : {best_p.atr_tp_mult}")
    print(f"  BE ATR mult          : {best_p.atr_be_mult:.2f}")
    print(f"  R:R ratio            : 1:{best_p.atr_tp_mult/best_p.atr_sl_mult:.1f}")
    print(f"{'─'*74}")
    print_stats(best_s)
    print(f"{'═'*74}\n")
    return best_p


def print_stats(s: dict):
    print(f"  Trades         : {s['n']}  ({s['wins']} wins, {s['n']-s['wins']} losses)")
    print(f"  Win rate       : {s['win_rate']:.1f}%")
    print(f"  Profit factor  : {s['profit_factor']:.2f}")
    print(f"  Avg win        : ${s['avg_win']:.2f}")
    print(f"  Avg loss       : ${s['avg_loss']:.2f}")
    print(f"  Net P&L        : ${s['net_pnl']:+.2f}")
    print(f"  ROI            : {s['roi']:+.2f}%")
    print(f"  Final equity   : ${10_000 + s['net_pnl']:,.2f}")


def cross_validate(m15, h1, h4, best_p: BacktestParams):
    """Run the best params on different data seeds to check robustness."""
    print(f"\n{'═'*74}")
    print("  CROSS-VALIDATION  —  5 different data seeds")
    print(f"{'─'*74}")
    print(f"  {'SEED':>5} {'N':>4} {'WIN%':>5} {'P/F':>5} {'NET P&L':>10} {'ROI':>6}")
    print(DIV)

    seeds = [7, 13, 42, 99, 123]
    total_n = 0
    total_pnl = 0.0
    all_wins = 0

    for seed in seeds:
        m15_v = generate_xauusd(n_bars=10000, timeframe_minutes=15, seed=seed)
        h1_v  = resample_to_h1(m15_v)
        h4_v  = resample_h4(m15_v)
        s = fast_run(m15_v, h1_v, h4_v, best_p, lot=FIXED_LOT)
        total_n   += s["n"]
        total_pnl += s["net_pnl"]
        all_wins  += s["wins"]
        print(f"  {seed:>5} {s['n']:>4} {s['win_rate']:>5.1f} "
              f"{s['profit_factor']:>5.2f} ${s['net_pnl']:>9.2f} {s['roi']:>5.1f}%")

    if total_n:
        avg_wr = all_wins / total_n * 100
        avg_pnl = total_pnl / len(seeds)
        print(DIV)
        print(f"  {'AVG':>5} {total_n//len(seeds):>4} {avg_wr:>5.1f}      "
              f"     ${avg_pnl:>9.2f}")
    print(f"{'═'*74}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAU/USD Strategy Optimizer")
    parser.add_argument("--seed",  type=int, default=7)
    parser.add_argument("--bars",  type=int, default=12000)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--no-crossval", action="store_true")
    args = parser.parse_args()

    print(f"\n  Generating {args.bars} bars (seed={args.seed})…", flush=True)
    m15 = generate_xauusd(n_bars=args.bars, timeframe_minutes=15, seed=args.seed)
    h1  = resample_to_h1(m15)
    h4  = resample_h4(m15)

    best_p = optimize(m15, h1, h4, min_trades=args.min_trades)

    if not args.no_crossval and best_p is not None:
        cross_validate(m15, h1, h4, best_p)
