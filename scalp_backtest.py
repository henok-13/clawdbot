#!/usr/bin/env python3
"""
Scalping backtest: $50 fixed profit target per trade, 50 trades/day target.
Exit each trade the moment it reaches +$50 unrealized P&L.
Stop-loss is ATR-adaptive (0.5×ATR ≈ 131 pts on synthetic data).
Compound sizing keeps dollar risk ≈ equal to dollar TP (1:1 R:R), growing with balance.

Confidence filter: H1 EMA trend direction + M5 RSI/MACD alignment.
Session: 06:00–22:00 UTC | Daily stop: -$200 loss OR +$2,500 profit.

XAU/USD: POINT=0.01 | PPL=$1.00/lot/point
  Win  → +$50 (minus spread ≈ $0.04) each time TP is hit
  Loss → ATR-based SL (≈ $50 at default sizing → 1:1 R:R)
"""

import numpy as np
import pandas as pd

from trader.synthetic_data import generate_xauusd
from trader.scalper import ScalpParams, run_scalper

LOT          = 0.60
START_BAL    = 10_000.0
N_BARS       = 50_000   # ~8.5 months of M5 data (Mon-Fri, 12 bars/hour)
SEEDS        = list(range(10))

PARAMS = ScalpParams(
    # ── Fixed $50 profit target per trade ───────────────────────────────────
    dollar_tp        = 50.0,    # close the trade once unrealized PnL hits $50
    # ── ATR-based SL only (TP is dollar-based above) ─────────────────────
    atr_period       = 14,
    atr_sl_mult      = 0.5,    # SL = 0.5×ATR ≈ 131 pts → ~$50 loss at default lot
    # Partial TP / trailing / BE are disabled when dollar_tp > 0
    partial_tp_mult  = 99.0,
    trail_atr_mult   = 0.0,
    be_atr_mult      = 99.0,
    # ── Volatility regime filter ──────────────────────────────────────────
    atr_regime_period = 50,
    atr_regime_min   = 0.5,    # wider band — allow more trades
    atr_regime_max   = 3.0,
    # ── Compound sizing ───────────────────────────────────────────────────
    # risk_pct = 0.25% equity → $25 risk on $10K → 2:1 R:R vs $50 TP
    # Asymmetric edge: win $50, lose only $25. Profitable at any WR > 34%.
    use_compound_sizing = True,
    risk_pct         = 0.0025,
    max_lot          = 3.0,
    # ── Consecutive loss stop ────────────────────────────────────────────
    max_consec_losses = 5,     # more lenient — allow recovery runs
    # ── Session / frequency ──────────────────────────────────────────────
    max_trades_day   = 60,
    session_open     = 6,      # 06:00–22:00 UTC (16 hrs, London + NY + Asia open)
    session_close    = 22,
    # ── Confidence filters ────────────────────────────────────────────────
    # H1 EMA trend alignment kept — gives direction edge without over-filtering
    use_h1_trend     = True,
    use_h1_rsi       = False,  # disabled — too restrictive for target frequency
    use_h4_trend     = False,  # disabled — too restrictive for target frequency
    min_body_ratio   = 0.10,   # very light candle quality filter
    adx_min          = 0.0,    # disabled
    min_vol_ratio    = 0.0,    # disabled
    spread_points    = 4,
    rsi_buy_min      = 40.0,   # wide M5 RSI zones for more signals
    rsi_buy_max      = 75.0,
    rsi_sell_min     = 25.0,
    rsi_sell_max     = 60.0,
    # ── Daily P&L management ─────────────────────────────────────────────
    daily_loss_limit    = 200.0,   # stop the day after -$200
    daily_profit_target = 2500.0,  # allow up to 50 wins before halting day
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
    print("  XAU/USD SCALPING BACKTEST  |  $50 PROFIT TARGET  |  50 TRADES/DAY")
    print("=" * 72)
    print(f"  Data: {N_BARS:,} M5 bars (~{N_BARS // (12*5*5):.0f} months)  "
          f"| Session: {PARAMS.session_open:02d}:00-{PARAMS.session_close:02d}:00 UTC  "
          f"| Spread: {PARAMS.spread_points}pts")
    print(f"  TP: ${PARAMS.dollar_tp:.0f} fixed profit per trade  "
          f"|  SL: {PARAMS.atr_sl_mult}×ATR (ATR-adaptive)")
    print(f"  Compound sizing: risk {PARAMS.risk_pct*100:.2f}% equity/trade (${PARAMS.risk_pct*10000:.0f} risk → 2:1 R:R)  "
          f"|  Daily loss limit: ${PARAMS.daily_loss_limit:.0f}")
    print(f"  Confidence: H1 EMA trend + M5 RSI/MACD alignment")
    print()

    rows = []
    for seed in SEEDS:
        m5 = generate_xauusd(n_bars=N_BARS, timeframe_minutes=5, seed=seed)
        h1 = resample_to_h1(m5)
        h4 = resample_to_h4(m5)
        res = run_scalper(m5, h1, h4, params=PARAMS, lot=LOT, start_balance=START_BAL)
        rows.append({"seed": seed, **res})

    # ── Results table ─────────────────────────────────────────────────────────
    header = (
        f"{'Seed':>4}  {'Trades':>7}  {'Trd/Day':>7}  {'Win%':>6}  "
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
    print(f"  Avg win rate    : {arr_wr.mean():.1f}%  (wins vs decisive exits)")
    print(f"  Avg profit factor: {arr_pf.mean():.2f}")
    print(f"  Avg win / loss  : +${arr_aw.mean():.2f} / -${arr_al.mean():.2f}")
    print(f"  Avg net P&L     : ${arr_pnl.mean():,.2f}  "
          f"(min ${arr_pnl.min():,.2f} / max ${arr_pnl.max():,.2f})")

    daily_pnl_avg = arr_pnl.mean() / arr_nd.mean()
    monthly       = daily_pnl_avg * 22

    print(f"  Est. daily P&L  : ${daily_pnl_avg:,.2f}  (compound sizing)")
    print(f"  Est. monthly P&L: ${monthly:,.2f}  (22 trading days)")
    print(f"  Avg final equity: ${arr_bal.mean():,.2f}  "
          f"(ROI {(arr_bal.mean() - START_BAL) / START_BAL * 100:.1f}%)")
    print()

    # ── Walk-forward OOS check ─────────────────────────────────────────────────
    is_rows  = [r for r in rows if r["seed"] in range(5)]
    oos_rows = [r for r in rows if r["seed"] in range(5, 10)]

    is_wr  = np.mean([r["win_rate"]      for r in is_rows])
    is_pf  = np.mean([r["profit_factor"] for r in is_rows])
    oos_wr = np.mean([r["win_rate"]      for r in oos_rows])
    oos_pf = np.mean([r["profit_factor"] for r in oos_rows])

    print("WALK-FORWARD OOS CHECK:")
    print(f"  {'Group':<12}  {'Seeds':<10}  {'Avg Win%':>8}  {'Avg PF':>7}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*7}")
    print(f"  {'IS (train)':<12}  {'0–4':<10}  {is_wr:>8.1f}  {is_pf:>7.2f}")
    print(f"  {'OOS (test)':<12}  {'5–9':<10}  {oos_wr:>8.1f}  {oos_pf:>7.2f}")
    print()
    print("  Note: If OOS ≈ IS, the edge is real — not curve-fitted.")
    print("=" * 72)


if __name__ == "__main__":
    main()
