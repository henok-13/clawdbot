#!/usr/bin/env python3
"""
Demo runner — paper-trades on a virtual $10,000 account using real
or synthetic XAU/USD data. Supports single backtest or parameter optimization.

Usage:
    python run_demo.py --backtest              # single run with default params
    python run_demo.py --optimize             # grid-search best params
    python run_demo.py --ticks 5             # live paper-trading (N ticks)
"""

import argparse
import itertools
import logging
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

import trader.demo_connector as dc
sys.modules["trader.mt5_connector"] = dc  # type: ignore[assignment]

from trader.config import strategy_cfg as scfg, risk_cfg as rcfg  # noqa: E402
from trader.synthetic_data import generate_xauusd, resample_to_h1  # noqa: E402
from trader.strategy_v2 import StrategyParams, evaluate, Signal     # noqa: E402

logging.basicConfig(
    level=logging.WARNING,   # suppress per-bar noise during optimization
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIXED_LOT = 0.30          # user-requested lot size
POINT     = 0.01          # 1 point = $0.01 for XAU/USD
PPL       = 100 * POINT   # P&L per lot per point = $1.00

DIVIDER = "─" * 66


def _resample_h4(m15: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd
    return m15.resample("4h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna()


# ── Core backtest engine ──────────────────────────────────────────────────────

class PaperAccount:
    def __init__(self, balance: float = 10_000.0):
        self.start_balance = balance
        self.balance = balance
        self.equity  = balance
        self.trades: list[dict] = []
        self._open: list[dict] = []

    def open_trade(self, direction: str, entry: float, sl: float,
                   tp: float, be_trigger: float, lot: float, bar_idx: int):
        self._open.append({
            "direction": direction, "entry": entry,
            "sl": sl, "tp": tp, "be_trigger": be_trigger,
            "lot": lot, "bar_idx": bar_idx, "be_moved": False,
        })

    def update(self, bar_high: float, bar_low: float, bar_idx: int) -> None:
        """Check each open position for SL/TP/breakeven on the current bar."""
        closed = []
        for pos in self._open:
            direction = pos["direction"]
            entry     = pos["entry"]
            tp        = pos["tp"]
            sl        = pos["sl"]
            lot       = pos["lot"]

            # Breakeven: move SL to entry once floating profit ≥ be_trigger
            if not pos["be_moved"]:
                if direction == "BUY"  and bar_high >= pos["be_trigger"]:
                    pos["sl"] = entry + POINT * 5   # 5 pts above entry
                    pos["be_moved"] = True
                elif direction == "SELL" and bar_low <= pos["be_trigger"]:
                    pos["sl"] = entry - POINT * 5
                    pos["be_moved"] = True

            # Check TP / SL
            result = None
            close_price = None
            if direction == "BUY":
                if bar_high >= tp:
                    result, close_price = "TP", tp
                elif bar_low <= pos["sl"]:
                    result, close_price = "SL", pos["sl"]
            else:
                if bar_low <= tp:
                    result, close_price = "TP", tp
                elif bar_high >= pos["sl"]:
                    result, close_price = "SL", pos["sl"]

            if result:
                pnl_pts = ((close_price - entry) / POINT
                           if direction == "BUY"
                           else (entry - close_price) / POINT)
                pnl = pnl_pts * lot * PPL
                self.balance += pnl
                self.equity   = self.balance
                self.trades.append({
                    "direction": direction, "entry": entry,
                    "exit": close_price, "lot": lot,
                    "pnl": round(pnl, 2), "result": result,
                    "bar_idx": bar_idx,
                })
                closed.append(pos)

        for p in closed:
            self._open.remove(p)

    def close_all(self, price: float, bar_idx: int):
        for pos in list(self._open):
            direction = pos["direction"]
            entry = pos["entry"]
            pnl_pts = ((price - entry) / POINT if direction == "BUY"
                       else (entry - price) / POINT)
            pnl = pnl_pts * pos["lot"] * PPL
            self.balance += pnl
            self.trades.append({
                "direction": direction, "entry": entry,
                "exit": price, "lot": pos["lot"],
                "pnl": round(pnl, 2), "result": "END",
                "bar_idx": bar_idx,
            })
        self._open.clear()
        self.equity = self.balance

    @property
    def n_open(self) -> int:
        return len(self._open)

    def stats(self) -> dict:
        if not self.trades:
            return {"n": 0, "wins": 0, "win_rate": 0.0,
                    "net_pnl": 0.0, "roi": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0}
        wins   = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        gross_win  = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        net = self.balance - self.start_balance
        return {
            "n":             len(self.trades),
            "wins":          len(wins),
            "win_rate":      len(wins) / len(self.trades) * 100,
            "net_pnl":       round(net, 2),
            "roi":           round(net / self.start_balance * 100, 2),
            "avg_win":       round(gross_win / max(len(wins), 1), 2),
            "avg_loss":      round(-gross_loss / max(len(losses), 1), 2),
            "profit_factor": round(gross_win / max(gross_loss, 0.01), 2),
        }


def run_backtest_engine(
    m15_full, h1_full, h4_full,
    params: StrategyParams,
    lot: float = FIXED_LOT,
    verbose: bool = False,
) -> dict:
    """
    Walk-forward backtest. Returns stats dict.
    Uses vectorised bar updates — no per-bar Python loop for indicators
    (only the signal check loop is in Python).
    """
    import logging
    if not verbose:
        logging.disable(logging.INFO)

    account = PaperAccount()
    # Need ~200 H1 bars (EMA200) = ~800 M15 bars; add buffer for H4 EMA50
    WARMUP = 900
    last_entry_bar = -9999
    n_signals = 0

    for i in range(WARMUP, len(m15_full)):
        bar_time  = m15_full.index[i]
        bar_high  = float(m15_full["high"].iloc[i])
        bar_low   = float(m15_full["low"].iloc[i])
        bar_close = float(m15_full["close"].iloc[i])
        bar_open  = float(m15_full["open"].iloc[i])

        # Update open positions first
        account.update(bar_high, bar_low, i)

        # Guards
        if account.n_open >= 1:
            continue
        if (i - last_entry_bar) < params.signal_cooldown_bars:
            continue
        if account.balance < account.start_balance * 0.85:
            break   # daily-style hard stop at -15%

        m15_w = m15_full.iloc[max(0, i - 350): i + 1]
        h1_w  = h1_full[h1_full.index <= bar_time].iloc[-250:]
        h4_w  = h4_full[h4_full.index <= bar_time].iloc[-120:]

        if len(h4_w) < 60 or len(h1_w) < 210:
            continue

        sig = evaluate(m15_w, h1_w, h4_w, params)
        if sig is None:
            continue

        n_signals += 1
        # Entry on next bar open (realistic slippage simulation)
        if i + 1 >= len(m15_full):
            break
        entry = float(m15_full["open"].iloc[i + 1])

        if sig.direction == "BUY":
            sl = entry - sig.sl_pts * POINT
            tp = entry + sig.tp_pts * POINT
            be = entry + sig.be_pts * POINT
        else:
            sl = entry + sig.sl_pts * POINT
            tp = entry - sig.tp_pts * POINT
            be = entry - sig.be_pts * POINT

        account.open_trade(sig.direction, entry, sl, tp, be, lot, i)
        last_entry_bar = i

        if verbose:
            logging.disable(logging.NOTSET)
            logger.warning("[%s] %s %s @ %.2f  SL=%d  TP=%d",
                           bar_time.strftime("%m-%d %H:%M"),
                           sig.direction, sig.entry_type, entry,
                           sig.sl_pts, sig.tp_pts)
            logging.disable(logging.INFO)

    account.close_all(float(m15_full["close"].iloc[-1]), len(m15_full) - 1)

    logging.disable(logging.NOTSET)
    s = account.stats()
    s["n_signals"] = n_signals
    return s


# ── Parameter grid search ─────────────────────────────────────────────────────

def run_optimizer(m15, h1, h4):
    print(f"\n{'═'*66}")
    print("  STRATEGY OPTIMIZER — searching for best parameter set")
    print(f"{'─'*66}")
    print(f"  {'ADX':>3} {'ST':>4} {'BO':>4} {'MOM':>4} "
          f"{'SL':>4} {'TP':>4} "
          f"{'N':>4} {'WIN%':>5} {'P/F':>5} {'NET P&L':>9} {'ROI':>5}")
    print(DIVIDER)

    adx_values    = [20, 25, 28]
    st_mults      = [2.5, 3.0]
    bo_atr_mults  = [0.8, 1.2]
    mom_scores    = [0.5, 1.0]
    sl_mults      = [2.0, 2.5, 3.0]
    tp_mults      = [3.5, 4.5, 5.5]

    results = []
    n_tested = 0

    for adx_v, st_m, bo_m, mom_s, sl_m, tp_m in itertools.product(
        adx_values, st_mults, bo_atr_mults, mom_scores, sl_mults, tp_mults
    ):
        p = StrategyParams(
            adx_min=adx_v,
            st_mult=st_m,
            breakout_atr_mult=bo_m,
            min_momentum_score=mom_s,
            atr_sl_mult=sl_m,
            atr_tp_mult=tp_m,
            atr_be_mult=sl_m * 0.6,
        )
        s = run_backtest_engine(m15, h1, h4, p, verbose=False)
        n_tested += 1

        # Require at least 15 trades for statistical validity
        if s["n"] < 15:
            continue

        score = s["win_rate"] * s["profit_factor"]
        results.append((score, p, s))

        print(f"  {adx_v:>3} {st_m:>4.1f} {bo_m:>4.1f} {mom_s:>4.1f} "
              f"{sl_m:>4.1f} {tp_m:>4.1f} "
              f"{s['n']:>4} {s['win_rate']:>5.1f} {s['profit_factor']:>5.2f} "
              f"${s['net_pnl']:>9.2f} {s['roi']:>5.1f}%")

    print(f"\n  [{n_tested} configs tested, {len(results)} with ≥15 trades]")

    if not results:
        print("  No valid parameter sets found (need ≥10 trades).")
        return StrategyParams()

    results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_params, best_stats = results[0]

    print(f"\n{'═'*66}")
    print("  BEST PARAMETER SET")
    print(f"{'─'*66}")
    print(f"  ADX min       : {best_params.adx_min}")
    print(f"  Supertrend ×  : {best_params.st_mult}")
    print(f"  Breakout ATR× : {best_params.breakout_atr_mult}")
    print(f"  SL ATR×       : {best_params.atr_sl_mult}")
    print(f"  TP ATR×       : {best_params.atr_tp_mult}")
    print(f"{'─'*66}")
    _print_stats(best_stats)
    print(f"{'═'*66}\n")

    return best_params


# ── Single backtest ───────────────────────────────────────────────────────────

def _print_stats(s: dict):
    print(f"  Trades        : {s['n']}  ({s['wins']} wins)")
    print(f"  Win rate      : {s['win_rate']:.1f}%")
    print(f"  Profit factor : {s['profit_factor']:.2f}")
    print(f"  Avg win       : ${s['avg_win']:.2f}")
    print(f"  Avg loss      : ${s['avg_loss']:.2f}")
    print(f"  Net P&L       : ${s['net_pnl']:+.2f}")
    print(f"  ROI           : {s['roi']:+.2f}%")
    print(f"  Final equity  : ${10_000 + s['net_pnl']:,.2f}")


def run_single(m15, h1, h4, params: StrategyParams, verbose: bool = True):
    print(f"\n{'═'*66}")
    print("  XAU/USD DEMO — BACKTEST (Strategy v2, 0.30 lot)")
    print(f"  M15 bars: {len(m15)}   H1 bars: {len(h1)}   H4 bars: {len(h4)}")
    print(f"  Session : {params.session_open}:00–{params.session_close}:00 UTC  "
          f"(London + New York)")
    print(f"  SL: {params.atr_sl_mult}×ATR   TP: {params.atr_tp_mult}×ATR   "
          f"R:R=1:{params.atr_tp_mult/params.atr_sl_mult:.1f}")
    print(f"{'─'*66}")

    logging.getLogger().setLevel(logging.INFO)
    s = run_backtest_engine(m15, h1, h4, params, lot=FIXED_LOT, verbose=verbose)
    logging.getLogger().setLevel(logging.WARNING)

    print(f"\n{'═'*66}")
    print("  FINAL REPORT")
    print(f"{'─'*66}")
    _print_stats(s)
    print(f"{'═'*66}\n")
    return s


# ── Live paper-trading loop ───────────────────────────────────────────────────

def run_live(params: StrategyParams, max_ticks: int = 0):
    logging.getLogger().setLevel(logging.INFO)
    logger.warning("Starting live paper-trading loop…")
    dc.connect()
    tick = 0
    last_entry_bar = -9999
    account = PaperAccount()
    bar_i = 0

    try:
        while True:
            m15 = dc.get_ohlcv("XAUUSD", "M15", count=350)
            h1  = dc.get_ohlcv("XAUUSD", "H1",  count=250)
            if m15 is None or h1 is None:
                time.sleep(60); continue

            h4 = _resample_h4(m15.iloc[-500:] if len(m15) >= 500 else m15)

            price = float(m15["close"].iloc[-1])
            account.update(float(m15["high"].iloc[-1]),
                           float(m15["low"].iloc[-1]), bar_i)

            s = account.stats()
            print(f"\n[Tick {tick+1}] {datetime.now(timezone.utc).strftime('%H:%M UTC')}  "
                  f"XAU/USD ${price:,.2f}  |  "
                  f"Equity ${10_000+s['net_pnl']:,.2f}  P&L ${s['net_pnl']:+.2f}  "
                  f"Trades {s['n']} (WR {s['win_rate']:.0f}%)")

            if account.n_open < 1 and (bar_i - last_entry_bar) >= params.signal_cooldown_bars:
                sig = evaluate(m15, h1, h4, params)
                if sig:
                    entry = price
                    sl = entry - sig.sl_pts * POINT if sig.direction == "BUY" else entry + sig.sl_pts * POINT
                    tp = entry + sig.tp_pts * POINT if sig.direction == "BUY" else entry - sig.tp_pts * POINT
                    be = entry + sig.be_pts * POINT if sig.direction == "BUY" else entry - sig.be_pts * POINT
                    account.open_trade(sig.direction, entry, sl, tp, be, FIXED_LOT, bar_i)
                    last_entry_bar = bar_i
                    print(f"  *** {sig.direction} {sig.entry_type} opened  "
                          f"SL={sig.sl_pts}pt TP={sig.tp_pts}pt ***")
                    print(f"  {sig.reason}")
                else:
                    print("  No signal.")

            bar_i += 1
            tick += 1
            if max_ticks and tick >= max_ticks:
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    account.close_all(float(m15["close"].iloc[-1]), bar_i)
    s = account.stats()
    print(f"\n{'═'*66}")
    print("  LIVE SESSION REPORT")
    print(f"{'─'*66}")
    _print_stats(s)
    print(f"{'═'*66}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAU/USD Demo Trader v2")
    parser.add_argument("--backtest",  action="store_true")
    parser.add_argument("--optimize",  action="store_true")
    parser.add_argument("--ticks",     type=int, default=0)
    parser.add_argument("--seed",      type=int, default=7)
    parser.add_argument("--bars",      type=int, default=10000)
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    print(f"\n  Generating {args.bars} bars of synthetic XAU/USD data (seed={args.seed})…")
    m15_full = generate_xauusd(n_bars=args.bars, timeframe_minutes=15, seed=args.seed)
    h1_full  = resample_to_h1(m15_full)
    h4_full  = _resample_h4(m15_full)
    print(f"  M15: {len(m15_full)}  H1: {len(h1_full)}  H4: {len(h4_full)}")

    if args.optimize:
        best_params = run_optimizer(m15_full, h1_full, h4_full)
        print("  Running final backtest with best params…")
        run_single(m15_full, h1_full, h4_full, best_params, verbose=args.verbose)

    elif args.backtest:
        run_single(m15_full, h1_full, h4_full, StrategyParams(), verbose=args.verbose)

    else:
        run_live(StrategyParams(), max_ticks=args.ticks)
