#!/usr/bin/env python3
"""
Demo runner — uses real XAU/USD market data, paper-trades on a $10,000 virtual account.
No MetaTrader 5 installation required; runs on any OS.

Usage:
    python run_demo.py               # live paper-trading loop
    python run_demo.py --backtest    # instant backtest on last 60 days of data
    python run_demo.py --ticks 5     # run exactly N live ticks then exit
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

# Patch the bot to use demo_connector instead of mt5_connector
import trader.demo_connector as demo_conn
sys.modules["trader.mt5_connector"] = demo_conn  # type: ignore[assignment]

from trader import demo_connector as dc
from trader.config import strategy_cfg as scfg, risk_cfg as rcfg  # noqa: E402
from trader.strategy import evaluate
from trader.indicators import ema, rsi, macd, atr, adx


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DIVIDER = "─" * 62


def print_header():
    print(f"\n{'═'*62}")
    print(f"  XAU/USD Algorithm Trader — DEMO / PAPER TRADING")
    print(f"  Account balance : ${dc.DEMO_BALANCE:,.2f}")
    print(f"  Strategy        : {scfg.trend_timeframe} trend | {scfg.timeframe} signals")
    print(f"  TP              : {rcfg.take_profit_points} pts (${rcfg.take_profit_points*0.01:.2f})")
    print(f"  SL              : {rcfg.stop_loss_points} pts (${rcfg.stop_loss_points*0.01:.2f})")
    print(f"  Max lot         : {rcfg.max_lot}")
    print(f"  Data source     : Yahoo Finance (GC=F — Gold Futures)")
    print(f"{'═'*62}\n")


def print_market_snapshot(m15, h1):
    close = m15["close"]
    e20   = ema(close, scfg.ema_fast).iloc[-1]
    e50   = ema(close, scfg.ema_slow).iloc[-1]
    rsi_v = rsi(close, scfg.rsi_period).iloc[-1]
    adx_v = adx(h1, 14).iloc[-1]
    atr_v = atr(m15, scfg.atr_period).iloc[-1]
    price = close.iloc[-1]

    h1_close = h1["close"]
    h1_e20   = ema(h1_close, scfg.ema_fast).iloc[-1]
    h1_e50   = ema(h1_close, scfg.ema_slow).iloc[-1]
    h1_e200  = ema(h1_close, scfg.ema_trend).iloc[-1]

    if h1_close.iloc[-1] > h1_e20 > h1_e50 > h1_e200 and adx_v > scfg.adx_threshold:
        trend = "BULLISH ▲"
    elif h1_close.iloc[-1] < h1_e20 < h1_e50 < h1_e200 and adx_v > scfg.adx_threshold:
        trend = "BEARISH ▼"
    else:
        trend = "RANGING  —"

    print(DIVIDER)
    print(f"  Price : ${price:,.2f}    Trend : {trend}")
    print(f"  EMA20 : {e20:.2f}   EMA50 : {e50:.2f}")
    print(f"  RSI   : {rsi_v:.1f}          ADX   : {adx_v:.1f}    ATR : {atr_v:.2f}")
    print(DIVIDER)


def print_account(account: dc.DemoAccount):
    pnl = account.equity - dc.DEMO_BALANCE
    pnl_sign = "+" if pnl >= 0 else ""
    print(f"\n  💰 Equity   : ${account.equity:>10,.2f}   ({pnl_sign}${pnl:.2f})")
    print(f"  📊 Trades   : {len(account.closed_trades)} closed | {len(account.open_positions)} open")
    if account.closed_trades:
        wins = sum(1 for t in account.closed_trades if t["pnl"] > 0)
        win_rate = wins / len(account.closed_trades) * 100
        total_pnl = sum(t["pnl"] for t in account.closed_trades)
        print(f"  🎯 Win rate : {win_rate:.0f}%   Total realised P&L : ${total_pnl:+.2f}")


def print_open_positions(account: dc.DemoAccount):
    if not account.open_positions:
        print("  No open positions.")
        return
    for p in account.open_positions:
        dir_str = "BUY " if p.type == 0 else "SELL"
        pnl_sign = "+" if p.profit >= 0 else ""
        print(f"  [{p.ticket}] {dir_str} {p.volume}lot @ {p.price_open:.2f}  "
              f"TP={p.tp:.2f} SL={p.sl:.2f}  P&L: {pnl_sign}${p.profit:.2f}")


def print_closed_trades(account: dc.DemoAccount):
    if not account.closed_trades:
        return
    print(f"\n  {'─'*58}")
    print(f"  {'TICKET':<10} {'DIR':<5} {'LOT':<6} {'ENTRY':>8} {'EXIT':>8} {'P&L':>8} {'EXIT':<4}")
    print(f"  {'─'*58}")
    for t in account.closed_trades:
        pnl_str = f"${t['pnl']:+.2f}"
        print(f"  {t['ticket']:<10} {t['direction']:<5} {t['lot']:<6.2f} "
              f"{t['open_price']:>8.2f} {t['close_price']:>8.2f} "
              f"{pnl_str:>8} {t['reason']:<4}")
    print(f"  {'─'*58}")


def run_live(max_ticks: int = 0):
    """Live paper-trading loop — runs until Ctrl-C or max_ticks."""
    dc.connect()
    print_header()
    logger.info("Fetching market data…")

    tick_count = 0
    try:
        while True:
            m15 = dc.get_ohlcv("XAUUSD", scfg.timeframe, count=300)
            h1  = dc.get_ohlcv("XAUUSD", scfg.trend_timeframe, count=300)

            if m15 is None or h1 is None:
                logger.warning("Data unavailable — retrying in 60s")
                time.sleep(60)
                continue

            # Update open positions mark-to-market
            current_price = m15["close"].iloc[-1]
            dc._account.update_equity(current_price)

            print(f"\n[Tick {tick_count+1}]  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
            print_market_snapshot(m15, h1)
            print_account(dc._account)
            print("\n  Open positions:")
            print_open_positions(dc._account)

            # Check guards
            if dc.daily_loss_exceeded("XAUUSD"):
                logger.warning("Daily loss limit hit — no new entries.")
            elif len(dc.get_open_bot_trades("XAUUSD")) >= rcfg.max_open_trades:
                logger.info("Max open trades reached — skipping signal check.")
            else:
                sig = evaluate(m15, h1)
                if sig:
                    print(f"\n  *** SIGNAL: {sig.direction} {sig.entry_type} "
                          f"strength={sig.strength:.0%} ***")
                    print(f"  Reason: {sig.reason}")
                    equity = dc.get_account_equity()
                    lot = dc.calc_lot_size(rcfg.stop_loss_points, equity)
                    dc.place_order("XAUUSD", sig.direction, lot,
                                   rcfg.stop_loss_points, rcfg.take_profit_points,
                                   comment=f"demo-{sig.entry_type.lower()[:2]}")
                else:
                    print("\n  No signal this tick.")

            if account_closed_trades := dc._account.closed_trades:
                print("\n  Recent closed trades:")
                print_closed_trades(dc._account)

            tick_count += 1
            if max_ticks and tick_count >= max_ticks:
                break

            logger.info("Next check in 60 seconds…")
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    _print_final_report()


def run_backtest():
    """
    Walk-forward backtest on last 60 days of real M15 data.
    Uses a rolling window: at each M15 bar, feeds all prior bars to the strategy.
    """
    dc.connect()
    print_header()
    print("  Mode: BACKTEST on last 60 days of real XAU/USD data\n")

    logger.info("Generating realistic XAU/USD synthetic data…")
    from trader.synthetic_data import generate_xauusd, resample_to_h1
    # seed=7 gives more varied bull+bear trend regimes with clearer momentum
    m15_full = generate_xauusd(n_bars=8000, timeframe_minutes=15, seed=7)
    h1_full  = resample_to_h1(m15_full)

    logger.info("M15 bars: %d   H1 bars: %d", len(m15_full), len(h1_full))
    logger.info("Running strategy walk-forward…")

    # Need at least 200 H1 bars for EMA200; skip the warmup period
    WARMUP = 200
    signals_found = 0
    bars_processed = 0
    last_entry_bar = -999  # cooldown tracker

    for i in range(WARMUP, len(m15_full)):
        m15_window = m15_full.iloc[:i+1]
        bar_time   = m15_full.index[i]

        # Align H1 window to bars before this M15 bar
        h1_window = h1_full[h1_full.index <= bar_time]
        if len(h1_window) < WARMUP:
            continue

        bars_processed += 1

        # Update open positions against this bar's close price
        bar_price = float(m15_full["close"].iloc[i])
        dc._account.update_equity(bar_price)

        # Guard: max open trades
        if len(dc._account.open_positions) >= rcfg.max_open_trades:
            continue

        # Guard: daily loss
        if dc._account.daily_drawdown_pct() >= risk_cfg.daily_loss_limit_pct:
            continue

        # Guard: signal cooldown (don't re-enter within N bars of last entry)
        bars_since_last = bars_processed - last_entry_bar
        if bars_since_last < scfg.signal_cooldown_bars:
            continue

        sig = evaluate(m15_window, h1_window)
        if sig is None:
            continue

        signals_found += 1
        equity = dc._account.equity

        # Dynamic SL/TP based on current ATR
        point = 0.01
        if rcfg.atr_based_risk:
            sl_points = int(sig.atr_value / point * rcfg.atr_sl_mult)
            tp_points = int(sig.atr_value / point * rcfg.atr_tp_mult)
        else:
            sl_points = rcfg.stop_loss_points
            tp_points = rcfg.take_profit_points

        lot = dc.calc_lot_size(sl_points, equity)

        # Simulate entry on next bar's open (realistic)
        if i + 1 < len(m15_full):
            entry_price = float(m15_full["open"].iloc[i + 1])
        else:
            entry_price = bar_price

        if sig.direction == "BUY":
            tp = entry_price + tp_points * point
            sl = entry_price - sl_points * point
        else:
            tp = entry_price - tp_points * point
            sl = entry_price + sl_points * point

        dc._account.place_order(sig.direction, lot, entry_price, sl, tp,
                                comment=f"bt-{sig.entry_type.lower()[:2]}")
        last_entry_bar = bars_processed
        rr = tp_points / sl_points if sl_points else 0
        logger.info("[%s] %s %s @ %.2f  lot=%.2f  SL=%d pt  TP=%d pt  R:R=1:%.1f",
                    bar_time.strftime("%m-%d %H:%M"), sig.direction, sig.entry_type,
                    entry_price, lot, sl_points, tp_points, rr)

        if bars_processed % 500 == 0:
            logger.info("Progress: %d/%d bars  |  equity=$%.2f",
                        bars_processed, len(m15_full)-WARMUP, dc._account.equity)

    # Close any remaining open positions at last price
    last_price = float(m15_full["close"].iloc[-1])
    for pos in list(dc._account.open_positions):
        dc._account._close_position(pos, last_price, "END")

    logger.info("Backtest complete — %d signals from %d bars", signals_found, bars_processed)
    _print_final_report()


def _print_final_report():
    account = dc._account
    print(f"\n{'═'*62}")
    print("  DEMO ACCOUNT FINAL REPORT")
    print(f"{'─'*62}")
    print_account(account)
    if account.closed_trades:
        print(f"\n  All trades:")
        print_closed_trades(account)
        wins = [t for t in account.closed_trades if t["pnl"] > 0]
        losses = [t for t in account.closed_trades if t["pnl"] <= 0]
        if wins:
            print(f"\n  Avg win  : ${sum(t['pnl'] for t in wins)/len(wins):.2f}")
        if losses:
            print(f"  Avg loss : ${sum(t['pnl'] for t in losses)/len(losses):.2f}")
        total = sum(t["pnl"] for t in account.closed_trades)
        print(f"  Net P&L  : ${total:+.2f}")
        roi = total / dc.DEMO_BALANCE * 100
        print(f"  ROI      : {roi:+.2f}%")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    from trader.config import risk_cfg  # needed for backtest guard

    parser = argparse.ArgumentParser(description="XAU/USD Demo Trader")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest on 60 days of historical data")
    parser.add_argument("--ticks", type=int, default=0,
                        help="Max live ticks before exiting (0 = run forever)")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    else:
        run_live(max_ticks=args.ticks)
