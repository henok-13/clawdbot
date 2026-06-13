"""
High-frequency scalping backtest engine for XAU/USD.

Strategy: Enter on M5 bars with H1 trend confirmation; exit at
fixed TP (+200 pts) or SL (-135 pts). Target 25+ trades per day.
Lot size: 0.60.

Win-rate boosters applied:
  - ADX ≥ 15 on M5 (trending market only — skips choppy ranging)
  - Tighter RSI zones: 50-68 for buys, 32-50 for sells
  - Stronger candle body filter (≥ 0.30)
  - Breakeven stop: once +60 pts in profit, SL moves to entry

XAU/USD constants (matches rest of codebase):
  POINT = 0.01   (1 point = $0.01 price move)
  PPL   = 1.00   ($1.00 P&L per lot per point)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import ema, rsi, macd, adx, candle_body_ratio

POINT = 0.01
PPL   = 100 * POINT   # $1.00 P&L per lot per point


@dataclass
class ScalpParams:
    tp_points: int        = 200   # take profit in points
    sl_points: int        = 135   # stop loss in points
    max_trades_day: int   = 60    # hard daily cap
    session_open: int     = 7     # UTC hour (inclusive)
    session_close: int    = 21    # UTC hour (exclusive)
    min_body_ratio: float = 0.30  # strong candle body required
    use_h1_trend: bool    = True  # require H1 EMA alignment
    spread_points: int    = 4     # spread cost deducted per trade (points)
    adx_min: float        = 15.0  # M5 ADX threshold — skip ranging markets
    rsi_buy_min: float    = 50.0  # bullish RSI zone lower bound
    rsi_buy_max: float    = 68.0  # bullish RSI zone upper bound
    rsi_sell_min: float   = 32.0  # bearish RSI zone lower bound
    rsi_sell_max: float   = 50.0  # bearish RSI zone upper bound
    be_trigger_points: int = 60   # move SL to entry once this profit (pts) is reached


def _build_h1_trend(m5: pd.DataFrame, h1: pd.DataFrame) -> np.ndarray:
    """
    Compute H1 trend direction (+1 BUY / -1 SELL / 0 neutral) using
    EMA20 > EMA50 > EMA100, then forward-fill to the M5 index.
    """
    h1_e20  = ema(h1["close"], 20)
    h1_e50  = ema(h1["close"], 50)
    h1_e100 = ema(h1["close"], 100)

    bull = (h1_e20 > h1_e50) & (h1_e50 > h1_e100)
    bear = (h1_e20 < h1_e50) & (h1_e50 < h1_e100)

    h1_dir = pd.Series(0, index=h1.index, dtype=np.int8)
    h1_dir[bull] = 1
    h1_dir[bear] = -1

    # Forward-fill H1 values onto M5 timestamps — no look-ahead bias
    m5_dir = h1_dir.reindex(m5.index, method="ffill").fillna(0).astype(np.int8)
    return m5_dir.values


def run_scalper(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    params: ScalpParams | None = None,
    lot: float = 0.60,
    start_balance: float = 10_000.0,
) -> dict:
    """
    Walk-forward scalping backtest.

    Parameters
    ----------
    m5 : M5 OHLCV DataFrame with DatetimeIndex (UTC).
    h1 : H1 OHLCV DataFrame with DatetimeIndex (UTC).
    params : ScalpParams (uses defaults if None).
    lot : fixed lot size.
    start_balance : starting account equity.

    Returns
    -------
    dict with keys: n, wins, losses, breakevens, win_rate, net_pnl, roi,
                    avg_win, avg_loss, profit_factor,
                    trades_per_day, final_balance.
    """
    if params is None:
        params = ScalpParams()

    n = len(m5)

    # ── Pre-compute indicators (vectorised) ───────────────────────────────────
    close  = m5["close"].values
    high   = m5["high"].values
    low    = m5["low"].values
    open_  = m5["open"].values

    rsi_arr  = rsi(m5["close"], 14).values
    _, _, macd_hist = macd(m5["close"])
    macd_arr = macd_hist.values
    e20_arr  = ema(m5["close"], 20).values
    body_arr = candle_body_ratio(m5).values
    adx_arr  = adx(m5, 14).values

    # H1 trend mapped to M5 bars
    if params.use_h1_trend:
        trend_arr = _build_h1_trend(m5, h1)
    else:
        trend_arr = np.zeros(n, dtype=np.int8)

    # Session mask
    hours = m5.index.hour
    in_session = ((hours >= params.session_open) & (hours < params.session_close))

    tp_price = params.tp_points * POINT
    sl_price = params.sl_points * POINT
    be_price = params.be_trigger_points * POINT
    spread_cost = params.spread_points * lot * PPL

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    balance  = start_balance
    trades: list[dict] = []

    in_trade      = False
    t_dir         = 0
    t_entry       = 0.0
    t_tp          = 0.0
    t_sl_price    = 0.0
    t_be_set      = False
    t_entry_date  = None

    daily_counts: dict = {}
    warmup = 200

    for i in range(warmup, n):
        c = close[i]
        h = high[i]
        l = low[i]
        bar_date = m5.index[i].date()

        # ── Update open position ──────────────────────────────────────────────
        if in_trade:
            # Slide SL to entry once trade is be_trigger_points in profit
            if not t_be_set:
                if t_dir == 1 and h >= t_entry + be_price:
                    t_sl_price = t_entry
                    t_be_set   = True
                elif t_dir == -1 and l <= t_entry - be_price:
                    t_sl_price = t_entry
                    t_be_set   = True

            if t_dir == 1:
                tp_hit = h >= t_tp
                sl_hit = l <= t_sl_price
            else:
                tp_hit = l <= t_tp
                sl_hit = h >= t_sl_price

            eod_close = (
                not in_session[i]
                and m5.index[i].hour >= params.session_close
                and t_entry_date == bar_date
            )

            if tp_hit:
                pnl = params.tp_points * lot * PPL - spread_cost
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": "TP"})
                in_trade = False

            elif sl_hit:
                if t_be_set:
                    pnl = -spread_cost          # only lose the spread
                    result = "BE"
                else:
                    pnl = -params.sl_points * lot * PPL - spread_cost
                    result = "SL"
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": result})
                in_trade = False

            elif eod_close:
                pnl_pts = (c - t_entry) / POINT if t_dir == 1 else (t_entry - c) / POINT
                pnl = pnl_pts * lot * PPL - spread_cost
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "result": "EOD"})
                in_trade = False

            if in_trade:
                continue

        # ── Check entry conditions ────────────────────────────────────────────
        if not in_session[i]:
            continue

        daily_count = daily_counts.get(bar_date, 0)
        if daily_count >= params.max_trades_day:
            continue

        d = trend_arr[i]
        if params.use_h1_trend and d == 0:
            continue

        rsi_v = rsi_arr[i]
        mh    = macd_arr[i]
        e20   = e20_arr[i]
        br    = body_arr[i]
        adx_v = adx_arr[i]

        if np.isnan(rsi_v) or np.isnan(mh) or np.isnan(e20) or np.isnan(br) or np.isnan(adx_v):
            continue

        # Skip ranging/choppy market bars
        if adx_v < params.adx_min:
            continue

        # Strong candle required
        if br < params.min_body_ratio:
            continue

        # Direction when H1 trend disabled
        if not params.use_h1_trend:
            if mh > 0 and c > e20:
                d = 1
            elif mh < 0 and c < e20:
                d = -1
            else:
                continue

        if d == 1:
            rsi_ok  = params.rsi_buy_min  <= rsi_v <= params.rsi_buy_max
            macd_ok = mh > 0
            ema_ok  = c > e20
        else:
            rsi_ok  = params.rsi_sell_min <= rsi_v <= params.rsi_sell_max
            macd_ok = mh < 0
            ema_ok  = c < e20

        if not (rsi_ok and macd_ok and ema_ok):
            continue

        if i + 1 >= n:
            break

        entry = open_[i + 1]
        if entry <= 0:
            continue

        if d == 1:
            t_tp       = entry + tp_price
            t_sl_price = entry - sl_price
        else:
            t_tp       = entry - tp_price
            t_sl_price = entry + sl_price

        in_trade     = True
        t_dir        = d
        t_entry      = entry
        t_be_set     = False
        t_entry_date = m5.index[i + 1].date()

        daily_counts[bar_date] = daily_count + 1

    # Close any leftover position at last close
    if in_trade:
        cp = close[-1]
        pnl_pts = (cp - t_entry) / POINT if t_dir == 1 else (t_entry - cp) / POINT
        pnl = pnl_pts * lot * PPL - spread_cost
        balance += pnl
        trades.append({"pnl": round(pnl, 2), "result": "END"})

    # ── Stats ─────────────────────────────────────────────────────────────────
    if not trades:
        return {
            "n": 0, "wins": 0, "losses": 0, "breakevens": 0, "win_rate": 0.0,
            "net_pnl": 0.0, "roi": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "trades_per_day": 0.0,
            "final_balance": round(start_balance, 2),
        }

    wins       = [t for t in trades if t["result"] == "TP"]
    breakevens = [t for t in trades if t["result"] == "BE"]
    losses     = [t for t in trades if t["result"] in ("SL", "EOD", "END") and t["pnl"] < 0]

    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    net = balance - start_balance

    trade_dates = set(daily_counts.keys())
    n_days = max(len(trade_dates), 1)

    # Win rate counts only TP as wins; BE exits not counted as losses
    decisive = len(wins) + len(losses)

    return {
        "n":              len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "breakevens":     len(breakevens),
        "win_rate":       round(len(wins) / max(decisive, 1) * 100, 1),
        "net_pnl":        round(net, 2),
        "roi":            round(net / start_balance * 100, 2),
        "avg_win":        round(gw / max(len(wins), 1), 2),
        "avg_loss":       round(gl / max(len(losses), 1), 2),   # positive magnitude
        "profit_factor":  round(gw / max(gl, 0.01), 2),
        "trades_per_day": round(len(trades) / n_days, 1),
        "n_days":         n_days,
        "final_balance":  round(balance, 2),
    }
