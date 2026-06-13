"""
High-frequency scalping backtest engine for XAU/USD.

Strategy: Enter on M5 bars with full multi-timeframe confirmation; exit via
ATR-adaptive TP/SL with partial close + trailing stop. Target 15+ trades/day.

Multi-timeframe confirmation stack (4 layers):
  H4  : Supertrend direction must agree
  H1  : EMA20>EMA50>EMA100 + RSI zone (>52 bull / <48 bear)
  M5  : ADX ≥ 22, RSI zone, MACD sign, body ≥ 0.38, price vs EMA20
  Risk: ATR-adaptive SL/TP with breakeven + partial TP + trailing stop

XAU/USD constants:
  POINT = 0.01  |  PPL = $1.00 per lot per point
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import ema, rsi, macd, adx, supertrend, candle_body_ratio, volume_ratio, atr

POINT = 0.01
PPL   = 100 * POINT   # $1.00 per lot per point (100 * 0.01 = 1.00)


@dataclass
class ScalpParams:
    # ── ATR-adaptive TP/SL (Fix 1) ───────────────────────────────────────────
    atr_period: int       = 14
    atr_sl_mult: float    = 1.5   # SL = 1.5 × ATR
    atr_tp_mult: float    = 3.5   # full TP = 3.5 × ATR (mostly closed by trail)

    # ── Volatility regime filter (Fix 2) ─────────────────────────────────────
    # Note: atr_regime_max also automatically skips news-spike bars (Fix 6)
    atr_regime_period: int   = 50
    atr_regime_min: float    = 0.7   # skip if ATR < 70% of rolling avg (too quiet)
    atr_regime_max: float    = 2.5   # skip if ATR > 250% of rolling avg (news spike)

    # ── Compound lot sizing (Fix 3) ───────────────────────────────────────────
    use_compound_sizing: bool = True
    risk_pct: float           = 0.005  # risk 0.5% of current equity per trade
    max_lot: float            = 3.0

    # ── Partial TP + trailing stop (Fix 4) ───────────────────────────────────
    partial_pct: float      = 0.50   # close 50% at partial TP
    partial_tp_mult: float  = 2.0    # partial TP at 2.0 × ATR
    trail_atr_mult: float   = 0.8    # trail stop: 0.8 × ATR behind running extreme
    be_atr_mult: float      = 0.5    # slide SL to entry when 0.5 × ATR in profit

    # ── Consecutive loss daily stop (Fix 7) ───────────────────────────────────
    max_consec_losses: int  = 3   # halt rest of day after 3 straight losses

    # ── Existing session / filter params ─────────────────────────────────────
    max_trades_day: int   = 60
    session_open: int     = 8    # London open (UTC)
    session_close: int    = 18   # NY mid-session (UTC) — peak trending hours
    min_body_ratio: float = 0.38   # strong candle required
    use_h1_trend: bool    = True   # H1 EMA alignment gate
    use_h1_rsi: bool      = True   # H1 RSI confirmation gate
    h1_rsi_bull_min: float = 56.0  # H1 RSI must be above this for buys
    h1_rsi_bear_max: float = 44.0  # H1 RSI must be below this for sells
    use_h4_trend: bool    = True   # H4 Supertrend gate
    spread_points: int    = 4
    adx_min: float        = 24.0   # M5 ADX — only trade strong trends
    rsi_buy_min: float    = 52.0   # tighter M5 RSI zones
    rsi_buy_max: float    = 68.0
    rsi_sell_min: float   = 32.0
    rsi_sell_max: float   = 48.0
    min_vol_ratio: float  = 0.0    # volume must be ≥ this ratio (0.0 = disabled)
    daily_loss_limit: float    = 200.0  # stop trading the day once this loss is hit
    daily_profit_target: float = 500.0  # stop trading the day once this profit is hit


def _build_h1_trend(m5: pd.DataFrame, h1: pd.DataFrame) -> np.ndarray:
    """H1 EMA alignment: +1 bull / -1 bear / 0 neutral, forward-filled to M5."""
    e20  = ema(h1["close"], 20)
    e50  = ema(h1["close"], 50)
    e100 = ema(h1["close"], 100)

    h1_dir = pd.Series(0, index=h1.index, dtype=np.int8)
    h1_dir[(e20 > e50) & (e50 > e100)] = 1
    h1_dir[(e20 < e50) & (e50 < e100)] = -1

    return h1_dir.reindex(m5.index, method="ffill").fillna(0).astype(np.int8).values


def _build_h1_rsi(m5: pd.DataFrame, h1: pd.DataFrame) -> np.ndarray:
    """H1 RSI (14) forward-filled to M5 index."""
    h1_rsi = rsi(h1["close"], 14)
    return h1_rsi.reindex(m5.index, method="ffill").fillna(50.0).values


def _build_h4_trend(m5: pd.DataFrame, h4: pd.DataFrame) -> np.ndarray:
    """H4 Supertrend direction (+1 bull / -1 bear) forward-filled to M5."""
    st = supertrend(h4, period=10, multiplier=3.0)
    return st.reindex(m5.index, method="ffill").fillna(1).astype(np.int8).values


def run_scalper(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame | None = None,
    params: ScalpParams | None = None,
    lot: float = 0.60,
    start_balance: float = 10_000.0,
) -> dict:
    """
    Walk-forward scalping backtest.

    Parameters
    ----------
    m5     : M5 OHLCV DataFrame with DatetimeIndex (UTC).
    h1     : H1 OHLCV DataFrame with DatetimeIndex (UTC).
    h4     : H4 OHLCV DataFrame (optional — built from m5 if None).
    params : ScalpParams (uses defaults if None).
    lot    : fixed lot size (used when use_compound_sizing=False).
    start_balance : starting account equity.

    Returns
    -------
    dict with keys: n, wins, losses, breakevens, win_rate, net_pnl, roi,
                    avg_win, avg_loss, profit_factor, trades_per_day,
                    n_days, final_balance.
    """
    if params is None:
        params = ScalpParams()

    # Build H4 from M5 if not supplied
    if h4 is None:
        h4 = m5.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

    n = len(m5)

    # ── Pre-compute all indicators (vectorised) ───────────────────────────────
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
    vol_arr  = volume_ratio(m5, 20).values

    # ATR array for adaptive SL/TP (Fix 1) and regime filter (Fix 2)
    atr_raw = atr(m5, params.atr_period).values
    # Rolling average ATR for regime filter
    atr_avg_arr = pd.Series(atr_raw).rolling(params.atr_regime_period).mean().values

    # H1 layers
    h1_trend_arr = _build_h1_trend(m5, h1) if params.use_h1_trend else np.ones(n, dtype=np.int8)
    h1_rsi_arr   = _build_h1_rsi(m5, h1)   if params.use_h1_rsi   else np.full(n, 50.0)

    # H4 layer
    h4_trend_arr = _build_h4_trend(m5, h4) if params.use_h4_trend else np.ones(n, dtype=np.int8)

    hours = m5.index.hour
    in_session = (hours >= params.session_open) & (hours < params.session_close)

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    balance  = start_balance
    trades: list[dict] = []

    in_trade = False

    # Per-trade state (Fix 4: partial TP + trailing)
    t_dir: int = 0
    t_entry: float = 0.0
    t_tp: float = 0.0          # full TP price
    t_partial_tp: float = 0.0  # partial TP price
    t_sl_price: float = 0.0
    t_be_trigger: float = 0.0  # price at which to slide SL to entry
    t_be_set: bool = False
    t_partial_hit: bool = False
    t_trail_dist: float = 0.0  # trailing distance in price units
    t_running_extreme: float = 0.0
    t_lot_initial: float = 0.0  # lot at trade open
    t_lot: float = 0.0          # current lot (reduced after partial close)
    t_accrued_pnl: float = 0.0  # P&L booked from partial close
    t_atr_v: float = 0.0        # ATR at trade entry (for trail/be calcs)
    t_entry_date = None

    daily_counts: dict = {}
    daily_pnl: dict = {}       # date -> running P&L for that day
    daily_consec: dict = {}    # date -> current consecutive loss count (Fix 7)
    warmup = 300               # slightly longer for H1/H4 indicator warmup

    for i in range(warmup, n):
        c = close[i]
        h = high[i]
        l = low[i]
        bar_date = m5.index[i].date()

        # ── Update open position ──────────────────────────────────────────────
        if in_trade:
            spread_cost = params.spread_points * t_lot_initial * PPL  # one-time at open

            # Step 1: Check breakeven trigger (if partial not yet hit)
            if not t_partial_hit and not t_be_set:
                if t_dir == 1 and h >= t_be_trigger:
                    t_sl_price = t_entry
                    t_be_set = True
                elif t_dir == -1 and l <= t_be_trigger:
                    t_sl_price = t_entry
                    t_be_set = True

            # Step 2: Check partial TP (if not yet hit)
            if not t_partial_hit:
                partial_hit = (t_dir == 1 and h >= t_partial_tp) or \
                              (t_dir == -1 and l <= t_partial_tp)
                if partial_hit:
                    # Book partial P&L: price move from entry to partial_tp × partial lot
                    if t_dir == 1:
                        partial_tp_pts = (t_partial_tp - t_entry) / POINT
                    else:
                        partial_tp_pts = (t_entry - t_partial_tp) / POINT
                    partial_lot = t_lot_initial * params.partial_pct
                    t_accrued_pnl = partial_tp_pts * partial_lot * PPL
                    # Switch to trailing: remaining lot, set trailing distance, running extreme
                    t_lot = t_lot_initial * (1.0 - params.partial_pct)
                    t_trail_dist = t_atr_v * params.trail_atr_mult
                    t_running_extreme = t_partial_tp  # extreme starts at partial TP price
                    t_partial_hit = True

            # Step 3: Update trailing stop (after partial hit)
            if t_partial_hit:
                if t_dir == 1:
                    # Update running high; slide SL up with trail
                    if h > t_running_extreme:
                        t_running_extreme = h
                    trail_sl = t_running_extreme - t_trail_dist
                    if trail_sl > t_sl_price:
                        t_sl_price = trail_sl
                else:
                    # Update running low; slide SL down with trail
                    if l < t_running_extreme:
                        t_running_extreme = l
                    trail_sl = t_running_extreme + t_trail_dist
                    if trail_sl < t_sl_price:
                        t_sl_price = trail_sl

            # Step 4: Check full TP (remaining lot)
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

            closed = False

            if tp_hit:
                if t_dir == 1:
                    exit_pts = (t_tp - t_entry) / POINT
                else:
                    exit_pts = (t_entry - t_tp) / POINT
                final_pnl = exit_pts * t_lot * PPL
                total_pnl = t_accrued_pnl + final_pnl - spread_cost
                balance += total_pnl
                daily_pnl[t_entry_date] = daily_pnl.get(t_entry_date, 0.0) + total_pnl
                result = "TP" if not t_partial_hit else "PARTIAL_TRAIL"
                trades.append({"pnl": round(total_pnl, 2), "result": result})
                closed = True

            elif sl_hit:
                # SL hit — check if it's breakeven (SL was slid to entry)
                if t_be_set and not t_partial_hit and abs(t_sl_price - t_entry) < POINT:
                    # Pure breakeven: only spread cost lost
                    total_pnl = -spread_cost
                    result = "BE"
                else:
                    if t_dir == 1:
                        exit_pts = (t_sl_price - t_entry) / POINT
                    else:
                        exit_pts = (t_entry - t_sl_price) / POINT
                    final_pnl = exit_pts * t_lot * PPL
                    total_pnl = t_accrued_pnl + final_pnl - spread_cost
                    result = "TRAIL" if t_partial_hit else "SL"
                balance += total_pnl
                daily_pnl[t_entry_date] = daily_pnl.get(t_entry_date, 0.0) + total_pnl
                trades.append({"pnl": round(total_pnl, 2), "result": result})
                closed = True

            elif eod_close:
                pnl_pts = (c - t_entry) / POINT if t_dir == 1 else (t_entry - c) / POINT
                final_pnl = pnl_pts * t_lot * PPL
                total_pnl = t_accrued_pnl + final_pnl - spread_cost
                balance += total_pnl
                daily_pnl[t_entry_date] = daily_pnl.get(t_entry_date, 0.0) + total_pnl
                trades.append({"pnl": round(total_pnl, 2), "result": "EOD"})
                closed = True

            if closed:
                # Fix 7: Update consecutive loss counter
                if total_pnl <= 0:
                    daily_consec[t_entry_date] = daily_consec.get(t_entry_date, 0) + 1
                else:
                    # Win resets the streak
                    daily_consec[t_entry_date] = 0
                in_trade = False

            if in_trade:
                continue

        # ── Entry conditions ──────────────────────────────────────────────────
        if not in_session[i]:
            continue

        daily_count = daily_counts.get(bar_date, 0)
        if daily_count >= params.max_trades_day:
            continue

        # Daily loss / profit limits — halt entries for the rest of this day
        day_pnl = daily_pnl.get(bar_date, 0.0)
        if day_pnl <= -params.daily_loss_limit:
            continue
        if day_pnl >= params.daily_profit_target:
            continue

        # Fix 7: Consecutive loss daily stop — skip if hit max consecutive losses
        if daily_consec.get(bar_date, 0) >= params.max_consec_losses:
            continue

        # ATR and regime checks
        atr_v = atr_raw[i]
        atr_avg = atr_avg_arr[i]
        if np.isnan(atr_v) or atr_v <= 0:
            continue

        # Fix 2: Volatility regime filter (also handles news spikes via atr_regime_max)
        if not np.isnan(atr_avg) and atr_avg > 0:
            ratio = atr_v / atr_avg
            if ratio < params.atr_regime_min or ratio > params.atr_regime_max:
                continue

        # Layer 1 — H1 EMA trend direction
        h1_d = h1_trend_arr[i]
        if params.use_h1_trend and h1_d == 0:
            continue

        # Layer 2 — H4 Supertrend must agree with H1 trend
        h4_d = h4_trend_arr[i]
        if params.use_h4_trend and h4_d != h1_d:
            continue

        # M5 indicators
        rsi_v = rsi_arr[i]
        mh    = macd_arr[i]
        e20   = e20_arr[i]
        br    = body_arr[i]
        adx_v = adx_arr[i]
        h1_r  = h1_rsi_arr[i]
        vol_v = vol_arr[i]

        if np.isnan(rsi_v) or np.isnan(mh) or np.isnan(e20) or np.isnan(br) or np.isnan(adx_v):
            continue

        # Layer 3 — M5 ADX (trending market)
        if adx_v < params.adx_min:
            continue

        # Layer 3 — candle body quality
        if br < params.min_body_ratio:
            continue

        # Layer 3 — volume confirmation (above-average participation)
        if not np.isnan(vol_v) and params.min_vol_ratio > 0 and vol_v < params.min_vol_ratio:
            continue

        # Layer 2 — H1 RSI confirmation
        if params.use_h1_rsi:
            if h1_d == 1 and h1_r < params.h1_rsi_bull_min:
                continue
            if h1_d == -1 and h1_r > params.h1_rsi_bear_max:
                continue

        d = h1_d  # trade direction follows H1 (and H4-confirmed) trend

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

        # Fix 1: ATR-adaptive SL/TP (in price units)
        sl_price_dist = atr_v * params.atr_sl_mult
        tp_price_dist = atr_v * params.atr_tp_mult
        partial_tp_dist = atr_v * params.partial_tp_mult
        be_trigger_dist = atr_v * params.be_atr_mult

        # Fix 3: Compound lot sizing
        if params.use_compound_sizing:
            sl_pts_num = sl_price_dist / POINT   # SL distance in points count
            risk_usd = balance * params.risk_pct
            t_lot_val = min(risk_usd / max(sl_pts_num * PPL, 0.001), params.max_lot)
            t_lot_val = max(round(t_lot_val, 2), 0.01)
        else:
            t_lot_val = lot

        if d == 1:
            t_tp        = entry + tp_price_dist
            t_sl_price  = entry - sl_price_dist
            t_partial_tp = entry + partial_tp_dist
            t_be_trigger = entry + be_trigger_dist
        else:
            t_tp        = entry - tp_price_dist
            t_sl_price  = entry + sl_price_dist
            t_partial_tp = entry - partial_tp_dist
            t_be_trigger = entry - be_trigger_dist

        in_trade         = True
        t_dir            = d
        t_entry          = entry
        t_be_set         = False
        t_partial_hit    = False
        t_accrued_pnl    = 0.0
        t_lot_initial    = t_lot_val
        t_lot            = t_lot_val
        t_trail_dist     = 0.0
        t_running_extreme = entry
        t_atr_v          = atr_v
        t_entry_date     = m5.index[i + 1].date()

        daily_counts[bar_date] = daily_count + 1

    # Close leftover at last bar
    if in_trade:
        cp = close[-1]
        spread_cost = params.spread_points * t_lot_initial * PPL
        pnl_pts = (cp - t_entry) / POINT if t_dir == 1 else (t_entry - cp) / POINT
        final_pnl = pnl_pts * t_lot * PPL
        total_pnl = t_accrued_pnl + final_pnl - spread_cost
        balance += total_pnl
        trades.append({"pnl": round(total_pnl, 2), "result": "END"})

    # ── Stats ─────────────────────────────────────────────────────────────────
    if not trades:
        return {
            "n": 0, "wins": 0, "losses": 0, "breakevens": 0, "win_rate": 0.0,
            "net_pnl": 0.0, "roi": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "trades_per_day": 0.0, "n_days": 1,
            "final_balance": round(start_balance, 2),
        }

    # Classification: wins = total_pnl > 0, losses = total_pnl <= 0, be = result == "BE"
    wins_list       = [t for t in trades if t["pnl"] > 0]
    losses_list     = [t for t in trades if t["pnl"] <= 0 and t["result"] != "BE"]
    breakevens_list = [t for t in trades if t["result"] == "BE"]

    gw  = sum(t["pnl"] for t in wins_list)
    gl  = abs(sum(t["pnl"] for t in losses_list))
    net = balance - start_balance

    trade_dates = set(daily_counts.keys())
    n_days = max(len(trade_dates), 1)

    decisive = len(wins_list) + len(losses_list)

    return {
        "n":              len(trades),
        "wins":           len(wins_list),
        "losses":         len(losses_list),
        "breakevens":     len(breakevens_list),
        "win_rate":       round(len(wins_list) / max(decisive, 1) * 100, 1),
        "net_pnl":        round(net, 2),
        "roi":            round(net / start_balance * 100, 2),
        "avg_win":        round(gw / max(len(wins_list), 1), 2),
        "avg_loss":       round(gl / max(len(losses_list), 1), 2),   # positive magnitude
        "profit_factor":  round(gw / max(gl, 0.01), 2),
        "trades_per_day": round(len(trades) / n_days, 1),
        "n_days":         n_days,
        "final_balance":  round(balance, 2),
    }
