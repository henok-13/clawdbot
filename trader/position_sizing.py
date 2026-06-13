"""
ATR-based position sizing — calculates the correct lot size so that
the SL distance represents exactly risk_pct% of account equity.

XAU/USD contract spec (standard lot):
  1 standard lot = 100 troy oz
  1 point = $0.01 price move
  P&L per lot per point = $1.00

Usage:
    lot = calculate_lot(equity=10_000, risk_pct=1.0, sl_points=1250)
    # → 0.08 lots (risks $100 on a $10,000 account with 1250-pt SL)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PPL       = 1.00   # $ P&L per lot per point
POINT     = 0.01   # 1 point = $0.01
MIN_LOT   = 0.01
MAX_LOT   = 100.0


def sl_to_points(sl_price_distance: float) -> float:
    """Convert price-unit SL distance to points (e.g. $5.00 → 500 pts)."""
    return sl_price_distance / POINT


def calculate_lot(
    equity: float,
    risk_pct: float,
    sl_points: float,
    min_lot: float = MIN_LOT,
    max_lot: float = MAX_LOT,
) -> float:
    """
    Returns lot size so that hitting SL costs exactly risk_pct% of equity.

    Args:
        equity    : current account balance in USD
        risk_pct  : fraction of equity to risk per trade (e.g. 1.0 = 1%)
        sl_points : stop-loss distance in points (e.g. 1250 for 2.5×$5 ATR)
        min_lot   : broker minimum lot size
        max_lot   : hard upper cap
    """
    if sl_points <= 0 or equity <= 0 or risk_pct <= 0:
        return min_lot

    risk_usd = equity * (risk_pct / 100.0)
    raw_lot  = risk_usd / (sl_points * PPL)
    lot      = round(max(min_lot, min(raw_lot, max_lot)), 2)

    logger.debug(
        "PositionSize: equity=$%.2f risk=%.1f%% ($%.2f) SL=%d pts → %.2f lots",
        equity, risk_pct, risk_usd, sl_points, lot,
    )
    return lot


def daily_target_lot(
    daily_target_usd: float,
    win_rate: float,
    avg_tp_points: float,
    avg_sl_points: float,
    risk_pct: float,
    equity: float,
) -> dict:
    """
    Show required lot and capital for a given daily profit target.
    Uses strategy statistics to project daily expected P&L.

    Returns a dict with sizing info and capital requirements.
    """
    # Expected value per trade (in points)
    ev_pts = win_rate * avg_tp_points - (1 - win_rate) * avg_sl_points

    # Lot required so that one avg winning trade = daily_target
    # (conservative: assume 1 trade/day)
    lot_for_1trade = daily_target_usd / (win_rate * avg_tp_points * PPL) if ev_pts > 0 else 0

    # Capital required at risk_pct per trade for that lot size
    sl_risk_per_lot = avg_sl_points * PPL
    capital_needed  = (lot_for_1trade * sl_risk_per_lot) / (risk_pct / 100)

    # Conservative lot from proper risk management
    safe_lot = calculate_lot(equity, risk_pct, avg_sl_points)
    safe_daily = safe_lot * ev_pts * PPL * 0.176  # 0.176 = avg trades/day

    return {
        "daily_target_usd":    daily_target_usd,
        "ev_per_trade_usd":    round(ev_pts * lot_for_1trade * PPL, 2),
        "lot_for_target":      round(lot_for_1trade, 2),
        "capital_needed_usd":  round(capital_needed, 2),
        "safe_lot":            safe_lot,
        "safe_daily_pnl_est":  round(safe_daily, 2),
        "win_rate_pct":        round(win_rate * 100, 1),
    }


def print_sizing_table(equity: float, risk_pct: float = 1.0):
    """Print position sizing for common SL distances."""
    print(f"\n  Position Sizing  |  Equity: ${equity:,.0f}  |  Risk: {risk_pct}% per trade")
    print(f"  {'SL (pts)':>8} {'SL ($)':>8} {'Risk ($)':>9} {'Lot':>6} {'TP (pts)':>9} {'TP ($)':>8}")
    print("  " + "─" * 56)
    for atr in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        sl_pts = int(2.5 * atr / POINT)
        tp_pts = int(5.5 * atr / POINT)
        lot      = calculate_lot(equity, risk_pct, sl_pts)
        risk_usd = sl_pts * lot * PPL
        tp_usd   = tp_pts * lot * PPL
        print(f"  ATR=${atr:.1f}  {sl_pts:>8} {atr*2.5:>8.2f}  ${risk_usd:>8.2f}  {lot:>6.2f}  "
              f"{tp_pts:>9}  ${tp_usd:>7.2f}")
