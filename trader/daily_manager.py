"""
Daily P&L Manager — controls when the bot trades each day.

Rules:
  - Stop trading for the day once daily_profit_target is reached
  - Stop trading for the day once daily_loss_limit is hit (capital protection)
  - Reset automatically at the start of each trading day
  - Log every trade and daily summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DailyManagerConfig:
    daily_profit_target: float  = 500.0   # stop trading once +$500 hit today
    daily_loss_limit: float     = 200.0   # stop trading once -$200 hit today
    max_trades_per_day: int     = 3        # hard cap — avoids overtrading on volatile days
    notify_on_target: bool      = True


class DailyManager:
    def __init__(self, config: Optional[DailyManagerConfig] = None):
        self.cfg = config or DailyManagerConfig()
        self._today: Optional[date] = None
        self._daily_pnl: float = 0.0
        self._trades_today: int = 0
        self._stopped: bool = False
        self._stop_reason: str = ""
        self._trade_log: list[dict] = []

    # ── State management ──────────────────────────────────────────────────────

    def _check_day_rollover(self) -> None:
        today = datetime.now(timezone.utc).date()
        if self._today != today:
            if self._today is not None:
                self._log_daily_summary()
            self._today = today
            self._daily_pnl = 0.0
            self._trades_today = 0
            self._stopped = False
            self._stop_reason = ""
            logger.info("[DailyManager] New trading day: %s | Target +$%.0f | Limit -$%.0f",
                        today, self.cfg.daily_profit_target, self.cfg.daily_loss_limit)

    def should_trade(self) -> bool:
        """Return True if the bot is allowed to open a new trade right now."""
        self._check_day_rollover()
        if self._stopped:
            logger.debug("[DailyManager] Blocked: %s", self._stop_reason)
            return False
        return True

    def record_trade(self, pnl: float, direction: str, entry: float,
                     result: str = "") -> None:
        """Call this after every trade closes."""
        self._check_day_rollover()
        self._daily_pnl += pnl
        self._trades_today += 1
        self._trade_log.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "direction": direction,
            "entry": entry,
            "pnl": round(pnl, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "result": result,
        })

        emoji = "✅" if pnl > 0 else "❌"
        logger.info("[DailyManager] Trade %d %s  P&L: $%+.2f  Daily: $%+.2f",
                    self._trades_today, emoji, pnl, self._daily_pnl)

        # Check stop conditions
        if self._daily_pnl >= self.cfg.daily_profit_target:
            self._stopped = True
            self._stop_reason = f"daily profit target hit (${self._daily_pnl:+.2f})"
            logger.info("[DailyManager] 🎯 PROFIT TARGET HIT — done for today. Daily P&L: $%+.2f",
                        self._daily_pnl)

        elif self._daily_pnl <= -self.cfg.daily_loss_limit:
            self._stopped = True
            self._stop_reason = f"daily loss limit hit (${self._daily_pnl:+.2f})"
            logger.warning("[DailyManager] 🛑 LOSS LIMIT HIT — stopping for today. Daily P&L: $%+.2f",
                           self._daily_pnl)

        elif self._trades_today >= self.cfg.max_trades_per_day:
            self._stopped = True
            self._stop_reason = f"max trades/day reached ({self._trades_today})"
            logger.info("[DailyManager] Max trades/day reached — done for today.")

    def _log_daily_summary(self) -> None:
        wins   = sum(1 for t in self._trade_log if t["pnl"] > 0)
        losses = self._trades_today - wins
        logger.info(
            "[DailyManager] ── Day summary %s ── "
            "Trades: %d | Wins: %d | Losses: %d | P&L: $%+.2f",
            self._today, self._trades_today, wins, losses, self._daily_pnl,
        )
        self._trade_log.clear()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def daily_pnl(self) -> float:
        self._check_day_rollover()
        return self._daily_pnl

    @property
    def trades_today(self) -> int:
        self._check_day_rollover()
        return self._trades_today

    @property
    def is_stopped(self) -> bool:
        self._check_day_rollover()
        return self._stopped

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def progress(self) -> str:
        """Human-readable status for the dashboard."""
        self._check_day_rollover()
        pct = self._daily_pnl / self.cfg.daily_profit_target * 100
        return (
            f"Daily P&L: ${self._daily_pnl:+.2f} "
            f"({pct:+.0f}% of ${self.cfg.daily_profit_target:.0f} target) | "
            f"Trades: {self._trades_today}/{self.cfg.max_trades_per_day} | "
            f"{'STOPPED — ' + self._stop_reason if self._stopped else 'TRADING'}"
        )
