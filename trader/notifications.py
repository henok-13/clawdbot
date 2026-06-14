"""
iPhone push notifications via ntfy.sh (free, no sign-up needed).

The user installs the free ntfy app on their iPhone and subscribes to
their private topic. The bot POSTs to https://ntfy.sh/<topic> on every
trade event.

Set NTFY_TOPIC in .env — choose any hard-to-guess string like:
  xaubot-henok-7x3k9p
"""

import logging
import os
import threading

import requests

logger = logging.getLogger(__name__)

_NTFY_BASE = "https://ntfy.sh"
_topic: str = os.getenv("NTFY_TOPIC", "")


def _send(title: str, body: str, priority: str = "default", tags: str = "") -> None:
    """Fire-and-forget POST to ntfy.sh in a background thread."""
    if not _topic:
        logger.debug("NTFY_TOPIC not set — skipping push notification")
        return

    def _post():
        try:
            headers = {
                "Title": title,
                "Priority": priority,
                "Tags": tags,
            }
            requests.post(
                f"{_NTFY_BASE}/{_topic}",
                data=body.encode("utf-8"),
                headers=headers,
                timeout=8,
            )
        except Exception as exc:
            logger.warning("ntfy push failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()


def notify_trade_opened(direction: str, entry_type: str, lot: float,
                        price: float, tp: float, sl: float) -> None:
    icon = "📈" if direction == "BUY" else "📉"
    _send(
        title=f"{icon} XAU/USD {direction} opened",
        body=(
            f"Type: {entry_type}\n"
            f"Lot: {lot}\n"
            f"Entry: {price:.2f}\n"
            f"TP: {tp:.2f}  |  SL: {sl:.2f}"
        ),
        priority="high",
        tags="moneybag,chart_with_upwards_trend" if direction == "BUY" else "moneybag,chart_with_downwards_trend",
    )


def notify_high_impact_news(headline: str) -> None:
    _send(
        title="⚠️ High-Impact Gold News — Bot Paused",
        body=headline,
        priority="urgent",
        tags="warning,newspaper",
    )


def notify_daily_limit_hit() -> None:
    _send(
        title="🛑 Daily Loss Limit Reached",
        body="XAU/USD bot has stopped taking new trades for today.",
        priority="high",
        tags="stop_sign",
    )


def notify_status(message: str) -> None:
    _send(
        title="XAU/USD Bot",
        body=message,
        priority="default",
        tags="robot",
    )
