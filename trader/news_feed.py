"""
Live XAU/USD news fetcher using NewsAPI.
Filters headlines for gold/macro keywords and flags high-impact events
that should pause trading (e.g. NFP, FOMC).
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from .config import news_cfg

logger = logging.getLogger(__name__)

# Events that carry very high volatility — pause bot entry during ±30 min window
HIGH_IMPACT_KEYWORDS = [
    "non-farm payroll", "nfp", "fomc", "federal reserve meeting",
    "cpi report", "inflation data", "fed rate decision",
    "interest rate decision", "powell speech", "us jobs report",
]


@dataclass
class NewsItem:
    title: str
    source: str
    published_at: datetime
    url: str
    is_high_impact: bool = False


@dataclass
class NewsFeed:
    _items: list[NewsItem] = field(default_factory=list)
    _last_fetch: float = 0.0

    def _fetch(self) -> None:
        if not news_cfg.api_key:
            logger.debug("NEWS_API_KEY not set — skipping news fetch")
            return

        query = " OR ".join(f'"{k}"' for k in ["gold", "XAUUSD", "XAU/USD"])
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": news_cfg.api_key,
        }
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("News fetch error: %s", exc)
            return

        items = []
        for art in data.get("articles", []):
            title = art.get("title") or ""
            pub_str = art.get("publishedAt") or ""
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except ValueError:
                pub_dt = datetime.now(timezone.utc)

            high_impact = any(
                kw in title.lower() for kw in HIGH_IMPACT_KEYWORDS
            )
            items.append(
                NewsItem(
                    title=title,
                    source=(art.get("source") or {}).get("name", ""),
                    published_at=pub_dt,
                    url=art.get("url") or "",
                    is_high_impact=high_impact,
                )
            )

        self._items = items
        logger.info("News refreshed — %d articles (%d high-impact)",
                    len(items), sum(1 for i in items if i.is_high_impact))

    def refresh_if_needed(self) -> None:
        if time.time() - self._last_fetch >= news_cfg.refresh_interval_sec:
            self._fetch()
            self._last_fetch = time.time()

    def latest_headlines(self, n: int = 5) -> list[NewsItem]:
        self.refresh_if_needed()
        return sorted(self._items, key=lambda i: i.published_at, reverse=True)[:n]

    def high_impact_near_now(self, window_minutes: int = 30) -> Optional[NewsItem]:
        """
        Return the first high-impact article published within ±window_minutes
        of now. If one exists, the bot should skip new entries.
        """
        self.refresh_if_needed()
        now = datetime.now(timezone.utc)
        cutoff = timedelta(minutes=window_minutes)
        for item in self._items:
            if item.is_high_impact and abs(now - item.published_at) <= cutoff:
                return item
        return None


# Module-level singleton
news_feed = NewsFeed()
