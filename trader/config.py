"""Central configuration for the XAU/USD algorithm trader bot."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MT5Config:
    login: int = int(os.getenv("MT5_LOGIN", "0"))
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "Vantage-Live")  # Vantage MT5 server
    path: str = os.getenv("MT5_PATH", "")  # Path to terminal64.exe if needed


@dataclass
class NewsConfig:
    api_key: str = os.getenv("NEWS_API_KEY", "")
    # Gold/XAU keywords for filtering relevant news
    keywords: list = field(default_factory=lambda: [
        "gold", "XAU", "XAUUSD", "Federal Reserve", "Fed", "inflation",
        "CPI", "USD", "dollar", "interest rate", "Treasury", "geopolitical"
    ])
    refresh_interval_sec: int = 300  # fetch news every 5 minutes


@dataclass
class StrategyConfig:
    symbol: str = "XAUUSD"
    timeframe: str = "M15"          # Primary timeframe for signals
    trend_timeframe: str = "H1"     # Higher timeframe for trend direction

    # EMA periods
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200

    # ADX trend-strength threshold
    adx_threshold: float = 25.0

    # Momentum indicators
    rsi_period: int = 14
    rsi_bull_min: float = 50.0      # RSI must be above this for longs
    rsi_bear_max: float = 50.0      # RSI must be below this for shorts
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ATR period for dynamic SL sizing
    atr_period: int = 14

    # Breakout: price must exceed this many ATR multiples above swing high/low
    breakout_atr_mult: float = 0.3

    # Pullback: price retraces this fraction of the last swing before entry
    pullback_min_retrace: float = 0.30
    pullback_max_retrace: float = 0.65


@dataclass
class RiskConfig:
    # XAU/USD: 1 point = $0.01 price move; 800 points = $8.00 move
    take_profit_points: int = 800    # $8.00 move on spot price
    stop_loss_points: int = 500      # $5.00 move — conservative 1:1.6 R:R

    max_risk_pct: float = 1.0        # max 1% of account equity per trade
    max_open_trades: int = 2         # no more than 2 concurrent positions
    daily_loss_limit_pct: float = 3.0  # halt trading if daily drawdown exceeds 3%

    # Minimum lot size Vantage allows; keep conservative
    min_lot: float = 0.01
    max_lot: float = 0.10

    # Magic number to tag bot orders in MT5
    magic_number: int = 20240601


# Singleton instances
mt5_cfg = MT5Config()
news_cfg = NewsConfig()
strategy_cfg = StrategyConfig()
risk_cfg = RiskConfig()
