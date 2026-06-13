#!/usr/bin/env python3
"""
Entry point for the XAU/USD trader bot.

Usage:
    python run_trader.py [--log-level DEBUG|INFO|WARNING]

Requirements:
    pip install -r trader/requirements.txt

Environment variables (copy .env.example → .env and fill in values):
    MT5_LOGIN      — your Vantage MT5 account number
    MT5_PASSWORD   — your Vantage MT5 account password
    MT5_SERVER     — Vantage MT5 server name (e.g. Vantage-Live)
    MT5_PATH       — (optional) full path to MetaTrader5 terminal64.exe
    NEWS_API_KEY   — NewsAPI.org free-tier key for live gold news
"""

import argparse
import sys
import os

# Allow running from project root: python run_trader.py
sys.path.insert(0, os.path.dirname(__file__))

from trader.bot import run


def main():
    parser = argparse.ArgumentParser(description="XAU/USD Algorithm Trader Bot")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()
    run(log_level=args.log_level)


if __name__ == "__main__":
    main()
