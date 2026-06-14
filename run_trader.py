#!/usr/bin/env python3
"""
Entry point for the XAU/USD trader bot.

Basic usage (bot only):
    python run_trader.py

With iPhone dashboard + push notifications:
    python run_trader.py --with-api --api-port 8080

Then open http://<your-server-ip>:8080 in Safari on your iPhone
and tap Share → "Add to Home Screen" to install it as an app.

Environment variables (copy .env.example → .env):
    MT5_LOGIN       — Vantage MT5 account number
    MT5_PASSWORD    — Vantage MT5 password
    MT5_SERVER      — e.g. Vantage-Live  or  Vantage-Demo
    MT5_PATH        — (optional) path to terminal64.exe
    NEWS_API_KEY    — free key from https://newsapi.org
    NTFY_TOPIC      — any unique string, e.g. xaubot-myprivate-abc123
                      Subscribe in the ntfy iPhone app to receive alerts
"""

import argparse
import os
import sys

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
    parser.add_argument(
        "--with-api",
        action="store_true",
        help="Start the iPhone web dashboard alongside the bot",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8080,
        help="Port for the iPhone dashboard (default: 8080)",
    )
    parser.add_argument(
        "--api-host",
        default="0.0.0.0",
        help="Host to bind the dashboard to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    run(
        log_level=args.log_level,
        with_api=args.with_api,
        api_host=args.api_host,
        api_port=args.api_port,
    )


if __name__ == "__main__":
    main()
