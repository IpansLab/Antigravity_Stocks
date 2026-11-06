#!/usr/bin/env python3
"""
One-command launcher for the NVDA Quant Terminal.

Usage:
  python run_terminal.py            # Starts the Web Dashboard server on http://localhost:8000 (auto-opens browser)
  python run_terminal.py --live-cli # Runs a 1-second live-updating ticker + forecast in the terminal (Ctrl+C to close)
"""

from __future__ import annotations

import argparse
import time
import webbrowser
from threading import Timer

import uvicorn

from simulator.engine_service import NVDAQuantService


def run_live_cli_loop() -> None:
    svc = NVDAQuantService()
    payload = svc.get_full_dashboard_payload(force_refresh=True)
    sim = payload["simulation"]
    opt = sim["entry_recommendations"]["most_suitable_entry"]
    safe = sim["entry_recommendations"]["safest_entry"]
    open_pred = sim["short_term_horizons"]["market_open"]["predicted_price"]
    week_pred = sim["one_week_forecast"]["end_of_week_target"]

    print("=" * 94)
    print(" NVDA LIVE 1-SECOND TERMINAL STREAM  (Press Ctrl+C to close anytime)")
    print("=" * 94)
    try:
        while True:
            tick = svc.fetcher.get_fast_live_tick()
            p = tick["price"]
            chg = tick["extended_change_pct"]
            arrow = "▲" if tick["tick_direction"] > 0 else ("▼" if tick["tick_direction"] < 0 else "•")
            line = (
                f"\r [{tick['timestamp_et']}] NVDA {arrow} ${p:6.2f} ({chg:+5.2f}%) | "
                f"Sim Open: ${open_pred:6.2f} | 1W Target: ${week_pred:6.2f} | "
                f"Opt Entry: ${opt['entry_price']:6.2f} | Safe Entry: ${safe['entry_price']:6.2f} "
            )
            print(line, end="", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\n[Closed NVDA Terminal Stream cleanly.]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start or close the NVDA Quant Terminal")
    parser.add_argument("--port", type=int, default=8000, help="Port for Web Dashboard (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    parser.add_argument("--live-cli", action="store_true", help="Run 1-second live price ticker in terminal")
    args = parser.parse_args()

    if args.live_cli:
        run_live_cli_loop()
        return

    url = f"http://localhost:{args.port}"
    print("=" * 76)
    print(f" Starting NVDA Quant Terminal at {url}")
    print(" To CLOSE the server at any time: Press Ctrl + C in this terminal window.")
    print(" To RUN IT BACK again:            Run `python run_terminal.py`")
    print("=" * 76)

    if not args.no_browser:
        Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run("app.main:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()
