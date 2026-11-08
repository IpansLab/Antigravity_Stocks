#!/usr/bin/env python3
"""
Command-Line Interface for the NVDA Multi-Factor Stock Tracker,
Overnight/Market-Open/1-Week Price Simulator, Whale Monitor, and Self-Tuning Engine.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

from simulator.engine_service import NVDAQuantService


def format_usd(val: float) -> str:
    if abs(val) >= 1e9:
        return f"${val / 1e9:,.2f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:,.2f}M"
    return f"${val:,.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVDA Multi-Factor Price Tracker, 1-Week Simulator, Whale Monitor & Self-Tuning Engine"
    )
    parser.add_argument("--refresh", action="store_true", help="Force fresh market data fetch from Yahoo Finance")
    parser.add_argument("--fine-tune", action="store_true", help="Run online self-tuning against latest market price")
    parser.add_argument(
        "--recalibrate-backtest",
        action="store_true",
        help="Re-run full 5-year (2021-2026) walk-forward historical calibration and update SQLite database",
    )
    parser.add_argument(
        "--whale-buy-musd",
        type=float,
        default=0.0,
        help="Simulate a What-If institutional/whale block purchase (+) or sale (-) in Millions USD",
    )
    parser.add_argument(
        "--futures-pct",
        type=float,
        default=0.0,
        help="Simulate a What-If overnight Nasdaq futures shock (%)",
    )
    parser.add_argument(
        "--sector-pct",
        type=float,
        default=0.0,
        help="Simulate a What-If semiconductor sector (SMH/TSM/AMD) shock (%)",
    )
    parser.add_argument(
        "--export-snapshot",
        action="store_true",
        help="Export latest 5Y calibration history, database status, and forecast JSON files to data/",
    )
    parser.add_argument("--json", action="store_true", help="Output full payload as JSON")
    args = parser.parse_args()

    svc = NVDAQuantService()

    if args.fine_tune or args.recalibrate_backtest:
        tune_res = svc.trigger_self_tuning(recalibrate_backtest=args.recalibrate_backtest)
        if not args.json:
            st = tune_res["updated_state"]
            print("=" * 84)
            print(" [SELF-TUNING 5-YEAR CALIBRATION EXECUTED]")
            print(f" Step Type : {st['step_type']} | Samples Evaluated: {st['samples_evaluated']}")
            print(
                f" Accuracy  : 30D Rolling MAE ${st['rolling_mae']:.2f} (vs RW Baseline ${st['baseline_mae']:.2f}) "
                f"| MAPE {st['rolling_mape']:.2f}% | Directional Hit Rate {st['directional_accuracy']:.1f}%"
            )
            print(f" Notes     : {st.get('notes', '')}")
            print("=" * 84)

    overrides: Dict[str, float] = {}
    if abs(args.whale_buy_musd) > 0.01:
        overrides["whale_net_buy_musd"] = args.whale_buy_musd
    if abs(args.futures_pct) > 0.01:
        overrides["overnight_futures_pct"] = args.futures_pct
    if abs(args.sector_pct) > 0.01:
        overrides["sector_peers_pct"] = args.sector_pct

    payload = svc.get_full_dashboard_payload(
        force_refresh=args.refresh,
        what_if_overrides=overrides if overrides else None,
    )

    if args.export_snapshot or args.fine_tune or args.recalibrate_backtest:
        import os
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        cal_export = {
            "updated_at": payload["generated_at"],
            "current_calibration_state": payload["calibration"]["current_state"],
            "database_status": payload.get("database_status", {}),
            "weight_evolution_history": payload["calibration"]["history"],
            "recent_predictions_vs_actuals": payload["simulation"]["live_vs_simulated"]["historical_comparison_series"],
        }
        forecast_export = {
            "generated_at": payload["generated_at"],
            "symbol": "NVDA",
            "current_price": payload["market"]["current_price"],
            "active_session": payload["market"]["active_session"],
            "regular_change_pct": payload["market"].get("regular_change_pct", 0.0),
            "extended_change_pct": payload["market"].get("extended_change_pct", 0.0),
            "composite_score": payload["factors"]["composite_score"],
            "composite_label": payload["factors"]["composite_label"],
            "factor_scores": payload["factors"]["scores"],
            "short_term_horizons": payload["simulation"]["short_term_horizons"],
            "one_week_forecast": payload["simulation"]["one_week_forecast"],
            "entry_recommendations": payload["simulation"]["entry_recommendations"],
            "database_status": payload.get("database_status", {}),
        }
        with open(os.path.join(data_dir, "calibration_history.json"), "w", encoding="utf-8") as f_cal:
            json.dump(cal_export, f_cal, indent=2)
        with open(os.path.join(data_dir, "latest_forecast.json"), "w", encoding="utf-8") as f_fc:
            json.dump(forecast_export, f_fc, indent=2)

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    mkt = payload["market"]
    fac = payload["factors"]
    sim = payload["simulation"]
    cal = payload["calibration"]["current_state"]
    whale = payload["whale_activity"]
    entries = sim["entry_recommendations"]
    comp = sim["live_vs_simulated"]

    print("\n" + "=" * 84)
    print(f" NVDA QUANTITATIVE MULTI-FACTOR TRACKER & 1-WEEK SIMULATOR  ({mkt['timestamp_et'][:19]})")
    print("=" * 84)
    print(
        f" Live Price      : ${mkt['current_price']:.2f}  ({mkt['extended_change_pct']:+.2f}% vs Last Reg Close ${mkt['last_regular_close']:.2f})"
    )
    print(
        f" Active Session  : {mkt['active_session']}  | Intraday VWAP: ${mkt['vwap_intraday']:.2f}  "
        f"| 20D Avg Vol: {mkt['avg_daily_volume_20d']:,}"
    )
    print(
        f" Live vs Sim     : {comp['active_benchmark_label']} = ${comp['active_simulated_price']:.2f} "
        f"({comp['implied_upside_dollars']:+.2f} / {comp['implied_upside_pct']:+.2f}%) -> {comp['valuation_status']}"
    )

    if sim.get("scenario_notes"):
        print("\n [ACTIVE WHAT-IF SCENARIO OVERRIDES]")
        for note in sim["scenario_notes"]:
            print(f"  * {note}")

    print("\n" + "-" * 84)
    print(f" 1. MULTI-FACTOR INFLUENCE BREAKDOWN  (Composite Score: {fac['composite_score']:+.3f} — {fac['composite_label']})")
    print("-" * 84)
    for b in fac["breakdown"]:
        bar_len = int(round(abs(b["score"]) * 12))
        bar_char = "+" if b["score"] >= 0 else "-"
        bar_str = (bar_char * bar_len).ljust(12)
        print(
            f"  • {b['name']:<36} Score: {b['score']:+6.3f} [{bar_str}] "
            f"Weight: {b['weight']*100:4.1f}%  ({b['signal_label']})"
        )
        for drv in b["drivers"]:
            print(f"      └─ {drv}")

    print("\n" + "-" * 84)
    print(" 2. HIGH-VOLUME WHALE, INSTITUTIONAL (13F) & INSIDER (FORM 4) MONITOR")
    print("-" * 84)
    bv = whale["block_volume"]
    inst = whale["institutional"]
    print(
        f"  Block Flow (10D + Intraday): Accumulation ${bv['accumulation_pressure_B']:.2f}B vs "
        f"Distribution ${bv['distribution_pressure_B']:.2f}B (Net {bv['net_block_flow_B']:+.2f}B | {bv['relative_volume_20d']:.2f}x RVOL)"
    )
    print(
        f"  Top-10 Institutions (13F)  : ${inst['top_holders_total_value_B']:.1f}B Held | "
        f"Weighted Position Change: {inst['weighted_position_change_pct']:+.2f}% ({inst['net_shares_added']:+,} shares)"
    )
    for ev in whale["events"][:5]:
        print(f"  [{ev['event_date']}] ({ev['action_direction']}) {ev['summary']}")

    print("\n" + "-" * 84)
    print(" 3. OVERNIGHT, MARKET OPEN & INTRADAY CLOSE SIMULATIONS")
    print("-" * 84)
    for key in ("overnight", "market_open", "intraday_close"):
        h = sim["short_term_horizons"][key]
        print(
            f"  • {h['title']:<44}: ${h['predicted_price']:7.2f} ({h['predicted_return_pct']:+5.2f}%) "
            f"| 68% Band [${h['low_68']:.2f} – ${h['high_68']:.2f}]"
        )

    wk = sim["one_week_forecast"]
    print("\n" + "-" * 84)
    print(
        f" 4. 1-WEEK (5-TRADING-DAY) TREND FORECAST  —  {wk['trend_regime']}"
    )
    print("-" * 84)
    print(
        f"  End-of-Week Target : ${wk['end_of_week_target']:.2f} ({wk['expected_weekly_return_pct']:+.2f}%) | "
        f"Expected Weekly Range: ${wk['expected_weekly_low']:.2f} – ${wk['expected_weekly_high']:.2f} | "
        f"Bullish Prob: {wk['weekly_bullish_probability_pct']:.1f}%"
    )
    for dc in wk["daily_cards"]:
        print(
            f"   Day {dc['day_number']} ({dc['day_label']}) -> Pred Open: ${dc['predicted_open']:6.2f} | "
            f"Pred Close: ${dc['predicted_close']:6.2f} ({dc['cumulative_return_pct']:+5.2f}%) | "
            f"68% Range: [${dc['expected_low_68']:.2f} – ${dc['expected_high_68']:.2f}]"
        )

    print("\n" + "-" * 84)
    print(" 5. RECOMMENDED ENTRY POINTS FOR PROFIT & SAFETY")
    print("-" * 84)
    opt = entries["most_suitable_entry"]
    safe = entries["safest_entry"]
    print(f"  ★ {opt['label'].upper()}")
    print(
        f"    Entry Limit Price : ${opt['entry_price']:.2f} ({opt['discount_from_current_pct']:+.2f}% vs spot) "
        f"| Fill Probability: {opt['fill_probability_pct']:.1f}%"
    )
    print(
        f"    Profit Targets    : Primary ${opt['primary_target_price']:.2f} ({opt['expected_gain_pct']:+.2f}%) "
        f"| Stretch ${opt['stretch_target_price']:.2f} ({opt['stretch_gain_pct']:+.2f}%)"
    )
    print(
        f"    Stop-Loss & R:R   : Stop ${opt['stop_loss_price']:.2f} (-{opt['max_risk_pct']:.2f}% risk) "
        f"| Reward-to-Risk: {opt['reward_to_risk_ratio']:.2f}:1"
    )
    print(f"    Timing Window     : {opt['best_timing_window']}")
    print(f"    Rationale         : {opt['rationale']}\n")

    print(f"  ◆ {safe['label'].upper()}")
    print(
        f"    Safe Limit Price  : ${safe['entry_price']:.2f} ({safe['discount_from_current_pct']:+.2f}% vs spot) "
        f"| Fill Probability: {safe['fill_probability_pct']:.1f}% | Safety Score: {safe['margin_of_safety_score']}/100"
    )
    print(
        f"    Target & Stop     : Target ${safe['conservative_target_price']:.2f} ({safe['expected_gain_pct']:+.2f}%) "
        f"| Stop-Loss ${safe['stop_loss_price']:.2f} (-{safe['max_drawdown_risk_pct']:.2f}% risk) "
        f"| R:R {safe['reward_to_risk_ratio']:.2f}:1"
    )
    print(f"    Rationale         : {safe['rationale']}")

    print("\n" + "-" * 84)
    print(" 6. ADAPTIVE SELF-TUNING CALIBRATION STATUS")
    print("-" * 84)
    mae_imp = (
        ((cal["baseline_mae"] - cal["rolling_mae"]) / cal["baseline_mae"]) * 100.0
        if cal["baseline_mae"] > 0
        else 0.0
    )
    print(
        f"  Evaluated Predictions : {cal['samples_evaluated']} | "
        f"Rolling MAE: ${cal['rolling_mae']:.2f} vs Uncalibrated ${cal['baseline_mae']:.2f} ({mae_imp:+.1f}% Error Reduction)"
    )
    print(
        f"  Directional Accuracy  : {cal['directional_accuracy']:.1f}% | "
        f"Rolling MAPE: {cal['rolling_mape']:.2f}% | Learned Drift Bias: {cal['bias_drift']*100:+.3f}%"
    )
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
