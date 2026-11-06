"""
Multi-Horizon Stock Price Simulator for NVDA.

Simulates & predicts:
1. Overnight Market Price (20:00 ET -> 04:00 ET)
2. Market Open Price (09:30 ET Opening Bell Gap)
3. Intraday Market Close Price (16:00 ET)
4. 1-Week (5-Trading-Day) Forward Trend Trajectory with 68% and 95% Confidence Bands
5. Live Price vs. Simulated Comparison & Custom "What-If" Factor/Whale Injection Scenarios
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.database import TrackerDatabase
from simulator.calibrator import BASE_DAILY_SENSITIVITY, FACTOR_KEYS
from simulator.entry_optimizer import EntryPointOptimizer

ET_ZONE = ZoneInfo("America/New_York")


class NVDAPricePredictor:
    """Runs Monte Carlo + Jump-Diffusion price simulations across Overnight, Market Open, and 1-Week horizons."""

    def __init__(self, db: TrackerDatabase) -> None:
        self.db = db
        self.entry_optimizer = EntryPointOptimizer()

    def run_full_simulation(
        self,
        session_info: Dict[str, Any],
        factor_result: Dict[str, Any],
        calibration_state: Dict[str, Any],
        log_to_db: bool = True,
        what_if_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generates Overnight, Market Open, Intraday Close, and 1-Week (5-day) predictions,
        along with Most Suitable & Safest Entry Points and Live-vs-Predicted comparisons.
        """
        current_price = float(session_info.get("current_price", 190.0))
        last_reg_close = float(session_info.get("last_regular_close", current_price))
        technicals = factor_result.get("technicals", {})
        weights = dict(calibration_state.get("weights", {}))
        bias_drift = float(calibration_state.get("bias_drift", 0.0))
        vol_mult = float(calibration_state.get("vol_multiplier", 1.0))

        base_scores = dict(factor_result.get("scores", {}))

        # Apply optional "What-If" overrides (e.g., high-volume whale purchase, futures shock, etc.)
        effective_scores = dict(base_scores)
        scenario_notes: List[str] = []
        if what_if_overrides:
            effective_scores, scenario_notes = self._apply_what_if_overrides(
                base_scores, what_if_overrides, current_price
            )

        total_w = sum(abs(weights.get(k, 0.2)) for k in FACTOR_KEYS) or 1.0
        composite_score = float(
            np.clip(
                sum(effective_scores.get(k, 0.0) * (weights.get(k, 0.2) / total_w) for k in FACTOR_KEYS),
                -1.0,
                1.0,
            )
        )

        daily_vol = float(technicals.get("daily_volatility", 0.028)) * vol_mult
        daily_vol = float(np.clip(daily_vol, 0.012, 0.065))

        # ---------------------------------------------------------------------
        # 1. OVERNIGHT, MARKET OPEN (09:30 ET), AND TODAY'S CLOSE (16:00 ET)
        # ---------------------------------------------------------------------
        overnight_lead_score = float(
            0.48 * effective_scores.get("overnight_premarket", 0.0)
            + 0.32 * effective_scores.get("whale_institutional", 0.0)
            + 0.20 * composite_score
        )
        open_lead_score = float(
            0.40 * effective_scores.get("overnight_premarket", 0.0)
            + 0.30 * effective_scores.get("whale_institutional", 0.0)
            + 0.18 * effective_scores.get("sector_peers", 0.0)
            + 0.12 * effective_scores.get("macro_regime", 0.0)
        )

        # Overnight session drift (20:00 ET -> 04:00 ET)
        overnight_drift = 0.0085 * overnight_lead_score + 0.30 * bias_drift
        overnight_sigma = daily_vol * 0.38
        overnight_pred = current_price * (1.0 + overnight_drift)

        # Market Open (09:30 ET) prediction
        open_drift = 0.0125 * open_lead_score + 0.45 * bias_drift
        open_sigma = daily_vol * 0.52
        open_pred = current_price * (1.0 + open_drift)

        # Intraday Close (16:00 ET) prediction
        close_drift = BASE_DAILY_SENSITIVITY * composite_score + bias_drift
        close_sigma = daily_vol * 0.90
        close_pred = current_price * (1.0 + close_drift)

        short_term_horizons = {
            "overnight": self._format_horizon_card(
                title="Overnight Market Session (20:00–04:00 ET)",
                horizon_type="OVERNIGHT",
                base_price=current_price,
                pred_price=overnight_pred,
                sigma=overnight_sigma,
                driver_summary="Driven by post-market whale flow, NQ=F futures, and TSMC Asian session lead.",
            ),
            "market_open": self._format_horizon_card(
                title="Next Market Open (09:30 ET Opening Bell)",
                horizon_type="MARKET_OPEN",
                base_price=current_price,
                pred_price=open_pred,
                sigma=open_sigma,
                driver_summary="Driven by pre-market volume accumulation, institutional block orders, and SMH/Nasdaq futures.",
            ),
            "intraday_close": self._format_horizon_card(
                title="Full Session Close (16:00 ET)",
                horizon_type="DAILY_CLOSE",
                base_price=current_price,
                pred_price=close_pred,
                sigma=close_sigma,
                driver_summary="Driven by full 5-factor calibrated composite score + intraday institutional VWAP flow.",
            ),
        }

        # ---------------------------------------------------------------------
        # 2. 1-WEEK (5-TRADING-DAY) MONTE CARLO + JUMP-DIFFUSION SIMULATION
        # ---------------------------------------------------------------------
        # 10 half-day steps (Day 1 Open, Day 1 Close, ..., Day 5 Open, Day 5 Close)
        n_paths = 2000
        n_steps = 10
        rng = np.random.default_rng(int( round(current_price * 100) ) + 20261106)

        paths = np.zeros((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = current_price

        sma_20 = float(technicals.get("sma_20", current_price))
        whale_score = float(effective_scores.get("whale_institutional", 0.0))

        # Jump-diffusion parameters: high whale score shifts jump mean upward
        jump_prob_per_step = 0.06 + 0.04 * abs(whale_score)
        jump_mean = 0.006 * whale_score
        jump_std = daily_vol * 0.55

        for step in range(1, n_steps + 1):
            is_open_step = (step % 2 == 1)
            day_idx = (step - 1) // 2  # 0..4
            # Factor momentum decays slightly over 5 days toward structural trend
            decay = 0.88 ** day_idx
            if is_open_step:
                step_mu = (0.42 * BASE_DAILY_SENSITIVITY * open_lead_score * decay) + 0.40 * bias_drift
                step_sig = daily_vol * 0.55
            else:
                step_mu = (0.58 * BASE_DAILY_SENSITIVITY * composite_score * decay) + 0.60 * bias_drift
                step_sig = daily_vol * 0.78

            z = rng.standard_normal(n_paths)
            jumps = (rng.random(n_paths) < jump_prob_per_step).astype(float) * rng.normal(
                jump_mean, jump_std, size=n_paths
            )
            # Mild elasticity toward rising/falling 20-day structure to prevent unrealistic runaway variance
            prev_p = paths[:, step - 1]
            mean_rev = -0.025 * ((prev_p - sma_20 * (1.0 + 0.015 * composite_score)) / max(sma_20, 1.0))

            step_ret = step_mu + mean_rev + step_sig * z + jumps
            paths[:, step] = prev_p * np.exp(step_ret)

        # Build 1-Week trajectory timeline (10 checkpoints + t=0 Spot)
        b_days = self._next_five_trading_days()
        weekly_timeline: List[Dict[str, Any]] = [
            {
                "step": 0,
                "label": "Current Spot",
                "date": datetime.now(ET_ZONE).strftime("%Y-%m-%d"),
                "session": session_info.get("active_session", "LIVE"),
                "median": round(current_price, 2),
                "mean": round(current_price, 2),
                "low_68": round(current_price, 2),
                "high_68": round(current_price, 2),
                "low_90": round(current_price, 2),
                "high_90": round(current_price, 2),
                "return_pct": 0.0,
            }
        ]

        daily_summary_cards: List[Dict[str, Any]] = []
        for d_i, b_date in enumerate(b_days):
            open_step = d_i * 2 + 1
            close_step = d_i * 2 + 2
            date_str = b_date.strftime("%Y-%m-%d")
            day_label = b_date.strftime("%a %b %d")

            for st_idx, sess_name, short_lbl in [
                (open_step, "MARKET_OPEN", f"D{d_i+1} Open ({day_label})"),
                (close_step, "DAILY_CLOSE", f"D{d_i+1} Close ({day_label})"),
            ]:
                col = paths[:, st_idx]
                med = float(np.median(col))
                mn = float(np.mean(col))
                l68, h68 = float(np.percentile(col, 16)), float(np.percentile(col, 84))
                l90, h90 = float(np.percentile(col, 5)), float(np.percentile(col, 95))
                ret_p = ((med - current_price) / current_price) * 100.0
                weekly_timeline.append(
                    {
                        "step": st_idx,
                        "label": short_lbl,
                        "date": date_str,
                        "session": sess_name,
                        "median": round(med, 2),
                        "mean": round(mn, 2),
                        "low_68": round(l68, 2),
                        "high_68": round(h68, 2),
                        "low_90": round(l90, 2),
                        "high_90": round(h90, 2),
                        "return_pct": round(ret_p, 2),
                    }
                )

            open_col = paths[:, open_step]
            close_col = paths[:, close_step]
            daily_summary_cards.append(
                {
                    "day_number": d_i + 1,
                    "date": date_str,
                    "day_label": day_label,
                    "predicted_open": round(float(np.median(open_col)), 2),
                    "predicted_close": round(float(np.median(close_col)), 2),
                    "expected_low_68": round(float(np.percentile(close_col, 16)), 2),
                    "expected_high_68": round(float(np.percentile(close_col, 84)), 2),
                    "cumulative_return_pct": round(
                        ((float(np.median(close_col)) - current_price) / current_price) * 100.0, 2
                    ),
                    "bullish_prob_pct": round(float(np.mean(close_col > current_price)) * 100.0, 1),
                }
            )

        end_col = paths[:, -1]
        week_end_median = float(np.median(end_col))
        week_return_pct = ((week_end_median - current_price) / current_price) * 100.0
        week_high_expected = float(np.percentile(np.max(paths[:, 1:], axis=1), 70))
        week_low_expected = float(np.percentile(np.min(paths[:, 1:], axis=1), 30))
        week_bull_prob = float(np.mean(end_col > current_price)) * 100.0

        # ---------------------------------------------------------------------
        # 3. ENTRY POINT OPTIMIZATION (MOST SUITABLE & SAFEST ENTRY)
        # ---------------------------------------------------------------------
        entry_recommendations = self.entry_optimizer.optimize_entries(
            current_price=current_price,
            simulated_paths=paths,
            technicals=technicals,
            composite_score=composite_score,
            whale_score=whale_score,
        )

        # ---------------------------------------------------------------------
        # 4. LIVE PRICE VS. SIMULATED COMPARISON
        # ---------------------------------------------------------------------
        live_comparison = self._build_live_vs_simulated_comparison(
            current_price=current_price,
            last_reg_close=last_reg_close,
            session_info=session_info,
            short_term_horizons=short_term_horizons,
        )

        # Log today's live prediction if requested and not a temporary What-If run
        if log_to_db and not what_if_overrides:
            now_iso = datetime.now(ET_ZONE).isoformat()
            next_day_str = b_days[0].strftime("%Y-%m-%d")
            self.db.log_prediction(
                created_at=now_iso,
                target_date=next_day_str,
                horizon_type="MARKET_OPEN",
                horizon_days=0.5,
                base_price=round(current_price, 2),
                predicted_price=short_term_horizons["market_open"]["predicted_price"],
                low_68=short_term_horizons["market_open"]["low_68"],
                high_68=short_term_horizons["market_open"]["high_68"],
                low_95=short_term_horizons["market_open"]["low_95"],
                high_95=short_term_horizons["market_open"]["high_95"],
                factor_scores=effective_scores,
                weights_used=weights,
                actual_price=None,
                is_backtest=False,
            )

        return {
            "symbol": "NVDA",
            "base_price": round(current_price, 2),
            "composite_score": round(composite_score, 4),
            "effective_scores": effective_scores,
            "scenario_notes": scenario_notes,
            "short_term_horizons": short_term_horizons,
            "one_week_forecast": {
                "horizon_trading_days": 5,
                "end_of_week_target": round(week_end_median, 2),
                "expected_weekly_return_pct": round(week_return_pct, 2),
                "expected_weekly_high": round(week_high_expected, 2),
                "expected_weekly_low": round(week_low_expected, 2),
                "weekly_bullish_probability_pct": round(week_bull_prob, 1),
                "trend_regime": self._classify_weekly_trend(week_return_pct, week_bull_prob),
                "timeline": weekly_timeline,
                "daily_cards": daily_summary_cards,
            },
            "entry_recommendations": entry_recommendations,
            "live_vs_simulated": live_comparison,
        }

    def _build_live_vs_simulated_comparison(
        self,
        current_price: float,
        last_reg_close: float,
        session_info: Dict[str, Any],
        short_term_horizons: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compares current NVDA price against historical resolved predictions and active session targets."""
        resolved = self.db.get_recent_predictions(limit=30, resolved_only=True)

        # Latest resolved Market Open and Daily Close predictions
        open_preds = [p for p in resolved if p.get("horizon_type") == "MARKET_OPEN"][:12]
        close_preds = [p for p in resolved if p.get("horizon_type") == "DAILY_CLOSE"][:12]

        latest_resolved_close = close_preds[0] if close_preds else None
        latest_resolved_open = open_preds[0] if open_preds else None

        active_sess = str(session_info.get("active_session", "REGULAR"))
        if active_sess in ("PRE_MARKET", "OVERNIGHT", "WEEKEND_OVERNIGHT"):
            benchmark_pred = float(short_term_horizons["market_open"]["predicted_price"])
            benchmark_label = "Simulated Next Market Open (09:30 ET)"
        elif active_sess == "POST_MARKET":
            benchmark_pred = float(short_term_horizons["overnight"]["predicted_price"])
            benchmark_label = "Simulated Overnight Session Target"
        else:
            benchmark_pred = float(short_term_horizons["intraday_close"]["predicted_price"])
            benchmark_label = "Simulated Intraday Close (16:00 ET)"

        spread_dollars = round(benchmark_pred - current_price, 2)
        spread_pct = round(((benchmark_pred - current_price) / current_price) * 100.0, 2)

        # Build chronological comparison series for chart (last 18 daily close predictions vs actuals)
        chart_records = []
        for p in reversed(close_preds[:18]):
            chart_records.append(
                {
                    "date": p["target_date"],
                    "horizon": p["horizon_type"],
                    "base_price": round(float(p["base_price"]), 2),
                    "predicted_price": round(float(p["predicted_price"]), 2),
                    "actual_price": round(float(p["actual_price"]), 2) if p.get("actual_price") else None,
                    "low_68": round(float(p["low_68"]), 2),
                    "high_68": round(float(p["high_68"]), 2),
                    "abs_error": round(float(p["abs_error"]), 2) if p.get("abs_error") is not None else 0.0,
                    "pct_error": round(float(p["pct_error"]), 2) if p.get("pct_error") is not None else 0.0,
                    "direction_correct": bool(p.get("direction_correct", 0)),
                }
            )

        return {
            "current_market_price": round(current_price, 2),
            "last_regular_close": round(last_reg_close, 2),
            "active_session": active_sess,
            "active_benchmark_label": benchmark_label,
            "active_simulated_price": round(benchmark_pred, 2),
            "implied_upside_dollars": spread_dollars,
            "implied_upside_pct": spread_pct,
            "valuation_status": (
                "TRADING BELOW SIMULATED FAIR VALUE (BULLISH GAP)"
                if spread_pct > 0.25
                else (
                    "TRADING ABOVE SIMULATED TARGET (EXTENDED)"
                    if spread_pct < -0.25
                    else "TRACKING INLINE WITH SIMULATION"
                )
            ),
            "latest_resolved_open": latest_resolved_open,
            "latest_resolved_close": latest_resolved_close,
            "historical_comparison_series": chart_records,
        }

    @staticmethod
    def _apply_what_if_overrides(
        base_scores: Dict[str, float],
        overrides: Dict[str, float],
        current_price: float,
    ) -> tuple[Dict[str, float], List[str]]:
        """
        Converts user-supplied scenario shocks (e.g. whale purchase in $M / shares,
        Nasdaq futures %, semiconductor peer shock %, VIX change %) into adjusted factor scores.
        """
        scores = dict(base_scores)
        notes: List[str] = []

        # 1. Whale / Institutional Block Purchase or Sale (in Millions USD or Shares)
        whale_musd = float(overrides.get("whale_net_buy_musd", 0.0))
        if abs(whale_musd) > 0.1:
            equiv_shares = (whale_musd * 1e6) / max(current_price, 1.0)
            # A $500M net institutional/whale block buy shifts whale_institutional score by ~+0.45
            delta_w = float(np.tanh(whale_musd / 650.0))
            scores["whale_institutional"] = round(float(np.clip(scores["whale_institutional"] + delta_w, -1.0, 1.0)), 4)
            action_word = "Purchase" if whale_musd > 0 else "Sale"
            notes.append(
                f"Simulated Whale {action_word}: ${abs(whale_musd):,.0f}M (~{abs(equiv_shares):,.0f} shares) "
                f"shifted Whale factor to {scores['whale_institutional']:+.2f}."
            )

        # 2. Overnight / Nasdaq Futures Shock (%)
        nq_shock_pct = float(overrides.get("overnight_futures_pct", 0.0))
        if abs(nq_shock_pct) > 0.01:
            delta_ov = float(np.tanh(nq_shock_pct / 1.8))
            scores["overnight_premarket"] = round(
                float(np.clip(scores["overnight_premarket"] + delta_ov, -1.0, 1.0)), 4
            )
            notes.append(
                f"Overnight Futures Shock ({nq_shock_pct:+.2f}%) shifted Overnight factor to {scores['overnight_premarket']:+.2f}."
            )

        # 3. Semiconductor Sector / AI Capex Shock (%)
        sector_shock_pct = float(overrides.get("sector_peers_pct", 0.0))
        if abs(sector_shock_pct) > 0.01:
            delta_sec = float(np.tanh(sector_shock_pct / 2.5))
            scores["sector_peers"] = round(float(np.clip(scores["sector_peers"] + delta_sec, -1.0, 1.0)), 4)
            notes.append(
                f"Semiconductor Peers Shock ({sector_shock_pct:+.2f}%) shifted Sector factor to {scores['sector_peers']:+.2f}."
            )

        # 4. Macro / VIX Volatility Shock (%)
        vix_shock_pct = float(overrides.get("vix_change_pct", 0.0))
        if abs(vix_shock_pct) > 0.01:
            delta_mac = float(-np.tanh(vix_shock_pct / 12.0))
            scores["macro_regime"] = round(float(np.clip(scores["macro_regime"] + delta_mac, -1.0, 1.0)), 4)
            notes.append(
                f"Macro VIX Shock ({vix_shock_pct:+.1f}%) shifted Macro factor to {scores['macro_regime']:+.2f}."
            )

        # 5. News / Earnings Catalyst Sentiment Override
        news_boost = float(overrides.get("sentiment_override", 0.0))
        if abs(news_boost) > 0.01:
            scores["technical_sentiment"] = round(
                float(np.clip(scores["technical_sentiment"] + news_boost * 0.6, -1.0, 1.0)), 4
            )
            notes.append(
                f"Catalyst Sentiment Shift ({news_boost:+.2f}) shifted Technical/Sentiment factor to {scores['technical_sentiment']:+.2f}."
            )

        return scores, notes

    @staticmethod
    def _format_horizon_card(
        title: str,
        horizon_type: str,
        base_price: float,
        pred_price: float,
        sigma: float,
        driver_summary: str,
    ) -> Dict[str, Any]:
        ret_pct = ((pred_price - base_price) / base_price) * 100.0 if base_price > 0 else 0.0
        low_68 = pred_price * (1.0 - sigma)
        high_68 = pred_price * (1.0 + sigma)
        low_95 = pred_price * (1.0 - 1.96 * sigma)
        high_95 = pred_price * (1.0 + 1.96 * sigma)
        return {
            "title": title,
            "horizon_type": horizon_type,
            "base_price": round(base_price, 2),
            "predicted_price": round(pred_price, 2),
            "predicted_change_dollars": round(pred_price - base_price, 2),
            "predicted_return_pct": round(ret_pct, 2),
            "direction": "BULLISH" if ret_pct >= 0.05 else ("BEARISH" if ret_pct <= -0.05 else "FLAT"),
            "low_68": round(low_68, 2),
            "high_68": round(high_68, 2),
            "low_95": round(low_95, 2),
            "high_95": round(high_95, 2),
            "driver_summary": driver_summary,
        }

    @staticmethod
    def _next_five_trading_days() -> List[datetime]:
        now_et = datetime.now(ET_ZONE)
        days: List[datetime] = []
        cur = now_et
        # If before 09:30 ET on a weekday, Day 1 includes today's regular session
        if cur.weekday() < 5 and (cur.hour * 60 + cur.minute) < (9 * 60 + 30):
            days.append(cur)
        while len(days) < 5:
            cur = cur + timedelta(days=1)
            if cur.weekday() < 5:
                days.append(cur)
        return days

    @staticmethod
    def _classify_weekly_trend(week_ret_pct: float, bull_prob: float) -> str:
        if week_ret_pct >= 2.5 and bull_prob >= 62.0:
            return "STRONG BULLISH EXPANSION"
        if week_ret_pct >= 0.6 and bull_prob >= 53.0:
            return "MODERATE BULLISH ACCUMULATION"
        if week_ret_pct <= -2.5 and bull_prob <= 38.0:
            return "BEARISH DISTRIBUTION / PULLBACK"
        if week_ret_pct <= -0.6:
            return "MILD CONSOLIDATION / DIP"
        return "RANGE-BOUND / VOLATILITY DIGESTION"
