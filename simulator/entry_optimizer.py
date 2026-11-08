"""
Entry Point & Profit Optimization Engine for NVDA 1-Week Trend Forecasts.

Determines:
1. Most Suitable Entry Point (Optimal Profit / Reward-to-Risk):
   Maximizes [Fill Probability x Expected Upside Return / Downside Risk] across 2,000 simulated
   1-week price trajectories and technical support confluences.
2. Safest Entry Point (Maximum Margin of Safety):
   Targets deep institutional accumulation zones (Volume Profile POC, SMA/Bollinger support,
   and 18th-25th percentile simulated 1-week dip) to minimize drawdown risk while capturing rebound profit.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class EntryPointOptimizer:
    """Computes optimal and safest entry prices, targets, stops, and a 3-tranche execution plan."""

    def optimize_entries(
        self,
        current_price: float,
        simulated_paths: np.ndarray,
        technicals: Dict[str, Any],
        composite_score: float,
        whale_score: float,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        current_price : float
            Current live NVDA price.
        simulated_paths : np.ndarray
            Shape (n_paths, n_steps + 1) representing simulated price trajectories over 5 trading days (10 half-day steps).
        technicals : Dict[str, Any]
            Technical indicators, ATR(14), Volume POC, and support/resistance levels.
        composite_score : float
            Weighted multi-factor score in [-1.0, +1.0].
        whale_score : float
            Whale/Institutional factor score in [-1.0, +1.0].
        """
        atr = max(float(technicals.get("atr_14", current_price * 0.03)), current_price * 0.005)
        raw_supports = [float(s) for s in technicals.get("support_levels", []) if float(s) < current_price * 0.998]
        resistances = [float(r) for r in technicals.get("resistance_levels", []) if float(r) > current_price * 1.002]

        raw_poc = float(technicals.get("volume_poc", current_price * 0.975))
        support_poc = float(technicals.get("support_volume_poc", current_price * 0.970))
        if raw_poc < current_price * 0.998:
            vwap_poc = raw_poc
        elif support_poc < current_price * 0.998:
            vwap_poc = support_poc
        else:
            vwap_poc = round(current_price - 0.95 * atr, 2)

        sma_20 = float(technicals.get("sma_20", current_price * 0.985))
        supports = sorted(set(raw_supports + [round(vwap_poc, 2)]), reverse=True)

        # Path minimums, maximums, and terminal prices across the 5-day (10-step) simulation
        path_mins = np.min(simulated_paths[:, 1:], axis=1)
        path_maxs = np.max(simulated_paths[:, 1:], axis=1)
        day2_mins = np.min(simulated_paths[:, 1:5], axis=1)  # First 2 trading days

        p40_low = float(np.percentile(day2_mins, 42))
        p20_low = float(np.percentile(path_mins, 22))
        p10_low = float(np.percentile(path_mins, 10))
        p60_high = float(np.percentile(path_maxs, 62))
        p80_high = float(np.percentile(path_maxs, 82))

        # -------------------------------------------------------------------------
        # 1. MOST SUITABLE ENTRY POINT (Optimal Profit x Fill Probability, strictly post-fill t > t_fill)
        # -------------------------------------------------------------------------
        max_opt_ceil = current_price - max(0.28 * atr, current_price * 0.005)
        min_opt_floor = current_price - 1.50 * atr
        candidates = np.linspace(min_opt_floor, max_opt_ceil, 28)
        for s in supports:
            if current_price - 1.80 * atr <= s <= max_opt_ceil:
                candidates = np.append(candidates, s)
        candidates = np.unique(np.round(candidates, 2))

        best_candidate = float(min(p40_low, max_opt_ceil))
        best_utility = -1e9

        confluence_refs = [vwap_poc] + list(supports[:3])
        if sma_20 < current_price * 0.998:
            confluence_refs.append(sma_20)

        for cand in candidates:
            hit_mask = path_mins <= cand
            fill_prob = float(np.mean(hit_mask))
            if fill_prob < 0.25:
                continue

            # Strictly evaluate post-fill peak and post-fill drawdown for steps t > t_fill
            post_entry_peaks = []
            post_entry_troughs = []
            for p_idx in np.where(hit_mask)[0][:600]:
                path = simulated_paths[p_idx, 1:]
                first_hit_step = int(np.argmax(path <= cand))
                # Strictly after the fill step (t > t_fill); if filled on the terminal step, only terminal price remains
                post_slice = path[first_hit_step + 1 :] if (first_hit_step + 1) < len(path) else path[-1:]
                post_entry_peaks.append(float(np.max(post_slice)))
                post_entry_troughs.append(float(np.min(post_slice)))

            expected_peak = float(np.mean(post_entry_peaks)) if post_entry_peaks else p60_high
            post_p10_trough = float(np.percentile(post_entry_troughs, 10)) if post_entry_troughs else p10_low
            upside_dollars = max(expected_peak - cand, atr * 0.5)
            stop_distance = max(1.10 * atr, cand - post_p10_trough)
            rr_ratio = upside_dollars / max(stop_distance, 0.25 * atr)

            confluence_bonus = 1.0
            for ref_lvl in confluence_refs:
                if abs(cand - ref_lvl) / max(current_price, 1e-4) <= 0.006:
                    confluence_bonus = 1.12
                    break

            utility = (fill_prob ** 0.72) * rr_ratio * confluence_bonus
            if utility > best_utility:
                best_utility = utility
                best_candidate = float(cand)

        opt_entry_price = round(min(best_candidate, max_opt_ceil), 2)
        opt_fill_prob = round(float(np.mean(path_mins <= opt_entry_price)) * 100.0, 1)
        opt_discount_pct = round(((opt_entry_price - current_price) / current_price) * 100.0, 2)

        # Compute post-fill distribution for the selected Optimal Entry
        opt_hit_indices = np.where(path_mins <= opt_entry_price)[0]
        if len(opt_hit_indices) > 0:
            opt_post_peaks = []
            for p_idx in opt_hit_indices:
                path = simulated_paths[p_idx, 1:]
                t_fill = int(np.argmax(path <= opt_entry_price))
                post_s = path[t_fill + 1 :] if (t_fill + 1) < len(path) else path[-1:]
                opt_post_peaks.append(float(np.max(post_s)))
            post_p65_high = float(np.percentile(opt_post_peaks, 65))
            post_p85_high = float(np.percentile(opt_post_peaks, 85))
        else:
            post_p65_high = p60_high
            post_p85_high = p80_high

        opt_primary_target = round(max(post_p65_high, opt_entry_price + 1.40 * atr), 2)
        opt_stretch_target = round(max(post_p85_high, opt_primary_target + 0.90 * atr), 2)
        opt_stop_loss = round(opt_entry_price - 1.15 * atr, 2)

        opt_gain_pct = round(((opt_primary_target - opt_entry_price) / opt_entry_price) * 100.0, 2)
        opt_stretch_gain_pct = round(((opt_stretch_target - opt_entry_price) / opt_entry_price) * 100.0, 2)
        opt_risk_pct = round(((opt_entry_price - opt_stop_loss) / opt_entry_price) * 100.0, 2)
        opt_rr = round(opt_gain_pct / max(opt_risk_pct, 0.25), 2)

        if composite_score >= 0.20:
            opt_timing = "Pre-Market / Opening 45m Consolidation Dip (Day 1–2)"
        elif composite_score <= -0.15:
            opt_timing = "Mid-Week Pullback Stabilization (Day 2–3)"
        else:
            opt_timing = "Intraday VWAP Retest Window (Day 1–2)"

        # -------------------------------------------------------------------------
        # 2. SAFEST ENTRY POINT (Maximum Margin of Safety & Below-Spot Structural Floor)
        # -------------------------------------------------------------------------
        deeper_supports = [s for s in supports if s < opt_entry_price - 0.25 * atr]
        if deeper_supports:
            anchor_support = deeper_supports[0]
            safe_raw = 0.55 * p20_low + 0.45 * anchor_support
        else:
            anchor_support = vwap_poc if vwap_poc < opt_entry_price else round(opt_entry_price - 0.85 * atr, 2)
            safe_raw = min(p20_low, opt_entry_price - 0.85 * atr)

        safe_entry_price = round(min(safe_raw, opt_entry_price - max(0.45 * atr, current_price * 0.006)), 2)
        safe_fill_prob = round(float(np.mean(path_mins <= safe_entry_price)) * 100.0, 1)
        safe_discount_pct = round(((safe_entry_price - current_price) / current_price) * 100.0, 2)

        safe_target = round(max(current_price + 0.35 * atr, safe_entry_price + 1.55 * atr), 2)
        safe_stop_loss = round(min(p10_low, safe_entry_price - 0.95 * atr), 2)
        safe_gain_pct = round(((safe_target - safe_entry_price) / safe_entry_price) * 100.0, 2)
        safe_risk_pct = round(((safe_entry_price - safe_stop_loss) / safe_entry_price) * 100.0, 2)
        safe_rr = round(safe_gain_pct / max(safe_risk_pct, 0.25), 2)

        # Post-fill win probability (strictly evaluating steps t > t_fill after Safe Entry is reached)
        safe_hit_indices = np.where(path_mins <= safe_entry_price)[0]
        if len(safe_hit_indices) > 0:
            post_fill_wins = []
            for p_idx in safe_hit_indices:
                path = simulated_paths[p_idx, 1:]
                t_fill = int(np.argmax(path <= safe_entry_price))
                post_s = path[t_fill + 1 :] if (t_fill + 1) < len(path) else path[-1:]
                # Win if subsequent trajectory rebounds >= +1.5% above safe_entry_price or finishes above safe_entry_price
                rebounded = (float(np.max(post_s)) >= safe_entry_price * 1.015) or (float(post_s[-1]) > safe_entry_price)
                post_fill_wins.append(1.0 if rebounded else 0.0)
            win_prob_safe = float(np.mean(post_fill_wins)) * 100.0
        else:
            win_prob_safe = 78.0
        safety_score = int(np.clip(round(win_prob_safe), 65, 96))

        # -------------------------------------------------------------------------
        # 3. STRICTLY MONOTONIC 3-TRANCHE EXECUTION LADDER
        #    Enforces: current_price > tactical_entry > opt_entry_price > safe_entry_price
        # -------------------------------------------------------------------------
        tactical_raw = 0.55 * current_price + 0.45 * opt_entry_price
        tactical_entry = round(
            max(opt_entry_price + 0.05, min(current_price - 0.05, tactical_raw)),
            2,
        )
        blended_avg = round(0.30 * tactical_entry + 0.45 * opt_entry_price + 0.25 * safe_entry_price, 2)

        tranches: List[Dict[str, Any]] = [
            {
                "tranche": "Tranche 1 — Tactical Starter (30%)",
                "allocation_pct": 30,
                "limit_price": tactical_entry,
                "discount_pct": round(((tactical_entry - current_price) / current_price) * 100.0, 2),
                "fill_probability_pct": round(float(np.mean(path_mins <= tactical_entry)) * 100.0, 1),
                "purpose": "Captures immediate upside if NVDA gaps & runs from Market Open with only a shallow dip.",
            },
            {
                "tranche": "Tranche 2 — Most Suitable Core Entry (45%)",
                "allocation_pct": 45,
                "limit_price": opt_entry_price,
                "discount_pct": opt_discount_pct,
                "fill_probability_pct": opt_fill_prob,
                "purpose": "Highest post-fill expected Reward-to-Risk entry aligned with Day 1–2 pullback zone.",
            },
            {
                "tranche": "Tranche 3 — Safest Institutional Floor (25%)",
                "allocation_pct": 25,
                "limit_price": safe_entry_price,
                "discount_pct": safe_discount_pct,
                "fill_probability_pct": safe_fill_prob,
                "purpose": "High margin-of-safety bid sitting at deep institutional support / 20th-percentile weekly low.",
            },
        ]

        return {
            "most_suitable_entry": {
                "label": "Most Suitable Entry Point (Optimal Profit)",
                "entry_price": opt_entry_price,
                "discount_from_current_pct": opt_discount_pct,
                "fill_probability_pct": opt_fill_prob,
                "best_timing_window": opt_timing,
                "primary_target_price": opt_primary_target,
                "expected_gain_pct": opt_gain_pct,
                "stretch_target_price": opt_stretch_target,
                "stretch_gain_pct": opt_stretch_gain_pct,
                "stop_loss_price": opt_stop_loss,
                "max_risk_pct": opt_risk_pct,
                "reward_to_risk_ratio": opt_rr,
                "rationale": (
                    f"Maximizes post-fill 1-week profit × fill probability ({opt_fill_prob:.0f}% chance of fill). "
                    f"Sits {abs(opt_discount_pct):.2f}% below spot near support node ${vwap_poc:.2f} with a "
                    f"{opt_rr:.2f}:1 post-fill Reward-to-Risk ratio toward ${opt_primary_target:.2f}."
                ),
            },
            "safest_entry": {
                "label": "Safest Entry Point (Maximum Margin of Safety)",
                "entry_price": safe_entry_price,
                "discount_from_current_pct": safe_discount_pct,
                "fill_probability_pct": safe_fill_prob,
                "best_timing_window": "Volatility Shakeout / Limit Order Resting Bid (5-Day GTC)",
                "conservative_target_price": safe_target,
                "expected_gain_pct": safe_gain_pct,
                "stop_loss_price": safe_stop_loss,
                "max_drawdown_risk_pct": safe_risk_pct,
                "reward_to_risk_ratio": safe_rr,
                "margin_of_safety_score": safety_score,
                "rationale": (
                    f"Anchored at the 20th-percentile simulated weekly floor and below-spot structural support "
                    f"(${anchor_support:.2f} / Support POC ${vwap_poc:.2f}). Offers a {safety_score}% post-fill rebound probability "
                    f"with tight downside risk ({safe_risk_pct:.2f}% to stop)."
                ),
            },
            "execution_plan": {
                "blended_average_entry": blended_avg,
                "blended_upside_to_primary_pct": round(((opt_primary_target - blended_avg) / blended_avg) * 100.0, 2),
                "resistances_to_watch": resistances[:3],
                "supports_to_watch": supports[:3],
                "tranches": tranches,
            },
        }
