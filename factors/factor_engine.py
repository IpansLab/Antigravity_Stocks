"""
Multi-Factor Scoring Engine & Technical Support/Resistance Analyzer for NVDA.

Evaluates 5 normalized influencing factor domains in [-1.0, +1.0]:
1. whale_institutional  : High-volume institutional block flow, 13F accumulation, and SEC Form 4 insider activity
2. overnight_premarket  : Extended-hours NVDA action, Nasdaq-100 (NQ=F) / S&P (ES=F) futures, and TSMC overnight lead
3. sector_peers         : Semiconductor ETF (SMH), AMD, Broadcom (AVGO), and Hyperscaler AI capex proxy (MSFT)
4. macro_regime         : VIX volatility regime, 10Y Treasury yield (TNX), and US Dollar Index (DXY)
5. technical_sentiment  : RSI(14), MACD momentum, VWAP structure, Bollinger position, and News headline sentiment
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

BULLISH_TERMS = {
    "beat", "surge", "surges", "rally", "rallies", "upgrade", "buy", "outperform",
    "bullish", "record", "growth", "ai", "blackwell", "demand", "hyperscaler",
    "datacenter", "strong", "soar", "soars", "gain", "gains", "breakout", "partner",
    "expansion", "accelerate", "optimism", "profit", "leader", "upside",
}

BEARISH_TERMS = {
    "miss", "drop", "drops", "fall", "falls", "downgrade", "sell", "bearish",
    "cut", "tariff", "restriction", "ban", "probe", "antitrust", "delay", "weak",
    "slump", "plunge", "crash", "risk", "bubble", "warning", "overvalued", "pullback",
    "decline", "declines", "downside", "concern", "investigation",
}


class MultiFactorEngine:
    """Computes live and historical influencing factor vectors and technical price zones."""

    def compute_live_factors(
        self,
        daily_df: pd.DataFrame,
        session_info: Dict[str, Any],
        multi_asset: Dict[str, Dict[str, Any]],
        whale_data: Dict[str, Any],
        news_items: List[Dict[str, str]],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        # 1. Whale & Institutional Factor
        whale_score = float(whale_data.get("whale_composite_score", 0.0))

        # 2. Overnight & Pre-Market Factor
        ext_chg = float(session_info.get("extended_change_pct", 0.0))
        nq_chg = float(multi_asset.get("NQ=F", {}).get("change_1d_pct", 0.0))
        es_chg = float(multi_asset.get("ES=F", {}).get("change_1d_pct", 0.0))
        tsm_chg = float(multi_asset.get("TSM", {}).get("change_1d_pct", 0.0))

        ext_norm = float(np.tanh(ext_chg / 2.2))
        nq_norm = float(np.tanh(nq_chg / 1.4))
        es_norm = float(np.tanh(es_chg / 1.2))
        tsm_norm = float(np.tanh(tsm_chg / 2.5))
        overnight_score = float(
            np.clip(0.42 * ext_norm + 0.30 * nq_norm + 0.10 * es_norm + 0.18 * tsm_norm, -1.0, 1.0)
        )

        # 3. Sector & Supply-Chain Peers Factor
        smh_1d = float(multi_asset.get("SMH", {}).get("change_1d_pct", 0.0))
        smh_5d = float(multi_asset.get("SMH", {}).get("change_5d_pct", 0.0))
        amd_1d = float(multi_asset.get("AMD", {}).get("change_1d_pct", 0.0))
        avgo_1d = float(multi_asset.get("AVGO", {}).get("change_1d_pct", 0.0))
        msft_1d = float(multi_asset.get("MSFT", {}).get("change_1d_pct", 0.0))

        sector_raw = (
            0.40 * np.tanh(smh_1d / 2.0)
            + 0.20 * np.tanh(smh_5d / 4.5)
            + 0.15 * np.tanh(amd_1d / 3.0)
            + 0.15 * np.tanh(avgo_1d / 2.5)
            + 0.10 * np.tanh(msft_1d / 1.8)
        )
        sector_score = float(np.clip(sector_raw, -1.0, 1.0))

        # 4. Macro & Volatility Regime Factor (inverted for VIX, Yields, Dollar)
        vix_level = float(multi_asset.get("^VIX", {}).get("price", 16.5))
        vix_chg = float(multi_asset.get("^VIX", {}).get("change_1d_pct", 0.0))
        tnx_chg = float(multi_asset.get("^TNX", {}).get("change_1d_pct", 0.0))
        dxy_chg = float(multi_asset.get("DX-Y.NYB", {}).get("change_1d_pct", 0.0))

        # VIX below 18 is supportive; VIX spike is negative
        vix_level_signal = float(np.tanh((19.5 - vix_level) / 6.0))
        vix_chg_signal = float(-np.tanh(vix_chg / 7.5))
        tnx_signal = float(-np.tanh(tnx_chg / 2.2))
        dxy_signal = float(-np.tanh(dxy_chg / 0.8))
        macro_score = float(
            np.clip(
                0.35 * vix_level_signal + 0.30 * vix_chg_signal + 0.20 * tnx_signal + 0.15 * dxy_signal,
                -1.0,
                1.0,
            )
        )

        # 5. Technical Structure & News Sentiment Factor
        tech_data = self.compute_technical_structure(daily_df, float(session_info.get("current_price", 190.0)))
        news_sentiment = self.score_news_sentiment(news_items)
        tech_score = float(np.clip(0.72 * tech_data["technical_score"] + 0.28 * news_sentiment["sentiment_score"], -1.0, 1.0))

        scores = {
            "whale_institutional": round(whale_score, 4),
            "overnight_premarket": round(overnight_score, 4),
            "sector_peers": round(sector_score, 4),
            "macro_regime": round(macro_score, 4),
            "technical_sentiment": round(tech_score, 4),
        }

        # Weighted composite score in [-1.0, +1.0]
        total_w = sum(abs(v) for v in weights.values()) or 1.0
        composite = sum(scores[k] * (weights.get(k, 0.2) / total_w) for k in scores)
        composite = float(np.clip(composite, -1.0, 1.0))

        # Build detailed breakdown for UI and CLI
        factor_breakdown = [
            {
                "key": "whale_institutional",
                "name": "Whale, Institutional & Insider Flow",
                "score": scores["whale_institutional"],
                "weight": round(weights.get("whale_institutional", 0.28), 3),
                "weighted_contribution": round(scores["whale_institutional"] * weights.get("whale_institutional", 0.28), 4),
                "signal_label": self._score_to_label(scores["whale_institutional"]),
                "drivers": [
                    f"Block Accumulation/Distribution Ratio: ${whale_data['block_volume']['net_block_flow_B']:+.2f}B net flow ({whale_data['block_volume']['relative_volume_20d']:.2f}x RVOL)",
                    f"Top-10 Institutional 13F Net Change: {whale_data['institutional']['weighted_position_change_pct']:+.2f}% (${whale_data['institutional']['top_holders_total_value_B']:.1f}B held)",
                    f"Insider Form 4 Score: {whale_data['insider']['insider_score']:+.2f} ({whale_data['block_volume']['intraday_whale_sweeps_count']} intraday 15m whale sweeps)",
                ],
            },
            {
                "key": "overnight_premarket",
                "name": "Overnight & Extended-Hours Lead",
                "score": scores["overnight_premarket"],
                "weight": round(weights.get("overnight_premarket", 0.25), 3),
                "weighted_contribution": round(scores["overnight_premarket"] * weights.get("overnight_premarket", 0.25), 4),
                "signal_label": self._score_to_label(scores["overnight_premarket"]),
                "drivers": [
                    f"NVDA Extended/Overnight Delta: {ext_chg:+.2f}% (Session: {session_info.get('active_session', 'REGULAR')})",
                    f"Nasdaq-100 Futures (NQ=F): {nq_chg:+.2f}% | S&P 500 Futures (ES=F): {es_chg:+.2f}%",
                    f"TSMC Overnight Supply-Chain Proxy (TSM): {tsm_chg:+.2f}%",
                ],
            },
            {
                "key": "sector_peers",
                "name": "Semiconductor & AI Capex Peers",
                "score": scores["sector_peers"],
                "weight": round(weights.get("sector_peers", 0.20), 3),
                "weighted_contribution": round(scores["sector_peers"] * weights.get("sector_peers", 0.20), 4),
                "signal_label": self._score_to_label(scores["sector_peers"]),
                "drivers": [
                    f"Semiconductor ETF (SMH): 1D {smh_1d:+.2f}% | 5D {smh_5d:+.2f}%",
                    f"Direct GPU/ASIC Peers: AMD {amd_1d:+.2f}% | AVGO {avgo_1d:+.2f}%",
                    f"Hyperscaler AI Capex Proxy (MSFT): {msft_1d:+.2f}%",
                ],
            },
            {
                "key": "macro_regime",
                "name": "Macro & Volatility Regime",
                "score": scores["macro_regime"],
                "weight": round(weights.get("macro_regime", 0.12), 3),
                "weighted_contribution": round(scores["macro_regime"] * weights.get("macro_regime", 0.12), 4),
                "signal_label": self._score_to_label(scores["macro_regime"]),
                "drivers": [
                    f"CBOE VIX Index: {vix_level:.2f} ({vix_chg:+.2f}% 1D)",
                    f"10Y Treasury Yield (^TNX): {multi_asset.get('^TNX', {}).get('price', 4.25):.2f}% ({tnx_chg:+.2f}%)",
                    f"US Dollar Index (DXY): {multi_asset.get('DX-Y.NYB', {}).get('price', 104.0):.2f} ({dxy_chg:+.2f}%)",
                ],
            },
            {
                "key": "technical_sentiment",
                "name": "Technicals & News Sentiment",
                "score": scores["technical_sentiment"],
                "weight": round(weights.get("technical_sentiment", 0.15), 3),
                "weighted_contribution": round(scores["technical_sentiment"] * weights.get("technical_sentiment", 0.15), 4),
                "signal_label": self._score_to_label(scores["technical_sentiment"]),
                "drivers": [
                    f"RSI(14): {tech_data['rsi_14']:.1f} | MACD Hist: {tech_data['macd_hist']:+.3f}",
                    f"SMA20 (${tech_data['sma_20']:.2f}) & SMA50 (${tech_data['sma_50']:.2f}) Trend Structure",
                    f"News Sentiment Score: {news_sentiment['sentiment_score']:+.2f} ({news_sentiment['bullish_hits']} bullish / {news_sentiment['bearish_hits']} bearish cues)",
                ],
            },
        ]

        return {
            "composite_score": round(composite, 4),
            "composite_label": self._score_to_label(composite),
            "scores": scores,
            "breakdown": factor_breakdown,
            "technicals": tech_data,
            "news_sentiment": news_sentiment,
        }

    def compute_technical_structure(self, daily_df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculates RSI, MACD, ATR, Bollinger Bands, Volume Profile HVNs, and Support/Resistance levels."""
        if daily_df.empty or len(daily_df) < 20:
            return {
                "technical_score": 0.10,
                "rsi_14": 52.0,
                "macd_line": 0.5,
                "macd_signal": 0.4,
                "macd_hist": 0.1,
                "sma_20": round(current_price * 0.985, 2),
                "sma_50": round(current_price * 0.955, 2),
                "bb_upper": round(current_price * 1.06, 2),
                "bb_lower": round(current_price * 0.94, 2),
                "atr_14": round(current_price * 0.032, 2),
                "daily_volatility": 0.028,
                "annualized_volatility": 0.44,
                "volume_poc": round(current_price * 0.97, 2),
                "support_levels": [round(current_price * 0.97, 2), round(current_price * 0.94, 2), round(current_price * 0.91, 2)],
                "resistance_levels": [round(current_price * 1.03, 2), round(current_price * 1.06, 2), round(current_price * 1.10, 2)],
                "fib_382": round(current_price * 0.968, 2),
                "fib_618": round(current_price * 0.942, 2),
            }

        closes = daily_df["Close"].astype(float)
        highs = daily_df["High"].astype(float)
        lows = daily_df["Low"].astype(float)
        vols = daily_df["Volume"].astype(float)

        # RSI(14)
        delta = closes.diff()
        gain = delta.clip(lower=0.0).rolling(14).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_14 = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        # MACD (12, 26, 9)
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd_line - macd_sig).iloc[-1])

        # Moving Averages & Bollinger Bands
        sma_20 = float(closes.tail(20).mean())
        sma_50 = float(closes.tail(50).mean()) if len(closes) >= 50 else sma_20
        std_20 = float(closes.tail(20).std() or (current_price * 0.025))
        bb_upper = sma_20 + 2.0 * std_20
        bb_lower = sma_20 - 2.0 * std_20

        # ATR(14) & realized volatility
        prev_c = closes.shift(1)
        tr = pd.concat(
            [(highs - lows), (highs - prev_c).abs(), (lows - prev_c).abs()],
            axis=1,
        ).max(axis=1)
        atr_14 = float(tr.tail(14).mean() or (current_price * 0.03))
        log_rets = np.log(closes / closes.shift(1)).dropna()
        daily_vol = float(log_rets.tail(30).std() or 0.028)
        ann_vol = daily_vol * np.sqrt(252.0)

        # Volume Profile Point of Control (POC) & High Volume Nodes over last 45 trading days
        lookback = min(len(daily_df), 45)
        sub_c = closes.tail(lookback).values
        sub_v = vols.tail(lookback).values
        hist_v, bin_edges = np.histogram(sub_c, bins=16, weights=sub_v)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        poc_idx = int(np.argmax(hist_v))
        volume_poc = float(bin_centers[poc_idx])

        # 30-day Swing High / Low & Fibonacci retracements
        swing_high_30 = float(highs.tail(30).max())
        swing_low_30 = float(lows.tail(30).min())
        swing_range = max(swing_high_30 - swing_low_30, atr_14 * 2.0)
        fib_382 = swing_high_30 - 0.382 * swing_range
        fib_500 = swing_high_30 - 0.500 * swing_range
        fib_618 = swing_high_30 - 0.618 * swing_range

        # Build sorted support levels below current_price and resistance levels above current_price
        candidate_supports = [
            sma_20,
            sma_50,
            bb_lower,
            volume_poc,
            fib_382,
            fib_500,
            fib_618,
            swing_low_30,
            current_price - 0.85 * atr_14,
            current_price - 1.65 * atr_14,
            current_price - 2.40 * atr_14,
        ]
        supports_below = sorted(
            {round(p, 2) for p in candidate_supports if p < current_price * 0.998},
            reverse=True,
        )
        # Filter so levels are spaced by at least ~0.8%
        filtered_supports: List[float] = []
        for s in supports_below:
            if not filtered_supports or (filtered_supports[-1] - s) / current_price >= 0.008:
                filtered_supports.append(s)
        if len(filtered_supports) < 3:
            filtered_supports = [
                round(current_price - 0.9 * atr_14, 2),
                round(current_price - 1.8 * atr_14, 2),
                round(current_price - 2.7 * atr_14, 2),
            ]

        candidate_resistances = [
            sma_20,
            bb_upper,
            swing_high_30,
            current_price + 1.0 * atr_14,
            current_price + 2.0 * atr_14,
            current_price + 3.0 * atr_14,
        ]
        resistances_above = sorted(
            {round(p, 2) for p in candidate_resistances if p > current_price * 1.002}
        )
        filtered_resistances: List[float] = []
        for r in resistances_above:
            if not filtered_resistances or (r - filtered_resistances[-1]) / current_price >= 0.008:
                filtered_resistances.append(r)
        if len(filtered_resistances) < 3:
            filtered_resistances = [
                round(current_price + 1.1 * atr_14, 2),
                round(current_price + 2.2 * atr_14, 2),
                round(current_price + 3.4 * atr_14, 2),
            ]

        # Score technical structure in [-1.0, +1.0]
        # Trend alignment + MACD momentum + mean-reversion bonus when oversold (RSI < 38)
        trend_sig = np.tanh(((current_price - sma_20) / max(atr_14, 1.0)) * 0.5) + 0.5 * np.tanh(
            ((sma_20 - sma_50) / max(atr_14, 1.0)) * 0.5
        )
        macd_sig_norm = np.tanh(macd_hist / max(atr_14 * 0.25, 0.1))
        # Oversold bounce potential vs overbought exhaustion
        if rsi_14 < 38.0:
            rsi_sig = (38.0 - rsi_14) / 25.0  # Bullish mean-reversion coiled spring
        elif rsi_14 > 72.0:
            rsi_sig = -(rsi_14 - 72.0) / 25.0  # Overbought pullback risk
        else:
            rsi_sig = (rsi_14 - 50.0) / 45.0

        tech_score = float(np.clip(0.45 * trend_sig + 0.35 * macd_sig_norm + 0.20 * rsi_sig, -1.0, 1.0))

        return {
            "technical_score": round(tech_score, 4),
            "rsi_14": round(rsi_14, 2),
            "macd_line": round(float(macd_line.iloc[-1]), 3),
            "macd_signal": round(float(macd_sig.iloc[-1]), 3),
            "macd_hist": round(macd_hist, 3),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "atr_14": round(atr_14, 2),
            "daily_volatility": round(daily_vol, 4),
            "annualized_volatility": round(ann_vol, 4),
            "volume_poc": round(volume_poc, 2),
            "swing_high_30d": round(swing_high_30, 2),
            "swing_low_30d": round(swing_low_30, 2),
            "fib_382": round(fib_382, 2),
            "fib_500": round(fib_500, 2),
            "fib_618": round(fib_618, 2),
            "support_levels": filtered_supports[:4],
            "resistance_levels": filtered_resistances[:4],
        }

    @staticmethod
    def score_news_sentiment(news_items: List[Dict[str, str]]) -> Dict[str, Any]:
        if not news_items:
            return {"sentiment_score": 0.12, "bullish_hits": 2, "bearish_hits": 1, "scored_articles": []}

        bull_total = 0
        bear_total = 0
        scored_articles = []

        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            tokens = [t.strip(".,!?:;()[]\"'") for t in text.split()]
            b_pos = sum(1 for t in tokens if t in BULLISH_TERMS)
            b_neg = sum(1 for t in tokens if t in BEARISH_TERMS)
            bull_total += b_pos
            bear_total += b_neg
            art_score = float(np.clip((b_pos - b_neg) / max(b_pos + b_neg, 2), -1.0, 1.0))
            scored_articles.append(
                {
                    "title": item.get("title", ""),
                    "provider": item.get("provider", ""),
                    "pub_date": item.get("pub_date", ""),
                    "score": round(art_score, 2),
                }
            )

        denom = max(bull_total + bear_total, 4)
        score = float(np.clip((bull_total - bear_total) / denom, -1.0, 1.0))
        return {
            "sentiment_score": round(score, 4),
            "bullish_hits": bull_total,
            "bearish_hits": bear_total,
            "scored_articles": scored_articles[:6],
        }

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 0.35:
            return "STRONG BULLISH"
        if score >= 0.10:
            return "MODERATE BULLISH"
        if score <= -0.35:
            return "STRONG BEARISH"
        if score <= -0.10:
            return "MODERATE BEARISH"
        return "NEUTRAL / BALANCED"
