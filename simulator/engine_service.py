"""
Unified Orchestration Service for NVDA Multi-Factor Price Tracker,
Overnight/Market-Open/1-Week Simulator, Whale Monitor, and Online Self-Tuning Engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from data.database import TrackerDatabase
from data.market_fetcher import MarketDataFetcher
from data.whale_tracker import WhaleActivityTracker
from factors.factor_engine import MultiFactorEngine
from simulator.calibrator import AdaptiveCalibrator
from simulator.predictor import NVDAPricePredictor


class NVDAQuantService:
    """High-level service coordinating live market ingestion, factor scoring, calibration, and simulation."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db = TrackerDatabase(db_path) if db_path else TrackerDatabase()
        self.fetcher = MarketDataFetcher(symbol="NVDA")
        self.whale_tracker = WhaleActivityTracker(symbol="NVDA")
        self.factor_engine = MultiFactorEngine()
        self.calibrator = AdaptiveCalibrator(self.db)
        self.predictor = NVDAPricePredictor(self.db)

    def get_full_dashboard_payload(
        self,
        force_refresh: bool = False,
        what_if_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Fetches live NVDA market state, scores all 5 factors, ensures calibration, and runs full simulation."""
        daily_df = self.fetcher.get_nvda_daily(force_refresh=force_refresh)
        intra_df = self.fetcher.get_nvda_intraday(force_refresh=force_refresh)
        session_info = self.fetcher.get_session_breakdown(force_refresh=force_refresh)
        multi_asset = self.fetcher.get_multi_asset_context(force_refresh=force_refresh)
        news_items = self.fetcher.get_news_headlines(force_refresh=force_refresh)

        current_price = float(session_info.get("current_price", 190.0))

        # Ensure historical walk-forward warm-start has been executed
        calibration_state = self.calibrator.ensure_warm_started(daily_df)
        weights = dict(calibration_state.get("weights", {}))

        # Analyze Whale, Institutional 13F, Insider Form 4, and Block Volume Flows
        whale_data = self.whale_tracker.analyze_whale_activity(
            daily_df=daily_df,
            intra_df=intra_df,
            current_price=current_price,
            force_refresh=force_refresh,
        )
        self.db.upsert_whale_events(whale_data.get("events", []))

        # Compute all 5 influencing factor groups
        factor_result = self.factor_engine.compute_live_factors(
            daily_df=daily_df,
            session_info=session_info,
            multi_asset=multi_asset,
            whale_data=whale_data,
            news_items=news_items,
            weights=weights,
        )

        # Save price & factor snapshots on fresh live runs
        now_iso = datetime.now(timezone.utc).isoformat()
        if not what_if_overrides:
            self.db.save_price_snapshot(
                timestamp=now_iso,
                session_type=str(session_info.get("active_session", "REGULAR")),
                price=current_price,
                open_price=session_info.get("day_open"),
                high_price=session_info.get("day_high"),
                low_price=session_info.get("day_low"),
                prev_close=session_info.get("prev_regular_close"),
                volume=session_info.get("latest_daily_volume"),
                vwap=session_info.get("vwap_intraday"),
            )
            self.db.save_factor_snapshot(
                timestamp=now_iso,
                composite_score=factor_result["composite_score"],
                scores=factor_result["scores"],
                details={"session": session_info.get("active_session")},
            )

        # Run Overnight, Market Open, Intraday Close, and 1-Week (5-day) simulations + Entry Optimization
        simulation = self.predictor.run_full_simulation(
            session_info=session_info,
            factor_result=factor_result,
            calibration_state=calibration_state,
            log_to_db=(not bool(what_if_overrides)),
            what_if_overrides=what_if_overrides,
        )

        calibration_history = self.db.get_calibration_history(limit=30)
        candles_1d = self.fetcher.get_ohlcv_candles(timeframe="1D", limit=220, force_refresh=force_refresh)

        return {
            "generated_at": now_iso,
            "market": session_info,
            "chart_candles": candles_1d,
            "multi_asset_peers": multi_asset,
            "whale_activity": whale_data,
            "factors": factor_result,
            "simulation": simulation,
            "calibration": {
                "current_state": calibration_state,
                "history": calibration_history,
            },
        }

    def trigger_self_tuning(self, recalibrate_backtest: bool = False) -> Dict[str, Any]:
        """
        Triggers an explicit online self-tuning cycle comparing predictions with current NVDA market price
        (and optionally re-running the walk-forward historical replay).
        """
        daily_df = self.fetcher.get_nvda_daily(force_refresh=True)
        intra_df = self.fetcher.get_nvda_intraday(force_refresh=True)
        session_info = self.fetcher.get_session_breakdown(force_refresh=False)
        multi_asset = self.fetcher.get_multi_asset_context(force_refresh=False)
        news_items = self.fetcher.get_news_headlines(force_refresh=False)
        current_price = float(session_info.get("current_price", 190.0))

        if recalibrate_backtest:
            self.calibrator.run_walk_forward_backtest(daily_df, lookback_days=45)

        state_before = self.db.get_latest_calibration_state()
        whale_data = self.whale_tracker.analyze_whale_activity(daily_df, intra_df, current_price)
        factor_result = self.factor_engine.compute_live_factors(
            daily_df=daily_df,
            session_info=session_info,
            multi_asset=multi_asset,
            whale_data=whale_data,
            news_items=news_items,
            weights=state_before["weights"],
        )

        updated_state = self.calibrator.fine_tune_with_live_price(
            current_price=current_price,
            current_factors=factor_result["scores"],
            daily_df=daily_df,
        )

        return {
            "status": "ok",
            "current_price": current_price,
            "previous_state": state_before,
            "updated_state": updated_state,
            "history": self.db.get_calibration_history(limit=30),
        }
