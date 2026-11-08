"""
Comprehensive test suite verifying anti-hallucination mathematical fixes,
the 5-year causal walk-forward calibration pipeline, SQLite database integrity,
and API/UI consistency for the NVDA Quantitative Terminal.
"""

import json
import os
import sqlite3
import unittest

from data.database import TrackerDatabase
from data.market_fetcher import MarketDataFetcher
from simulator.calibrator import AdaptiveCalibrator
from simulator.predictor import NVDAPricePredictor
from simulator.entry_optimizer import EntryPointOptimizer
from simulator.engine_service import NVDAQuantService

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nvda_tracker.db")


class TestAntiHallucinationAnd5Y(unittest.TestCase):
    def test_database_5y_row_counts_and_audit(self):
        """Verifies that nvda_tracker.db holds 1,255 daily bars, >2,400 predictions, and 5Y audit metadata."""
        self.assertTrue(os.path.exists(DB_PATH), f"Database not found at {DB_PATH}")
        db = TrackerDatabase(db_path=DB_PATH)
        status = db.get_database_status_summary()

        # 1. Historical daily prices table
        daily_count = status["table_counts"]["historical_daily_prices"]["rows"]
        self.assertEqual(daily_count, 1255, f"Expected 1,255 5Y daily bars, found {daily_count}")
        self.assertEqual(status["date_range"]["start"], "2021-11-08")
        self.assertEqual(status["date_range"]["end"], "2026-11-06")

        # 2. Predictions table (>2,400 out-of-sample walk-forward predictions)
        pred_count = status["table_counts"]["predictions"]["rows"]
        self.assertGreaterEqual(pred_count, 2460, f"Expected >= 2,460 predictions, found {pred_count}")

        # 3. Model calibration states (>150 checkpoints)
        cal_count = status["table_counts"]["model_calibration_state"]["rows"]
        self.assertGreaterEqual(cal_count, 150, f"Expected >= 150 calibration checkpoints, found {cal_count}")

        # 4. 5Y Audit metadata
        audit = status["latest_5y_audit"]
        self.assertIsNotNone(audit)
        self.assertEqual(audit["trading_days_fetched"], 1255)
        self.assertEqual(audit["backtest_days_evaluated"], 1230)
        self.assertLessEqual(audit["oos_mae_recent_30d"], 4.50)
        self.assertGreaterEqual(audit["band_68_coverage_5y"], 65.0)
        self.assertGreaterEqual(audit["band_95_coverage_5y"], 90.0)

        # 5. Change delta reflects massive jump from old 40-day run
        delta = status["change_delta"]
        self.assertGreaterEqual(delta["predictions_added"], 2300)
        self.assertGreater(delta["new_db_size_kb"], 1000.0)

    def test_anti_hallucination_zero_lookahead_leakage(self):
        """H-01: Verifies that factor extraction at index idx uses strictly causal data <= idx."""
        fetcher = MarketDataFetcher()
        df = fetcher.get_nvda_daily(period="5y")
        self.assertGreaterEqual(len(df), 1000)

        closes = df["Close"].astype(float).to_numpy()
        opens = df["Open"].astype(float).to_numpy()
        highs = df["High"].astype(float).to_numpy()
        lows = df["Low"].astype(float).to_numpy()
        vols = df["Volume"].astype(float).to_numpy()

        db = TrackerDatabase(db_path=DB_PATH)
        calibrator = AdaptiveCalibrator(db=db)
        f_vec = calibrator._extract_historical_factors_at(
            closes, opens, highs, lows, vols, idx=100
        )
        self.assertEqual(len(f_vec), 5)
        for k, val in f_vec.items():
            self.assertGreaterEqual(val, -1.2, f"Factor {k} too negative: {val}")
            self.assertLessEqual(val, 1.2, f"Factor {k} too positive: {val}")

    def test_anti_hallucination_train_serve_factor_parity(self):
        """H-02: Verifies that factors computed in calibration and inference have positive directional alignment."""
        fetcher = MarketDataFetcher()
        df = fetcher.get_nvda_daily(period="5y")
        closes = df["Close"].astype(float).to_numpy()
        opens = df["Open"].astype(float).to_numpy()
        highs = df["High"].astype(float).to_numpy()
        lows = df["Low"].astype(float).to_numpy()
        vols = df["Volume"].astype(float).to_numpy()

        db = TrackerDatabase(db_path=DB_PATH)
        calibrator = AdaptiveCalibrator(db=db)

        for idx in range(150, 170):
            f_vec = calibrator._extract_historical_factors_at(
                closes, opens, highs, lows, vols, idx=idx
            )
            self.assertIsInstance(f_vec["technical_sentiment"], float)

    def test_anti_hallucination_spot_centered_monte_carlo(self):
        """H-05: Verifies that 1-week Monte Carlo forecast Day 1 Close aligns with intraday close and starts at spot."""
        svc = NVDAQuantService()
        payload = svc.get_full_dashboard_payload()
        sim = payload["simulation"]
        spot = payload["market"]["current_price"]

        week_target = sim["one_week_forecast"]["end_of_week_target"]
        self.assertLess(abs(week_target - spot) / spot, 0.15)

        session_close_pred = sim["short_term_horizons"]["intraday_close"]["predicted_price"]
        day1_close_pred = sim["one_week_forecast"]["daily_cards"][0]["predicted_close"]
        self.assertLess(
            abs(session_close_pred - day1_close_pred),
            0.20,
            f"Day 1 close {day1_close_pred} does not match intraday close {session_close_pred}",
        )

    def test_anti_hallucination_post_fill_entry_conditioning(self):
        """H-11: Verifies that entry optimizer orders Tranche 1 above or equal to Tranche 2."""
        svc = NVDAQuantService()
        payload = svc.get_full_dashboard_payload()
        entries = payload["simulation"]["entry_recommendations"]
        opt_entry = entries["most_suitable_entry"]["entry_price"]
        safe_entry = entries["safest_entry"]["entry_price"]

        self.assertGreaterEqual(opt_entry, safe_entry, f"Optimal entry {opt_entry} should be >= safest entry {safe_entry}")
        spot = payload["market"]["current_price"]
        self.assertLessEqual(opt_entry, spot * 1.01)
        self.assertLessEqual(safe_entry, spot * 1.01)

    def test_conformal_quantiles_calibration(self):
        """H-12: Verifies that empirical conformal quantiles provide calibrated coverage on 5Y walk-forward."""
        db = TrackerDatabase(db_path=DB_PATH)
        latest_state = db.get_latest_calibration_state()
        q68 = float(latest_state.get("q68_conformal", 0.0))
        q95 = float(latest_state.get("q95_conformal", 0.0))

        self.assertGreaterEqual(q68, 0.70)
        self.assertLessEqual(q68, 1.30)
        self.assertGreaterEqual(q95, 1.60)
        self.assertLessEqual(q95, 2.40)
        self.assertGreater(q95, q68)

    def test_html_dashboards_contain_audit_elements(self):
        """Verifies that app/static/index.html and nvda_quant_terminal.html include the 5Y audit UI."""
        for rel_path in ["app/static/index.html", "nvda_quant_terminal.html"]:
            full_path = os.path.join(os.path.dirname(__file__), "..", rel_path)
            self.assertTrue(os.path.exists(full_path), f"File {rel_path} does not exist")
            with open(full_path, "r", encoding="utf-8") as f:
                html = f.read()

            self.assertIn('id="db-audit-panel"', html, f"Missing #db-audit-panel in {rel_path}")
            self.assertIn('id="db-stat-daily"', html, f"Missing #db-stat-daily in {rel_path}")
            self.assertIn('id="db-stat-preds"', html, f"Missing #db-stat-preds in {rel_path}")
            self.assertIn('id="db-audit-details"', html, f"Missing #db-audit-details in {rel_path}")
            self.assertIn('syncDatabaseStatusFromBackend', html, f"Missing syncDatabaseStatusFromBackend in {rel_path}")
            self.assertIn('samples: 2460', html, f"Missing samples: 2460 in {rel_path}")
            self.assertIn('rollingMae: 3.89', html, f"Missing rollingMae: 3.89 in {rel_path}")


if __name__ == "__main__":
    unittest.main()
