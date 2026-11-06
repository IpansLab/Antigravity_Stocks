"""
SQLite persistence layer for NVDA price tracking, multi-factor snapshots,
predictions vs. actual market prices, whale events, and online self-tuning weight history.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(DB_DIR, "nvda_tracker.db")


DEFAULT_WEIGHTS: Dict[str, float] = {
    "whale_institutional": 0.28,
    "overnight_premarket": 0.25,
    "sector_peers": 0.20,
    "macro_regime": 0.12,
    "technical_sentiment": 0.15,
}


class TrackerDatabase:
    """Manages SQLite persistence for predictions, actuals, factor history, and model calibration."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL DEFAULT 'NVDA',
                    timestamp TEXT NOT NULL,
                    session_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    prev_close REAL,
                    volume INTEGER,
                    vwap REAL
                );

                CREATE INDEX IF NOT EXISTS idx_price_ts ON price_snapshots(timestamp);

                CREATE TABLE IF NOT EXISTS factor_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL DEFAULT 'NVDA',
                    composite_score REAL NOT NULL,
                    whale_institutional REAL NOT NULL,
                    overnight_premarket REAL NOT NULL,
                    sector_peers REAL NOT NULL,
                    macro_regime REAL NOT NULL,
                    technical_sentiment REAL NOT NULL,
                    details_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    horizon_type TEXT NOT NULL,
                    horizon_days REAL NOT NULL DEFAULT 1.0,
                    base_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    predicted_return_pct REAL NOT NULL,
                    low_68 REAL NOT NULL,
                    high_68 REAL NOT NULL,
                    low_95 REAL NOT NULL,
                    high_95 REAL NOT NULL,
                    factor_scores_json TEXT NOT NULL,
                    weights_used_json TEXT NOT NULL,
                    actual_price REAL,
                    actual_return_pct REAL,
                    abs_error REAL,
                    pct_error REAL,
                    direction_correct INTEGER,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    calibrated INTEGER NOT NULL DEFAULT 0,
                    is_backtest INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_date, horizon_type);
                CREATE INDEX IF NOT EXISTS idx_pred_resolved ON predictions(resolved, calibrated);

                CREATE TABLE IF NOT EXISTS model_calibration_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    updated_at TEXT NOT NULL,
                    step_type TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    bias_drift REAL NOT NULL DEFAULT 0.0,
                    vol_multiplier REAL NOT NULL DEFAULT 1.0,
                    learning_rate REAL NOT NULL DEFAULT 0.08,
                    rolling_mae REAL NOT NULL DEFAULT 0.0,
                    rolling_mape REAL NOT NULL DEFAULT 0.0,
                    directional_accuracy REAL NOT NULL DEFAULT 0.0,
                    baseline_mae REAL NOT NULL DEFAULT 0.0,
                    samples_evaluated INTEGER NOT NULL DEFAULT 0,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS whale_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_date TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    role_or_source TEXT,
                    action_direction TEXT NOT NULL,
                    shares REAL NOT NULL,
                    estimated_value_usd REAL NOT NULL,
                    impact_score REAL NOT NULL,
                    summary TEXT NOT NULL,
                    UNIQUE(event_date, entity_name, shares, event_type)
                );
                """
            )

            # Seed initial calibration state if empty
            cur = conn.execute("SELECT COUNT(*) AS cnt FROM model_calibration_state")
            if cur.fetchone()["cnt"] == 0:
                now_str = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    INSERT INTO model_calibration_state (
                        updated_at, step_type, weights_json, bias_drift,
                        vol_multiplier, learning_rate, rolling_mae, rolling_mape,
                        directional_accuracy, baseline_mae, samples_evaluated, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_str,
                        "INITIAL_DEFAULT",
                        json.dumps(DEFAULT_WEIGHTS),
                        0.0,
                        1.0,
                        0.08,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0,
                        "Initial prior weights before historical warm-start and live error tuning.",
                    ),
                )

    def save_price_snapshot(
        self,
        timestamp: str,
        session_type: str,
        price: float,
        open_price: Optional[float] = None,
        high_price: Optional[float] = None,
        low_price: Optional[float] = None,
        prev_close: Optional[float] = None,
        volume: Optional[int] = None,
        vwap: Optional[float] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO price_snapshots (
                    symbol, timestamp, session_type, price,
                    open_price, high_price, low_price, prev_close, volume, vwap
                ) VALUES ('NVDA', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    session_type,
                    price,
                    open_price,
                    high_price,
                    low_price,
                    prev_close,
                    volume,
                    vwap,
                ),
            )

    def save_factor_snapshot(
        self,
        timestamp: str,
        composite_score: float,
        scores: Dict[str, float],
        details: Dict[str, Any],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO factor_snapshots (
                    timestamp, symbol, composite_score,
                    whale_institutional, overnight_premarket, sector_peers,
                    macro_regime, technical_sentiment, details_json
                ) VALUES (?, 'NVDA', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    composite_score,
                    float(scores.get("whale_institutional", 0.0)),
                    float(scores.get("overnight_premarket", 0.0)),
                    float(scores.get("sector_peers", 0.0)),
                    float(scores.get("macro_regime", 0.0)),
                    float(scores.get("technical_sentiment", 0.0)),
                    json.dumps(details),
                ),
            )

    def log_prediction(
        self,
        created_at: str,
        target_date: str,
        horizon_type: str,
        horizon_days: float,
        base_price: float,
        predicted_price: float,
        low_68: float,
        high_68: float,
        low_95: float,
        high_95: float,
        factor_scores: Dict[str, float],
        weights_used: Dict[str, float],
        actual_price: Optional[float] = None,
        is_backtest: bool = False,
    ) -> int:
        pred_ret = ((predicted_price - base_price) / base_price) * 100.0 if base_price > 0 else 0.0
        resolved = 0
        calibrated = 0
        act_ret = None
        abs_err = None
        pct_err = None
        dir_ok = None

        if actual_price is not None and actual_price > 0:
            resolved = 1
            act_ret = ((actual_price - base_price) / base_price) * 100.0 if base_price > 0 else 0.0
            abs_err = abs(actual_price - predicted_price)
            pct_err = (abs_err / actual_price) * 100.0
            pred_sign = 1 if predicted_price >= base_price else -1
            act_sign = 1 if actual_price >= base_price else -1
            dir_ok = 1 if pred_sign == act_sign else 0

        with self._conn() as conn:
            # Avoid duplicate backtest entries for identical target_date + horizon_type
            if is_backtest:
                existing = conn.execute(
                    "SELECT id FROM predictions WHERE target_date = ? AND horizon_type = ? AND is_backtest = 1",
                    (target_date, horizon_type),
                ).fetchone()
                if existing:
                    return int(existing["id"])

            cur = conn.execute(
                """
                INSERT INTO predictions (
                    created_at, target_date, horizon_type, horizon_days,
                    base_price, predicted_price, predicted_return_pct,
                    low_68, high_68, low_95, high_95,
                    factor_scores_json, weights_used_json,
                    actual_price, actual_return_pct, abs_error, pct_error,
                    direction_correct, resolved, calibrated, is_backtest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    target_date,
                    horizon_type,
                    horizon_days,
                    base_price,
                    predicted_price,
                    pred_ret,
                    low_68,
                    high_68,
                    low_95,
                    high_95,
                    json.dumps(factor_scores),
                    json.dumps(weights_used),
                    actual_price,
                    act_ret,
                    abs_err,
                    pct_err,
                    dir_ok,
                    resolved,
                    calibrated,
                    1 if is_backtest else 0,
                ),
            )
            return int(cur.lastrowid)

    def resolve_prediction(self, pred_id: int, actual_price: float) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT base_price, predicted_price FROM predictions WHERE id = ?",
                (pred_id,),
            ).fetchone()
            if not row or actual_price <= 0:
                return
            base_price = float(row["base_price"])
            predicted_price = float(row["predicted_price"])
            act_ret = ((actual_price - base_price) / base_price) * 100.0 if base_price > 0 else 0.0
            abs_err = abs(actual_price - predicted_price)
            pct_err = (abs_err / actual_price) * 100.0
            pred_sign = 1 if predicted_price >= base_price else -1
            act_sign = 1 if actual_price >= base_price else -1
            dir_ok = 1 if pred_sign == act_sign else 0

            conn.execute(
                """
                UPDATE predictions
                SET actual_price = ?, actual_return_pct = ?, abs_error = ?,
                    pct_error = ?, direction_correct = ?, resolved = 1
                WHERE id = ?
                """,
                (actual_price, act_ret, abs_err, pct_err, dir_ok, pred_id),
            )

    def get_latest_calibration_state(self) -> Dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM model_calibration_state ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not row:
                return {
                    "weights": dict(DEFAULT_WEIGHTS),
                    "bias_drift": 0.0,
                    "vol_multiplier": 1.0,
                    "learning_rate": 0.08,
                    "rolling_mae": 0.0,
                    "rolling_mape": 0.0,
                    "directional_accuracy": 0.0,
                    "baseline_mae": 0.0,
                    "samples_evaluated": 0,
                    "step_type": "INITIAL_DEFAULT",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "notes": "",
                }
            data = dict(row)
            data["weights"] = json.loads(data.pop("weights_json"))
            return data

    def save_calibration_state(
        self,
        step_type: str,
        weights: Dict[str, float],
        bias_drift: float,
        vol_multiplier: float,
        learning_rate: float,
        rolling_mae: float,
        rolling_mape: float,
        directional_accuracy: float,
        baseline_mae: float,
        samples_evaluated: int,
        notes: str = "",
    ) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO model_calibration_state (
                    updated_at, step_type, weights_json, bias_drift,
                    vol_multiplier, learning_rate, rolling_mae, rolling_mape,
                    directional_accuracy, baseline_mae, samples_evaluated, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str,
                    step_type,
                    json.dumps(weights),
                    bias_drift,
                    vol_multiplier,
                    learning_rate,
                    rolling_mae,
                    rolling_mape,
                    directional_accuracy,
                    baseline_mae,
                    samples_evaluated,
                    notes,
                ),
            )

    def get_calibration_history(self, limit: int = 40) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM model_calibration_state ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
            out = []
            for r in rows:
                item = dict(r)
                item["weights"] = json.loads(item.pop("weights_json"))
                out.append(item)
            return out

    def get_recent_predictions(
        self, limit: int = 35, resolved_only: bool = False
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM predictions "
        if resolved_only:
            query += "WHERE resolved = 1 "
        query += "ORDER BY target_date DESC, id DESC LIMIT ?"
        with self._conn() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["factor_scores"] = json.loads(d.pop("factor_scores_json"))
                d["weights_used"] = json.loads(d.pop("weights_used_json"))
                out.append(d)
            return out

    def upsert_whale_events(self, events: List[Dict[str, Any]]) -> None:
        with self._conn() as conn:
            for ev in events:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO whale_events (
                        event_date, event_type, entity_name, role_or_source,
                        action_direction, shares, estimated_value_usd, impact_score, summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(ev.get("event_date", "")),
                        str(ev.get("event_type", "INSTITUTIONAL")),
                        str(ev.get("entity_name", "Unknown")),
                        str(ev.get("role_or_source", "")),
                        str(ev.get("action_direction", "BUY")),
                        float(ev.get("shares", 0.0)),
                        float(ev.get("estimated_value_usd", 0.0)),
                        float(ev.get("impact_score", 0.0)),
                        str(ev.get("summary", "")),
                    ),
                )
