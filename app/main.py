"""
FastAPI Backend for NVDA Quantitative Multi-Factor Price Tracker,
Overnight/Market-Open/1-Week Simulator, Whale Monitor, and Online Self-Tuning Engine.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simulator.engine_service import NVDAQuantService

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="NVDA Multi-Factor Tracker & Self-Tuning Price Simulator",
    description="Tracks NVDA price drivers, whale/institutional block activity, overnight/open/1-week simulations, optimal entry points, and online error-driven model calibration.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

service = NVDAQuantService()


class WhatIfScenarioRequest(BaseModel):
    whale_net_buy_musd: float = Field(0.0, description="Net institutional/whale block purchase (+) or sale (-) in Millions USD")
    overnight_futures_pct: float = Field(0.0, description="Overnight Nasdaq-100 futures shock (%)")
    sector_peers_pct: float = Field(0.0, description="Semiconductor sector peers (SMH/TSM/AMD) shock (%)")
    vix_change_pct: float = Field(0.0, description="Macro CBOE VIX volatility index change (%)")
    sentiment_override: float = Field(0.0, description="News/Catalyst sentiment shift in [-1.0, +1.0]")


class CalibrateRequest(BaseModel):
    recalibrate_backtest: bool = Field(False, description="Whether to re-run full walk-forward historical replay")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/dashboard")
def get_dashboard(refresh: bool = Query(False)) -> Dict[str, Any]:
    return service.get_full_dashboard_payload(force_refresh=refresh)


@app.post("/api/simulate")
def simulate_what_if(req: WhatIfScenarioRequest) -> Dict[str, Any]:
    overrides: Dict[str, float] = {}
    if abs(req.whale_net_buy_musd) > 0.01:
        overrides["whale_net_buy_musd"] = req.whale_net_buy_musd
    if abs(req.overnight_futures_pct) > 0.01:
        overrides["overnight_futures_pct"] = req.overnight_futures_pct
    if abs(req.sector_peers_pct) > 0.01:
        overrides["sector_peers_pct"] = req.sector_peers_pct
    if abs(req.vix_change_pct) > 0.01:
        overrides["vix_change_pct"] = req.vix_change_pct
    if abs(req.sentiment_override) > 0.01:
        overrides["sentiment_override"] = req.sentiment_override

    return service.get_full_dashboard_payload(
        force_refresh=False,
        what_if_overrides=overrides if overrides else None,
    )


@app.post("/api/calibrate")
def calibrate_model(req: CalibrateRequest) -> Dict[str, Any]:
    tune_result = service.trigger_self_tuning(recalibrate_backtest=req.recalibrate_backtest)
    dashboard = service.get_full_dashboard_payload(force_refresh=False)
    return {
        "calibration_result": tune_result,
        "dashboard": dashboard,
    }
