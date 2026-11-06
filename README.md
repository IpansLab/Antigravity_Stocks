# Antigravity_Stocks — NVDA Multi-Factor Stock Tracker & Self-Tuning Price Simulator

Quantitative stock price tracker, institutional/whale block monitor, multi-horizon price simulator (**Overnight**, **Market Open**, **Intraday Close**, and **1-Week 5-Trading-Day Trend**), **Entry Point Optimizer**, and **Online Error-Driven Self-Tuning Engine** focused on **NVIDIA (`NVDA`)**.

---

## Core Capabilities

1. **Multi-Factor Price Influence Engine (`factors/factor_engine.py`)**
   Scores 5 normalized influencing domains in `[-1.0, +1.0]`:
   - **Whale, Institutional & Insider Activity (`whale_institutional`):** Institutional 13F position additions/trims (BlackRock, Vanguard, FMR, etc.), SEC Form 4 executive transactions, 10-day & 15-minute high-volume block accumulation/distribution (`>1.6σ` volume spikes), and post-capitulation institutional dip absorption.
   - **Overnight & Extended-Hours Lead (`overnight_premarket`):** Live NVDA pre-market / post-market price & volume action, Nasdaq-100 (`NQ=F`) & S&P 500 (`ES=F`) futures, and TSMC (`TSM`) overnight supply-chain lead.
   - **Semiconductor & AI Capex Peers (`sector_peers`):** `SMH`, `AMD`, `AVGO`, and hyperscaler AI capex proxy (`MSFT`).
   - **Macro & Volatility Regime (`macro_regime`):** CBOE Volatility Index (`^VIX`), 10-Year Treasury Yield (`^TNX`), and US Dollar Index (`DX-Y.NYB`).
   - **Technicals & News Sentiment (`technical_sentiment`):** RSI(14), MACD(12,26,9), SMA20/SMA50, Bollinger Bands, Volume Profile Point-of-Control (POC), ATR(14), and financial headline sentiment scoring.

2. **Multi-Horizon Monte Carlo + Jump-Diffusion Simulator (`simulator/predictor.py`)**
   - **Overnight Session Prediction** (`20:00–04:00 ET`)
   - **Next Market Open Prediction** (`09:30 ET Opening Bell`)
   - **Intraday Session Close Prediction** (`16:00 ET`)
   - **1-Week (5-Trading-Day) Forward Trend Trajectory** across 2,000 Monte Carlo Jump-Diffusion paths with 68% and 90%/95% confidence intervals.
   - **Live vs. Simulated Price Comparison** & **Interactive "What-If" Scenario Simulator** (test the impact of a major firm/whale buying or selling `$50M–$1.5B` of NVDA stock, overnight futures shocks, or sector moves).

3. **Profit & Safety Entry Point Optimizer (`simulator/entry_optimizer.py`)**
   - **Most Suitable Entry Point (Optimal Profit):** Maximizes `Fill Probability × Expected Post-Entry Upside / Stop Risk`, providing exact limit entry price, timing window, primary & stretch take-profit targets, stop-loss, and Reward-to-Risk ratio.
   - **Safest Entry Point (Maximum Margin of Safety):** Anchors at the confluence of the 20th-percentile simulated weekly floor, Volume Profile POC, and structural ATR support to minimize drawdown risk.
   - **3-Tranche Capital Scaling Plan:** Splits capital across Tactical Starter (30%), Most Suitable Core (45%), and Safest Floor (25%).

4. **Adaptive Self-Tuning Calibration Engine (`simulator/calibrator.py`)**
   - Logs predictions and actual NVDA prices in SQLite (`data/nvda_tracker.db`).
   - Runs a **Walk-Forward Historical Warm-Start** on startup and supports **Continuous Online Fine-Tuning** via Ridge-Regularized Gradient Descent + Kalman-style drift bias correction whenever new market prices arrive.

---

## Quick Start

### 1. Install Dependencies
```bash
uv venv .venv
uv pip install -r requirements.txt
```

### 2. Run the CLI Tracker & Simulator
```bash
# Standard live report (Market state, 5 factors, Whale feed, Overnight/Open/1-Week forecast, Entry points)
.venv/bin/python cli.py

# Trigger online self-tuning against latest live market price + simulate a $600M whale block purchase
.venv/bin/python cli.py --fine-tune --whale-buy-musd 600 --futures-pct 0.8
```

### 3. Launch the Interactive Quant Web Dashboard
```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```
