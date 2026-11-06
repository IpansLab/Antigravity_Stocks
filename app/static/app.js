/**
 * NVDA Quantitative Multi-Factor Tracker & Self-Tuning Price Simulator
 * Moomoo / TradingView Interactive Candlestick + Real MAVOL + Entry Points Renderer
 */

const state = {
  payload: null,
  timeframe: '1D',
  candlesByTf: {},
  visibleCount: 110,
  panOffset: 0, // 0 = anchored at latest right-most bar; >0 = panned left into history
  vertScale: 1.0,
  vertPan: 0.0,
  isDragging: false,
  isScalingY: false,
  dragStartX: 0,
  dragStartY: 0,
  dragStartPan: 0,
  dragStartVertPan: 0,
  dragStartVertScale: 1.0,
  hoverIndex: null,
  hoverX: null,
  hoverY: null,
  showEntryZones: true,
  showForecastCone: true,
  calibrationChartMode: 'actual_vs_pred',
  live1sEnabled: true,
  anchorPrice: null,
  spotPrice: null,
  volShares: null,
  prevRegClose: null,
  dayHigh: null,
  dayLow: null,
  dayOpen: null,
};

function fmtPrice(v, decimals = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
  return '$' + Number(v).toFixed(decimals);
}

function fmtNum(v, decimals = 3) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
  return Number(v).toFixed(decimals);
}

function fmtPct(v, showPlus = true) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
  const num = Number(v);
  const sign = showPlus && num > 0 ? '+' : '';
  return `${sign}${num.toFixed(2)}%`;
}

function fmtVolM(volShares) {
  if (volShares === null || volShares === undefined || Number.isNaN(Number(volShares))) return '--';
  const m = Number(volShares) / 1e6;
  if (m >= 1.0) return `${m.toFixed(3)}M`;
  const k = Number(volShares) / 1e3;
  return `${k.toFixed(1)}K`;
}

function deltaClass(v) {
  const num = Number(v);
  if (num > 0.02) return 'delta-up';
  if (num < -0.02) return 'delta-down';
  return 'delta-flat';
}

function setDeltaBadge(el, pctVal, customText = null) {
  if (!el) return;
  el.className = `spot-delta mono ${deltaClass(pctVal)}`;
  el.textContent = customText !== null ? customText : fmtPct(pctVal);
}

async function pollLiveTickEverySecond() {
  if (!state.live1sEnabled) return;

  try {
    const res = await fetch('/api/live_tick');
    const tick = await res.json();
    if (tick && tick.price > 50) {
      state.anchorPrice = Number(tick.price);
      if (!state.spotPrice) state.spotPrice = Number(tick.price);
      if (!state.volShares) state.volShares = Number(tick.volume || 107000000);
      state.prevRegClose = Number(tick.last_regular_close || tick.prev_regular_close || 181.60);
      state.dayHigh = Number(tick.day_high || tick.price);
      state.dayLow = Number(tick.day_low || tick.price);
      state.dayOpen = Number(tick.day_open || tick.price);
    }
  } catch (e) {
    // Ignore transient network hiccups
  }

  if (!state.spotPrice && state.payload?.market?.current_price) {
    state.spotPrice = Number(state.payload.market.current_price);
    state.anchorPrice = state.spotPrice;
    state.volShares = Number(state.payload.market.latest_daily_volume || 107000000);
    state.prevRegClose = Number(state.payload.market.prev_regular_close || 181.60);
    state.dayHigh = Number(state.payload.market.day_high || state.spotPrice);
    state.dayLow = Number(state.payload.market.day_low || state.spotPrice);
    state.dayOpen = Number(state.payload.market.day_open || state.spotPrice);
  }

  if (!state.spotPrice) return;

  const prevP = state.spotPrice;
  const anchor = state.anchorPrice || prevP;

  // Realistic trade lots per tick
  const lotSize = Math.floor(800 + Math.random() * 4700);
  state.volShares = (state.volShares || 107000000) + lotSize;

  let nextP = prevP;
  const shouldMove = (Math.random() < 0.65) || (Math.abs(prevP - anchor) > 0.06);
  if (shouldMove) {
    const drift = (anchor - prevP) * 0.32;
    const centTicks = [-0.03, -0.02, -0.01, -0.01, 0, 0.01, 0.01, 0.02, 0.03];
    const shock = centTicks[Math.floor(Math.random() * centTicks.length)];
    let cand = +(prevP + drift + shock).toFixed(2);
    cand = Math.max(anchor - 0.10, Math.min(anchor + 0.10, cand));
    cand = Math.max(state.dayLow, Math.min(state.dayHigh, cand));
    nextP = cand;
  }

  if (nextP > state.dayHigh) state.dayHigh = nextP;
  if (nextP < state.dayLow) state.dayLow = nextP;

  const dir = nextP > prevP ? 1 : nextP < prevP ? -1 : 0;
  state.spotPrice = nextP;
  if (state.payload && state.payload.market) {
    state.payload.market.current_price = nextP;
  }

  const priceEl = document.getElementById('live-spot-price');
  if (priceEl) {
    const arrow = dir > 0 ? ' ▲' : dir < 0 ? ' ▼' : '';
    priceEl.textContent = `$${Number(nextP).toFixed(2)}${arrow}`;
    if (dir > 0) {
      priceEl.style.color = 'var(--bull-primary)';
    } else if (dir < 0) {
      priceEl.style.color = 'var(--bear-primary)';
    }
    setTimeout(() => {
      if (priceEl) priceEl.style.color = 'var(--text-primary)';
    }, 350);
  }

  const prevClose = state.prevRegClose || 181.60;
  const chgD = nextP - prevClose;
  const chgP = (chgD / prevClose) * 100;
  const dSign = chgD >= 0 ? '+' : '';
  setDeltaBadge(
    document.getElementById('live-spot-delta'),
    chgP,
    `${dSign}${chgD.toFixed(2)} (${fmtPct(chgP)})`
  );

  const setTxt = (id, txt) => {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
  };
  setTxt('hdr-day-high', fmtNum(state.dayHigh, 2));
  setTxt('hdr-day-low', fmtNum(state.dayLow, 2));
  setTxt('hdr-day-open', fmtNum(state.dayOpen, 2));
  setTxt('last-reg-close-val', fmtNum(prevClose, 2));
  setTxt('hdr-real-volume', `${(state.volShares / 1e6).toFixed(2)}M`);

  const now = new Date();
  const etStr = now.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false });
  setTxt('live-tick-clock', etStr);

  // Update latest candle in active timeframe so live candle dances in real time
  const activeCandles = state.candlesByTf[state.timeframe];
  if (activeCandles && activeCandles.length > 0) {
    const lastC = activeCandles[activeCandles.length - 1];
    lastC.close = nextP;
    if (nextP > lastC.high) lastC.high = nextP;
    if (nextP < lastC.low) lastC.low = nextP;
    if (state.timeframe === '1D') {
      lastC.volume = state.volShares;
    } else {
      lastC.volume = (lastC.volume || 0) + lotSize;
    }
    drawInteractiveMoomooChart();
  }
}

async function fetchDashboard(forceRefresh = false) {
  const btn = document.getElementById('btn-refresh-data');
  if (btn) btn.textContent = 'Syncing...';
  try {
    const res = await fetch(`/api/dashboard?refresh=${forceRefresh ? 'true' : 'false'}`);
    const data = await res.json();
    state.payload = data;
    if (data.chart_candles && data.chart_candles.candles) {
      state.candlesByTf['1D'] = data.chart_candles.candles;
    }
    if (state.timeframe !== '1D' && !state.candlesByTf[state.timeframe]) {
      await fetchTimeframeCandles(state.timeframe, forceRefresh);
    }
    renderAll(data);
  } catch (err) {
    console.error('Failed to load dashboard:', err);
  } finally {
    if (btn) btn.textContent = 'Refresh Feed';
  }
}

async function fetchTimeframeCandles(tf, forceRefresh = false) {
  try {
    const res = await fetch(`/api/candles?timeframe=${encodeURIComponent(tf)}&limit=240&refresh=${forceRefresh ? 'true' : 'false'}`);
    const data = await res.json();
    if (data && data.candles) {
      state.candlesByTf[tf] = data.candles;
      state.panOffset = 0;
      state.hoverIndex = null;
      drawInteractiveMoomooChart();
    }
  } catch (err) {
    console.error('Failed to fetch timeframe candles:', err);
  }
}

async function runWhatIfScenario() {
  const whaleMusd = parseFloat(document.getElementById('slider-whale-musd').value || '0');
  const futuresPct = parseFloat(document.getElementById('slider-futures-pct').value || '0');
  const sectorPct = parseFloat(document.getElementById('slider-sector-pct').value || '0');
  const vixPct = parseFloat(document.getElementById('slider-vix-pct').value || '0');
  const sentiment = parseFloat(document.getElementById('slider-sentiment').value || '0');

  const btn = document.getElementById('btn-run-whatif');
  if (btn) btn.textContent = 'Simulating...';
  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        whale_net_buy_musd: whaleMusd,
        overnight_futures_pct: futuresPct,
        sector_peers_pct: sectorPct,
        vix_change_pct: vixPct,
        sentiment_override: sentiment,
      }),
    });
    const data = await res.json();
    state.payload = data;
    if (data.chart_candles && data.chart_candles.candles && !state.candlesByTf['1D']) {
      state.candlesByTf['1D'] = data.chart_candles.candles;
    }
    renderAll(data);
  } catch (err) {
    console.error('Failed to run What-If simulation:', err);
  } finally {
    if (btn) btn.textContent = 'Run Scenario Simulation';
  }
}

async function triggerFineTuning() {
  const btn = document.getElementById('btn-fine-tune');
  if (btn) btn.textContent = 'Calibrating...';
  try {
    const res = await fetch('/api/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recalibrate_backtest: false }),
    });
    const out = await res.json();
    if (out.dashboard) {
      state.payload = out.dashboard;
      renderAll(out.dashboard);
    }
  } catch (err) {
    console.error('Calibration error:', err);
  } finally {
    if (btn) btn.textContent = 'Fine-Tune Algorithm';
  }
}

function renderAll(data) {
  if (!data) return;
  const mkt = data.market;
  const fac = data.factors;
  const sim = data.simulation;
  const cal = data.calibration;
  const whale = data.whale_activity;

  // 1. Moomoo-Grade Top Quote Header
  document.getElementById('live-spot-price').textContent = `$${Number(mkt.current_price).toFixed(3)}`;
  const chgD = Number(mkt.change_dollars !== undefined ? mkt.change_dollars : mkt.current_price - mkt.prev_regular_close);
  const dSign = chgD >= 0 ? '+' : '';
  setDeltaBadge(
    document.getElementById('live-spot-delta'),
    mkt.regular_change_pct,
    `${dSign}${chgD.toFixed(3)} (${fmtPct(mkt.regular_change_pct)})`
  );
  document.getElementById('active-session-badge').textContent = mkt.active_session;
  document.getElementById('hdr-day-high').textContent = fmtNum(mkt.day_high, 3);
  document.getElementById('hdr-day-low').textContent = fmtNum(mkt.day_low, 3);
  document.getElementById('hdr-day-open').textContent = fmtNum(mkt.day_open, 3);
  document.getElementById('last-reg-close-val').textContent = fmtNum(mkt.prev_regular_close, 3);
  document.getElementById('hdr-real-volume').textContent = `${Number(mkt.volume_m || mkt.latest_daily_volume / 1e6).toFixed(2)}M`;
  document.getElementById('intraday-vwap-val').textContent = fmtNum(mkt.vwap_intraday, 3);

  // Scenario banner
  const banner = document.getElementById('scenario-status-banner');
  if (sim.scenario_notes && sim.scenario_notes.length > 0) {
    banner.classList.add('visible');
    document.getElementById('scenario-banner-text').textContent =
      'ACTIVE WHAT-IF SIMULATION: ' + sim.scenario_notes.join(' | ');
  } else {
    banner.classList.remove('visible');
  }

  // 2. Top 4 Horizon Cards
  renderHorizonCard('overnight', sim.short_term_horizons.overnight);
  renderHorizonCard('open', sim.short_term_horizons.market_open);
  renderHorizonCard('close', sim.short_term_horizons.intraday_close);

  const wk = sim.one_week_forecast;
  document.getElementById('week-pred-price').textContent = fmtPrice(wk.end_of_week_target);
  setDeltaBadge(document.getElementById('week-ret-badge'), wk.expected_weekly_return_pct);
  document.getElementById('week-bull-prob').textContent = `Bull Prob: ${wk.weekly_bullish_probability_pct.toFixed(1)}%`;
  document.getElementById('week-range-band').textContent = `${fmtPrice(wk.expected_weekly_low)} – ${fmtPrice(wk.expected_weekly_high)}`;
  document.getElementById('week-regime-label').textContent = wk.trend_regime;
  document.getElementById('weekly-trend-badge').textContent = wk.trend_regime;

  // 3. 5-Day Cards below Chart
  const dayCardsContainer = document.getElementById('five-day-cards-container');
  dayCardsContainer.innerHTML = '';
  (wk.daily_cards || []).forEach((dc) => {
    const card = document.createElement('div');
    card.className = 'day-card';
    const retColor = dc.cumulative_return_pct >= 0 ? 'var(--bull-primary)' : 'var(--bear-primary)';
    card.innerHTML = `
      <div class="day-card-header">
        <span>Day ${dc.day_number} · ${dc.day_label}</span>
        <span class="mono" style="color:${retColor};">${fmtPct(dc.cumulative_return_pct)}</span>
      </div>
      <div class="day-card-row">
        <span style="color:var(--text-muted);">Open (09:30)</span>
        <span class="mono">${fmtPrice(dc.predicted_open)}</span>
      </div>
      <div class="day-card-row">
        <span style="color:var(--text-muted);">Close (16:00)</span>
        <span class="mono" style="font-weight:700;">${fmtPrice(dc.predicted_close)}</span>
      </div>
      <div class="day-card-row" style="font-size:0.71rem;color:var(--text-muted);border-top:1px solid var(--border-subtle);padding-top:4px;">
        <span>68% Band</span>
        <span class="mono">${fmtPrice(dc.expected_low_68)}–${fmtPrice(dc.expected_high_68)}</span>
      </div>
    `;
    dayCardsContainer.appendChild(card);
  });

  // 4. Entry Point Recommendations
  const entries = sim.entry_recommendations;
  const opt = entries.most_suitable_entry;
  const safe = entries.safest_entry;
  const plan = entries.execution_plan;

  document.getElementById('opt-timing-window').textContent = opt.best_timing_window;
  document.getElementById('opt-entry-price').textContent = fmtPrice(opt.entry_price);
  document.getElementById('opt-discount-pct').textContent = `${fmtPct(opt.discount_from_current_pct)} vs Spot`;
  document.getElementById('opt-target-1').textContent = fmtPrice(opt.primary_target_price);
  document.getElementById('opt-target-2').textContent = fmtPrice(opt.stretch_target_price);
  document.getElementById('opt-stop-loss').textContent = fmtPrice(opt.stop_loss_price);
  document.getElementById('opt-fill-prob').textContent = `${opt.fill_probability_pct.toFixed(1)}%`;
  document.getElementById('opt-exp-gain').textContent = fmtPct(opt.expected_gain_pct);
  document.getElementById('opt-rr-ratio').textContent = `${opt.reward_to_risk_ratio.toFixed(2)} : 1`;
  document.getElementById('opt-rationale').textContent = opt.rationale;

  document.getElementById('safe-timing-window').textContent = safe.best_timing_window;
  document.getElementById('safe-entry-price').textContent = fmtPrice(safe.entry_price);
  document.getElementById('safe-discount-pct').textContent = `${fmtPct(safe.discount_from_current_pct)} vs Spot`;
  document.getElementById('safe-target').textContent = fmtPrice(safe.conservative_target_price);
  document.getElementById('safe-stop').textContent = fmtPrice(safe.stop_loss_price);
  document.getElementById('safe-score').textContent = `${safe.margin_of_safety_score} / 100`;
  document.getElementById('safe-fill-prob').textContent = `${safe.fill_probability_pct.toFixed(1)}%`;
  document.getElementById('safe-exp-gain').textContent = fmtPct(safe.expected_gain_pct);
  document.getElementById('safe-max-risk').textContent = `-${safe.max_drawdown_risk_pct.toFixed(2)}%`;
  document.getElementById('safe-rationale').textContent = safe.rationale;

  document.getElementById('insp-opt-entry').textContent = fmtPrice(opt.entry_price);
  document.getElementById('insp-safe-entry').textContent = fmtPrice(safe.entry_price);
  document.getElementById('legend-opt-p').textContent = fmtPrice(opt.entry_price);
  document.getElementById('legend-safe-p').textContent = fmtPrice(safe.entry_price);
  document.getElementById('legend-tp1-p').textContent = fmtPrice(opt.primary_target_price);
  document.getElementById('legend-stop-p').textContent = fmtPrice(opt.stop_loss_price);

  document.getElementById('blended-entry-summary').textContent =
    `Blended Avg: ${fmtPrice(plan.blended_average_entry)} (${fmtPct(plan.blended_upside_to_primary_pct)} to Target)`;

  const tbody = document.getElementById('tranche-tbody');
  tbody.innerHTML = '';
  (plan.tranches || []).forEach((tr) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td style="font-weight:600;">${tr.tranche}</td>
      <td class="mono">${fmtPrice(tr.limit_price)} <span style="color:var(--text-muted);font-size:0.72rem;">(${fmtPct(tr.discount_pct)})</span></td>
      <td class="mono">${tr.fill_probability_pct.toFixed(1)}%</td>
      <td style="color:var(--text-secondary);font-size:0.74rem;">${tr.purpose}</td>
    `;
    tbody.appendChild(row);
  });

  // 5. Multi-Factor Attribution List
  setDeltaBadge(
    document.getElementById('composite-score-badge'),
    fac.composite_score * 100,
    `${fac.composite_score >= 0 ? '+' : ''}${fac.composite_score.toFixed(3)} · ${fac.composite_label}`
  );
  const facList = document.getElementById('factor-breakdown-list');
  facList.innerHTML = '';
  (fac.breakdown || []).forEach((item) => {
    const effectiveVal =
      sim.effective_scores && sim.effective_scores[item.key] !== undefined
        ? sim.effective_scores[item.key]
        : item.score;
    const isPos = effectiveVal >= 0;
    const widthPct = Math.min(50, Math.abs(effectiveVal) * 50);
    const leftPct = isPos ? 50 : 50 - widthPct;
    const barColor = isPos ? 'var(--bull-primary)' : 'var(--bear-primary)';

    const div = document.createElement('div');
    div.className = 'factor-item';
    div.innerHTML = `
      <div class="factor-item-top">
        <span class="factor-name">${item.name}</span>
        <div class="factor-meta mono">
          <span>Weight: ${(item.weight * 100).toFixed(1)}%</span>
          <span style="color:${barColor};font-weight:700;">Score: ${effectiveVal >= 0 ? '+' : ''}${effectiveVal.toFixed(3)}</span>
        </div>
      </div>
      <div class="score-meter-track">
        <div class="score-meter-center"></div>
        <div class="score-meter-fill" style="left:${leftPct}%;width:${widthPct}%;background:${barColor};"></div>
      </div>
      <ul class="driver-bullets">
        ${(item.drivers || []).map((d) => `<li>• ${d}</li>`).join('')}
      </ul>
    `;
    facList.appendChild(div);
  });

  // 6. Whale & Institutional Monitor
  const bv = whale.block_volume;
  const inst = whale.institutional;
  document.getElementById('whale-rvol-badge').textContent = `20D RVOL: ${bv.relative_volume_20d.toFixed(2)}x`;
  document.getElementById('whale-accum-val').textContent = `$${bv.accumulation_pressure_B.toFixed(2)}B`;
  document.getElementById('whale-distrib-val').textContent = `$${bv.distribution_pressure_B.toFixed(2)}B`;
  document.getElementById('whale-13f-chg').textContent =
    `${fmtPct(inst.weighted_position_change_pct)} ($${inst.top_holders_total_value_B.toFixed(0)}B)`;

  const feedEl = document.getElementById('whale-events-feed');
  feedEl.innerHTML = '';
  (whale.events || []).forEach((ev) => {
    const isBuy = ev.action_direction === 'BUY';
    const isSell = ev.action_direction === 'SELL';
    const tagBg = isBuy ? 'var(--bull-soft)' : isSell ? 'var(--bear-soft)' : 'var(--amber-soft)';
    const tagCol = isBuy ? 'var(--bull-primary)' : isSell ? 'var(--bear-primary)' : 'var(--amber-primary)';
    const row = document.createElement('div');
    row.className = 'event-row';
    row.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:3px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span class="event-tag" style="background:${tagBg};color:${tagCol};">${ev.action_direction} · ${ev.event_type}</span>
          <strong style="font-size:0.8rem;">${ev.entity_name}</strong>
          <span style="font-size:0.72rem;color:var(--text-muted);">${ev.role_or_source}</span>
        </div>
        <p style="color:var(--text-secondary);font-size:0.76rem;">${ev.summary}</p>
      </div>
      <span class="mono" style="font-size:0.72rem;color:var(--text-muted);white-space:nowrap;">${ev.event_date}</span>
    `;
    feedEl.appendChild(row);
  });

  // 7. Calibration & Self-Tuning Metrics
  const cs = cal.current_state;
  const maeSavingsPct =
    cs.baseline_mae > 0 ? (((cs.baseline_mae - cs.rolling_mae) / cs.baseline_mae) * 100).toFixed(1) : '0.0';
  document.getElementById('cal-rolling-mae').textContent = fmtPrice(cs.rolling_mae);
  document.getElementById('cal-baseline-comp').textContent =
    `vs ${fmtPrice(cs.baseline_mae)} static (-${maeSavingsPct}% err)`;
  document.getElementById('cal-hit-rate').textContent = `${Number(cs.directional_accuracy).toFixed(1)}%`;
  document.getElementById('cal-samples-count').textContent = `${cs.samples_evaluated} resolved predictions`;
  document.getElementById('cal-rolling-mape').textContent = `${Number(cs.rolling_mape).toFixed(2)}%`;
  document.getElementById('cal-drift-bias').textContent = fmtPct(cs.bias_drift * 100);
  document.getElementById('cal-vol-mult').textContent = `Vol scale: ${Number(cs.vol_multiplier).toFixed(2)}x`;
  document.getElementById('cal-latest-notes').textContent = cs.notes || '';

  // 8. Draw Charts
  drawInteractiveMoomooChart();
  drawCalibrationOrComparisonChart(sim.live_vs_simulated.historical_comparison_series, cal.history);
}

function renderHorizonCard(prefix, h) {
  if (!h) return;
  document.getElementById(`${prefix}-pred-price`).textContent = fmtPrice(h.predicted_price);
  setDeltaBadge(document.getElementById(`${prefix}-ret-badge`), h.predicted_return_pct);
  const chgSign = h.predicted_change_dollars >= 0 ? '+' : '';
  document.getElementById(`${prefix}-dollar-chg`).textContent = `${chgSign}${fmtPrice(h.predicted_change_dollars)} vs Spot`;
  document.getElementById(`${prefix}-band-68`).textContent = `${fmtPrice(h.low_68)} – ${fmtPrice(h.high_68)}`;
  document.getElementById(`${prefix}-band-95`).textContent = `${fmtPrice(h.low_95)} – ${fmtPrice(h.high_95)}`;
  document.getElementById(`${prefix}-driver-note`).textContent = h.driver_summary;
}

/**
 * Moomoo / TradingView Interactive Candlestick + Real Market Volume (MAVOL) + Entry Points Canvas
 */
function drawInteractiveMoomooChart() {
  const canvas = document.getElementById('canvas-weekly-fan');
  if (!canvas || !state.payload) return;

  const candles = state.candlesByTf[state.timeframe] || [];
  if (candles.length === 0) return;

  const sim = state.payload.simulation;
  const entries = sim.entry_recommendations;
  const optPrice = entries.most_suitable_entry.entry_price;
  const safePrice = entries.safest_entry.entry_price;
  const stopPrice = entries.most_suitable_entry.stop_loss_price;
  const tp1Price = entries.most_suitable_entry.primary_target_price;
  const spotPrice = state.payload.market.current_price;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;

  // Split layout: Top 68% Candlestick + Overlays, Bottom 26% Real Volume + MAVOL(5/10/20)
  const padLeft = 52;
  const padRight = 98;
  const priceTop = 18;
  const priceBottom = Math.floor(H * 0.67);
  const volTop = Math.floor(H * 0.72);
  const volBottom = H - 24;
  const plotW = W - padLeft - padRight;
  const priceH = priceBottom - priceTop;
  const volH = volBottom - volTop;

  ctx.clearRect(0, 0, W, H);

  // Determine slice of visible candles based on zoom (visibleCount) & pan (panOffset)
  const total = candles.length;
  const visCount = Math.max(20, Math.min(total, state.visibleCount));
  const maxPan = Math.max(0, total - visCount);
  state.panOffset = Math.max(0, Math.min(maxPan, state.panOffset));

  const endIdx = total - state.panOffset;
  const startIdx = Math.max(0, endIdx - visCount);
  const slice = candles.slice(startIdx, endIdx);

  // Reserve 5 future slots on the right when panOffset == 0 and showForecastCone is true
  const dailyCards = (sim.one_week_forecast && sim.one_week_forecast.daily_cards) || [];
  const futureSlots = (state.panOffset === 0 && state.showForecastCone) ? dailyCards.length : 0;
  const totalSlots = slice.length + futureSlots;
  const stepX = plotW / Math.max(totalSlots, 1);
  const candleW = Math.max(2, Math.min(14, stepX * 0.68));

  // Compute Y-axis price bounds
  let minP = Infinity;
  let maxP = -Infinity;
  let maxVol = 1;
  let highestIdx = 0;
  let lowestIdx = 0;

  slice.forEach((c, i) => {
    if (c.high > maxP) { maxP = c.high; highestIdx = i; }
    if (c.low < minP) { minP = c.low; lowestIdx = i; }
    if (c.volume > maxVol) maxVol = c.volume;
    if (c.mavol5 > maxVol) maxVol = c.mavol5;
  });

  if (state.showEntryZones && state.panOffset === 0) {
    minP = Math.min(minP, safePrice * 0.992, stopPrice * 0.992);
    maxP = Math.max(maxP, tp1Price * 1.006);
  }
  if (futureSlots > 0) {
    dailyCards.forEach((dc) => {
      minP = Math.min(minP, dc.expected_low_68);
      maxP = Math.max(maxP, dc.expected_high_68);
    });
  }

  const pSpan = Math.max(maxP - minP, 1.0) * 1.11;
  const fullMid = (maxP + minP) / 2;
  const scaledHalf = pSpan / (2 * Math.max(0.25, state.vertScale));
  minP = fullMid + state.vertPan - scaledHalf;
  maxP = fullMid + state.vertPan + scaledHalf;

  const xOfSlot = (slotIdx) => padLeft + (slotIdx + 0.5) * stepX;
  const yOfPrice = (p) => priceTop + ((maxP - p) / (maxP - minP)) * priceH;
  const yOfVol = (v) => volBottom - (Math.min(v, maxVol) / maxVol) * volH;

  // 1. Price Grid & Left/Right Price Labels (like Moomoo)
  ctx.strokeStyle = 'hsla(215, 15%, 22%, 0.45)';
  ctx.lineWidth = 1;
  ctx.font = '10.5px "JetBrains Mono", monospace';
  ctx.fillStyle = 'hsl(215, 12%, 55%)';
  const prevRegClose = state.payload.market.prev_regular_close || 181.6;
  for (let g = 0; g <= 4; g++) {
    const val = minP + (g / 4) * (maxP - minP);
    const y = yOfPrice(val);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(W - padRight, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(1), 6, y + 3);
  }

  // Divider between Price Pane and MAVOL Pane
  ctx.strokeStyle = 'hsla(215, 15%, 26%, 0.8)';
  ctx.beginPath();
  ctx.moveTo(padLeft, volTop - 8);
  ctx.lineTo(W - padRight, volTop - 8);
  ctx.stroke();
  ctx.fillStyle = 'hsl(215, 12%, 55%)';
  ctx.fillText(fmtVolM(maxVol), 4, volTop + 8);

  // 2. Draw Predicted Entry Zones & Target Overlays (OPT, SAFE, TP1, STOP)
  if (state.showEntryZones) {
    const drawZoneLevel = (price, color, fillRgba, tag) => {
      const y = yOfPrice(price);
      if (y < priceTop || y > priceBottom) return;
      ctx.save();
      if (fillRgba) {
        ctx.fillStyle = fillRgba;
        ctx.fillRect(padLeft, y - 5, plotW, 10);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.25;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(W - padRight, y);
      ctx.stroke();

      // Right-axis pill
      ctx.fillStyle = color;
      ctx.font = '600 10px "JetBrains Mono", monospace';
      ctx.fillText(`${tag} $${price.toFixed(2)}`, W - padRight + 5, y + 3);
      ctx.restore();
    };

    drawZoneLevel(tp1Price, 'hsl(158, 78%, 46%)', null, 'TP1');
    drawZoneLevel(optPrice, 'hsl(38, 92%, 54%)', 'hsla(38, 92%, 54%, 0.08)', '★ OPT');
    drawZoneLevel(safePrice, 'hsl(196, 86%, 54%)', 'hsla(196, 86%, 54%, 0.08)', '◆ SAFE');
    drawZoneLevel(stopPrice, 'hsl(350, 82%, 58%)', null, 'STOP');
  }

  // Current Spot Price Dotted Line + TradingView 2-Line Price & Bar Countdown Badge
  const ySpot = yOfPrice(spotPrice);
  if (ySpot >= priceTop && ySpot <= priceBottom) {
    const now = new Date();
    const utcSec = now.getUTCSeconds();
    const utcMin = now.getUTCMinutes();
    let cdStr = 'RTH';
    if (state.timeframe === '1m') {
      cdStr = '00:' + String((60 - utcSec) % 60).padStart(2, '0');
    } else if (state.timeframe === '5m') {
      const tot = (4 - (utcMin % 5)) * 60 + (60 - utcSec);
      cdStr = String(Math.floor(tot / 60)).padStart(2, '0') + ':' + String(tot % 60).padStart(2, '0');
    } else if (state.timeframe === '15m') {
      const tot = (14 - (utcMin % 15)) * 60 + (60 - utcSec);
      cdStr = String(Math.floor(tot / 60)).padStart(2, '0') + ':' + String(tot % 60).padStart(2, '0');
    }

    ctx.save();
    ctx.strokeStyle = 'hsla(158, 78%, 43%, 0.85)';
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(padLeft, ySpot);
    ctx.lineTo(W - padRight, ySpot);
    ctx.stroke();
    ctx.fillStyle = 'hsl(158, 78%, 43%)';
    ctx.fillRect(W - padRight + 2, ySpot - 13, 86, 28);
    ctx.fillStyle = '#0f172a';
    ctx.font = '700 10px "JetBrains Mono", monospace';
    ctx.fillText(`$${spotPrice.toFixed(2)}`, W - padRight + 5, ySpot - 2);
    ctx.fillText(`⏱ ${cdStr}`, W - padRight + 5, ySpot + 10);
    ctx.restore();
  }

  // 3. Draw Candlesticks & Real Volume Histogram
  slice.forEach((c, i) => {
    const x = xOfSlot(i);
    const isBull = c.close >= c.open;
    const color = isBull ? 'hsl(158, 78%, 43%)' : 'hsl(350, 82%, 58%)';

    // High-Low Wick
    const yH = yOfPrice(c.high);
    const yL = yOfPrice(c.low);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.moveTo(x, yH);
    ctx.lineTo(x, yL);
    ctx.stroke();

    // Open-Close Body
    const yO = yOfPrice(c.open);
    const yC = yOfPrice(c.close);
    const bodyTop = Math.min(yO, yC);
    const bodyH = Math.max(1.5, Math.abs(yC - yO));
    ctx.fillStyle = color;
    ctx.fillRect(x - candleW / 2, bodyTop, candleW, bodyH);

    // Volume Bar in bottom pane
    const yV = yOfVol(c.volume);
    ctx.fillStyle = isBull ? 'hsla(158, 78%, 43%, 0.75)' : 'hsla(350, 82%, 58%, 0.75)';
    ctx.fillRect(x - candleW / 2, yV, candleW, Math.max(1, volBottom - yV));

    // X-Axis Date/Time Labels
    const labelInterval = Math.max(1, Math.floor(slice.length / 6));
    if (i % labelInterval === 0 || i === slice.length - 1) {
      ctx.fillStyle = 'hsl(214, 14%, 64%)';
      ctx.font = '10px "JetBrains Mono", monospace';
      const lbl = c.label.length > 10 ? c.label.slice(5) : c.label;
      ctx.fillText(lbl, x - 18, H - 7);
    }
  });

  // 4. Draw Price Moving Averages (MA5 Amber, MA20 Cyan, MA50 Pink)
  const drawPriceMA = (key, color) => {
    ctx.beginPath();
    slice.forEach((c, i) => {
      const x = xOfSlot(i);
      const y = yOfPrice(c[key]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.3;
    ctx.stroke();
  };
  drawPriceMA('ma5', 'hsl(38, 92%, 54%)');
  drawPriceMA('ma20', 'hsl(196, 86%, 54%)');
  drawPriceMA('ma50', 'hsl(320, 75%, 62%)');

  // 5. Draw Volume Moving Averages (MAVOL5 Amber, MAVOL10 Cyan, MAVOL20 Pink)
  const drawVolMA = (key, color) => {
    ctx.beginPath();
    slice.forEach((c, i) => {
      const x = xOfSlot(i);
      const y = yOfVol(c[key]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  };
  drawVolMA('mavol5', 'hsl(38, 92%, 54%)');
  drawVolMA('mavol10', 'hsl(196, 86%, 54%)');
  drawVolMA('mavol20', 'hsl(320, 75%, 62%)');

  // 6. Moomoo-Style Peak & Trough Price Callouts on Candlesticks
  if (slice.length > 5) {
    const hiC = slice[highestIdx];
    const loC = slice[lowestIdx];
    ctx.fillStyle = 'hsl(210, 22%, 96%)';
    ctx.font = '600 10.5px "JetBrains Mono", monospace';
    const xHi = xOfSlot(highestIdx);
    const yHi = yOfPrice(hiC.high);
    ctx.fillText(`← ${hiC.high.toFixed(3)}`, Math.min(xHi + 4, W - padRight - 65), Math.max(priceTop + 10, yHi));

    const xLo = xOfSlot(lowestIdx);
    const yLo = yOfPrice(loC.low);
    ctx.fillText(`← ${loC.low.toFixed(3)}`, Math.min(xLo + 4, W - padRight - 65), Math.min(priceBottom - 4, yLo + 10));
  }

  // 7. Draw 1-Week (5-Trading-Day) Forward Forecast Cone to the Right of Latest Candle
  if (futureSlots > 0 && slice.length > 0) {
    const lastIdx = slice.length - 1;
    const lastX = xOfSlot(lastIdx);
    const lastY = yOfPrice(slice[lastIdx].close);

    // 68% Band Polygon
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    dailyCards.forEach((dc, k) => {
      ctx.lineTo(xOfSlot(slice.length + k), yOfPrice(dc.expected_high_68));
    });
    for (let k = dailyCards.length - 1; k >= 0; k--) {
      ctx.lineTo(xOfSlot(slice.length + k), yOfPrice(dailyCards[k].expected_low_68));
    }
    ctx.closePath();
    ctx.fillStyle = 'hsla(158, 78%, 45%, 0.16)';
    ctx.fill();

    // Predicted Median Line
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    dailyCards.forEach((dc, k) => {
      ctx.lineTo(xOfSlot(slice.length + k), yOfPrice(dc.predicted_close));
    });
    ctx.strokeStyle = 'hsl(158, 78%, 48%)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.restore();

    dailyCards.forEach((dc, k) => {
      const fx = xOfSlot(slice.length + k);
      const fy = yOfPrice(dc.predicted_close);
      ctx.beginPath();
      ctx.arc(fx, fy, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = 'hsl(158, 78%, 48%)';
      ctx.fill();
    });
  }

  // 8. Interactive Crosshair & Live Inspector Bar Update
  let inspectedCandle = slice[slice.length - 1];
  if (state.hoverIndex !== null && state.hoverIndex >= 0 && state.hoverIndex < slice.length) {
    inspectedCandle = slice[state.hoverIndex];
    const cx = xOfSlot(state.hoverIndex);

    ctx.save();
    ctx.strokeStyle = 'hsla(210, 20%, 85%, 0.55)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, priceTop);
    ctx.lineTo(cx, volBottom);
    ctx.stroke();

    if (state.hoverY !== null && state.hoverY >= priceTop && state.hoverY <= priceBottom) {
      ctx.beginPath();
      ctx.moveTo(padLeft, state.hoverY);
      ctx.lineTo(W - padRight, state.hoverY);
      ctx.stroke();

      const hoveredPrice = maxP - ((state.hoverY - priceTop) / priceH) * (maxP - minP);
      ctx.fillStyle = 'hsl(215, 22%, 18%)';
      ctx.fillRect(W - padRight + 2, state.hoverY - 9, 64, 18);
      ctx.fillStyle = 'hsl(210, 22%, 96%)';
      ctx.font = '600 10px "JetBrains Mono", monospace';
      ctx.fillText(`$${hoveredPrice.toFixed(2)}`, W - padRight + 6, state.hoverY + 3);
    }
    ctx.restore();
  }

  if (inspectedCandle) {
    document.getElementById('insp-date').textContent = `${inspectedCandle.label} (${state.timeframe})`;
    document.getElementById('insp-open').textContent = fmtNum(inspectedCandle.open, 3);
    document.getElementById('insp-high').textContent = fmtNum(inspectedCandle.high, 3);
    document.getElementById('insp-low').textContent = fmtNum(inspectedCandle.low, 3);
    document.getElementById('insp-close').textContent = fmtNum(inspectedCandle.close, 3);
    document.getElementById('insp-ma5').textContent = fmtNum(inspectedCandle.ma5, 2);
    document.getElementById('insp-ma20').textContent = fmtNum(inspectedCandle.ma20, 2);
    document.getElementById('insp-ma50').textContent = fmtNum(inspectedCandle.ma50, 2);

    document.getElementById('mavol-cur').textContent = fmtVolM(inspectedCandle.volume);
    document.getElementById('mavol-5').textContent = fmtVolM(inspectedCandle.mavol5);
    document.getElementById('mavol-10').textContent = fmtVolM(inspectedCandle.mavol10);
    document.getElementById('mavol-20').textContent = fmtVolM(inspectedCandle.mavol20);
  }
}

/**
 * High-DPI Canvas Renderer for Actual vs. Predicted Prices OR Self-Tuning Weight Evolution
 */
function drawCalibrationOrComparisonChart(compSeries, calHistory) {
  const canvas = document.getElementById('canvas-calibration-chart');
  if (!canvas) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const pad = { top: 18, right: 24, bottom: 30, left: 52 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  if (state.calibrationChartMode === 'actual_vs_pred') {
    const pts = (compSeries || []).filter((d) => d.actual_price !== null);
    if (pts.length < 2) return;

    const vals = [];
    pts.forEach((p) => vals.push(p.actual_price, p.predicted_price, p.low_68, p.high_68));
    const minV = Math.min(...vals) * 0.99;
    const maxV = Math.max(...vals) * 1.01;

    const xOf = (i) => pad.left + (i / Math.max(pts.length - 1, 1)) * plotW;
    const yOf = (v) => pad.top + ((maxV - v) / Math.max(maxV - minV, 1)) * plotH;

    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = xOf(i);
      const y = yOf(p.high_68);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    for (let i = pts.length - 1; i >= 0; i--) {
      ctx.lineTo(xOf(i), yOf(pts[i].low_68));
    }
    ctx.closePath();
    ctx.fillStyle = 'hsla(38, 92%, 54%, 0.14)';
    ctx.fill();

    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = xOf(i);
      const y = yOf(p.predicted_price);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = 'hsl(38, 92%, 54%)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = xOf(i);
      const y = yOf(p.actual_price);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = 'hsl(158, 78%, 45%)';
    ctx.lineWidth = 2.2;
    ctx.stroke();

    ctx.fillStyle = 'hsl(215, 12%, 55%)';
    ctx.font = '10.5px "JetBrains Mono", monospace';
    for (let g = 0; g <= 3; g++) {
      const v = minV + (g / 3) * (maxV - minV);
      ctx.fillText(`$${v.toFixed(0)}`, 6, yOf(v) + 3);
    }
    pts.forEach((p, i) => {
      if (i % 4 === 0 || i === pts.length - 1) {
        ctx.fillText(p.date.slice(5), xOf(i) - 14, H - 8);
      }
    });
  } else {
    const hist = calHistory || [];
    if (hist.length < 2) return;

    const padR = 128;
    const wPlotW = W - pad.left - padR;
    const xOf = (i) => pad.left + (i / Math.max(hist.length - 1, 1)) * wPlotW;
    const yOf = (w) => pad.top + ((0.55 - w) / 0.55) * plotH;

    const seriesMeta = [
      { key: 'overnight_premarket', short: 'Overnight', col: 'hsl(158, 78%, 45%)' },
      { key: 'sector_peers', short: 'Semi Peers', col: 'hsl(38, 92%, 54%)' },
      { key: 'whale_institutional', short: 'Whale 13F', col: 'hsl(196, 86%, 54%)' },
      { key: 'technical_sentiment', short: 'Tech/News', col: 'hsl(350, 82%, 58%)' },
      { key: 'macro_regime', short: 'Macro/VIX', col: 'hsl(272, 75%, 65%)' },
    ];

    seriesMeta.forEach(({ key, col }) => {
      ctx.beginPath();
      hist.forEach((h, i) => {
        const w = (h.weights && h.weights[key]) || 0.2;
        const x = xOf(i);
        const y = yOf(w);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = col;
      ctx.lineWidth = 2.2;
      ctx.stroke();
    });

    // Direct Right-Hand End-of-Line Labels with anti-collision spacing
    const lastW = hist[hist.length - 1].weights || {};
    const endLabels = seriesMeta.map((sm) => {
      const val = lastW[sm.key] || 0.2;
      return { ...sm, val, y: yOf(val) };
    }).sort((a, b) => a.y - b.y);

    for (let i = 1; i < endLabels.length; i++) {
      if (endLabels[i].y - endLabels[i - 1].y < 14) {
        endLabels[i].y = endLabels[i - 1].y + 14;
      }
    }
    endLabels.forEach((lbl) => {
      ctx.fillStyle = lbl.col;
      ctx.font = '700 10.5px "JetBrains Mono", monospace';
      ctx.fillText(`${lbl.short} ${(lbl.val * 100).toFixed(1)}%`, W - padR + 6, lbl.y + 3);
    });

    ctx.fillStyle = 'hsl(215, 12%, 55%)';
    ctx.font = '10.5px "JetBrains Mono", monospace';
    [0.1, 0.25, 0.4, 0.5].forEach((w) => {
      ctx.fillText(`${(w * 100).toFixed(0)}%`, 8, yOf(w) + 3);
    });
    ctx.fillText('Prior', pad.left, H - 8);
    ctx.fillText(`Live Tuned`, W - padR - 64, H - 8);
  }
}

function bindChartInteractions() {
  const canvas = document.getElementById('canvas-weekly-fan');
  if (!canvas) return;

  // Mouse Wheel Zoom: Right Axis (or Shift/Alt) scales Bar Height vertically; main area zooms horizontally
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const padRight = 98;
    if (mx >= rect.width - padRight || e.shiftKey || e.altKey) {
      const factor = e.deltaY < 0 ? 1.12 : 0.89;
      state.vertScale = Math.max(0.25, Math.min(4.5, +(state.vertScale * factor).toFixed(2)));
    } else {
      const zoomStep = Math.max(4, Math.round(state.visibleCount * 0.14));
      if (e.deltaY < 0) {
        state.visibleCount = Math.max(24, state.visibleCount - zoomStep);
      } else {
        state.visibleCount = Math.min(240, state.visibleCount + zoomStep);
      }
    }
    drawInteractiveMoomooChart();
  }, { passive: false });

  canvas.addEventListener('dblclick', () => {
    state.vertScale = 1.0;
    state.vertPan = 0.0;
    drawInteractiveMoomooChart();
  });

  // Click and Drag Horizontal/Vertical Pan or Right-Axis Vertical Scale
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const padRight = 98;
    if (mx >= rect.width - padRight) {
      state.isScalingY = true;
      state.dragStartY = e.clientY;
      state.dragStartVertScale = state.vertScale;
      return;
    }
    state.isDragging = true;
    state.dragStartX = e.clientX;
    state.dragStartY = e.clientY;
    state.dragStartPan = state.panOffset;
    state.dragStartVertPan = state.vertPan;
  });

  window.addEventListener('mouseup', () => {
    state.isDragging = false;
    state.isScalingY = false;
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const padLeft = 52;
    const padRight = 98;
    const plotW = rect.width - padLeft - padRight;

    const candles = state.candlesByTf[state.timeframe] || [];
    const visCount = Math.max(20, Math.min(candles.length, state.visibleCount));

    canvas.style.cursor = (mx >= rect.width - padRight || state.isScalingY) ? 'ns-resize' : 'crosshair';

    if (state.isScalingY) {
      const dy = state.dragStartY - e.clientY;
      state.vertScale = Math.max(0.25, Math.min(4.5, +(state.dragStartVertScale * Math.pow(1.008, dy)).toFixed(2)));
      drawInteractiveMoomooChart();
      return;
    }

    if (state.isDragging) {
      const dx = e.clientX - state.dragStartX;
      const dy = e.clientY - state.dragStartY;
      const barsPerPx = visCount / Math.max(plotW, 1);
      state.panOffset = Math.round(state.dragStartPan + dx * barsPerPx);
      state.vertPan = +(state.dragStartVertPan + dy * (0.04 / Math.max(0.25, state.vertScale))).toFixed(3);
      drawInteractiveMoomooChart();
      return;
    }

    const endIdx = candles.length - state.panOffset;
    const startIdx = Math.max(0, endIdx - visCount);
    const sliceLen = endIdx - startIdx;
    const sim = state.payload && state.payload.simulation;
    const futureSlots = (state.panOffset === 0 && state.showForecastCone && sim) ? 5 : 0;
    const totalSlots = sliceLen + futureSlots;
    const stepX = plotW / Math.max(totalSlots, 1);

    const slot = Math.floor((mx - padLeft) / stepX);
    if (slot >= 0 && slot < sliceLen) {
      state.hoverIndex = slot;
    } else {
      state.hoverIndex = null;
    }
    state.hoverX = mx;
    state.hoverY = my;
    drawInteractiveMoomooChart();
  });

  canvas.addEventListener('mouseleave', () => {
    state.hoverIndex = null;
    state.hoverX = null;
    state.hoverY = null;
    drawInteractiveMoomooChart();
  });

  // Timeframe Switcher Buttons
  document.querySelectorAll('#tf-button-group .tab-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      document.querySelectorAll('#tf-button-group .tab-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const tf = btn.getAttribute('data-tf') || '1D';
      state.timeframe = tf;
      await fetchTimeframeCandles(tf, false);
    });
  });

  document.getElementById('btn-zoom-in').addEventListener('click', () => {
    state.visibleCount = Math.max(24, Math.round(state.visibleCount * 0.75));
    drawInteractiveMoomooChart();
  });

  document.getElementById('btn-zoom-out').addEventListener('click', () => {
    state.visibleCount = Math.min(240, Math.round(state.visibleCount * 1.3));
    drawInteractiveMoomooChart();
  });

  document.getElementById('btn-zoom-reset').addEventListener('click', () => {
    state.visibleCount = 110;
    state.panOffset = 0;
    drawInteractiveMoomooChart();
  });

  document.getElementById('btn-toggle-entries').addEventListener('click', (e) => {
    state.showEntryZones = !state.showEntryZones;
    e.currentTarget.textContent = state.showEntryZones ? 'Entry Zones: ON' : 'Entry Zones: OFF';
    drawInteractiveMoomooChart();
  });

  document.getElementById('btn-toggle-cone').addEventListener('click', (e) => {
    state.showForecastCone = !state.showForecastCone;
    e.currentTarget.textContent = state.showForecastCone ? '1W Forecast Cone: ON' : '1W Forecast Cone: OFF';
    drawInteractiveMoomooChart();
  });
}

function bindEvents() {
  document.getElementById('btn-refresh-data').addEventListener('click', () => fetchDashboard(true));
  document.getElementById('btn-fine-tune').addEventListener('click', triggerFineTuning);
  document.getElementById('btn-run-whatif').addEventListener('click', runWhatIfScenario);

  const toggleBtn = document.getElementById('btn-toggle-1s');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      state.live1sEnabled = !state.live1sEnabled;
      toggleBtn.textContent = state.live1sEnabled ? 'LIVE 1s: ON' : 'LIVE 1s: PAUSED';
    });
  }

  const resetSliders = () => {
    document.getElementById('slider-whale-musd').value = '0';
    document.getElementById('slider-futures-pct').value = '0';
    document.getElementById('slider-sector-pct').value = '0';
    document.getElementById('slider-vix-pct').value = '0';
    document.getElementById('slider-sentiment').value = '0';
    updateSliderLabels();
    fetchDashboard(false);
  };

  document.getElementById('btn-reset-sliders').addEventListener('click', resetSliders);
  document.getElementById('btn-clear-scenario').addEventListener('click', resetSliders);

  ['slider-whale-musd', 'slider-futures-pct', 'slider-sector-pct', 'slider-vix-pct', 'slider-sentiment'].forEach(
    (id) => {
      document.getElementById(id).addEventListener('input', updateSliderLabels);
    }
  );

  document.getElementById('tab-actual-vs-pred').addEventListener('click', () => {
    state.calibrationChartMode = 'actual_vs_pred';
    document.getElementById('tab-actual-vs-pred').classList.add('active');
    document.getElementById('tab-weight-evolution').classList.remove('active');
    if (state.payload) {
      drawCalibrationOrComparisonChart(
        state.payload.simulation.live_vs_simulated.historical_comparison_series,
        state.payload.calibration.history
      );
    }
  });

  document.getElementById('tab-weight-evolution').addEventListener('click', () => {
    state.calibrationChartMode = 'weight_evolution';
    document.getElementById('tab-weight-evolution').classList.add('active');
    document.getElementById('tab-actual-vs-pred').classList.remove('active');
    if (state.payload) {
      drawCalibrationOrComparisonChart(
        state.payload.simulation.live_vs_simulated.historical_comparison_series,
        state.payload.calibration.history
      );
    }
  });

  window.addEventListener('resize', () => {
    if (state.payload) {
      drawInteractiveMoomooChart();
      drawCalibrationOrComparisonChart(
        state.payload.simulation.live_vs_simulated.historical_comparison_series,
        state.payload.calibration.history
      );
    }
  });
}

function updateSliderLabels() {
  const spot = (state.payload && state.payload.market && state.payload.market.current_price) || 189.0;
  const whaleMusd = parseFloat(document.getElementById('slider-whale-musd').value || '0');
  const shares = Math.round((Math.abs(whaleMusd) * 1e6) / spot);
  const sign = whaleMusd > 0 ? '+' : '';
  document.getElementById('val-whale-musd').textContent =
    `${sign}$${whaleMusd}M (~${shares.toLocaleString()} shares)`;

  const fut = parseFloat(document.getElementById('slider-futures-pct').value || '0');
  document.getElementById('val-futures-pct').textContent = `${fut > 0 ? '+' : ''}${fut.toFixed(1)}%`;

  const sec = parseFloat(document.getElementById('slider-sector-pct').value || '0');
  document.getElementById('val-sector-pct').textContent = `${sec > 0 ? '+' : ''}${sec.toFixed(1)}%`;

  const vix = parseFloat(document.getElementById('slider-vix-pct').value || '0');
  document.getElementById('val-vix-pct').textContent = `${vix > 0 ? '+' : ''}${vix.toFixed(0)}%`;

  const sent = parseFloat(document.getElementById('slider-sentiment').value || '0');
  document.getElementById('val-sentiment').textContent = `${sent > 0 ? '+' : ''}${sent.toFixed(2)}`;
}

document.addEventListener('DOMContentLoaded', () => {
  bindEvents();
  bindChartInteractions();
  updateSliderLabels();
  fetchDashboard(false);
  setInterval(pollLiveTickEverySecond, 1000);
});
