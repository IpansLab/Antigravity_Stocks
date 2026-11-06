/**
 * NVDA Quantitative Multi-Factor Tracker & Self-Tuning Price Simulator
 * Frontend Controller & High-DPI Financial Canvas Renderer
 */

const state = {
  payload: null,
  calibrationChartMode: 'actual_vs_pred', // 'actual_vs_pred' | 'weight_evolution'
};

function fmtPrice(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
  return '$' + Number(v).toFixed(2);
}

function fmtPct(v, showPlus = true) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
  const num = Number(v);
  const sign = showPlus && num > 0 ? '+' : '';
  return `${sign}${num.toFixed(2)}%`;
}

function deltaClass(v) {
  const num = Number(v);
  if (num > 0.05) return 'delta-up';
  if (num < -0.05) return 'delta-down';
  return 'delta-flat';
}

function setDeltaBadge(el, pctVal, customText = null) {
  if (!el) return;
  el.className = `spot-delta mono ${deltaClass(pctVal)}`;
  el.textContent = customText !== null ? customText : fmtPct(pctVal);
}

async function fetchDashboard(forceRefresh = false) {
  const btn = document.getElementById('btn-refresh-data');
  if (btn) btn.textContent = 'Syncing...';
  try {
    const res = await fetch(`/api/dashboard?refresh=${forceRefresh ? 'true' : 'false'}`);
    const data = await res.json();
    state.payload = data;
    renderAll(data);
  } catch (err) {
    console.error('Failed to load dashboard:', err);
  } finally {
    if (btn) btn.textContent = 'Refresh Live Feed';
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
    renderAll(data);
  } catch (err) {
    console.error('Failed to run What-If simulation:', err);
  } finally {
    if (btn) btn.textContent = 'Run Scenario Simulation';
  }
}

async function triggerFineTuning() {
  const btn = document.getElementById('btn-fine-tune');
  if (btn) btn.textContent = 'Calibrating Weights...';
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
    if (btn) btn.textContent = 'Fine-Tune Algorithm (Learn from Error)';
  }
}

function renderAll(data) {
  if (!data) return;
  const mkt = data.market;
  const fac = data.factors;
  const sim = data.simulation;
  const cal = data.calibration;
  const whale = data.whale_activity;

  // 1. Top Header
  document.getElementById('live-spot-price').textContent = fmtPrice(mkt.current_price);
  setDeltaBadge(document.getElementById('live-spot-delta'), mkt.extended_change_pct);
  document.getElementById('active-session-badge').textContent = mkt.active_session;
  document.getElementById('last-reg-close-val').textContent = fmtPrice(mkt.last_regular_close);
  document.getElementById('intraday-vwap-val').textContent = fmtPrice(mkt.vwap_intraday);

  const comp = sim.live_vs_simulated;
  const simSpreadEl = document.getElementById('live-vs-sim-badge');
  simSpreadEl.textContent = `${fmtPrice(comp.active_simulated_price)} (${fmtPct(comp.implied_upside_pct)})`;
  simSpreadEl.style.color =
    comp.implied_upside_pct > 0.15
      ? 'var(--bull-primary)'
      : comp.implied_upside_pct < -0.15
      ? 'var(--bear-primary)'
      : 'var(--text-primary)';

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

  // 3. 5-Day Cards below Fan Chart
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
  drawWeeklyFanChart(sim.one_week_forecast.timeline, entries);
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
 * High-DPI Canvas Renderer for 1-Week Trajectory Fan Chart + Entry & Stop Levels
 */
function drawWeeklyFanChart(timeline, entries) {
  const canvas = document.getElementById('canvas-weekly-fan');
  if (!canvas || !timeline || timeline.length === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const pad = { top: 22, right: 94, bottom: 34, left: 54 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  const optPrice = entries.most_suitable_entry.entry_price;
  const safePrice = entries.safest_entry.entry_price;
  const stopPrice = entries.most_suitable_entry.stop_loss_price;
  const targetPrice = entries.most_suitable_entry.primary_target_price;

  const allVals = [];
  timeline.forEach((pt) => {
    allVals.push(pt.low_90, pt.high_90, pt.median);
  });
  allVals.push(optPrice, safePrice, stopPrice, targetPrice);

  const minP = Math.min(...allVals) * 0.992;
  const maxP = Math.max(...allVals) * 1.008;

  const xOf = (idx) => pad.left + (idx / Math.max(timeline.length - 1, 1)) * plotW;
  const yOf = (val) => pad.top + ((maxP - val) / Math.max(maxP - minP, 1)) * plotH;

  // Horizontal price grid lines
  const gridSteps = 5;
  ctx.strokeStyle = 'hsla(215, 15%, 22%, 0.55)';
  ctx.lineWidth = 1;
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillStyle = 'hsl(215, 12%, 55%)';
  for (let i = 0; i <= gridSteps; i++) {
    const val = minP + (i / gridSteps) * (maxP - minP);
    const y = yOf(val);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();
    ctx.fillText(`$${val.toFixed(0)}`, 8, y + 4);
  }

  // 90% Outer Confidence Band
  ctx.beginPath();
  timeline.forEach((pt, idx) => {
    const x = xOf(idx);
    const y = yOf(pt.high_90);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  for (let idx = timeline.length - 1; idx >= 0; idx--) {
    ctx.lineTo(xOf(idx), yOf(timeline[idx].low_90));
  }
  ctx.closePath();
  ctx.fillStyle = 'hsla(158, 78%, 42%, 0.08)';
  ctx.fill();

  // 68% Inner Confidence Band
  ctx.beginPath();
  timeline.forEach((pt, idx) => {
    const x = xOf(idx);
    const y = yOf(pt.high_68);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  for (let idx = timeline.length - 1; idx >= 0; idx--) {
    ctx.lineTo(xOf(idx), yOf(timeline[idx].low_68));
  }
  ctx.closePath();
  ctx.fillStyle = 'hsla(158, 78%, 42%, 0.19)';
  ctx.fill();

  // Horizontal Reference Lines: Target, Optimal Entry, Safe Entry, Stop-Loss
  const drawHorizontalLevel = (price, color, label, dashed = true) => {
    const y = yOf(price);
    if (y < pad.top || y > H - pad.bottom) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.3;
    if (dashed) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = '600 10.5px "JetBrains Mono", monospace';
    ctx.fillText(`${label} $${price.toFixed(2)}`, W - pad.right + 6, y + 3);
    ctx.restore();
  };

  drawHorizontalLevel(targetPrice, 'hsl(158, 78%, 48%)', 'TP1');
  drawHorizontalLevel(optPrice, 'hsl(38, 92%, 54%)', 'OPT');
  drawHorizontalLevel(safePrice, 'hsl(196, 86%, 54%)', 'SAFE');
  drawHorizontalLevel(stopPrice, 'hsl(350, 82%, 58%)', 'STOP');

  // Simulated Median Trajectory Line
  ctx.beginPath();
  timeline.forEach((pt, idx) => {
    const x = xOf(idx);
    const y = yOf(pt.median);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = 'hsl(158, 78%, 45%)';
  ctx.lineWidth = 2.4;
  ctx.stroke();

  // Data Nodes & X-Axis Labels
  timeline.forEach((pt, idx) => {
    const x = xOf(idx);
    const y = yOf(pt.median);
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = idx === 0 ? 'hsl(210, 22%, 95%)' : 'hsl(158, 78%, 45%)';
    ctx.fill();

    if (idx === 0 || idx % 2 === 0) {
      ctx.fillStyle = 'hsl(214, 14%, 68%)';
      ctx.font = '10.5px "JetBrains Mono", monospace';
      const shortLbl = idx === 0 ? 'Spot' : `D${idx / 2} Close`;
      ctx.fillText(shortLbl, x - 18, H - 10);
    }
  });
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

    // 68% Band around Predicted Price
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

    // Predicted Line (Amber)
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

    // Actual Market Price Line (Emerald)
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

    // Axis labels
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
    // Self-Tuning Weight Evolution Mode
    const hist = calHistory || [];
    if (hist.length < 2) return;

    const xOf = (i) => pad.left + (i / Math.max(hist.length - 1, 1)) * plotW;
    const yOf = (w) => pad.top + ((0.55 - w) / 0.55) * plotH;

    const seriesColors = {
      overnight_premarket: 'hsl(158, 78%, 45%)',
      sector_peers: 'hsl(38, 92%, 54%)',
      whale_institutional: 'hsl(196, 86%, 54%)',
      technical_sentiment: 'hsl(350, 82%, 58%)',
      macro_regime: 'hsl(214, 25%, 72%)',
    };

    Object.entries(seriesColors).forEach(([key, col]) => {
      ctx.beginPath();
      hist.forEach((h, i) => {
        const w = (h.weights && h.weights[key]) || 0.2;
        const x = xOf(i);
        const y = yOf(w);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    ctx.fillStyle = 'hsl(215, 12%, 55%)';
    ctx.font = '10.5px "JetBrains Mono", monospace';
    [0.1, 0.25, 0.4, 0.5].forEach((w) => {
      ctx.fillText(`${(w * 100).toFixed(0)}%`, 8, yOf(w) + 3);
    });
    ctx.fillText('Step 1 (Prior)', pad.left, H - 8);
    ctx.fillText(`Step ${hist.length} (Latest Tuned)`, W - pad.right - 115, H - 8);
  }
}

function bindEvents() {
  document.getElementById('btn-refresh-data').addEventListener('click', () => fetchDashboard(true));
  document.getElementById('btn-fine-tune').addEventListener('click', triggerFineTuning);
  document.getElementById('btn-run-whatif').addEventListener('click', runWhatIfScenario);

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
      drawWeeklyFanChart(state.payload.simulation.one_week_forecast.timeline, state.payload.simulation.entry_recommendations);
      drawCalibrationOrComparisonChart(
        state.payload.simulation.live_vs_simulated.historical_comparison_series,
        state.payload.calibration.history
      );
    }
  });
}

function updateSliderLabels() {
  const spot = (state.payload && state.payload.market && state.payload.market.current_price) || 190.0;
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
  updateSliderLabels();
  fetchDashboard(false);
});
