/**
 * Trading Bot Dashboard — Main JS entry point
 */
import { initUI } from './modules/ui.js?v=4.5';
import { initWebSocket } from './modules/websocket.js?v=4.5';

const API_BASE = '';

async function fetchJSON(path) {
  try {
    const res = await fetch(API_BASE + path);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function setKpi(id, value, style = 'neutral') {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = value;
    el.className = `kpi-value ${style}`;
  }
}

function renderKpis(status, stats) {
  const grid = document.getElementById('kpi-grid');
  if (!grid) return;
  const items = [
    { label: 'Cycle',       value: status?.cycle        ?? '--',    id: 'kpi-cycle' },
    { label: 'Decision',    value: status?.last_decision ?? '--',   id: 'kpi-decision' },
    { label: 'Symbol',      value: status?.last_symbol  ?? '--',    id: 'kpi-symbol' },
    { label: 'Win Rate',    value: stats?.win_rate != null ? `${stats.win_rate.toFixed(1)}%` : '--', id: 'kpi-win' },
    { label: 'Total P&L',   value: stats?.total_pnl != null ? `$${stats.total_pnl.toFixed(2)}` : '--', id: 'kpi-pnl' },
    { label: 'Sharpe',      value: stats?.sharpe_ratio != null ? stats.sharpe_ratio.toFixed(2) : '--', id: 'kpi-sharpe' },
  ];
  grid.innerHTML = items.map(({ label, value, id }) => `
    <div class="kpi-card">
      <div class="kpi-label">${label}</div>
      <div class="kpi-value neutral" id="${id}">${value}</div>
    </div>`).join('');
}

async function pollAll() {
  const [status, history, stats, prompt, response] = await Promise.allSettled([
    fetchJSON('/api/monitor/status'),
    fetchJSON('/api/performance/history'),
    fetchJSON('/api/performance/stats'),
    fetchJSON('/api/monitor/last-prompt'),
    fetchJSON('/api/monitor/last-response'),
  ]);

  const s = status.value;
  const t = stats.value?.stats ?? {};
  renderKpis(s, t);

  const promptEl = document.getElementById('prompt-panel');
  if (promptEl && prompt.value?.prompt) promptEl.textContent = prompt.value.prompt;

  const responseEl = document.getElementById('response-panel');
  if (responseEl && response.value?.response) responseEl.textContent = response.value.response;

  // Render statistics
  const statsEl = document.getElementById('statistics-panel');
  if (statsEl && Object.keys(t).length) {
    statsEl.innerHTML = `<div class="stats-grid">${
      Object.entries(t).map(([k, v]) =>
        `<div class="stat-card"><div class="stat-label">${k}</div><div class="stat-value">${
          typeof v === 'number' ? v.toFixed(2) : v
        }</div></div>`
      ).join('')
    }</div>`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initUI();
  initWebSocket();
  pollAll();
  setInterval(pollAll, 10000);
});
