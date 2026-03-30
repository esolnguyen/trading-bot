/**
 * position_panel.js — Current open position display with SL/TP gauge and P&L.
 */

export function initPositionPanel(position) {
  const panel   = document.getElementById('position-panel');
  const kpiPos  = document.getElementById('kpi-position-value');
  const kpiPnl  = document.getElementById('kpi-pnl-value');
  if (!panel) return;

  if (!position || !position.side) {
    panel.innerHTML = '<p class="empty-state">No open position.</p>';
    if (kpiPos) { kpiPos.textContent = 'None'; kpiPos.className = 'kpi-value text-muted'; }
    if (kpiPnl) { kpiPnl.textContent = '—';    kpiPnl.className = 'kpi-value'; }
    return;
  }

  const side     = position.side ?? 'LONG';
  const entry    = parseFloat(position.entry_price   ?? position.entry ?? 0);
  const current  = parseFloat(position.current_price ?? position.price ?? 0);
  const sl       = parseFloat(position.stop_loss     ?? position.sl    ?? 0);
  const tp       = parseFloat(position.take_profit   ?? position.tp    ?? 0);
  const qty      = parseFloat(position.quantity      ?? position.qty   ?? 0);
  const pnlPct   = current && entry ? ((current - entry) / entry * 100 * (side === 'SHORT' ? -1 : 1)) : 0;
  const pnlUSD   = qty ? ((current - entry) * qty * (side === 'SHORT' ? -1 : 1)) : null;

  // SL/TP gauge (0–100 where 0=SL, 100=TP)
  const slGaugePct = calculateGaugePct(current, entry, sl, tp, side);

  panel.innerHTML = `
    <div class="position-grid">
      <div class="position-field">
        <div class="position-field-label">Side</div>
        <div class="position-field-value ${side.toLowerCase()}">${side}</div>
      </div>
      <div class="position-field">
        <div class="position-field-label">Entry Price</div>
        <div class="position-field-value">${formatPrice(entry)}</div>
      </div>
      <div class="position-field">
        <div class="position-field-label">Current Price</div>
        <div class="position-field-value">${formatPrice(current)}</div>
      </div>
      <div class="position-field">
        <div class="position-field-label">Stop Loss</div>
        <div class="position-field-value text-danger">${sl ? formatPrice(sl) : '—'}</div>
      </div>
      <div class="position-field">
        <div class="position-field-label">Take Profit</div>
        <div class="position-field-value text-success">${tp ? formatPrice(tp) : '—'}</div>
      </div>
      <div class="position-field">
        <div class="position-field-label">Quantity</div>
        <div class="position-field-value">${qty || '—'}</div>
      </div>
      <div class="position-field">
        <div class="position-field-label">Open P&L</div>
        <div class="position-field-value ${pnlPct >= 0 ? 'text-success' : 'text-danger'}">
          ${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%
          ${pnlUSD !== null ? `<span style="font-size:12px;margin-left:4px">(${pnlUSD >= 0 ? '+' : ''}$${Math.abs(pnlUSD).toFixed(2)})</span>` : ''}
        </div>
      </div>
    </div>
    ${sl && tp ? `
    <div style="margin-top:4px;">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--color-text-muted);margin-bottom:2px;">
        <span>SL ${formatPrice(sl)}</span>
        <span>TP ${formatPrice(tp)}</span>
      </div>
      <div class="gauge-bar"><div class="gauge-fill" style="width:${slGaugePct.toFixed(1)}%;background:${gaugeColor(slGaugePct)}"></div></div>
    </div>` : ''}
    ${position.confluence_factors?.length ? `
    <div style="margin-top:12px;">
      <div style="font-size:11px;font-weight:600;color:var(--color-text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">Confluence Factors</div>
      <div style="display:flex;flex-wrap:wrap;gap:4px;">
        ${position.confluence_factors.map(f => `<span class="topic-pill">${f}</span>`).join('')}
      </div>
    </div>` : ''}
  `;

  if (kpiPos) { kpiPos.textContent = side; kpiPos.className = `kpi-value ${side.toLowerCase()}`; }
  if (kpiPnl) {
    kpiPnl.textContent = `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%`;
    kpiPnl.className   = `kpi-value ${pnlPct >= 0 ? 'positive' : 'negative'}`;
  }
}

function formatPrice(v) {
  if (!v) return '—';
  return Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 8 });
}

function calculateGaugePct(current, entry, sl, tp, side) {
  if (!sl || !tp || !current) return 50;
  const lo = side === 'SHORT' ? tp : sl;
  const hi = side === 'SHORT' ? sl : tp;
  if (hi === lo) return 50;
  const pct = ((current - lo) / (hi - lo)) * 100;
  return Math.max(0, Math.min(100, pct));
}

function gaugeColor(pct) {
  if (pct < 30) return 'var(--color-danger)';
  if (pct < 60) return 'var(--color-warning)';
  return 'var(--color-success)';
}
