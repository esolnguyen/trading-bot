/**
 * vector_panel.js — Vector memory table, semantic rules cards, context pills.
 */

let _sortKey = null;
let _sortAsc = true;

export function initVectorPanel(data = {}) {
  const panel = document.getElementById('vector-panel');
  if (!panel) return;

  const memories = Array.isArray(data.memories) ? data.memories : [];
  const rules    = Array.isArray(data.rules)    ? data.rules    : [];
  const stats    = data.stats ?? {};

  panel.innerHTML = '';

  // Stats row
  if (Object.keys(stats).length) {
    panel.appendChild(buildStatsRow(stats));
  }

  // Context factor pills
  const factors = Array.isArray(data.context_factors) ? data.context_factors : [];
  if (factors.length) {
    panel.appendChild(buildFactorPills(factors));
  }

  // Memory table
  if (memories.length) {
    const wrapper = document.createElement('div');
    wrapper.className = 'vector-table-wrapper';
    wrapper.appendChild(buildMemoryTable(memories));
    panel.appendChild(wrapper);
  } else {
    const empty = document.createElement('p');
    empty.className   = 'empty-state';
    empty.textContent = 'No vector memories yet.';
    panel.appendChild(empty);
  }

  // Semantic rules
  if (rules.length) {
    const section = document.createElement('div');
    section.style.marginTop = '16px';
    const heading = document.createElement('h3');
    heading.textContent = 'Semantic Rules';
    heading.style.cssText = 'font-size:13px;font-weight:600;color:var(--color-text-muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;';
    section.appendChild(heading);
    const rulesGrid = buildRulesGrid(rules);
    section.appendChild(rulesGrid);
    panel.appendChild(section);
  }
}

function buildStatsRow(stats) {
  const row = document.createElement('div');
  row.className = 'kpi-grid';
  row.style.marginBottom = '12px';

  for (const [key, val] of Object.entries(stats)) {
    const card = document.createElement('div');
    card.className = 'kpi-card';
    card.innerHTML = `<div class="kpi-label">${formatKey(key)}</div><div class="kpi-value">${formatVal(val)}</div>`;
    row.appendChild(card);
  }
  return row;
}

function buildFactorPills(factors) {
  const row = document.createElement('div');
  row.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;';
  for (const f of factors) {
    const pill = document.createElement('span');
    pill.className   = 'context-pill';
    pill.textContent = typeof f === 'string' ? f : JSON.stringify(f);
    row.appendChild(pill);
  }
  return row;
}

const MEM_COLS = [
  { key: 'timestamp', label: 'Time',    fmt: v => v ? new Date(v).toLocaleString() : '—' },
  { key: 'action',    label: 'Action',  fmt: v => v ?? '—' },
  { key: 'symbol',    label: 'Symbol',  fmt: v => v ?? '—' },
  { key: 'pnl_pct',   label: 'P&L %',  fmt: v => v != null ? `${parseFloat(v).toFixed(2)}%` : '—' },
  { key: 'reason',    label: 'Reason',  fmt: v => v ? String(v).slice(0, 60) : '—' },
];

function buildMemoryTable(memories) {
  const table = document.createElement('table');
  table.className = 'vector-table';

  // Head
  const thead = table.createTHead();
  const hrow  = thead.insertRow();
  MEM_COLS.forEach(col => {
    const th = document.createElement('th');
    th.textContent = col.label;
    th.dataset.key = col.key;
    if (_sortKey === col.key) th.classList.add(_sortAsc ? 'sort-asc' : 'sort-desc');
    th.addEventListener('click', () => {
      if (_sortKey === col.key) { _sortAsc = !_sortAsc; }
      else { _sortKey = col.key; _sortAsc = true; }
      initVectorPanel({ memories, rules: [], stats: {} });
    });
    hrow.appendChild(th);
  });

  // Sort
  const sorted = [...memories].sort((a, b) => {
    if (!_sortKey) return 0;
    const av = a[_sortKey] ?? '';
    const bv = b[_sortKey] ?? '';
    const cmp = av < bv ? -1 : av > bv ? 1 : 0;
    return _sortAsc ? cmp : -cmp;
  });

  // Body
  const tbody = table.createTBody();
  for (const mem of sorted) {
    const row = tbody.insertRow();
    MEM_COLS.forEach(col => {
      const td = row.insertCell();
      td.textContent = col.fmt(mem[col.key]);
      if (col.key === 'pnl_pct' && mem[col.key] != null) {
        td.className = parseFloat(mem[col.key]) >= 0 ? 'text-success' : 'text-danger';
      }
    });
  }

  return table;
}

function buildRulesGrid(rules) {
  const grid = document.createElement('div');
  grid.className = 'rules-grid';
  for (const rule of rules) {
    const card = document.createElement('div');
    card.className   = 'rule-card';
    card.textContent = typeof rule === 'string' ? rule : (rule.rule ?? rule.text ?? JSON.stringify(rule));
    grid.appendChild(card);
  }
  return grid;
}

function formatKey(k) {
  return k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatVal(v) {
  if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
  return v ?? '—';
}
