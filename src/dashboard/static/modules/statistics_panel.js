/**
 * statistics_panel.js — Statistics grid cards with descriptions.
 */

const STAT_META = {
  total_trades:          { label: 'Total Trades',    desc: 'Number of closed positions.' },
  winning_trades:        { label: 'Winning Trades',  desc: 'Positions closed with positive P&L.' },
  losing_trades:         { label: 'Losing Trades',   desc: 'Positions closed with negative P&L.' },
  win_rate:              { label: 'Win Rate',         desc: '% of trades that were profitable.', unit: '%', decimals: 1 },
  total_pnl_pct:         { label: 'Total P&L',        desc: 'Cumulative return across all trades.', unit: '%', decimals: 2 },
  avg_trade_pct:         { label: 'Avg Trade P&L',   desc: 'Average per-trade return.', unit: '%', decimals: 2 },
  best_trade_pct:        { label: 'Best Trade',       desc: 'Single best trade return.', unit: '%', decimals: 2 },
  worst_trade_pct:       { label: 'Worst Trade',      desc: 'Single worst trade return.', unit: '%', decimals: 2 },
  sharpe_ratio:          { label: 'Sharpe Ratio',     desc: 'Risk-adjusted return (excess return / volatility). >1 is good.', decimals: 2 },
  sortino_ratio:         { label: 'Sortino Ratio',    desc: 'Like Sharpe but only penalises downside volatility. >1 is good.', decimals: 2 },
  max_drawdown_pct:      { label: 'Max Drawdown',     desc: 'Largest peak-to-trough decline.', unit: '%', decimals: 2 },
  profit_factor:         { label: 'Profit Factor',    desc: 'Gross profit / gross loss. >1 is profitable.', decimals: 2 },
  avg_win_pct:           { label: 'Avg Win',          desc: 'Average P&L of winning trades.', unit: '%', decimals: 2 },
  avg_loss_pct:          { label: 'Avg Loss',         desc: 'Average P&L of losing trades.', unit: '%', decimals: 2 },
  largest_consecutive_wins:  { label: 'Max Win Streak',  desc: 'Longest consecutive winning run.' },
  largest_consecutive_losses:{ label: 'Max Loss Streak', desc: 'Longest consecutive losing run.' },
  total_pnl_quote:       { label: 'Total P&L ($)',    desc: 'Cumulative P&L in quote currency.', prefix: '$', decimals: 2 },
};

export function initStatisticsPanel(stats = {}) {
  const panel = document.getElementById('statistics-panel');
  if (!panel) return;

  const entries = Object.entries(stats).filter(([k]) => k !== 'calculation_date');

  if (!entries.length) {
    panel.innerHTML = '<p class="empty-state">No statistics yet. Close a trade to see metrics.</p>';
    return;
  }

  const grid = document.createElement('div');
  grid.className = 'stats-grid';

  for (const [key, val] of entries) {
    const meta = STAT_META[key] ?? { label: formatKey(key), desc: '' };
    const card = document.createElement('div');
    card.className = 'stat-card';

    const formatted = formatStatValue(val, meta);
    const colorClass = getColorClass(key, val);

    card.innerHTML = `
      <div class="stat-name">${meta.label}</div>
      <div class="stat-value ${colorClass}">${formatted}</div>
      ${meta.desc ? `<div class="stat-desc">${meta.desc}</div>` : ''}
    `;
    grid.appendChild(card);
  }

  panel.innerHTML = '';
  panel.appendChild(grid);
}

function formatStatValue(val, meta) {
  if (val == null || val === '') return '—';
  const num = parseFloat(val);
  if (isNaN(num)) return String(val);
  const decimals = meta.decimals ?? 0;
  const prefix   = meta.prefix  ?? '';
  const unit     = meta.unit    ?? '';
  return `${prefix}${num.toFixed(decimals)}${unit}`;
}

function getColorClass(key, val) {
  const num = parseFloat(val);
  if (isNaN(num)) return '';
  if (key === 'max_drawdown_pct') return num > 0 ? 'text-danger' : '';
  if (key === 'worst_trade_pct')  return num < 0  ? 'text-danger' : 'text-success';
  if (key === 'best_trade_pct')   return num >= 0 ? 'text-success' : 'text-danger';
  if (['total_pnl_pct', 'total_pnl_quote', 'avg_trade_pct'].includes(key)) {
    return num >= 0 ? 'text-success' : 'text-danger';
  }
  if (key === 'sharpe_ratio' || key === 'sortino_ratio') {
    return num >= 1 ? 'text-success' : num >= 0 ? 'text-warning' : 'text-danger';
  }
  return '';
}

function formatKey(k) {
  return k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
