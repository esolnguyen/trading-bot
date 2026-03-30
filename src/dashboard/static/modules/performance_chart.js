/**
 * performance_chart.js — ApexCharts area chart for trade P&L history.
 */

let _chart = null;

export function initPerformanceChart(tradeHistory = []) {
  const el = document.getElementById('performance-chart');
  if (!el) return;

  const { series, annotations } = buildSeriesData(tradeHistory);

  if (_chart) {
    _chart.updateSeries(series);
    _chart.updateOptions({ annotations });
    return;
  }

  const options = {
    chart: {
      type:       'area',
      height:     280,
      background: 'transparent',
      toolbar:    { show: false },
      zoom:       { enabled: true },
      animations: { enabled: false },
    },
    theme: { mode: 'dark' },
    colors: ['#58a6ff'],
    stroke: { curve: 'smooth', width: 2 },
    fill: {
      type: 'gradient',
      gradient: {
        shadeIntensity: 1,
        opacityFrom:    0.4,
        opacityTo:      0.0,
        stops:          [0, 100],
      },
    },
    series,
    xaxis: {
      type:   'datetime',
      labels: { style: { colors: '#8b949e' } },
    },
    yaxis: {
      labels: {
        style:     { colors: '#8b949e' },
        formatter: v => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`,
      },
    },
    grid: { borderColor: '#30363d' },
    dataLabels:  { enabled: false },
    tooltip: {
      x: { format: 'dd MMM HH:mm' },
      y: { formatter: v => `${v >= 0 ? '+' : ''}${v.toFixed(3)}%` },
    },
    annotations,
    noData: { text: 'No trade data yet.', style: { color: '#8b949e' } },
  };

  _chart = new ApexCharts(el, options);  // eslint-disable-line no-undef
  _chart.render();
}

function buildSeriesData(tradeHistory) {
  const cumulative = [];
  const points     = [];
  let runningPnl   = 0;

  for (const trade of tradeHistory) {
    const ts = trade.close_time || trade.timestamp || trade.time;
    if (!ts) continue;
    const t   = new Date(ts).getTime();
    const pnl = parseFloat(trade.pnl_pct ?? trade.pnl ?? 0);
    runningPnl += pnl;
    cumulative.push({ x: t, y: parseFloat(runningPnl.toFixed(4)) });

    const isWin = pnl >= 0;
    points.push({
      x:          t,
      y:          parseFloat(runningPnl.toFixed(4)),
      marker:     { size: 5, fillColor: isWin ? '#3fb950' : '#f85149', strokeColor: '#fff', strokeWidth: 1 },
      label:      { text: `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%`, style: { fontSize: '10px', color: isWin ? '#3fb950' : '#f85149', background: 'transparent', border: 0 } },
    });
  }

  return {
    series:      [{ name: 'Cumulative P&L', data: cumulative }],
    annotations: { points },
  };
}
