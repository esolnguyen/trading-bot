/**
 * synapse_viewer.js — vis-network graph for trade sequence chains.
 * BUY → UPDATE → CLOSE chains visualised as a directed graph.
 */

let _network = null;

export function initSynapseViewer(memories = []) {
  const container = document.getElementById('synapse-viewer');
  if (!container) return;

  const { nodes, edges } = buildGraph(memories);

  const data = {
    nodes: new vis.DataSet(nodes),  // eslint-disable-line no-undef
    edges: new vis.DataSet(edges),  // eslint-disable-line no-undef
  };

  const options = {
    nodes: {
      shape:       'dot',
      size:        14,
      font:        { size: 11, color: '#e6edf3' },
      borderWidth: 1,
      shadow:      false,
    },
    edges: {
      arrows:     { to: { enabled: true, scaleFactor: 0.6 } },
      color:      { color: '#30363d', highlight: '#58a6ff' },
      smooth:     { type: 'curvedCW', roundness: 0.2 },
      font:       { size: 9, color: '#8b949e', align: 'middle' },
      width:      1.5,
    },
    layout: { improvedLayout: true },
    physics: {
      enabled:   true,
      stabilization: { iterations: 150 },
      barnesHut: { gravitationalConstant: -6000, springLength: 100 },
    },
    interaction: { hover: true, tooltipDelay: 200 },
    background: { color: 'transparent' },
  };

  if (_network) {
    _network.setData(data);
  } else {
    _network = new vis.Network(container, data, options);  // eslint-disable-line no-undef
    // Expose fit function for the UI tab-switch trigger
    window.fitSynapseNetwork = () => _network && _network.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
  }
}

function buildGraph(memories) {
  const nodes = [];
  const edges = [];
  const seen  = new Set();

  memories.forEach((mem, idx) => {
    const id    = mem.id    ?? `m-${idx}`;
    const label = mem.action ?? mem.type ?? 'TRADE';
    const pnl   = mem.pnl_pct ?? mem.pnl ?? null;
    const color = pnl === null ? '#58a6ff' : (parseFloat(pnl) >= 0 ? '#3fb950' : '#f85149');
    const title = buildTooltip(mem);

    if (!seen.has(id)) {
      seen.add(id);
      nodes.push({ id, label, color: { background: color, border: color }, title });
    }

    if (mem.prev_id && seen.has(mem.prev_id)) {
      edges.push({ from: mem.prev_id, to: id });
    }
  });

  return { nodes, edges };
}

function buildTooltip(mem) {
  const lines = [];
  if (mem.timestamp || mem.time) lines.push(`Time: ${new Date(mem.timestamp || mem.time).toLocaleString()}`);
  if (mem.symbol)  lines.push(`Symbol: ${mem.symbol}`);
  if (mem.action)  lines.push(`Action: ${mem.action}`);
  if (mem.pnl_pct != null) lines.push(`P&L: ${parseFloat(mem.pnl_pct).toFixed(2)}%`);
  if (mem.reason)  lines.push(`Reason: ${mem.reason}`);
  return lines.join('\n') || 'No details';
}
