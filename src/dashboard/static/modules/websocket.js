/**
 * websocket.js — WebSocket client with exponential-backoff reconnection.
 */

const WS_URL    = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
const MIN_DELAY = 1_000;
const MAX_DELAY = 30_000;

let _socket  = null;
let _retries = 0;
let _cb      = null;

function setStatus(connected) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  if (dot)  { dot.className  = `status-dot ${connected ? 'connected' : 'disconnected'}`; }
  if (text) { text.textContent = connected ? 'Connected' : 'Disconnected'; }
}

function connect() {
  try { _socket = new WebSocket(WS_URL); }
  catch (err) { console.warn('[ws] open failed:', err); schedule(); return; }

  _socket.addEventListener('open', () => {
    _retries = 0;
    setStatus(true);
  });

  _socket.addEventListener('message', ({ data }) => {
    try {
      const payload = JSON.parse(data);
      if (typeof _cb === 'function') {
        _cb(
          payload.bot_status    ?? payload.type ?? null,
          payload.next_check_utc ?? null,
          payload.position       ?? null,
        );
      }
      if (payload.type === 'analysis_complete') {
        document.dispatchEvent(new CustomEvent('analysis-complete'));
      }
    } catch (err) { console.warn('[ws] parse error:', err); }
  });

  _socket.addEventListener('close', () => { setStatus(false); schedule(); });
  _socket.addEventListener('error', () => { /* close fires next */ });
}

function schedule() {
  const delay = Math.min(MIN_DELAY * 2 ** _retries, MAX_DELAY);
  _retries++;
  setTimeout(connect, delay);
}

/**
 * @param {function} onUpdate  Called with (botStatus, nextCheckUTC, position).
 */
export function initWebSocket(onUpdate) {
  _cb = onUpdate;
  connect();
}

export function formatDuration(seconds) {
  if (seconds < 0) seconds = 0;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
