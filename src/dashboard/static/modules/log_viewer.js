/**
 * log_viewer.js — Renders prompt / response text with marked.js + DOMPurify.
 */

export function initLogViewer(elementId, text = '') {
  const el = document.getElementById(elementId);
  if (!el) return;

  if (!text) {
    el.innerHTML = '<p class="empty-state">No data yet.</p>';
    return;
  }

  try {
    // Use marked.js if available, otherwise plain text
    if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {  // eslint-disable-line no-undef
      const raw  = marked.parse(text);                                          // eslint-disable-line no-undef
      el.innerHTML = DOMPurify.sanitize(raw);                                   // eslint-disable-line no-undef
    } else {
      el.textContent = text;
    }
  } catch (err) {
    console.warn('[log_viewer] render error:', err);
    el.textContent = text;
  }
}
