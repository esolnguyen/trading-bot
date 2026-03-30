/**
 * fullscreen.js — Panel fullscreen overlay.
 * Moves the actual DOM element (not a clone) so canvas/chart elements render.
 */

let _originalParent  = null;
let _originalNextSib = null;
let _activeEl        = null;

export function initFullscreen() {
  const overlay  = document.getElementById('fullscreen-overlay');
  const closeBtn = document.getElementById('fullscreen-close');
  const content  = document.getElementById('fullscreen-content');
  if (!overlay || !closeBtn || !content) return;

  // Open on fullscreen button clicks
  document.addEventListener('click', e => {
    const btn = e.target.closest('.fullscreen-btn');
    if (!btn) return;
    const targetId = btn.dataset.target;
    const el = document.getElementById(targetId);
    if (!el) return;
    openFullscreen(el, content, overlay);
  });

  // Close button
  closeBtn.addEventListener('click', () => closeFullscreen(overlay, content));

  // Escape key
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !overlay.classList.contains('hidden')) {
      closeFullscreen(overlay, content);
    }
  });

  // Click outside content
  overlay.addEventListener('click', e => {
    if (e.target === overlay) closeFullscreen(overlay, content);
  });
}

function openFullscreen(el, content, overlay) {
  _originalParent  = el.parentNode;
  _originalNextSib = el.nextSibling;
  _activeEl        = el;

  content.appendChild(el);
  overlay.classList.remove('hidden');
  overlay.focus?.();
  window.dispatchEvent(new Event('resize'));
}

function closeFullscreen(overlay, content) {
  overlay.classList.add('hidden');
  if (_activeEl && _originalParent) {
    _originalParent.insertBefore(_activeEl, _originalNextSib);
    window.dispatchEvent(new Event('resize'));
  }
  content.innerHTML = '';
  _activeEl = _originalParent = _originalNextSib = null;
}
