/**
 * ui.js — Sidebar navigation and tab switching.
 */

export function initUI() {
  const navBtns = document.querySelectorAll('.nav-btn');
  const panels  = document.querySelectorAll('.tab-panel');

  function activateTab(tabId) {
    navBtns.forEach(btn => {
      const active = btn.dataset.tab === tabId;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', String(active));
      btn.setAttribute('tabindex', active ? '0' : '-1');
    });
    panels.forEach(panel => {
      panel.classList.toggle('active', panel.id === `tab-${tabId}`);
    });

    // Trigger resize for charts that need it
    window.dispatchEvent(new Event('resize'));

    if (tabId === 'brain' && window.fitSynapseNetwork) {
      setTimeout(() => window.fitSynapseNetwork(), 60);
    }
  }

  navBtns.forEach((btn, index) => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));

    btn.addEventListener('keydown', e => {
      const items = [...navBtns];
      const idx   = items.indexOf(btn);
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        items[(idx + 1) % items.length]?.focus();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        items[(idx - 1 + items.length) % items.length]?.focus();
      } else if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        activateTab(btn.dataset.tab);
      }
    });
  });
}
