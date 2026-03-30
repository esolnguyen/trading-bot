/**
 * visuals.js — Chart image display with lightbox (pan/zoom).
 */

export function initVisuals(charts = {}) {
  const panel = document.getElementById('visuals-panel');
  if (!panel) return;

  const entries = Object.entries(charts);
  if (!entries.length) {
    panel.innerHTML = '<p class="empty-state">No charts available.</p>';
    return;
  }

  const grid = document.createElement('div');
  grid.className = 'visuals-grid';

  for (const [name, data] of entries) {
    if (!data) continue;
    const src = data.startsWith('data:') ? data : `data:image/png;base64,${data}`;

    const item = document.createElement('div');
    item.className = 'visuals-item';
    item.title     = `Click to enlarge — ${name}`;

    const img = document.createElement('img');
    img.src = src;
    img.alt = name;
    img.loading = 'lazy';

    const ts = document.createElement('div');
    ts.className   = 'visuals-timestamp';
    ts.textContent = new Date().toLocaleTimeString();

    item.appendChild(img);
    item.appendChild(ts);
    item.addEventListener('click', () => openLightbox(src, name));
    grid.appendChild(item);
  }

  panel.innerHTML = '';
  panel.appendChild(grid);
  initLightbox();
}

let _lightboxInit = false;

function initLightbox() {
  if (_lightboxInit) return;
  _lightboxInit = true;

  const lightbox = document.getElementById('image-lightbox');
  const closeBtn = document.getElementById('lightbox-close');
  const imgEl    = document.getElementById('lightbox-img');
  if (!lightbox || !closeBtn || !imgEl) return;

  closeBtn.addEventListener('click', () => lightbox.classList.add('hidden'));
  lightbox.addEventListener('click', e => {
    if (e.target === lightbox) lightbox.classList.add('hidden');
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') lightbox.classList.add('hidden');
  });
}

function openLightbox(src, caption) {
  const lightbox  = document.getElementById('image-lightbox');
  const imgEl     = document.getElementById('lightbox-img');
  const captionEl = document.getElementById('lightbox-caption');
  if (!lightbox || !imgEl) return;

  imgEl.src              = src;
  if (captionEl) captionEl.textContent = caption ?? '';
  lightbox.classList.remove('hidden');
}
