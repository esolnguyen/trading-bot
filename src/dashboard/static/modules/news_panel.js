/**
 * news_panel.js — News article cards with topic pills and XSS-safe URLs.
 */

export function initNewsPanel(news = []) {
  const panel = document.getElementById('news-panel');
  if (!panel) return;

  if (!news.length) {
    panel.innerHTML = '<p class="empty-state">No news articles yet.</p>';
    return;
  }

  const grid = document.createElement('div');
  grid.className = 'news-grid';

  for (const article of news) {
    grid.appendChild(buildNewsCard(article));
  }

  panel.innerHTML = '';
  panel.appendChild(grid);
}

function buildNewsCard(article) {
  const card = document.createElement('div');
  card.className = 'news-card';

  const title   = sanitize(article.title   ?? article.headline ?? 'Untitled');
  const source  = sanitize(article.source  ?? article.publisher ?? '');
  const summary = sanitize(article.summary ?? article.description ?? article.body ?? '');
  const url     = safeUrl(article.url      ?? article.link ?? '');
  const topics  = Array.isArray(article.topics) ? article.topics : (article.categories ? [article.categories] : []);
  const dateStr = formatDate(article.published_utc ?? article.published_at ?? article.date ?? article.timestamp);

  const titleHtml = url
    ? `<a href="${url}" target="_blank" rel="noopener noreferrer" class="news-title">${title}</a>`
    : `<span class="news-title">${title}</span>`;

  const summaryHtml = summary
    ? `<div class="news-summary">${summary.slice(0, 250)}${summary.length > 250 ? '…' : ''}</div>`
    : '';

  const topicsHtml = topics.length
    ? `<div class="news-topics">${topics.map(t => `<span class="topic-pill">${sanitize(t)}</span>`).join('')}</div>`
    : '';

  card.innerHTML = `
    <div class="news-card-header">
      ${titleHtml}
      <span class="news-date">${dateStr}</span>
    </div>
    ${source ? `<div class="news-source">${source}</div>` : ''}
    ${summaryHtml}
    ${topicsHtml}
  `;

  return card;
}

function sanitize(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;');
}

function safeUrl(url) {
  if (!url) return '';
  try {
    const u = new URL(url);
    if (u.protocol === 'https:' || u.protocol === 'http:') return u.href;
  } catch {/* ignore */}
  return '';
}

function formatDate(raw) {
  if (!raw) return '';
  try {
    const d = new Date(raw);
    if (isNaN(d.getTime())) return String(raw);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  } catch {
    return String(raw);
  }
}
