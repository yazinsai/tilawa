import { icon } from './item.js';

// Shared by the Autolab page and its live progress page.

export const $ = (id) => document.getElementById(id);
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
export const NAMES = { listener: 'Listener', tracker: 'Tracker', checker: 'Checker', overall: 'Overall' };
export const qs = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v != null && v !== '')).toString();
export const day = (s) => (s ? new Date(s).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '');
export const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
export const fewer = (n) => (n == null ? '–' : n === 0 ? 'no change' : n < 0 ? `${-n} fewer` : `${n} more`);
export const sentence = (s) => (s ? s[0].toUpperCase() + s.slice(1) : '');
export const sw = (c) => (NAMES[c] ? `<i class="sw" style="--c:var(--${c === 'overall' ? 'g-900' : c})"></i>` : '');

export function span(s) {
  if (s == null) return '–';
  if (s < 90) return `${Math.max(0, s)} s`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m} min`;
  const h = Math.floor(m / 60);
  if (h < 36) return `${h} h${m % 60 ? ` ${m % 60} min` : ''}`;
  return plural(Math.round(h / 24), 'day');
}
export const ago = (at) => (at ? span(Math.round((Date.now() - new Date(at)) / 1000)) + ' ago' : 'never');

export async function api(path) {
  const r = await fetch(path, { cache: 'no-store' });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${r.status}`);
  return body;
}

// An experiment in plain words: what it is about, what it expects, and what came of it once decided.

export const RESULT_KIND = { folded: 'better', discarded: 'worse', superseded: 'worse', abandoned: '', open: '' };

export function plainLines(r) {
  const e = r.eli5?.text ? r.eli5 : { text: r.hypothesis || '', plain: false };
  return `<p class="pl-eli5${e.plain ? '' : ' tech'}">${e.plain ? '' : '<i>Not yet in plain words:</i> '}${esc(e.text)}</p>
    ${r.expectText ? `<p class="pl-exp">${esc(r.expectText)}</p>` : ''}
    ${r.resultText ? `<p class="pl-res ${RESULT_KIND[r.state] ?? ''}">${esc(r.resultText)}</p>` : ''}`;
}

// Top bar: the same tabs as the dashboard, and the supervisor's heartbeat on the right.

export function beat(started, h) {
  if (!started || !h) return '<span class="who"><i class="dot"></i>Not started</span>';
  const kind = h.stale ? 'fail' : h.paused ? 'wait' : 'pass';
  const text = h.stale ? `No check-in for ${h.at ? span(h.ageSeconds) : 'ever'}` : h.paused ? 'Paused' : `Checked in ${span(h.ageSeconds)} ago`;
  return `<span class="who al-beat ${kind}" title="The supervisor checks in every ${span(h.tickSeconds)}; it counts as stopped after 3 missed check-ins"><i class="dot ${kind}"></i>${text}</span>`;
}

export function topbar(v) {
  const tab = (href, text, on) => `<a href="${href}" ${on ? 'aria-current="page"' : ''}>${text}</a>`;
  $('head').innerHTML = `<a class="brand" href="/">${icon('mark')}Tilawa lab</a>
    <nav class="tabs">${tab('/', 'Score')}${tab('/#/release', 'Release')}${tab('/review', 'Listening check')}${tab('/record', 'Record')}${tab('/live/', 'Live mic')}${tab('/autolab', 'Autolab', true)}</nav>
    ${v ? beat(v.started, v.heartbeat) : ''}`;
}

export function staleNotice(h) {
  if (!h?.stale) return '';
  return `<div class="notice err"><b>The supervisor seems to have stopped.</b> Its last check-in was ${h.at ? `${span(h.ageSeconds)} ago (${day(h.at)})` : 'never'}; it should check in every ${span(h.tickSeconds)}. Nothing on this page moves until it’s running again.</div>`;
}
