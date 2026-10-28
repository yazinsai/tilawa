import { mountItem, VERDICTS } from './item.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const url = new URL(location.href);
const queuePath = url.searchParams.get('queue');
const reviewer = url.searchParams.get('reviewer') || 'me';
const st = { queue: null, reviews: {}, tally: null, i: 0, view: null, unsaved: new Map() };

async function api(path, init) {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = body.detail;
    throw new Error(d?.problems ? `This queue has problems:\n• ${d.problems.join('\n• ')}` : typeof d === 'string' ? d : `HTTP ${r.status}`);
  }
  return body;
}

function header() {
  const t = st.tally;
  const state = { pass: 'Pass', fail: 'Fail', open: 'Passing so far' }[t.state];
  $('head').innerHTML = `
    <span class="big">Reviewed ${t.reviewed} of ${t.total}</span>
    <span>Bad: <b>${t.bad}</b></span>
    <span class="bar ${t.state}" title="Bad = bad label, bad audio or bad splice">${state} · at most ${t.maxBad} bad in ${t.total}</span>
    ${st.unsaved.size ? `<span class="bar fail">Not saved: ${esc([...st.unsaved.values()].join(', '))}. Go back and choose again.</span>` : ''}
    <span class="who">${esc(st.queue.testSet)} · ${esc(queuePath)} · reviewing as <a href="#" id="who">${esc(reviewer)}</a></span>`;
  $('who').onclick = (e) => {
    e.preventDefault();
    const n = prompt('Your name (letters, digits, - or _)', reviewer);
    if (n) {
      url.searchParams.set('reviewer', n);
      location.href = url;
    }
  };
}

function show(i, autoplay = true) {
  const items = st.queue.items;
  st.i = Math.max(0, Math.min(items.length - 1, i));
  const item = items[st.i];
  st.view?.destroy();
  st.view = mountItem($('main'), item, {
    review: st.reviews[item.id],
    verdicts: VERDICTS,
    onVerdict: (verdict, note, how) => {
      const saving = api('/api/verdict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ queue: queuePath, reviewer, id: item.id, verdict, note }),
      }).then(
        (r) => {
          st.reviews[item.id] = r.review;
          st.tally = r.tally;
          st.unsaved.delete(item.id);
          header();
          return r.review;
        },
        (e) => {
          st.unsaved.set(item.id, item.title || item.id);
          header();
          throw e;
        },
      );
      if (how !== 'note') setTimeout(() => st.queue.items[st.i] === item && show(st.i + 1), 250);
      return saving;
    },
  });
  if (autoplay) st.view.play('audio');
  $('pos').textContent = `${st.i + 1} / ${items.length}`;
  url.searchParams.set('item', item.id);
  history.replaceState(null, '', url);
}

const nextTodo = () => {
  const items = st.queue.items;
  for (let k = 1; k <= items.length; k++) {
    const j = (st.i + k) % items.length;
    if (!st.reviews[items[j].id]) return show(j);
  }
};

document.addEventListener('keydown', (e) => {
  if (!st.view || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target.closest('textarea, input, select')) {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  const k = e.key;
  if (e.target.tagName === 'BUTTON') e.target.blur();
  if (k === ' ') st.view.play('audio');
  else if (k === 'b') st.view.play('before');
  else if (k === 'a') st.view.play('after');
  else if (k === 'j') show(st.i + 1);
  else if (k === 'k') show(st.i - 1);
  else if (/^[1-9]$/.test(k)) st.view.choose(Number(k) - 1).catch(() => {});
  else return;
  e.preventDefault();
});

async function start() {
  if (!queuePath) {
    const { queues } = await api('/api/queues');
    $('main').innerHTML = `<div class="msg"><h2>Pick a review queue</h2>
      <ul>${queues.map((q) => `<li><a href="?queue=${encodeURIComponent(q)}&reviewer=${encodeURIComponent(reviewer)}">${esc(q)}</a></li>`).join('') || '<li>No queues yet.</li>'}</ul>
      <p>Or open <code>?queue=&lt;path under /lab/data&gt;</code>.</p></div>`;
    return;
  }
  const d = await api(`/api/queue?path=${encodeURIComponent(queuePath)}&reviewer=${encodeURIComponent(reviewer)}`);
  Object.assign(st, { queue: d.queue, reviews: d.reviews, tally: d.tally });
  header();
  $('nav').hidden = $('keys').hidden = false;
  $('prev').onclick = () => show(st.i - 1);
  $('next').onclick = () => show(st.i + 1);
  $('todo').onclick = nextTodo;
  const at = st.queue.items.findIndex((it) => it.id === url.searchParams.get('item'));
  const todo = st.queue.items.findIndex((it) => !st.reviews[it.id]);
  show(at >= 0 ? at : Math.max(0, todo), false);
}

start().catch((e) => {
  $('main').innerHTML = `<div class="msg err">${esc(e.message)}</div>`;
});
