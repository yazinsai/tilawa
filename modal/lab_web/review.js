import { mountItem, VERDICTS, icon, keySheet, railHtml } from './item.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const url = new URL(location.href);
const queuePath = url.searchParams.get('queue');
const reviewer = url.searchParams.get('reviewer') || 'me';
const st = { queue: null, reviews: {}, tally: null, i: 0, view: null, unsaved: new Map() };
const keys = keySheet([
  ['space', 'Play or pause'],
  ['b', 'Original recording before the edit'],
  ['a', 'Original recording after the edit'],
  ['1', 'Sounds right'],
  ['2 3 4', 'Bad label, bad audio, bad splice'],
  ['j', 'Next item'],
  ['k', 'Previous item'],
  ['?', 'Show or hide this'],
]);

async function api(path, init) {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = body.detail;
    throw new Error(d?.problems ? `This queue has problems:\n• ${d.problems.join('\n• ')}` : typeof d === 'string' ? d : `HTTP ${r.status}`);
  }
  return body;
}

function topbar() {
  $('head').innerHTML = `<a class="brand" href="/">${icon('mark')}Tilawa lab</a>
    <nav class="tabs"><a href="/">Score</a><a href="/#/release">Release</a><a href="/review?reviewer=${encodeURIComponent(reviewer)}" aria-current="page">Listening check</a></nav>
    <span class="who">Reviewing as <b>${esc(reviewer)}</b> · <button id="who">change</button></span>`;
  $('who').onclick = () => {
    const n = prompt('Your name (letters, digits, - or _)', reviewer);
    if (n) {
      url.searchParams.set('reviewer', n);
      location.href = url;
    }
  };
}

const verdictText = (v) => VERDICTS.find((x) => x.key === v)?.text || v;

function intro() {
  const t = st.tally;
  const items = st.queue.items;
  const state = { pass: 'Passed', fail: 'Failed', open: 'Passing so far' }[t.state];
  const name = queuePath.split('/').pop().replace(/\.json$/, '');
  $('intro').innerHTML = `<div class="rv-head">
    <h1>${name === 'v1-check' ? 'Listening check' : `Review: ${esc(name)}`} <span class="muted" style="font-weight:400">${esc(st.queue.testSet)}</span></h1>
    <p class="rv-tally"><span>Heard <b class="num">${t.reviewed}</b> of ${t.total}</span><span><b class="num">${t.bad}</b> bad</span>
      <span class="state ${t.state}" title="Bad = bad label, bad audio or bad splice">${state} · at most ${t.maxBad} bad allowed</span></p>
    ${st.unsaved.size ? `<div class="notice err">Not saved: ${esc([...st.unsaved.values()].join(', '))}. Go back and choose again.</div>` : ''}
    ${railHtml(items, st.i, (it) => st.reviews[it.id]?.verdict, (j, v) => `${j + 1}. ${items[j].title || items[j].id}${v ? ` · ${verdictText(v)}` : ''}`)}
  </div>`;
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
          intro();
          return r.review;
        },
        (e) => {
          st.unsaved.set(item.id, item.title || item.id);
          intro();
          throw e;
        },
      );
      if (how !== 'note') setTimeout(() => st.queue.items[st.i] === item && show(st.i + 1), 250);
      return saving;
    },
  });
  if (autoplay) st.view.play('audio');
  $('pos').textContent = `${st.i + 1} of ${items.length}`;
  $('prev').disabled = st.i === 0;
  $('next').disabled = st.i === items.length - 1;
  intro();
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
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target.closest('textarea, input, select')) {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  const k = e.key;
  if (k === 'Escape' && keys.open()) return keys.close();
  if (!st.view) return;
  if (e.target.tagName === 'BUTTON') e.target.blur();
  if (k === '?') keys.toggle();
  else if (k === ' ') st.view.play('audio');
  else if (k === 'b') st.view.play('before');
  else if (k === 'a') st.view.play('after');
  else if (k === 'j') show(st.i + 1);
  else if (k === 'k') show(st.i - 1);
  else if (/^[1-9]$/.test(k)) st.view.choose(Number(k) - 1).catch(() => {});
  else return;
  e.preventDefault();
});

async function start() {
  topbar();
  if (!queuePath) {
    const { queues } = await api('/api/queues');
    $('main').innerHTML = `<div class="empty"><h1>Pick a review queue</h1>
      <p>Each queue is a set of clips to listen to. Your verdicts are saved as <b>${esc(reviewer)}</b>.</p>
      <ul class="queues">${queues
        .map((q) => {
          const [set, , file] = q.split('/');
          return `<li><a href="?queue=${encodeURIComponent(q)}&reviewer=${encodeURIComponent(reviewer)}"><b>${esc(file?.replace(/\.json$/, '') || q)}</b><span>${esc(set)}</span>${icon('right')}</a></li>`;
        })
        .join('') || '<li class="muted" style="padding:16px 0">No queues yet.</li>'}</ul>
      <p class="fine">Or open <code>?queue=&lt;path under /lab/data&gt;</code>.</p></div>`;
    return;
  }
  const d = await api(`/api/queue?path=${encodeURIComponent(queuePath)}&reviewer=${encodeURIComponent(reviewer)}`);
  Object.assign(st, { queue: d.queue, reviews: d.reviews, tally: d.tally });
  $('nav').hidden = false;
  $('prev').onclick = () => show(st.i - 1);
  $('next').onclick = () => show(st.i + 1);
  $('todo').onclick = nextTodo;
  $('intro').onclick = (e) => {
    const b = e.target.closest('[data-go]');
    if (b) show(Number(b.dataset.go));
  };
  const at = st.queue.items.findIndex((it) => it.id === url.searchParams.get('item'));
  const todo = st.queue.items.findIndex((it) => !st.reviews[it.id]);
  show(at >= 0 ? at : Math.max(0, todo), false);
}

start().catch((e) => {
  $('main').innerHTML = `<div class="notice err">${esc(e.message)}</div>`;
});
