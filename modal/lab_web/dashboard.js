import { mountItem, VERDICTS, FAULTS } from './item.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const url = new URL(location.href);
const run = url.searchParams.get('run');
const ledger = url.searchParams.get('ledger');
const reviewer = url.searchParams.get('reviewer') || 'me';
const NAMES = { listener: 'Listener', tracker: 'Tracker', checker: 'Checker' };
const CASE_VERDICTS = [...VERDICTS.filter((v) => v.key !== 'ok'), ...FAULTS];
const st = { dash: null, lists: {}, view: null, nav: null };

const qs = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v != null && v !== '')).toString();
const signed = (n) => (n == null ? '–' : n === 0 ? '0' : `${n > 0 ? '+' : '−'}${Math.abs(n)}`);
const wc = (c) => (c ? `${c.wrongCalls} <span class="note">(${c.missed} + ${c.falseWarnings})</span>` : '–');
const bold = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
const day = (s) => (s ? new Date(s).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '');
const cmd = (c) => `<span class="cmd"><code>${esc(c)}</code><button data-copy="${esc(c)}">Copy</button></span>`;
const verdictPill = (v) => (v ? `<span class="pill ${v.startsWith('better') ? 'better' : v.startsWith('worse') ? 'worse' : ''}">${esc(v)}</span>` : '–');

async function api(path, init) {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${r.status}`);
  return body;
}

document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-copy]');
  if (!b) return;
  navigator.clipboard?.writeText(b.dataset.copy).then(
    () => {
      b.textContent = 'Copied';
      setTimeout(() => (b.textContent = 'Copy'), 1200);
    },
    () => prompt('Copy this command', b.dataset.copy),
  );
});

function header() {
  const h = st.dash.header;
  const s = h.score;
  const base = (h.testSet || '').replace(/\.v\d+$/, '');
  const where = location.hash.startsWith('#/release') ? 'release' : 'score';
  const score = s
    ? `<div class="score"><span class="big">${s.per5Min == null ? '–' : s.per5Min.toFixed(1)}</span>
        <div><div class="unit">wrong calls per 5 min · lower is better</div>
        <div class="counts">${s.missed} missed + ${s.falseWarnings} false warnings</div>
        <div class="meta">${s.wrongCalls} wrong calls in ${s.minutes} min of the ${s.split === 'exam' ? 'exam set' : `${s.split} sessions`}, ${esc(s.audio)} audio ·
        ${st.dash.asked ? `run ${esc(st.dash.run)}` : `main line ${esc(s.sha)}`}</div></div></div>`
    : `<div class="score"><span class="big">–</span><div class="unit">no baseline yet</div></div>`;
  const check = h.check ? `<span class="pill ${h.check.state === 'pass' ? 'pass' : h.check.state === 'fail' ? 'fail' : ''}">${esc(h.check.text)}</span>` : '<span class="pill">not checked by ear yet</span>';
  const pending = h.pending
    ? `<span><span class="pill wait">${h.pending} flag${h.pending === 1 ? '' : 's'} pending</span> apply: ${cmd(h.applyFlags)}</span>`
    : '<span class="note">0 flags pending</span>';
  $('head').innerHTML = `${score}
    <div class="facts"><span>${h.testSet ? `Test set <b>${esc(h.testSet)}</b>${h.flagsApplied ? ` · ${h.flagsApplied} flagged windows left out` : ''} · ${check}` : 'No test set in the ledger yet'}</span>${pending}</div>
    <nav class="tabs"><a href="#/" class="${where === 'score' ? 'on' : ''}">Score</a><a href="#/release" class="${where === 'release' ? 'on' : ''}">Release</a>
    <a href="/review?${qs({ queue: h.check ? `${base}/review/v1-check.json` : null, reviewer })}">Listening check</a></nav>`;
}

function notices(d) {
  const out = [];
  if (d.asked) out.push(`Showing run <b>${esc(d.run)}</b>, not the main line. <a href="?${qs({ ledger, reviewer: url.searchParams.get('reviewer') })}">Back to the main line</a>`);
  else if (!d.header.inLedger) {
    out.push(`No baseline yet: the ledger has no main-line score. Start one with ${cmd(d.header.baseline)}` +
      (d.runs.length ? `<br>Meanwhile, look at a run: ${d.runs.map((r) => `<a href="?${qs({ run: r, ledger })}">${esc(r)}</a>`).join(' · ')}` : ''));
  } else if (!d.overview) out.push(`The main line's run ${esc(d.run)} has no score.json on the volume.`);
  for (const w of d.flagWarnings || []) out.push(`Flag not usable: ${esc(w)}`);
  return out.map((m) => `<div class="msg">${m}</div>`).join('');
}

function experiments(rows) {
  if (!rows.length)
    return `<p class="note">No experiments yet. Start one: ${cmd("python3 tools/lab.py new <id> --component checker --hypothesis '...' --expect-component -5 --expect-overall -3")}</p>`;
  const pair = (x, k) => (x ? `${signed(x.overall)} overall<br><span class="note">${signed(x.component)} ${esc(k)}</span>` : '–');
  return `<table><tr><th>Experiment</th><th>Part</th><th class="num">Expected</th><th class="num">Got (practice)</th><th class="num">Exam after fold</th><th>Verdict</th><th>State</th></tr>
    ${rows
      .map((r) => {
        const v = r.rebased || r.practice;
        return `<tr class="${r.state.startsWith('waiting') ? 'waiting' : ''}">
          <td><b>${esc(r.id)}</b><div class="hyp">${esc(r.hypothesis)}</div></td>
          <td>${esc(NAMES[r.component])}</td>
          <td class="num">${pair(r.expected, r.component)}</td>
          <td class="num">${pair(r.practice, r.component)}${r.rebased ? `<div class="note">after rebase: ${signed(r.rebased.overall)}</div>` : ''}</td>
          <td class="num">${r.exam ? `${pair(r.exam, r.component)}<div class="note">${r.exam.main.wrongCalls} → ${r.exam.this.wrongCalls}</div>` : '–'}</td>
          <td>${verdictPill(v?.verdict)}</td>
          <td><span class="pill ${r.state.startsWith('waiting') ? 'wait' : ''}">${esc(r.stateText)}</span>
            ${r.moved ? '<div class="note">main line moved since; fold re-checks</div>' : ''}
            ${r.note ? `<div class="note">${esc(r.note)}</div>` : ''}${r.next.map(cmd).join('')}</td></tr>`;
      })
      .join('')}</table>
    <p class="note">Counts of wrong calls; fewer is better. A change of 2 or fewer is noise. Fold or discard from the command line.</p>`;
}

function history(rows) {
  if (!rows.length) return '<p class="note">No main-line scores yet.</p>';
  const what = (r) =>
    r.kind === 'fold' ? `Folded in <b>${esc(r.experiment)}</b>: ${esc(r.note)}` : r.kind === 'new test set' ? `New test set ${esc(r.testSet)}: ${esc(r.note)}` : esc(r.note || r.kind);
  return `<table><tr><th>When</th><th>What changed</th><th class="num">Exam wrong calls</th><th class="num">Change</th><th class="num">Practice</th><th>Commit</th></tr>
    ${rows
      .map((r) => `<tr><td>${day(r.at)}</td><td>${what(r)}</td><td class="num">${wc(r.exam)}</td>
        <td class="num">${r.change == null ? '–' : `${signed(r.change)} ${verdictPill(r.verdict)}`}</td><td class="num">${wc(r.practice)}</td><td><code>${esc(r.sha)}</code></td></tr>`)
      .join('')}</table>`;
}

function scoreView() {
  const d = st.dash;
  let html = notices(d);
  const o = d.overview;
  if (o) {
    const where = o.split === 'exam' ? 'exam set' : `${o.split} sessions`;
    html += `<h2>Where the ${o.bar.total} wrong calls come from <span class="note">(${where}, phone audio)</span></h2>
      <div class="stack">${o.bar.parts.map((p) => `<div style="flex:${p.n} 0 0;background:var(--${p.key})" title="${NAMES[p.key]}: ${p.n}">${p.share > 0.06 ? `${NAMES[p.key]} ${p.n}` : ''}</div>`).join('')}</div>
      <div class="legend">${o.bar.parts.map((p) => `<span><span class="dot" style="background:var(--${p.key})"></span>${NAMES[p.key]} ${p.n}</span>`).join('')}
        <span>= ${o.bar.total}${o.bar.adds ? '' : ' (does not add up: scorer bug)'}</span></div>
      <div class="cards">${o.cards
        .map((c) => `<a class="card" href="#/cases/${c.key}" style="--c:var(--${c.key})">
          <span class="name">${c.name}</span><span class="job">${esc(c.job)}</span>
          <span class="n">${c.wrongCalls} <small>wrong calls are its fault</small></span>
          <span class="note">${c.missed} missed + ${c.falseWarnings} false warnings</span>
          <span class="own">${esc(c.own.text)}</span><span class="note">${esc(c.own.detail)}</span>
          <span class="note">${esc(c.explain)}</span>
          <span class="go">See its failing practice cases →</span></a>`)
        .join('')}</div>
      ${o.teacher ? `<div class="ref"><b>Teacher (reference, not a target):</b> the same engine on the teacher model's hearing makes ${o.teacher.wrongCalls} wrong calls (${o.teacher.missed} missed + ${o.teacher.falseWarnings} false warnings, ${o.teacher.per5Min} per 5 min)${o.teacher.mishears != null ? ` and mishears ${o.teacher.mishears} of every 1,000 sounds` : ''}.</div>` : ''}`;
  }
  if (!d.asked) html += `<h2>Experiments</h2>${experiments(d.experiments)}<h2>Main line history</h2>${history(d.history)}`;
  $('main').innerHTML = html;
}

async function listFor(component, variant) {
  const k = `${component}/${variant}`;
  if (!st.lists[k]) st.lists[k] = await api(`/api/cases?${qs({ component, variant, run, ledger })}`);
  return st.lists[k];
}

const caseHref = (c, v, key) => `#/case/${c}/${v}/${encodeURIComponent(key)}`;

async function casesView(component, variant) {
  const d = await listFor(component, variant);
  const c = d.card;
  $('main').innerHTML = `<p><a href="#/">← Score</a></p>
    <h2>${NAMES[component]}: failing practice cases
      <span class="seg">${['phone', 'studio']
        .map((v) => `<button data-variant="${v}" aria-pressed="${v === variant}">${v} audio${v === variant ? ` (${d.cases.length})` : ` (${d.other.n})`}</button>`)
        .join('')}</span></h2>
    <p class="note">${c && variant === 'phone' ? `${c.wrongCalls} wrong calls on the practice set are charged to the ${component} (${c.missed} missed + ${c.falseWarnings} false warnings). ` : ''}
      Likely bad data first. Open one to listen, then mark it bad data or a fault. Exam cases are never shown.</p>
    <ul class="cases">${
      d.cases
        .map((r) => `<li><a href="${caseHref(component, variant, r.key)}"><span class="t">${esc(r.title)}</span>
          <span class="w">${esc(r.where)} · ${esc(r.session)} at ${r.timeS.toFixed(1)} s</span>
          ${r.suspects.map((s) => `<span class="chip" title="${esc(s.text)}">${esc({ dataCheck: 'failed a data check', allFail: 'everyone fails it', everywhere: 'fails in every experiment' }[s.key])}</span>`).join('')}
          ${r.verdict ? `<span class="chip ${r.verdict.startsWith('bad_') ? 'bad' : 'v'}">${esc(r.verdictText)}</span>` : ''}</a></li>`)
        .join('') || '<li class="note">No failing cases here.</li>'
    }</ul>`;
  for (const b of document.querySelectorAll('[data-variant]')) b.onclick = () => (location.hash = `#/cases/${component}/${b.dataset.variant}`);
}

async function caseView(component, variant, key) {
  const list = (await listFor(component, variant)).cases;
  const i = list.findIndex((r) => r.key === key);
  const d = await api(`/api/case?${qs({ key, run, reviewer, ledger })}`);
  const go = (j) => list[j] && (location.hash = caseHref(component, variant, list[j].key));
  st.nav = { prev: () => go(i - 1), next: () => go(i + 1), back: () => (location.hash = `#/cases/${component}/${variant}`) };
  $('main').innerHTML = `<div class="casebar"><a href="#/cases/${component}/${variant}">← ${NAMES[component]} cases</a>
      <span>${i >= 0 ? `${i + 1} / ${list.length}` : ''}</span>
      <span><button id="prev">← Back (k)</button> <button id="next">Next (j) →</button></span></div>
    <div class="story">${d.suspects.map((s) => `<span class="chip">${esc(s.text)}</span>`).join('')}${d.story.map((p) => `<p>${esc(p)}</p>`).join('')}</div>
    <div id="item"></div>
    <div class="keys">Keys: space play · b / a original before / after · 1–3 bad data · 4–6 fault (then next) · j next · k back · Esc list.
      Saved to the review store for ${esc(d.testSet)} as <b>${esc(reviewer)}</b>; bad-data verdicts become flags for <code>lab apply-flags</code>.</div>`;
  $('prev').onclick = st.nav.prev;
  $('next').onclick = st.nav.next;
  st.view = mountItem($('item'), d.item, {
    review: d.review,
    verdicts: CASE_VERDICTS,
    onVerdict: async (verdict, note, how) => {
      const r = await api('/api/case-verdict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key, verdict, note, reviewer, run, ledger }),
      });
      const row = list.find((x) => x.key === key);
      if (row) Object.assign(row, { verdict, verdictText: CASE_VERDICTS.find((v) => v.key === verdict)?.text });
      st.dash.header.pending = r.pending;
      header();
      if (how !== 'note') setTimeout(() => location.hash === caseHref(component, variant, key) && go(i + 1), 250);
      return r.review;
    },
  });
}

function releaseView() {
  return api('/api/release').then(({ release: r, source }) => {
    if (!r) return ($('main').innerHTML = `<p>No release gates found in <code>${esc(source)}</code>.</p>`);
    $('main').innerHTML = `<h2>${esc(r.title)}</h2><p>${bold(r.summary)}</p>
      <table><tr>${r.columns.map((c) => `<th>${esc(c)}</th>`).join('')}</tr>
      ${r.gates.map((g) => `<tr>${r.columns.map((c) => `<td>${bold(g[c])}</td>`).join('')}</tr>`).join('')}</table>
      ${r.notes.map((n) => `<p class="note">${bold(n)}</p>`).join('')}
      <p class="note">From the old scoreboard (<code>${esc(source)}</code>, as of this deploy). The lab score above replaces it as the front page; these gates still decide a release.</p>`;
  });
}

async function route() {
  st.view?.destroy();
  st.view = st.nav = null;
  const [, page, a, b, ...rest] = location.hash.split('/');
  header();
  try {
    if (page === 'release') await releaseView();
    else if (page === 'cases' && NAMES[a]) await casesView(a, b === 'studio' ? 'studio' : 'phone');
    else if (page === 'case' && NAMES[a]) await caseView(a, b, decodeURIComponent(rest.join('/')));
    else scoreView();
  } catch (e) {
    $('main').innerHTML = `<div class="msg err">${esc(e.message)}</div>`;
  }
  if (page !== 'case') window.scrollTo(0, 0);
}

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
  else if (k === 'j') st.nav.next();
  else if (k === 'k') st.nav.prev();
  else if (k === 'Escape') st.nav.back();
  else if (/^[1-9]$/.test(k)) st.view.choose(Number(k) - 1).catch(() => {});
  else return;
  e.preventDefault();
});

api(`/api/dashboard?${qs({ run, ledger })}`)
  .then((d) => {
    st.dash = d;
    addEventListener('hashchange', route);
    return route();
  })
  .catch((e) => {
    $('head').innerHTML = '';
    $('main').innerHTML = `<div class="msg err">${esc(e.message)}</div>`;
  });
