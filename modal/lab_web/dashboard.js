import { mountItem, VERDICTS, FAULTS, icon, keySheet, railHtml, verdictKind } from './item.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const url = new URL(location.href);
const run = url.searchParams.get('run');
const ledger = url.searchParams.get('ledger');
const reviewer = url.searchParams.get('reviewer') || 'me';
const NAMES = { listener: 'Listener', tracker: 'Tracker', checker: 'Checker' };
const CASE_VERDICTS = [...VERDICTS.filter((v) => v.key !== 'ok'), ...FAULTS];
const st = { dash: null, lists: {}, view: null, nav: null, watch: null };

const qs = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v != null && v !== '')).toString();
const bold = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
const day = (s) => (s ? new Date(s).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '');
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const fewer = (n) => (n == null ? '–' : n === 0 ? 'no change' : n < 0 ? `${-n} fewer` : `${n} more`);
const sentence = (s) => (s ? s[0].toUpperCase() + s.slice(1) : '');
const copyBtn = (c, label = '') =>
  `<button class="copy" data-copy="${esc(c)}" title="Copy this command">${icon('copy')}${label ? `<span>${esc(label)}</span>` : ''}<code>${esc(c)}</code></button>`;
const VERB = { fold: 'Fold in', discard: 'Discard', run: 'Run' };
const verbOf = (c) => VERB[c.split(' ')[2]] || '';

const chip = (kind, text) => `<span class="verdict ${kind}">${icon(kind === 'better' ? 'down' : kind === 'worse' ? 'up' : 'minus')}${esc(text)}</span>`;
const kindOf = (v) => (v?.startsWith('better') ? 'better' : v?.startsWith('worse') ? 'worse' : '');

function verdictChip(v) {
  if (!v) return '';
  return chip(kindOf(v), v === 'noise' ? 'No real change' : sentence(v.replace(/^noise overall, (\w+) (\w+)$/, '$1 $2, overall no real change')));
}

async function api(path, init) {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${r.status}`);
  return body;
}

document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-copy]');
  if (!b) return;
  const done = () => {
    b.classList.add('done');
    setTimeout(() => b.classList.remove('done'), 1200);
  };
  navigator.clipboard?.writeText(b.dataset.copy).then(done, () => prompt('Copy this command', b.dataset.copy));
});

const keys = keySheet([
  ['space', 'Play or pause the case'],
  ['b', 'Original recording before the edit'],
  ['a', 'Original recording after the edit'],
  ['1 2 3', 'Bad data: label, audio, splice'],
  ['4 5 6', 'Fault: listener, tracker, checker'],
  ['j', 'Next case'],
  ['k', 'Previous case'],
  ['Esc', 'Back to the list'],
  ['?', 'Show or hide this'],
]);

// Top bar: wordmark, tabs, and the score once the big one is out of view.

function topbar(where) {
  const h = st.dash.header;
  const s = h.score;
  const base = (h.testSet || '').replace(/\.v\d+$/, '');
  const tab = (href, text, on) => `<a href="${href}" ${on ? 'aria-current="page"' : ''}>${text}</a>`;
  $('head').innerHTML = `<a class="brand" href="#/">${icon('mark')}Tilawa lab</a>
    <nav class="tabs">${tab('#/', 'Score', where !== 'release')}${tab('#/release', 'Release', where === 'release')}
      ${tab(`/review?${qs({ queue: h.check ? h.check.queue || `${base}/review/v1-check.json` : null, reviewer: url.searchParams.get('reviewer') })}`, 'Listening check', false)}</nav>
    ${s ? `<a class="mini ${where === 'score' ? 'off' : ''}" id="mini" href="#/" title="The overall score"><b class="num">${s.per5Min.toFixed(1)}</b><span>wrong calls / 5 min</span></a>` : ''}`;
  st.watch?.disconnect();
  const big = document.querySelector('.score');
  if (where === 'score' && big && $('mini')) {
    st.watch = new IntersectionObserver(([e]) => $('mini')?.classList.toggle('off', e.isIntersecting), { rootMargin: '-56px 0px 0px 0px' });
    st.watch.observe(big);
  }
}

function notices(d) {
  const out = [];
  if (d.asked) out.push(['', `Showing run <b>${esc(d.run)}</b>, not the main line. <a href="?${qs({ ledger, reviewer: url.searchParams.get('reviewer') })}">Back to the main line</a>`]);
  else if (d.header.inLedger && !d.overview) out.push(['warn', `The main line's run ${esc(d.run)} has no score on the volume.`]);
  for (const w of d.flagWarnings || []) out.push(['warn', `A flag can't be used: ${esc(w)}`]);
  return out.map(([k, m]) => `<div class="notice ${k}">${m}</div>`).join('');
}

// Overview

const DEFS = [
  ['Wrong call', 'A planted mistake the engine didn’t warn about, or a warning where nothing was wrong. Both count the same.'],
  ['Per 5 minutes', 'Wrong calls divided by the minutes of recitation, times 5, so sessions of any length compare.'],
  ['Exam set', 'Sessions experiments never see; only their numbers are shown. The practice set is the part you can open and listen to.'],
  ['The three parts', 'The listener hears the sounds, the tracker finds the place in the Quran, the checker decides if it’s a mistake. Each wrong call is charged to one part by re-running the engine with perfect inputs: still wrong with perfect hearing and the true place is the checker’s; fixed by perfect hearing is the listener’s; wrong only with the real tracker is the tracker’s.'],
  ['Noise', 'A change of 2 or fewer wrong calls. Treat it as no change.'],
  ['Big reference listener', 'A large listening model used to check the test data and as a reference. The app ships a smaller one.'],
];

function hero(d) {
  const h = d.header;
  const s = h.score;
  const where = s.split === 'exam' ? 'the exam set' : s.split === 'all' ? 'all sessions' : `the ${s.split} set`;
  const last = !d.asked ? d.history[0] : null;
  const ch = last && last.change != null ? last.change : null;
  const kind = ch == null ? '' : Math.abs(ch) <= 2 ? '' : ch < 0 ? 'better' : 'worse';
  const delta = ch == null ? '' : `<p class="delta ${kind}">${icon(kind === 'better' ? 'down' : kind === 'worse' ? 'up' : 'minus')}
      ${kind ? `${Math.abs(ch)} ${kind === 'better' ? 'fewer' : 'more'} wrong calls` : 'No real change'} <span>since the last main-line score${kind ? '' : ` (${ch > 0 ? '+' : ch < 0 ? '−' : ''}${Math.abs(ch)}, within noise)`}</span></p>`;
  const check = h.check
    ? `<span title="${esc(h.check.text)}"><i class="dot ${h.check.state === 'pass' ? 'pass' : h.check.state === 'fail' ? 'fail' : ''}"></i>${esc(sentence(h.check.text))}</span>`
    : '<span><i class="dot"></i>Not checked by ear yet</span>';
  const flags = h.pending
    ? `<span><i class="dot wait"></i>${plural(h.pending, 'flag')} waiting to be applied ${copyBtn(h.applyFlags)}</span>`
    : '<span>No flags pending</span>';
  return `<div class="hero">
    <div class="score"><span class="score-n">${s.per5Min == null ? '–' : s.per5Min.toFixed(1)}</span>
      <p class="score-u"><strong>wrong calls every 5 minutes</strong><span>Lower is better</span></p></div>
    <p class="parts"><b class="num">${s.missed}</b> <dfn title="A planted mistake the engine didn’t warn about">missed mistakes</dfn> +
      <b class="num">${s.falseWarnings}</b> <dfn title="A warning where nothing was wrong">false warnings</dfn> =
      <b class="num">${s.wrongCalls}</b> wrong calls in ${s.minutes} minutes of recitation on ${where}.</p>
    ${delta}
    <p class="facts"><span>Test set ${esc(h.testSet || '–')}${h.flagsApplied ? `, ${plural(h.flagsApplied, 'flagged window')} left out` : ''}</span><span>Phone audio</span>
      <span>${d.asked ? `Run ${esc(d.run)}` : `Main line <code>${esc(s.sha)}</code>`}</span>${check}${flags}</p>
    <details class="more"><summary>${icon('right')}How this is measured</summary>
      <dl class="defs">${DEFS.map(([t, x]) => `<dt>${t}</dt><dd>${x}</dd>`).join('')}
        <dt>Raw numbers</dt><dd class="num">${s.mistakes} planted mistakes, ${s.caught} caught${s.misclassified ? ` (${s.misclassified} named as the wrong kind)` : ''};
          ${plural(s.sessions, 'session')}, ${s.minutes} minutes of recitation; listener ${esc(s.listener || '–')}.</dd></dl></details>
  </div>`;
}

function components(o) {
  const lever = new Set(o.levers || []);
  const tied = lever.size > 1;
  const leverTip = tied ? 'These are within 2 wrong calls of each other, so improving either moves the overall score about as much' : 'The most wrong calls are charged here, so improving it moves the overall score most';
  const cards = Object.fromEntries(o.cards.map((c) => [c.key, c]));
  return `<section aria-labelledby="split-h">
    <div class="sec-h"><h2 id="split-h">Where the ${o.bar.total} wrong calls come from</h2></div>
    <p class="sec-p">Each wrong call is charged to the one part of the engine that caused it, so the three add up. Open a part to listen to its failing cases.</p>
    <div class="split"><div class="bar" role="img" aria-label="${o.bar.parts.map((p) => `${NAMES[p.key]} ${p.n}`).join(', ')}, total ${o.bar.total}">${o.bar.parts
      .map((p) => `<div style="flex:${p.n} 0 0;background:var(--${p.key})" title="${NAMES[p.key]}: ${p.n}"></div>`)
      .join('')}</div></div>
    <p class="split-sum num">${o.bar.parts.map((p) => `<span><i class="sw" style="--c:var(--${p.key})"></i>${NAMES[p.key]} <b>${p.n}</b></span>`).join('<span class="op">+</span>')}<span class="op">=</span><b>${o.bar.total}</b></p>
    ${o.bar.adds ? '' : '<div class="notice err">The three parts don’t add up to the total. That’s a scorer bug.</div>'}
    <div class="parts-list">${o.bar.parts
      .map(({ key }) => {
        const c = cards[key];
        return `<a class="part" href="#/cases/${key}" style="--c:var(--${key})">
          <span class="part-name"><b><i class="sw"></i>${c.name}${lever.has(key) ? `<span class="lever" title="${leverTip}">${tied ? 'Tied for biggest lever' : 'Biggest lever'}</span>` : ''}</b><span>${esc(c.job)}</span></span>
          <span class="part-n"><b>${c.wrongCalls}</b><span>${c.missed} missed · ${c.falseWarnings} false</span></span>
          <span class="part-own"><p>${esc(c.own.text)}</p><p>${esc(c.own.detail)}</p></span>
          ${icon('right')}</a>`;
      })
      .join('')}</div>
    ${o.teacher ? `<p class="ref">For reference, not a target: with the big reference listener (too large to ship in the app) the same engine makes ${o.teacher.wrongCalls} wrong calls (${o.teacher.missed} missed + ${o.teacher.falseWarnings} false warnings)${o.teacher.mishears != null ? ` and mishears ${o.teacher.mishears} of every 1,000 sounds` : ''}.</p>` : ''}
  </section>`;
}

// Expected vs got on one axis: fewer wrong calls to the left, the noise band shaded around zero.
function axis(expected, got) {
  const m = Math.max(5, Math.abs(expected ?? 0), Math.abs(got ?? 0)) + 1;
  const x = (v) => 50 + (v / m) * 50;
  const kind = got == null ? '' : Math.abs(got) <= 2 ? 'noise' : got < 0 ? 'better' : 'worse';
  const fill = { better: 'var(--better)', worse: 'var(--worse)', noise: 'var(--g-500)' }[kind];
  return `<div class="axis"><svg viewBox="0 0 100 22" preserveAspectRatio="none" role="img" aria-label="Expected ${fewer(expected)}, got ${fewer(got)}">
    <rect x="0" y="10.5" width="100" height="1" fill="var(--g-300)"/>
    <rect x="${x(-2)}" y="2" width="${x(2) - x(-2)}" height="18" fill="var(--g-200)"/>
    ${got == null ? '' : `<rect x="${Math.min(x(0), x(got))}" y="6" width="${Math.max(0.8, Math.abs(x(got) - x(0)))}" height="10" fill="${fill}"/>`}
    <line x1="50" x2="50" y1="0" y2="22" stroke="var(--g-500)" stroke-width="1" vector-effect="non-scaling-stroke"/>
    ${expected == null ? '' : `<line x1="${x(expected)}" x2="${x(expected)}" y1="0" y2="22" stroke="var(--g-900)" stroke-width="2.5" vector-effect="non-scaling-stroke"><title>Expected ${fewer(expected)}</title></line>`}
  </svg><div class="axis-l"><span>← fewer wrong calls</span><span>more →</span></div></div>`;
}

function expRow(r) {
  const v = r.rebased || r.practice;
  const comp = NAMES[r.component];
  const superseded = r.state === 'superseded';
  const lines = v
    ? `<p class="res-line"><span>Overall</span><span><b class="num">${fewer(v.overall)}</b> · expected ${fewer(r.expected.overall)}</span></p>
       <p class="res-line"><span>${comp}</span><span><b class="num">${fewer(v.component)}</b> · expected ${fewer(r.expected.component)}</span></p>`
    : `<p class="res-line"><span>Expected</span><span>${fewer(r.expected.overall)} overall · ${fewer(r.expected.component)} ${comp.toLowerCase()}</span></p>`;
  const notes = [
    superseded && r.practice ? `On its branch it got ${fewer(r.practice.overall)}; after rebasing onto the main line the gain is gone.` : '',
    r.rebased && !superseded ? `After rebase onto the main line: ${fewer(r.rebased.overall)} overall.` : '',
    r.moved ? 'The main line moved since this was measured; folding re-checks it.' : '',
    r.note || '',
  ].filter(Boolean);
  return `<article class="exp">
    <div><h3 class="exp-id"><i class="sw" style="--c:var(--${r.component})"></i>${esc(r.id)} <small>${comp}</small>
        ${superseded ? '<span class="verdict">Gain gone after rebase</span>' : verdictChip(v?.verdict)}</h3>
      <p class="exp-hyp">${esc(r.hypothesis)}</p>
      ${notes.map((n) => `<p class="exp-note">${icon('info')}${esc(n)}</p>`).join('')}
      <div class="exp-cmds">${r.next.map((c) => copyBtn(c, `${verbOf(c)}:`)).join('')}</div></div>
    <div class="exp-res">${v ? axis(r.expected.overall, v.overall) : ''}${lines}</div>
  </article>`;
}

function pastRow(r) {
  const state = r.state === 'folded' ? 'Folded in' : 'Discarded';
  const res = r.exam
    ? `exam ${r.exam.main.wrongCalls} → ${r.exam.this.wrongCalls} (${fewer(r.exam.overall)})`
    : r.practice ? `practice ${fewer(r.practice.overall)}` : 'not scored';
  return `<article class="exp compact">
    <div><h3 class="exp-id"><i class="sw" style="--c:var(--${r.component})"></i>${esc(r.id)} <small>${state} · ${esc(r.hypothesis)}</small></h3>
      ${r.note ? `<p class="exp-hyp small">${esc(r.note)}</p>` : ''}</div>
    <p class="res-line num"><span>${res}</span></p></article>`;
}

function experiments(rows) {
  const head = `<div class="sec-h"><h2 id="exp-h">Experiments</h2></div>
    <p class="sec-p">Each experiment changes one part and predicts how many wrong calls it saves, counted on the practice set. You fold it into the main line or discard it, from the command line.</p>
    <p class="axis-key"><span><i class="k-bar"></i>what it got</span><span><i class="k-line"></i>what it expected</span><span><i class="k-noise"></i>noise: a change of 2 or fewer</span></p>`;
  if (!rows.length)
    return `<section aria-labelledby="exp-h">${head}<p class="group-h">No experiments yet</p>
      <div class="exp-cmds">${copyBtn("python3 tools/lab.py new <id> --component checker --hypothesis '...' --expect-component -5 --expect-overall -3", 'Start one:')}</div></section>`;
  const need = rows.filter((r) => r.state.startsWith('waiting') || r.state === 'superseded');
  const open = rows.filter((r) => r.state === 'open');
  const past = rows.filter((r) => r.state === 'folded' || r.state === 'discarded');
  return `<section aria-labelledby="exp-h">${head}
    ${need.length ? `<div class="decide"><h3 class="group-h">Needs your decision <span>${need.length}</span></h3>${need.map(expRow).join('')}</div>` : '<p class="group-h">Nothing needs your decision</p>'}
    ${open.length ? `<h3 class="group-h">Running or not scored yet <span>${open.length}</span></h3>${open.map(expRow).join('')}` : ''}
    ${past.length ? `<details class="more"><summary>${icon('right')}Past experiments (${past.length})</summary>${past.map(pastRow).join('')}</details>` : ''}
  </section>`;
}

function history(rows) {
  if (!rows.length) return '';
  const what = (r) =>
    r.kind === 'fold' ? `Folded in <b>${esc(r.experiment)}</b><div class="muted">${esc(r.note)}</div>`
      : r.kind === 'new test set' ? `New test set ${esc(r.testSet)}${r.note ? `<div class="muted">${esc(r.note)}</div>` : ''}`
      : r.kind === 'first score' ? `First score${r.note ? `: ${esc(r.note)}` : ''}` : esc(r.note || r.kind);
  const row = (r) => `<tr><td class="muted">${day(r.at)}</td><td>${what(r)}</td>
    <td class="r">${r.exam ? `${r.exam.wrongCalls} <span class="muted">(${r.exam.missed} + ${r.exam.falseWarnings})</span>` : '–'}</td>
    <td class="r">${r.change == null ? '<span class="muted">–</span>' : chip(kindOf(r.verdict), r.verdict === 'noise' ? `${fewer(r.change)}, noise` : fewer(r.change))}</td>
    <td><code class="muted">${esc(r.sha)}</code></td></tr>`;
  const table = (rs) => `<div class="tbl-wrap"><table class="tbl"><tr><th>When</th><th>What changed</th><th class="r">Exam wrong calls</th><th class="r">Change</th><th>Commit</th></tr>${rs.map(row).join('')}</table></div>`;
  return `<section aria-labelledby="hist-h"><div class="sec-h"><h2 id="hist-h">Main line history</h2></div>
    <p class="sec-p">Every score of the main line, newest first. Changes compare exam wrong calls on the same test set.</p>
    ${table(rows.slice(0, 5))}${rows.length > 5 ? `<details class="more"><summary>${icon('right')}Older (${rows.length - 5})</summary>${table(rows.slice(5))}</details>` : ''}</section>`;
}

function noScore(d) {
  const parse = (r) => {
    const m = r.match(/^(.*?)-([0-9a-f]{7}(?:-dirty)?)-(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)\d\dZ$/);
    return m ? { kind: m[1], sha: m[2], at: day(`${m[3]}-${m[4]}-${m[5]}T${m[6]}:${m[7]}:00Z`) } : { kind: r, sha: '', at: '' };
  };
  return `<div class="empty"><h1>No score yet</h1>
    <p>The lab hasn’t scored the main line on a test set. Run the baseline once and this page fills in by itself.</p>
    <div class="cmd-box">${copyBtn(d.header.baseline)}</div>
    ${d.runs.length ? `<p class="group-h">Meanwhile, open a recent run</p><nav class="runs">${d.runs
      .map((r) => {
        const p = parse(r);
        return `<a href="?${qs({ run: r, ledger })}"><span>${esc(p.kind)}</span><code class="muted">${esc(p.sha)}</code><span class="muted">${esc(p.at)}</span>${icon('right')}</a>`;
      })
      .join('')}</nav>` : ''}</div>`;
}

function scoreView() {
  const d = st.dash;
  let html = notices(d);
  html += d.header.score && d.overview ? hero(d) + components(d.overview) : d.header.score ? hero(d) : noScore(d);
  if (!d.asked) html += experiments(d.experiments) + history(d.history);
  $('main').innerHTML = html;
}

// Drill-down

const GROUPS = {
  dataCheck: ['Failed an automatic data check', 'The test data itself looks suspect. Start here.'],
  allFail: ['Wrong whatever the hearing', 'The big reference listener, the app’s listener and perfect hearing all get it wrong: often a sign of bad data.'],
  everywhere: ['Fails in every experiment', 'No experiment fixes these.'],
  other: ['Other cases', 'Probably real faults. Listen to confirm.'],
};

async function listFor(component, variant) {
  const k = `${component}/${variant}`;
  if (!st.lists[k]) {
    const d = await api(`/api/cases?${qs({ component, variant, run, ledger })}`);
    d.order = Object.keys(GROUPS).flatMap((g) => d.cases.filter((r) => (r.group || 'other') === g));
    st.lists[k] = d;
  }
  return st.lists[k];
}

const caseHref = (c, v, key) => `#/case/${c}/${v}/${encodeURIComponent(key)}`;
const statusChip = (r) => (r.verdict ? `<span class="st ${verdictKind(r.verdict)}">${esc(r.verdictText || r.verdict)}</span>` : '<span class="st todo">Not reviewed</span>');

async function casesView(component, variant) {
  const d = await listFor(component, variant);
  const c = d.card;
  const done = d.cases.filter((r) => r.verdict).length;
  const name = NAMES[component];
  $('main').innerHTML = `<a class="crumb" href="#/">${icon('back')}Overview</a>
    <div class="page-h"><div><h1><i class="sw" style="--c:var(--${component})"></i>${name}: ${plural(d.cases.length, 'failing case')}</h1>
      <p class="lede">${c && variant === 'phone' ? `On the practice set, ${c.wrongCalls} wrong calls are charged to the ${component} (${c.missed} missed + ${c.falseWarnings} false warnings). ` : ''}Listen to each one and say whether the test data is bad or the ${component} really got it wrong.</p></div>
      <div class="seg" role="group" aria-label="Audio">${['phone', 'studio']
        .map((v) => `<button data-variant="${v}" aria-pressed="${v === variant}">${v === 'phone' ? 'Phone' : 'Studio'} audio · ${v === variant ? d.cases.length : d.other.n}</button>`)
        .join('')}</div></div>
    <p class="progress"><span class="track"><i style="width:${d.cases.length ? (100 * done) / d.cases.length : 0}%"></i></span>${done} of ${d.cases.length} reviewed · exam cases stay hidden</p>
    ${Object.entries(GROUPS)
      .map(([g, [title, sub]]) => {
        const rows = d.order.filter((r) => (r.group || 'other') === g);
        if (!rows.length) return '';
        return `<div class="cgroup"><h2>${title} <span>${rows.length}</span></h2><p>${sub}</p><ul class="cases">${rows
          .map((r) => `<li><a href="${caseHref(component, variant, r.key)}"><span class="t">${esc(r.title)}</span>
            <span class="w">${esc(r.where)} · ${esc(r.session)} at ${r.timeS.toFixed(1)} s</span>${statusChip(r)}${icon('right')}</a></li>`)
          .join('')}</ul></div>`;
      })
      .join('') || '<p class="group-h">No failing cases here.</p>'}`;
  for (const b of document.querySelectorAll('[data-variant]')) b.onclick = () => (location.hash = `#/cases/${component}/${b.dataset.variant}`);
}

function whyHtml(d) {
  const e = d.engine;
  if (!e) return '';
  const mark = (a) => (a.right == null ? '' : a.right ? icon('check', 'yes') : icon('x', 'no'));
  return `<details class="why"><summary><b>Why it’s here</b><span>Charged to the ${esc(e.charged)}: ${esc(e.why)}.</span>${icon('right')}</summary>
    <div class="why-body"><p>${esc(e.what)}</p>
      <dl class="arms">${e.arms.map((a) => `<dt>${esc(a.who)}</dt><dd>${mark(a)}${esc(a.did)}</dd>`).join('')}</dl>
      ${e.onPlace != null ? `<p>The tracker was on the right place ${e.onPlace}% of the time around here.</p>` : ''}</div></details>`;
}

async function caseView(component, variant, key) {
  const list = (await listFor(component, variant)).order;
  const i = list.findIndex((r) => r.key === key);
  const d = await api(`/api/case?${qs({ key, run, reviewer, ledger })}`);
  const go = (j) => list[j] && (location.hash = caseHref(component, variant, list[j].key));
  st.nav = { prev: () => go(i - 1), next: () => go(i + 1), back: () => (location.hash = `#/cases/${component}/${variant}`) };
  const rail = () => railHtml(list, i, (r) => r.verdict, (j, v) => `${j + 1}. ${list[j].title}${v ? ` · ${CASE_VERDICTS.find((x) => x.key === v)?.text || v}` : ''}`);
  $('main').innerHTML = `<div class="narrow">
    <div class="casebar"><a class="crumb" style="margin:0" href="#/cases/${component}/${variant}">${icon('back')}${NAMES[component]} cases</a>
      <span class="pager"><span class="num">${i >= 0 ? `${i + 1} of ${list.length}` : ''}</span>
        <button class="btn icon-only" id="prev" title="Previous (k)" aria-label="Previous case" ${i <= 0 ? 'disabled' : ''}>${icon('left')}</button>
        <button class="btn icon-only" id="next" title="Next (j)" aria-label="Next case" ${i < 0 || i >= list.length - 1 ? 'disabled' : ''}>${icon('right')}</button></span></div>
    <div id="rail">${rail()}</div>
    ${d.suspects.length ? `<div class="suspect">${icon('flag')}<p><b>Likely bad data.</b> ${d.suspects.map((s) => (s.raw ? `<span title="${esc(s.raw)}">${esc(s.text)}</span>` : esc(s.text))).join('. ')}.</p></div>` : ''}
    <div id="item"></div>
    <p class="li-context" style="margin-top:var(--s6)">Saved for ${esc(d.testSet)} as <b>${esc(reviewer)}</b>. Bad-data verdicts become flags for <code>lab apply-flags</code>.</p></div>`;
  $('prev').onclick = st.nav.prev;
  $('next').onclick = st.nav.next;
  $('rail').onclick = (e) => {
    const b = e.target.closest('[data-go]');
    if (b) go(Number(b.dataset.go));
  };
  st.view = mountItem($('item'), d.item, {
    review: d.review,
    verdicts: CASE_VERDICTS,
    aside: whyHtml(d),
    onVerdict: async (verdict, note, how) => {
      const r = await api('/api/case-verdict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key, verdict, note, reviewer, run, ledger }),
      });
      const row = list.find((x) => x.key === key);
      if (row) Object.assign(row, { verdict, verdictText: CASE_VERDICTS.find((v) => v.key === verdict)?.text });
      st.dash.header.pending = r.pending;
      $('rail').innerHTML = rail();
      if (how !== 'note') setTimeout(() => location.hash === caseHref(component, variant, key) && go(i + 1), 250);
      return r.review;
    },
  });
}

// Release

const COLS = { Bar: 'Needs', Shortfall: 'Gap', 'Δ previous run': 'Change since last run' };
const GATES = { 'Listener gold PER': 'Listener: sounds misheard (checked clips)', 'Real-time factor': 'Time to process, as a share of the audio length' };
const first = (v) => {
  const m = String(v || '').match(/(-?\d+(?:\.\d+)?)(%?)/);
  return m && { n: +m[1], pct: !!m[2] };
};

// "46% short" becomes the gap in the gate's own units, worked out from Now and Needs.
function gap(g, cols) {
  const v = g[cols[3]] || '';
  if (!/short/i.test(v)) return v;
  const now = first(g[cols[1]]);
  const bar = first(g[cols[2]]);
  if (!now || !bar) return 'short';
  const le = /≤|<=|</.test(g[cols[2]]);
  const d = Math.abs(now.n - bar.n);
  const n = d >= 10 ? d.toFixed(0) : d >= 1 ? d.toFixed(1) : d.toFixed(2);
  return `${n}${bar.pct ? ' points' : ''} ${le ? 'too high' : 'too low'}`;
}

function releaseView() {
  return api('/api/release').then(({ release: r, source }) => {
    if (!r) return ($('main').innerHTML = `<div class="empty"><h1>No release gates</h1><p>None found in <code>${esc(source)}</code>.</p></div>`);
    const cell = (col, g) => {
      const v = g[col];
      if (col === r.columns[0]) return GATES[v] ? `<span title="${esc(v)}">${esc(GATES[v])}</span>` : bold(v);
      if (!/shortfall/i.test(col)) return bold(v);
      const k = /^pass/i.test(v) ? 'better' : /short/i.test(v) ? 'worse' : 'muted';
      const shown = gap(g, r.columns);
      return `<span class="${k === 'muted' ? 'muted' : ''}" ${shown !== v ? `title="${esc(v)}"` : ''} style="${k !== 'muted' ? `color:var(--${k});font-weight:600` : ''}">${esc(shown)}</span>`;
    };
    $('main').innerHTML = `<div class="page-h"><div><h1>${esc(r.title)}</h1><p class="lede">${bold(r.summary)}</p></div></div>
      <div class="tbl-wrap"><table class="tbl"><tr>${r.columns.map((c) => `<th>${esc(COLS[c] || c)}</th>`).join('')}</tr>
      ${r.gates.map((g) => `<tr>${r.columns.map((c) => `<td>${cell(c, g)}</td>`).join('')}</tr>`).join('')}</table></div>
      ${r.notes.map((n) => `<p class="sec-p small">${bold(n)}</p>`).join('')}
      <p class="sec-p small">From the old scoreboard (<code>${esc(source)}</code>, as of this deploy). The lab score replaces it as the front page; these gates still decide a release.</p>`;
  });
}

async function route() {
  st.view?.destroy();
  st.view = st.nav = null;
  keys.close();
  const [, page, a, b, ...rest] = location.hash.split('/');
  const where = page === 'release' ? 'release' : page === 'cases' || page === 'case' ? 'cases' : 'score';
  try {
    if (page === 'release') await releaseView();
    else if (page === 'cases' && NAMES[a]) await casesView(a, b === 'studio' ? 'studio' : 'phone');
    else if (page === 'case' && NAMES[a]) await caseView(a, b, decodeURIComponent(rest.join('/')));
    else scoreView();
  } catch (e) {
    $('main').innerHTML = `<div class="notice err">${esc(e.message)}</div>`;
  }
  topbar(where);
  if (page !== 'case') window.scrollTo(0, 0);
}

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
    $('head').innerHTML = `<a class="brand" href="#/">${icon('mark')}Tilawa lab</a>`;
    $('main').innerHTML = `<div class="notice err">${esc(e.message)}</div>`;
  });
