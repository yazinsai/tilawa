import { icon } from './item.js';
import { $, esc, NAMES, qs, day, plural, fewer, sentence, sw, span, ago, api, topbar, staleNotice, plainLines } from './al.js';

const url = new URL(location.href);
const root = url.searchParams.get('root');
const POLL_MS = 30_000;
const CLI = 'python3 tools/autolab.py';

const copyBtn = (c, label = '') =>
  `<button class="copy" data-copy="${esc(c)}" title="Copy">${icon('copy')}${label ? `<span>${esc(label)}</span>` : ''}<code>${esc(c)}</code></button>`;
const limit = (min) => (min == null ? '' : min < 90 ? `${min} min` : `${Math.round(min / 60)} h`);

document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-copy]');
  if (!b) return;
  const done = () => {
    b.classList.add('done');
    setTimeout(() => b.classList.remove('done'), 1200);
  };
  navigator.clipboard?.writeText(b.dataset.copy).then(done, () => prompt('Copy this', b.dataset.copy));
});

function notices(v) {
  const h = v.heartbeat;
  const out = [];
  if (h.paused) out.push(['warn', `<b>Paused:</b> ${esc(h.paused)}. Work already running finishes; nothing new starts until you resume it. ${copyBtn(`${CLI} resume`)}`]);
  if (v.spend?.atCap.length) out.push(['warn', `<b>Daily cap reached:</b> ${esc(v.spend.atCap.join(', '))}. Running work finishes; nothing new starts today.`]);
  return staleNotice(h) + out.map(([k, m]) => `<div class="notice ${k}">${m}</div>`).join('');
}

const liveLink = () => `<a class="al-live" href="/autolab/live${root ? `?${qs({ root })}` : ''}">${icon('right')}Progress, live: the branches, the folds and how close the line is</a>`;

// Hero: the autolab line's practice score, the one big number.

function hero(v) {
  const m = v.main;
  const o = v.trend.series.overall;
  const kind = !o || Math.abs(o.change) <= 2 ? '' : o.change < 0 ? 'better' : 'worse';
  const delta = o && o.points.length > 1
    ? `<p class="delta ${kind}">${icon(kind === 'better' ? 'down' : kind === 'worse' ? 'up' : 'minus')}${kind ? `${Math.abs(o.change)} ${kind === 'better' ? 'fewer' : 'more'} wrong calls` : 'No real change'}
        <span>since the first score on ${esc(v.trend.testSet)}</span></p>`
    : '';
  const p = v.portfolio;
  const facts = [
    `Test set ${esc(v.testSet || '–')}`,
    m?.sha ? `Autolab line <code>${esc(m.sha)}</code>` : '',
    p.lastFoldAt ? `Last fold ${ago(p.lastFoldAt)}` : 'No fold yet',
    `${plural(p.held, 'slot')} in use${p.slots ? ` of ${p.slots}` : ''}`,
  ].filter(Boolean);
  if (!m || m.wrongCalls == null)
    return `<div class="hero"><p class="lede">The autolab line hasn’t been scored yet.</p><p class="facts">${facts.map((f) => `<span>${f}</span>`).join('')}</p>${liveLink()}</div>`;
  return `<div class="hero">
    <div class="score"><span class="score-n">${m.wrongCalls}</span>
      <p class="score-u"><strong>wrong calls on the practice set</strong><span>The autolab line. Lower is better</span></p></div>
    <p class="parts">${m.missed != null ? `<b class="num">${m.missed}</b> missed mistakes + <b class="num">${m.falseWarnings}</b> false warnings. ` : ''}Charged to the
      ${['listener', 'tracker', 'checker'].map((k) => `${k} <b class="num">${m.parts[k] ?? '–'}</b>`).join(', ')}.</p>
    ${delta}
    <p class="facts">${facts.map((f) => `<span>${f}</span>`).join('')}</p>
    ${liveLink()}
  </div>`;
}

// Experiments

const about = (r) =>
  plainLines(r) +
  (r.eli5?.plain && r.hypothesis
    ? `<details class="more" data-k="hyp"><summary>${icon('right')}In technical terms</summary><p class="exp-hyp small">${esc(r.hypothesis)}</p></details>`
    : '');

const head = (r, extra = '') =>
  `<h3 class="exp-id">${sw(r.component)}${esc(r.id)} <small>${esc(NAMES[r.component] || r.component || '')}${r.tierText ? ` · ${esc(r.tierText)}` : ''}</small>${extra}</h3>`;

function needsYou(rows) {
  if (!rows.length) return '';
  return `<div class="decide"><h3 class="group-h">Waiting for you <span>${rows.length}</span></h3>${rows
    .map((h) => `<article class="exp" data-k="h-${esc(h.id)}"><div>${head(h)}${about(h)}
        <p class="exp-note">${icon('info')}${esc(sentence(h.what))}</p></div>
      <div class="exp-res"><p class="res-line"><span>Asked</span><span>${ago(h.at)}</span></p></div></article>`)
    .join('')}</div>`;
}

function running(rows) {
  const body = rows.length
    ? rows
        .map((r) => `<article class="exp" data-k="r-${esc(r.id)}"><div>${head(r)}${about(r)}
          ${r.attempts > 1 ? `<p class="exp-note">${icon('info')}Try ${r.attempts}</p>` : ''}</div>
        <div class="exp-res"><p class="al-state">${esc(r.stateText)}</p>
          <p class="res-line"><span>For</span><span class="${r.late ? 'al-late' : ''}"><b class="num">${span(r.ageSeconds)}</b>${r.limitMinutes ? ` · gives up after ${limit(r.limitMinutes)}` : ''}</span></p>
          ${r.executor ? `<p class="res-line"><span>Agent</span><span>${esc(r.executor)}</span></p>` : ''}</div></article>`)
        .join('')
    : '<p class="group-h">Nothing running</p>';
  return `<section aria-labelledby="run-h"><div class="sec-h"><h2 id="run-h">Running now</h2></div>
    <p class="sec-p">Each experiment changes one part of the engine and predicts how many wrong calls it saves on the practice set. An agent writes it; the supervisor checks, scores and folds it.</p>${body}</section>`;
}

function queue(rows) {
  if (!rows.length) return '';
  return `<section aria-labelledby="q-h"><div class="sec-h"><h2 id="q-h">Up next <span class="muted al-n">${rows.length}</span></h2></div>
    ${rows
      .map((r) => `<article class="exp compact" data-k="q-${esc(r.id)}"><div>${head(r)}${about(r)}</div>
        <p class="res-line"><span>${esc(r.stateText)} · ${span(r.ageSeconds)}</span></p></article>`)
      .join('')}</section>`;
}

const OUTCOME = { folded: ['better', 'Folded in'], discarded: ['worse', 'Discarded'], superseded: ['', 'Gain gone after catching up'], abandoned: ['', 'Given up'] };
const MARK = { passed: ['check', 'yes'], failed: ['x', 'no'], note: ['info', 'na'] };

function gateList(g) {
  if (!g) return '';
  const lead = g.verdict === 'pass' ? 'Passed every required check' : `Failed: ${g.failed.join(', ').toLowerCase()}`;
  return `<details class="more" data-k="gate"><summary>${icon('right')}${esc(lead)}${g.tuning ? ' (a number tweak, so it had to save more)' : ''}</summary>
    <ul class="al-checks">${g.checks
      .map((c) => `<li class="${c.result}" title="${esc(c.detail)}">${icon(...MARK[c.result])}<span><b>${esc(c.title)}</b>${c.blocking ? '' : ' <i>(for information)</i>'}<br>${esc(c.text)}</span></li>`)
      .join('')}</ul></details>`;
}

function verdictRow(r) {
  const [kind, text] = OUTCOME[r.state] || ['', r.stateText];
  const p = r.practice;
  const lines = p
    ? `<p class="res-line"><span>Practice set, ${esc(p.where)}</span></p>
       <p class="res-line"><span>Overall</span><span><b class="num">${fewer(p.overall)}</b> · expected ${fewer(p.expected?.overall)}</span></p>
       <p class="res-line"><span>${esc(NAMES[r.component] || r.component)}</span><span><b class="num">${fewer(p.component)}</b> · expected ${fewer(p.expected?.component)}</span></p>`
    : '<p class="res-line"><span>Not scored</span></p>';
  const trip = r.examTripwire ? `<p class="res-line"><span>Exam check</span><span>${r.examTripwire === 'pass' ? 'no worse' : 'got worse: folding paused'}</span></p>` : '';
  return `<article class="exp" data-k="v-${esc(r.id)}"><div>
      ${head(r, `<span class="verdict ${kind}">${esc(text)}</span>`)}
      ${about(r)}
      ${r.note && !r.gate ? `<p class="exp-note">${icon('info')}${esc(sentence(r.note))}</p>` : ''}
      ${gateList(r.gate)}
      <p class="exp-cmds">${copyBtn(r.journal, 'Journal:')}</p></div>
    <div class="exp-res"><p class="res-line"><span>Decided</span><span>${ago(r.at)}</span></p>${lines}${trip}</div></article>`;
}

function verdicts(rows) {
  const intro = `<div class="sec-h"><h2 id="v-h">Recent verdicts</h2></div>
    <p class="sec-p">Newest first. An experiment is folded in only if it passes every required check on the practice set; each check is listed in the order the gate runs them.</p>`;
  if (!rows.length) return `<section aria-labelledby="v-h">${intro}<p class="group-h">No verdicts yet</p></section>`;
  return `<section aria-labelledby="v-h">${intro}${rows.slice(0, 6).map(verdictRow).join('')}
    ${rows.length > 6 ? `<details class="more" data-k="older"><summary>${icon('right')}Older (${rows.length - 6})</summary>${rows.slice(6).map(verdictRow).join('')}</details>` : ''}</section>`;
}

// Trend: one small line per part, practice wrong calls at each main-line score.

function spark(s, key) {
  const pts = s.points;
  const ns = pts.map((p) => p.n);
  const lo = Math.min(...ns);
  const hi = Math.max(...ns, lo + 1);
  const x = (i) => (pts.length === 1 ? 50 : 2 + (i / (pts.length - 1)) * 96);
  const y = (n) => 4 + (1 - (n - lo) / (hi - lo)) * 32;
  const c = key === 'overall' ? 'var(--g-900)' : `var(--${key})`;
  return `<svg class="al-spark" viewBox="0 0 100 40" preserveAspectRatio="none" role="img" aria-label="${NAMES[key]}: ${ns.join(', ')}">
    <polyline points="${pts.map((p, i) => `${x(i)},${y(p.n)}`).join(' ')}" fill="none" stroke="${c}" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>
    ${pts.map((p, i) => `<circle cx="${x(i)}" cy="${y(p.n)}" r="2.5" fill="${c}" vector-effect="non-scaling-stroke"><title>${p.n} · ${day(p.at)}${p.experiment ? ` · after ${p.experiment}` : ''}</title></circle>`).join('')}
  </svg>`;
}

function trend(t) {
  const keys = ['overall', 'listener', 'tracker', 'checker'].filter((k) => t.series[k]);
  if (!keys.length) return '';
  return `<section aria-labelledby="t-h"><div class="sec-h"><h2 id="t-h">Practice trend</h2></div>
    <p class="sec-p">Wrong calls on the practice set at every score of the autolab line on ${esc(t.testSet)}. A change of 2 or fewer is noise.</p>
    <div class="al-trend">${keys
      .map((k) => {
        const s = t.series[k];
        const kind = Math.abs(s.change) <= 2 ? '' : s.change < 0 ? 'better' : 'worse';
        return `<div class="al-tr"><p class="al-tr-h">${sw(k)}${NAMES[k]}</p><p class="al-tr-n num"><b>${s.last}</b><span class="${kind ? `al-${kind}` : 'muted'}">${s.points.length > 1 ? `${fewer(s.change)} since ${s.first}` : 'one score so far'}</span></p>${spark(s, k)}</div>`;
      })
      .join('')}</div></section>`;
}

// Portfolio, coverage, spend

function portfolio(p) {
  const tiers = Object.keys(p.tiers);
  const row = (c) => `<tr><td>${sw(c.key)} ${esc(c.name)}${c.frozenSince ? ` <span class="verdict" title="Five experiments in a row without a fold">Quick fixes paused since ${day(c.frozenSince)}</span>` : ''}</td>
    ${tiers.map((t) => `<td class="r">${c.tiers[t] || '<span class="muted">0</span>'}</td>`).join('')}<td class="r"><b>${c.held}</b></td><td class="r">${c.sinceFold}</td></tr>`;
  return `<section aria-labelledby="p-h"><div class="sec-h"><h2 id="p-h">Portfolio</h2></div>
    <p class="sec-p">Slots in use by part and kind. Parts get slots in line with how many wrong calls they cause; about half go to quick fixes, three in ten to new ideas and two in ten to big bets.</p>
    <div class="tbl-wrap"><table class="tbl"><tr><th>Part</th>${tiers.map((t) => `<th class="r">${esc(p.tiers[t])}s</th>`).join('')}<th class="r">In use</th><th class="r">Tries since its last fold</th></tr>
      ${p.components.map(row).join('')}
      <tr><td class="muted">All parts</td>${tiers.map((t) => `<td class="r">${p.byTier[t]}</td>`).join('')}<td class="r"><b>${p.held}</b>${p.slots ? `<span class="muted"> of ${p.slots}</span>` : ''}</td><td></td></tr></table></div></section>`;
}

function coverage(c) {
  if (!c.clusters.length) return '';
  const kind = { fixed: 'better', tried: '', untouched: '' };
  return `<section aria-labelledby="c-h"><div class="sec-h"><h2 id="c-h">Coverage</h2></div>
    <p class="sec-p">Groups of similar failing cases on the practice set, and what has been tried on each. ${c.counts.fixed || 0} fixed, ${c.counts.tried || 0} tried and not fixed, ${c.counts.untouched || 0} not tried yet.</p>
    <div class="tbl-wrap"><table class="tbl"><tr><th>Group</th><th class="r">Cases</th><th>State</th><th>Tried by</th></tr>${c.clusters
      .map((r) => `<tr><td>${sw(r.component)} ${esc(NAMES[r.component] || '')} <span class="muted">${esc(r.label)}</span></td><td class="r">${r.incidents}</td>
        <td><span class="verdict ${kind[r.state]}">${esc(r.stateText)}</span>${r.triedTwice ? '<div class="muted small">Failed twice: another try needs a written reason</div>' : ''}</td>
        <td class="muted">${r.experiments.map(esc).join(', ') || '–'}</td></tr>`)
      .join('')}</table></div></section>`;
}

const amount = (r, n) => (n == null ? '–' : r.unit === 'usd' ? `$${n.toFixed(2)}` : r.unit === 'hours' ? `${n.toFixed(1)} h` : `${n}`);

function spend(s) {
  return `<section aria-labelledby="s-h"><div class="sec-h"><h2 id="s-h">Spend today</h2></div>
    <p class="sec-p">Against the daily caps${s.day ? ` for ${esc(s.day)}` : ''}. At a cap the supervisor lets running work finish and starts nothing new.</p>
    <div class="al-spend">${s.rows
      .map((r) => `<div class="al-sp${r.over ? ' over' : ''}"><span>${esc(r.label)}</span><span class="track"><i style="width:${r.share == null ? 0 : r.share * 100}%"></i></span>
        <span class="num"><b>${amount(r, r.used)}</b>${r.cap != null ? ` of ${amount(r, r.cap)}` : ' <span class="muted">no cap</span>'}</span></div>`)
      .join('')}</div></section>`;
}

function activity(rows) {
  if (!rows.length) return '';
  const what = (e) =>
    e.id ? `<b>${esc(e.id)}</b>: ${esc(e.toText)}${e.note ? `<div class="muted">${esc(sentence(e.note))}</div>` : ''}` : esc(sentence(e.note || e.toText));
  return `<section aria-labelledby="a-h"><details class="more" data-k="activity"><summary>${icon('right')}Recent activity (${rows.length})</summary>
    <div class="tbl-wrap"><table class="tbl">${rows.map((e) => `<tr><td class="muted">${day(e.at)}</td><td>${what(e)}</td></tr>`).join('')}</table></div></details></section>`;
}

function notStarted(v) {
  return `<div class="empty"><h1>Autolab hasn’t started</h1>
    <p>Nothing has been written yet. Once the supervisor runs, this page shows what it’s working on and refreshes by itself.</p>
    <div class="cmd-box">${copyBtn(`${CLI} loop --every 120`)}</div>${liveLink()}</div>${trend(v.trend)}${activity(v.activity)}`;
}

function render(v) {
  const open = new Set([...document.querySelectorAll('details[open]')].map((d) => (d.closest('[data-k]:not(details)')?.dataset.k || '') + '/' + d.dataset.k));
  topbar(v);
  $('main').innerHTML = v.started
    ? notices(v) + hero(v) + needsYou(v.needsHuman) + running(v.running) + queue(v.queue) + verdicts(v.verdicts) + trend(v.trend) + portfolio(v.portfolio) + coverage(v.coverage) + spend(v.spend) + activity(v.activity)
    : notStarted(v);
  for (const d of document.querySelectorAll('details[data-k]')) if (open.has((d.closest('[data-k]:not(details)')?.dataset.k || '') + '/' + d.dataset.k)) d.open = true;
}

async function load() {
  try {
    render(await api(`/api/autolab?${qs({ root })}`));
  } catch (e) {
    topbar(null);
    $('main').innerHTML = `<div class="notice err">Can’t load the autolab state: ${esc(e.message)}</div>`;
  }
}

load();
setInterval(() => document.hidden || load(), POLL_MS);
document.addEventListener('visibilitychange', () => document.hidden || load());
