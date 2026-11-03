import { icon } from './item.js';
import { $, esc, NAMES, qs, day, plural, fewer, sentence, sw, span, ago, api, topbar, staleNotice, plainLines, RESULT_KIND } from './al.js';
import { waitingPanel, pauseControl, pausedNotice, forgetLink, dropButton, typing, wire } from './act.js';

const root = new URL(location.href).searchParams.get('root');
const POLL_MS = 10_000;
const PARTS = ['listener', 'tracker', 'checker'];
const TIER_MARK = { exploit: 'circle', explore: 'diamond', bet: 'square' };
const SHORT = { proposed: 'proposed', accepted: 'waiting for an agent', implementing: 'writing', validating: 'checking', scoring: 'scoring', folding: 'folding', 'needs-human': 'waiting for you' };
const NOISE = 2;

let scope = localStorage.getItem('pg-scope') || 'testSet';
let last = null;
let seen = null; // id -> outcome|state at the previous render of this scope, for what to animate
let seenScope = null;
let seenTicks = null;
let tips = [];
let hover = null; // experiment under the pointer, in the graph or the list
let pinned = null; // experiment clicked or tapped: its branch and row stay lit
let labOpen = false;
let listHtml = '';

const clock = (t) => new Date(t).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
const dayOnly = (t) => new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

// Layout: one column per state of the main line, a node on the trunk with a comb after it holding one tooth per
// experiment that started from that state. The tooth that made the next node lies on the trunk, so the line runs
// through it; the others stack below and above, oldest nearest the trunk, each ending in a cap (its kind). Caps
// alternate short and long so neighbours never sit side by side, which lets rows pack tighter than a cap is wide;
// a comb with labels keeps rows a label apart.

const ROW = 8;
const ROW_LABELLED = 13;
const SHORT_T = 22;
const LONG_T = 42;
const GAP = 10;
const CH = 5.9;
const R = 112;

const minutes = (b) => Math.max(0, Date.parse(b.end || b.start) - Date.parse(b.start)) / 60e3;
const toothLen = (b, k) => (b.slot === 0 ? SHORT_T : Math.abs(b.slot) % 2 ? SHORT_T : LONG_T) * k;
const sum = (xs) => xs.reduce((s, x) => s + x, 0);

function toothLabel(b, i) {
  if (b.outcome === 'live') return `${b.id} · ${SHORT[b.state] || b.state} ${span(b.ageSeconds)}`;
  if (b.outcome === 'open') return b.id;
  if (b.outcome === 'folded' && b.foldCol !== i + 1 && b.foldN != null) return `→ ${b.foldN}`;
  return '';
}

function combs(view) {
  const cs = view.columns.map(() => ({ teeth: [], above: 0, below: 0 }));
  for (const b of view.branches) if (b.col != null && cs[b.col]) cs[b.col].teeth.push(b);
  cs.forEach((c, i) => {
    const win = c.teeth.find((b) => b.foldCol === i + 1);
    let k = 0;
    for (const b of c.teeth) {
      b.slot = b === win ? 0 : k % 2 ? -(k + 1) / 2 : k / 2 + 1;
      if (b !== win) k++;
      c.below = Math.max(c.below, b.slot);
      c.above = Math.max(c.above, -b.slot);
    }
  });
  return cs;
}

// Only the last column (the line as it is now, where live work usually sits) makes room for its labels, which may run
// on into the right margin; earlier ones are cut to the room before the next node.
const labelAt = (k) => LONG_T * k + 11;
const widths = (cs, k) =>
  cs.map((c, i) => {
    const short = (b) => (b.outcome === 'folded' ? labelAt(k) + toothLabel(b, i).length * CH : 0);
    const teeth = Math.max(40, GAP + 16 + Math.max(0, ...c.teeth.map((b) => Math.max(toothLen(b, k) + 6, short(b)))));
    if (i < cs.length - 1) return teeth;
    return Math.max(teeth, GAP + labelAt(k) + Math.max(0, ...c.teeth.map((b) => toothLabel(b, i).length * CH)) - R + 40);
  });

// A comb's rows can sit closer than a label is tall unless two labelled teeth are neighbours on the same side.
const rowOf = (c, i) =>
  c.teeth.some((a) => a.slot && toothLabel(a, i) && c.teeth.some((b) => b.slot === a.slot + Math.sign(a.slot) && toothLabel(b, i))) ? ROW_LABELLED : ROW;

function fit(text, px) {
  const n = Math.floor(px / CH);
  return text.length <= n ? text : n >= 8 ? text.slice(0, n - 1) + '…' : '';
}

// Graph pieces

const tip = (html) => tips.push(html) - 1;

function mark(kind, cx, cy, cls, s = 1) {
  if (kind === 'diamond') return `<path class="${cls}" d="M${cx} ${cy - 5 * s}L${cx + 5 * s} ${cy}L${cx} ${cy + 5 * s}L${cx - 5 * s} ${cy}Z"/>`;
  if (kind === 'square') return `<rect class="${cls}" x="${cx - 4 * s}" y="${cy - 4 * s}" width="${8 * s}" height="${8 * s}" rx="1"/>`;
  return `<circle class="${cls}" cx="${cx}" cy="${cy}" r="${(kind === 'hollow' ? 3 : 4) * s}"/>`;
}

function branchTip(b) {
  const who = [NAMES[b.component] || b.component, b.tierText, b.source === 'lab' ? 'hand-run lab' : 'autolab', b.arm ? `built by ${b.arm}` : ''].filter(Boolean).join(' · ');
  const res =
    b.change != null
      ? `<p><span>Practice</span><b>${fewer(b.change)}</b>${b.expected != null ? ` · expected ${fewer(b.expected)}` : ''}${Math.abs(b.change) <= NOISE ? ' <i>(noise)</i>' : ''}</p>`
      : '';
  const state =
    b.outcome === 'live'
      ? `<p><span>Now</span><b>${esc(b.stateText)}</b> for ${span(b.ageSeconds)}</p>`
      : b.outcome === 'folded'
        ? `<p><span>Folded</span>${day(b.foldAt)}${b.foldN != null ? ` · the line went to <b>${b.foldN}</b>` : ''}</p>`
        : b.outcome === 'open'
          ? `<p><span>Open</span>${esc(b.stateText)}</p>`
          : `<p><span>${esc(sentence(b.state))}</span>${day(b.end)} · ran ${span(Math.round(minutes(b) * 60))}</p>`;
  return `<h4>${sw(b.component)}${esc(b.id)}</h4><small>${esc(who)}</small>
    <div class="pg-plain">${plainLines(b)}</div>
    <div class="pg-techd">${b.eli5?.plain && b.hypothesis ? `<q>${esc(b.hypothesis)}</q>` : ''}
    <p><span>Started</span>${day(b.start)}${b.base != null ? ` · line at ${b.base}` : ''}</p>${state}${res}
    ${b.reason && !b.resultText ? `<p class="pg-why">${esc(sentence(b.reason))}</p>` : ''}</div>`;
}

function draw(v, view, W) {
  tips = [];
  const s = v.score;
  const pts = view.spine;
  const cols = view.columns;
  if (!cols.length) return '<p class="group-h">No main-line score in this range yet</p>';
  const L = 48;
  const chartTop = 26;
  const chartH = 150;
  const chartBot = chartTop + chartH;
  const cs = combs(view);
  let k = 1;
  let ws = widths(cs, k);
  while (k > 0.5 && sum(ws) > W - L - R - 10) ws = widths(cs, (k -= 0.1));
  const extra = Math.max(0, W - L - R - 10 - sum(ws)) / cols.length;
  ws = ws.map((w) => w + extra);
  const X = [L + 10];
  ws.forEach((w) => X.push(X.at(-1) + w));
  const end = X.pop();
  const Wd = Math.max(W, end + R);
  const row = cs.map(rowOf);
  const above = Math.max(0, ...cs.map((c, i) => c.above * row[i]));
  const below = Math.max(0, ...cs.map((c, i) => c.below * row[i]));
  const trunkY = chartBot + 46 + above;
  const graphBot = trunkY + below + 20;
  const H = graphBot + 30;
  const P = cols.map((c) => pts[c.point]);
  const right = (i) => (i + 1 < cols.length ? X[i + 1] : end);
  const out = [];

  // background: the hand-run era, test-set boundaries
  const firstAuto = P.findIndex((p) => p.source === 'autolab');
  if (P.some((p) => p.source === 'lab')) {
    const x1 = firstAuto >= 0 ? X[firstAuto] - 8 : end;
    out.push(`<rect class="pg-era" x="${L}" y="${chartBot + 16}" width="${Math.max(0, x1 - L)}" height="${graphBot - chartBot - 16}"/>
      <text class="pg-era-t" x="${L + 6}" y="${graphBot - 6}">Hand-run lab</text>`);
  }
  if (firstAuto > 0)
    out.push(`<line class="pg-auto" x1="${X[firstAuto] - 8}" x2="${X[firstAuto] - 8}" y1="${chartBot + 16}" y2="${graphBot}"/><text class="pg-era-t pg-auto-t" x="${X[firstAuto] - 2}" y="${graphBot - 6}">Autolab</text>`);
  for (const b of view.boundaries) {
    const i = P.findIndex((p) => p.at === b.at);
    if (i < 0) continue;
    const bx = X[i] - 8;
    out.push(`<g class="pg-bound" data-i="${tip(`<h4>New test set</h4><p>${esc(b.from)} → <b>${esc(b.to)}</b> on ${day(b.at)}</p><p class="pg-why">The line read ${b.before} on ${esc(b.from)} and ${b.after} on ${esc(b.to)}: counts on two test sets don’t compare, so the line is broken here.</p>`)}">
      <line x1="${bx}" x2="${bx}" y1="${chartTop - 14}" y2="${graphBot}"/><text x="${bx + 4}" y="${chartTop - 16}">${esc(b.to)}</text><rect class="hit" x="${bx - 6}" y="${chartTop - 26}" width="12" height="${graphBot - chartTop + 26}"/></g>`);
  }

  // score chart: one step per column, practice wrong calls stacked by part, broken at a new test set
  const top = Math.max(10, ...P.map((p) => p.n), s.teacher != null && P.some((p) => p.testSet === s.testSet) ? s.teacher : 0);
  const stepY = top <= 30 ? 5 : top <= 60 ? 10 : top <= 150 ? 25 : 50;
  const ymax = Math.ceil((top * 1.06) / stepY) * stepY;
  const y = (n) => chartTop + chartH * (1 - n / ymax);
  for (let g = 0; g <= ymax; g += stepY * (ymax / stepY > 6 ? 2 : 1))
    out.push(`<line class="pg-grid" x1="${L}" x2="${end}" y1="${y(g)}" y2="${y(g)}"/><text class="pg-yl" x="${L - 8}" y="${y(g) + 4}" text-anchor="end">${g}</text>`);
  let line = '';
  P.forEach((p, i) => {
    const xa = X[i];
    const xb = right(i);
    if (PARTS.every((q) => p.parts[q] != null)) {
      let acc = 0;
      for (const q of PARTS) {
        out.push(`<rect class="pg-stack" style="--c:var(--${q})" x="${xa}" y="${y(acc + p.parts[q])}" width="${Math.max(0, xb - xa)}" height="${y(acc) - y(acc + p.parts[q])}"/>`);
        acc += p.parts[q];
      }
    }
    line += i && P[i - 1].testSet === p.testSet ? `V${y(p.n)}H${xb}` : `M${xa} ${y(p.n)}H${xb}`;
  });
  out.push(`<path class="pg-line" d="${line}"/>`);
  const cur = P.findIndex((p) => p.testSet === s.testSet);
  if (s.teacher != null && cur >= 0) {
    out.push(`<g data-i="${tip(`<h4>Big reference listener</h4><p>${s.teacher} wrong calls on the ${esc(s.testSet)} practice set with the big reference listener in place of the app’s.</p>`)}">
      <line class="pg-ref" x1="${X[cur]}" x2="${end + 4}" y1="${y(s.teacher)}" y2="${y(s.teacher)}"/><text class="pg-ref-t" x="${end + 8}" y="${y(s.teacher) + 4}">Reference ${s.teacher}</text>
      <rect class="hit" x="${X[cur]}" y="${y(s.teacher) - 5}" width="${end - X[cur]}" height="10"/></g>`);
  }
  out.push(`<text class="pg-goal-t" x="${end + 8}" y="${y(s.goal.n) + 4}">Goal ${s.goal.n}</text>`);
  if (s.goal.n > 0) out.push(`<line class="pg-ref goal" x1="${L}" x2="${end + 4}" y1="${y(s.goal.n)}" y2="${y(s.goal.n)}"/>`);

  // the trunk, then each column's comb
  // green where the line moved by a fold, thin grey otherwise (a new test set, a re-score, the stretch to now)
  P.forEach((p, i) => {
    const fold = P[i + 1]?.kind === 'fold' && P[i + 1].testSet === p.testSet;
    out.push(`<line class="pg-spine ${fold ? 'fold' : 'quiet'}" x1="${X[i]}" x2="${right(i)}" y1="${trunkY}" y2="${trunkY}"/>`);
  });
  out.push(`<text class="pg-now-t" x="${end + 8}" y="${trunkY + 4}">now</text>`);
  const labels = [];
  cs.forEach((c, i) => {
    const sx = X[i] + GAP;
    const ty = (slot) => trunkY + slot * row[i];
    if (c.above || c.below) out.push(`<line class="pg-comb" x1="${sx}" x2="${sx}" y1="${ty(-c.above)}" y2="${ty(c.below)}"/>`);
    for (const b of c.teeth) {
      const yl = ty(b.slot);
      const xe = sx + toothLen(b, k);
      const was = seen?.get(b.id);
      const anim = seen && seenScope === scope ? (!was ? ' enter' : b.outcome === 'folded' && !was.startsWith('folded') ? ' merging' : was !== `${b.outcome}|${b.state}` ? ' changed' : '') : '';
      const cls = `pg-br ${b.outcome} ${b.source}${b.slot === 0 ? ' win' : ''}${b.waiting ? ' waiting' : ''}${anim}`;
      const parts = [`<path class="br" pathLength="1" d="M${sx} ${yl}H${xe}"/>`];
      if (b.outcome === 'live') parts.push(`<circle class="pulse" cx="${xe}" cy="${yl}" r="4.5"/>`);
      if (b.slot !== 0) parts.push(mark(b.source === 'lab' ? 'hollow' : TIER_MARK[b.tier] || 'circle', xe, yl, 'cap', b.source === 'lab' ? 1.4 : 1.05));
      const lx = sx + labelAt(k);
      const text = fit(toothLabel(b, i), (i + 1 < cols.length ? X[i + 1] : Wd) - lx - 4);
      if (text && lx - xe > 12) labels.push(`<line class="pg-lead" x1="${xe + 7}" x2="${lx - 4}" y1="${yl}" y2="${yl}"/>`);
      if (text) labels.push(`<text class="pg-bl ${b.outcome} ${b.source}${b.waiting ? ' waiting' : ''}${anim}" x="${lx}" y="${yl + 3.5}">${esc(text)}</text>`);
      out.push(`<g class="${cls}" style="--c:var(--${b.component || 'g-500'})" data-k="${esc(b.id)}" data-i="${tip(branchTip(b))}" tabindex="0" role="img" aria-label="${esc(`${b.id}: ${b.stateText}`)}">${parts.join('')}<path class="hit" d="M${sx} ${yl}H${xe + 6}"/></g>`);
    }
  });
  out.push(...labels);

  // nodes on top, with the count after them; the date of each under the graph where there is room
  let lastX = -1e9;
  let lastDay = null;
  cols.forEach((col, i) => {
    const p = P[i];
    const isNew = seen && seenScope === scope && p.experiment && seen.get(p.experiment) && !seen.get(p.experiment).startsWith('folded');
    const what = p.experiment ? `Fold of <b>${esc(p.experiment)}</b>` : esc(sentence(p.note) || 'Main-line score');
    const parts = p.parts && PARTS.every((q) => p.parts[q] != null) ? `<p>${PARTS.map((q) => `${sw(q)}${p.parts[q]}`).join(' ')}</p>` : '';
    const also = col.also.length ? `<p class="pg-why">Re-scored ${plural(col.also.length, 'more time')} after, still ${p.n}: ${col.also.map((j) => esc(pts[j].note || day(pts[j].at))).join('; ')}.</p>` : '';
    const n = cs[i].teeth.length;
    out.push(`<g class="pg-node ${p.kind} ${p.source}${isNew ? ' pop' : ''}" data-i="${tip(`<h4>${p.n} wrong calls</h4><small>${esc(p.testSet)} · ${day(p.at)} · <code>${esc(p.sha)}</code></small><p>${what}</p>${parts}${also}<p>${plural(n, 'experiment')} started from here</p>`)}">
      <circle cx="${X[i]}" cy="${trunkY}" r="${p.kind === 'fold' ? 6 : 3.5}"/><text x="${X[i]}" y="${trunkY - 11}">${p.n}</text><circle class="hit" cx="${X[i]}" cy="${trunkY}" r="10"/></g>`);
    if (X[i] - lastX < 70) return;
    const d = dayOnly(p.at);
    out.push(`<line class="pg-axt" x1="${X[i]}" x2="${X[i]}" y1="${graphBot + 4}" y2="${graphBot + 9}"/><text class="pg-xl" x="${X[i]}" y="${graphBot + 22}" text-anchor="middle">${d !== lastDay ? `${d} ` : ''}${clock(p.at)}</text>`);
    lastX = X[i];
    lastDay = d;
  });
  return `<svg class="pg-svg" width="${Wd}" height="${H}" viewBox="0 0 ${Wd} ${H}" role="group" aria-label="Main line and the experiments started from each of its states">${out.join('')}</svg>`;
}

// Sections

function hero(v) {
  const s = v.score;
  const c = s.current;
  if (!c) return `<div class="hero"><p class="lede">No main-line score yet.</p></div>`;
  const part = PARTS.map((k) => `${k} <b class="num">${c.parts?.[k] ?? '–'}</b>`).join(', ');
  const start = s.start;
  const removed = start ? start.n - c.wrongCalls : 0;
  const span0 = start ? Math.max(start.n - s.goal.n, 1) : 1;
  const pos = (n) => `${Math.min(100, Math.max(0, ((start.n - n) / span0) * 100))}%`;
  const meter = start
    ? `<div class="pg-meter" role="img" aria-label="${removed} of ${start.n - s.goal.n} wrong calls removed since the start of ${esc(s.testSet)}">
        <div class="pg-mt"><span class="now" style="left:${pos(c.wrongCalls)}"><b class="num">${c.wrongCalls}</b> now</span>
          ${s.teacher != null && s.teacher <= start.n && Math.abs(s.teacher - c.wrongCalls) > start.n / 8 ? `<span style="left:${pos(s.teacher)}">Reference ${s.teacher}</span>` : ''}</div>
        <div class="pg-track"><i style="width:${pos(c.wrongCalls)}"></i>
          ${s.teacher != null && s.teacher <= start.n ? `<b class="pg-mk ref" style="left:${pos(s.teacher)}" title="The big reference listener: ${s.teacher}"></b>` : ''}
          ${s.autolab ? `<b class="pg-mk auto" style="left:${pos(s.autolab.start)}" title="Where autolab took over: ${s.autolab.start}"></b>` : ''}
          <b class="pg-mk now" style="left:${pos(c.wrongCalls)}"></b></div>
        <div class="pg-ml"><span><b class="num">${start.n}</b> at the start of ${esc(s.testSet)}</span>
          <span>Goal <b class="num">${s.goal.n}</b>${s.goal.source === 'zero' ? ' (none set)' : ''}</span></div></div>`
    : '';
  const lines = [];
  if (start) lines.push(`<b>${removed > 0 ? `${removed} fewer` : removed < 0 ? `${-removed} more` : 'No change'}</b> than at the start of ${esc(s.testSet)} (${start.n} on ${day(start.at)}); <b>${s.toGo}</b> to go to reach ${s.goal.n}.`);
  if (s.teacher != null) lines.push(c.wrongCalls < s.teacher ? `${s.teacher - c.wrongCalls} fewer than the big reference listener makes on the same set (${s.teacher}).` : `${c.wrongCalls - s.teacher} more than the big reference listener makes on the same set (${s.teacher}).`);
  if (s.autolab) lines.push(`Since autolab took over at ${s.autolab.start} (${day(s.autolab.at)}): ${s.autolab.change ? fewer(s.autolab.change) : 'no change yet'}, ${plural(s.autolab.folds, 'fold')}.`);
  const pace = s.rate
    ? `<p class="pg-pace">${icon(s.rate.perDay > 0 ? 'down' : 'minus')}<span>Pace: about <b class="num">${Math.abs(Math.round(s.rate.perDay))}</b> ${s.rate.perDay >= 0 ? 'fewer' : 'more'} a day over ${span(Math.round(s.rate.days * 86400))}${s.lastDay != null ? `; ${fewer(s.lastDay)} in the last day` : ''}.
        ${s.projection ? `<em>Projection:</em> at that pace the line would reach ${s.goal.n} around <b>${dayOnly(s.projection.at)}</b> (${s.projection.days} days). A straight line from the pace so far, not a forecast.` : ''}</span></p>`
    : '';
  return `<div class="hero pg-hero">
    <div class="score"><span class="score-n" id="pg-n">${c.wrongCalls}</span>
      <p class="score-u"><strong>wrong calls on the practice set</strong><span>${s.autolab ? 'The autolab line' : 'The main line'} on ${esc(s.testSet)}. Lower is better</span></p></div>
    <p class="parts">${c.missed != null ? `<b class="num">${c.missed}</b> missed mistakes + <b class="num">${c.falseWarnings}</b> false warnings. ` : ''}Charged to the ${part}.</p>
    ${meter}<p class="pg-close">${lines.join(' ')}</p>${pace}</div>`;
}

function ticker(v) {
  const rows = v.ticker;
  const fresh = new Set(seenTicks ? rows.filter((e) => !seenTicks.has(e.key)).map((e) => e.key) : []);
  seenTicks = new Set(rows.map((e) => e.key));
  const kind = (e) => (e.to === 'folded' ? 'better' : ['discarded', 'abandoned', 'superseded', 'rejected'].includes(e.to) ? 'worse' : e.to === 'needs-human' || e.to === 'paused' ? 'wait' : '');
  const h = v.heartbeat;
  return `<div class="pg-tick-h"><h2>Live</h2><span class="muted small">${h ? `${h.stale ? 'Supervisor silent' : 'Supervisor checked in'} ${span(h.ageSeconds)} ago` : 'Supervisor not started'} · refreshes every 10 s</span></div>
    ${rows.length ? `<ol>${rows
      .map((e) => `<li class="${fresh.has(e.key) ? 'new ' : ''}${kind(e)}" title="${esc(e.note)}"><time>${clock(e.at)}</time>
          <span>${e.to === 'rejected' ? `${e.id ? `${sw(e.component)}<b>${esc(e.id)}</b> ` : ''}Your command was turned down: ${esc((e.note || '').replace(/^your \w+: /, ''))}` : e.id ? `${sw(e.component)}<b>${esc(e.id)}</b> ${esc(e.toText)}` : esc(sentence(e.note || e.toText))}</span></li>`)
      .join('')}</ol>` : '<p class="muted small">Nothing has happened yet.</p>'}`;
}

function key() {
  const k = (c, t) => `<span>${c}${t}</span>`;
  const svg = (inner) => `<svg width="18" height="12" viewBox="0 0 18 12">${inner}</svg>`;
  return `<div class="axis-key pg-key">
    ${PARTS.map((p) => k(sw(p), NAMES[p])).join('')}
    ${k(svg('<circle cx="9" cy="6" r="4" fill="var(--g-700)"/>'), 'Quick fix')}${k(svg('<path d="M9 1L14 6L9 11L4 6Z" fill="var(--g-700)"/>'), 'New idea')}${k(svg('<rect x="5" y="2" width="8" height="8" rx="1" fill="var(--g-700)"/>'), 'Big bet')}
    ${k(svg('<circle cx="9" cy="6" r="5" fill="var(--better)"/>'), 'Folded in')}${k(svg('<path d="M2 6h14" stroke="var(--g-500)" stroke-width="2" opacity="0.4"/>'), 'Let go')}
    ${k(svg('<circle cx="9" cy="6" r="4.5" fill="var(--gold-400)"/>'), 'Waiting for you')}${k(svg('<path d="M1 6h16" stroke="var(--g-400)" stroke-width="1.5" stroke-dasharray="3 2"/>'), 'Open, hand-run')}</div>`;
}

function section(v) {
  const seg = [['autolab', 'Autolab only'], ['testSet', `Since ${v.testSet || 'the test set'}`], ['all', 'Everything']]
    .map(([k, t]) => `<button data-scope="${k}" aria-pressed="${k === scope}">${esc(t)}</button>`)
    .join('');
  return `<section aria-labelledby="g-h"><div class="sec-h"><h2 id="g-h">Branches and folds</h2><div class="seg" role="group" aria-label="How far back">${seg}</div></div>
    <p class="sec-p">The main line runs left to right, one node per state with its practice count. The comb after a node holds every experiment started from that state, one tooth each,
      capped with its kind; the line is green where a fold moved it, through the tooth that made the next node. A faded tooth was let go (point at it for why), a pulsing one is still running,
      and <b>→ n</b> marks a fold that landed later, after the line had moved on. <span id="pg-counts"></span></p>
    ${key()}<div class="pg-fig" id="pg-fig"><div class="pg-tip" id="pg-tip" hidden></div><div id="pg-svg"></div></div>
    <div class="pg-list" id="pg-list" aria-label="The experiments in this range, in plain words"></div></section>`;
}

// The experiments in this range, in words: always on the page, not only on hover.

const CHIP = { folded: ['better', 'Folded in'], discarded: ['worse', 'Discarded'], superseded: ['', 'Gain gone'], abandoned: ['', 'Given up'] };

function chip(b) {
  if (b.outcome === 'live') return `<span class="verdict${b.waiting ? ' wait' : ''}">${esc(sentence(SHORT[b.state] || b.state))}</span>`;
  if (b.outcome === 'open') return '<span class="verdict">Open</span>';
  const [k, t] = CHIP[b.state] || [RESULT_KIND[b.state] ?? '', b.stateText];
  return `<span class="verdict ${k}">${esc(t)}</span>`;
}

function rowMark(b) {
  const kind = b.source === 'lab' ? 'hollow' : TIER_MARK[b.tier] || 'circle';
  return `<svg class="pg-rm ${b.source}" width="14" height="14" viewBox="0 0 14 14" style="--c:var(--${b.component || 'g-500'})" aria-hidden="true">${mark(kind, 7, 7, '')}</svg>`;
}

function row(b) {
  const when = b.outcome === 'live' ? `for ${span(b.ageSeconds)}` : b.outcome === 'open' ? `since ${dayOnly(b.start)}` : day(b.end);
  const who = [NAMES[b.component] || b.component, b.tierText].filter(Boolean).join(' · ');
  return `<li class="pg-row ${b.outcome} ${b.source}" data-k="${esc(b.id)}" tabindex="0">
    <div class="pg-row-h">${rowMark(b)}<b>${esc(b.id)}</b>${chip(b)}<span class="muted small">${esc(who)}${who ? ' · ' : ''}${when}</span></div>
    <div class="pg-row-b">${plainLines(b)}${b.source === 'autolab' && b.outcome === 'live' && b.state !== 'needs-human' ? `<p class="pg-row-a">${dropButton(b.id)}</p>` : ''}</div></li>`;
}

function list(view) {
  const br = view.branches;
  const t = (x) => Date.parse(x || 0);
  const live = br.filter((b) => b.outcome === 'live').sort((a, b) => t(b.start) - t(a.start));
  const done = br.filter((b) => b.source === 'autolab' && b.outcome !== 'live').sort((a, b) => t(b.end) - t(a.end));
  const lab = br.filter((b) => b.source === 'lab').sort((a, b) => (b.outcome === 'open') - (a.outcome === 'open') || t(b.end) - t(a.end));
  const group = (h, rows) => (rows.length ? `<h3 class="group-h">${h} <span>${rows.length}</span></h3><ol class="pg-rows">${rows.map(row).join('')}</ol>` : '');
  const labOpenN = lab.filter((b) => b.outcome === 'open').length;
  return (
    group('In flight', live) +
    group('Decided by autolab, newest first', done) +
    (lab.length
      ? `<details class="more pg-lab" id="pg-lab"${labOpen ? ' open' : ''}><summary>${icon('right')}Hand-run lab: ${plural(lab.length, 'experiment')}${labOpenN ? `, ${labOpenN} still open` : ''}</summary><ol class="pg-rows">${lab.map(row).join('')}</ol></details>`
      : '') || '<p class="group-h">No experiments in this range</p>'
  );
}

const counts = (c) => `${c.live} running, ${c.folded} folded, ${c.ended} let go${c.open ? `, ${c.open} still open in the hand-run lab` : ''}.`;

// Render and refresh

function width() {
  if ($('pg-fig')?.clientWidth) return Math.max(720, $('pg-fig').clientWidth);
  const m = $('main');
  const cs = getComputedStyle(m);
  return Math.max(720, m.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight));
}

function tween(el, from, to) {
  if (!el || from == null || from === to || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 900);
    el.textContent = Math.round(from + (to - from) * (1 - (1 - k) ** 3));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function render(v) {
  const view = v.views[scope] || v.views.testSet;
  const before = last?.score?.current?.wrongCalls;
  topbar(v, pauseControl(v));
  if (!$('pg-top'))
    $('main').innerHTML = `<div id="pg-notice"></div><div id="pg-wait"></div><div class="pg-top" id="pg-top"><div id="pg-hero"></div><aside class="pg-tick" id="pg-tick" aria-live="polite"></aside></div><div id="pg-sec"></div>
      <p class="pg-back"><a href="/autolab${root ? `?${qs({ root })}` : ''}">${icon('right')}Running now, verdicts, portfolio and spend</a><span id="pg-forget"></span></p>`;
  $('pg-notice').innerHTML = staleNotice(v.heartbeat) + pausedNotice(v, v.ticker);
  if (!typing()) $('pg-wait').innerHTML = waitingPanel(v.waiting, v, v.ticker);
  $('pg-forget').innerHTML = forgetLink();
  $('pg-hero').innerHTML = hero(v);
  $('pg-tick').innerHTML = ticker(v);
  const fresh = !$('pg-fig') || seenScope !== scope || last?.testSet !== v.testSet;
  if (fresh) {
    $('pg-sec').innerHTML = section(v);
    listHtml = '';
  }
  $('pg-counts').textContent = counts(view.counts);
  $('pg-svg').innerHTML = draw(v, view, width());
  if (fresh) $('pg-fig').scrollLeft = $('pg-fig').scrollWidth;
  const lh = list(view);
  if (lh !== listHtml) $('pg-list').innerHTML = listHtml = lh;
  if (pinned && !view.branches.some((b) => b.id === pinned)) pinned = null;
  light();
  const t = $('pg-tip');
  if (!t.hidden) {
    const el = t.dataset.k && $('pg-svg').querySelector(`[data-k="${CSS.escape(t.dataset.k)}"]`);
    if (el) t.innerHTML = tips[+el.dataset.i];
    else t.hidden = true;
  }
  tween($('pg-n'), before, v.score.current?.wrongCalls);
  seen = new Map(view.branches.map((b) => [b.id, `${b.outcome}|${b.state}`]));
  seenScope = scope;
  last = v;
}

function showTip(el, px, py) {
  const t = $('pg-tip');
  const fig = $('pg-fig');
  t.innerHTML = tips[+el.dataset.i];
  t.dataset.k = el.dataset.k || '';
  t.hidden = false;
  const fw = fig.clientWidth;
  const left = Math.min(Math.max(8, px + 14 - fig.scrollLeft), fw - t.offsetWidth - 8) + fig.scrollLeft;
  const top = py + 16 + t.offsetHeight > fig.clientHeight ? py - t.offsetHeight - 12 : py + 16;
  t.style.left = `${left}px`;
  t.style.top = `${Math.max(0, top)}px`;
}

// Highlight: an experiment under the pointer (or pinned) lights its branch and its row together.

const sel = (id) => `#pg-sec [data-k="${CSS.escape(id)}"]`;

function light() {
  const id = hover ?? pinned;
  for (const el of document.querySelectorAll('#pg-sec .hl, #pg-sec .pinned')) el.classList.remove('hl', 'pinned');
  $('pg-svg')?.firstElementChild?.classList.toggle('focus', !!id);
  if (id) for (const el of document.querySelectorAll(sel(id))) el.classList.add('hl');
  if (pinned) $('pg-list')?.querySelector(`[data-k="${CSS.escape(pinned)}"]`)?.classList.add('pinned');
}

function pin(id, scroll = false) {
  pinned = id;
  light();
  const row = id && $('pg-list')?.querySelector(`[data-k="${CSS.escape(id)}"]`);
  if (!row || !scroll) return;
  const d = row.closest('details');
  if (d && !d.open) d.open = labOpen = true;
  row.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
}

function tipAt(el) {
  const fig = $('pg-fig');
  const a = el.getBoundingClientRect();
  const r = fig.getBoundingClientRect();
  showTip(el, a.left - r.left + fig.scrollLeft + 24, a.bottom - r.top - 8);
}

document.addEventListener('pointerover', (e) => {
  const id = e.target.closest?.('#pg-sec [data-k]')?.dataset.k ?? null;
  if (id === hover) return;
  hover = id;
  light();
});
document.addEventListener('toggle', (e) => e.target.id === 'pg-lab' && (labOpen = e.target.open), true);

document.addEventListener('pointermove', (e) => {
  const fig = $('pg-fig');
  if (!fig || pinned) return;
  const el = e.target.closest?.('[data-i]');
  if (!el || !fig.contains(el)) {
    if (!$('pg-tip').hidden && !e.target.closest?.('.pg-tip')) $('pg-tip').hidden = true;
    return;
  }
  const r = fig.getBoundingClientRect();
  showTip(el, e.clientX - r.left + fig.scrollLeft, e.clientY - r.top);
});
document.addEventListener('focusin', (e) => {
  const k = e.target.closest?.('#pg-sec [data-k]')?.dataset.k;
  if (k && !pinned) {
    hover = k;
    light();
  }
  const el = e.target.closest?.('#pg-fig [data-i]');
  if (el && !pinned) tipAt(el);
});
document.addEventListener('focusout', (e) => !pinned && e.target.closest?.('#pg-fig [data-i]') && ($('pg-tip').hidden = true));

// Click or tap a branch: its tooltip stays and its row is scrolled to and pinned. Click a row to pin its branch.
function choose(e, el) {
  const id = el.dataset.k;
  if (el.closest('#pg-list')) {
    pin(pinned === id ? null : id);
    $('pg-tip').hidden = true;
    return;
  }
  pin(id, true);
  if (!e.clientX && !e.clientY) return tipAt(el);
  const fig = $('pg-fig');
  const r = fig.getBoundingClientRect();
  showTip(el, e.clientX - r.left + fig.scrollLeft, e.clientY - r.top);
}

document.addEventListener('keydown', (e) => {
  const el = (e.key === 'Enter' || e.key === ' ') && e.target.closest?.('#pg-sec [data-k]');
  if (!el) return;
  e.preventDefault();
  choose(e, el);
});
document.addEventListener('click', (e) => {
  if (!$('pg-sec') || e.target.closest('[data-scope], summary, a, button')) return;
  const el = e.target.closest('#pg-sec [data-k]');
  if (el) return choose(e, el);
  if (pinned && !e.target.closest('.pg-tip')) {
    pin(null);
    $('pg-tip').hidden = true;
  }
});
document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-scope]');
  if (!b || b.dataset.scope === scope) return;
  scope = b.dataset.scope;
  localStorage.setItem('pg-scope', scope);
  seen = null;
  if (last) render(last);
});
let resize;
addEventListener('resize', () => {
  clearTimeout(resize);
  resize = setTimeout(() => last && render(last), 150);
});

async function load() {
  try {
    render(await api(`/api/autolab/progress?${qs({ root })}`));
  } catch (e) {
    if (last) return;
    topbar(null);
    $('main').innerHTML = `<div class="notice err">Can’t load the autolab progress: ${esc(e.message)}</div>`;
  }
}

wire({ root, refresh: load, rerender: () => last && render(last) });
load();
setInterval(() => document.hidden || load(), POLL_MS);
document.addEventListener('visibilitychange', () => document.hidden || load());
