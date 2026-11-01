import { icon } from './item.js';
import { $, esc, NAMES, qs, day, plural, fewer, sentence, sw, span, ago, api, topbar, staleNotice } from './al.js';

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

const clock = (t) => new Date(t).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
const dayOnly = (t) => new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

// Time axis: windows of activity side by side, quiet stretches squeezed into fixed-width breaks.

// The stretch holding now gets at least NOW_SHARE of the width, so live work isn't squashed by a long history.
const NOW_SHARE = 0.3;

function xscale(view, x0, x1, B = 36) {
  const ws = view.windows.map((w) => [Date.parse(w.from), Date.parse(w.to)]);
  const dur = ws.map(([a, b]) => Math.max(b - a, 1));
  const avail = Math.max(0, x1 - x0 - B * (ws.length - 1));
  const total = dur.reduce((s, d) => s + d, 0);
  let px = dur.map((d) => (d / total) * avail);
  if (ws.length > 1 && px.at(-1) < avail * NOW_SHARE) {
    const rest = total - dur.at(-1);
    px = dur.map((d, i) => (i === ws.length - 1 ? avail * NOW_SHARE : (d / rest) * avail * (1 - NOW_SHARE)));
  }
  const win = [];
  let x = x0;
  ws.forEach(([a, b], i) => {
    win.push({ a, b, x0: x, x1: x + px[i], k: px[i] / dur[i] });
    x += px[i] + B;
  });
  const f = (t) => {
    const v = typeof t === 'number' ? t : Date.parse(t);
    for (let i = 0; i < win.length; i++) {
      const w = win[i];
      if (v < w.a) return i === 0 ? w.x0 : win[i - 1].x1 + (B * (v - win[i - 1].b)) / (w.a - win[i - 1].b);
      if (v <= w.b) return w.x0 + (v - w.a) * w.k;
    }
    return x1;
  };
  f.windows = win;
  f.breaks = win.slice(1).map((w, i) => ({ x0: win[i].x1, x1: w.x0, ms: w.a - win[i].b }));
  return f;
}

function ticks(x) {
  const out = [];
  const steps = [0.25, 0.5, 1, 2, 3, 6, 12, 24].map((h) => h * 3600e3);
  for (const w of x.windows) {
    const step = steps.find((s) => s * w.k >= 72) || steps.at(-1);
    out.push({ x: w.x0, t: w.a, first: true });
    for (let t = Math.ceil(w.a / step) * step; t <= w.b; t += step) if ((t - w.a) * w.k >= 80 && (w.b - t) * w.k >= 8) out.push({ x: x(t), t });
  }
  return out;
}

// Graph pieces

const tip = (html) => tips.push(html) - 1;

function mark(kind, cx, cy, cls) {
  if (kind === 'diamond') return `<path class="${cls}" d="M${cx} ${cy - 5}L${cx + 5} ${cy}L${cx} ${cy + 5}L${cx - 5} ${cy}Z"/>`;
  if (kind === 'square') return `<rect class="${cls}" x="${cx - 4}" y="${cy - 4}" width="8" height="8" rx="1"/>`;
  return `<circle class="${cls}" cx="${cx}" cy="${cy}" r="${kind === 'hollow' ? 3 : 4}"/>`;
}

function branchTip(b) {
  const who = [NAMES[b.component] || b.component, b.tierText, b.source === 'lab' ? 'hand-run lab' : 'autolab'].filter(Boolean).join(' · ');
  const res =
    b.change != null
      ? `<p><span>Practice</span><b>${fewer(b.change)}</b>${b.expected != null ? ` · expected ${fewer(b.expected)}` : ''}${Math.abs(b.change) <= NOISE ? ' <i>(noise)</i>' : ''}</p>`
      : b.expected != null
        ? `<p><span>Expects</span>${fewer(b.expected)}</p>`
        : '';
  const state =
    b.outcome === 'live'
      ? `<p><span>Now</span><b>${esc(b.stateText)}</b> for ${span(b.ageSeconds)}</p>`
      : b.outcome === 'folded'
        ? `<p><span>Folded</span>${day(b.foldAt)}${b.foldN != null ? ` · the line went to <b>${b.foldN}</b>` : ''}</p>`
        : b.outcome === 'open'
          ? `<p><span>Open</span>${esc(b.stateText)}</p>`
          : `<p><span>${esc(sentence(b.state))}</span>${day(b.end)}</p>`;
  return `<h4>${sw(b.component)}${esc(b.id)}</h4><small>${esc(who)}</small>
    ${b.hypothesis ? `<q>${esc(b.hypothesis)}</q>` : ''}
    <p><span>Started</span>${day(b.start)}${b.base != null ? ` · line at ${b.base}` : ''}</p>${state}${res}
    ${b.reason ? `<p class="pg-why">${esc(sentence(b.reason))}</p>` : ''}`;
}

function fit(text, px) {
  const n = Math.floor(px / 6.3);
  return n >= text.length ? text : n >= 16 ? text.slice(0, n - 1) + '…' : '';
}

function draw(v, view, W) {
  tips = [];
  const s = v.score;
  const L = 48;
  const R = 112;
  const chartTop = 26;
  const chartH = 168;
  const chartBot = chartTop + chartH;
  const spineY = chartBot + 54;
  const LANE = 22;
  const laneY = (i) => spineY + 30 + i * LANE;
  const graphBot = laneY(Math.max(view.lanes, 1) - 1) + 30;
  const H = graphBot + 34;
  const x = xscale(view, L, W - R);
  const now = Date.parse(v.now);
  const xNow = x(now);
  const pts = view.spine;
  const out = [];

  // background: breaks, the hand-run era, boundaries
  for (const b of x.breaks)
    out.push(`<rect class="pg-break" x="${b.x0}" y="${chartTop - 8}" width="${b.x1 - b.x0}" height="${graphBot - chartTop + 8}"/>
      <text class="pg-break-t" x="${(b.x0 + b.x1) / 2}" y="${chartBot + 22}"><tspan x="${(b.x0 + b.x1) / 2}">${span(Math.round(b.ms / 1000)).replace(/ \d+ min$/, '')}</tspan><tspan x="${(b.x0 + b.x1) / 2}" dy="12">quiet</tspan></text>`);
  const auto = view.autolabStart ? x(view.autolabStart) : null;
  const firstLab = pts.find((p) => p.source === 'lab') || view.branches.find((b) => b.source === 'lab');
  if (firstLab) {
    const x1 = auto ?? xNow;
    out.push(`<rect class="pg-era" x="${L}" y="${spineY - 26}" width="${Math.max(0, x1 - L)}" height="${graphBot - spineY + 26}"/>
      <text class="pg-era-t" x="${L + 6}" y="${graphBot - 6}">Hand-run lab</text>`);
  }
  if (auto != null) out.push(`<line class="pg-auto" x1="${auto}" x2="${auto}" y1="${spineY - 26}" y2="${graphBot}"/><text class="pg-era-t pg-auto-t" x="${auto + 6}" y="${graphBot - 6}">Autolab</text>`);
  for (const b of view.boundaries) {
    const bx = x(b.at);
    out.push(`<g class="pg-bound" data-i="${tip(`<h4>New test set</h4><p>${esc(b.from)} → <b>${esc(b.to)}</b> on ${day(b.at)}</p><p class="pg-why">The line read ${b.before} on ${esc(b.from)} and ${b.after} on ${esc(b.to)}: counts on two test sets don’t compare, so the line is broken here.</p>`)}">
      <line x1="${bx}" x2="${bx}" y1="${chartTop - 14}" y2="${graphBot}"/><text x="${bx + 4}" y="${chartTop - 16}">${esc(b.to)}</text><rect class="hit" x="${bx - 6}" y="${chartTop - 26}" width="12" height="${graphBot - chartTop + 26}"/></g>`);
  }

  // score chart: practice wrong calls stacked by part, one step line per test set
  const ns = pts.map((p) => p.n);
  const top = Math.max(10, ...ns, s.teacher != null && view.spine.some((p) => p.testSet === s.testSet) ? s.teacher : 0);
  const stepY = top <= 30 ? 5 : top <= 60 ? 10 : top <= 150 ? 25 : 50;
  const ymax = Math.ceil((top * 1.06) / stepY) * stepY;
  const y = (n) => chartTop + chartH * (1 - n / ymax);
  for (let g = 0; g <= ymax; g += stepY * (ymax / stepY > 6 ? 2 : 1))
    out.push(`<line class="pg-grid" x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}"/><text class="pg-yl" x="${L - 8}" y="${y(g) + 4}" text-anchor="end">${g}</text>`);
  const segs = [];
  for (const p of pts) (segs.at(-1)?.ts === p.testSet ? segs.at(-1).pts : segs[segs.push({ ts: p.testSet, pts: [] }) - 1].pts).push(p);
  segs.forEach((sg, i) => {
    const end = i + 1 < segs.length ? x(segs[i + 1].pts[0].at) : xNow;
    const line = [];
    sg.pts.forEach((p, j) => {
      const xa = x(p.at);
      const xb = j + 1 < sg.pts.length ? x(sg.pts[j + 1].at) : end;
      if (PARTS.every((k) => p.parts[k] != null)) {
        let acc = 0;
        for (const k of PARTS) {
          out.push(`<rect class="pg-stack" style="--c:var(--${k})" x="${xa}" y="${y(acc + p.parts[k])}" width="${Math.max(0, xb - xa)}" height="${y(acc) - y(acc + p.parts[k])}"/>`);
          acc += p.parts[k];
        }
      }
      line.push(`${j ? `L${xa} ${y(sg.pts[j - 1].n)}` : `M${xa} ${y(p.n)}`}L${xa} ${y(p.n)}`);
      if (j + 1 === sg.pts.length) line.push(`L${xb} ${y(p.n)}`);
    });
    out.push(`<path class="pg-line" d="${line.join('')}"/>`);
  });
  const cur = segs.at(-1);
  if (s.teacher != null && cur?.ts === s.testSet) {
    const x0 = x(cur.pts[0].at);
    out.push(`<g data-i="${tip(`<h4>Big reference listener</h4><p>${s.teacher} wrong calls on the ${esc(s.testSet)} practice set with the big reference listener in place of the app’s.</p>`)}">
      <line class="pg-ref" x1="${x0}" x2="${W - R + 4}" y1="${y(s.teacher)}" y2="${y(s.teacher)}"/><text class="pg-ref-t" x="${W - R + 8}" y="${y(s.teacher) + 4}">Reference ${s.teacher}</text>
      <rect class="hit" x="${x0}" y="${y(s.teacher) - 5}" width="${W - R - x0}" height="10"/></g>`);
  }
  out.push(`<text class="pg-goal-t" x="${W - R + 8}" y="${y(s.goal.n) + 4}">Goal ${s.goal.n}</text>`);
  if (s.goal.n > 0) out.push(`<line class="pg-ref goal" x1="${L}" x2="${W - R + 4}" y1="${y(s.goal.n)}" y2="${y(s.goal.n)}"/>`);
  if (s.rate && s.rate.perDay > 0 && s.current && cur?.ts === s.testSet) {
    const n1 = Math.max(s.goal.n, s.current.wrongCalls - s.rate.perDay);
    const x1 = W - R + 52;
    out.push(`<g data-i="${tip(`<h4>Projection</h4><p>If the line kept removing about ${Math.round(s.rate.perDay)} wrong calls a day, it would be near <b>${Math.round(n1)}</b> a day from now. A straight line from the pace so far, not a forecast.</p>`)}">
      <path class="pg-proj" d="M${xNow} ${y(s.current.wrongCalls)}L${x1} ${y(n1)}"/><text class="pg-proj-t" x="${x1 - 2}" y="${y(n1) + 14}" text-anchor="end">+1 day ≈ ${Math.round(n1)}</text>
      <rect class="hit" x="${xNow}" y="${Math.min(y(s.current.wrongCalls), y(n1)) - 6}" width="${x1 - xNow}" height="${Math.abs(y(n1) - y(s.current.wrongCalls)) + 12}"/></g>`);
  }

  // the spine
  if (pts.length) {
    const x0 = x(pts[0].at);
    for (const w of x.windows) {
      const a = Math.max(x0, w.x0);
      const b = Math.min(xNow, w.x1);
      if (b > a) out.push(`<line class="pg-spine" x1="${a}" x2="${b}" y1="${spineY}" y2="${spineY}"/>`);
    }
    for (const b of x.breaks) if (b.x1 > x0 && b.x0 < xNow) out.push(`<line class="pg-spine gap" x1="${Math.max(b.x0, x0)}" x2="${b.x1}" y1="${spineY}" y2="${spineY}"/>`);
  }

  // branches, in lanes under the spine
  const r = 8;
  const labels = [];
  for (const b of view.branches) {
    const xs = x(b.start);
    const yl = laneY(b.lane);
    const xe = Math.max(x(b.end), xs + r + 3);
    const xf = b.outcome === 'folded' ? Math.max(x(b.foldAt), xs + 2 * r + 2) : null;
    const was = seen?.get(b.id);
    const anim = seen && seenScope === scope ? (!was ? ' enter' : b.outcome === 'folded' && !was.startsWith('folded') ? ' merging' : was !== `${b.outcome}|${b.state}` ? ' changed' : '') : '';
    const cls = `pg-br ${b.outcome} ${b.source}${b.waiting ? ' waiting' : ''}${anim}`;
    const body = `M${xs} ${spineY}V${yl - r}Q${xs} ${yl} ${xs + r} ${yl}H${xf != null ? xf - r : xe}`;
    const parts = [`<path class="br" pathLength="1" d="${body}"/>`];
    if (xf != null) parts.push(`<path class="br-merge" pathLength="1" d="M${xf - r} ${yl}Q${xf} ${yl} ${xf} ${yl - r}V${spineY}"/>`);
    if (b.source === 'autolab') for (const st of b.stages.slice(1, -1)) parts.push(`<line class="stage" x1="${x(st.at)}" x2="${x(st.at)}" y1="${yl - 3}" y2="${yl + 3}"/>`);
    parts.push(mark(b.source === 'lab' ? 'hollow' : TIER_MARK[b.tier] || 'circle', xs + r + 4, yl, 'tier'));
    if (b.outcome === 'ended') parts.push(b.state === 'discarded' ? `<path class="end" d="M${xe - 3.5} ${yl - 3.5}l7 7m0 -7l-7 7"/>` : `<path class="end" d="M${xe} ${yl - 5}v10"/>`);
    if (b.outcome === 'live') parts.push(`<circle class="pulse" cx="${xe}" cy="${yl}" r="4.5"/><circle class="tip" cx="${xe}" cy="${yl}" r="4.5"/>`);
    if (b.outcome === 'open') parts.push(`<circle class="tip" cx="${xe}" cy="${yl}" r="3.5"/>`);
    const room = (xf ?? xe) - xs - r - 14;
    const text = b.outcome === 'live' ? fit(`${b.id} · ${SHORT[b.state] || b.state} ${span(b.ageSeconds)}`, room) || fit(`${SHORT[b.state] || b.state} ${span(b.ageSeconds)}`, room) : fit(b.id, room);
    const lc = `pg-bl ${b.outcome} ${b.source}${b.waiting ? ' waiting' : ''}${anim}`;
    if (text) labels.push(`<text class="${lc}" x="${xs + r + 12}" y="${yl - 5}">${esc(text)}</text>`);
    else if (b.outcome === 'live') labels.push(`<text class="${lc} left" x="${xs - 6}" y="${yl + 4}" text-anchor="end">${esc(`${b.id} · ${SHORT[b.state] || b.state} ${span(b.ageSeconds)}`)}</text>`);
    const hit = `M${xs} ${spineY}V${yl}H${xf ?? xe}${xf != null ? `V${spineY}` : ''}`;
    out.push(`<g class="${cls}" style="--c:var(--${b.component || 'g-500'})" data-k="${esc(b.id)}" data-i="${tip(branchTip(b))}" tabindex="0" role="img" aria-label="${esc(`${b.id}: ${b.stateText}`)}">${parts.join('')}<path class="hit" d="${hit}"/></g>`);
  }
  out.push(...labels);

  // spine nodes on top, labelled with the count after them where there is room
  const placed = [];
  const order = pts.map((p, i) => i).sort((a, b) => (pts[b].kind === 'fold') - (pts[a].kind === 'fold') || b - a);
  const show = new Set(order.filter((i) => {
    const px = x(pts[i].at);
    if (placed.some((q) => Math.abs(q - px) < 22)) return false;
    placed.push(px);
    return true;
  }));
  pts.forEach((p, i) => {
    const px = x(p.at);
    const isNew = seen && seenScope === scope && p.experiment && seen.get(p.experiment) && !seen.get(p.experiment).startsWith('folded');
    const what = p.experiment ? `Fold of <b>${esc(p.experiment)}</b>` : esc(sentence(p.note) || 'Main-line score');
    const parts = p.parts && PARTS.every((k) => p.parts[k] != null) ? `<p>${PARTS.map((k) => `${sw(k)}${p.parts[k]}`).join(' ')}</p>` : '';
    out.push(`<g class="pg-node ${p.kind} ${p.source}${isNew ? ' pop' : ''}" data-i="${tip(`<h4>${p.n} wrong calls</h4><small>${esc(p.testSet)} · ${day(p.at)} · <code>${esc(p.sha)}</code></small><p>${what}</p>${parts}`)}">
      <circle cx="${px}" cy="${spineY}" r="${p.kind === 'fold' ? 6 : 3.5}"/>${show.has(i) ? `<text x="${px}" y="${spineY - 11}" text-anchor="middle">${p.n}</text>` : ''}<circle class="hit" cx="${px}" cy="${spineY}" r="10"/></g>`);
  });

  // now, and the time axis
  out.push(`<line class="pg-now" x1="${xNow}" x2="${xNow}" y1="${chartTop - 8}" y2="${graphBot}"/><text class="pg-now-t" x="${xNow}" y="${chartTop - 14}" text-anchor="middle">now</text>`);
  let lastDay = null;
  for (const t of ticks(x)) {
    const d = dayOnly(t.t);
    const label = t.first || d !== lastDay ? `${d} ${clock(t.t)}` : clock(t.t);
    lastDay = d;
    out.push(`<line class="pg-axt" x1="${t.x}" x2="${t.x}" y1="${graphBot + 4}" y2="${graphBot + 9}"/><text class="pg-xl" x="${t.x}" y="${graphBot + 22}">${label}</text>`);
  }
  return `<svg class="pg-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="group" aria-label="Main line and experiments over time">${out.join('')}</svg>`;
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
  const kind = (e) => (e.to === 'folded' ? 'better' : ['discarded', 'abandoned', 'superseded'].includes(e.to) ? 'worse' : e.to === 'needs-human' ? 'wait' : '');
  const h = v.heartbeat;
  return `<div class="pg-tick-h"><h2>Live</h2><span class="muted small">${h ? `${h.stale ? 'Supervisor silent' : 'Supervisor checked in'} ${span(h.ageSeconds)} ago` : 'Supervisor not started'} · refreshes every 10 s</span></div>
    ${rows.length ? `<ol>${rows
      .map((e) => `<li class="${fresh.has(e.key) ? 'new ' : ''}${kind(e)}" title="${esc(e.note)}"><time>${clock(e.at)}</time>
          <span>${e.id ? `${sw(e.component)}<b>${esc(e.id)}</b> ${esc(e.toText)}` : esc(sentence(e.note || e.toText))}</span></li>`)
      .join('')}</ol>` : '<p class="muted small">Nothing has happened yet.</p>'}`;
}

function key() {
  const k = (c, t) => `<span>${c}${t}</span>`;
  const svg = (inner) => `<svg width="18" height="12" viewBox="0 0 18 12">${inner}</svg>`;
  return `<div class="axis-key pg-key">
    ${PARTS.map((p) => k(sw(p), NAMES[p])).join('')}
    ${k(svg('<circle cx="9" cy="6" r="4" fill="var(--g-700)"/>'), 'Quick fix')}${k(svg('<path d="M9 1L14 6L9 11L4 6Z" fill="var(--g-700)"/>'), 'New idea')}${k(svg('<rect x="5" y="2" width="8" height="8" rx="1" fill="var(--g-700)"/>'), 'Big bet')}
    ${k(svg('<circle cx="9" cy="6" r="5" fill="var(--better)"/>'), 'Folded in')}${k(svg('<path d="M5.5 2.5l7 7m0-7l-7 7" stroke="var(--g-500)" stroke-width="1.75" fill="none"/>'), 'Discarded')}
    ${k(svg('<circle cx="9" cy="6" r="4.5" fill="var(--gold-400)"/>'), 'Waiting for you')}${k(svg('<path d="M1 6h16" stroke="var(--g-400)" stroke-width="1.5" stroke-dasharray="3 2"/>'), 'Open, hand-run')}</div>`;
}

function section(v) {
  const seg = [['autolab', 'Autolab only'], ['testSet', `Since ${v.testSet || 'the test set'}`], ['all', 'Everything']]
    .map(([k, t]) => `<button data-scope="${k}" aria-pressed="${k === scope}">${esc(t)}</button>`)
    .join('');
  return `<section aria-labelledby="g-h"><div class="sec-h"><h2 id="g-h">Branches and folds</h2><div class="seg" role="group" aria-label="How far back">${seg}</div></div>
    <p class="sec-p">The main line runs left to right with the practice count after each score; each experiment leaves it when it starts. A green join is a fold;
      a faded branch was let go (point at it for why); branches ending at now are still running. Quiet stretches are squeezed into breaks.
      <span id="pg-counts"></span></p>
    ${key()}<div class="pg-fig" id="pg-fig"><div class="pg-tip" id="pg-tip" hidden></div><div id="pg-svg"></div></div></section>`;
}

const counts = (c) => `${c.live} running, ${c.folded} folded, ${c.ended} let go${c.open ? `, ${c.open} still open in the hand-run lab` : ''}.`;

// Render and refresh

function width() {
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
  topbar(v);
  if (!$('pg-top'))
    $('main').innerHTML = `<div id="pg-notice"></div><div class="pg-top" id="pg-top"><div id="pg-hero"></div><aside class="pg-tick" id="pg-tick" aria-live="polite"></aside></div><div id="pg-sec"></div>
      <p class="pg-back"><a href="/autolab${root ? `?${qs({ root })}` : ''}">${icon('right')}Running now, verdicts, portfolio and spend</a></p>`;
  $('pg-notice').innerHTML = staleNotice(v.heartbeat) + (v.heartbeat?.paused ? `<div class="notice warn"><b>Paused:</b> ${esc(v.heartbeat.paused)}. Running work finishes; nothing new starts.</div>` : '');
  $('pg-hero').innerHTML = hero(v);
  $('pg-tick').innerHTML = ticker(v);
  if (!$('pg-fig') || seenScope !== scope || last?.testSet !== v.testSet) $('pg-sec').innerHTML = section(v);
  $('pg-counts').textContent = counts(view.counts);
  $('pg-svg').innerHTML = draw(v, view, width());
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

document.addEventListener('pointermove', (e) => {
  const fig = $('pg-fig');
  if (!fig) return;
  const el = e.target.closest?.('[data-i]');
  if (!el || !fig.contains(el)) {
    if (!$('pg-tip').hidden && !e.target.closest?.('.pg-tip')) $('pg-tip').hidden = true;
    return;
  }
  const r = fig.getBoundingClientRect();
  showTip(el, e.clientX - r.left + fig.scrollLeft, e.clientY - r.top);
});
document.addEventListener('focusin', (e) => {
  const el = e.target.closest?.('#pg-fig [data-i]');
  if (!el) return;
  const fig = $('pg-fig');
  const a = el.getBoundingClientRect();
  const r = fig.getBoundingClientRect();
  showTip(el, a.left - r.left + fig.scrollLeft + 24, a.bottom - r.top - 8);
});
document.addEventListener('focusout', (e) => e.target.closest?.('#pg-fig [data-i]') && ($('pg-tip').hidden = true));
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

load();
setInterval(() => document.hidden || load(), POLL_MS);
document.addEventListener('visibilitychange', () => document.hidden || load());
