'use strict';
// Mirrors gap/moves/verdict/parentOf/biggestLever in packages/core/src/bench/scoreboard.ts; gates come from scoreboard.json.
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const plain = s => String(s ?? '').replaceAll('lead-out recitation', 'clean recitation after the test clip').replaceAll('lead-in recitation', 'clean recitation before the test clip').replaceAll('lead-outs', 'added endings').replaceAll('lead-ins', 'added beginnings').replaceAll('lead-in', 'surrounding recitation').replaceAll('lead-out', 'recitation after the test clip').replaceAll('stitched sessions', 'combined recordings');
const num = m => typeof m?.value === 'number' && Number.isFinite(m.value);
const metric = (r,id) => r?.metrics.find(m => m.id === id);
const fmt = (v,u) => v == null ? '—' : u === '%' ? `${(v*100).toFixed(1)}%` : u === 's' ? `${v.toFixed(2)} s` : String(+v.toFixed(3));
const show = m => m ? (m.display ?? (num(m) ? fmt(m.value,m.unit) : m.status === 'pending' ? 'not run' : '—')) : '—';
const barText = m => m?.bar ? `${m.bar.op === '>=' ? '≥' : '≤'} ${fmt(m.bar.target,m.unit)}` : '';
const gap = m => { if(!m?.bar || !num(m)) return Infinity; const s = m.bar.op === '>=' ? m.bar.target - m.value : m.value - m.bar.target; return s <= 0 ? 0 : s / (m.bar.target || 1); };
const counts = m => { const x = m?.display?.match(/\((\d+)\/(\d+)\)/); return x ? {hit:+x[1], of:+x[2]} : null; };
const lowerBetter = m => m?.bar ? m.bar.op === '<=' : /PER|false|jump|lost|rtf|latency|time/i.test(`${m?.id} ${m?.name}`);
const VERDICT = {better:'↑ Improved',worse:'↓ Regressed',mixed:'Mixed results',noise:'Small change',unchanged:'No release check changed'};

function move(a,b,flipOnly){
 if(!num(a) || !num(b) || Math.abs(b.value-a.value) < 1e-9) return null;
 if(flipOnly && a.status === b.status) return null;
 const ca = counts(a), cb = counts(b), d = b.value - a.value;
 const small = ca && cb ? Math.abs(cb.hit-ca.hit) <= 2 && Math.abs((cb.of-cb.hit)-(ca.of-ca.hit)) <= 2 : Math.abs(d)/Math.max(Math.abs(a.value),1e-9) < .03;
 return {from:a, to:b, better:lowerBetter(b) ? d < 0 : d > 0, small};
}
const moves = (from,to,gates) => gates.map(g => ({g, mv: move(metric(from,g.id), metric(to,g.id), g.flipOnly)})).filter(x => x.mv);
function verdict(ms){
 if(!ms.length) return 'unchanged';
 const big = ms.filter(x => !x.mv.small); if(!big.length) return 'noise';
 const up = big.some(x => x.mv.better), down = big.some(x => !x.mv.better);
 return up && down ? 'mixed' : up ? 'better' : 'worse';
}
function parentOf(rows,i,cand){
 for(let j=i-1;j>=0;j--) if(rows[j].listener === rows[i].listener) return rows[j];
 for(let j=i-1;j>=0;j--) if(rows[j].listener === cand) return rows[j];
}
function lever(row){
 const e0 = [['evidence','Listener'],['tracker','Tracker'],['checker','Checker']].map(([k,label]) => ({label, m:metric(row,`checker.e0.${k}`)}));
 if(!e0.every(x => num(x.m))) return null;
 const of = e0.reduce((n,x) => n + x.m.value, 0); return of ? {e0, of} : null;
}
const delta = (a,b) => {
 const mv = move(a,b,false); if(!mv) return num(a) && num(b) ? '<span class="small">Unchanged</span>' : '';
 const d = b.value - a.value, n = b.unit === '%' ? `${(d*100).toFixed(1)} points` : fmt(d,b.unit);
 if(parseFloat(n) === 0) return '<span class="small">Almost unchanged</span>';
 return `<span title="${esc(b.unit === '%' ? 'Change in percentage points' : 'Change in the measured value')}" class="mv ${mv.better ? 'up' : 'down'} ${mv.small ? 'small-move' : ''}">${mv.better ? '▲' : '▼'} ${d > 0 ? '+' : ''}${n}</span>`;
};
const label = r => `${r.listener} @ ${r.sha.slice(0,7)} · ${r.date}`;

let board, rows, gates, blocked, cand;

const CHECKS = {
 'e2e.leadin.recall': ['Mistakes caught', 'Automatic passage recognition', 'How many real mistakes the engine detects, with clean recitation before and after the test clip.'],
 'e2e.leadin.precision': ['Warnings you can trust', 'Automatic passage recognition', 'How many warnings point to a real mistake, rather than a false alarm.'],
 'checker.oracleLock.recall': ['Mistakes caught', 'Passage provided', 'How many real mistakes the engine detects when it is given the passage.'],
 'checker.oracleLock.precision': ['Warnings you can trust', 'Passage provided', 'How many warnings are correct when the engine is given the passage.'],
 'e2e.cleanPer5': ['False alarms in test clips', 'Per 5 minutes of clean recitation', 'Warnings raised on clean short recordings, normalized to five minutes.'],
 'e2e.stitched.falseWords': ['False alarms in longer sessions', 'Per 5 minutes of combined recordings', 'Longer recordings test whether false alarms build up over time.'],
 'e2e.repeatFp': ['False alarms on repeated words', 'Repeating is allowed', 'A repeated word should not be treated as a mistake.'],
 'e2e.leadin.leadWarnings': ['False alarms before the test clip', 'Per 5 minutes of added clean recitation', 'These warnings happened in the clean recitation added before the original test clip.'],
 'e2e.leadin.followWarnings': ['False alarms after the test clip', 'Per 5 minutes of added clean recitation', 'These warnings happened in the clean recitation added after the original test clip.'],
 'listener.goldPer': ['Sound recognition errors', 'Audio recognition', 'The share of speech sounds the listener gets wrong. Lower is better.'],
 'tracker.clean.within5s': ['Passage found within 5 seconds', 'Passage recognition', 'How often the engine recognizes the correct ayah within five seconds.'],
 'tracker.clean.falseJumps': ['Unexpected passage changes', 'Passage recognition', 'How often the engine incorrectly jumps to another passage.'],
 'e2e.perfRtf': ['Processing speed', 'Compute time ÷ recording time', 'A value of 0.05 means five seconds of processing for every 100 seconds of audio. Lower is better.'],
};
function readiness(){
 const mine = rows.filter(r => r.listener === cand), last = mine.at(-1), prev = mine.at(-2);
 if(!last) return `<div class="notice">No runs for candidate ${esc(cand)}.</div>`;
 const ranked = gates.map(g => ({g, m:metric(last,g.id)})).sort((a,b) => (Number(!num(a.m)) - Number(!num(b.m))) || gap(b.m) - gap(a.m));
 const failing = ranked.filter(x => x.m?.status === 'fail').length, open = ranked.filter(x => !num(x.m)).length;
 const unmeasured = blocked.filter(b => !last.metrics.some(m => (m.id === b.id || m.id.startsWith(b.id + '.')) && num(m)));
 const ready = !failing && !open && !unmeasured.length;
 const summary = ready ? 'All release checks meet their targets.' : !failing && !open ? `Measured checks pass. Evidence is still missing in ${unmeasured.length} areas.` : `${failing ? `${failing} ${failing === 1 ? 'check needs' : 'checks need'} improvement` : 'Measured checks pass'}${open ? `${failing ? ' and' : ', but'} ${open} still ${open === 1 ? 'needs' : 'need'} to be measured` : ''}. Explore the results below, or listen to the recordings.`;
 const passing = ranked.filter(x => num(x.m) && x.m.status === 'pass');
 const attention = ranked.filter(x => !num(x.m) || x.m.status !== 'pass');
 const card = ({g,m}) => {
  const [title, context, help] = CHECKS[g.id] || [plain(g.label),g.area,''];
  const pass = m?.status === 'pass', available = num(m), review = window.tilawaReview?.hasPreset(g.id);
  const previous = metric(prev,g.id);
  const reading = m?.unit === '%' ? fmt(m.value,m.unit) : m?.unit === 's' ? fmt(m.value,m.unit) : String(+m?.value?.toFixed(g.id === 'e2e.perfRtf' ? 3 : 2));
  return `<article class="check-card ${pass ? 'is-pass' : available ? 'is-fail' : 'is-pending'}"><div class="check-top"><span class="check-context">${esc(context)}</span><span class="check-status">${pass ? '✓ Passing' : available ? 'Needs improvement' : 'Not measured'}</span></div><h3>${esc(title)}</h3><div class="check-reading"><strong>${available ? esc(reading) : '—'}</strong><span>${m?.bar ? `Target: ${m.bar.op === '>=' ? 'at least' : 'at most'} ${esc(fmt(m.bar.target,m.unit))}` : 'Awaiting results'}</span></div><p>${esc(help)}</p><div class="check-bottom"><span class="check-movement">${available && num(previous) ? `${delta(previous,m)} <span class="small">since last run</span>` : '<span class="small">No earlier measurement</span>'}</span>${review ? `<button class="text-button" data-gate="${esc(g.id)}">Listen to examples <span aria-hidden="true">↗</span></button>` : ''}</div><details class="check-evidence"><summary>Measurement details</summary><p>${esc(plain(show(m)))}</p></details></article>`;
 };
 const lv = lever(last), stageNames = {Listener:'Recognizing the sounds', Tracker:'Finding the passage', Checker:'Checking for mistakes'};
 return `<section class="readiness" aria-labelledby="readiness-title"><div class="release-summary"><div class="release-copy"><span class="eyebrow">RELEASE READINESS</span><h2 id="readiness-title">${ready ? 'Ready for the next step.' : 'A few things to improve.'}</h2><p>${esc(summary)}</p><div class="release-meta"><span class="release-state ${ready ? 'pass' : 'fail'}">${ready ? '✓ Ready to release' : 'Not ready to release'}</span><span>${esc(last.date)} · <code>${esc(last.sha.slice(0,7))}</code></span></div></div><div class="release-progress"><div class="progress-total"><strong>${passing.length}</strong><span>of ${gates.length}<br>checks passing</span></div><div class="progress-segments" aria-label="${passing.length} of ${gates.length} checks passing">${ranked.map(({m}) => `<span class="${m?.status === 'pass' ? 'passed' : num(m) ? 'failed' : 'pending'}"></span>`).join('')}</div><a class="primary-link" href="#review">Listen & review <span aria-hidden="true">↓</span></a></div></div><div class="section-heading"><h2>What needs attention</h2><span class="small">Latest recorded results</span></div><div class="check-grid">${attention.map(card).join('') || '<p class="empty">All checks are passing.</p>'}</div><details class="passed-checks"><summary><span>✓ ${passing.length} checks are passing</span><span class="small">View results</span></summary><div class="check-grid">${passing.map(card).join('')}</div></details>${unmeasured.length ? `<details class="coverage"><summary><span>More evidence needed before release</span><span class="count-pill">${unmeasured.length} areas</span></summary><p>These areas also block release. Recordings or measurements are still missing.</p><ul>${unmeasured.map(b => `<li>${esc(plain(b.label))}</li>`).join('')}</ul></details>` : ''}${lv ? `<details class="coverage"><summary>Where are mistakes being missed?</summary><p>For test clips with deliberately added mistakes, these are the stages responsible for misses in automatic passage recognition.</p><div class="loss-stages">${lv.e0.map(x => `<div><span>${stageNames[x.label]}</span><strong>${x.m.value} <small>misses</small></strong><div class="track"><div class="fill" style="width:${x.m.value/lv.of*100}%"></div></div></div>`).join('')}</div></details>` : ''}</section>`;
}

function experiments(){
 const listeners = [...new Set(rows.map(r => r.listener))];
 return `<section class="panel"><div class="section-top"><div><h2>Experiments</h2><p>Each experiment is compared with its previous run. For a new audio model, the comparison uses the release candidate. Changes affecting two clips or fewer are shown as small changes.</p></div><div class="filters"><select id="exp-listener" aria-label="Audio model"><option value="">All audio models</option>${listeners.map(l => `<option>${esc(l)}</option>`).join('')}</select><select id="exp-verdict" aria-label="Experiment outcome"><option value="">All outcomes</option>${Object.entries(VERDICT).map(([k,v]) => `<option value="${k}">${v}</option>`).join('')}</select></div></div><div id="exp-list"></div></section>`;
}
function renderExperiments(){
 const l = document.getElementById('exp-listener').value, v = document.getElementById('exp-verdict').value;
 const items = rows.map((r,i) => { const p = parentOf(rows,i,cand), ms = p ? moves(p,r,gates) : []; return {r, i, p, ms, vd: p ? verdict(ms) : 'first'}; })
  .filter(x => (!l || x.r.listener === l) && (!v || x.vd === v)).reverse();
 document.getElementById('exp-list').innerHTML = items.map(({r,i,p,ms,vd}) => `<article class="exp ${vd}"><div class="exp-head"><span class="vd">${vd === 'first' ? 'First run' : VERDICT[vd]}</span><b>${esc(r.listener)}</b><code>${esc(r.sha.slice(0,7))}</code><span class="small">${esc(r.date)}${p ? ` · vs ${p.listener === r.listener ? '' : esc(p.listener) + ' '}${esc(p.sha.slice(0,7))}` : ''}</span>${(r.modules || []).map(m => `<span class="tag">${esc(m)}</span>`).join('')}${p ? `<button data-a="${rows.indexOf(p)}" data-b="${i}">compare</button>` : ''}</div><p class="note">${esc(r.note || 'No note.')}</p>${ms.length ? `<ul class="moves">${ms.map(({g,mv}) => `<li class="mv ${mv.better ? 'up' : 'down'} ${mv.small ? 'small-move' : ''}">${mv.better ? '▲' : '▼'} ${esc(plain(g.label))}: ${esc(show(mv.from))} → <b>${esc(show(mv.to))}</b></li>`).join('')}</ul>` : ''}</article>`).join('') || '<div class="empty">No runs match.</div>';
 document.querySelectorAll('#exp-list button').forEach(b => b.addEventListener('click', () => { document.getElementById('cmp-a').value = b.dataset.a; document.getElementById('cmp-b').value = b.dataset.b; renderCompare(); document.getElementById('compare').scrollIntoView({behavior:'smooth'}); }));
}

function compare(){
 const opts = rows.map((r,i) => `<option value="${i}">${esc(label(r))}</option>`).reverse().join('');
 return `<section class="panel" id="compare"><div class="section-top"><div><h2>Compare two runs</h2><p>See which measurements changed. Release checks are marked ●.</p></div><div class="filters"><select id="cmp-a" aria-label="Earlier run">${opts}</select><span>→</span><select id="cmp-b" aria-label="Later run">${opts}</select><label class="small"><input type="checkbox" id="cmp-all"> show unchanged</label></div></div><div id="cmp-out"></div></section>`;
}
function renderCompare(){
 const a = rows[+document.getElementById('cmp-a').value], b = rows[+document.getElementById('cmp-b').value], all = document.getElementById('cmp-all').checked;
 const gateIds = new Set(gates.map(g => g.id));
 const list = b.metrics.map(m => { const o = metric(a,m.id), mv = move(o,m,false), rel = mv ? Math.abs(m.value - o.value) / Math.max(Math.abs(o.value), .01) : -1; return {m, o, mv, rel}; })
  .filter(x => all || x.mv || (x.o && x.o.status !== x.m.status)).sort((x,y) => (gateIds.has(y.m.id) - gateIds.has(x.m.id)) || ((y.mv && !y.mv.small) - (x.mv && !x.mv.small)) || y.rel - x.rel);
 const v = verdict(moves(a,b,gates));
 document.getElementById('cmp-out').innerHTML = `<p><span class="vd ${v}">${VERDICT[v]}</span> on release checks · ${list.filter(x => x.mv?.better).length} metrics better, ${list.filter(x => x.mv && !x.mv.better).length} worse</p><div class="table-wrap"><table class="gates"><thead><tr><th>Metric</th><th>${esc(label(a))}</th><th>${esc(label(b))}</th><th>Change</th><th>Target</th></tr></thead><tbody>${list.map(({m,o}) => `<tr><td>${gateIds.has(m.id) ? '● ' : ''}<span class="area">${esc(m.group)}</span> ${esc(plain(m.name))}</td><td>${esc(show(o))}</td><td><b>${esc(plain(show(m)))}</b></td><td>${delta(o,m)}</td><td>${esc(barText(m))} ${m.status === 'pass' ? '✓' : m.status === 'fail' ? '✗' : ''}</td></tr>`).join('') || '<tr><td colspan="5">No differences.</td></tr>'}</tbody></table></div>`;
}

function render(){
 const rv = window.tilawaReview;
 document.getElementById('content').innerHTML = readiness() + (rv ? rv.panel() : '') + '<details class="details"><summary>Experiment history & detailed comparisons</summary>' + experiments() + compare() + '</details>';
 if(rv){
  document.querySelectorAll('[data-gate]').forEach(tr => { if(!rv.hasPreset(tr.dataset.gate)) return; tr.addEventListener('click', () => rv.focusGate(tr.dataset.gate)); });
  rv.init(cand);
 }
 const mine = rows.map((r,i) => i).filter(i => rows[i].listener === cand);
 document.getElementById('cmp-b').value = mine.at(-1) ?? rows.length - 1;
 document.getElementById('cmp-a').value = mine.at(-2) ?? Math.max(0, rows.length - 2);
 ['exp-listener','exp-verdict'].forEach(id => document.getElementById(id).addEventListener('change', renderExperiments));
 ['cmp-a','cmp-b','cmp-all'].forEach(id => document.getElementById(id).addEventListener('change', renderCompare));
 renderExperiments(); renderCompare();
}

async function init(){
 try{
  const res = await fetch('scoreboard.json',{cache:'no-store'}); if(!res.ok) throw new Error(`HTTP ${res.status}`);
  board = await res.json(); rows = board.rows; gates = board.gates || []; blocked = board.blocked || [];
  if(!rows?.length || !gates.length) throw new Error('No runs or gates; run pnpm scoreboard --render');
  const sel = document.getElementById('candidate');
  sel.innerHTML = [...new Set(rows.map(r => r.listener))].map(l => `<option ${l === board.candidate ? 'selected' : ''}>${esc(l)}</option>`).join('');
  cand = sel.value; sel.addEventListener('change', () => { cand = sel.value; render(); });
  render();
 }catch(e){ document.getElementById('content').innerHTML = `<div class="notice"><b>Couldn’t load the scoreboard.</b> ${esc(e.message)}. Serve with <code>uv run tools/scoreboard_serve.py</code>.</div>`; }
}
init();
