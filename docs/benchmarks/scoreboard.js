'use strict';
// Mirrors gap/moves/verdict/parentOf/biggestLever in packages/core/src/bench/scoreboard.ts; gates come from scoreboard.json.
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = m => typeof m?.value === 'number' && Number.isFinite(m.value);
const metric = (r,id) => r?.metrics.find(m => m.id === id);
const fmt = (v,u) => v == null ? '—' : u === '%' ? `${(v*100).toFixed(1)}%` : u === 's' ? `${v.toFixed(2)} s` : String(+v.toFixed(3));
const show = m => m ? (m.display ?? (num(m) ? fmt(m.value,m.unit) : m.status === 'pending' ? 'not run' : '—')) : '—';
const barText = m => m?.bar ? `${m.bar.op === '>=' ? '≥' : '≤'} ${fmt(m.bar.target,m.unit)}` : '';
const gap = m => { if(!m?.bar || !num(m)) return Infinity; const s = m.bar.op === '>=' ? m.bar.target - m.value : m.value - m.bar.target; return s <= 0 ? 0 : s / (m.bar.target || 1); };
const counts = m => { const x = m?.display?.match(/\((\d+)\/(\d+)\)/); return x ? {hit:+x[1], of:+x[2]} : null; };
const lowerBetter = m => m?.bar ? m.bar.op === '<=' : /PER|false|jump|lost|rtf|latency|time/i.test(`${m?.id} ${m?.name}`);
const VERDICT = {better:'▲ Better',worse:'▼ Worse',mixed:'◆ Mixed',noise:'· Noise',unchanged:'= No gate moved'};

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
 const mv = move(a,b,false); if(!mv) return num(a) && num(b) ? '<span class="small">=</span>' : '';
 const d = b.value - a.value, n = b.unit === '%' ? `${(d*100).toFixed(1)} pp` : fmt(d,b.unit);
 return `<span class="mv ${mv.better ? 'up' : 'down'} ${mv.small ? 'small-move' : ''}">${mv.better ? '▲' : '▼'} ${d > 0 ? '+' : ''}${n}</span>`;
};
const label = r => `${r.listener} @ ${r.sha.slice(0,7)} · ${r.date}`;

let board, rows, gates, blocked, cand;

function readiness(){
 const mine = rows.filter(r => r.listener === cand), last = mine.at(-1), prev = mine.at(-2);
 if(!last) return `<div class="notice">No runs for candidate ${esc(cand)}.</div>`;
 const ranked = gates.map(g => ({g, m:metric(last,g.id)})).sort((a,b) => gap(b.m) - gap(a.m));
 const failing = ranked.filter(x => x.m?.status === 'fail').length, open = ranked.filter(x => !num(x.m)).length;
 const unmeasured = blocked.filter(b => !last.metrics.some(m => (m.id === b.id || m.id.startsWith(b.id + '.')) && num(m)));
 const ready = !failing && !open && !unmeasured.length;
 const lv = lever(last);
 const trend = id => { const pts = mine.map(r => metric(r,id)).filter(num); if(pts.length < 2) return ''; const v = pts.map(m => m.value), lo = Math.min(...v), hi = Math.max(...v), W = 80, H = 20; return `<svg class="spark" viewBox="0 0 ${W} ${H}"><polyline fill="none" stroke="currentColor" stroke-width="1.5" points="${v.map((x,i) => `${i*W/(v.length-1)},${hi === lo ? H/2 : H - (x-lo)/(hi-lo)*H}`).join(' ')}"/></svg>`; };
 return `<section class="panel"><div class="verdict ${ready ? 'pass' : 'fail'}"><div><h2>${ready ? 'Ready to release' : 'Not ready to release'}</h2><div class="small">${esc(label(last))}</div></div><div class="tally"><b>${gates.length - failing - open}</b>/${gates.length} gates pass · <b>${failing}</b> fail · <b>${open}</b> not run · <b>${unmeasured.length}</b> unmeasurable</div></div>
 ${lv ? `<div class="lever"><div><b>Biggest lever on recall:</b> ${esc(lv.e0.reduce((a,b) => b.m.value > a.m.value ? b : a).label)} · planted mistakes missed in free recitation, by the stage that lost them</div><div class="stack">${lv.e0.map(x => `<span style="flex:${x.m.value}" title="${esc(x.label)}: ${x.m.value}/${lv.of}">${esc(x.label)} ${x.m.value}</span>`).join('')}</div></div>` : ''}
 <table class="gates"><thead><tr><th>Gate (worst first)</th><th>Now</th><th>Bar</th><th>Shortfall</th><th>vs previous run</th><th>Trend</th></tr></thead><tbody>${ranked.map(({g,m}) => { const s = gap(m); return `<tr class="${s === 0 ? 'ok' : s === Infinity ? 'open' : 'bad'}"><td><span class="area">${esc(g.area)}</span> ${esc(g.label)}</td><td>${esc(show(m))}</td><td>${esc(barText(m))}</td><td>${s === Infinity ? (m?.status === 'pending' ? 'not run' : 'no data') : s === 0 ? '✓' : `<div class="short"><i style="width:${Math.min(100,s*200)}%"></i></div>${(s*100).toFixed(0)}% short`}</td><td>${delta(metric(prev,g.id), m)}</td><td>${trend(g.id)}</td></tr>`; }).join('')}</tbody></table>
 ${unmeasured.length ? `<p class="small"><b>Also blocks release, no data yet:</b> ${unmeasured.map(b => esc(b.label)).join(' · ')}</p>` : ''}</section>`;
}

function experiments(){
 const listeners = [...new Set(rows.map(r => r.listener))];
 return `<section class="panel"><div class="section-top"><div><h2>Experiments</h2><p>Each run against its parent, the previous run of the same listener, or the candidate for a listener's first run. Only gates that moved are listed. <span class="small-move">Faded</span> moves are two clips or fewer and count as noise.</p></div><div class="filters"><select id="exp-listener"><option value="">All listeners</option>${listeners.map(l => `<option>${esc(l)}</option>`).join('')}</select><select id="exp-verdict"><option value="">Any verdict</option>${Object.entries(VERDICT).map(([k,v]) => `<option value="${k}">${v}</option>`).join('')}</select></div></div><div id="exp-list"></div></section>`;
}
function renderExperiments(){
 const l = document.getElementById('exp-listener').value, v = document.getElementById('exp-verdict').value;
 const items = rows.map((r,i) => { const p = parentOf(rows,i,cand), ms = p ? moves(p,r,gates) : []; return {r, i, p, ms, vd: p ? verdict(ms) : 'first'}; })
  .filter(x => (!l || x.r.listener === l) && (!v || x.vd === v)).reverse();
 document.getElementById('exp-list').innerHTML = items.map(({r,i,p,ms,vd}) => `<article class="exp ${vd}"><div class="exp-head"><span class="vd">${vd === 'first' ? 'First run' : VERDICT[vd]}</span><b>${esc(r.listener)}</b><code>${esc(r.sha.slice(0,7))}</code><span class="small">${esc(r.date)}${p ? ` · vs ${p.listener === r.listener ? '' : esc(p.listener) + ' '}${esc(p.sha.slice(0,7))}` : ''}</span>${(r.modules || []).map(m => `<span class="tag">${esc(m)}</span>`).join('')}${p ? `<button data-a="${rows.indexOf(p)}" data-b="${i}">compare</button>` : ''}</div><p class="note">${esc(r.note || 'No note.')}</p>${ms.length ? `<ul class="moves">${ms.map(({g,mv}) => `<li class="mv ${mv.better ? 'up' : 'down'} ${mv.small ? 'small-move' : ''}">${mv.better ? '▲' : '▼'} ${esc(g.label)}: ${esc(show(mv.from))} → <b>${esc(show(mv.to))}</b></li>`).join('')}</ul>` : ''}</article>`).join('') || '<div class="empty">No runs match.</div>';
 document.querySelectorAll('#exp-list button').forEach(b => b.addEventListener('click', () => { document.getElementById('cmp-a').value = b.dataset.a; document.getElementById('cmp-b').value = b.dataset.b; renderCompare(); document.getElementById('compare').scrollIntoView({behavior:'smooth'}); }));
}

function compare(){
 const opts = rows.map((r,i) => `<option value="${i}">${esc(label(r))}</option>`).reverse().join('');
 return `<section class="panel" id="compare"><div class="section-top"><div><h2>Compare two runs</h2><p>Every metric that differs, biggest moves first. Gates are marked ●.</p></div><div class="filters"><select id="cmp-a">${opts}</select><span>→</span><select id="cmp-b">${opts}</select><label class="small"><input type="checkbox" id="cmp-all"> show unchanged</label></div></div><div id="cmp-out"></div></section>`;
}
function renderCompare(){
 const a = rows[+document.getElementById('cmp-a').value], b = rows[+document.getElementById('cmp-b').value], all = document.getElementById('cmp-all').checked;
 const gateIds = new Set(gates.map(g => g.id));
 const list = b.metrics.map(m => { const o = metric(a,m.id), mv = move(o,m,false), rel = mv ? Math.abs(m.value - o.value) / Math.max(Math.abs(o.value), .01) : -1; return {m, o, mv, rel}; })
  .filter(x => all || x.mv || (x.o && x.o.status !== x.m.status)).sort((x,y) => (gateIds.has(y.m.id) - gateIds.has(x.m.id)) || ((y.mv && !y.mv.small) - (x.mv && !x.mv.small)) || y.rel - x.rel);
 const v = verdict(moves(a,b,gates));
 document.getElementById('cmp-out').innerHTML = `<p><span class="vd ${v}">${VERDICT[v]}</span> on gates · ${list.filter(x => x.mv?.better).length} metrics better, ${list.filter(x => x.mv && !x.mv.better).length} worse</p><table class="gates"><thead><tr><th>Metric</th><th>${esc(label(a))}</th><th>${esc(label(b))}</th><th>Change</th><th>Bar</th></tr></thead><tbody>${list.map(({m,o}) => `<tr><td>${gateIds.has(m.id) ? '● ' : ''}<span class="area">${esc(m.group)}</span> ${esc(m.name)}</td><td>${esc(show(o))}</td><td><b>${esc(show(m))}</b></td><td>${delta(o,m)}</td><td>${esc(barText(m))} ${m.status === 'pass' ? '✓' : m.status === 'fail' ? '✗' : ''}</td></tr>`).join('') || '<tr><td colspan="5">No differences.</td></tr>'}</tbody></table>`;
}

function render(){
 document.getElementById('content').innerHTML = readiness() + experiments() + compare();
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
