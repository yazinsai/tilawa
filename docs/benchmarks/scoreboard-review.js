'use strict';
// Mistake review: per-clip outcomes from pnpm scoreboard's clip dump (/clips/, served by tools/scoreboard_serve.py).
(() => {
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ARMS = {leadFree:'Free, with lead-in/out (gated)', declared:'Declared passage (gated)', leadDeclared:'Declared, with lead-in/out', free:'Free, no lead-in (info)', oracle:'Oracle lock + evidence'};
const TYPES = {miss:'Missed mistakes', fp:'False warnings (in clip)', leadin:'False warnings in the lead-in', leadout:'False warnings in the lead-out', late:'Tracker: no correct lock within 5 s'};
/** Gate → the outcomes behind it. */
const PRESETS = {
 'e2e.leadin.precision':{arm:'leadFree', type:'fp'}, 'e2e.leadin.recall':{arm:'leadFree', type:'miss'},
 'e2e.leadin.leadWarnings':{arm:'leadFree', type:'leadin'}, 'e2e.leadin.followWarnings':{arm:'leadFree', type:'leadout'},
 'checker.oracleLock.precision':{arm:'declared', type:'fp'}, 'checker.oracleLock.recall':{arm:'declared', type:'miss'},
 'tracker.clean.within5s':{arm:'free', type:'late', kind:'none'}, 'e2e.cleanPer5':{arm:'free', type:'fp', kind:'none'},
};
/** Expected and decoded text on separate lines: on one line, bidi reordering makes the direction ambiguous. */
const said = e => `<div class="rv-said"><span class="small">expected</span> <bdi class="ar" lang="ar">${esc(e.expected || '—')}</bdi></div><div class="rv-said"><span class="small">said</span> <bdi class="ar" lang="ar">${esc(e.observed || '—')}</bdi></div>`;
const st = {runs:[], dumps:{}, audio:{}, sel:null};
const $ = id => document.getElementById(id);

function items(dump, arm){
 if(!dump) return [];
 const out = [];
 for(const c of dump.clips){
  const a = c.arms[arm]; if(!a) continue;
  const push = (type, e) => out.push({type, clip:c, arm:a, e, key:`${type}|${c.id}|${e ? `${e.errorClass}/${e.at}` : ''}`});
  a.misses.forEach(e => push('miss', e));
  a.falsePositives.forEach(e => push('fp', e));
  (a.leadInWarnings || []).forEach(e => push('leadin', e));
  (a.leadOutWarnings || []).forEach(e => push('leadout', e));
  if(!a.gate.correctWithin5s && (arm === 'free' || arm === 'leadFree')) push('late', null);
 }
 return out;
}

async function loadDump(file){
 if(!st.dumps[file]){ const r = await fetch(`clips/${file}`, {cache:'no-store'}); if(!r.ok) throw new Error(`clips/${file}: HTTP ${r.status}`); st.dumps[file] = await r.json(); }
 return st.dumps[file];
}

function panel(){
 return `<section class="panel" id="review"><div class="section-top"><div><h2>Mistake review</h2><p>Every missed mistake, false warning and late tracker lock behind the gates, one clip at a time. Click a failing gate above to jump to its clips.</p></div></div><div id="rv-body"><p class="small">Loading clip dumps…</p></div></section>`;
}

function controls(){
 const runOpts = st.runs.map(r => `<option value="${esc(r.file)}">${esc(r.listener)} @ ${esc(r.sha.slice(0,7))}</option>`).reverse().join('');
 const opt = (o, all) => `${all ? `<option value="">${all}</option>` : ''}${Object.entries(o).map(([k,v]) => `<option value="${k}">${esc(v)}</option>`).join('')}`;
 return `<div class="filters rv-filters">
  <label>Run <select id="rv-run">${runOpts}</select></label>
  <label>vs <select id="rv-vs"><option value="">— no diff —</option>${runOpts}</select></label>
  <label>Show <select id="rv-diff"><option value="all">all</option><option value="new">only in this run (new)</option><option value="fixed">only in the vs run (fixed)</option></select></label>
  <label>Arm <select id="rv-arm">${opt(ARMS)}</select></label>
  <label>Outcome <select id="rv-type">${opt(TYPES, 'any outcome')}</select></label>
  <label>Class <select id="rv-class"><option value="">any class</option><option>wrong_word</option><option>omitted_word</option><option>added_word</option><option>skipped_passage</option></select></label>
  <label>Lost to <select id="rv-cause"><option value="">any stage</option><option>tracker</option><option>evidence</option><option>checker</option></select></label>
  <label>Slice <select id="rv-slice"><option value="">all</option><option value="u">unambiguous</option><option value="m">mutashabihat</option></select></label>
  <label>Clips <select id="rv-kind"><option value="">all clips</option><option value="none">clean clips</option><option value="err">error clips</option></select></label>
  <label>Device <select id="rv-device"><option value="">any device</option></select></label>
 </div><div class="rv-split"><div id="rv-list"></div><div id="rv-detail"><p class="empty">Pick a row to hear the clip and see what was decoded.</p></div></div>`;
}

async function refresh(){
 const f = id => $(id).value;
 const cur = await loadDump(f('rv-run')), vs = f('rv-vs') ? await loadDump(f('rv-vs')) : null;
 const devices = [...new Set(cur.clips.map(c => c.device).filter(Boolean))].sort();
 const dsel = $('rv-device'), dv = dsel.value;
 dsel.innerHTML = `<option value="">any device</option>${devices.map(d => `<option ${d === dv ? 'selected' : ''}>${esc(d)}</option>`).join('')}`;
 const arm = f('rv-arm');
 let list = items(cur, arm), other = vs ? items(vs, arm) : [];
 if(vs && f('rv-diff') !== 'all'){
  const keys = new Set((f('rv-diff') === 'new' ? other : list).map(x => x.key));
  list = (f('rv-diff') === 'new' ? list : other).filter(x => !keys.has(x.key));
 }
 const keep = x => (!f('rv-type') || x.type === f('rv-type'))
  && (!f('rv-class') || x.e?.errorClass === f('rv-class'))
  && (!f('rv-cause') || x.e?.cause === f('rv-cause'))
  && (!f('rv-slice') || (x.type === 'miss' && (f('rv-slice') === 'm') === !!x.e.forms_other_ayah))
  && (!f('rv-kind') || (f('rv-kind') === 'none') === (x.clip.kind === 'none'))
  && (!f('rv-device') || x.clip.device === f('rv-device'));
 const shown = list.filter(keep);
 const counts = Object.keys(TYPES).map(t => [t, shown.filter(x => x.type === t).length]).filter(([,n]) => n);
 const word = x => x.e && x.clip.words.find(w => w.at === x.e.at)?.text;
 $('rv-list').innerHTML = `<p class="small">${shown.length} outcome${shown.length === 1 ? '' : 's'}${counts.length ? ' · ' + counts.map(([t,n]) => `${n} ${esc(TYPES[t].toLowerCase())}`).join(' · ') : ''}</p>
 <table class="gates rv-table"><thead><tr><th>Outcome</th><th>Clip</th><th>Class · word</th><th>Expected → said</th><th>Why</th></tr></thead><tbody>${shown.map((x,i) => `<tr data-i="${i}" class="${st.sel?.key === x.key && st.sel?.clip.id === x.clip.id ? 'rv-on' : ''}"><td><span class="rv-type rv-${x.type}">${esc({miss:'miss',fp:'false',leadin:'lead-in',leadout:'lead-out',late:'late lock'}[x.type])}</span></td><td><code>${esc(x.clip.id)}</code><div class="small">${esc(x.clip.ayahs.join(', '))} · ${esc(x.clip.device || '')}</div></td><td>${x.e ? `${esc(x.e.errorClass)} <span class="small">${esc(x.e.at)}</span> <span class="ar" lang="ar" dir="rtl">${esc(word(x) || '')}</span>` : `<span class="small">${x.arm.gate.firstCorrectLockMs == null ? 'never locked on the passage' : `locked at ${((x.arm.gate.firstCorrectLockMs - x.arm.gate.onsetMs)/1000).toFixed(1)} s`}${x.arm.gate.wrongLockFirst ? ' · wrong lock first' : ''}</span>`}</td><td>${x.e ? said(x.e) : ''}</td><td class="small">${x.e?.cause ? `lost to <b>${esc(x.e.cause)}</b>` : ''}${x.e?.raisedAs ? ` · raised, ended ${esc(x.e.raisedAs)}` : x.type === 'miss' ? ' · never raised' : ''}${x.e?.forms_other_ayah ? ' · mutashabihat' : ''}${x.e?.confidence != null && x.type !== 'miss' ? ` · conf ${(+x.e.confidence).toFixed(2)}` : ''}</td></tr>`).join('') || '<tr><td colspan="5" class="empty">Nothing matches.</td></tr>'}</tbody></table>`;
 $('rv-list').querySelectorAll('tr[data-i]').forEach(tr => tr.addEventListener('click', () => { st.sel = shown[+tr.dataset.i]; $('rv-list').querySelectorAll('tr').forEach(t => t.classList.toggle('rv-on', t === tr)); detail(st.sel); }));
}

async function audioUrl(path){
 if(!st.audio[path]){ const r = await fetch(`audio/${path}`); if(!r.ok) throw new Error(`audio/${path}: HTTP ${r.status}`); st.audio[path] = URL.createObjectURL(await r.blob()); }
 return st.audio[path];
}

async function play(path, at){
 const el = $('rv-audio');
 try{ const url = await audioUrl(path); if(el.dataset.path !== path){ el.src = url; el.dataset.path = path; await new Promise(r => el.addEventListener('loadedmetadata', r, {once:true})); } el.currentTime = at; el.play(); $('rv-now').textContent = `${path} @ ${at.toFixed(1)} s`; }
 catch(e){ $('rv-now').textContent = `Couldn’t load audio: ${e.message}`; }
}

function detail(x){
 const c = x.clip, a = x.arm, at = x.e?.at;
 const flags = new Map();
 for(const e of a.misses) flags.set(e.at, 'miss');
 for(const e of a.falsePositives) flags.set(e.at, 'fp');
 for(const l of c.labels) if(!flags.has(`${l.location.surah}:${l.location.ayah}:${l.location.word}`)) flags.set(`${l.location.surah}:${l.location.ayah}:${l.location.word}`, 'hit');
 const ev = (e, kind) => `<li><span class="rv-type rv-${kind}">${kind}</span> ${esc(e.errorClass)} <span class="small">${esc(e.at)}</span> ${said(e)} ${e.state ? `<span class="small">${esc(e.state)}</span>` : ''} <button data-path="${esc(e.audio)}" data-at="${e.seekS}">▶ ${e.seekS.toFixed(1)} s</button></li>`;
 const seek = x.e ? {path:x.e.audio, at:x.e.seekS} : {path:c.audio, at:0};
 $('rv-detail').innerHTML = `<div class="rv-head"><h3><code>${esc(c.id)}</code></h3><div class="small">${esc(c.ayahs.join(', '))} · ${esc(c.kind)} · ${esc(c.device || '')} · ${esc(c.level || '')} · speaker ${esc(c.speaker.slice(0,8))} · ${c.durationS.toFixed(1)} s · ${esc(ARMS[$('rv-arm').value])}</div></div>
 <audio id="rv-audio" controls preload="none"></audio><div class="small" id="rv-now"></div>
 <p><button class="rv-play" data-path="${esc(seek.path)}" data-at="${seek.at}">▶ Play from the ${x.e ? 'event' : 'start'}</button> <button class="rv-play" data-path="${esc(c.audio)}" data-at="0">▶ Whole clip</button></p>
 <div class="small">Written words (top) and what the listener decoded for each (bottom); click a word to hear it. Red = decode differs from the written phonemes.</div>
 <div class="rv-words" dir="rtl">${c.words.map(w => `<span class="rv-word ${w.at === at ? 'rv-focus' : ''} ${flags.get(w.at) ? 'rv-f-' + flags.get(w.at) : ''} ${w.said !== w.ref ? 'rv-diff' : ''}" ${w.fromS != null ? `data-path="${esc(c.audio)}" data-at="${Math.max(0, w.fromS - .3)}"` : ''} title="${esc(w.at)}"><b lang="ar">${esc(w.text)}</b><i>${esc(w.ref)}</i><i class="said">${esc(w.said || '∅')}</i></span>`).join('')}</div>
 <h3>Labels</h3><ul class="rv-events">${c.labels.map(l => `<li>${esc(l.errorClass)} <span class="small">${l.location.surah}:${l.location.ayah}:${l.location.word} · ${esc(l.state)}${l.forms_other_ayah ? ' · mutashabihat' : ''}</span> ${said(l)}</li>`).join('') || '<li class="small">Clean recitation.</li>'}${c.repeats.length ? `<li class="small">Planted repeats at ${esc(c.repeats.join(', '))}</li>` : ''}</ul>
 <h3>This arm on this clip</h3><ul class="rv-events">${a.hits.map(h => `<li><span class="rv-type rv-hit">hit</span> ${esc(h)}</li>`).join('')}${a.misses.map(e => ev(e, 'miss')).join('')}${a.falsePositives.map(e => ev(e, 'fp')).join('')}${(a.leadInWarnings || []).map(e => ev(e, 'lead-in')).join('')}${(a.leadOutWarnings || []).map(e => ev(e, 'lead-out')).join('')}</ul>
 <p class="small">Tracker: first decoded phoneme at ${(a.gate.onsetMs/1000).toFixed(1)} s, ${a.gate.firstCorrectLockMs == null ? 'never locked on the passage' : `correct lock ${((a.gate.firstCorrectLockMs - a.gate.onsetMs)/1000).toFixed(1)} s later`}${a.gate.wrongLockFirst ? ', wrong lock first' : ''}${a.gate.falseJumps ? `, ${a.gate.falseJumps} false jumps` : ''}${a.segments ? ' (session time)' : ''}.</p>
 ${a.segments ? `<h3>Stitched session</h3><ul class="rv-events">${a.segments.map(s => `<li>${s.source === c.id ? '<b>clip</b>' : esc(s.source)} <span class="small">${esc(s.ayahs.join(', '))} · ${(s.startMs/1000).toFixed(1)}–${(s.endMs/1000).toFixed(1)} s</span> <button data-path="${esc(s.audio)}" data-at="0">▶</button></li>`).join('')}</ul>` : ''}`;
 $('rv-detail').querySelectorAll('[data-path]').forEach(b => b.addEventListener('click', () => play(b.dataset.path, +b.dataset.at)));
}

async function init(cand){
 try{
  const r = await fetch('clips/index.json', {cache:'no-store'}); if(!r.ok) throw new Error(`HTTP ${r.status}`);
  st.runs = await r.json(); if(!st.runs.length) throw new Error('no runs');
 }catch(e){
  $('rv-body').innerHTML = `<div class="notice"><b>No clip dump.</b> Run <code>pnpm scoreboard</code> (or add <code>--dump-only --sha &lt;commit&gt;</code> to skip the row), then serve with <code>uv run tools/scoreboard_serve.py</code>. (${esc(e.message)})</div>`;
  return;
 }
 $('rv-body').innerHTML = controls();
 const mine = st.runs.filter(x => x.listener === cand).at(-1) ?? st.runs.at(-1);
 $('rv-run').value = mine.file;
 ['rv-run','rv-vs','rv-diff','rv-arm','rv-type','rv-class','rv-cause','rv-slice','rv-kind','rv-device'].forEach(id => $(id).addEventListener('change', () => refresh().catch(fail)));
 await refresh().catch(fail);
}
const fail = e => { $('rv-list').innerHTML = `<div class="notice">${esc(e.message)}</div>`; };

/** Jump from a gate row to the outcomes behind it. */
function focusGate(id){
 const p = PRESETS[id]; if(!p || !$('rv-arm')) return false;
 for(const k of ['rv-type','rv-class','rv-cause','rv-slice','rv-kind','rv-device','rv-vs']) $(k).value = '';
 $('rv-arm').value = p.arm; $('rv-type').value = p.type; $('rv-kind').value = p.kind || '';
 refresh().catch(fail); $('review').scrollIntoView({behavior:'smooth'});
 return true;
}

window.tilawaReview = {panel, init, focusGate, hasPreset: id => id in PRESETS};
})();
