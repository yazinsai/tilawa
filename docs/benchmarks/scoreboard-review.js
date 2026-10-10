'use strict';
// Mistake review: per-clip outcomes from pnpm scoreboard's clip dump (/clips/, served by tools/scoreboard_serve.py).
(() => {
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const WAVE = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true"><path d="M4 12v4m5-9v14m5-18v22m5-18v14m5-9v4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
const CLASSES = {wrong_word:'Wrong word', omitted_word:'Missing word', added_word:'Extra word', skipped_passage:'Skipped passage', short_vowel:'Vowel mistake', none:'Clean recitation'};
const readable = s => CLASSES[s] || String(s || '').replaceAll('_', ' ');
const SETUPS = {leadFree:'The engine finds the passage itself. Where available, clean recitation is added before and after the test clip.', declared:'The engine is given the passage. Only the original test recording is evaluated.', leadDeclared:'The engine is given the passage, with clean recitation added before and after each test clip.', free:'The engine finds the passage itself, using only the original test recording.', oracle:'A diagnostic baseline with ideal passage recognition and sound evidence.'};
const BADGES = {miss:'Missed mistake', fp:'False alarm', leadin:'False alarm', leadout:'False alarm', late:'Slow passage recognition', hit:'Detected mistake'};
const ARMS = {leadFree:'Automatic · with surrounding recitation', declared:'Passage provided · clip only', leadDeclared:'Passage provided · with surrounding recitation', free:'Automatic · clip only', oracle:'Ideal recognition · diagnostic baseline'};
const TYPES = {miss:'Missed mistakes', fp:'False warnings during the test clip', leadin:'False warnings before the test clip', leadout:'False warnings after the test clip', late:'Passage not recognized within 5 seconds'};
/** Gate → the outcomes behind it. */
const PRESETS = {
 'e2e.leadin.precision':{arm:'leadFree', type:'fp'}, 'e2e.leadin.recall':{arm:'leadFree', type:'miss'},
 'e2e.leadin.leadWarnings':{arm:'leadFree', type:'leadin'}, 'e2e.leadin.followWarnings':{arm:'leadFree', type:'leadout'},
 'checker.oracleLock.precision':{arm:'declared', type:'fp'}, 'checker.oracleLock.recall':{arm:'declared', type:'miss'},
 'tracker.clean.within5s':{arm:'free', type:'late', kind:'none'}, 'e2e.cleanPer5':{arm:'free', type:'fp', kind:'none'},
};
/** Expected and decoded text on separate lines: on one line, bidi reordering makes the direction ambiguous. */
const said = (e, words = false) => `<div class="rv-said"><span class="small">${words ? 'Expected word' : 'Expected sounds'}</span> <bdi class="ar" lang="ar">${esc(e.expected || '—')}</bdi></div><div class="rv-said"><span class="small">${words ? 'Recited instead' : 'Recognized sounds'}</span> <bdi class="ar" lang="ar">${esc(e.observed || '—')}</bdi></div>`;
const st = {runs:[], dumps:{}, audio:{}, sel:null, shown:[], revision:0, playRevision:0, initRevision:0};
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
 return `<section class="panel review-panel" id="review" aria-labelledby="review-title"><div class="section-top"><div><span class="eyebrow">THE RECORDINGS BEHIND THE NUMBERS</span><h2 id="review-title">Listen. See what happened.</h2><p>Explore missed mistakes and false alarms, one recording at a time.</p></div><span class="review-mark" aria-hidden="true">${WAVE}</span></div><div id="rv-body"><p class="notice">Loading recordings…</p></div></section>`;
}

function controls(){
 const runOpts = st.runs.map(r => `<option value="${esc(r.file)}">${esc(r.listener)} · ${esc(r.sha.slice(0,7))}</option>`).reverse().join('');
 const opt = (o, all) => `${all ? `<option value="">${all}</option>` : ''}${Object.entries(o).map(([k,v]) => `<option value="${k}">${esc(v)}</option>`).join('')}`;
 return `<div class="rv-primary-filters">
  <label>Recorded run<select id="rv-run">${runOpts}</select></label>
  <label>Test setup<select id="rv-arm">${opt(ARMS)}</select></label>
  <label>Show me<select id="rv-type">${opt(TYPES, 'Everything that needs attention')}</select></label>
 </div><div class="rv-filter-tools"><details class="rv-extra"><summary>Refine results</summary><div class="rv-more-filters">
  <label>Mistake type<select id="rv-class"><option value="">All mistake types</option><option value="wrong_word">Wrong word</option><option value="omitted_word">Missing word</option><option value="added_word">Extra word</option><option value="skipped_passage">Skipped passage</option></select></label>
  <label>Where it was missed<select id="rv-cause"><option value="">Any stage</option><option value="tracker">Passage recognition</option><option value="evidence">Sound recognition</option><option value="checker">Mistake checking</option></select></label>
  <label>Passage similarity<select id="rv-slice"><option value="">All passages</option><option value="u">Distinct passage</option><option value="m">Similar to another ayah</option></select></label>
  <label>Recording type<select id="rv-kind"><option value="">All recordings</option><option value="none">Clean recitation</option><option value="err">Contains a mistake</option></select></label>
  <label>Device<select id="rv-device"><option value="">Any device</option></select></label>
 </div></details><details class="rv-extra"><summary>Compare with another run</summary><div class="rv-more-filters">
  <label>Compare with<select id="rv-vs"><option value="">No comparison</option>${runOpts}</select></label>
  <label>Show differences<select id="rv-diff" disabled><option value="all">All results</option><option value="new">New issues in this run</option><option value="fixed">Resolved since the comparison run</option></select></label>
 </div><p class="small">“Resolved” shows the earlier recording and engine output, so you can hear what changed.</p></details><button class="text-button" id="rv-reset" type="button">Reset filters</button></div>
 <div class="rv-setup-note"><span aria-hidden="true">ⓘ</span><p id="rv-setup-copy"></p></div>
 <div class="rv-split"><div class="rv-browser"><div class="rv-list-heading" id="rv-count" role="status"></div><div id="rv-list"></div></div><div id="rv-detail" tabindex="-1" aria-label="Selected result"><p class="empty">Choose a recording to get started.</p></div></div>`;
}

function resetFilters(){
 for(const id of ['rv-type','rv-class','rv-cause','rv-slice','rv-kind','rv-device','rv-vs']) $(id).value = '';
 $('rv-diff').value = 'all';
 refresh().catch(fail);
}

async function refresh(){
 const revision = ++st.revision;
 const filters = Object.fromEntries(['run','vs','diff','arm','type','class','cause','slice','kind','device'].map(k => [k,$(`rv-${k}`).value]));
 const [cur,vs] = await Promise.all([loadDump(filters.run), filters.vs ? loadDump(filters.vs) : null]);
 if(revision !== st.revision || !$('rv-list')) return;
 const devices = [...new Set(cur.clips.map(c => c.device).filter(Boolean))].sort();
 $('rv-device').innerHTML = `<option value="">Any device</option>${devices.map(d => `<option value="${esc(d)}" ${d === filters.device ? 'selected' : ''}>${esc(d)}</option>`).join('')}`;
 $('rv-diff').disabled = !vs;
 $('rv-setup-copy').textContent = SETUPS[filters.arm];
 let list = items(cur,filters.arm), other = vs ? items(vs,filters.arm) : [];
 if(vs && filters.diff !== 'all'){
  const keys = new Set((filters.diff === 'new' ? other : list).map(x => x.key));
  list = (filters.diff === 'new' ? list : other).filter(x => !keys.has(x.key));
 }
 const keep = x => (!filters.type || x.type === filters.type)
  && (!filters.class || x.e?.errorClass === filters.class)
  && (!filters.cause || x.e?.cause === filters.cause)
  && (!filters.slice || (x.type === 'miss' && (filters.slice === 'm') === !!x.e.forms_other_ayah))
  && (!filters.kind || (filters.kind === 'none') === (x.clip.kind === 'none'))
  && (!filters.device || x.clip.device === filters.device);
 const shown = list.filter(keep);
 st.shown = shown;
 const unique = new Set(shown.map(x => x.clip.id)).size;
 $('rv-count').innerHTML = `${vs && filters.diff === 'fixed' ? '<p class="rv-comparison-note">Resolved results · audio from the earlier run</p>' : ''}<div><b>${shown.length} result${shown.length === 1 ? '' : 's'}</b><span class="small">from ${unique} recording${unique === 1 ? '' : 's'}</span></div><span class="small">Select to explore</span>`;
 $('rv-list').innerHTML = shown.map((x,i) => {
  const location = x.e?.at?.split(':').slice(0,2).join(':') || x.clip.ayahs.join(', ');
  const context = x.type === 'leadout' ? 'After the test clip' : x.type === 'leadin' ? 'Before the test clip' : 'In the test clip';
  return `<button type="button" data-i="${i}" class="rv-card" aria-pressed="false"><span class="rv-card-top"><span class="rv-type rv-${x.type}">${esc(BADGES[x.type])}</span><span class="rv-card-arrow" aria-hidden="true">↗</span></span><strong>Ayah ${esc(location)}</strong><span class="rv-card-description">${x.e ? esc(readable(x.e.errorClass)) : 'Passage not found within 5 seconds'}<span aria-hidden="true"> · </span>${esc(x.clip.device || 'Unknown device')}</span><span class="rv-card-context">${esc(context)}${x.type === 'leadin' || x.type === 'leadout' ? ` · test ayah ${esc(x.clip.ayahs.join(', '))}` : ''}</span></button>`;
 }).join('') || `<div class="empty"><span class="empty-icon" aria-hidden="true">✓</span><h3>No matching results</h3><p>Try another filter or test setup to explore more recordings.</p><button class="text-button" id="rv-empty-reset">Clear filters</button></div>`;
 $('rv-empty-reset')?.addEventListener('click',resetFilters);
 $('rv-list').querySelectorAll('[data-i]').forEach(b => b.addEventListener('click', () => { selectResult(+b.dataset.i); if(window.innerWidth <= 850) { $('rv-detail').scrollIntoView({behavior:'smooth',block:'start'}); $('rv-detail').focus({preventScroll:true}); } }));
 if(shown.length){ const i = shown.findIndex(x => x.key === st.sel?.key); selectResult(i < 0 ? 0 : i); }
 else { st.sel = null; ++st.playRevision; $('rv-detail').innerHTML = `<div class="rv-detail-empty"><span aria-hidden="true">${WAVE}</span><p>Your recording will appear here.</p></div>`; }
}

function selectResult(i){
 if(!st.shown[i]) return;
 st.sel = st.shown[i]; ++st.playRevision;
 $('rv-list').querySelectorAll('[data-i]').forEach(b => { b.classList.toggle('rv-on',+b.dataset.i === i); b.setAttribute('aria-pressed',String(+b.dataset.i === i)); });
 detail(st.sel);
 $('rv-position').textContent = `${i+1} of ${st.shown.length}`;
 $('rv-back').addEventListener('click',() => { $('rv-count').scrollIntoView({behavior:'smooth',block:'start'}); $('rv-list').querySelector('.rv-on')?.focus({preventScroll:true}); });
 $('rv-prev').disabled = i === 0; $('rv-next').disabled = i === st.shown.length-1;
 $('rv-prev').addEventListener('click',() => navigateResult(i-1,'rv-prev'));
 $('rv-next').addEventListener('click',() => navigateResult(i+1,'rv-next'));
}

function navigateResult(i,focusId){
 selectResult(i);
 const list = $('rv-list'), card = list.querySelector(`[data-i="${i}"]`);
 if(card){
  const top = card.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop;
  if(top < list.scrollTop) list.scrollTop = top - 4;
  else if(top + card.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = top + card.offsetHeight - list.clientHeight + 4;
 }
 const button = $(focusId);
 (button.disabled ? $(focusId === 'rv-next' ? 'rv-prev' : 'rv-next') : button).focus({preventScroll:true});
}

async function audioUrl(path){
 if(!st.audio[path]){ const r = await fetch(`audio/${path}`); if(!r.ok) throw new Error(`audio/${path}: HTTP ${r.status}`); st.audio[path] = URL.createObjectURL(await r.blob()); }
 return st.audio[path];
}

async function play(path,at,label){
 const revision = ++st.playRevision, el = $('rv-audio'), status = $('rv-now');
 status.textContent = 'Loading audio…';
 try{
  const url = await audioUrl(path);
  if(revision !== st.playRevision || el !== $('rv-audio')) return;
  if(el.dataset.path !== path){
   await new Promise((resolve,reject) => {
    const clean = () => { el.removeEventListener('loadedmetadata',loaded); el.removeEventListener('error',failed); };
    const loaded = () => { clean(); resolve(); };
    const failed = () => { clean(); reject(new Error('This recording could not be played.')); };
    el.addEventListener('loadedmetadata',loaded,{once:true}); el.addEventListener('error',failed,{once:true});
    el.src = url; el.dataset.path = path; el.load();
   });
  }
  if(revision !== st.playRevision || el !== $('rv-audio')) return;
  el.hidden = false;
  el.currentTime = Math.max(0,Math.min(at,Number.isFinite(el.duration) ? el.duration : at));
  await el.play();
  if(revision !== st.playRevision) return;
  status.textContent = `${label || 'Recording'} · started at ${at.toFixed(1)} seconds`;
  $('rv-detail').querySelectorAll('.rv-timeline button').forEach(b => b.classList.toggle('is-playing',b.dataset.path === path));
 }catch(e){ if(revision === st.playRevision) status.textContent = e.name === 'NotAllowedError' ? 'Press play on the audio player to listen.' : 'This recording couldn’t be played. Try again or choose another section.'; }
}

function detail(x){
 const c = x.clip, a = x.arm, at = x.e?.at;
 const location = at?.split(':').slice(0,2).join(':') || c.ayahs.join(', ');
 const wordNumber = at?.split(':')[2];
 const outside = x.type === 'leadin' || x.type === 'leadout';
 const flags = new Map();
 for(const e of a.misses) flags.set(e.at,'miss');
 for(const e of a.falsePositives) flags.set(e.at,'fp');
 for(const l of c.labels) if(!flags.has(`${l.location.surah}:${l.location.ayah}:${l.location.word}`)) flags.set(`${l.location.surah}:${l.location.ayah}:${l.location.word}`,'hit');
 const ev = (e,kind) => `<li><span class="rv-type rv-${kind}">${esc(BADGES[kind] || kind)}</span> ${esc(readable(e.errorClass))}<span class="small"> · ${esc(e.at)}</span>${said(e,kind === 'miss')}${e.state ? `<span class="small">${esc(e.state)}</span>` : ''}<button data-path="${esc(e.audio)}" data-at="${e.seekS}" data-label="Event audio">▶ Listen at ${e.seekS.toFixed(1)} s</button></li>`;
 const seek = x.e ? {path:x.e.audio,at:x.e.seekS} : {path:c.audio,at:0};
 const explanation = {leadout:'This recitation was correct. The engine raised a warning in the clean audio added after the test clip.',leadin:'This recitation was correct. The engine raised a warning in the clean audio added before the test clip.',miss:'This recording contains a real mistake that the engine did not confirm.',fp:'The engine raised a warning, but there was no labeled mistake at this location.',late:'The engine did not find the correct passage within five seconds of the first recognized sound.'};
 const actual = x.type === 'miss' ? c.labels.find(l => `${l.location.surah}:${l.location.ayah}:${l.location.word}` === at && l.errorClass === x.e.errorClass) : null;
 const word = !outside && c.words.find(w => w.at === at);
 const focus = a.segments?.find(seg => seg.audio === seek.path);
 const main = a.segments?.find(seg => seg.source === c.id);
 const segmentLabel = seg => seg.source === c.id ? 'Test clip' : seg.endMs <= main?.startMs ? 'Before' : 'After';
 $('rv-detail').innerHTML = `<div class="rv-detail-nav"><div class="rv-nav-label"><button class="text-button" id="rv-back">↑ All results</button><span id="rv-position" class="small" role="status"></span></div><div><button id="rv-prev" aria-label="Previous result">←</button><button id="rv-next" aria-label="Next result">→</button></div></div><div class="rv-head"><span class="rv-type rv-${x.type}">${esc(BADGES[x.type])}</span><h3>Ayah ${esc(location)}${wordNumber ? `<span class="word-number">Word ${esc(wordNumber)}</span>` : ''}</h3><p>${esc(explanation[x.type])}</p><div class="rv-facts"><span>${esc(c.device || 'Unknown device')}</span><span>${c.durationS.toFixed(1)} s test clip</span>${c.level ? `<span>${esc(c.level)}</span>` : ''}</div></div>
 ${x.e ? `<div class="rv-verdict"><span>${x.type === 'miss' ? 'Actual mistake' : 'Engine flagged'}</span><strong>${esc(readable(x.e.errorClass))}</strong>${word ? `<bdi class="ar" lang="ar">${esc(word.text)}</bdi>` : ''}</div>` : ''}
 ${actual ? `<div class="rv-actual"><div><span>Expected word</span><bdi lang="ar" dir="rtl">${esc(word?.text || actual.expected || '—')}</bdi></div><div><span>Recited instead</span>${actual.observed ? `<bdi lang="ar" dir="rtl">${esc(actual.observed)}</bdi>` : `<span class="rv-not-recited">${actual.errorClass === 'omitted_word' ? 'Word not recited' : 'Not recorded'}</span>`}</div></div>` : ''}
 <div class="rv-player"><button class="rv-play primary-button" data-path="${esc(seek.path)}" data-at="${seek.at}" data-label="${outside ? 'Clean recitation '+(x.type === 'leadout' ? 'after' : 'before')+' the test clip' : 'Test clip'}"><span aria-hidden="true">▶</span> ${x.type === 'miss' ? 'Hear the missed mistake' : x.type === 'late' ? 'Listen from the start' : 'Hear the false alarm'}</button><audio id="rv-audio" controls hidden preload="none" aria-label="Selected recording"></audio><p class="small" id="rv-now" role="status">${x.e ? `Starts just before the ${x.type === 'miss' ? 'mistake' : 'warning'}, at ${seek.at.toFixed(1)} seconds.` : 'Press play to load the recording.'}</p></div>
 ${a.segments ? `<div class="rv-session"><div class="rv-section-label"><h4>Where it happened</h4><span class="small">${outside ? 'In added clean recitation' : 'In the test clip'}</span></div><div class="rv-timeline">${a.segments.map(seg => `<button class="${seg === focus ? 'is-event' : ''}" data-path="${esc(seg.audio)}" data-at="0" data-label="${segmentLabel(seg)} · ayah ${esc(seg.ayahs.join(', '))}"><span class="segment-label">${segmentLabel(seg)}${seg === focus ? '<span class="event-dot" aria-label="Selected result happened here"></span>' : ''}</span><strong>${esc(seg.ayahs.join(', '))}</strong><span>${((seg.endMs-seg.startMs)/1000).toFixed(1)} s · ${seg.source === c.id ? 'Original recording' : 'Clean recitation'}</span><span class="segment-play">▶ Listen</span></button>`).join('')}</div><p class="rv-sequence-note">Before and after are clean recordings added to simulate a longer recitation. The highlighted section contains this result.</p></div>` : `<button class="text-button" data-path="${esc(c.audio)}" data-at="0" data-label="Complete test clip">Listen to the complete test clip ↗</button>`}
 <details class="rv-diagnostics"><summary>Word recognition & technical details</summary><div class="small">The top line is the Quran word. The lower lines show expected and recognized sounds in the engine’s phoneme notation, not standard Arabic spelling. Select a word to hear it; red text marks a sound mismatch.</div>
 <div class="rv-words" dir="rtl">${c.words.map(w => `<button type="button" class="rv-word ${w.at === at ? 'rv-focus' : ''} ${flags.get(w.at) ? 'rv-f-' + flags.get(w.at) : ''} ${w.said !== w.ref ? 'rv-diff' : ''}" ${w.fromS != null ? `data-path="${esc(c.audio)}" data-at="${Math.max(0, w.fromS - .3)}"` : ''} ${w.fromS == null ? 'disabled' : ''} aria-label="Listen to word ${esc(w.at)}" title="${esc(w.at)}"><b lang="ar">${esc(w.text)}</b><i>${esc(w.ref)}</i><i class="said">${esc(w.said || '∅')}</i></button>`).join('')}</div>
 <h3>Actual mistakes in the recording</h3><ul class="rv-events">${c.labels.map(l => `<li>${esc(readable(l.errorClass))} <span class="small">${l.location.surah}:${l.location.ayah}:${l.location.word} · ${esc(l.state)}${l.forms_other_ayah ? ' · mutashabihat' : ''}</span> ${said(l,true)}</li>`).join('') || '<li class="small">Clean recitation.</li>'}${c.repeats.length ? `<li class="small">Planted repeats at ${esc(c.repeats.join(', '))}</li>` : ''}</ul>
 <h3>What the engine reported</h3><ul class="rv-events">${a.hits.map(h => `<li><span class="rv-type rv-hit">Detected mistake</span> ${esc(h)}</li>`).join('')}${a.misses.map(e => ev(e, 'miss')).join('')}${a.falsePositives.map(e => ev(e, 'fp')).join('')}${(a.leadInWarnings || []).map(e => ev(e, 'leadin')).join('')}${(a.leadOutWarnings || []).map(e => ev(e, 'leadout')).join('')}</ul>
 <p class="small">Tracker: first decoded phoneme at ${(a.gate.onsetMs/1000).toFixed(1)} s, ${a.gate.firstCorrectLockMs == null ? 'never locked on the passage' : `correct lock ${((a.gate.firstCorrectLockMs - a.gate.onsetMs)/1000).toFixed(1)} s later`}${a.gate.wrongLockFirst ? ', wrong lock first' : ''}${a.gate.falseJumps ? `, ${a.gate.falseJumps} false jumps` : ''}${a.segments ? ' (session time)' : ''}.</p>
 ${a.segments ? `<h3>Recording source files</h3><ul class="rv-events">${a.segments.map(s => `<li>${s.source === c.id ? '<b>clip</b>' : esc(s.source)} <span class="small">${esc(s.ayahs.join(', '))} · ${(s.startMs/1000).toFixed(1)}–${(s.endMs/1000).toFixed(1)} s</span> <button data-path="${esc(s.audio)}" data-at="0">▶</button></li>`).join('')}</ul>` : ''}<p class="small">Recording ID: <code>${esc(c.id)}</code></p></details>`;
 $('rv-detail').querySelectorAll('[data-path]').forEach(b => b.addEventListener('click',() => play(b.dataset.path,+b.dataset.at,b.dataset.label || 'Selected word')));
}

async function init(cand){
 const revision = ++st.initRevision;
 ++st.revision; ++st.playRevision;
 try{
  const r = await fetch('clips/index.json', {cache:'no-store'}); if(!r.ok) throw new Error(`HTTP ${r.status}`);
  st.runs = await r.json(); if(!st.runs.length) throw new Error('no runs');
 }catch(e){
  if(revision !== st.initRevision) return;
  $('rv-body').innerHTML = `<div class="notice"><b>No clip dump.</b> Run <code>pnpm scoreboard</code> (or add <code>--dump-only --sha &lt;commit&gt;</code> to skip the row), then serve with <code>uv run tools/scoreboard_serve.py</code>. (${esc(e.message)})</div>`;
  return;
 }
 if(revision !== st.initRevision) return;
 $('rv-body').innerHTML = controls();
 const matching = st.runs.filter(x => x.listener === cand).at(-1);
 const mine = matching ?? st.runs.at(-1);
 if(!matching) $('rv-body').insertAdjacentHTML('afterbegin',`<p class="notice rv-availability">No recording results are available for ${esc(cand)} yet. Showing ${esc(mine.listener)}, the latest available recorded run.</p>`);
 $('rv-run').value = mine.file;
 $('rv-reset').addEventListener('click',resetFilters);
 ['rv-run','rv-vs','rv-diff','rv-arm','rv-type','rv-class','rv-cause','rv-slice','rv-kind','rv-device'].forEach(id => $(id).addEventListener('change', () => refresh().catch(fail)));
 await refresh().catch(fail);
}
const fail = e => { if($('rv-list')) $('rv-list').innerHTML = `<div class="notice">Couldn’t load recordings. ${esc(e.message)}</div>`; };

/** Jump from a gate row to the outcomes behind it. */
function focusGate(id){
 const p = PRESETS[id]; if(!p || !$('rv-arm')) return false;
 for(const k of ['rv-type','rv-class','rv-cause','rv-slice','rv-kind','rv-device','rv-vs']) $(k).value = '';
 $('rv-diff').value = 'all';
 $('rv-arm').value = p.arm; $('rv-type').value = p.type; $('rv-kind').value = p.kind || '';
 refresh().catch(fail); $('review').scrollIntoView({behavior:'smooth'});
 return true;
}

window.tilawaReview = {panel, init, focusGate, hasPreset: id => id in PRESETS};
})();
