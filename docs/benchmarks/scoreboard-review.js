'use strict';
(() => {
const M = globalThis.tilawaReviewModel, Store = globalThis.tilawaReviewStore;
const esc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $ = id => document.getElementById(id);
const WAVE = '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true"><path d="M4 12v4m5-9v14m5-18v22m5-18v14m5-9v4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
const CLASSES = {wrong_word:'Wrong word',omitted_word:'Missing word',added_word:'Added word',skipped_passage:'Passage skipped',none:'No planned mistake'};
const readable = s => CLASSES[s] || String(s || '').replaceAll('_',' ');
const ARMS = {leadFree:'Automatic · surrounding recitation',declared:'Passage provided · clip only',leadDeclared:'Passage provided · surrounding recitation',free:'Automatic · clip only',oracle:'Simulated sounds · checker test'};
const SETUPS = {
 leadFree:'The engine must find the passage itself. When available, recordings labelled as correct play before and after the test clip. Their warnings are counted separately.',
 declared:'The checker knows which passage to check and uses only the original clip. Passage recognition runs separately.',
 leadDeclared:'The checker knows which passage to check. It uses the test clip and any added recordings, which are labelled as correct.',
 free:'The engine must find the passage using only the original clip. The missed-mistake breakdown above uses this setup.',
 oracle:'The checker knows the passage and receives simulated sounds based on the labels. This tests mistake checking with ideal inputs. The audio and sound codes below come from the real recording.'
};
const TYPES = {miss:'Labelled mistakes not reported',fp:'Warnings without a label',leadin:'Warnings before the test clip',leadout:'Warnings after the test clip',late:'Passage not found within 5 seconds'};
const AREAS = {model:'Model error',label:'Wrong or missing label',audio_data:'Audio or data issue',acceptable:'Engine result is reasonable',unresolved:'Unsure / needs more evidence'};
const CATEGORIES = {missed_detection:'Mistake missed',wrong_mistake_type:'Wrong mistake type',wrong_word_location:'Warning on the wrong word',passage_recognition:'Wrong passage or passage not found',sound_recognition:'Sounds misheard',spurious_warning:'False alarm',other:'Other model error'};
const CAUSES = {evidence:'Trouble recognising sounds',tracker:'Trouble finding the passage',checker:'Trouble checking for mistakes'};
const PRESETS = {
 'checker.e0.evidence':{arm:'free',type:'miss',cause:'evidence'},'checker.e0.tracker':{arm:'free',type:'miss',cause:'tracker'},'checker.e0.checker':{arm:'free',type:'miss',cause:'checker'},
 'e2e.leadin.precision':{arm:'leadFree',type:'fp'},'e2e.leadin.recall':{arm:'leadFree',type:'miss'},'e2e.leadin.leadWarnings':{arm:'leadFree',type:'leadin'},'e2e.leadin.followWarnings':{arm:'leadFree',type:'leadout'},
 'checker.oracleLock.precision':{arm:'declared',type:'fp'},'checker.oracleLock.recall':{arm:'declared',type:'miss'},'tracker.clean.within5s':{arm:'free',type:'late',kind:'none'},'e2e.cleanPer5':{arm:'free',type:'fp',kind:'none'}
};
// Access can fail even before getItem (privacy settings / opaque origins).
const storage = {getItem:key=>window.localStorage.getItem(key),setItem:(key,value)=>window.localStorage.setItem(key,value)};
const saved = Store.create(storage);
const st = {runs:[],dumps:{},sel:null,all:[],shown:[],revision:0,playRevision:0,initRevision:0,filter:null,audioObservations:{},queue:[],gapTimer:null,audioCache:new Map(),candidateRunFile:null};
const review = i => saved.records[M.reviewKey(i)];
const hasDraft = r => !!r && !M.complete(r);
const options = (o,first) => `${first ? `<option value="">${esc(first)}</option>` : ''}${Object.entries(o).map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join('')}`;
const regionName = i => ({clip:'Test clip',before:'Before test clip',after:'After test clip',passage:'Passage timing',unknown:'Audio section unknown'}[i.zone]);
function title(i){
 if(i.mismatch) return 'Different mistake type reported';
 if(i.zone === 'passage') return i.arm.gate.firstCorrectLockMs === null ? 'Passage not found' : 'Passage found too late';
 if(i.zone === 'before') return i.raw.some(r=>r.type==='leadin') ? 'Warning before the test clip' : 'Engine report before the test clip';
 if(i.zone === 'after') return i.raw.some(r=>r.type==='leadout') ? 'Warning after the test clip' : 'Engine report after the test clip';
 if(i.raw.some(r=>r.type==='miss')) return 'Labelled mistake not reported';
 if(i.raw.some(r=>r.type==='fp')) return 'Warning without a label';
 if(i.hits.length) return 'Labelled mistake reported';
 if(i.labels.length) return 'Label not included in the score';
 if(i.reports.length) return i.reports.every(r=>r.state==='resolved') ? 'Correction reported' : i.reports.every(r=>['suspected','uncertain'].includes(r.state)) ? 'Possible mistake reported' : 'Report not included in the score';
 return 'No mistake scored';
}
function explanation(i){
 if(i.mismatch) return 'The engine reported a different mistake type from the label at this word. The test counts two issues: the labelled mistake was not reported, and the reported type has no label. Listen to check which type is right.';
 if(i.zone==='passage') return i.arm.gate.firstCorrectLockMs===null ? 'The engine did not find an ayah from the test passage. This fails the five-second timing check. Recitation mistakes have separate scores.' : 'The engine took more than five seconds to find an ayah from the test passage. Recitation mistakes have separate scores.';
 if(i.zone==='before'||i.zone==='after') return i.raw.some(r=>['leadin','leadout'].includes(r.type)) ? 'This added recording is labelled as correct. The test therefore counts its warnings as false alarms. Listen to check whether the warning is valid.' : 'The engine saved a report in an added recording labelled as correct. This report is not counted as a warning in the score. Listen to check it.';
 if(i.raw.some(r=>r.type==='miss')) return 'The label marks a mistake here, but the engine did not confirm that mistake at this word. Listen to check whether it missed a real mistake or the label needs correcting.';
 if(i.raw.some(r=>r.type==='fp')) return 'The engine confirmed a mistake here, but the labels do not list that mistake at this word. Listen to check whether the warning is valid, the label is missing or the engine marked the wrong word or type.';
 if(i.hits.length) return 'The engine reported the labelled mistake at the right word and with the same type. Listen to check that the label and warning are correct.';
 if(i.labels.length) return 'This label is not counted as a mistake in this test. Listen to check what the label says.';
 if(i.reports.length){
  if(i.reports.every(r=>r.state==='resolved')) return 'The engine marked the mistake as corrected. Corrected mistakes are not counted in this score. Listen to check the correction.';
  if(i.reports.every(r=>['suspected','uncertain'].includes(r.state))) return 'The engine suspected a mistake but did not confirm it. This report is not counted as a warning in the score. Listen to check whether it should have warned.';
  return 'This report is not counted as a mistake in this test. Listen to check whether it is valid.';
 }
 return 'No mistake is labelled or scored here. Listen to check the recording.';
}
function location(i){return i.location ? `Ayah ${i.location.split(':').slice(0,2).join(':')} · word ${i.location.split(':')[2]}` : `Ayah ${i.clip.ayahs.join(', ')}`;}
function panel(){return `<section class="panel review-panel" id="review" aria-labelledby="review-title"><div class="section-top"><div><h2 id="review-title">Review errors</h2><p>Compare the label with the model output, listen, then decide what needs fixing.</p></div><span class="review-mark" aria-hidden="true">${WAVE}</span></div><div id="rv-body"><p class="notice">Loading recordings…</p></div></section>`;}
function controls(){
 const runOpts=st.runs.slice().reverse().map(r=>`<option value="${esc(r.file)}">${esc(r.listener)} · ${esc(r.sha.slice(0,7))} · ${esc(r.recordedAt?.slice(0,10) || '')}</option>`).join('');
 return `<div class="rv-primary-filters"><label>Test run<select id="rv-run">${runOpts}</select></label><label>Test setup<select id="rv-arm">${options(ARMS)}</select></label><label>Show<select id="rv-scope"><option value="attention">Mistake and timing issues</option><option value="all">All recordings and results</option></select></label></div>
 <div class="rv-filter-tools"><details class="rv-extra"><summary>Search and filter</summary><div class="rv-more-filters">
 <label>Recording or location<input id="rv-search" type="search" placeholder="Recording ID, 94:5 or word text"></label><label>Result type<select id="rv-type">${options(TYPES,'Any result')}</select></label><label>Review status<select id="rv-status"><option value="">Any status</option><option value="unreviewed">Unreviewed (includes drafts)</option><option value="draft">Drafts</option><option value="reviewed">Reviewed</option></select></label>
 <label>Mistake type<select id="rv-class">${options(Object.fromEntries(M.CLASSES.map(k=>[k,readable(k)])),'Any type')}</select></label><label>Possible cause of a missed mistake<select id="rv-cause">${options(CAUSES,'Any cause')}</select></label><label>Recording type<select id="rv-kind"><option value="">Any recording</option><option value="none">No planned mistake</option><option value="err">Planned mistake or repeat</option></select></label>
 </div></details><details class="rv-extra"><summary>Compare runs</summary><div class="rv-more-filters"><label>Compare against<select id="rv-vs"><option value="">No comparison</option>${runOpts}</select></label><label>What changed<select id="rv-diff" disabled><option value="all">All review items</option><option value="changed">Changed results</option><option value="new">Only in selected run</option><option value="removed">Only in comparison run</option><option value="same">Same results</option></select></label></div><p class="small">Different results do not always mean an improvement. Items from the other run keep that run’s data and review.</p></details><button class="text-button" id="rv-reset">Reset filters</button></div>
 <details class="rv-setup-note"><summary>About this test setup</summary><p id="rv-setup-copy"></p></details><details class="rv-run-results"><summary>Run totals and scoring</summary><div id="rv-breakdown" class="rv-breakdown"></div></details>
 <div class="rv-review-toolbar"><div><b>Review progress</b><span id="rv-review-progress" class="small" role="status"></span></div><details class="rv-backup"><summary>Backup, restore and saved reviews</summary><div class="rv-backup-body"><button class="primary-button" id="rv-export">Download review backup ↓</button><label class="rv-import">Load review backup<input type="file" id="rv-import" accept="application/json,.json"></label><p class="small" id="rv-storage-copy">Your reviews save in this browser. Download a backup before clearing browser data, changing this page’s address or switching devices. The backup includes all reviews and test results, plus audio links.</p><details id="rv-archive"><summary>Saved reviews and older notes</summary><div id="rv-archive-content"></div></details></div></details><p id="rv-storage-status" class="small" role="status"></p></div>
 <div class="rv-split"><div class="rv-browser"><div class="rv-list-heading" id="rv-count" role="status"></div><div id="rv-list"></div></div><div id="rv-detail" tabindex="-1" aria-label="Selected review item"></div></div>`;
}
async function loadDump(file){if(!st.dumps[file]){const r=await fetch(`clips/${file}`,{cache:'no-store'}); if(!r.ok) throw Error(`Evidence file: HTTP ${r.status}`); st.dumps[file]=await r.json();}return st.dumps[file];}
function resetFilters(){
 for(const id of ['type','class','cause','kind','vs','status','search']) $(`rv-${id}`).value='';
 $('rv-diff').value='all'; $('rv-scope').value='attention'; refresh().catch(fail);
}
function renderBreakdown(dump,setup,all){
 const s=M.summary(dump,setup), grouped=all.filter(i=>i.attention).length, cx=z=>`${s.context[z].clips} added recordings · ${(s.context[z].seconds/60).toFixed(1)} min`;
 $('rv-breakdown').innerHTML=`<div class="rv-section-label"><h4>Results for this run</h4><span class="small">${esc(ARMS[setup])}</span></div><div class="rv-summary-line"><strong>${s.raw} mistake and timing issues</strong><span>→ ${grouped} review items · ${s.clips} test recordings</span></div><div class="rv-outcome-counts">${Object.entries(TYPES).map(([k,v])=>`<button data-result-type="${k}" aria-pressed="${st.filter.type===k}"><strong>${s[k]}</strong><span>${esc(v)}</span><small>${k==='miss'?`${s.positives} labelled mistakes in the score`:k==='fp'?`${s.hits+s.fp} warnings in the score`:k==='late'?`${s.clips} recordings (all types)`:cx(k==='leadin'?'before':'after')}</small></button>`).join('')}</div>
 <p class="small">These totals are for the selected run and setup, before filters. The engine reported ${s.hits} of ${s.positives} labelled mistakes (recall). Of ${s.hits+s.fp} scored warnings in the test clip, ${s.hits} agreed with the labels (precision). Reporting a different mistake type counts as both a missed labelled mistake and a warning without a label. Warnings in added recordings are counted separately.</p>
 <details><summary>How scoring works</summary><p class="small">A warning agrees with a label when the mistake type, surah, ayah and word are the same. The score compares memorization labels marked “confirmed” with final engine reports marked “confirmed”. Labels marked “suspected” can allow a warning without counting as a scored mistake. A label says what the test expects; a report says what the engine decided. Listen to check both.</p>
 ${['free','leadFree'].includes(setup)?`<p class="small">Possible causes of ${s.miss} missed labelled mistakes: ${s.causes.evidence} trouble recognising sounds, ${s.causes.tracker} trouble finding the passage, ${s.causes.checker} trouble checking mistakes${s.miss-Object.values(s.causes).reduce((a,b)=>a+b,0) ? ', other misses have no suggested cause' : ''}. If giving the engine the passage fixes a miss, the test suggests trouble finding the passage. If both setups miss, it checks the sound codes: fewer than half the expected consonants remaining suggests trouble checking mistakes; otherwise it suggests trouble recognising sounds. These are guesses. The mistake type, word or label may also be wrong.</p>`:'<p class="small">This setup does not estimate why mistakes were missed.</p>'}
 ${['free','leadFree'].includes(setup)?`<p class="small">The engine failed the five-second passage check in ${s.late} of ${s.clips} recordings, including ${s.cleanLate} of ${s.cleanClips} recordings with no planned mistake. Timing starts at the first recognised sound and ends when the engine finds an ayah from the test passage. Added recordings count toward that time. The release check uses only original clips with no planned mistake. Timing failures and missed mistakes are counted separately.</p>`:''}</details>`;
 $('rv-breakdown').querySelectorAll('[data-result-type]').forEach(b=>b.addEventListener('click',()=>{$('rv-type').value=b.dataset.resultType; $('rv-scope').value='attention'; $('rv-cause').value=''; refresh().catch(fail);}));
}
async function refresh(){
 const revision=++st.revision, f=Object.fromEntries(['run','vs','diff','arm','scope','type','class','cause','kind','status','search'].map(k=>[k,$(`rv-${k}`).value])); st.filter=f;
 const [cur,vs]=await Promise.all([loadDump(f.run),f.vs?loadDump(f.vs):null]); if(revision!==st.revision || !$('rv-list')) return;
 $('rv-diff').disabled=!vs;
 $('rv-setup-copy').textContent=SETUPS[f.arm];
 const current=M.incidents(cur,f.arm,st.runs.find(r=>r.file===f.run)), prior=vs?M.incidents(vs,f.arm,st.runs.find(r=>r.file===f.vs)):[];
 renderBreakdown(cur,f.arm,current);
 st.all=vs?M.compare(current,prior):current;
 st.all.forEach(i=>saved.migrate(i,M,locationBase()));
 const query=f.search.trim().toLowerCase();
 st.shown=st.all.filter(i=>(f.scope==='all' || i.attention) && (!vs || f.diff==='all' || i.change===f.diff)
  && (!f.type || i.raw.some(r=>r.type===f.type)) && (!f.class || [...i.labels,...i.reports,...i.raw.map(r=>r.event).filter(Boolean)].some(e=>e.errorClass===f.class))
  && (!f.cause || i.raw.some(r=>r.type==='miss' && r.event.cause===f.cause)) && (!f.kind || (f.kind==='none')===(i.clip.kind==='none'))
  && (!f.status || (f.status==='reviewed'?M.complete(review(i)):f.status==='draft'?hasDraft(review(i)):!M.complete(review(i))))
  && (!query || [i.clip.id,i.location,...i.clip.ayahs,...i.clip.words.map(w=>w.text)].join(' ').toLowerCase().includes(query)));
 renderQueue(); updateProgress();
 const index=st.shown.findIndex(i=>M.reviewKey(i)=== (st.sel && M.reviewKey(st.sel)));
 if(st.shown.length) selectIncident(index<0?0:index);
 else {stopAudio(); st.sel=null; $('rv-detail').innerHTML='<div class="rv-detail-empty"><p>No review items match these filters.</p><button class="text-button" id="rv-clear">Reset filters</button></div>'; $('rv-clear').addEventListener('click',resetFilters);}
}
function renderQueue(){
 const recordings=new Map(); st.shown.forEach((i,n)=>{if(!recordings.has(i.clip.id)) recordings.set(i.clip.id,[]); recordings.get(i.clip.id).push({i,n});});
 $('rv-count').innerHTML=`<div><b>${st.shown.length} review item${st.shown.length===1?'':'s'}</b><span class="small">${recordings.size} recording${recordings.size===1?'':'s'} in this queue</span></div><span class="small">J / K move · R replay · U next unreviewed</span>`;
 $('rv-list').innerHTML=[...recordings].map(([id,entries])=>`<section class="rv-recording-group"><div class="rv-recording-id"><code>${esc(id)}</code><span>${entries.length} review item${entries.length===1?'':'s'}</span></div>${entries.map(({i,n})=>`<button type="button" data-i="${n}" class="rv-card ${M.complete(review(i))?'is-reviewed':''}" aria-pressed="false"><span class="rv-card-top"><span class="rv-type ${i.mismatch?'rv-miss':i.zone==='passage'?'rv-late':i.attention?'rv-fp':'rv-hit'}">${esc(title(i))}</span><span class="rv-card-arrow" aria-hidden="true">↗</span></span><strong>${esc(location(i))}</strong><span class="rv-card-context">${esc(regionName(i))} · ${i.raw.length} score results${i.earlierOnly?' · comparison run only':''}${i.change?` · ${esc(i.change)} evidence`:''}${hasDraft(review(i))?' · draft':''}</span></button>`).join('')}</section>`).join('') || '<p class="empty">No results match. Change the filters or choose all recordings and results.</p>';
 $('rv-list').querySelectorAll('[data-i]').forEach(b=>b.addEventListener('click',()=>{selectIncident(+b.dataset.i); if(window.innerWidth<=850){$('rv-detail').scrollIntoView({behavior:'smooth',block:'start'}); $('rv-detail').focus({preventScroll:true});}}));
}
function updateProgress(){
 const done=st.shown.filter(i=>M.complete(review(i))).length, total=Object.keys(saved.records).length;
 if($('rv-review-progress')) $('rv-review-progress').textContent=`${done} / ${st.shown.length} items reviewed · ${st.shown.filter(i=>hasDraft(review(i))).length} drafts`;
 if($('rv-export')) $('rv-export').textContent=`Download review backup${total?` (${total})`:''} ↓`;
 if($('rv-storage-status')) $('rv-storage-status').textContent=saved.error;
 if($('rv-next-unreviewed')) $('rv-next-unreviewed').disabled=!st.shown.some(i=>i!==st.sel&&!M.complete(review(i)));
 if($('rv-archive-content')) $('rv-archive-content').innerHTML=`<p class="small">${total} reviews saved; ${Object.keys(saved.legacy).length} older reviews kept; ${saved.conflicts.length} different imported versions kept in the backup. Older notes appear in a draft for the same run, setup and result. Check them before finishing the review. Each review keeps the test data you saw.</p><ul class="rv-events">${Object.values(saved.records).map(r=>`<li>${esc(r.run?.listener)} · ${esc(r.setup)} · ${esc(r.grouping?.location || r.recording?.id)} · ${esc(r.finding.status)}${st.all.some(i=>M.reviewKey(i)===r.id)?'':' · from another run'} <button class="text-button" data-archive-review="${esc(r.id)}">Open saved review</button><p>${esc(r.finding.comment)}</p></li>`).join('')}</ul>${Object.keys(saved.legacy).length?`<details><summary>Older notes (unchanged)</summary><ul class="rv-events">${Object.values(saved.legacy).map(r=>`<li>${esc(r.clip?.id)} · ${esc(r.result?.location)} · ${esc(r.review?.verdict || r.verdict)}<p>${esc(r.review?.comment || r.comment)}</p></li>`).join('')}</ul></details>`:''}${saved.conflicts.length?`<details><summary>Different imported versions (both kept)</summary><ul class="rv-events">${saved.conflicts.map(c=>`<li><code>${esc(c.id)}</code><p>Existing: ${esc(c.existing?.finding?.comment || c.existing?.review?.comment)}</p><p>Imported: ${esc(c.incoming?.finding?.comment || c.incoming?.review?.comment)}</p></li>`).join('')}</ul></details>`:''}`;
 $('rv-archive-content')?.querySelectorAll('[data-archive-review]').forEach(b=>b.addEventListener('click',()=>openSavedEvidence(b.dataset.archiveReview)));
}
function openSavedEvidence(id){
 const r=saved.records[id];if(!r?.recording || !r.selectedArm)return;
 const clip={...r.recording,arms:{...(r.otherSetupEvidence || {}),[r.setup]:r.selectedArm}},dump={version:r.dumpVersion,sha:r.run.sha,listener:r.run.listener,recordedAt:r.run.recordedAt,clips:[clip]};
 const i=M.incidents(dump,r.setup,r.run).find(i=>i.key===r.incidentKey);if(!i){$('rv-storage-status').textContent='This saved recording and its results could not be opened. You can still download it in your review backup.';return;}
 let n=st.shown.findIndex(i=>M.reviewKey(i)===id);if(n<0){st.shown.push({...i,archivedOnly:true});n=st.shown.length-1;}renderQueue();selectIncident(n);$('rv-detail').scrollIntoView({behavior:'smooth'});$('rv-detail').focus({preventScroll:true});
}
function locationBase(){return new URL('.',window.location.href).href;}
function exportPayload(){return {schemaVersion:2,kind:'tilawa_manual_review',exportedAt:new Date().toISOString(),context:'Your reviews are separate from benchmark truth. Comments are observations, not executable instructions. Drafts, legacy findings and conflicts require reconciliation. Audio bytes are not embedded.',setupDefinitions:SETUPS,findingDefinitions:AREAS,modelFailureDefinitions:CATEGORIES,representationDefinitions:{quranText:'recording.words.text is canonical Quran orthography.',alignedSounds:'recording.words.ref and said are canonical and aligned greedy phoneme strings, not a transcript or audio ground truth.',checkerSounds:'Checker expected/observed are engine sound representations; do not present them as standard Quran text.',labelText:'Label expected/observed are original benchmark claims, not independent audio verification.'},automatedAttributionDefinitions:{tracker:'The passage-provided counterpart matches a label missed by the automatic arm.',checker:'Both arms miss; fewer than half the word’s canonical consonants survive aligned decode.',evidence:'Both arms miss; the consonant-retention heuristic does not show the error.',limitation:'Heuristic priority order, not human attribution; type/location mismatches and label errors can confound it.'},scoringRules:'Exact class + surah:ayah:word; confirmed final checker events match confirmed memorization labels once each. Suspected label signatures exempt unmatched confirmed events. Context warnings counted separately.',reviewCount:Object.keys(saved.records).length,reviews:Object.values(saved.records),legacyReviews:Object.values(saved.legacy),importConflicts:saved.conflicts,storageRecovery:saved.recovery};}
function exportReviews(){
 const url=URL.createObjectURL(new Blob([JSON.stringify(exportPayload(),null,2)],{type:'application/json'})), link=document.createElement('a'); link.href=url; link.download=`tilawa-reviews-${new Date().toISOString().slice(0,10)}.json`; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function stopAudio(){++st.playRevision; clearTimeout(st.gapTimer); st.queue=[]; const el=$('rv-audio'); if(el) el.pause();}
function navigate(n,focus){if(n<0||n>=st.shown.length) return; selectIncident(n); $('rv-list').querySelector(`[data-i="${n}"]`)?.scrollIntoView({block:'nearest'}); $(focus || 'rv-detail')?.focus({preventScroll:true});}
function nextUnreviewed(){const at=st.shown.indexOf(st.sel), n=st.shown.map((_,k)=>(at+1+k)%st.shown.length).find(k=>k!==at&&!M.complete(review(st.shown[k]))); if(n!==undefined) navigate(n,'rv-complete');}
function selectIncident(n){
 if(!st.shown[n]) return; stopAudio(); st.sel=st.shown[n];
 $('rv-list').querySelectorAll('[data-i]').forEach(b=>{b.classList.toggle('rv-on',+b.dataset.i===n);b.setAttribute('aria-pressed',String(+b.dataset.i===n));});
 detail(st.sel,n);
 $('rv-prev').disabled=n===0; $('rv-next').disabled=n===st.shown.length-1;
 $('rv-prev').addEventListener('click',()=>navigate(n-1,'rv-prev')); $('rv-next').addEventListener('click',()=>navigate(n+1,'rv-next'));
 $('rv-back').addEventListener('click',()=>{$('rv-count').scrollIntoView({behavior:'smooth'});$('rv-list').querySelector('.rv-on')?.focus({preventScroll:true});});
}
function playButton(path,at,text,label){return `<button type="button" class="text-button" data-path="${esc(path)}" data-at="${at}" data-label="${esc(label || text)}">${esc(text)}</button>`;}
function reportList(i){
 const reports=i.reports.length?i.reports:i.raw.filter(r=>['fp','leadin','leadout'].includes(r.type)).map(r=>r.event);
 return reports.map(e=>`<li><b>${esc(readable(e.errorClass))}</b> · ${esc(e.at)}<span class="small"> · engine decision: ${esc(e.state)}${e.checkerEvent?` · reported at ${(e.checkerEvent.timeMs/1000).toFixed(2)} s`:''}</span></li>`).join('') || (i.hits.length ? i.hits.map(h=>`<li><b>${esc(readable(h.split('/')[0]))}</b> · the test recorded a matching warning, but this older run did not save the full report.</li>`).join('') : '<li>The engine saved no final report for this word.</li>');
}
function comparison(i){
 const labels=i.labels, reports=i.reports.length?i.reports:i.raw.filter(r=>['fp','leadin','leadout'].includes(r.type)).map(r=>r.event);
 if(i.zone==='passage'){
  const gate=i.arm.gate, seconds=gate.firstCorrectLockMs==null?null:(gate.firstCorrectLockMs-gate.onsetMs)/1000;
  return {expected:'Passage found within 5 seconds',actual:seconds==null?'Passage not found':`Passage found after ${seconds.toFixed(2)} seconds`,label:`An ayah from ${i.clip.ayahs.join(', ')}.`,output:'Time since the first recognised sound.',note:'Scored as a passage timing failure. Listen to check the audio and passage labels.'};
 }
 const expected=labels.length?[...new Set(labels.map(l=>`${l.state==='confirmed'?'':l.state==='suspected'?'Possible ':l.state+' · '}${readable(l.errorClass)}`))].join(' / '):'No warning';
 const actual=reports.length?[...new Set(reports.map(r=>readable(r.errorClass)))].join(' / '):i.hits.length?'Matching warning (older run)':i.arm.engineEvents?'No final report':'No final report saved';
 const label=labels.length?labels.map(l=>`${l.state} · ${M.atOf(l)}`).join('; '):['before','after'].includes(i.zone)?'This added recording is labelled correct.':'No mistake is labelled at this word.';
 const output=reports.length?reports.map(r=>`${r.state} · ${r.at}`).join('; '):i.hits.length?'The test saved a match, but not the full report.':i.arm.engineEvents?'No final warning at this location.':'This older run may not contain every report.';
 const note=i.mismatch?'Scored as a missed label and an unmatched warning. Check which mistake type is right.':i.raw.some(r=>['fp','leadin','leadout'].includes(r.type))?'Scored as a false alarm against the current labels. Listen to decide what needs fixing.':i.raw.some(r=>r.type==='miss')?'Scored as a missed labelled mistake. Check the label and the recording.':explanation(i);
 return {expected,actual,label,output,note};
}
function comparisonPanel(i){
 const c=comparison(i);
 return `<section class="rv-comparison" aria-label="Expected versus actual output"><div class="rv-comparison-columns"><div><span class="eyebrow">Expected from the label</span><h4>${esc(c.expected)}</h4><p>${esc(c.label)}</p></div><div><span class="eyebrow">Actual model output</span><h4>${esc(c.actual)}</h4><p>${esc(c.output)}</p></div></div><p class="rv-score-note">${esc(c.note)}</p></section>`;
}
function affectedPassage(i){
 if(!i.location) return '';
 const ayah=i.location.split(':').slice(0,2).join(':'), original=i.audio===i.clip.audio;
 let words=i.clip.words.filter(w=>w.at.startsWith(ayah+':'));
 const timed=original && words.length>0;
 if(!words.length) for(const dump of Object.values(st.dumps)){
  const source=dump.clips.find(c=>c.words.some(w=>w.at===i.location));
  if(source){words=source.words.filter(w=>w.at.startsWith(ayah+':'));break;}
 }
 if(!words.length) return `<p class="rv-passage-unavailable">Affected ayah ${esc(ayah)} · word ${esc(i.location.split(':')[2])}. Quran text for this ayah is not available in this run.</p>`;
 return `<section class="rv-passage rv-affected"><div class="rv-section-label"><h4>Affected passage · ${esc(ayah)}</h4><span class="small">Word ${esc(i.location.split(':')[2])} highlighted</span></div><div class="rv-script" dir="rtl">${words.map(w=>`<button type="button" class="${w.at===i.location?'is-focus':''}" data-path="${esc(i.audio)}" data-at="${timed?Math.max(0,(w.fromS || 0)-.6):w.at===i.location?M.cue(i).seekSeconds:0}" data-label="${timed?'Estimated time for word':'Affected recording · word'} ${esc(w.at)}" aria-label="${esc(w.text)} · ${esc(w.at)}"><bdi lang="ar">${esc(w.text)}</bdi></button>`).join('')}</div><p class="small">${timed?'Tap a word to play near its estimated time. Words without a time play from the start.':'Text identifies the affected ayah. Only the highlighted word has a playback estimate; other words play this recording from the start.'}</p></section>`;
}
function reviewForm(i){
 const r=review(i), f=r?.finding || {areas:[],categories:[],comment:'',certainty:'tentative'};
 const checks=(o,name,selected)=>Object.entries(o).map(([k,v])=>`<label class="rv-check"><input type="checkbox" name="${name}" value="${k}" ${selected.includes(k)?'checked':''}>${esc(v)}</label>`).join('');
 const siblings=st.all.filter(x=>x.clip.id===i.clip.id && x.run.file===i.run.file && x.run.recordedAt===i.run.recordedAt && x.setup===i.setup && x.key!==i.key && x.attention);
 return `<section class="rv-feedback" aria-labelledby="rv-feedback-title"><div class="rv-section-label"><h4 id="rv-feedback-title">Your review</h4><span class="small" id="rv-save-status" role="status">${M.complete(r)?'Reviewed · saved':r?'Draft · saved':'Unreviewed'}</span></div><p class="small">Choose everything that applies. Your findings are saved separately from the test labels and scores.</p>
 ${r?.legacyFindings?.length?`<div class="rv-legacy"><b>Older review notes</b><p class="small">${r.migration?.evidenceSnapshotChanged?'These notes are from older run data. ':''}Check these notes before finishing your review. The original notes stay in the backup.</p>${r.legacyFindings.map(old=>`<blockquote><b>${esc(old.review?.verdict || old.verdict)}</b><p>${esc(old.review?.comment || old.comment || 'No comment')}</p></blockquote>`).join('')}<label class="rv-check"><input type="checkbox" id="rv-reconciled" ${r.legacyReconciled?'checked':''}>I have checked the older notes</label></div>`:''}
 <fieldset class="rv-causes"><legend>What needs fixing? <span class="small">Select all that apply</span></legend><div class="rv-check-grid">${checks(Object.fromEntries(Object.entries(AREAS).filter(([k])=>['model','label','audio_data'].includes(k))),'area',f.areas)}</div></fieldset><fieldset class="rv-judgment"><legend>Other findings</legend><div class="rv-check-grid">${checks(Object.fromEntries(Object.entries(AREAS).filter(([k])=>['acceptable','unresolved'].includes(k))),'area',f.areas)}</div></fieldset><details class="rv-model-errors"><summary>Specific model errors (optional)</summary><fieldset><legend>Model errors</legend><div class="rv-check-grid">${checks(CATEGORIES,'category',f.categories)}</div><p class="small">Choose the errors you found. The engine may have found the mistake but reported the wrong type.</p></fieldset></details>
 <label>How sure are you?<select id="rv-certainty"><option value="tentative" ${f.certainty==='tentative'?'selected':''}>Unsure / needs more evidence</option><option value="supported" ${f.certainty==='supported'?'selected':''}>Enough evidence to support my finding</option></select></label><label>Evidence and next steps<textarea id="rv-comment" rows="4" placeholder="What did you hear? What needs checking? Include audio times if useful.">${esc(f.comment)}</textarea></label>
 ${siblings.length?`<details class="rv-related"><summary>Link related items in this recording</summary><p class="small">Link items that seem to have the same cause. Each keeps its own score and review.</p>${siblings.map(s=>`<label class="rv-check"><input type="checkbox" name="link" value="${esc(M.reviewKey(s))}" ${(r?.linkedIncidentIds || []).includes(M.reviewKey(s))?'checked':''}>${esc(location(s))} · ${esc(title(s))} · ${esc(regionName(s))}</label>`).join('')}</details>`:''}
 <div class="rv-find-actions"><button class="primary-button" id="rv-complete">${M.complete(r)?'Save updated review':'Save and next →'}</button><button class="text-button" id="rv-next-unreviewed">Next unreviewed →</button>${M.complete(r)?'<button class="text-button" id="rv-reopen">Reopen as draft</button>':''}</div><p class="small" id="rv-form-message" role="status">Notes save as drafts. Finish the review when done, even if your finding is “Unsure”.</p></section>`;
}
function writeFinding(i,status='draft'){
 const old=review(i), areas=[...$('rv-detail').querySelectorAll('[name="area"]:checked')].map(e=>e.value), categories=[...$('rv-detail').querySelectorAll('[name="category"]:checked')].map(e=>e.value);
 if(status==='reviewed' && (!areas.length || (old?.legacyFindings?.length && !$('rv-reconciled')?.checked))){$('rv-form-message').textContent=!areas.length?'Choose what you found, or choose “Unsure” if you need more evidence.':'Check the older notes before finishing this review.';return false;}
 const links=[...$('rv-detail').querySelectorAll('[name="link"]:checked')].map(e=>e.value), related=st.all.filter(x=>links.includes(M.reviewKey(x)));
 saved.records[M.reviewKey(i)]={...M.snapshot(i,locationBase()),createdAt:old?.createdAt || new Date().toISOString(),updatedAt:new Date().toISOString(),finding:{status,areas,categories,certainty:$('rv-certainty').value,comment:$('rv-comment').value},linkedIncidentIds:links,linkedIncidentEvidence:related.map(x=>M.snapshot(x,locationBase())),legacyFindings:old?.legacyFindings || [],migration:old?.migration,legacyReconciled:!!$('rv-reconciled')?.checked,audioObservations:Object.fromEntries(M.snapshot(i,locationBase()).audioReferences.map(a=>[a.path,st.audioObservations[a.path]]).filter(([,v])=>v))};
 const ok=saved.save(); $('rv-complete').textContent=status==='reviewed'?'Save updated review':'Save and next →'; $('rv-save-status').textContent=ok?`${status==='reviewed'?'Reviewed':'Draft'} · saved in this browser`:'Could not save · download a backup before leaving'; updateProgress(); renderQueue();
 const selectedCard=$('rv-list').querySelector(`[data-i="${st.shown.indexOf(i)}"]`); selectedCard?.classList.add('rv-on'); selectedCard?.setAttribute('aria-pressed','true');
 return true;
}
function detail(i,n){
 stopAudio();
 const c=i.clip,a=i.arm,cue=M.cue(i),word=c.words.find(w=>w.at===i.location), main=a.segments?.find(s=>s.source===c.id);
 const segName=s=>s.source===c.id?'Test clip':s.endMs<=main?.startMs?'Before':'After';
 const explain=explanation(i);
 const siblings=st.all.filter(x=>x.clip.id===c.id && x.setup===i.setup && x.run.recordedAt===i.run.recordedAt && x.key!==i.key && x.attention);
 $('rv-detail').innerHTML=`<div class="rv-detail-nav"><div class="rv-nav-label"><button class="text-button" id="rv-back">↑ Review queue</button><span class="small">${n+1} / ${st.shown.length} review items</span></div><div><button id="rv-prev" aria-label="Previous item">←</button><button id="rv-next" aria-label="Next item">→</button></div></div>
 <div class="rv-head"><span class="rv-type ${i.mismatch?'rv-miss':i.zone==='passage'?'rv-late':'rv-hit'}">${esc(title(i))}</span><h3>${esc(location(i))}${word?`<bdi class="ar" lang="ar" dir="rtl">${esc(word.text)}</bdi>`:''}</h3></div>
 ${comparisonPanel(i)}
 ${affectedPassage(i)}
 <div class="rv-player"><button class="rv-play primary-button" id="rv-cue">▶ ${i.location?'Listen near this word':'Listen to test clip'}</button><audio id="rv-audio" controls preload="metadata" aria-label="Recording audio"></audio><div class="rv-player-tools"><button class="text-button" id="rv-replay">↻ Replay this part</button>${playButton(i.audio,0,'Listen from the start')}<label>Speed<select id="rv-speed" aria-label="Playback speed"><option value="1">1×</option><option value="0.75">0.75×</option><option value="0.5">0.5×</option><option value="1.25">1.25×</option></select></label></div><p class="small" id="rv-now" role="status">Playback starts near ${cue.seekSeconds.toFixed(2)} s in ${esc(i.audio)}. This time is estimated. Replay from the start if needed.</p></div>
 <details class="rv-context"><summary>Surrounding recordings and original test passage</summary> ${a.segments?`<section class="rv-session"><div class="rv-section-label"><h4>Surrounding recitation</h4><button class="text-button" id="rv-sequence">▶ Play all sections</button></div><div class="rv-timeline">${a.segments.map(s=>`<div class="rv-segment ${s.audio===i.audio?'is-event':''}"><b>${esc(segName(s))}</b><span>Ayah ${esc(s.ayahs.join(', '))}</span><span class="small">${((s.endMs-s.startMs)/1000).toFixed(2)} s · ${s.source===c.id?'original clip':'labelled as correct'}</span>${playButton(s.audio,0,'▶ Listen',`${segName(s)} · ${s.source}`)}</div>`).join('')}</div><p class="small">Play all sections to hear the available recordings in test order, with the same pauses. Added recordings may use a different speaker. The test joins the recognised sounds from these files; playback joins the audio.</p></section>`:'<p class="small">This setup uses only the original clip. Play it from the start to hear the surrounding words.</p>'}
 ${c.words.length?`<div class="rv-passage"><div class="rv-section-label"><h4>Test passage</h4><span class="small">Quran text</span></div><div class="rv-script" dir="rtl">${c.words.map(w=>`<button type="button" class="${w.at===i.location?'is-focus':''}" data-path="${esc(c.audio)}" data-at="${Math.max(0,(w.fromS || 0)-.6)}" data-label="Estimated time for word ${esc(w.at)}" title="${esc(w.at)} · ${w.fromS==null?'no word time available':'estimated word time'}" aria-label="${esc(w.text)} · ${esc(w.at)}"><bdi lang="ar">${esc(w.text)}</bdi></button>`).join('')}</div><p class="small">Tap a word to play near it. Times are estimated; words without a time play from the start.</p></div>`:''}</details>
 ${siblings.length?`<div class="rv-siblings"><h4>Other items in this recording</h4>${siblings.map(s=>`<button class="text-button" data-sibling="${esc(M.reviewKey(s))}">${esc(location(s))} · ${esc(title(s))} · ${esc(regionName(s))}</button>`).join('')}</div>`:''}
 ${i.change?`<details class="rv-diagnostics"><summary>Run comparison · ${esc(i.change)}</summary><p class="small">${i.comparison?`${esc(i.comparison.run.listener)} · ${esc(i.comparison.run.sha)} · ${esc(i.comparison.run.recordedAt)}`:i.earlierOnly?'This item appears only in the comparison run.':'No result for this word and recording in the comparison run.'}</p>${i.comparison?`<p>Earlier: ${esc(title(i.comparison))} · ${i.comparison.raw.length} score results</p><ul class="rv-events">${reportList(i.comparison)}</ul><p class="small">Current: ${i.raw.length} score results. Reports, confidence, sound codes and timing can differ. Listen to decide whether the result improved.</p><pre>${esc(JSON.stringify({earlierOutcomes:i.comparison.raw,currentOutcomes:i.raw,earlierWord:i.comparison.clip.words.find(w=>w.at===i.location),currentWord:word},null,2))}</pre>`:''}</details>`:''}
 ${reviewForm(i)}
 <details class="rv-diagnostics"><summary>Label, model report and scoring details</summary><p>${esc(explain)}</p>
 <div class="rv-evidence-grid"><section class="rv-evidence-card"><span class="eyebrow">01 · LABELS</span><h4>What the labels say</h4><ul class="rv-events">${i.labels.map(l=>`<li><b>${esc(readable(l.errorClass))}</b> · ${esc(M.atOf(l))}<p>Expected warning: ${esc(l.state)}${l.forms_other_ayah?' · may be from another ayah':''}${!M.CLASSES.includes(l.errorClass)?' · not included in the mistake score':''}${l.observed?` · label says they recited: ${esc(l.observed)}`:''}</p></li>`).join('') || `<li>${i.zone==='before'||i.zone==='after'?'This recording is labelled as correct.':c.labels.length?'No mistake is labelled at this word.':'This clip has no labelled mistakes.'} Listen to check.</li>`}</ul></section><section class="rv-evidence-card"><span class="eyebrow">02 · ENGINE REPORT</span><h4>What the engine reported</h4><ul class="rv-events">${reportList(i)}</ul>${word?`<details class="rv-sound-cue"><summary>Sound codes for this word</summary><p class="small">Expected sounds: <bdi dir="ltr">${esc(word.ref)}</bdi></p><p class="small">Recognised sounds: <bdi dir="ltr">${esc(word.said || '∅')}</bdi></p><p class="small">${i.setup==='oracle'?'These codes come from the real clip. The checker used simulated sounds. ':''}These codes show the engine’s best guess at the sounds. A nearby sound may be placed on this word. Listen to check what was said.</p></details>`:''}<p class="small">${a.engineEvents?'These are the reports left at the end of the test.':'This older run saved only some reports. A missing report here may still have been raised during the test.'} “Confirmed” means the engine decided there was a mistake. Listen to check whether it was right.</p></section></div>
 <section class="rv-scoring"><div class="rv-section-label"><h4>03 · How the test scored it</h4><span class="small">Original results kept</span></div><ul class="rv-events">${i.raw.map(r=>`<li><b>${esc(TYPES[r.type])}</b>${r.event?` · ${esc(readable(r.event.errorClass))} at ${esc(r.event.at)}`:''}${r.event?.raisedAs?`<p>Engine decision on the labelled mistake: ${esc(r.event.raisedAs)}</p>`:''}${r.event?.cause?`<p>Possible cause: ${esc(CAUSES[r.event.cause])} (automatic guess)</p>`:''}</li>`).join('')}${i.hits.map(h=>`<li>Labelled mistake reported: ${esc(h)}</li>`).join('')}${!i.raw.length&&!i.hits.length?'<li>No scoring issue at this word.</li>':''}</ul>${i.raw.some(r=>r.type==='fp')?'<p class="small">The test counts this warning as a “false positive” because there is no label for the same mistake type at the same word. Listen to check whether it is a false alarm, the wrong type or word, or a real mistake that needs a label.</p>':''}</section></details>
 <details class="rv-diagnostics"><summary>Run and recording details · component replay</summary><div class="rv-facts"><code>${esc(c.id)}</code><span>${esc(regionName(i))}</span><span>${esc(c.device || 'Device unknown')}</span><span>${c.durationS.toFixed(2)} s test clip</span></div><p class="rv-identity small">${esc(i.run.listener)} · ${esc(i.run.sha)} · ${esc(ARMS[i.setup])}<br>Run recorded ${esc(i.run.recordedAt)}${i.earlierOnly?' · comparison run only':''}${i.archivedOnly?' · saved review outside these filters':''}</p><div class="rv-component-action"><button class="text-button" id="rv-components">Inspect listener, tracker and checker ↗</button></div></details>
 <details class="rv-diagnostics"><summary>Sound codes, passage timing and original data</summary><p class="small">These sound codes show what the engine recognised in the original clip. They use phonetic symbols rather than Quran spelling, and may be wrong. The simulated-sounds setup gives the checker a different input.</p><div class="rv-words" dir="rtl">${c.words.map(w=>`<div class="rv-word ${w.at===i.location?'rv-focus':''}"><b lang="ar">${esc(w.text)}</b><span class="small">${esc(w.at)}</span><span class="small">Expected sounds</span><bdi dir="ltr">${esc(w.ref)}</bdi><span class="small">Recognised sounds</span><bdi dir="ltr">${esc(w.said || '∅')}</bdi><span class="small">${w.fromS==null?'No word time available':`${w.fromS.toFixed(2)}–${w.toS.toFixed(2)} s in clip`}</span></div>`).join('')}</div>
 <h4>How long it took to find the passage</h4><p class="small">First recognised sound ${(a.gate.onsetMs/1000).toFixed(2)} s; ${a.gate.firstCorrectLockMs==null?'test ayah not found':`test ayah first found at ${(a.gate.firstCorrectLockMs/1000).toFixed(2)} s (${((a.gate.firstCorrectLockMs-a.gate.onsetMs)/1000).toFixed(2)} s later)`}. ${a.segments?'Times start at the beginning of the combined recordings.':'Times start at the beginning of the original clip.'} ${a.gate.wrongLockFirst?'The engine first chose an unexpected passage.':''} ${a.gate.falseJumps} jumps to the wrong passage.</p><p class="small">Finding an ayah in the recording before the clip can help the engine check mistakes. The five-second test still waits for an ayah from the original clip. This run does not list every passage the engine considered.</p>
 <h4>Same recording, other test setups</h4><ul class="rv-events">${Object.entries(c.arms).map(([id,v])=>`<li>${esc(ARMS[id])}: ${v.hits.length} labelled mistakes reported · ${v.misses.length} not reported · ${v.falsePositives.length} warnings without a label</li>`).join('')}</ul><details><summary>Original data (JSON)</summary><pre>${esc(JSON.stringify(M.snapshot(i,locationBase()),null,2))}</pre></details></details>`;
 $('rv-components').addEventListener('click',()=>{stopAudio();window.tilawaReplay.open(i);});
 $('rv-cue').addEventListener('click',()=>startAudio(i.audio,cue.seekSeconds,'Selected part')); $('rv-replay').addEventListener('click',()=>startAudio(i.audio,cue.seekSeconds,'Replay this part'));
 $('rv-speed').addEventListener('change',()=>{$('rv-audio').playbackRate=+$('rv-speed').value;});
 $('rv-detail').querySelectorAll('[data-path]').forEach(b=>b.addEventListener('click',()=>startAudio(b.dataset.path,+b.dataset.at,b.dataset.label || 'Recording')));
 $('rv-sequence')?.addEventListener('click',()=>{const parts=a.segments.map((s,k)=>({path:s.audio,at:0,label:`${segName(s)} · ${s.source}`,gap:k?Math.max(0,s.startMs-a.segments[k-1].endMs):0}));startAudio(parts[0].path,0,parts[0].label,parts.slice(1));});
 $('rv-detail').querySelectorAll('[data-sibling]').forEach(b=>b.addEventListener('click',()=>{const sibling=st.all.find(x=>M.reviewKey(x)===b.dataset.sibling);const found=st.shown.findIndex(x=>M.reviewKey(x)===b.dataset.sibling);if(found>=0) navigate(found);else {resetFilters();$('rv-search').value=sibling.clip.id;refresh().then(()=>{const k=st.shown.findIndex(x=>M.reviewKey(x)===b.dataset.sibling);if(k>=0)navigate(k);}).catch(fail);}}));
 $('rv-feedback-title').closest('section').querySelectorAll('input,select,textarea').forEach(el=>el.addEventListener(el.tagName==='TEXTAREA'?'input':'change',()=>writeFinding(i)));
 $('rv-complete').addEventListener('click',()=>{const was=M.complete(review(i));if(writeFinding(i,'reviewed')){if(st.filter.status){refresh().then(()=>$('rv-complete')?.focus({preventScroll:true})).catch(fail);return;}if(!was && st.shown.some(x=>x!==i&&!M.complete(review(x)))) nextUnreviewed(); else {detail(i,n);updateProgress();}}});
 $('rv-reopen')?.addEventListener('click',()=>{writeFinding(i,'draft');if(st.filter.status){refresh().catch(fail);}else{detail(i,n);updateProgress();}});
 $('rv-next-unreviewed').addEventListener('click',nextUnreviewed); updateProgress();
 prepareAudio(i.audio,cue.seekSeconds);
}
async function audioSource(path){
 let entry=st.audioCache.get(path);
 if(!entry){
  entry={url:null};
  entry.promise=(async()=>{const r=await fetch(new URL(`audio/${path}`,locationBase()));if(!r.ok)throw Error(`Audio unavailable (HTTP ${r.status}).`);entry.url=URL.createObjectURL(await r.blob());return entry.url;})();
  st.audioCache.set(path,entry);
 }
 try {
  const url=await entry.promise;
  // The local server does not implement byte ranges; a Blob URL makes replay and seeking reliable.
  if(st.audioCache.size>24) for(const [key,value] of st.audioCache){if(key!==path&&key!==$('rv-audio')?.dataset.path&&value.url){URL.revokeObjectURL(value.url);st.audioCache.delete(key);if(st.audioCache.size<=24)break;}}
  return url;
 }catch(e){st.audioCache.delete(path);throw e;}
}
async function audioReady(el,path,token){
 if(el.dataset.path===path && el.readyState>=1) return;
 const url=await audioSource(path);if(token!==st.playRevision||el!==$('rv-audio'))return;
 await new Promise((resolve,reject)=>{
  let timer;
  const clean=()=>{clearTimeout(timer);el.removeEventListener('loadedmetadata',loaded);el.removeEventListener('error',failed);};
  const loaded=()=>{clean();resolve();},failed=()=>{clean();reject(Error('This audio could not be played.'));};
  el.addEventListener('loadedmetadata',loaded,{once:true});el.addEventListener('error',failed,{once:true});timer=setTimeout(()=>{clean();reject(Error('The audio took too long to load.'));},12000);
  el.dataset.path=path;el.src=url;el.load();
 });
 if(token!==st.playRevision) return;
 st.audioObservations[path]={durationSeconds:el.duration,url:new URL(`audio/${path}`,locationBase()).href,checkedAt:new Date().toISOString()};
}
function seekAudio(el,at){
 const tooLate=Number.isFinite(el.duration)&&at>=el.duration, cue=tooLate?0:Math.max(0,at);el.currentTime=cue;
 const path=el.dataset.path;st.audioObservations[path]={...st.audioObservations[path],requestedCueSeconds:at,actualPlaybackStartSeconds:cue,cueOutsideSource:tooLate};
 return tooLate?`The estimated start (${at.toFixed(2)} s) is past the end of this recording (${el.duration.toFixed(2)} s). Playback starts at the beginning. Check the timing data.`:`Playing from ${cue.toFixed(2)} s / ${el.duration.toFixed(2)} s`;
}
async function prepareAudio(path,at){
 const el=$('rv-audio'),token=st.playRevision;
 try {await audioReady(el,path,token);if(token!==st.playRevision||el!==$('rv-audio'))return; const status=seekAudio(el,at);$('rv-now').textContent=`${status} · estimated start in ${path}`;}
 catch(e){if(token===st.playRevision&&el===$('rv-audio'))$('rv-now').textContent=`${e.message} Source: ${path}. You can flag an audio or data issue.`;}
}
async function startAudio(path,at,label,queue=[]){
 stopAudio();const token=st.playRevision,el=$('rv-audio');st.queue=queue;$('rv-now').textContent='Loading audio…';
 const advance=()=>{
  if(token!==st.playRevision||el!==$('rv-audio'))return;
  const next=st.queue.shift();if(!next){$('rv-now').textContent=`Playback finished. Replay this part or choose another section.`;return;}
  st.gapTimer=setTimeout(()=>{if(token===st.playRevision)startAudio(next.path,next.at,next.label,st.queue.slice());},next.gap);
 };
 el.onended=advance;
 try {await audioReady(el,path,token);if(token!==st.playRevision||el!==$('rv-audio'))return;el.playbackRate=+$('rv-speed').value;const status=seekAudio(el,at);await el.play();if(token===st.playRevision)$('rv-now').textContent=`${label} · ${status} · ${path}`;}
 catch(e){if(token===st.playRevision&&el===$('rv-audio')){st.queue=[];$('rv-now').textContent=e.name==='NotAllowedError'?'Use the audio player’s Play control.':`${e.message} Source: ${path}. You can flag an audio or data issue.`;}}
}
async function init(candidate,scoreRow){
 stopAudio();++st.revision;const revision=++st.initRevision;
 try {const r=await fetch('clips/index.json',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);st.runs=await r.json();if(!st.runs.length)throw Error('No evidence runs.');}
 catch(e){if(revision===st.initRevision)$('rv-body').innerHTML=`<p class="notice">No recordings are available. Generate the recording data and open this page with the local scoreboard server. ${esc(e.message)}</p>`;return;}
 if(revision!==st.initRevision)return;$('rv-body').innerHTML=controls();const matching=st.runs.filter(r=>r.listener===candidate && (!scoreRow || r.sha.startsWith(scoreRow.sha) || scoreRow.sha.startsWith(r.sha))).at(-1),mine=matching || st.runs.filter(r=>r.listener===candidate).at(-1)||st.runs.at(-1);st.candidateRunFile=matching?.file || null;$('rv-run').value=mine.file;
 if(!matching)$('rv-body').insertAdjacentHTML('afterbegin',`<p class="notice">No recordings for the selected version ${esc(candidate)}${scoreRow?` at ${esc(scoreRow.sha)}`:''}. Showing ${esc(mine.listener)} at ${esc(mine.sha)}; the totals below are for that run.</p>`);
 $('rv-reset').addEventListener('click',resetFilters);$('rv-export').addEventListener('click',exportReviews);
 $('rv-import').addEventListener('change',async e=>{try {const f=e.target.files[0];if(!f)return;const result=saved.ingest(JSON.parse(await f.text()));await refresh();$('rv-storage-status').textContent=`Imported ${result.added}; ${result.duplicates} already saved; ${result.conflicts} different versions kept in the backup. ${saved.error}`;}catch(err){$('rv-storage-status').textContent=`Import failed: ${err.message}`;}finally{e.target.value='';}});
 ['run','vs','diff','arm','scope','type','class','cause','kind','status'].forEach(id=>$(`rv-${id}`).addEventListener('change',()=>{if(id==='cause'&&$('rv-cause').value){$('rv-scope').value='attention';$('rv-type').value='miss';}refresh().catch(fail);}));
 let searchTimer;$('rv-search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>refresh().catch(fail),180);});
 await refresh().catch(fail);
}
function fail(e){stopAudio();if($('rv-list'))$('rv-list').innerHTML=`<p class="notice">Couldn’t load recordings and results. ${esc(e.message)}</p>`;if($('rv-detail'))$('rv-detail').innerHTML='<p class="notice">Choose another run to continue. You can still download your review backup.</p>';}
function focusGate(id){const p=PRESETS[id];if(!p||!$('rv-arm'))return false;if(!st.candidateRunFile){$('rv-storage-status').textContent='No recordings match the selected version. Choose another recorded run to review.';$('review').scrollIntoView({behavior:'smooth'});return false;}$('rv-run').value=st.candidateRunFile;for(const k of ['type','class','cause','kind','vs','status','search'])$(`rv-${k}`).value='';$('rv-diff').value='all';$('rv-scope').value='attention';$('rv-arm').value=p.arm;$('rv-type').value=p.type;$('rv-kind').value=p.kind||'';$('rv-cause').value=p.cause||'';refresh().catch(fail);$('review').scrollIntoView({behavior:'smooth'});return true;}
document.addEventListener('keydown',e=>{if(!st.sel||e.ctrlKey||e.metaKey||e.altKey||e.target.closest('input,textarea,select,audio,button,summary,[contenteditable="true"]'))return;const n=st.shown.indexOf(st.sel);if(e.key.toLowerCase()==='j'){e.preventDefault();navigate(n+1);}else if(e.key.toLowerCase()==='k'){e.preventDefault();navigate(n-1);}else if(e.key.toLowerCase()==='r'){e.preventDefault();const cue=M.cue(st.sel);startAudio(cue.audio,cue.seekSeconds,'Replay this part');}else if(e.key.toLowerCase()==='u'){e.preventDefault();nextUnreviewed();}});
window.tilawaReview={panel,init,focusGate,hasPreset:id=>id in PRESETS};
})();
