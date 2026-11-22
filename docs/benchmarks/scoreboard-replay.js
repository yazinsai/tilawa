'use strict';
(() => {
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $=id=>document.getElementById(id);
const names={free:'Automatic · clip only',leadFree:'Automatic · surrounding recitation',declared:'Passage provided · clip only',leadDeclared:'Passage provided · surrounding recitation',oracle:'Simulated sounds · checker test'};
const classes={wrong_word:'Wrong word',omitted_word:'Missing word',added_word:'Added word',skipped_passage:'Passage skipped'};
const stageNames={audio:'Audio',listener:'Listener',tracker:'Tracker',checker:'Checker'};
let dialog,context,meta,result,step=0,stage='listener',revision=0,controller,audioRevision=0,playbackRevision=0,gapTimer,focusBefore;
const blobs=new Map();
const sec=n=>`${(n/1000).toFixed(2)} s`;
const word=w=>w?`${w.surah}:${w.ayah} · word ${w.word}`:'No position';
function shell(){
return `<div class="rp-top"><div><span class="eyebrow">CLIP REPLAY</span><h2 id="rp-title">Inside the engine</h2><p id="rp-context"></p></div><button class="rp-close" id="rp-close" aria-label="Close replay">×</button></div>
<p class="small" id="rp-provenance"></p><p class="notice" id="rp-status" role="status">Loading test setup…</p>
<div class="rp-run-controls"><label>Listener input<select id="rp-mode"><option value="saved">Use saved sounds</option><option value="fresh">Run listener again</option></select></label><button class="primary-button" id="rp-run" disabled>Run test</button><details class="rp-settings"><summary>Test settings</summary><p class="small">Change these to test an idea. Saved scores and reviews stay unchanged.</p><div class="rp-settings-grid"><label>Sound step (ms)<input id="rp-chunkMs" type="number" min="40" max="2000" step="40"></label><label>Passage search interval (ms)<input id="rp-acquireMs" type="number" min="40" max="5000" step="40"></label><label>Minimum confidence<input id="rp-minConfidence" type="number" min="0" max="1" step="0.05"></label><label>Warning delay (ms)<input id="rp-confirmMs" type="number" min="0" max="10000" step="100"></label><label class="rv-check"><input id="rp-localWindow" type="checkbox">Check nearby sound probabilities</label><label class="rv-check"><input id="rp-finaliseFree" type="checkbox">Confirm open warnings when the clip ends</label></div><button class="text-button" id="rp-reset">Reset to this run’s settings</button></details></div>
<div id="rp-body" hidden><div class="rp-playback"><div class="rp-section-heading"><h3>Listen and step through</h3><button class="text-button" id="rp-play-all">▶ Play all sections</button></div><div id="rp-segments" class="rp-segments"></div><audio id="rp-audio" controls preload="metadata" aria-label="Replay recording"></audio><p class="small" id="rp-audio-status" role="status"></p><div class="rp-step-controls"><button id="rp-prev" aria-label="Previous step">←</button><input id="rp-step" type="range" min="0" max="0" value="0" aria-label="Replay step"><button id="rp-next" aria-label="Next step">→</button><button class="text-button" id="rp-next-warning">Next warning</button></div><p class="small" id="rp-step-label"></p></div>
<div class="rp-flow" role="tablist" aria-label="Engine components">${Object.entries(stageNames).map(([k,n],i)=>`<button role="tab" id="rp-tab-${k}" aria-controls="rp-component" aria-selected="${k==='listener'}" tabindex="${k==='listener'?0:-1}" data-stage="${k}"><span>${String(i+1).padStart(2,'0')}</span><b>${n}</b><small>${{audio:'Recording',listener:'Recognises sounds',tracker:'Finds the passage',checker:'Checks mistakes'}[k]}</small></button>`).join('')}</div><p class="small rp-route">Audio → Listener. Listener sounds → Tracker and Checker. Tracker position → Checker.</p>
<section id="rp-component" role="tabpanel" aria-labelledby="rp-tab-listener" tabindex="0"></section><details class="rp-data"><summary>Full test settings and source files</summary><pre id="rp-config"></pre></details><button class="text-button" id="rp-download">Download this replay</button></div>`;
}
function ensure(){
 if(dialog)return;
 dialog=document.createElement('dialog');dialog.className='rp-dialog';dialog.setAttribute('aria-labelledby','rp-title');dialog.innerHTML=shell();document.body.append(dialog);
 $('rp-close').addEventListener('click',close);
 dialog.addEventListener('close',cleanup);
 dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)close();}});
 $('rp-run').addEventListener('click',()=>run());
 $('rp-reset').addEventListener('click',()=>setSettings(meta.settings));
 $('rp-prev').addEventListener('click',()=>pickStep(step-1,true));$('rp-next').addEventListener('click',()=>pickStep(step+1,true));
 $('rp-step').addEventListener('input',e=>pickStep(+e.target.value,true));
 $('rp-next-warning').addEventListener('click',()=>{const at=result.steps.findIndex((s,i)=>i>step&&s.checkerOutput.length);if(at>=0)pickStep(at,true);});
 dialog.querySelectorAll('[data-stage]').forEach(b=>{
  b.addEventListener('click',()=>pickStage(b.dataset.stage));
  b.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const keys=Object.keys(stageNames),at=keys.indexOf(stage),next=e.key==='Home'?0:e.key==='End'?keys.length-1:(at+(e.key==='ArrowRight'?1:-1)+keys.length)%keys.length;pickStage(keys[next]);$(`rp-tab-${keys[next]}`).focus();});
 });
 $('rp-audio').addEventListener('timeupdate',()=>{
  audioStatus();if(!$('rp-audio').paused)followAudio();
 });
 $('rp-audio').addEventListener('seeked',()=>{if($('rp-audio').dataset.seekByStep){delete $('rp-audio').dataset.seekByStep;return;}followAudio();});
 $('rp-play-all').addEventListener('click',()=>playSequence());
 $('rp-download').addEventListener('click',()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download=`tilawa-replay-${context.clip}-${context.setup}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});
}
function cleanup(){++revision;++audioRevision;controller?.abort();stopSequence();$('rp-audio').removeAttribute('src');$('rp-audio').load();delete $('rp-audio').dataset.seekByStep;for(const b of blobs.values())URL.revokeObjectURL(b);blobs.clear();result=null;focusBefore?.isConnected&&focusBefore.focus();}
function close(){dialog?.close();}
async function api(action,extra={}){
 controller?.abort();controller=new AbortController();
 const response=await fetch('/api/replay',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...context,action,...extra}),signal:controller.signal});
 if(response.status===404||response.status===501)throw Error('Replay needs the updated local scoreboard server. Restart it, then try again.');
 let data;try{data=await response.json();}catch{throw Error('The local server did not return replay data.');}
 if(!response.ok||data.error)throw Error(data.error||'Replay failed.');return data;
}
function setSettings(s){for(const [k,v]of Object.entries(s)){if(!$(`rp-${k}`))continue;if(typeof v==='boolean')$(`rp-${k}`).checked=v;else $(`rp-${k}`).value=v;}}
function settings(){return Object.fromEntries(Object.keys(meta.settings).map(k=>[k,['localWindow','finaliseFree'].includes(k)?$(`rp-${k}`).checked:Number($(`rp-${k}`).value)]));}
async function open(i){
 ensure();if(dialog.open)dialog.close();focusBefore=document.activeElement;context={run:i.run.file,recordedAt:i.run.recordedAt,clip:i.clip.id,setup:i.setup};
 meta=null;result=null;step=0;stage='listener';$('rp-body').hidden=true;$('rp-run').disabled=true;$('rp-mode').value='saved';$('rp-status').textContent='Loading test setup…';$('rp-context').textContent=`${i.clip.id} · ${names[i.setup]} · ${i.run.listener}`;$('rp-provenance').textContent='';
 dialog.showModal();const token=++revision;
 try{meta=await api('prepare');if(token!==revision)return;setSettings(meta.settings);$('rp-mode').querySelector('[value="fresh"]').disabled=!meta.freshAvailable;
  $('rp-provenance').textContent=`Saved run ${meta.recordedSha}. Replay uses current code ${meta.localSha}${meta.localChanges?' with local changes':''}. ${meta.configurationRecorded?'Using the saved test settings.':'This old run has no saved settings. Using benchmark defaults.'}${!meta.freshAvailable?' No local listener model is available for this setup.':''}`;
  $('rp-run').disabled=false;await run();
 }catch(e){if(token===revision&&e.name!=='AbortError')$('rp-status').textContent=e.message;}
}
async function run(){
 const token=++revision;++audioRevision;$('rp-run').disabled=true;stopSequence();$('rp-body').hidden=true;
 $('rp-status').textContent=$('rp-mode').value==='fresh'?'Running the listener, tracker and checker…':'Replaying saved sounds through the tracker and checker…';
 try{
  const data=await api('run',{listenerMode:$('rp-mode').value,settings:settings()});if(token!==revision)return;result=data;step=0;
  $('rp-body').hidden=false;$('rp-step').max=result.steps.length-1;
  const baseline=data.matchesRecordedReports===null?'This older run has no saved reports to compare.':data.matchesRecordedReports?'Final reports match the recorded run.':'Final reports differ from the recorded run.';
  $('rp-status').textContent=`${data.synthetic?'The checker used simulated sounds from the labels. Their timing does not match the audio.':data.listenerMode==='fresh'?'Listener ran again on each audio file.':'Listener output loaded from the saved sound files.'} ${baseline}${data.changedSettings?' Test settings changed.':''}${!data.sourceMatches?' The saved word sounds differ from the files now on disk.':''}`;
  $('rp-segments').innerHTML=result.segments.map((s,k)=>`<button class="text-button" data-part="${k}">${s.source===context.clip?'Test clip':s.endMs<=result.segments.find(p=>p.source===context.clip)?.startMs?'Before clip':'After clip'} · ${s.ayahs.join(', ')}</button>`).join('');
  $('rp-segments').querySelectorAll('[data-part]').forEach(b=>b.addEventListener('click',()=>{stopSequence();const p=result.segments[+b.dataset.part];const at=result.steps.findIndex(s=>s.toMs>p.startMs);pickStep(Math.max(0,at),false);loadAudio(+b.dataset.part,0).catch(audioFail);}));
  $('rp-config').textContent=JSON.stringify({run:context,recordedSha:data.recordedSha,localSha:data.localSha,localChanges:data.localChanges,listenerMode:data.listenerMode,configurationRecorded:data.configurationRecorded,settings:data.settings,tracker:data.trackerOptions,checker:data.checkerOptions,model:data.model,reference:data.reference,phrases:data.phrases,sources:data.sources},null,2);
  pickStage(stage);pickStep(0,false);await loadAudio(0,0);
 }catch(e){if(token===revision&&e.name!=='AbortError'){$('rp-status').textContent=e.message;$('rp-body').hidden=true;}}
 finally{if(token===revision)$('rp-run').disabled=false;}
}
function pickStage(value){stage=value;dialog.querySelectorAll('[data-stage]').forEach(b=>{const on=b.dataset.stage===stage;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;});$('rp-component').setAttribute('aria-labelledby',`rp-tab-${stage}`);renderComponent();}
function pickStep(at,seek){if(!result)return;step=Math.max(0,Math.min(result.steps.length-1,at));const s=result.steps[step];$('rp-step').value=step;$('rp-step-label').textContent=`Step ${step+1} of ${result.steps.length} · ${sec(s.fromMs)}–${sec(s.toMs)}${s.phase==='end'?' · Clip ended; open reports decided':''}. Times follow the test’s sound frames.`;$('rp-prev').disabled=step===0;$('rp-next').disabled=step===result.steps.length-1;$('rp-next-warning').disabled=!result.steps.slice(step+1).some(s=>s.checkerOutput.length);renderComponent();if(seek){stopSequence();if(!result.synthetic)seekTime(s.fromMs).catch(audioFail);}}
const codes=e=>e?.phonemes?.length?`<div class="rp-phonemes" dir="ltr">${e.phonemes.map(p=>`<span title="${esc(`${sec(p.start*e.frameMs)}–${sec((p.end+1)*e.frameMs)} · confidence ${p.conf.toFixed(3)}`)}">${esc(p.p)}</span>`).join('')}</div>`:'<p class="small">No new sounds in this step.</p>';
const raw=(label,data)=>`<details class="rp-data"><summary>${label}</summary><pre>${esc(JSON.stringify(data,null,2))}</pre></details>`;
const events=es=>es.length?`<ul class="rv-events">${es.map(e=>`<li><b>${esc(e.type==='position'?e.state:classes[e.errorClass]||e.errorClass)}</b> · ${esc(e.type==='position'?word(e.observed[0]?.word):word(e.location))}${e.type==='checker'?` · ${esc(e.state)}`:''}<span class="small"> · ${sec(e.timeMs)}</span></li>`).join('')}</ul>`:'<p class="small">No new reports in this step.</p>';
function renderComponent(){
 if(!result)return;const s=result.steps[step],all=result.steps.slice(0,step+1),positions=all.flatMap(s=>s.trackerOutput).filter(e=>e.type==='position'),last=positions.at(-1),checks=[...new Map(all.flatMap(s=>s.checkerOutput).map(e=>[e.id,e])).values()];
 let input,output;
 if(stage==='audio'){
  input=`<h4>Recordings in this test</h4><ul class="rv-events">${result.segments.map(p=>`<li>${esc(p.audio)} · ${sec(p.startMs)}–${sec(p.endMs)}</li>`).join('')}</ul><p class="small">Use the audio player above to listen.</p>`;
  output=`<h4>Sent to the listener</h4><p>${result.listenerMode==='fresh'?'Each audio file was decoded separately, with 20 ms audio pushes.':'This replay uses saved listener output. Audio is available for listening; the listener was not run again.'}</p><p class="small">Added recordings are joined after decoding, as in the benchmark.</p>`;
 }else if(stage==='listener'){
  const part=result.segments.find(p=>s.fromMs>=p.startMs&&s.fromMs<p.endMs);
  input=`<h4>Audio input</h4><p>${part?esc(part.audio):s.phase==='end'?'End of the test':'Gap between recordings'}</p><p class="small">${result.synthetic?'The checker test uses sounds simulated from labels. This audio is for listening.':result.listenerMode==='saved'?'Saved listener output is being replayed.':'The local model decoded this audio again.'}</p>`;
  output=`<h4>Output phonemes</h4>${codes(s.listenerOutput)}<p class="small">Grouped into the same steps used by the test. These sound codes feed both the tracker and checker.</p>${raw('Full listener output for this step',s.listenerOutput)}`;
 }else if(stage==='tracker'){
  input=`<h4>Input phonemes</h4>${codes(s.trackerInput)}<p class="small">The tracker gets sound codes, timing and confidence. The benchmark does not send it the sound probability table.</p>${raw('Full tracker input',s.trackerInput)}`;
  output=`<h4>Position reports in this step</h4>${events(s.trackerOutput)}<p><b>Current position:</b> ${last?`${esc(last.state)} · ${esc(word(last.observed[0]?.word))}`:'Passage not found yet'}</p>${raw('Full tracker output and candidate scores',s.trackerOutput)}`;
 }else{
  input=`<h4>Sounds and tracker reports</h4>${codes(s.checkerInput.evidence)}<p class="small">${s.checkerInput.evidence?.posteriors?'Sound probability table included.':'No sound probability table in this step.'} ${result.checkerOptions.passage?'The checker was also given the passage.':'The checker follows the tracker’s passage.'}</p>${events(s.checkerInput.trackerEvents)}${raw('Full checker input',s.checkerInput)}`;
  output=`<h4>New mistake reports</h4>${events(s.checkerOutput)}<h4 class="rp-so-far">Reports so far</h4>${events(checks)}${raw('Full checker output for this step',s.checkerOutput)}`;
 }
 $('rp-component').innerHTML=`<h3>${stageNames[stage]} · step ${step+1}</h3><div class="rp-io"><section><span class="eyebrow">INPUT</span>${input}</section><section><span class="eyebrow">OUTPUT</span>${output}</section></div>`;
}
function followAudio(){if(!result||result.synthetic)return;const el=$('rp-audio'),p=result.segments[+el.dataset.part];if(!p)return;const time=p.startMs+el.currentTime*1000;let at=result.steps.findIndex(s=>s.toMs>time);if(at<0)at=result.steps.length-1;pickStep(at,false);}
function audioStatus(){if(!result)return;const el=$('rp-audio'),p=result.segments[+el.dataset.part];if(p&&Number.isFinite(el.duration))$('rp-audio-status').textContent=`${p.audio} · ${sec(el.currentTime*1000)} of ${sec(el.duration*1000)}`;}
function stopSequence(){++playbackRevision;clearTimeout(gapTimer);$('rp-audio').onended=null;$('rp-audio').pause();}
function audioFail(e){if(dialog?.open)$('rp-audio-status').textContent=`Could not play the audio. ${e.message}`;}
async function loadAudio(part,at){
 const token=++audioRevision,p=result.segments[part],el=$('rp-audio');el.pause();
 if(!blobs.has(p.audio)){const r=await fetch(`audio/${p.audio}`);if(!r.ok)throw Error(`HTTP ${r.status}`);const blob=await r.blob();if(token!==audioRevision)return;blobs.set(p.audio,URL.createObjectURL(blob));}
 if(token!==audioRevision)return;
 if(+el.dataset.part!==part||!el.src){el.dataset.part=part;await new Promise((resolve,reject)=>{let timeout;const clean=()=>{clearTimeout(timeout);el.removeEventListener('loadedmetadata',ok);el.removeEventListener('error',bad);};const ok=()=>{clean();resolve();},bad=()=>{clean();reject(Error('Audio format unavailable.'));};el.addEventListener('loadedmetadata',ok,{once:true});el.addEventListener('error',bad,{once:true});timeout=setTimeout(()=>{clean();reject(Error('Audio took too long to load.'));},12000);el.src=blobs.get(p.audio);el.load();});}
 if(token!==audioRevision||!dialog.open)return;
 const time=Math.max(0,Math.min(at,Math.max(0,el.duration-.01)));if(Math.abs(el.currentTime-time)>.001){el.dataset.seekByStep='1';el.currentTime=time;}audioStatus();if(at>=el.duration)$('rp-audio-status').textContent+=' · Step time is past the end of this audio.';
}
async function seekTime(time){let part=result.segments.findIndex(p=>time>=p.startMs&&time<p.endMs);if(part<0)part=result.segments.findIndex(p=>p.startMs>=time);if(part<0)part=result.segments.length-1;await loadAudio(part,Math.max(0,(time-result.segments[part].startMs)/1000));}
async function playSequence(){
 if(!result)return;stopSequence();const token=revision,playToken=playbackRevision;let part=+$('rp-audio').dataset.part||0;
 const next=async()=>{if(token!==revision||playToken!==playbackRevision||!dialog.open)return;const el=$('rp-audio'),at=part===+el.dataset.part&&el.currentTime<el.duration-.02?el.currentTime:0;await loadAudio(part,at);if(token!==revision||playToken!==playbackRevision||!dialog.open)return;el.onended=()=>{if(part+1>=result.segments.length){if(!result.synthetic)pickStep(result.steps.length-1,false);return;}const gap=Math.max(0,result.segments[part+1].startMs-result.segments[part].endMs);part++;gapTimer=setTimeout(()=>next().catch(audioFail),gap);};await el.play();};
 try{await next();}catch(e){audioFail(e);}
}
window.tilawaReplay={open,close};
})();
