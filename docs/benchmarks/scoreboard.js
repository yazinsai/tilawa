'use strict';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const numeric = m => typeof m?.value === 'number' && Number.isFinite(m.value);
const metric = (r,id) => r?.metrics.find(m => m.id === id);
const fmt = (v,u) => v == null ? '—' : u === '%' ? `${(v*100).toFixed(1)}%` : u === 's' ? `${v.toFixed(2)} s` : String(+v.toFixed(3));
const value = m => numeric(m) ? fmt(m.value,m.unit) : 'Not evaluated';
const badge = m => {const s = m?.status || 'pending';return `<span class="pill ${esc(s)}">${({pass:'✓ Meets target',fail:'Outside target',pending:'Awaiting evaluation',unmeasurable:'No measurement',info:'Reference metric'})[s] || esc(s)}</span>`;};
const target = m => m?.bar ? `Target ${esc(m.bar.op === '>=' ? '≥' : '≤')} ${fmt(m.bar.target,m.unit)}` : 'No acceptance target';
const lowerBetter = (id,m) => m?.bar ? m.bar.op === '<=' : /goldPer|perFair|perRef|perTeacher|falseJumps|cleanPer|repeatFp|lockS|perfRtf|\.e0\./.test(id);
function delta(m,prev,id) {
 if(!numeric(m) || !numeric(prev)) return '<span class="small">No earlier measurement</span>';
 const d=m.value-prev.value, improved=lowerBetter(id,m)?d<0:d>0;
 if(Math.abs(d)<1e-10) return '<span class="small">Unchanged from previous run</span>';
 const n=m.unit==='%'?`${Math.abs(d*100).toFixed(1)} pp`:fmt(Math.abs(d),m.unit);
 return `<span class="delta ${improved?'':'worse'}">${improved?'↗ Improved':'↘ Worsened'} by ${n}</span>`;
}
function source(m){
 const path=m?.source;
 // Benchmark sources are local filenames; never interpret data as an arbitrary URL.
 return typeof path==='string' && /^[a-zA-Z0-9_-]+\.md$/.test(path) ? `<a href="${esc(path)}">Backing report ↗</a>` : '';
}
function chart(rows,specs,{compact=false}={}){
 const series=specs.map(s=>({...s,points:rows.map((r,i)=>({r,i,m:metric(r,s.id)}))}));
 if(!series.some(s=>s.points.some(p=>numeric(p.m)))) return '<div class="empty">No results yet. This chart will appear when the evaluation is recorded.</div>';
 const W=compact?360:600,H=compact?160:256,L=48,R=16,T=24,B=40;
 const vals=series.flatMap(s=>s.points.filter(p=>numeric(p.m)).map(p=>p.m.value));
 const bars=series.map(s=>[...s.points].reverse().find(p=>numeric(p.m))?.m.bar);
 const unit=series.flatMap(s=>s.points).find(p=>numeric(p.m)).m.unit;
 const peak=Math.max(...vals,...bars.filter(Boolean).map(b=>b.target));
 const hi=unit==='%'?(compact && specs[0].id==='listener.goldPer'?Math.max(.05,Math.ceil(peak/.05)*.05):Math.max(1,peak)):Math.max(...vals,...bars.filter(Boolean).map(b=>b.target),.01)*1.2;
 const x=i=>rows.length===1?(L+W-R)/2:L+i*(W-L-R)/(rows.length-1),y=v=>H-B-v/hi*(H-T-B);
 let svg=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(specs.map(s=>s.label).join(' and '))} across ${rows.length} recorded runs"><title>${esc(specs.map(s=>s.label).join(' and '))} over recorded runs; gaps mean missing measurements</title>`;
 for(let i=0;i<=4;i++){const v=hi*i/4;svg+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="hsl(0,0%,86%)"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end">${fmt(v,unit)}</text>`;}
 bars.forEach((b,i)=>{if(b)svg+=`<line x1="${L}" x2="${W-R}" y1="${y(b.target)}" y2="${y(b.target)}" stroke="${series[i].color}" stroke-opacity=".6" stroke-dasharray="3 5"><title>${esc(series[i].label)} target: ${target({bar:b,unit})}</title></line>`;});
 series.forEach(s=>{
 // Preserve missing runs as gaps, rather than connecting across absent evidence.
 let segment=[];const flush=()=>{if(segment.length>1)svg+=`<polyline fill="none" stroke="${s.color}" stroke-width="3" ${s.dashed?'stroke-dasharray="7 5"':''} points="${segment.join(' ')}"/>`;segment=[];};
 s.points.forEach(p=>{if(numeric(p.m))segment.push(`${x(p.i)},${y(p.m.value)}`);else flush();});flush();
 s.points.filter(p=>numeric(p.m)).forEach(p=>{svg+=`<circle cx="${x(p.i)}" cy="${y(p.m.value)}" r="5" fill="${s.color}" stroke="white" stroke-width="2"><title>${esc(p.r.date)} · ${esc(p.r.sha)} · ${esc(s.label)}: ${value(p.m)} · ${esc(p.r.note)}</title></circle>`;});
 });
 const indices=rows.length===1?[0]:[0,rows.length-1];indices.forEach(i=>{svg+=`<text x="${x(i)}" y="${H-16}" text-anchor="${rows.length===1?'middle':i===0?'start':'end'}">${esc(rows[i].date)} · ${esc(rows[i].sha.slice(0,7))}</text>`;});
 return svg+'</svg>'+`<div class="chart-caption">${rows.length===1?'One recorded run — a baseline, not a trend.':'Runs ordered by date; same-day runs keep their recorded order. Gaps indicate missing evidence.'} ${bars.some(Boolean)?'Dotted lines show targets.':''}</div>`;
}
const primary='hsl(0,0%,9%)',secondary='hsl(0,0%,45%)';
function stat(row,prev,id,label,description){const m=metric(row,id);return `<div><div class="metric-label">${label}</div><div class="number ${numeric(m)?'':'unavailable'}">${value(m)}</div><p>${description}</p><div class="target">${target(m)}</div>${delta(m,metric(prev,id),id)}</div>`;}
const MODULES = ['listener','tracker','checker'];
const moduleLabels = {listener:'Listener',tracker:'Tracker',checker:'Checker'};
const impacted = row => MODULES.filter(m => row.modules?.includes(m));
const impactMetric = {listener:['listener.goldPer','Listener · sound error rate'],tracker:['tracker.clean.within5s','Tracker · correct ayah ≤ 5 s'],checker:['checker.oracleAll.recall','Checker · isolated recall']};
function historyFigure(entry,old,id,label){
 const m=metric(entry,id),before=metric(old,id);
 const d=numeric(m)&&numeric(before)?m.value-before.value:null;
 const changed=d!==null && Math.abs(d)>1e-10;
 const movement=d===null?'No comparison':!changed?'Unchanged':`${d>0?'+':'−'}${m.unit==='%'?(Math.abs(d)*100).toFixed(1)+' pp':fmt(Math.abs(d),m.unit)}`;
 const direction=changed?(lowerBetter(id,m)?d<0:d>0)?'Improved':'Worsened':'';
 return `<div class="history-figure"><span class="small">${esc(label)}</span><div class="history-value">${numeric(m)?value(m):'—'}<span class="history-movement ${changed?'changed':''}" title="${esc(direction || movement)}">${direction?direction==='Improved'?'↗ ':'↘ ':''}${movement}</span></div></div>`;
}
const when = r => r.recordedAt || r.date;
function ago(r){
 if(!r.recordedAt) return r.date;
 const s=(Date.now()-Date.parse(r.recordedAt))/1000;
 if(s<60) return 'just now';
 const [n,u]=s<3600?[s/60,'min']:s<86400?[s/3600,'hr']:s<86400*30?[s/86400,'day']:[s/86400/30,'mo'];
 const k=Math.floor(n);return `${k} ${u}${k===1||u==='min'?'':'s'} ago`;
}
function historyEntry(entry,old,showModel){
 const modules=impacted(entry),baseline=Array.isArray(entry.modules)&&entry.modules.length===0;
 const tags=(showModel?`<span class="module-tag neutral" title="Listener model">${esc(entry.listener)}</span>`:'')+(modules.length?modules.map(m=>`<span class="module-tag">${moduleLabels[m]}</span>`).join(''):`<span class="module-tag neutral">${baseline?'Baseline':'Module not tagged'}</span>`);
 const figures=[['e2e.precision','End-to-end precision'],['e2e.recall','End-to-end recall'],...modules.map(m=>impactMetric[m])];
 const note=entry.note || 'No change note recorded.';
 const headline=note.split(';')[0];
 return `<article class="history-item"><div class="history-meta"><div class="history-tags">${tags}</div><time datetime="${esc(when(entry))}" title="${esc(entry.recordedAt?new Date(entry.recordedAt).toLocaleString():entry.date)}">${esc(ago(entry))}</time><code>${esc(entry.sha.slice(0,7))}</code></div><div class="history-body"><h3 class="history-note">${esc(headline)}</h3><div class="history-figures">${figures.map(([id,label])=>historyFigure(entry,old,id,label)).join('')}</div><details class="change-details"><summary>Change details</summary><p>${esc(note)}</p><p class="small">Figures compare against the previous recorded run for this model, even when the list is filtered. Isolated checker recall uses perfect passage and sound evidence.</p></details></div></article>`;
}
let data=[];
function render(){
 const selected=document.getElementById('listener').value;
 const rows=data.filter(r=>r.listener===selected).sort((a,b)=>String(a.date).localeCompare(String(b.date)));
 const row=rows.at(-1),prev=rows.at(-2);if(!row)return;
 document.getElementById('run-meta').textContent=`Latest: ${row.date} · ${row.sha.slice(0,7)} · ${rows.length} run${rows.length===1?'':'s'}`;
 const tracked=metric(row,'e2e.leadin.precision')?.value!=null?'e2e.leadin':metric(row,'e2e.unambiguous.precision')?'e2e.unambiguous':'e2e',pid=`${tracked}.precision`,rid=`${tracked}.recall`;
 const p=metric(row,pid),r=metric(row,rid),warn=metric(row,'e2e.cleanPer5'),lead=tracked==='e2e.leadin',mut=metric(row,lead?'e2e.leadin.mutashabihat.recall':'e2e.mutashabihat.recall'),leadWarn=metric(row,'e2e.leadin.leadWarnings');
 const measured=numeric(p)&&numeric(r);
 const state=!measured?'Awaiting end-to-end evaluation':p.status==='pass'&&r.status==='pass'?'✓ Precision & recall meet targets':'Precision & recall need work';
 let html=`<div class="notice">${rows.length===1?'There’s only a baseline for this model. Record another benchmark run to see whether quality is improving.':`Comparing ${rows.length} runs for ${esc(selected)}. Changes below compare the latest run with the previous run.`}</div>`;
 html+=`<section class="hero" aria-labelledby="e2e-title"><div class="section-top"><div><div class="eyebrow">01 / THE WHOLE EXPERIENCE</div><h2 id="e2e-title">End-to-end quality</h2><p>Does Tilawa catch memorization mistakes, and can you trust its warnings?</p></div><span class="pill ${measured?(p.status==='pass'&&r.status==='pass'?'pass':'fail'):'pending'}">${state}</span></div><div class="hero-grid"><div><div class="score-pair">${stat(row,prev,pid,'Warnings you can trust','Precision · Of the warnings raised, how many were correct?')}${stat(row,prev,rid,'Mistakes caught','Recall · Of the actual mistakes, how many were caught?')}</div><div class="submetric"><span>False warnings / 5 min <span class="small">(short clips)</span></span><strong>${value(warn)}</strong></div><div class="target">${target(warn)} · Long-session evaluation still needed.</div>${mut?`<div class="submetric"><span>Mutashabihat slips caught <span class="small">(${lead?'free, after 1–2 recited ayahs':'free; needs declared passage or position hint'})</span></span><strong>${value(mut)}</strong></div>`:''}${leadWarn?`<div class="submetric"><span>False warnings in the lead-in</span><strong>${value(leadWarn)}</strong></div>`:''}</div><div><div class="chart-legend"><span><i class="legend-line"></i>Precision</span><span><i class="legend-line secondary"></i>Recall</span></div>${chart(rows,[{id:pid,label:'Precision',color:primary},{id:rid,label:'Recall',color:secondary,dashed:true}])}</div></div><p class="small">These are memorization-error metrics, not an overall pronunciation score. No single “accuracy” number is recorded.</p>${!measured?`<p class="small">${esc(p?.note || r?.note || 'The end-to-end evaluation has not been recorded.')}</p>`:''}</section>`;
 const defs=[{name:'Listener',step:'HEAR THE SOUNDS',id:'listener.goldPer',label:'Phoneme error rate',description:'Turns recitation audio into sounds the engine can use.',explain:'Gold PER · On human-verified exact recitations, the share of sound edits needed to match the reference. Lower is better; this is not word accuracy.',secondary:'listener.visible',secondaryLabel:'Planted mistakes visible in decode'}, {name:'Tracker',step:'FIND THE PASSAGE',id:'tracker.clean.within5s',label:'Correct ayah within 5 seconds',description:'Finds where the reciter is in the Quran.',explain:'Measured on clean recitation clips. Higher is better.',secondary:'tracker.all.within5s',secondaryLabel:'Including clips with mistakes'}, {name:'Checker',step:'RECOGNIZE THE MISTAKE',id:'checker.oracleAll.recall',label:'Mistakes caught with perfect inputs',description:'Decides when a difference deserves a warning.',explain:'Recall with the correct passage and label-synthesized sound evidence. Isolates the checker; higher is better.',secondary:'checker.oracleLock.recall',secondaryLabel:'Correct passage + real sound evidence'}];
 html+=`<div class="section-heading"><div><h2>Inside the engine</h2><p>Three different jobs. Three different measures of quality.</p></div><span class="small">Audio → Listener → Tracker → Checker</span></div><section class="components">`;
 defs.forEach((d,i)=>{let m=metric(row,d.id);html+=`<article class="component"><div class="step">0${i+1} / ${d.step}</div><h3>${d.name}</h3><p class="description">${d.description}</p>${badge(m)}<div class="metric-label">${d.label}</div><div class="number ${numeric(m)?'':'unavailable'}">${value(m)}</div><p class="small">${d.explain}</p><div class="target">${target(m)}</div>${delta(m,metric(prev,d.id),d.id)}${chart(rows,[{id:d.id,label:d.label,color:primary}],{compact:true})}<div class="submetric"><span>${d.secondaryLabel}</span><strong>${value(metric(row,d.secondary))}</strong></div>${!numeric(m)?`<p class="small">${esc(m?.note || 'This isolated evaluation has not been recorded for this model.')}</p>`:''}</article>`;});
 html+='</section><section class="analysis"><article class="panel"><h3>Which mistakes get caught?</h3><p>Latest end-to-end results, by mistake type.</p><div class="error-row head"><span>Mistake</span><span>Precision</span><span>Recall</span></div>';
 [['wrong_word','Wrong word'],['omitted_word','Omitted word'],['added_word','Added word'],['skipped_passage','Skipped passage']].forEach(([id,label])=>{html+=`<div class="error-row"><span>${label}</span>${['precision','recall'].map(s=>{const m=metric(row,`e2e.${id}.${s}`);return `<span title="${esc(m?.display || m?.note || 'No measurement')}">${numeric(m)?value(m):'—'}</span>`;}).join('')}</div>`;});
 html+='<p class="small">— means no measurement or no confirmed examples; it does not mean zero accuracy. Hover values for sample counts.</p></article><article class="panel"><h3>Where are mistakes being missed?</h3><p>Attribution of missed memorization errors with real upstream evidence.</p>';
 const losses=[['checker.e0.evidence','Listener evidence'],['checker.e0.tracker','Tracker'],['checker.e0.checker','Checker']].map(([id,label])=>({m:metric(row,id),label}));
 const total=losses.reduce((n,l)=>n+(numeric(l.m)?l.m.value:0),0);
 if(losses.every(l=>numeric(l.m))&&total>0){losses.forEach(l=>{html+=`<div class="loss"><div class="loss-label"><span>${l.label}</span><strong>${l.m.value} of ${total} misses · ${Math.round(l.m.value/total*100)}%</strong></div><div class="track"><div class="fill" style="width:${l.m.value/total*100}%"></div></div></div>`;});html+='<p class="small">Diagnostic attribution, not independent component accuracy.</p>';}else html+='<div class="empty">Miss attribution has not been measured for this model.</div>';
 html+='</article></section><div class="section-heading"><div><h2>What changed?</h2><p>Latest runs across every model. Filter by module or model.</p></div></div><section class="panel history-panel"><div class="history-toolbar"><div class="history-filters" role="group" aria-label="Filter changes by module">'+['all',...MODULES].map(m=>`<button type="button" data-module="${m}" aria-pressed="${m==='all'}">${m==='all'?'All changes':moduleLabels[m]}</button>`).join('')+`</div><label class="small history-model">Model <select id="history-model"><option value="">All models</option>${[...new Set(data.map(r=>r.listener))].map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join('')}</select></label><span class="small" id="history-count"></span></div><div id="history-list"></div>`;
 html+='</section><details class="details"><summary>Explore all measurements <span class="small">· targets, sample counts & reports</span></summary><p>Meets / outside target applies only to that metric. Pending means an evaluation hasn’t run; no measurement means the data or capability is missing.</p><div class="detail-tools"><label>Find a metric <input id="search" type="search" placeholder="e.g. pronunciation, false jumps"></label><label>Area <select id="area"><option value="">All areas</option><option>End to end</option><option>Listener</option><option>Tracker</option><option>Checker</option></select></label></div><div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Latest result</th><th>Target / status</th><th>Change vs previous run</th></tr></thead><tbody id="metrics"></tbody></table></div></details>';
 document.getElementById('content').innerHTML=html;
 const timeline=data.map((r,i)=>({r,i})).sort((a,b)=>String(when(a.r)).localeCompare(String(when(b.r)))||a.i-b.i).map(x=>x.r);
 const last={},all=timeline.map(entry=>{const old=last[entry.listener];last[entry.listener]=entry;return {entry,old};});
 let filter='all';
 function history(){
  const model=document.getElementById('history-model').value;
  const pool=all.filter(({entry})=>!model||entry.listener===model);
  const entries=pool.filter(({entry})=>filter==='all'||impacted(entry).includes(filter)).reverse();
  document.getElementById('history-count').textContent=`${entries.length} of ${pool.length} runs`;
  document.getElementById('history-list').innerHTML=entries.map(({entry,old})=>historyEntry(entry,old,!model)).join('') || `<div class="empty">No recorded ${esc(moduleLabels[filter]?.toLowerCase() || '')} changes${model?' for this model':''}.</div>`;
  document.querySelectorAll('[data-module]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.module===filter)));
 }
 history();document.querySelectorAll('[data-module]').forEach(button=>button.addEventListener('click',()=>{filter=button.dataset.module;history();}));document.getElementById('history-model').addEventListener('change',history);
 function details(){const q=document.getElementById('search').value.toLowerCase(),area=document.getElementById('area').value;const matches=row.metrics.filter(m=>(!area||m.group===area)&&`${m.name} ${m.group} ${m.note || ''}`.toLowerCase().includes(q));document.getElementById('metrics').innerHTML=matches.map(m=>`<tr><td><span class="small">${esc(m.group)}</span><br>${esc(m.name)}<small>${esc(m.note)} ${source(m)}</small></td><td>${esc(m.display || (numeric(m)?value(m):m.status==='pending'?'Awaiting evaluation':'No measurement'))}</td><td>${badge(m)}<div class="target">${target(m)}</div></td><td>${delta(m,metric(prev,m.id),m.id)}</td></tr>`).join('')||'<tr><td colspan="4">No matching measurements.</td></tr>';}
 details();document.getElementById('search').addEventListener('input',details);document.getElementById('area').addEventListener('change',details);
}
async function init(){try{const response=await fetch('scoreboard.json',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);const json=await response.json();if(!Array.isArray(json.rows)||!json.rows.length)throw new Error('No benchmark runs recorded');data=json.rows;const models=[...new Set(data.map(r=>r.listener))];document.getElementById('listener').innerHTML=models.map(s=>`<option value="${esc(s)}">${esc(s)}${s==='teacher'?' · reference model':''}</option>`).join('');document.getElementById('listener').addEventListener('change',render);render();}catch(error){document.getElementById('content').innerHTML=`<div class="notice"><strong>Couldn’t load benchmark results.</strong><br>${esc(error.message)}. Serve this page using <code>uv run tools/scoreboard_serve.py</code> and reload.</div>`;}}
init();
