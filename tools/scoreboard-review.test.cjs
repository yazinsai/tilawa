// Run: node --test tools/scoreboard-review.test.cjs
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const context=vm.createContext({URL,Date});
for(const f of ['model','store']) vm.runInContext(fs.readFileSync(path.join(root,`docs/benchmarks/scoreboard-review-${f}.js`),'utf8'),context);
const M=context.tilawaReviewModel,S=context.tilawaReviewStore;
const clone=x=>JSON.parse(JSON.stringify(x));
const run={file:'example.json',sha:'example',listener:'test',recordedAt:'2026-10-10T00:00:00Z'};
const label={errorClass:'omitted_word',state:'confirmed',location:{surah:94,ayah:5,word:2},expected:'مع'};
const miss={errorClass:'omitted_word',state:'confirmed',at:'94:5:2',audio:'help/example.wav',seekS:1.6,cause:'checker'};
const report={errorClass:'wrong_word',state:'confirmed',at:'94:5:2',audio:'help/example.wav',seekS:1.6,observed:'هءَ'};
const gate={correctWithin5s:true,onsetMs:1200,firstCorrectLockMs:5000};
function fixture(){return {version:2,...run,clips:[{id:'example',audio:'help/example.wav',ayahs:['94:5'],words:[{at:'94:5:2',text:'مَعَ',ref:'مَعَ',said:'هءَ',fromS:2.6,toS:3.28}],labels:[clone(label)],kind:'skip_word',arms:{free:{hits:[],misses:[clone(miss)],falsePositives:[clone(report)],gate:{...gate},engineEvents:[clone(report)]}}}]};}
const memory=()=>{const map=new Map();return {getItem:k=>map.get(k)||null,setItem:(k,v)=>map.set(k,v),map};};
test('same-word different-class scoring records become one incident without changing evidence',()=>{
 const d=fixture(), before=JSON.stringify(d), all=M.incidents(d,'free',run);
 assert.equal(all.length,1);assert.equal(all[0].mismatch,true);assert.equal(all[0].raw.length,2);assert.equal(all[0].reports.length,1);assert.equal(all[0].labels.length,1);assert.equal(JSON.stringify(d),before);
});
test('neighboring words and passage timing remain separate, duplicate outcomes stay traceable',()=>{
 const d=fixture(),a=d.clips[0].arms.free;a.falsePositives.push({...report,at:'94:5:3'},clone(report));a.gate.correctWithin5s=false;
 const all=M.incidents(d,'free',run);assert.equal(all.length,3);assert.equal(all.find(i=>i.location==='94:5:2').raw.length,3);
 assert.equal(new Set(all.flatMap(i=>i.raw.map(r=>r.id))).size,5);
});
test('a benchmark label is not represented as an engine report',()=>{
 const d=fixture();d.version=1;delete d.clips[0].arms.free.engineEvents;d.clips[0].arms.free.falsePositives=[];
 const i=M.incidents(d,'free',run)[0];assert.equal(i.reports.length,0);assert.equal(i.labels.length,1);assert.equal(M.snapshot(i,'http://localhost/').limitations[0].includes('Version 1'),true);
});
test('suspected labels and non-confirmed engine states are reviewable without becoming scored positives',()=>{
 const d=fixture();d.clips[0].labels[0].state='suspected';d.clips[0].arms.free.misses=[];d.clips[0].arms.free.falsePositives=[];d.clips[0].arms.free.engineEvents[0].state='uncertain';
 assert.equal(M.summary(d,'free').positives,0);assert.equal(M.incidents(d,'free',run)[0].attention,false);assert.equal(M.incidents(d,'free',run)[0].reports[0].state,'uncertain');
});
test('sound or confidence changes are comparison differences even with unchanged scoring signatures',()=>{
 const a=fixture(),b=fixture();b.clips[0].arms.free.engineEvents[0].confidence=.8;
 const comparison=M.compare(M.incidents(b,'free',run),M.incidents(a,'free',run));assert.equal(comparison[0].change,'changed');assert.equal(comparison[0].comparison.reports.length,1);
});
test('removed evidence retains earlier run identity',()=>{
 const a=fixture(),b=fixture();b.clips=[];const all=M.compare(M.incidents(b,'free',run),M.incidents(a,'free',run));assert.equal(all[0].earlierOnly,true);assert.equal(all[0].change,'removed');assert.equal(all[0].run.file,run.file);
});
test('new dump timestamp has a separate review identity',()=>{
 const a=fixture(),b=fixture();b.recordedAt='2026-10-11';assert.notEqual(M.reviewKey(M.incidents(a,'free',run)[0]),M.reviewKey(M.incidents(b,'free',run)[0]));
});
test('export snapshot preserves all labels, checker states, raw outcome indices, other setups and audio timing',()=>{
 const i=M.incidents(fixture(),'free',run)[0],r=M.snapshot(i,'http://127.0.0.1:8765/');
 assert.equal(r.originalScoringOutcomes.length,2);assert.equal(r.selectedArm.engineEvents.length,1);assert.equal(r.otherSetupEvidence.free.misses.length,1);assert.equal(r.cue.alignedWordSpan.fromSeconds,2.6);assert.equal(r.audioReferences[0].url,'http://127.0.0.1:8765/audio/help/example.wav');
});
test('legacy findings for both outcomes are preserved in a draft, never reconciled automatically',()=>{
 const mem=memory(),i=M.incidents(fixture(),'free',run)[0],old={};
 i.raw.forEach((r,n)=>old[n]={id:String(n),run:{...run,recordedAt:'older'},setup:{id:'free'},result:{id:r.key},review:{verdict:n?'engine_correct':'missed_mistake',comment:'old note'}});
 const original=JSON.stringify(old);mem.setItem(S.LEGACY_KEY,original);const store=S.create(mem),r=store.migrate(i,M,'http://localhost/');assert.equal(r.legacyFindings.length,2);assert.equal(M.complete(r),false);assert.equal(r.migration.evidenceSnapshotChanged,true);assert.equal(mem.getItem(S.LEGACY_KEY),original);assert.equal(S.create(mem).records[r.id].legacyFindings.length,2);
});
test('review progress requires explicit reviewed status, not a comment',()=>{assert.equal(M.complete({finding:{status:'draft',comment:'Notes'}}),false);assert.equal(M.complete({finding:{status:'reviewed',areas:['unresolved']}}),true);});
test('corrupt stored JSON is preserved and protected from overwrites',()=>{
 const mem=memory();mem.setItem(S.KEY,'{broken');const store=S.create(mem);assert.equal(store.writable,false);assert.equal(store.save(),false);assert.equal(store.recovery[0].raw,'{broken');assert.equal(mem.getItem(S.KEY),'{broken');
});
test('storage failures leave the in-memory review available',()=>{
 const store=S.create({getItem:()=>null,setItem:()=>{throw Error('quota');}});store.records.example={id:'example',finding:{status:'draft'}};assert.equal(store.save(),false);assert.equal(store.records.example.id,'example');assert.match(store.error,/backup/i);
});
test('backup import deduplicates identical records and retains divergent versions',()=>{
 const mem=memory(),store=S.create(mem),r={id:'r',run,setup:'free',finding:{status:'draft',comment:'A',areas:[],categories:[]}};
 assert.equal(store.ingest({schemaVersion:2,reviews:[r]}).added,1);assert.equal(store.ingest({schemaVersion:2,reviews:[r]}).duplicates,1);
 assert.equal(store.ingest({schemaVersion:2,reviews:[{...r,finding:{...r.finding,comment:'B'}}]}).conflicts,1);assert.equal(store.records.r.finding.comment,'A');assert.equal(store.conflicts[0].incoming.finding.comment,'B');
});
test('invalid import is atomic',()=>{
 const store=S.create(memory());assert.throws(()=>store.ingest({schemaVersion:2,reviews:[{id:'r',run,finding:{}},{run}]}));assert.equal(Object.keys(store.records).length,0);
});
// Optional real-record integration: private dumps stay private, while this check remains reproducible.
const actual=path.join(root,'.scratch/scoreboard/clips/a4cc202-spliced-clean-int8.json');
test('actual benchmark: exact mismatch, context warnings, misses, passage timing and metrics agree', {skip:!fs.existsSync(actual)},()=>{
 const d=JSON.parse(fs.readFileSync(actual,'utf8')),s=M.summary(d,'leadFree'),all=M.incidents(d,'leadFree',{file:path.basename(actual)});
 const i=all.filter(i=>i.clip.id==='muo5gkq2_857a61f4'&&i.location==='94:5:2');assert.equal(i.length,1);assert.equal(i[0].mismatch,true);assert.equal(i[0].raw.length,2);
 assert.equal(s.raw,s.miss+s.fp+s.leadin+s.leadout+s.late);assert.equal(s.positives,s.hits+s.miss);
 assert.equal(all.filter(i=>i.attention).reduce((n,i)=>n+i.raw.length,0),s.raw);
 assert.ok(all.some(i=>i.clip.kind==='none'&&i.raw.some(r=>r.type==='fp')));assert.ok(all.some(i=>i.zone==='before'));assert.ok(all.some(i=>i.zone==='after'));assert.ok(all.some(i=>i.zone==='passage'));
 const board=JSON.parse(fs.readFileSync(path.join(root,'docs/benchmarks/scoreboard.json'),'utf8')),row=board.rows.filter(r=>r.listener===d.listener&&r.sha.startsWith(d.sha)).at(-1);
 const free=M.summary(d,'free');for(const k of ['evidence','tracker','checker']) assert.equal(free.causes[k],row.metrics.find(m=>m.id===`checker.e0.${k}`).value);
 assert.equal(s.hits/s.positives,row.metrics.find(m=>m.id==='e2e.leadin.recall').value);
});

// Exercise the presentation with the existing incident model. No review schema changes.
function presentation(){
 const context=vm.createContext({URL,Date,window:{localStorage:memory()},document:{addEventListener(){}}});
 for(const f of ['model','store']) vm.runInContext(fs.readFileSync(path.join(root,`docs/benchmarks/scoreboard-review-${f}.js`),'utf8'),context);
 const source=fs.readFileSync(path.join(root,'docs/benchmarks/scoreboard-review.js'),'utf8').replace('window.tilawaReview={panel,init,focusGate,hasPreset:id=>id in PRESETS};','window.tilawaReview={comparison,comparisonPanel,affectedPassage,reviewForm,st};');
 vm.runInContext(source,context);return context.window.tilawaReview;
}
test('comparison explains a type mismatch without merging its original score outcomes',()=>{
 const ui=presentation(),i=M.incidents(fixture(),'free',run)[0],before=JSON.stringify(i),c=ui.comparison(i);
 assert.equal(c.expected,'Missing word');assert.equal(c.actual,'Wrong word');assert.match(c.note,/missed label and an unmatched warning/);assert.equal(JSON.stringify(i),before);
});
test('comparison distinguishes an incomplete older report from a known absent report',()=>{
 const ui=presentation(),d=fixture();delete d.clips[0].arms.free.engineEvents;d.clips[0].arms.free.falsePositives=[];
 assert.equal(ui.comparison(M.incidents(d,'free',run)[0]).actual,'No final report saved');
 d.clips[0].arms.free.engineEvents=[];assert.equal(ui.comparison(M.incidents(d,'free',run)[0]).actual,'No final report');
});
test('suspected label stays tentative in the expected output',()=>{
 const ui=presentation(),d=fixture();d.clips[0].labels[0].state='suspected';
 assert.equal(ui.comparison(M.incidents(d,'free',run)[0]).expected,'Possible Missing word');
});
test('passage comparison uses elapsed time since onset, including a passage never found',()=>{
 const ui=presentation(),d=fixture();d.clips[0].arms.free.gate={...gate,correctWithin5s:false,firstCorrectLockMs:9000};
 assert.equal(ui.comparison(M.incidents(d,'free',run).find(i=>i.zone==='passage')).actual,'Passage found after 7.80 seconds');
 d.clips[0].arms.free.gate.firstCorrectLockMs=null;
 assert.equal(ui.comparison(M.incidents(d,'free',run).find(i=>i.zone==='passage')).actual,'Passage not found');
});
test('context warning never shows original passage text or borrows another recording’s word times',()=>{
 const ui=presentation(),d=fixture(),i={...M.incidents(d,'free',run)[0],location:'2:184:6',audio:'leadin/context.wav',zone:'after'};
 assert.match(ui.affectedPassage(i),/text for this ayah is not available/);assert.doesNotMatch(ui.affectedPassage(i),/مَعَ/);
 ui.st.dumps.other={clips:[{words:[{at:'2:184:6',text:'مَّرِيضًا',fromS:99}]}]};
 const html=ui.affectedPassage(i);assert.match(html,/مَّرِيضًا/);assert.match(html,/data-path="leadin\/context.wav"/);assert.doesNotMatch(html,/data-at="98/);
});
test('reflow retains every original finding and model-error option',()=>{
 const ui=presentation(),html=ui.reviewForm(M.incidents(fixture(),'free',run)[0]);
 for(const area of ['model','label','audio_data','acceptable','unresolved'])assert.match(html,new RegExp(`name="area" value="${area}"`));
 for(const category of ['missed_detection','wrong_mistake_type','wrong_word_location','passage_recognition','sound_recognition','spurious_warning','other'])assert.match(html,new RegExp(`name="category" value="${category}"`));
 for(const id of ['rv-certainty','rv-comment','rv-complete','rv-next-unreviewed'])assert.match(html,new RegExp(`id="${id}"`));
});
