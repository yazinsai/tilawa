'use strict';
// Human review is a separate layer. Nothing here changes acceptance matching or benchmark labels.
(() => {
const CLASSES = ['wrong_word','omitted_word','added_word','skipped_passage'];
const atOf = l => `${l.location.surah}:${l.location.ayah}:${l.location.word}`;
const signature = e => `${e.errorClass}/${e.at || atOf(e)}`;
function outcomes(clip, arm, setup){
 const out = [];
 for(const [type,field] of [['miss','misses'],['fp','falsePositives'],['leadin','leadInWarnings'],['leadout','leadOutWarnings']])
  (arm[field] || []).forEach((event,index) => out.push({type,event,sourceField:field,sourceIndex:index,key:`${type}|${clip.id}|${signature(event)}`,id:JSON.stringify([clip.id,field,index])}));
 if(['free','leadFree'].includes(setup) && !arm.gate.correctWithin5s) out.push({type:'late',event:null,sourceField:'gate',key:`late|${clip.id}|`,id:JSON.stringify([clip.id,'gate'])});
 return out;
}
function incidents(dump, setup, run){
 const result = [];
 for(const clip of dump.clips){
  const arm = clip.arms[setup]; if(!arm) continue;
  const groups = new Map(), main = arm.segments?.find(s => s.source === clip.id);
  const region = e => {
   if(e.audio === clip.audio) return 'clip';
   const segment = arm.segments?.find(s => s.audio === e.audio);
   return segment && main ? segment.endMs <= main.startMs ? 'before' : 'after' : 'unknown';
  };
  const put = (zone,location,audio) => {
   const key = JSON.stringify([clip.id,zone,location,audio]);
   if(!groups.has(key)) groups.set(key,{key,comparisonKey:JSON.stringify([clip.id,zone,location,audio]),zone,location,audio,clip,arm,run:{...run,sha:dump.sha,listener:dump.listener,recordedAt:dump.recordedAt},setup,raw:[],labels:[],hits:[],reports:[],dumpVersion:dump.version});
   return groups.get(key);
  };
  for(const raw of outcomes(clip,arm,setup)){
   const zone = raw.type === 'late' ? 'passage' : raw.type === 'leadin' ? 'before' : raw.type === 'leadout' ? 'after' : 'clip';
   put(zone,raw.event?.at || null,raw.event?.audio || clip.audio).raw.push(raw);
  }
  for(const label of clip.labels) put('clip',atOf(label),clip.audio).labels.push(label);
  for(const hit of arm.hits) put('clip',hit.split('/')[1],clip.audio).hits.push(hit);
  for(const report of arm.engineEvents || []) put(region(report),report.at,report.audio).reports.push(report);
  for(const group of groups.values()){
   group.attention = group.raw.length > 0;
   group.mismatch = group.raw.some(r => r.type === 'miss') && group.raw.some(r => r.type === 'fp');
   // Confidence, sound output and final state matter even when the scoring category stays the same.
   group.fingerprint = JSON.stringify({raw:group.raw,hits:group.hits,labels:group.labels,reports:group.reports,gate:group.zone === 'passage' ? arm.gate : undefined,words:clip.words.filter(w=>w.at===group.location)});
   result.push(group);
  }
  if(!groups.size) result.push({...put('clip',null,clip.audio),attention:false,mismatch:false,fingerprint:'clear'});
 }
 return result;
}
function compare(current, earlier){
 const left = new Map(earlier.map(i => [i.comparisonKey,i])), right = new Map(current.map(i => [i.comparisonKey,i]));
 return [...current.map(i => {
  const prior = left.get(i.comparisonKey);
  return {...i,comparison:prior || null,change:!prior ? 'new' : prior.fingerprint === i.fingerprint ? 'same' : 'changed'};
 }),...earlier.filter(i => !right.has(i.comparisonKey)).map(i => ({...i,comparison:null,change:'removed',earlierOnly:true}))];
}
const reviewKey = i => JSON.stringify([i.run.file,i.run.recordedAt,i.setup,i.key]);
const complete = r => r?.finding?.status === 'reviewed';
function legacyFor(i, legacy){
 return Object.values(legacy).filter(r => r && r.run?.file === i.run.file && (r.setup?.id || r.setup) === i.setup && i.raw.some(raw => raw.key === r.result?.id));
}
function summary(dump, setup){
 const clips = dump.clips.filter(c => c.arms[setup]);
 const raw = clips.flatMap(c => outcomes(c,c.arms[setup],setup)), count = type => raw.filter(r => r.type === type).length;
 const positives = clips.reduce((n,c) => n+c.labels.filter(l => l.state === 'confirmed' && CLASSES.includes(l.errorClass)).length,0);
 const hits = clips.reduce((n,c) => n+c.arms[setup].hits.length,0);
 const causes = Object.fromEntries(['evidence','tracker','checker'].map(k => [k,raw.filter(r => r.type === 'miss' && r.event.cause === k).length]));
 const context = {before:{clips:0,seconds:0},after:{clips:0,seconds:0}};
 for(const c of clips){
  const parts = c.arms[setup].segments || [], main = parts.find(s => s.source === c.id);
  for(const zone of ['before','after']){
   const segs = main ? parts.filter(s => zone === 'before' ? s.endMs <= main.startMs : s.startMs >= main.endMs) : [];
   if(segs.length) context[zone].clips++;
   context[zone].seconds += segs.reduce((n,s) => n+(s.endMs-s.startMs)/1000,0);
  }
 }
 return {clips:clips.length,cleanClips:clips.filter(c => c.kind === 'none').length,positives,hits,raw:raw.length,miss:count('miss'),fp:count('fp'),leadin:count('leadin'),leadout:count('leadout'),late:count('late'),cleanLate:clips.filter(c=>c.kind==='none' && !c.arms[setup].gate.correctWithin5s).length,causes,context};
}
function cue(i){
 const event = i.raw.find(r=>r.event)?.event || i.reports[0], word = i.clip.words.find(w=>w.at===i.location);
 return {audio:i.audio,seekSeconds:event?.seekS ?? Math.max(0,(word?.fromS ?? 1)-1),alignedWordSpan:i.audio===i.clip.audio && word?.fromS!=null ? {fromSeconds:word.fromS,toSeconds:word.toS} : null,
  meaning:'Approximate pre-roll in source audio from decode alignment or report emission. Neither an annotated onset nor proof of what was recited.'};
}
function snapshot(i, baseUrl){
 const paths = new Set([i.clip.audio,i.audio,...(i.arm.segments || []).map(s=>s.audio)]);
 const recording = {...i.clip}; delete recording.arms;
 return {id:reviewKey(i),incidentKey:i.key,comparisonKey:i.comparisonKey,run:i.run,setup:i.setup,dumpVersion:i.dumpVersion,
  grouping:{rule:'Same recording, exact word location, source audio and session region. Passage timing stays separate. Adjacent locations are not automatically equivalent.',region:i.zone,location:i.location,originalOutcomeIds:i.raw.map(r=>r.id),legacyOutcomeKeys:i.raw.map(r=>r.key)},
  recording,selectedArm:i.arm,otherSetupEvidence:i.clip.arms,originalScoringOutcomes:i.raw,benchmarkClaims:i.labels,matchedSignatures:i.hits,engineReports:i.reports,
  audioReferences:[...paths].map(path=>({path,url:new URL(`audio/${path}`,baseUrl).href,availability:'Local server required; audio bytes are not embedded.'})),cue:cue(i),
  limitations:i.arm.engineEvents ? ['Final checker states only; no full event transition or tracker candidate trace. Word sounds are aligned greedy decode of the original clip; oracle arm uses synthetic evidence.'] : ['Version 1 dump: matched reports are signatures only; non-confirmed checker states and original emission times are unavailable.'],
  comparison:i.comparison ? {change:i.change,run:i.comparison.run,recording:i.comparison.clip,selectedArm:i.comparison.arm,originalScoringOutcomes:i.comparison.raw,engineReports:i.comparison.reports} : i.change ? {change:i.change,earlierOnly:!!i.earlierOnly} : null};
}
globalThis.tilawaReviewModel = Object.freeze({CLASSES,atOf,signature,outcomes,incidents,compare,reviewKey,complete,legacyFor,summary,cue,snapshot});
})();
