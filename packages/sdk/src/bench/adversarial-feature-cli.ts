/** Prepare training-speaker acoustic contrasts or inspect engine-requested comparisons.
 * No label supplies a performed mistake transcript or changes an inference window. */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, SequenceEvidenceCandidate, WordRef } from "@tilawa/core";
import { addAdversarialEvidence, featureObservations, type AdversarialModel } from "../adversarial-feature.js";
import { loadNodeModel } from "../node.js";
import { ortBackend } from "../ort.js";
import type { OrtModule } from "../ort.js";
import { createListener } from "../listener.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{model:{type:"string"},extractor:{type:"string"},labels:{type:"string"},
  "excluded-speakers":{type:"string"},out:{type:"string"},audio:{type:"string"},evidence:{type:"string"}}});
if(!values.model||!values.out)throw new Error("--model and --out are required");
const json=(path:string)=>JSON.parse(readFileSync(path,"utf8"));
const asset=(name:string)=>json(new URL(`../../../core/assets/${name}`,import.meta.url).pathname);
const index=loadReferenceIndex(asset("ref-v2.json")),phrases=asset("phrases-v1.json");
const audio=(path:string)=> {
  const wav=decodeWav(readFileSync(path));
  if(wav.sampleRate!==16000)throw new Error("Frozen scorer requires 16 kHz WAV audio");
  return wav.samples;
};
/** Forced alignment supplies boundaries only on none-labelled full canonical recordings. */
function alignment(lp:number[][],ids:number[]):number[]|null {
  const labels=[0,...ids.flatMap(p=>[p,0])],n=labels.length;
  let a=labels.map(()=>-Infinity);a[0]=lp[0][0];a[1]=lp[0][ids[0]];
  const back:number[][]=[labels.map(()=>-1)];
  for(let t=1;t<lp.length;t++) {
    const b=labels.map(()=>-Infinity),parent=labels.map(()=>-1);
    for(let s=0;s<n;s++) {
      const previous=[s,...(s>0?[s-1]:[]),...(s>1&&labels[s]!==0&&labels[s]!==labels[s-2]?[s-2]:[])];
      const best=previous.reduce((x,y)=>a[x]>=a[y]?x:y);
      b[s]=a[best]+lp[t][labels[s]];parent[s]=best;
    }
    a=b;back.push(parent);
  }
  let last=a[n-1]>=a[n-2]?n-1:n-2;
  if(!Number.isFinite(a[last]))return null;
  const states=new Array(lp.length);states[lp.length-1]=last;
  for(let t=lp.length-1;t>0;t--) {last=back[t][last];states[t-1]=last;}
  return states;
}
const key=(w:WordRef)=>`${w.surah}:${w.ayah}:${w.word}`;
const primary=await loadNodeModel(resolve(values.model));
try {
  if(values.labels) {
    if(!values.extractor||!values["excluded-speakers"])throw new Error("--extractor and --excluded-speakers are required");
    const excluded=new Set<string>(json(values["excluded-speakers"]));
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for(const r of labels)if(r.split!=="train")excluded.add(r.speaker);
    const metadata=json(resolve(values.extractor,"adversarial-feature-metadata.json")) as AdversarialModel;
    if(metadata.primarySha256!==primary.manifest.sha256)throw new Error("Extractor differs from primary graph");
    const extractor=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"adversarial-feature.onnx"),"onnxruntime-node",{
      executionProviders:["cpu"],intraOpNumThreads:1,graphOptimizationLevel:"disabled"});
    try {
      const rows:any[]=[],skipped:string[]=[];
      for(const r of labels) {
        if(r.split!=="train"||excluded.has(r.speaker))continue;
        const substitution=(r.events??[]).some((e:any)=>e.errorClass==="wrong_word");
        if(r.kind!=="none"&&!substitution)continue;
        if(r.kind==="none"&&((r.events??[]).length||(r.non_errors??[]).length))throw new Error("None-labelled anchor contains scripted deviations");
        if(!r.speaker||!r.wav||!r.ayahs?.length)throw new Error("Training label lacks speaker, WAV or ayahs");
        const words=(r.ayahs as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
        if(!words.length)throw new Error("No canonical training pronunciation");
        const samples=audio(resolve(dirname(values.labels),r.wav));
        const features=await featureObservations(extractor,primary.manifest,metadata,samples);
        const provenance={source:"help-train",split:"train",excluded:false,speaker:r.speaker,id:r.id};
        if(r.kind==="none") {
          const pronunciation=words.map(w=>index.phonemes.slice(w.start,w.end)).join("");
          const ids=[...pronunciation].map(p=>metadata.tokens.indexOf(p)+1);
          if(ids.some(p=>p<1))throw new Error("Unknown canonical phone");
          const states=features.logProbs.length?alignment(features.logProbs,ids):null;
          if(!states){skipped.push(r.id);continue;}
          let offset=0;
          const examples=[];
          for(const w of words) {
            const count=w.end-w.start,from=states.findIndex(s=>s>=2*offset+1),
              next=states.findIndex(s=>s>2*(offset+count)),to=next<0?states.length:next;
            const correct=ids.slice(offset,offset+count);offset+=count;
            if(from<0||to<=from||to-from<correct.length)continue;
            examples.push({features:features.observations.slice(from,to),correct,pronunciation:index.phonemes.slice(w.start,w.end)});
          }
          rows.push({...provenance,kind:"none",features:features.observations,
            frameLabels:states.map(s=>s%2?ids[(s-1)/2]:0),examples});
        } else {
          // Only actual engine requests overlapping a weakly labelled substitution
          // become preservation controls. No script word is forced onto its frames.
          const listener=createListener(primary,{posteriors:true});
          const chunks:PhonemeEvidence[]=[];
          for(let s=0;s<samples.length;s+=5120){const e=await listener.push(samples.slice(s,s+5120),16000);if(e)chunks.push(e);}
          const tail=await listener.end();if(tail)chunks.push(tail);
          const wanted=new Set((r.events??[]).filter((e:any)=>e.errorClass==="wrong_word").map((e:any)=>key(e.location)));
          const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000});
          const requests=new Map<string,SequenceEvidenceCandidate>();
          const checker=createChecker(index,{phrases,sequenceEvidence(c){if(wanted.has(key(c.word)))requests.set(JSON.stringify(c),c);return true;}});
          for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
          const controls=[...requests.values()].filter(c=>c.fromFrame>=0&&c.toFrame<=features.observations.length).map(c=>({features:features.observations.slice(c.fromFrame,c.toFrame),
            written:[...c.written].map(p=>metadata.tokens.indexOf(p)+1),rival:[...c.rival].map(p=>metadata.tokens.indexOf(p)+1)}));
          rows.push({...provenance,kind:"substituted_word",features:features.observations,controls});
        }
      }
      // Both lexical identities have recorded positive examples; nearest attested
      // pronunciation rivals are deterministic and independent of practice targets.
      const vocabulary=[...new Map(rows.filter(r=>r.kind==="none").flatMap(r=>r.examples)
        .map((e:any)=>[e.pronunciation,e.correct] as [string,number[]])).entries()];
      const edit=(a:number[],b:number[])=> {
        let row=b.map((_,i)=>i+1);row.unshift(0);
        for(let i=0;i<a.length;i++){const next=[i+1];for(let j=0;j<b.length;j++)next.push(Math.min(next[j]+1,row[j+1]+1,row[j]+Number(a[i]!==b[j])));row=next;}
        return row[b.length];
      };
      for(const r of rows.filter(r=>r.kind==="none"))r.examples=r.examples.flatMap((e:any)=>
        vocabulary.filter(([p])=>p!==e.pronunciation).sort((a,b)=>edit(e.correct,a[1])-edit(e.correct,b[1])||a[0].localeCompare(b[0]))
          .filter(([,rival])=>rival.length+rival.slice(1).filter((p,i)=>p===rival[i]).length<=e.features.length)
          .slice(0,2).map(([,rival])=>({...e,rival})));
      writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
      console.log(JSON.stringify({recordings:rows.length,none:rows.filter(r=>r.kind==="none").length,
        speakers:new Set(rows.map(r=>r.speaker)).size,excludedSpeakers:[...excluded],impossibleAlignments:skipped}));
    } finally {await extractor.release();}
  } else {
    if(!primary.adversarialFeature||!values.audio||!values.evidence)throw new Error("A fitted bundle, --audio and --evidence are required");
    const raw=json(values.evidence);
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const comparisons:unknown[]=[];
    const enriched=await addAdversarialEvidence(primary.adversarialFeature,primary.manifest,audio(values.audio),chunks,index,phrases,{},"free",
      (comparison,candidate)=>comparisons.push({candidate,comparison}));
    const replay=(evidence:PhonemeEvidence[])=> {
      const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases});
      const events=[];
      for(const e of insertionCells(evidence)){const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t);}
      const t=tracker.end();events.push(...t);checker.end(t);return {events,mistakes:checker.mistakes()};
    };
    const before=replay(chunks),after=replay(enriched);
    if(JSON.stringify(before.events)!==JSON.stringify(after.events))throw new Error("Feature certificates changed tracker events");
    if(JSON.stringify(before.mistakes.filter(e=>e.errorClass==="added_word"))!==JSON.stringify(after.mistakes.filter(e=>e.errorClass==="added_word")))
      throw new Error("Feature certificates changed insertion decisions");
    writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
    console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.adversarialFeatureVetoes??[])).length}));
  }
} finally {await primary.backend.release();}
