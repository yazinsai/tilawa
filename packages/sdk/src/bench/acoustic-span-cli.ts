/** Prepare training-speaker acoustic contrasts or inspect engine-requested comparisons.
 * Scripted spoken spellings need complete acoustic support; labels never set an inference window. */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { addAcousticSpanEvidence, compatibilityScore, acousticSpanFeatures, type AcousticSpanModel } from "../acoustic-span.js";
import { loadNodeModel } from "../node.js";
import { ortBackend } from "../ort.js";
import type { OrtModule } from "../ort.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";
import { Resampler } from "../audio.js";
import { trainingPronunciations, transcriptLogProbability } from "../acoustic-span-training.js";

const {values}=parseArgs({options:{model:{type:"string"},extractor:{type:"string"},labels:{type:"string"},
  "excluded-speakers":{type:"string"},out:{type:"string"},audio:{type:"string"},evidence:{type:"string"},parity:{type:"string"}}});
if(!values.model||!values.out)throw new Error("--model and --out are required");
const json=(path:string)=>JSON.parse(readFileSync(path,"utf8"));
const asset=(name:string)=>json(new URL(`../../../core/assets/${name}`,import.meta.url).pathname);
const index=loadReferenceIndex(asset("ref-v2.json")),phrases=asset("phrases-v1.json");
const audio=(path:string)=> {
  const wav=decodeWav(readFileSync(path));
  if(wav.sampleRate===16000)return wav.samples;
  const resampler=new Resampler(wav.sampleRate,16000);
  const body=resampler.push(wav.samples),tail=resampler.flush();
  const samples=new Float32Array(body.length+tail.length);
  samples.set(body);samples.set(tail,body.length);return samples;
};
/** Forced alignment supplies boundaries for complete supported training transcripts. */
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
const primary=await loadNodeModel(resolve(values.model));
try {
  if(values.parity) {
    if(!primary.acousticSpans)throw new Error("Parity requires the complete fitted bundle");
    const scorer=primary.acousticSpans,errors:number[]=[];
    for(const arm of ["contrast","control"] as const) {
      const fixtures=json(resolve(values.parity,`acoustic-span-${arm}-parity.json`));
      for(const f of fixtures) {
        const pronunciation=f.ids.map((i:number)=>scorer.model.tokens[i-1]).join("");
        const score=compatibilityScore(scorer[arm],scorer.model,f.features,pronunciation);
        if(score===null||Math.abs(score-f.score)>1e-5)throw new Error("Exported compatibility head parity failed");
        errors.push(Math.abs(score-f.score));
      }
    }
    if(!errors.length)throw new Error("No training-only parity fixtures");
    writeFileSync(values.out,JSON.stringify({comparisons:errors.length,maxError:Math.max(...errors)}));
  } else if(values.labels) {
    if(!values.extractor||!values["excluded-speakers"])throw new Error("--extractor and --excluded-speakers are required");
    const excluded=new Set<string>(json(values["excluded-speakers"]));
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for(const r of labels)if(r.split!=="train")excluded.add(r.speaker);
    const metadata=json(resolve(values.extractor,"acoustic-span-features-metadata.json")) as AcousticSpanModel;
    if(metadata.primarySha256!==primary.manifest.sha256)throw new Error("Extractor differs from primary graph");
    const extractor=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"acoustic-span-features.onnx"),"onnxruntime-node",{
      executionProviders:["cpu"],intraOpNumThreads:1,graphOptimizationLevel:"disabled"});
    try {
      const rows:any[]=[],skipped:string[]=[];
      for(const r of labels) {
        if(r.split!=="train"||excluded.has(r.speaker)||r.excluded||r.exclude)continue;
        const transcript=trainingPronunciations(index,r);
        if(!transcript)continue;
        if(r.kind==="none"&&((r.events??[]).length||(r.non_errors??[]).length))throw new Error("None-labelled anchor contains scripted deviations");
        if(!r.speaker||!r.wav||!r.ayahs?.length)throw new Error("Training label lacks speaker, WAV or ayahs");
        const words=(r.ayahs as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
        if(!words.length)throw new Error("No canonical training pronunciation");
        const samples=audio(resolve(dirname(values.labels),r.wav));
        const features=await acousticSpanFeatures(extractor,primary.manifest,metadata,samples);
        const capture=String(r.device??r.channel??r.recording_device??"unknown");
        const rms=Math.sqrt(samples.reduce((v,s)=>v+s*s,0)/Math.max(1,samples.length));
        const descriptor=[Math.log(Math.max(rms,1e-8))];
        for(let d=0;d<8;d++) descriptor.push(features.observations.reduce((v,r)=>v+r[d],0)/Math.max(1,features.observations.length));
        const provenance={capture,descriptor,source:"help-train",split:"train",excluded:false,speaker:r.speaker,id:r.id};
        if(transcript) {
          const pronunciation=transcript.recorded.join("");
          const ids=[...pronunciation].map(p=>metadata.tokens.indexOf(p)+1);
          if(ids.some(p=>p<1))throw new Error("Unknown canonical phone");
          if(transcript.kind==="supported-substitution") {
            const writtenIds=[...transcript.written.join("")].map(p=>metadata.tokens.indexOf(p)+1);
            if(!(transcriptLogProbability(features.logProbs,ids)>transcriptLogProbability(features.logProbs,writtenIds))) {
              skipped.push(r.id);continue;
            }
          }
          const states=features.logProbs.length?alignment(features.logProbs,ids):null;
          if(!states){skipped.push(r.id);continue;}
          let offset=0;
          const wordCrops=[];
          for(const phones of transcript.recorded) {
            const count=phones.length, first=offset;
            offset+=count;
            const phoneBounds=[];
            for(let p=first;p<offset;p++) {
              const frames=states.flatMap((s,t)=>s===2*p+1?[t]:[]);
              if(!frames.length)break;
              phoneBounds.push([frames[0],frames.at(-1)!+1]);
            }
            if(phoneBounds.length!==count)continue;
            const firstState=2*first+1,lastState=2*(offset-1)+1;
            let previous=-1;
            for(let t=0;t<states.length;t++)if(states[t]<firstState-1)previous=t;
            const next=states.findIndex(s=>s>lastState+1);
            const from=Math.floor((Math.max(0,previous+1)+phoneBounds[0][0])/2);
            const to=Math.ceil((phoneBounds.at(-1)![1]+(next<0?states.length:next))/2);
            wordCrops.push({from,to,ids:ids.slice(first,offset),phoneBounds});
          }
          rows.push({...provenance,kind:transcript.kind,features:features.observations,words:wordCrops});

        }
      }
      writeFileSync(resolve(dirname(values.out),"excluded-speakers.json"),JSON.stringify([...excluded]));
      writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
      console.log(JSON.stringify({recordings:rows.length,none:rows.filter(r=>r.kind==="none").length,substitutions:rows.filter(r=>r.kind==="supported-substitution").length,
        speakers:new Set(rows.map(r=>r.speaker)).size,excludedSpeakers:[...excluded],impossibleAlignments:skipped}));
    } finally {await extractor.release();}
  } else {
    if(!primary.acousticSpans||!values.audio||!values.evidence)throw new Error("A fitted bundle or --extractor, --audio and --evidence are required");
    const raw=json(values.evidence);
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const comparisons:unknown[]=[];
    const enriched=await addAcousticSpanEvidence(primary.acousticSpans,primary.manifest,audio(values.audio),chunks,index,phrases,{},"free",
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
    console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.acousticSpanVetoes??[])).length}));
  }
} finally {await primary.backend.release();}
