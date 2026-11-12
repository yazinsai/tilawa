import { createHash } from "node:crypto";
/** Prepare training-speaker soft-occupancy data or inspect engine-requested comparisons.
 * No label supplies a performed mistake transcript or changes an inference window. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, SequenceEvidenceCandidate, WordRef } from "@tilawa/core";
import { addAngularEvidence, featureObservations, type AngularModel } from "../angular-phone.js";
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
const key=(w:WordRef)=>`${w.surah}:${w.ayah}:${w.word}`;
const primary=await loadNodeModel(resolve(values.model));
try {
  if(values.labels) {
    if(!values.extractor||!values["excluded-speakers"])throw new Error("--extractor and --excluded-speakers are required");
    const excluded=new Set<string>(json(values["excluded-speakers"]));
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for(const r of labels)if(r.split!=="train")excluded.add(r.speaker);
    const metadata=json(resolve(values.extractor,"angular-phone-metadata.json")) as AngularModel;
    if(metadata.primarySha256!==primary.manifest.sha256)throw new Error("Extractor differs from primary graph");
    const extractor=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"angular-phone.onnx"),"onnxruntime-node",{
      executionProviders:["cpu"],intraOpNumThreads:1,graphOptimizationLevel:"disabled"});
    try {
      const rows:any[]=[];
      const cache=resolve(values.extractor,"training-cache");
      mkdirSync(cache,{recursive:true});
      const extractorHash=createHash("sha256").update(readFileSync(resolve(values.extractor,"angular-phone.onnx"))).digest("hex");
      for(const r of labels) {
        if(r.split!=="train"||excluded.has(r.speaker))continue;
        const substitution=(r.events??[]).some((e:any)=>e.errorClass==="wrong_word");
        if(r.kind!=="none"&&!substitution)continue;
        if(r.kind==="none"&&((r.events??[]).length||(r.non_errors??[]).length))throw new Error("None-labelled anchor contains scripted deviations");
        if(!r.speaker||!r.wav||!r.ayahs?.length)throw new Error("Training label lacks speaker, WAV or ayahs");
        const words=(r.ayahs as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
        if(!words.length)throw new Error("No canonical training pronunciation");
        const wavPath=resolve(dirname(values.labels),r.wav);
        const cacheKey=createHash("sha256").update(readFileSync(wavPath)).update(extractorHash).update(JSON.stringify(r)).digest("hex");
        const cachePath=resolve(cache,cacheKey+".json");
        if(existsSync(cachePath)){rows.push(json(cachePath));continue;}
        const samples=audio(wavPath);
        const features=await featureObservations(extractor,primary.manifest,metadata,samples);
        const provenance={source:"help-train",split:"train",excluded:false,speaker:r.speaker,id:r.id};
        if(r.kind==="none") {
          const pronunciation=words.map(w=>index.phonemes.slice(w.start,w.end)).join("");
          const ids=[...pronunciation].map(p=>metadata.tokens.indexOf(p)+1);
          if(ids.some(p=>p<1))throw new Error("Unknown canonical phone");
          rows.push({...provenance,kind:"none",features:features.observations,
            logProbs:features.logProbs,pronunciation:ids});
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
            logProbs:features.logProbs.slice(c.fromFrame,c.toFrame),written:[...c.written].map(p=>metadata.tokens.indexOf(p)+1),rival:[...c.rival].map(p=>metadata.tokens.indexOf(p)+1)}));
          rows.push({...provenance,kind:"substituted_word",features:features.observations,controls});
        }
        writeFileSync(cachePath,JSON.stringify(rows[rows.length-1]));
      }
      writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
      console.log(JSON.stringify({recordings:rows.length,none:rows.filter(r=>r.kind==="none").length,
        speakers:new Set(rows.map(r=>r.speaker)).size,excludedSpeakers:[...excluded],}));
    } finally {await extractor.release();}
  } else {
    if(!primary.angularPhone||!values.audio||!values.evidence)throw new Error("A fitted bundle, --audio and --evidence are required");
    const raw=json(values.evidence);
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const comparisons:unknown[]=[];
    const enriched=await addAngularEvidence(primary.angularPhone,primary.manifest,audio(values.audio),chunks,index,phrases,{},"free",
      (comparison,candidate,probe)=>comparisons.push({candidate,comparison,probe}));
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
    console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.angularPhoneVetoes??[])).length}));
  }
} finally {await primary.backend.release();}
