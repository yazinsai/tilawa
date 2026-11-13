/** Prepare training-speaker soft-occupancy data or inspect engine-requested comparisons.
 * No label supplies a performed mistake transcript or changes an inference window. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { addChannelTransitionEvidence, featureObservations, channelTransitionTrainingRole, type ChannelTransitionModel } from "../channel-transition.js";
import { loadNodeModel } from "../node.js";
import { ortBackend } from "../ort.js";
import type { OrtModule } from "../ort.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{model:{type:"string"},extractor:{type:"string"},labels:{type:"string"},
  views:{type:"string"},"excluded-speakers":{type:"string"},out:{type:"string"},audio:{type:"string"},evidence:{type:"string"}}});
if(!values.model||!values.out)throw new Error("--model and --out are required");
const json=(path:string)=>JSON.parse(readFileSync(path,"utf8"));
const asset=(name:string)=>json(new URL(`../../../core/assets/${name}`,import.meta.url).pathname);
const index=loadReferenceIndex(asset("ref-v2.json")),phrases=asset("phrases-v1.json");
const audio=(path:string)=> {
  const wav=decodeWav(readFileSync(path));
  if(wav.sampleRate!==16000)throw new Error("Frozen scorer requires 16 kHz WAV audio");
  return wav.samples;
};
const primary=await loadNodeModel(resolve(values.model));
try {
  if(values.labels) {
    if(!values.extractor||!values["excluded-speakers"]||!values.views)throw new Error("--extractor, --views and --excluded-speakers are required");
    const excluded=new Set<string>(json(values["excluded-speakers"]));
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for(const r of labels)if(r.split!=="train")excluded.add(r.speaker);
    const metadata=json(resolve(values.extractor,"channel-transition-metadata.json")) as ChannelTransitionModel;
    if(metadata.primarySha256!==primary.manifest.sha256)throw new Error("Extractor differs from primary graph");
    const extractor=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"channel-transition.onnx"),"onnxruntime-node",{
      executionProviders:["cpu"],intraOpNumThreads:1,graphOptimizationLevel:"disabled"});
    try {
      const rows:any[]=[];
      const views=json(values.views);
      for(const r of labels) {
        const role=channelTransitionTrainingRole(r,excluded);
        if(role!=="none"||!views[r.id])continue;
        if(!r.speaker||!r.wav||!r.ayahs?.length)throw new Error("Training label lacks speaker, WAV or ayahs");
        const words=(r.ayahs as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
        if(!words.length)throw new Error("No canonical training pronunciation");
        const pronunciation=words.map(w=>index.phonemes.slice(w.start,w.end)).join("");
        const ids=[...pronunciation].map(p=>metadata.tokens.indexOf(p)+1);
        if(ids.some(p=>p<1))throw new Error("Unknown canonical phone");
        const extract=async(path:string)=>featureObservations(extractor,primary.manifest,metadata,audio(path));
        const clean=await extract(views[r.id].clean);
        const partitions:Record<string,number[][][][]>={};
        for(const partition of ["train","validation"]){
          const triplets:number[][][][]=[];
          for(const paths of views[r.id][partition]){
            const features:number[][][]=[];
            for(const path of paths)features.push((await extract(path)).observations);
            triplets.push(features);
          }
          partitions[partition]=triplets;
        }
        rows.push({source:"help-train",split:"train",excluded:false,speaker:r.speaker,id:r.id,
          kind:"none",clean:{features:clean.observations,logProbs:clean.logProbs},
          pronunciation:ids,...partitions});
      }
      writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
      console.log(JSON.stringify({recordings:rows.length,none:rows.filter(r=>r.kind==="none").length,
        speakers:new Set(rows.map(r=>r.speaker)).size,excludedSpeakers:[...excluded],}));
    } finally {await extractor.release();}
  } else {
    if(!primary.channelTransition||!values.audio||!values.evidence)throw new Error("A fitted bundle, --audio and --evidence are required");
    const raw=json(values.evidence);
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const comparisons:unknown[]=[];
    const enriched=await addChannelTransitionEvidence(primary.channelTransition,primary.manifest,audio(values.audio),chunks,index,phrases,{},"free",
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
    console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.channelTransitionVetoes??[])).length}));
  }
} finally {await primary.backend.release();}
