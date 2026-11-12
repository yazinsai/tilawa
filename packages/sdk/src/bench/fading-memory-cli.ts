/** Prepare eligible frozen training observations or inspect engine-requested comparisons.
 * Labels never choose inference windows, pronunciations or acceptance rules. */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { addFadingMemoryEvidence, fadingMemoryObservations, eligibleFadingMemoryTake, readouts, validateFadingMemoryModel, type FadingMemoryMetadata } from "../fading-memory.js";
import { loadNodeModel } from "../node.js";
import { ortBackend } from "../ort.js";
import type { OrtModule } from "../ort.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{model:{type:"string"},extractor:{type:"string"},labels:{type:"string"},
  "excluded-speakers":{type:"string"},out:{type:"string"},audio:{type:"string"},evidence:{type:"string"},verify:{type:"string"},fitted:{type:"string"}}});
if (!values.model||!values.out) throw new Error("--model and --out are required");
const json=(path:string)=>JSON.parse(readFileSync(path,"utf8"));
const asset=(name:string)=>json(new URL(`../../../core/assets/${name}`,import.meta.url).pathname);
const index=loadReferenceIndex(asset("ref-v2.json")),phrases=asset("phrases-v1.json");
const audio=(path:string)=> {
  const wav=decodeWav(readFileSync(path));
  if (wav.sampleRate!==16000) throw new Error("Frozen scorer requires 16 kHz WAV audio");
  return wav.samples;
};
const primary=await loadNodeModel(resolve(values.model));
try {
  if (values.verify) {
    if(!values.fitted)throw new Error("--fitted required for readout parity");
    const fitted=validateFadingMemoryModel(json(values.fitted),primary.manifest), parity=json(values.verify);
    const actual=readouts(fitted,parity.features);
    let maximumError=0;
    for(let a=0;a<3;a++)for(let t=0;t<actual[a].length;t++)for(let p=0;p<actual[a][t].length;p++) {
      const error=Math.abs(actual[a][t][p]-parity.expected[a][t][p]);
      if(!Number.isFinite(error))throw new Error("Nonfinite readout parity");
      maximumError=Math.max(maximumError,error);
    }
    if(maximumError>1e-4)throw new Error("Exported readouts differ from training runtime");
    writeFileSync(values.out,JSON.stringify({maximumError,frames:actual[0].length}));
    console.log(JSON.stringify({maximumError,frames:actual[0].length}));
  } else if (values.labels) {
    if (!values.extractor||!values["excluded-speakers"]) throw new Error("--extractor and --excluded-speakers are required");
    const excluded=new Set<string>(json(values["excluded-speakers"]));
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for (const r of labels) if (r.split!=="train") excluded.add(r.speaker);
    const metadata=json(resolve(values.extractor,"fading-memory-metadata.json")) as FadingMemoryMetadata;
    if (metadata.primarySha256!==primary.manifest.sha256) throw new Error("Extractor differs from primary graph");
    const extractor=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"fading-memory.onnx"),"onnxruntime-node",{
      executionProviders:["cpu"],intraOpNumThreads:1});
    try {
      const rows=[];
      for (const r of labels) {
        if (!eligibleFadingMemoryTake(r,excluded)) continue;
        if (!r.speaker||!r.wav||!r.ayahs?.length) throw new Error("Training label lacks speaker, WAV or ayahs");
        const words=(r.ayahs as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
        if (!words.length) throw new Error("No canonical training pronunciation");
        const pronunciation=words.map(w=>index.phonemes.slice(w.start,w.end)).join("");
        const features=await fadingMemoryObservations(extractor,primary.manifest,metadata,audio(resolve(dirname(values.labels),r.wav)));
        rows.push({source:"help-train",kind:"none",split:"train",excluded:false,speaker:r.speaker,id:r.id,
          pronunciation,features:features.encoder,log_probs:features.logProbs});
      }
      writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
      console.log(JSON.stringify({recordings:rows.length,speakers:new Set(rows.map(r=>r.speaker)).size}));
    } finally { await extractor.release(); }
  } else {
    if (!primary.fadingMemory||!values.audio||!values.evidence) throw new Error("A fitted bundle, --audio and --evidence are required");
    const raw=json(values.evidence);
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const comparisons:unknown[]=[];
    const enriched=await addFadingMemoryEvidence(primary.fadingMemory,primary.manifest,audio(values.audio),chunks,index,phrases,{},"free",
      (comparison,candidate)=>comparisons.push({candidate,comparison}));
    const replay=(evidence:PhonemeEvidence[])=> {
      const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases});
      const events=[];
      for (const e of insertionCells(evidence)) { const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t); }
      const t=tracker.end();events.push(...t);checker.end(t);
      return {events,mistakes:checker.mistakes()};
    };
    const before=replay(chunks),after=replay(enriched);
    if (JSON.stringify(before.events)!==JSON.stringify(after.events)) throw new Error("Fading-memory metadata changed tracker events");
    if (JSON.stringify(before.mistakes.filter(e=>e.errorClass!=="wrong_word"))!==
      JSON.stringify(after.mistakes.filter(e=>e.errorClass!=="wrong_word"))) throw new Error("Fading-memory metadata changed another decision class");
    writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
    console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.fadingMemoryVetoes??[])).length}));
  }
} finally { await primary.backend.release(); }
