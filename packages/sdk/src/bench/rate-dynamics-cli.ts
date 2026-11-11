/** Prepare frozen training observations, or inspect complete comparisons requested by the ordinary engine.
 * No incident labels, target vocabulary, scorer or practice-set selection participates in inference. */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, RateEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { addRateEvidence, rateFeatures, validateRateModel } from "../rate-dynamics.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{labels:{type:"string"},"excluded-speakers":{type:"string"},out:{type:"string"},
  emissions:{type:"string"},audio:{type:"string"},evidence:{type:"string"}}});
if (!values.out) throw new Error("--out is required");
const json=(path:string)=>JSON.parse(readFileSync(path,"utf8"));
const asset=(name:string)=>json(new URL(`../../../core/assets/${name}`,import.meta.url).pathname);
const index=loadReferenceIndex(asset("ref-v2.json")),phrases=asset("phrases-v1.json"),tokens=[...index.tokens].join("");
function audio(path:string):Float32Array {
  const wav=decodeWav(readFileSync(path));
  if (wav.sampleRate!==16000) throw new Error("Frozen scorer observations require 16 kHz WAV audio");
  return wav.samples;
}
if (values.labels) {
  const excluded=new Set<string>(values["excluded-speakers"] ? json(values["excluded-speakers"]) : []);
  const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
  const rows=[];
  // A speaker with any held-out recording is excluded from all of its takes.
  for (const r of labels) if (r.split!=="train") excluded.add(r.speaker);
  for (const r of labels) {
    if (r.kind!=="none"||r.split!=="train"||excluded.has(r.speaker)) continue;
    if (!r.speaker||!r.wav||!Array.isArray(r.ayahs)||!r.ayahs.length) throw new Error("Training label lacks speaker, audio or ayahs");
    const words=(r.ayahs as string[]).flatMap((a:string)=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
    if (!words.length) throw new Error("No pronunciation for the training label");
    const pronunciation=words.map(w=>index.phonemes.slice(w.start,w.end)).join("");
    rows.push({source:"help-train",kind:"none",split:"train",excluded:false,speaker:r.speaker,id:r.id,
      pronunciation,features:rateFeatures(audio(resolve(dirname(values.labels),r.wav)))});
  }
  writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
  writeFileSync(values.out+".excluded.json",JSON.stringify([...excluded].sort()));
  const metadata={recordings:rows.length,speakers:new Set(rows.map(r=>r.speaker)).size,inventory:index.inventoryId,tokens};
  writeFileSync(values.out+".metadata.json",JSON.stringify(metadata));
  console.log(JSON.stringify(metadata));
} else {
  if (!values.emissions||!values.audio||!values.evidence) throw new Error("--emissions, --audio and --evidence are required for a comparison probe");
  const model=validateRateModel(json(values.emissions),index.inventoryId,tokens);
  const raw=json(values.evidence);
  const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
  const comparisons:{candidate:SequenceEvidenceCandidate;comparison:RateEvidence|null}[]=[];
  const enriched=addRateEvidence(model,audio(values.audio),chunks,index,phrases,{},"free",(comparison,candidate)=>comparisons.push({candidate,comparison}));
  const replay=(evidence:PhonemeEvidence[])=> {
    const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases});
    const events=[];
    for (const e of insertionCells(evidence)) {const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t);}
    const t=tracker.end();events.push(...t);checker.end(t);
    return {events,mistakes:checker.mistakes()};
  };
  const before=replay(chunks),after=replay(enriched);
  if (JSON.stringify(before.events)!==JSON.stringify(after.events)) throw new Error("Rate metadata changed tracker events");
  const otherDecisions=(r:ReturnType<typeof replay>)=>r.mistakes.filter(e=>e.errorClass!=="wrong_word");
  if (JSON.stringify(otherDecisions(before))!==JSON.stringify(otherDecisions(after))) throw new Error("Rate metadata changed another error class");
  const primary=(es:PhonemeEvidence[])=>es.map(e=>({...e,phonemes:e.phonemes.map(p=>{
    const {rateVetoes:_,...token}=p; return token;
  })}));
  if (JSON.stringify(primary(chunks))!==JSON.stringify(primary(enriched))) throw new Error("Rate metadata changed primary tokens or posteriors");
  writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
  console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.rateVetoes??[])).length}));
}
