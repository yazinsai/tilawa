/** Cache training-speaker observations or inspect exact engine-requested comparison windows. */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { carryMultigramEvidence, multigramComparison, multigramCtcScore, multigramObservations, eligibleMultigramTake, readout, validateMultigramModel } from "../multigram.js";
import type { MultigramMetadata } from "../multigram.js";
import { loadNodeModel } from "../node.js";
import { ortBackend } from "../ort.js";
import type { OrtModule } from "../ort.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{model:{type:"string"},extractor:{type:"string"},labels:{type:"string"},
  "excluded-speakers":{type:"string"},out:{type:"string"},audio:{type:"string"},evidence:{type:"string"},fitted:{type:"string"},verify:{type:"string"}}});
if(!values.model||!values.extractor||!values.out)throw new Error("--model, --extractor, --out required");
const json=(p:string)=>JSON.parse(readFileSync(p,"utf8"));
const asset=(name:string)=>json(new URL(`../../../core/assets/${name}`,import.meta.url).pathname);
const index=loadReferenceIndex(asset("ref-v2.json")),phrases=asset("phrases-v1.json");
const audio=(p:string)=> {
  const wav=decodeWav(readFileSync(p));
  if(wav.sampleRate!==16000)throw new Error("16 kHz WAV required");
  return wav.samples;
};
const primary=await loadNodeModel(resolve(values.model));
const metadata=json(resolve(values.extractor,"multigram-metadata.json")) as MultigramMetadata;
if(metadata.primarySha256!==primary.manifest.sha256)throw new Error("Extractor primary mismatch");
const extractor=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"multigram-extractor.onnx"),
  "onnxruntime-node",{executionProviders:["cpu"],intraOpNumThreads:1});
try {
  if(values.labels) {
    if(!values["excluded-speakers"])throw new Error("Explicit excluded speaker list required");
    const excluded=new Set<string>(json(values["excluded-speakers"]));
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(s=>JSON.parse(s));
    for(const r of labels)if(r.split!=="train")excluded.add(r.speaker);
    const rows=[];
    for(const r of labels) {
      if(!eligibleMultigramTake(r,excluded))continue;
      if(!r.wav||!r.ayahs?.length)throw new Error("Incomplete training label");
      const parts=(r.ayahs as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start)
        .map(w=>index.phonemes.slice(w.start,w.end)));
      if(!parts.length)throw new Error("Missing training pronunciation");
      const features=await multigramObservations(extractor,primary.manifest,metadata,audio(resolve(dirname(values.labels),r.wav)));
      rows.push({source:"help-train",kind:"none",split:"train",speaker:r.speaker,id:r.id,
        pronunciation:parts.join(""),parts,features:features.encoder});
    }
    writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
    console.log(JSON.stringify({recordings:rows.length,speakers:new Set(rows.map(r=>r.speaker)).size}));
  } else if(values.verify) {
    if(!values.fitted)throw new Error("--fitted required");
    const m=validateMultigramModel(json(values.fitted),primary.manifest),parity=json(values.verify);
    const actual=[m.multigram,m.control].map(h=>readout(h,parity.features,m.center,m.scale));
    let maximumError=0;
    for(let a=0;a<2;a++)for(let t=0;t<actual[a].length;t++)for(let p=0;p<actual[a][t].length;p++) {
      const error=Math.abs(actual[a][t][p]-parity.expected[a][t][p]);
      if(!Number.isFinite(error))throw new Error("Invalid readout parity values");
      maximumError=Math.max(maximumError,error);
    }
    if(maximumError>1e-4)throw new Error("Exported readout differs from training runtime");
    const scores=[m.multigram,m.control].map((h,a)=> {
      const rows=readout(h,parity.scoreFeatures,m.center,m.scale);
      return multigramCtcScore(a ? rows.map(r=>r.slice(0,m.tokens.length+1)) : rows,
        parity.pronunciation,a ? [...m.tokens] : m.units);
    });
    if(scores.some((s,a)=>!Number.isFinite(s)||Math.abs(s-parity.scores[a])>1e-3))
      throw new Error("Exported segmentation sum differs from training runtime");
    writeFileSync(values.out,JSON.stringify({maximumError,frames:parity.features.length}));
    console.log(JSON.stringify({maximumError,frames:parity.features.length}));
  } else {
    if(!values.fitted||!values.audio||!values.evidence)throw new Error("--fitted, --audio, --evidence required");
    const model=validateMultigramModel(json(values.fitted),primary.manifest),raw=json(values.evidence);
    if(model.primarySha256!==primary.manifest.sha256)throw new Error("Readout primary mismatch");
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{...raw.posteriors,
      startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const candidates=new Map<string,SequenceEvidenceCandidate>();
    const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000});
    const checker=createChecker(index,{phrases,sequenceEvidence:c=>{candidates.set(JSON.stringify(c),c);}});
    for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));
    checker.end(tracker.end());
    const features=await multigramObservations(extractor,primary.manifest,model,audio(values.audio));
    const comparisons=[...candidates.values()].map(c=>({candidate:c,comparison:
      multigramComparison(model,features.encoder.slice(c.fromFrame,c.toFrame),c)}));
    const certificates=comparisons.flatMap(c=>c.comparison?.veto?[c.comparison]:[]);
    const replay=(evidence:PhonemeEvidence[])=> {
      const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases});
      const events=[];
      for(const e of insertionCells(evidence)) {
        const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t);
      }
      const t=tracker.end();events.push(...t);checker.end(t);
      return {events,mistakes:checker.mistakes()};
    };
    const before=replay(chunks),after=replay(carryMultigramEvidence(chunks,certificates));
    if(JSON.stringify(before.events)!==JSON.stringify(after.events))throw new Error("Certificate changed tracker events");
    if(JSON.stringify(before.mistakes.filter(e=>e.errorClass!=="wrong_word"))!==
      JSON.stringify(after.mistakes.filter(e=>e.errorClass!=="wrong_word")))throw new Error("Certificate changed another decision class");
    writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
    console.log(JSON.stringify({comparisons:comparisons.length,vetoes:comparisons.filter(c=>c.comparison?.veto).length}));
  }
} finally {await extractor.release();await primary.backend.release();}
