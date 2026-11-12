/** Inspect actual engine requests; labels never choose comparison windows or rivals. */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { addGlottalSourceEvidence } from "../glottal-source.js";
import { loadNodeModel, readWav } from "../node.js";
import { insertionCells } from "../seam-insertion.js";

const {values}=parseArgs({options:{model:{type:"string"},audio:{type:"string"},evidence:{type:"string"},out:{type:"string"}}});
if (!values.model||!values.audio||!values.evidence||!values.out) throw new Error("model, audio, evidence and out are required");
const json=(path:string)=>JSON.parse(readFileSync(path,"utf8"));
const index=loadReferenceIndex(json(new URL("../../../core/assets/ref-v2.json",import.meta.url).pathname));
const phrases=json(new URL("../../../core/assets/phrases-v1.json",import.meta.url).pathname);
const model=await loadNodeModel(values.model);
try {
  if (!model.glottalSource) throw new Error("A complete glottal-source bundle is required");
  const wav=await readWav(values.audio);
  if (wav.sampleRate!==16000) throw new Error("The evidence clock requires 16 kHz WAV audio");
  const raw=json(values.evidence);
  const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
    ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
  const comparisons:unknown[]=[];
  const enriched=await addGlottalSourceEvidence(model.glottalSource,model.manifest,wav.samples,chunks,index,phrases,{},"free",
    (comparison,candidate)=>comparisons.push({candidate,comparison}));
  const replay=(evidence:PhonemeEvidence[])=> {
    const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases});
    const events=[];
    for (const e of insertionCells(evidence)) {const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t);}
    const t=tracker.end();events.push(...t);checker.end(t);
    return {events,mistakes:checker.mistakes()};
  };
  const before=replay(chunks),after=replay(enriched);
  if (JSON.stringify(before.events)!==JSON.stringify(after.events)) throw new Error("Glottal metadata changed tracker events");
  if (JSON.stringify(before.mistakes.filter(e=>e.errorClass!=="wrong_word"))!==
      JSON.stringify(after.mistakes.filter(e=>e.errorClass!=="wrong_word"))) throw new Error("Glottal metadata changed other decisions");
  writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
  console.log(JSON.stringify({comparisons:comparisons.length,
    certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.glottalSourceVetoes??[])).length}));
} finally {await model.backend.release();}
