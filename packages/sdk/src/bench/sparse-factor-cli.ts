/** Prepare eligible frozen training observations or inspect engine-requested comparisons.
 * Labels never choose inference windows, pronunciations or acceptance rules. */
import { appendFileSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { addSparseFactorEvidence, sparseFactorObservations, sparseFactorRows, eligibleSparseFactorTake, type SparseFactorMetadata } from "../sparse-factor.js";
import { loadNodeModel } from "../node.js";
import { ortBackend } from "../ort.js";
import type { OrtModule } from "../ort.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{model:{type:"string"},extractor:{type:"string"},labels:{type:"string"},
  "excluded-speakers":{type:"string"},out:{type:"string"},audio:{type:"string"},evidence:{type:"string"},verify:{type:"string"}}});
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
    if (!primary.sparseFactor) throw new Error("A complete fitted bundle is required");
    const report=json(values.verify), probe=report.final.parity, fitted=primary.sparseFactor.model;
    let error=0;
    for (const [i,arm] of [fitted.sparseFactor,fitted.dense,fitted.control].entries()) {
      const observed=sparseFactorRows(fitted,arm,probe.features);
      for (const [t,row] of observed.entries()) for (const [p,v] of row.entries())
        error=Math.max(error,Math.abs(v-probe.logRows[i][t][p]));
    }
    if (!Number.isFinite(error)||error>1e-4) throw new Error("Python/SDK readout parity failed");
    writeFileSync(values.out,JSON.stringify({maximumLogProbabilityError:error}));
    console.log(JSON.stringify({maximumLogProbabilityError:error}));
  } else if (values.labels) {
    if (!values.extractor) throw new Error("--extractor is required");
    const excluded=new Set<string>(values["excluded-speakers"] ? json(values["excluded-speakers"]) : []);
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for (const r of labels) if (r.split!=="train") excluded.add(r.speaker);
    const began=performance.now();
    const metadata=json(resolve(values.extractor,"sparse-factor-metadata.json")) as SparseFactorMetadata;
    if (metadata.primarySha256!==primary.manifest.sha256) throw new Error("Extractor differs from primary graph");
    const extractor=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"sparse-factor.onnx"),"onnxruntime-node",{
      executionProviders:["cpu"],intraOpNumThreads:1});
    try {
      let recordings=0;
      const speakers=new Set<string>();
      writeFileSync(values.out,"");
      for (const r of labels) {
        if (performance.now()-began>7200*1000) throw new Error("Feature extraction CPU budget reached");
        if (r.split!=="train" || !r.speaker || excluded.has(r.speaker) || r.exclude || r.excluded || r.scored===false) continue;
        const supervised=eligibleSparseFactorTake(r,excluded);
        if (!r.speaker||!r.wav||!r.ayahs?.length) throw new Error("Training label lacks speaker, WAV or ayahs");
        const words=(r.ayahs as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
        if (!words.length) throw new Error("No canonical training pronunciation");
        const pronunciation=words.map(w=>index.phonemes.slice(w.start,w.end)).join("");
        const features=await sparseFactorObservations(extractor,primary.manifest,metadata,audio(resolve(dirname(values.labels),r.wav)));
        appendFileSync(values.out,JSON.stringify({source:"help-train",kind:r.kind,split:"train",excluded:false,speaker:r.speaker,id:r.id,
          events:r.events,non_errors:r.non_errors,pronunciation:supervised ? pronunciation : null,
          features:features.features})+"\n");
        recordings++;speakers.add(r.speaker);
      }
      console.log(JSON.stringify({recordings,speakers:speakers.size}));
    } finally { await extractor.release(); }
  } else {
    if (!primary.sparseFactor||!values.audio||!values.evidence) throw new Error("A fitted bundle, --audio and --evidence are required");
    const raw=json(values.evidence);
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const comparisons:unknown[]=[];
    const enriched=await addSparseFactorEvidence(primary.sparseFactor,primary.manifest,audio(values.audio),chunks,index,phrases,{},"free",
      (comparison,candidate)=>comparisons.push({candidate,comparison}));
    const replay=(evidence:PhonemeEvidence[])=> {
      const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases});
      const events=[];
      for (const e of insertionCells(evidence)) { const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t); }
      const t=tracker.end();events.push(...t);checker.end(t);
      return {events,mistakes:checker.mistakes()};
    };
    const before=replay(chunks),after=replay(enriched);
    if (JSON.stringify(before.events)!==JSON.stringify(after.events)) throw new Error("Sparse-factor metadata changed tracker events");
    if (JSON.stringify(before.mistakes.filter(e=>e.errorClass==="added_word"))!==
      JSON.stringify(after.mistakes.filter(e=>e.errorClass==="added_word"))) throw new Error("Sparse-factor metadata changed insertion decisions");
    writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
    console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.sparseFactorVetoes??[])).length}));
  }
} finally { await primary.backend.release(); }
