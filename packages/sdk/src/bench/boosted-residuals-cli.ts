/** Prepare eligible frozen training observations or inspect engine-requested comparisons.
 * Labels never choose inference windows, pronunciations or acceptance rules. */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { addResidualEvidence, residualObservations, eligibleResidualTake, eligiblePreservationTake, type ResidualMetadata } from "../boosted-residuals.js";
import { loadNodeModel } from "../node.js";
import { ortBackend } from "../ort.js";
import type { OrtModule } from "../ort.js";
import { createListener } from "../listener.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{model:{type:"string"},extractor:{type:"string"},labels:{type:"string"},
  "excluded-speakers":{type:"string"},out:{type:"string"},verify:{type:"boolean"},audio:{type:"string"},evidence:{type:"string"}}});
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
    if (!primary.boostedResiduals) throw new Error("No declared residual ensemble");
    const wave=Float32Array.from({length:24000},(_,i)=>.1*Math.sin(i*.071)+.03*Math.sin(i*.29));
    const observed=await residualObservations(primary.boostedResiduals.backend,primary.manifest,primary.boostedResiduals.model,wave);
    const listener=createListener(primary,{posteriors:true,topK:primary.manifest.tokens.length+1});
    const chunks:PhonemeEvidence[]=[];
    for(let i=0;i<wave.length;i+=113) { const e=await listener.push(wave.slice(i,i+113),16000);if(e)chunks.push(e); }
    const end=await listener.end();if(end)chunks.push(end);
    let frames=0;
    for(const e of chunks) if(e.posteriors) {
      const p=e.posteriors;
      for(let t=0;t<p.nFrames;t++) for(let k=0;k<p.k;k++) {
        const at=t*p.k+k,id=p.ids[at],lp=observed.logProbs[p.startFrame+t]?.[id];
        if(!Number.isFinite(lp)||Math.round(Math.exp(lp)*1e4)/1e4!==p.probs[at])throw new Error("Primary/extractor streaming parity failed");
      }
      frames+=p.nFrames;
    }
    if(frames!==observed.features.length)throw new Error("Primary/extractor frame clocks differ");
    const report={verified:true,frames,trees:primary.boostedResiduals.model.trees.length,primarySha256:primary.manifest.sha256};
    writeFileSync(values.out,JSON.stringify(report,null,2));console.log(JSON.stringify(report));
  } else if (values.labels) {
    if (!values.extractor||!values["excluded-speakers"]) throw new Error("--extractor and --excluded-speakers are required");
    const excluded=new Set<string>(json(values["excluded-speakers"]));
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for (const r of labels) if (r.split!=="train") excluded.add(r.speaker);
    const metadata=json(resolve(values.extractor,"boosted-residuals-metadata.json")) as ResidualMetadata;
    if (metadata.primarySha256!==primary.manifest.sha256) throw new Error("Extractor differs from primary graph");
    const extractor=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"boosted-residuals.onnx"),"onnxruntime-node",{
      executionProviders:["cpu"],intraOpNumThreads:1});
    try {
      const rows=[];
      for (const r of labels) {
        if (!eligiblePreservationTake(r,excluded)) continue;
        const transcript=eligibleResidualTake(r,excluded);
        if (!transcript && r.kind === "none") continue;
        if (!r.speaker||!r.wav|| (transcript&&!r.ayahs?.length)) throw new Error("Training label lacks speaker, WAV or ayahs");
        const words=((r.ayahs??[]) as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
        if (transcript&&!words.length) throw new Error("No canonical training pronunciation");
        const pronunciation=words.map(w=>index.phonemes.slice(w.start,w.end)).join("");
        const features=await residualObservations(extractor,primary.manifest,metadata,audio(resolve(dirname(values.labels),r.wav)));
        rows.push({source:"help-train",kind:r.kind,split:"train",excluded:false,speaker:r.speaker,id:r.id,
          role:transcript?"transcript":"preservation", ...(transcript?{pronunciation}:{}),
          events:r.events??[],non_errors:r.non_errors??[],features:features.features,logits:features.logits});
      }
      writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
      console.log(JSON.stringify({recordings:rows.length,speakers:new Set(rows.map(r=>r.speaker)).size}));
    } finally { await extractor.release(); }
  } else {
    if (!primary.boostedResiduals||!values.audio||!values.evidence) throw new Error("A fitted bundle, --audio and --evidence are required");
    const raw=json(values.evidence);
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const comparisons:unknown[]=[];
    const enriched=await addResidualEvidence(primary.boostedResiduals,primary.manifest,audio(values.audio),chunks,index,phrases,{},"free",
      (comparison,candidate)=>comparisons.push({candidate,comparison}));
    const replay=(evidence:PhonemeEvidence[])=> {
      const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases});
      const events=[];
      for (const e of insertionCells(evidence)) { const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t); }
      const t=tracker.end();events.push(...t);checker.end(t);
      return {events,mistakes:checker.mistakes()};
    };
    const before=replay(chunks),after=replay(enriched);
    if (JSON.stringify(before.events)!==JSON.stringify(after.events)) throw new Error("Residual metadata changed tracker events");
    if (JSON.stringify(before.mistakes.filter(e=>e.errorClass==="added_word"))!==
      JSON.stringify(after.mistakes.filter(e=>e.errorClass==="added_word"))) throw new Error("Residual metadata changed insertion decisions");
    writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
    console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.boostedResidualVetoes??[])).length}));
  }
} finally { await primary.backend.release(); }
