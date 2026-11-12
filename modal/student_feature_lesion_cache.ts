/** Training feature cache and read-only requested-window diagnostics. */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { createRequire } from "node:module";
const ort = createRequire(new URL("../packages/sdk/package.json", import.meta.url))("onnxruntime-node");
import { loadReferenceIndex } from "../packages/core/src/reference.js";
import type { PhonemeEvidence } from "../packages/core/src/evidence.js";
import { createListener } from "../packages/sdk/src/listener.js";
import { featureLesionObservations, featureLesionRequests } from "../packages/sdk/src/feature-lesion.js";
import type { FeatureLesionMetadata } from "../packages/sdk/src/feature-lesion.js";
import { loadNodeModel } from "../packages/sdk/src/node.js";
import { ortBackend } from "../packages/sdk/src/ort.js";
import type { OrtModule } from "../packages/sdk/src/ort.js";
import { decodeWav } from "../packages/sdk/src/wav.js";
const {values}=parseArgs({options:{model:{type:"string"},extractor:{type:"string"},labels:{type:"string"},
  audio:{type:"string"},evidence:{type:"string"},out:{type:"string"}}});
if (!values.model || !values.extractor || !values.out) throw new Error("--model, --extractor and --out required");
const json=(p:string)=>JSON.parse(readFileSync(p,"utf8"));
const index=loadReferenceIndex(json(new URL("../packages/core/assets/ref-v2.json",import.meta.url).pathname));
const phrases=json(new URL("../packages/core/assets/phrases-v1.json",import.meta.url).pathname);
const meta=json(resolve(values.extractor,"feature-lesion-metadata.json")) as FeatureLesionMetadata;
const primary=await loadNodeModel(values.model);
if (meta.primarySha256!==primary.manifest.sha256) throw new Error("Extractor does not match primary");
const backend=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"feature-lesion.onnx"),"onnxruntime-node",{
 executionProviders:["cpu"],intraOpNumThreads:1});
mkdirSync(values.out,{recursive:true});
const rows:any[]=[];
try {
 const labels=values.labels?readFileSync(values.labels,"utf8").trim().split(/\r?\n/).map(s=>JSON.parse(s)):[];
 // A nontraining take excludes its entire speaker, even if another take says train.
 const excluded=new Set(labels.filter(r=>r.split!=="train").map(r=>r.speaker));
 const inputs=values.labels?labels.filter(r=>r.split==="train"&&r.speaker&&!excluded.has(r.speaker)&&!r.exclude&&!r.excluded&&r.scored!==false):
  [{id:"diagnostic",wav:values.audio}];
 for (let i=0;i<inputs.length;i++) {
  const r=inputs[i];
  if (!r.wav) throw new Error("No recording path");
  const wav=decodeWav(readFileSync(values.labels?resolve(dirname(values.labels),r.wav):r.wav));
  if (wav.sampleRate!==16000) throw new Error("16 kHz WAV required");
  const observations=await featureLesionObservations(backend,primary.manifest,meta,wav.samples);
  let chunks:PhonemeEvidence[]=[];
  if (values.evidence) {
   const raw=json(values.evidence);
   chunks=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{...raw.posteriors,
    startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
  } else {
   const listener=createListener(primary,{posteriors:true});
   for (let a=0;a<wav.samples.length;a+=5120) { const e=await listener.push(wav.samples.subarray(a,a+5120),16000); if(e)chunks.push(e); }
   const tail=await listener.end();if(tail)chunks.push(tail);
  }
  const result=featureLesionRequests(chunks,index,phrases);
  const words=(r.ayahs??[]).flatMap((a:string)=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
  const base=createHash("sha256").update(r.id).digest("hex").slice(0,16);
  for (const [key,width] of [["features",meta.width],["logits",meta.tokens.length+1],["logProbs",meta.tokens.length+1]] as const) {
   const data=Float32Array.from(observations[key].flat());
   writeFileSync(resolve(values.out,`${base}-${key}.f32`),Buffer.from(data.buffer));
   if (data.length!==observations.features.length*width)throw new Error("Feature shape mismatch");
  }
  const row={...r,source:values.labels?"help-train":"diagnostic",excluded:false,
   cache:base,frames:observations.features.length,width:meta.width,
   pronunciation:words.map(w=>index.phonemes.slice(w.start,w.end)).join(""),
   words:words.map(w=>({word:{surah:w.surah,ayah:w.ayah,word:w.word},pronunciation:index.phonemes.slice(w.start,w.end)})),
   candidates:result.candidates,mistakes:result.mistakes};
  rows.push(row);
  // Save completed rows so a bounded job can resume without losing evidence.
  writeFileSync(resolve(values.out,"rows.jsonl"),rows.map(x=>JSON.stringify(x)).join("\n")+"\n");
  console.log(JSON.stringify({done:i+1,total:inputs.length,kind:r.kind,comparisons:result.candidates.length}));
 }
 writeFileSync(resolve(values.out,"excluded-speakers.json"),JSON.stringify([...excluded]));
} finally {await backend.release();await primary.backend.release();}
