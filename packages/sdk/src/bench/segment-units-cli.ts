/** Cache frozen training trajectories or inspect complete comparisons requested by the ordinary checker. */
import { appendFileSync, copyFileSync, createReadStream, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { canonicalSegmentTake, eligibleSegmentTake, segmentComparison, segmentObservations, segmentReversal, validateSegmentModel } from "../segment-units.js";
import type { SegmentMetadata, SegmentModel } from "../segment-units.js";
import { loadNodeModel } from "../node.js";
import { ortBackend } from "../ort.js";
import type { OrtModule } from "../ort.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";
import { Resampler } from "../audio.js";
import { MODEL_REGISTRY } from "../registry.js";

const {values}=parseArgs({options:{model:{type:"string"},extractor:{type:"string"},labels:{type:"string"},
  "excluded-speakers":{type:"string"},out:{type:"string"},audio:{type:"string"},evidence:{type:"string"},scorer:{type:"string"},
  bundle:{type:"boolean"},"auxiliary-dir":{type:"string"}}});
if(!values.model||!values.extractor||!values.out)throw new Error("--model, --extractor and --out are required");
const json=(path:string)=>JSON.parse(readFileSync(path,"utf8"));
const fingerprint=async(path:string)=> {
  const hash=createHash("sha256");let bytes=0;
  for await(const chunk of createReadStream(path)){hash.update(chunk);bytes+=chunk.length;}
  return {bytes,sha256:hash.digest("hex")};
};
if(values.bundle) {
  if(!values.scorer)throw new Error("--bundle requires a fitted --scorer");
  if(resolve(values.out)===resolve(values.model)||resolve(values.out)===resolve(values.extractor))
    throw new Error("Bundle output must differ from its inputs");
  const manifest=json(resolve(values.model,"manifest.json"));
  const fitted=validateSegmentModel(json(values.scorer),manifest);
  const meta=json(resolve(values.extractor,"segment-units-metadata.json"));
  if(meta.primarySha256!==manifest.sha256||meta.featureOutput!==fitted.featureOutput||meta.width!==fitted.width)
    throw new Error("Fitted scorer differs from verified extractor");
  const primary=await fingerprint(resolve(values.model,manifest.file));
  if(primary.sha256!==manifest.sha256)throw new Error("Primary checksum mismatch");
  mkdirSync(values.out,{recursive:true});
  for(const file of ["manifest.json",manifest.file])copyFileSync(resolve(values.model,file),resolve(values.out,file));
  // Include and hash the already shipped teacher so this bundle retains the existing insertion arm.
  for(const asset of MODEL_REGISTRY.auxiliary??[]) for(const entry of [asset,...asset.externalData]) {
    const source=resolve(values["auxiliary-dir"]??values.model,entry.file),actual=await fingerprint(source);
    if(actual.sha256!==entry.sha256||actual.bytes!==entry.bytes)throw new Error("Existing auxiliary checksum mismatch");
    copyFileSync(source,resolve(values.out,entry.file));
  }
  copyFileSync(values.scorer,resolve(values.out,"segment-units-model.json"));
  copyFileSync(resolve(values.extractor,"segment-units.onnx"),resolve(values.out,"segment-units.onnx"));
  const files=[];
  for(const file of ["segment-units-model.json","segment-units.onnx"])files.push({file,...await fingerprint(resolve(values.out,file))});
  writeFileSync(resolve(values.out,"segment-units-asset.json"),JSON.stringify({schema:1,files}));
  const verified=await loadNodeModel(resolve(values.out));
  if(!verified.segmentUnits||!verified.insertionTeacher)throw new Error("Incomplete listener bundle");
  await verified.backend.release();
  console.log(JSON.stringify({bundle:resolve(values.out),primarySha256:manifest.sha256,verified:true}));
  process.exit(0);
}
const asset=(name:string)=>json(new URL(`../../../core/assets/${name}`,import.meta.url).pathname);
const index=loadReferenceIndex(asset("ref-v2.json")),phrases=asset("phrases-v1.json");
const audio=(path:string)=> {
  const wav=decodeWav(readFileSync(path));if(wav.sampleRate===16000)return wav.samples;
  const resampler=new Resampler(wav.sampleRate,16000),body=resampler.push(wav.samples),tail=resampler.flush();
  const samples=new Float32Array(body.length+tail.length);samples.set(body);samples.set(tail,body.length);return samples;
};
const primary=await loadNodeModel(resolve(values.model));
const meta=json(resolve(values.extractor,"segment-units-metadata.json")) as SegmentMetadata;
if(meta.primarySha256!==primary.manifest.sha256)throw new Error("Extractor differs from primary graph");
const extractor=await ortBackend(ort as unknown as OrtModule,resolve(values.extractor,"segment-units.onnx"),"onnxruntime-node",{
  executionProviders:["cpu"],intraOpNumThreads:1});
const pack=(rows:number[][])=>Buffer.from(Float32Array.from(rows.flat()).buffer).toString("base64");
try {
  if(values.labels) {
    const excluded=new Set<string>(values["excluded-speakers"]?json(values["excluded-speakers"]):[]);
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for(const r of labels)if(r.split!=="train"&&r.speaker)excluded.add(r.speaker);
    const eligible=labels.filter(r=>eligibleSegmentTake(r,excluded));
    const shard=Number(process.env.LAB_SHARD??0),shards=Number(process.env.LAB_SHARDS??1);
    writeFileSync(values.out,"");let count=0,canonical=0;
    for(let i=0;i<eligible.length;i++) {
      if(i%shards!==shard)continue;
      const r=eligible[i];if(!r.wav||!r.ayahs?.length)throw new Error("Training take lacks WAV or ayahs");
      const words=(r.ayahs as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
      if(!words.length)throw new Error("Missing canonical forced-control pronunciation");
      const pronunciation=words.map(w=>index.phonemes.slice(w.start,w.end)).join("");
      const rows=await segmentObservations(extractor,primary.manifest,meta,audio(resolve(dirname(values.labels),r.wav)));
      const isCanonical=canonicalSegmentTake(r);canonical+=Number(isCanonical);
      appendFileSync(values.out,JSON.stringify({source:"help-train",split:"train",speaker:r.speaker,id:r.id,canonical:isCanonical,
        pronunciation,width:meta.width,frames:rows.features.length,featuresBase64:pack(rows.features),log_probs:rows.log_probs})+"\n");
      if(++count%20===0)console.log(JSON.stringify({cached:count,canonical}));
    }
    console.log(JSON.stringify({eligible:eligible.length,cached:count,canonical,shard,shards,excluded:[...excluded].sort()}));
  } else {
    if(!values.audio||!values.evidence)throw new Error("--audio and --evidence are required");
    const raw=json(values.evidence),rows=await segmentObservations(extractor,primary.manifest,meta,audio(values.audio));
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const candidates=new Map<string,SequenceEvidenceCandidate>();
    const fitted:SegmentModel|undefined=values.scorer?validateSegmentModel(json(values.scorer),primary.manifest):undefined;
    if(fitted&&(fitted.width!==meta.width||fitted.featureOutput!==meta.featureOutput))throw new Error("Diagnostic scorer differs from extractor");
    const comparisons:unknown[]=[];
    const replay=(review:boolean)=> {
      const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000});
      const checker=createChecker(index,{phrases,sequenceEvidence(c) {
        if(!review){candidates.set(JSON.stringify(c),c);return true;}
        if(!fitted)return true;
        const result=segmentComparison(fitted,rows.features.slice(c.fromFrame,c.toFrame),c);
        comparisons.push({candidate:c,comparison:result});return !result||!segmentReversal(result);
      }});
      const events=[];
      for(const e of insertionCells(chunks)){const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t);}
      const t=tracker.end();events.push(...t);checker.end(t);return {events,mistakes:checker.mistakes()};
    };
    const before=replay(false),after=replay(true);
    if(JSON.stringify(before.events)!==JSON.stringify(after.events))throw new Error("Changed tracker events");
    if(JSON.stringify(before.mistakes.filter(e=>e.errorClass==="added_word"))!==JSON.stringify(after.mistakes.filter(e=>e.errorClass==="added_word")))throw new Error("Changed insertion decisions");
    writeFileSync(values.out,JSON.stringify({metadata:meta,width:meta.width,frames:rows.features.length,featuresBase64:pack(rows.features),
      candidates:[...candidates.values()],comparisons,before:before.mistakes,after:after.mistakes}));
    console.log(JSON.stringify({frames:rows.features.length,candidates:candidates.size,comparisons:comparisons.length}));
  }
} finally {await extractor.release();await primary.backend.release();}
