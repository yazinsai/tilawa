/** Training-only extraction or read-only replay of engine-requested sequence comparisons. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, PosteriorFrames, trackerInput } from "@tilawa/core";
import type { BlankCauseEvidence, PhonemeEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { addBlankCauseEvidence, blankFeatures, blankObservation, validateBlankCauseModel } from "../blank-cause.js";
import { Resampler } from "../audio.js";
import { createListener } from "../listener.js";
import { loadNodeModel } from "../node.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{labels:{type:"string"},"excluded-speakers":{type:"string"},out:{type:"string"},
  model:{type:"string"},cache:{type:"string"},emissions:{type:"string"},audio:{type:"string"},evidence:{type:"string"}}});
if (!values.out) throw new Error("--out is required");
const json=(path:string)=>JSON.parse(readFileSync(path,"utf8"));
const asset=(name:string)=>json(new URL(`../../../core/assets/${name}`,import.meta.url).pathname);
const index=loadReferenceIndex(asset("ref-v2.json")),phrases=asset("phrases-v1.json"),tokens=[...index.tokens].join("");
function audio(path:string):Float32Array {
  const wav=decodeWav(readFileSync(path));
  if (wav.sampleRate===16000) return wav.samples;
  const r=new Resampler(wav.sampleRate,16000),first=r.push(wav.samples),tail=r.flush(),samples=new Float32Array(first.length+tail.length);
  samples.set(first);samples.set(tail,first.length);return samples;
}
function chunks(raw:any):PhonemeEvidence[] {
  return Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{...raw.posteriors,
    startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
}
if (values.labels) {
  if (!values["excluded-speakers"]||!values.model||!values.cache) throw new Error("--excluded-speakers, --model and --cache are required for training extraction");
  const excluded=new Set<string>(json(values["excluded-speakers"]));
  const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
  for (const r of labels) if (r.split!=="train") excluded.add(r.speaker);
  const model=await loadNodeModel(values.model);
  if (model.blankCause) throw new Error("Training extraction must use the unchanged primary listener");
  if (model.manifest.inventory!==index.inventoryId||model.manifest.tokens!==tokens||model.manifest.frameMs!==40||model.manifest.sampleRate!==16000)
    throw new Error("Primary model inventory or clock differs");
  const rows=[];mkdirSync(values.cache,{recursive:true});
  try {
    for (const r of labels) {
      if (r.kind!=="none"||r.split!=="train"||excluded.has(r.speaker)) continue;
      if (!r.speaker||!r.wav||!r.id||r.events?.length||r.non_errors?.length) throw new Error("None-labelled training input contains missing metadata, a mistake or repeat");
      const samples=audio(resolve(dirname(values.labels),r.wav));
      const cache=join(values.cache,encodeURIComponent(r.id)+".json");
      const audioSha256=createHash("sha256").update(readFileSync(resolve(dirname(values.labels),r.wav))).digest("hex");
      let evidence:PhonemeEvidence[];
      if (existsSync(cache)) {
        const saved=json(cache);
        if (saved.audioSha256!==audioSha256||saved.modelSha256!==model.manifest.sha256) throw new Error("Cached primary observations belong to different audio or weights");
        evidence=chunks(saved.evidence);
      }
      else {
        const listener=createListener(model,{posteriors:true}),decoded:PhonemeEvidence[]=[];
        const e=await listener.push(samples,16000);if (e) decoded.push(e);
        const end=await listener.end();if (end) decoded.push(end);
        evidence=decoded;writeFileSync(cache,JSON.stringify({audioSha256,modelSha256:model.manifest.sha256,evidence}));
      }
      const frames=new PosteriorFrames(index.tokens.size);
      for (const e of evidence) {
        if (!e.posteriors||e.frameMs!==40||e.inventory!==index.inventoryId) throw new Error("Invalid cached primary posterior rows");
        frames.append(e.posteriors);
      }
      const f=blankFeatures(samples);
      if (frames.startFrame!==0||frames.endFrame!==f.length) throw new Error("Acoustic observations and primary subsampling disagree");
      rows.push({source:"help-train",kind:"none",split:"train",excluded:false,speaker:r.speaker,id:r.id,audioSha256,modelSha256:model.manifest.sha256,
        features:f.map((x,i)=>blankObservation(x,frames.row(i)))});
      console.log(JSON.stringify({recording:r.id,frames:f.length,done:rows.length}));
    }
  } finally {await model.backend.release();}
  writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
  console.log(JSON.stringify({recordings:rows.length,speakers:new Set(rows.map(r=>r.speaker)).size,inventory:index.inventoryId,tokens}));
} else {
  if (!values.emissions||!values.audio||!values.evidence) throw new Error("--emissions, --audio and --evidence are required for a comparison probe");
  const model=validateBlankCauseModel(json(values.emissions),index.inventoryId,tokens),evidence=chunks(json(values.evidence));
  const comparisons:{candidate:SequenceEvidenceCandidate;comparison:BlankCauseEvidence|null}[]=[];
  const enriched=addBlankCauseEvidence(model,audio(values.audio),evidence,index,phrases,{},"free",(comparison,candidate)=>comparisons.push({candidate,comparison}));
  const replay=(chunks:PhonemeEvidence[])=> {
    const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases}),events=[];
    for (const e of insertionCells(chunks)) {const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t);}
    const t=tracker.end();events.push(...t);checker.end(t);return {events,mistakes:checker.mistakes()};
  };
  const before=replay(evidence),after=replay(enriched);
  if (JSON.stringify(before.events)!==JSON.stringify(after.events)) throw new Error("Blank-cause metadata changed tracker events");
  const insertions=(r:ReturnType<typeof replay>)=>r.mistakes.filter(e=>e.errorClass==="added_word");
  if (JSON.stringify(insertions(before))!==JSON.stringify(insertions(after))) throw new Error("Blank-cause metadata changed insertion decisions");
  const primary=(e:PhonemeEvidence[])=>e.map(c=>({...c,phonemes:c.phonemes.map(({blankCauseVetoes,...p})=>p)}));
  if (JSON.stringify(primary(evidence))!==JSON.stringify(primary(enriched))) throw new Error("Blank-cause metadata changed primary evidence");
  writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
  console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.blankCauseVetoes??[])).length}));
}
