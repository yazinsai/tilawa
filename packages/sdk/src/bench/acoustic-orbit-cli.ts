/** Prepare eligible frozen training observations or inspect engine-requested comparisons.
 * Labels never choose inference windows, pronunciations or acceptance rules. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { addAcousticOrbitEvidence, eligibleAcousticOrbitTake } from "../acoustic-orbit.js";
import { loadNodeModel } from "../node.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{model:{type:"string"},labels:{type:"string"},
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
    if (!primary.acousticOrbit) throw new Error("No declared fitted scorer");
    const m=primary.acousticOrbit.model;
    writeFileSync(values.out,JSON.stringify({primarySha256:primary.manifest.sha256,
      dimensions:m.dimensions,trainingRecordings:m.trainingRecordings,verified:true},null,2));
    console.log("Complete primary and orbit assets loaded and validated");
  } else if (values.labels) {
    if (!values["excluded-speakers"]) throw new Error("--excluded-speakers is required");
    const excluded=new Set<string>(json(values["excluded-speakers"]));
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for (const r of labels) if (r.split!=="train") excluded.add(r.speaker);
    const transcripts: Record<string,{pronunciation:string;controlPronunciations:string[]}> = {};
      const lexicon=[...new Set(index.words.filter(w=>w.end>w.start).map(w=>index.phonemes.slice(w.start,w.end)))].sort();
      for (const r of labels) {
        if (!eligibleAcousticOrbitTake(r,excluded)) continue;
        if (!r.speaker||!r.wav||!r.ayahs?.length) throw new Error("Training label lacks speaker, WAV or ayahs");
        const words=(r.ayahs as string[]).flatMap(a=>index.words.filter(w=>`${w.surah}:${w.ayah}`===a&&w.end>w.start));
        if (!words.length) throw new Error("No canonical training pronunciation");
        const pronunciation=words.map(w=>index.phonemes.slice(w.start,w.end)).join("");
        // Genuine recorded canonical speech is the spoken rival to a counterfactual
        // written reference. These controls alter text only, never audio or fitting labels.
        // Choose lexical single-phone neighbours by a fixed rule, without practice input.
        const controls:string[]=[];
        let offset=0;
        for (const w of words) {
          const spoken=index.phonemes.slice(w.start,w.end);
          const alternative=lexicon.find(p=>p.length===spoken.length &&
            [...p].filter((v,j)=>v!==spoken[j]).length===1);
          if (alternative && controls.length<3) controls.push(pronunciation.slice(0,offset)+alternative+pronunciation.slice(offset+spoken.length));
          offset+=spoken.length;
        }
        transcripts[r.id]={pronunciation,controlPronunciations:controls};
      }
    writeFileSync(values.out,JSON.stringify({transcripts,excluded:[...excluded]}));
    console.log(JSON.stringify({recordings:Object.keys(transcripts).length}));
  } else {
    if (!primary.acousticOrbit||!values.audio||!values.evidence) throw new Error("A fitted bundle, --audio and --evidence are required");
    const raw=json(values.evidence);
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const comparisons:unknown[]=[];
    const enriched=await addAcousticOrbitEvidence(primary.acousticOrbit,primary.manifest,audio(values.audio),chunks,index,phrases,{},"free",
      (comparison,candidate)=>comparisons.push({candidate,comparison}));
    const replay=(evidence:PhonemeEvidence[])=> {
      const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases});
      const events=[];
      for (const e of insertionCells(evidence)) { const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t); }
      const t=tracker.end();events.push(...t);checker.end(t);
      return {events,mistakes:checker.mistakes()};
    };
    const before=replay(chunks),after=replay(enriched);
    if (JSON.stringify(before.events)!==JSON.stringify(after.events)) throw new Error("Orbit-phone metadata changed tracker events");
    if (JSON.stringify(before.mistakes.filter(e=>e.errorClass==="added_word"))!==
      JSON.stringify(after.mistakes.filter(e=>e.errorClass==="added_word"))) throw new Error("Orbit-phone metadata changed insertion decisions");
    writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
    console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.acousticOrbitVetoes??[])).length}));
  }
} finally { await primary.backend.release(); }
