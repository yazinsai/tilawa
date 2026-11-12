/** Prepare eligible frozen training observations or inspect engine-requested comparisons.
 * Labels never choose inference windows, pronunciations or acceptance rules. */
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { addFingerprintEvidence, fingerprintObservations, eligibleFingerprintTake, fingerprintMetadata, fingerprintComparison } from "../response-fingerprint.js";
import { loadNodeModel } from "../node.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{model:{type:"string"},metadata:{type:"string"},deadline:{type:"string"},verify:{type:"boolean"},labels:{type:"string"},
  "excluded-speakers":{type:"string"},out:{type:"string"},audio:{type:"string"},evidence:{type:"string"}}});
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
  if (values.labels) {
    if (!values.metadata||!values.deadline||!values["excluded-speakers"]) throw new Error("--metadata, --deadline and --excluded-speakers are required");
    const deadline=Number(values.deadline);
    if (!Number.isFinite(deadline)||deadline<=Date.now()||deadline>Date.now()+86400000)
      throw new Error("Extraction requires a shared deadline no more than one day away");
    const budgetFile=values.metadata+".budget.json";
    if (existsSync(budgetFile) && json(budgetFile).deadlineMs!==deadline)
      throw new Error("The extraction deadline is frozen; a resume cannot restart the budget");
    writeFileSync(budgetFile,JSON.stringify({deadlineMs:deadline}));
    const excluded=new Set<string>(json(values["excluded-speakers"]));
    const labels=readFileSync(values.labels,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
    for (const r of labels) if (r.split!=="train" && r.speaker) excluded.add(r.speaker);
    const metadata=fingerprintMetadata(primary.manifest);
    if (existsSync(values.metadata) && JSON.stringify(json(values.metadata))!==JSON.stringify(metadata))
      throw new Error("Cached extraction metadata differ from the frozen graph");
    writeFileSync(values.metadata,JSON.stringify(metadata));
    const existing=existsSync(values.out)?readFileSync(values.out,"utf8").split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line)):[];
    if (existing.some(r=>r.primarySha256!==metadata.primarySha256 || r.deadlineMs!==deadline))
      throw new Error("Cached recordings differ from this frozen extraction");
    const completed=new Set(existing.map(r=>r.id));
    const shard=Number(process.env.LAB_SHARD??0), shards=Number(process.env.LAB_SHARDS??1);
    if (!Number.isInteger(shard)||!Number.isInteger(shards)||shards<1||shard<0||shard>=shards) throw new Error("Invalid extraction shard");
    const rows:{speaker:string}[]=[];
    const lexicon=[...new Set(index.words.filter(w=>w.end>w.start).map(w=>index.phonemes.slice(w.start,w.end)))].sort();
    for (const [ri,r] of labels.entries()) {
      if (ri%shards!==shard) continue;
      if (!eligibleFingerprintTake(r,excluded)) continue;
      if (!r.id||!r.speaker||!r.wav||!r.ayahs?.length) throw new Error("Training label lacks speaker, WAV or ayahs");
      if (completed.has(r.id)) continue;
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
      const features=await fingerprintObservations(primary.backend,primary.manifest,metadata,audio(resolve(dirname(values.labels),r.wav)),undefined,deadline);
      const row={source:"help-train",kind:"none",split:"train",excluded:false,speaker:r.speaker,id:r.id,
        pronunciation,controlPronunciations:controls,
        response:features.response,control:features.control,primarySha256:metadata.primarySha256,
        deadlineMs:deadline,log_probs:features.logProbs};
      rows.push({speaker:r.speaker});
      completed.add(r.id);
      appendFileSync(values.out,JSON.stringify(row)+"\n");
    }
    console.log(JSON.stringify({recordings:rows.length,speakers:new Set(rows.map(r=>r.speaker)).size}));
  } else if (values.verify) {
    if (!primary.responseFingerprint) throw new Error("A complete fitted bundle is required");
    // Exercise the actual declared graph and frozen extraction, including recurrent
    // state and all symmetric interventions, before declaring a scoring bundle.
    const signal=Float32Array.from({length:16000},(_,i)=>.02*Math.sin(i*.17));
    const rows=await fingerprintObservations(primary.responseFingerprint.backend,primary.manifest,
      fingerprintMetadata(primary.manifest),signal,f=>f===0);
    if(rows.response[0]?.length!==primary.responseFingerprint.model.dimensions) throw new Error("Scorer output dimension differs");
    // Also exercise Gaussian scoring and pronunciation completion on real extracted rows.
    const fitted=primary.responseFingerprint.model;
    const phone=fitted.response.phones.findIndex((e,p)=>p>0 && e.count>0 && fitted.control.phones[p].count>0);
    if(phone<1) throw new Error("Fitted arms have no supported pronunciation");
    const token=primary.manifest.tokens[phone-1];
    if(!fingerprintComparison(fitted,rows.response.slice(0,1),rows.control.slice(0,1),{
        word:{surah:0,ayah:0,word:0},written:token,rival:token,fromFrame:0,toFrame:1,
      })) throw new Error("Declared emissions fail finite pronunciation scoring");
    if(primary.insertionTeacher) await primary.insertionTeacher.decode(signal,5);
    writeFileSync(values.out,JSON.stringify({verified:true,primarySha256:primary.manifest.sha256,
      dimensions:rows.response[0].length,frames:rows.logProbs.length,insertionTeacher:!!primary.insertionTeacher}));
  } else {
    if (!primary.responseFingerprint||!values.audio||!values.evidence) throw new Error("A fitted bundle, --audio and --evidence are required");
    const raw=json(values.evidence);
    const chunks:PhonemeEvidence[]=Array.isArray(raw)?raw:[{...raw,posteriors:raw.posteriors&&{
      ...raw.posteriors,startFrame:raw.posteriors.startFrame??0,nFrames:raw.posteriors.nFrames??raw.nFrames}}];
    const comparisons:unknown[]=[];
    const enriched=await addFingerprintEvidence(primary.responseFingerprint,primary.manifest,audio(values.audio),chunks,index,phrases,{},"free",
      (comparison,candidate)=>comparisons.push({candidate,comparison}));
    const replay=(evidence:PhonemeEvidence[])=> {
      const tracker=createTracker(index,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(index,{phrases});
      const events=[];
      for (const e of insertionCells(evidence)) { const t=tracker.push(trackerInput(e));events.push(...t);checker.push(e,t); }
      const t=tracker.end();events.push(...t);checker.end(t);
      return {events,mistakes:checker.mistakes()};
    };
    const before=replay(chunks),after=replay(enriched);
    if (JSON.stringify(before.events)!==JSON.stringify(after.events)) throw new Error("Response metadata changed tracker events");
    if (JSON.stringify(before.mistakes.filter(e=>e.errorClass==="added_word"))!==
      JSON.stringify(after.mistakes.filter(e=>e.errorClass==="added_word"))) throw new Error("Response metadata changed insertion decisions");
    writeFileSync(values.out,JSON.stringify({comparisons,before:before.mistakes,after:after.mistakes},null,2));
    console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(e=>e.phonemes.flatMap(p=>p.responseFingerprintVetoes??[])).length}));
  }
} finally { await primary.backend.release(); }
