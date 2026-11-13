/** Generic exact-window probe; no labels, session IDs or oracle positions. */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "../packages/core/src/index.js";
import type { EngineEvent, MovingZeroEvidence, PhonemeEvidence, SequenceEvidenceCandidate } from "../packages/core/src/index.js";
import { addMovingZeroEvidence, validateMovingZeroModel } from "../packages/sdk/src/moving-spectral-zero.js";
import { insertionCells } from "../packages/sdk/src/seam-insertion.js";
import { decodeWav } from "../packages/sdk/src/wav.js";
import { Resampler } from "../packages/sdk/src/audio.js";

const {values}=parseArgs({options:{audio:{type:"string"},evidence:{type:"string"},model:{type:"string"},out:{type:"string"}}});
if(!values.audio||!values.evidence||!values.model||!values.out) throw new Error("--audio, --evidence, --model and --out required");
const reference=loadReferenceIndex(JSON.parse(readFileSync(new URL("../packages/core/assets/ref-v2.json",import.meta.url),"utf8")));
const phrases=JSON.parse(readFileSync(new URL("../packages/core/assets/phrases-v1.json",import.meta.url),"utf8"));
const model=validateMovingZeroModel(JSON.parse(readFileSync(values.model,"utf8")),reference.inventoryId,[...reference.tokens].join(""));
const saved=JSON.parse(readFileSync(values.evidence,"utf8"));
const chunks:PhonemeEvidence[]=Array.isArray(saved)?saved:[{...saved,decodedTo:saved.nFrames,
  posteriors:saved.posteriors?{...saved.posteriors,startFrame:0,nFrames:saved.posteriors.ids.length/saved.posteriors.k}:undefined}];
if(chunks.some(c=>c.inventory!==model.inventory||c.frameMs!==model.frameMs||!c.posteriors)) throw new Error("Probe needs complete SDK evidence with matching clock and inventory");
const wav=decodeWav(readFileSync(values.audio));
let audio=wav.samples;
if(wav.sampleRate!==16000) {
  const r=new Resampler(wav.sampleRate,16000),body=r.push(audio),tail=r.flush();
  audio=new Float32Array(body.length+tail.length);audio.set(body);audio.set(tail,body.length);
}
const comparisons:{candidate:SequenceEvidenceCandidate;comparison:MovingZeroEvidence|null}[]=[];
const enriched=addMovingZeroEvidence(model,audio,chunks,reference,phrases,{},"free",(comparison,candidate)=>comparisons.push({candidate,comparison}));
function replay(evidence:PhonemeEvidence[]) {
  const tracker=createTracker(reference,{mode:"free",phrases,acquireMs:1000}),checker=createChecker(reference,{phrases});
  const positions:EngineEvent[]=[],events:EngineEvent[]=[];
  for(const c of insertionCells(evidence)) {const p=tracker.push(trackerInput(c));positions.push(...p);events.push(...checker.push(c,p));}
  const p=tracker.end();positions.push(...p);events.push(...checker.end(p));
  return {positions,events,mistakes:checker.mistakes()};
}
const baseline=replay(chunks),changed=replay(enriched);
if(JSON.stringify(baseline.positions)!==JSON.stringify(changed.positions)) throw new Error("Side evidence changed tracker events");
if(baseline.events.length!==changed.events.length||baseline.events.some((e,i)=>JSON.stringify(e)!==JSON.stringify(changed.events[i])&&
  !(e.type==="checker"&&e.errorClass==="wrong_word"&&e.state==="confirmed"&&JSON.stringify({...e,state:"uncertain"})===JSON.stringify(changed.events[i]))))
  throw new Error("Side evidence changed a decision outside the allowed wrong-word veto");
writeFileSync(values.out,JSON.stringify({comparisons,baseline,changed},null,2));
console.log(JSON.stringify({comparisons:comparisons.length,certificates:enriched.flatMap(c=>c.phonemes.flatMap(p=>p.movingZeroVetoes??[])).length,
  warningsBefore:baseline.mistakes.length,warningsAfter:changed.mistakes.length}));
