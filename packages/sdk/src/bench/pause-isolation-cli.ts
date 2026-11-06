/** Fails before scoring if a side ear changes primary output or folded teacher
 * judgments. Inputs are ordinary model bundles and WAVs, with no benchmark keys. */
import { createHash } from "node:crypto";
import { readFileSync,readdirSync,writeFileSync } from "node:fs";
import { join,resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { loadNodeModel } from "../node.js";
import { createEngine } from "../session.js";
import { decodeWav } from "../wav.js";
const {values}=parseArgs({options:{model:{type:"string"},primary:{type:"string"},wavs:{type:"string"},out:{type:"string"}}});
if(!values.model||!values.primary||!values.wavs||!values.out)throw new Error("--model, --primary, --wavs, --out required");
const bundle=resolve(values.model),original=resolve(values.primary);
const manifestBytes=readFileSync(join(bundle,"manifest.json"));
if(!manifestBytes.equals(readFileSync(join(original,"manifest.json"))))throw new Error("Primary manifest bytes changed");
const model=await loadNodeModel(bundle);
if(!model.pauseModel||!model.insertionTeacher)throw new Error("Both declared side ears must be installed for isolation verification");
const primaryHash=createHash("sha256").update(readFileSync(join(bundle,model.manifest.file))).digest("hex");
if(primaryHash!==model.manifest.sha256)throw new Error("Primary ONNX bytes changed");
const asset=(name:string)=>JSON.parse(readFileSync(new URL(`../../../core/assets/${name}`,import.meta.url),"utf8"));
const reference=loadReferenceIndex(asset("ref-v2.json")),phrases=asset("phrases-v1.json");
const files=readdirSync(values.wavs).filter(f=>f.endsWith(".wav")).sort();
const shard=Number(process.env.LAB_SHARD??0),shards=Number(process.env.LAB_SHARDS??1);
const strip=(chunks:PhonemeEvidence[])=>chunks.map(e=>({...e,phonemes:e.phonemes.map(({pauseMaskEvidence:_,...p})=>p)}));
const rows=[];
for(const [i,name] of files.entries()){
  if(i%shards!==shard)continue;
  const wav=decodeWav(readFileSync(join(values.wavs,name)));
  const run=async(side:boolean)=>{
    const session=createEngine({reference,phrases,model:side?model:{...model,pauseModel:undefined},recordEvidence:true}).startFreeRecitation();
    const events=[];
    for(let a=0;a<wav.samples.length;a+=Math.round(wav.sampleRate*.02))events.push(...await session.pushAudio(wav.samples.subarray(a,a+Math.round(wav.sampleRate*.02)),wav.sampleRate));
    events.push(...await session.end());
    return {evidence:session.evidence(),mistakes:session.mistakes(),positions:events.filter(e=>e.type==="position")};
  };
  const baseline=await run(false),candidate=await run(true);
  const primary=JSON.stringify(baseline.evidence),actual=JSON.stringify(strip(candidate.evidence));
  if(primary!==actual)throw new Error(`Primary decode or folded teacher certificates changed: ${name}`);
  if(JSON.stringify(baseline.positions)!==JSON.stringify(candidate.positions))throw new Error(`Primary tracker changed: ${name}`);
  const retained=candidate.mistakes.filter(e=>!e.id.startsWith("pause-mask/"));
  if(JSON.stringify(baseline.mistakes)!==JSON.stringify(retained))throw new Error(`Primary mistakes or folded teacher wins changed: ${name}`);
  rows.push({name,primarySha256:createHash("sha256").update(primary).digest("hex"),primaryBytes:Buffer.byteLength(primary),
    baselineMistakes:baseline.mistakes,candidateMistakes:candidate.mistakes,
    scopes:candidate.evidence.flatMap(e=>e.phonemes.flatMap(p=>p.pauseMaskEvidence??[])).map(s=>({from:s.fromFrame,to:s.toFrame,phones:s.phonemes.map(p=>p.p).join(""),certificates:s.seamInsertions}))});
  console.log(JSON.stringify({name,verified:true,newAdditions:candidate.mistakes.length-baseline.mistakes.length}));
}
if(!rows.length)throw new Error("No WAVs selected for isolation verification");
writeFileSync(values.out,JSON.stringify({verified:true,manifestSha256:createHash("sha256").update(manifestBytes).digest("hex"),primaryHash,shard,shards,rows}));
await model.backend.release();
