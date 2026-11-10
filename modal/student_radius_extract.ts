/** Training-speaker engine accusation mining and native-state acoustic-radius measurements.
 * Run on Modal with tsx; output is analysis/training data, never benchmark evidence. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "@tilawa/core";
import { loadNodeModel, decodeWav, createEngine, Resampler } from "../packages/sdk/src/node.js";
import { estimateRadius, radiusCandidates, RADIUS_SEARCH } from "../packages/sdk/src/acoustic-radius.js";

const {values}=parseArgs({options:{"help-dir":{type:"string"},model:{type:"string"},out:{type:"string"},
  "excluded-speakers":{type:"string"}}});
if (!values["help-dir"] || !values.model || !values.out || !values["excluded-speakers"]) throw new Error("--help-dir, --model, --out and --excluded-speakers required");
const root=resolve(values["help-dir"]);
if (!root.endsWith(`${sep}help-train`)) throw new Error("This miner accepts only help-train");
const labels=readFileSync(resolve(root,"labels.jsonl"),"utf8").trim().split("\n").map(s=>JSON.parse(s));
const excluded=new Set<string>(JSON.parse(readFileSync(values["excluded-speakers"],"utf8")));
for (const l of labels) if (l.split!=="train") excluded.add(l.speaker);
const asset=(file:string)=>JSON.parse(readFileSync(new URL(`../packages/core/assets/${file}`,import.meta.url),"utf8"));
const index=loadReferenceIndex(asset("ref-v2.json")), phrases=asset("phrases-v1.json");
const model=await loadNodeModel(resolve(values.model));
if (model.acousticRadius) throw new Error("Calibration must mine the original listener, without a radius gate");
const shard=Number(process.env.LAB_SHARD??0), shards=Number(process.env.LAB_SHARDS??1);
const recordings=labels.filter(l=>l.split==="train" && !excluded.has(l.speaker));
const rows=[];
for (let i=shard;i<recordings.length;i+=shards) {
  const l=recordings[i], path=resolve(root,l.wav);
  if (!path.startsWith(root+sep)) throw new Error("Training WAV escapes help root");
  const wav=decodeWav(readFileSync(path));
  let audio=wav.samples;
  if (wav.sampleRate!==16000) {
    const rs=new Resampler(wav.sampleRate,16000), body=rs.push(audio), tail=rs.flush();
    audio=new Float32Array(body.length+tail.length);audio.set(body);audio.set(tail,body.length);
  }
  const session=createEngine({reference:index,phrases,model,recordEvidence:true}).startFreeRecitation();
  for (let s=0;s<audio.length;s+=5120) await session.pushAudio(audio.subarray(s,s+5120),16000);
  await session.end();
  const chunks=session.evidence();
  for (const c of radiusCandidates(chunks,index,phrases)) {
    const positive=(l.events??[]).some((e:any)=>e.errorClass==="wrong_word" && e.state==="confirmed" &&
      e.location.surah===c.word.surah && e.location.ayah===c.word.ayah && e.location.word===c.word.word);
    // Scripted non-swap recordings are preservation controls, not verified clean transcriptions.
    if (!positive && (l.kind!=="none" || l.events?.length || l.non_errors?.length)) continue;
    const result=await estimateRadius(model,audio,chunks,c);
    rows.push({speaker:l.speaker,recording:l.id,split:"train",positive,candidate:c,result,
      primarySha256:model.manifest.sha256,inventory:model.manifest.inventory,search:RADIUS_SEARCH});
  }
  console.log(JSON.stringify({recording:i+1,total:recordings.length,accusations:rows.length}));
}
writeFileSync(values.out,JSON.stringify({schema:1,shard,shards,recordings:recordings.length,
  excludedSpeakers:[...excluded].sort(),rows}));
await model.backend.release();
