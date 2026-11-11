/** Training-only SDK feature preparation. Run with tsx from the repository root.
 * The supplied help split plus an explicit exclusion list define eligible speakers.
 * Scripts are never converted into performed transcripts; only none-labelled takes
 * supply canonical pronunciations. No practice data or warning windows are inputs.
 */
import { readFileSync,writeFileSync,mkdirSync,appendFileSync } from "node:fs";
import { resolve,dirname } from "node:path";
import { parseArgs } from "node:util";
import { Resampler } from "../packages/sdk/src/audio.js";
import { decodeWav } from "../packages/sdk/src/wav.js";
import { regularityFeatures } from "../packages/sdk/src/regularity.js";

const {values}=parseArgs({options:{
  "help-dir":{type:"string"},"excluded-speakers":{type:"string"},out:{type:"string"},
}});
if (!values["help-dir"]||!values["excluded-speakers"]||!values.out) throw new Error("--help-dir, --excluded-speakers and --out are required");
interface Label { id: string; speaker: string; kind: string; split: string; wav: string; ayahs: string[];
  events: unknown[]; non_errors: unknown[] }
const root=resolve(values["help-dir"]);
const labels=readFileSync(resolve(root,"labels.jsonl"),"utf8").trim().split("\n").map(l=>JSON.parse(l) as Label);
const excluded=new Set<string>(JSON.parse(readFileSync(values["excluded-speakers"],"utf8")));
// Exclude the whole speaker whenever any recording is outside the training partition.
for (const l of labels) if (l.split!=="train") excluded.add(l.speaker);
const reference=JSON.parse(readFileSync(new URL("../packages/core/assets/ref-v2.json",import.meta.url),"utf8"));
const ayahs=new Map<string,string>(reference.ayahs.map((a: {s:number;a:number;p:string})=>[`${a.s}:${a.a}`,a.p]));
const out=resolve(values.out);mkdirSync(dirname(out),{recursive:true});writeFileSync(out,"");
let count=0;
for (const l of labels) {
  if (l.kind!=="none"||l.split!=="train"||excluded.has(l.speaker)) continue;
  if (!l.speaker||!l.ayahs.length||l.events.length||l.non_errors.length) throw new Error("None-labelled training take carries deviations or lacks provenance");
  const pronunciation=l.ayahs.map(a=>{const p=ayahs.get(a);if (!p) throw new Error(`Missing canonical pronunciation ${a}`);return p;}).join("");
  const path=resolve(root,l.wav);
  if (!path.startsWith(root+"/")) throw new Error("Training audio path leaves help-train");
  const wav=decodeWav(readFileSync(path));
  let audio=wav.samples;
  if (wav.sampleRate!==16000) {
    const resampler=new Resampler(wav.sampleRate,16000),head=resampler.push(audio),tail=resampler.flush();
    audio=new Float32Array(head.length+tail.length);audio.set(head);audio.set(tail,head.length);
  }
  const features={fine:regularityFeatures(audio,[2,4,8,16,32,64,128]),wide:regularityFeatures(audio,[4,8,16,32,64,128,256])};
  appendFileSync(out,JSON.stringify({id:l.id,speaker:l.speaker,kind:"none",split:"train",excluded:false,pronunciation,features})+"\n");
  count++;
}
if (!count) throw new Error("No eligible none-labelled training recordings");
writeFileSync(out+".exclusions.json",JSON.stringify([...excluded].sort()));
console.log(JSON.stringify({recordings:count,excludedSpeakers:excluded.size,out,inventory:reference.inventory}));
