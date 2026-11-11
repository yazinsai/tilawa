/** Training-only native acoustic extraction; no practice input is accepted. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import type { ReferenceAsset } from "@tilawa/core";
import { Resampler } from "../audio.js";
import { reverseFeatures } from "../reverse-predictive.js";
import { decodeWav } from "../wav.js";

const { values } = parseArgs({ options: {
  help: { type: "string" }, out: { type: "string" }, "excluded-speakers": { type: "string" },
} });
if (!values.help || !values.out || !values["excluded-speakers"]) throw new Error("--help, --out and --excluded-speakers required");
const root=resolve(values.help);
const excluded=new Set<string>(JSON.parse(readFileSync(values["excluded-speakers"],"utf8")));
const asset: ReferenceAsset=JSON.parse(readFileSync(new URL("../../../core/assets/ref-v2.json",import.meta.url),"utf8"));
const pronunciations=new Map(asset.ayahs.map(a=>[`${a.s}:${a.a}`,a.p]));
const labels=readFileSync(resolve(root,"labels.jsonl"),"utf8").trim().split("\n").map(s=>JSON.parse(s));
const denied=new Set<string>([...excluded,...labels.filter(r=>r.split!=="train").map(r=>r.speaker)]);
const output: string[]=[];
for (const r of labels) {
  if (r.kind!=="none" || r.split!=="train" || denied.has(r.speaker)) continue;
  if (r.events.length || r.non_errors.length) throw new Error("None-labelled take carries a deviation or repeat");
  const path=resolve(root,r.wav);
  if (!path.startsWith(root+"/")) throw new Error("Audio path escapes training dataset");
  const wav=decodeWav(readFileSync(path));
  let samples=wav.samples;
  if (wav.sampleRate!==16000) {
    const resampler=new Resampler(wav.sampleRate,16000), body=resampler.push(samples), tail=resampler.flush();
    samples=new Float32Array(body.length+tail.length);samples.set(body);samples.set(tail,body.length);
  }
  const pronunciation=r.ayahs.map((a: string)=> {
    const p=pronunciations.get(a);
    if (!p) throw new Error("Unknown canonical training ayah");
    return p;
  }).join("");
  output.push(JSON.stringify({speaker:r.speaker,kind:r.kind,split:r.split,excluded:false,pronunciation,features:reverseFeatures(samples)}));
}
writeFileSync(values.out,output.join("\n")+"\n");
console.log(JSON.stringify({recordings:output.length,excludedSpeakers:[...denied],inventory:asset.inventory.id,tokens:asset.inventory.tokens}));
