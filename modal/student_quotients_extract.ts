/** Training-only SDK observation export. Run with tsx on Modal, not on the Mac. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import { parseArgs } from "node:util";
import { quotientFeatures } from "../packages/sdk/src/spectral-quotients.js";
import { decodeWav } from "../packages/sdk/src/wav.js";
import { Resampler } from "../packages/sdk/src/audio.js";

const { values } = parseArgs({ options: {
  "help-dir": { type:"string" }, out:{type:"string"}, "excluded-speakers":{type:"string"},
} });
if (!values["help-dir"] || !values.out || !values["excluded-speakers"]) throw new Error("--help-dir, --out and --excluded-speakers required");
const root=resolve(values["help-dir"]);
const excluded=new Set<string>(JSON.parse(readFileSync(values["excluded-speakers"],"utf8")));
const labels=readFileSync(resolve(root,"labels.jsonl"),"utf8").trim().split("\n").map(s => JSON.parse(s));
for (const l of labels) if (l.split!=="train") excluded.add(l.speaker);
const reference=JSON.parse(readFileSync(new URL("../packages/core/assets/ref-v2.json",import.meta.url),"utf8"));
const ayahs=new Map<string,{p:string;o:number[];iso:string[]}>(reference.ayahs.map((a: {s:number;a:number;p:string;o:number[];iso:string[]}) => [`${a.s}:${a.a}`,a]));
const rows=[];
for (const l of labels) {
  if (l.kind!=="none" || l.split!=="train" || excluded.has(l.speaker)) continue;
  if (l.events?.length || l.non_errors?.length || l.ayahs?.length!==1) throw new Error("None training take has contradictory labels or is not a single ayah");
  const path=resolve(root,l.wav);
  if (!path.startsWith(root+sep)) throw new Error("Training WAV escapes help root");
  const wav=decodeWav(readFileSync(path));
  let audio=wav.samples;
  if (wav.sampleRate!==16000) {
    const resampler=new Resampler(wav.sampleRate,16000), body=resampler.push(audio), tail=resampler.flush();
    audio=new Float32Array(body.length+tail.length); audio.set(body); audio.set(tail,body.length);
  }
  const ayah=ayahs.get(l.ayahs[0]);
  if (!ayah) throw new Error("Missing canonical training pronunciation");
  const pronunciation=ayah.p.slice(0,ayah.o.at(-1))+ayah.iso.at(-1);
  rows.push({speaker:l.speaker,kind:"none",split:"train",excluded:false,pronunciation,features:quotientFeatures(audio)});
}
writeFileSync(values.out,rows.map(r => JSON.stringify(r)).join("\n")+"\n");
// The trainer checks the same union, including any held-out speakers in label metadata.
writeFileSync(values.out+".excluded.json",JSON.stringify([...excluded].sort()));
console.log(JSON.stringify({recordings:rows.length,speakers:new Set(rows.map(r => r.speaker)).size,excluded:excluded.size}));
