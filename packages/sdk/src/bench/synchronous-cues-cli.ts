/** Extract only none-labelled help-train training speakers for the acoustic scorer. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "@tilawa/core";
import { Resampler } from "../audio.js";
import { cueFeatures } from "../synchronous-cues.js";
import { decodeWav } from "../wav.js";

const {values}=parseArgs({options:{help:{type:"string"},out:{type:"string"},"excluded-speakers":{type:"string"}}});
if(!values.help || !values.out || !values["excluded-speakers"]) throw new Error("--help, --out and --excluded-speakers required");
const excluded=new Set<string>(JSON.parse(readFileSync(values["excluded-speakers"],"utf8")));
const index=loadReferenceIndex(JSON.parse(readFileSync(new URL("../../../core/assets/ref-v2.json",import.meta.url),"utf8")));
const labels=readFileSync(resolve(values.help,"labels.jsonl"),"utf8").trim().split("\n").map(s=>JSON.parse(s));
const rows=[];
for(const l of labels) {
  if(l.kind!=="none" || l.split!=="train" || excluded.has(l.speaker)) continue;
  if(typeof l.speaker!=="string" || !l.speaker || !Array.isArray(l.events) || l.events.length ||
    !Array.isArray(l.non_errors) || l.non_errors.length || !Array.isArray(l.ayahs) || !l.ayahs.length)
    throw new Error("None-labelled training record lacks trustworthy provenance");
  const pronunciation=l.ayahs.map((key:string)=> {
    const [surah,ayah]=key.split(":").map(Number);
    const words=index.words.filter(w=>w.surah===surah&&w.ayah===ayah);
    if(!words.length) throw new Error("Training ayah missing from the reference");
    return index.phonemes.slice(words[0].start,words.at(-1)!.end);
  }).join("");
  const wav=decodeWav(readFileSync(resolve(values.help,l.wav)));
  let audio=wav.samples;
  if(wav.sampleRate!==16000) {
    const resampler=new Resampler(wav.sampleRate,16000), part=resampler.push(audio), tail=resampler.flush();
    audio=new Float32Array(part.length+tail.length);audio.set(part);audio.set(tail,part.length);
  }
  rows.push({id:l.id,speaker:l.speaker,kind:l.kind,split:l.split,excluded:false,pronunciation,features:cueFeatures(audio)});
}
if(new Set(rows.map(r=>r.speaker)).size<3) throw new Error("Fewer than three eligible training speakers");
writeFileSync(values.out,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
console.log(JSON.stringify({recordings:rows.length,speakers:new Set(rows.map(r=>r.speaker)).size,inventory:index.inventoryId}));
