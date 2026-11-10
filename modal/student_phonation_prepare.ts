/** Training-only SDK feature extraction. Never accepts practice audio or scripted transcripts. */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadReferenceIndex } from "../packages/core/src/reference.js";
import { phonationFeatures } from "../packages/sdk/src/phonation.js";
import { decodeWav } from "../packages/sdk/src/wav.js";
import { Resampler } from "../packages/sdk/src/audio.js";

/** The importer omits `excluded` on eligible rows and stores a reason on rejected rows. */
interface TrainingLabel {
  speaker: string;
  kind: string;
  split?: string;
  excluded?: boolean | string;
  scored?: boolean;
  events?: unknown[];
  non_errors?: unknown[];
  unsupported?: unknown[];
  id?: string;
  ayahs?: string[];
  wav?: string;
}
export function selectPhonationTrainingLabels(labels: TrainingLabel[]): { rows: TrainingLabel[]; excluded: Set<string> } {
  const excluded=new Set<string>(labels.filter(r=>r.split!=="train").map(r=>r.speaker));
  const rows=labels.filter(r=>r.split==="train" && r.kind==="none" && !r.excluded && r.scored!==false &&
    !excluded.has(r.speaker) && !r.events?.length && !r.non_errors?.length && !r.unsupported?.length);
  return {rows,excluded};
}

function main() {
const args=process.argv.slice(2);
const get=(name:string)=>{const i=args.indexOf(name);if(i<0||!args[i+1])throw new Error(`Required ${name}`);return args[i+1];};
const help=resolve(get("--help-train")), output=resolve(get("--out"));
const index=loadReferenceIndex(JSON.parse(readFileSync(resolve(get("--reference")),"utf8")));
const source=readFileSync(resolve(help,"labels.jsonl"),"utf8");
const labels=source.trim().split("\n").map(s=>JSON.parse(s));
const {rows:eligible,excluded}=selectPhonationTrainingLabels(labels);
const rows=[];
for (const r of eligible) {
  if (!Array.isArray(r.ayahs)||r.ayahs.length!==1||typeof r.speaker!=="string"||typeof r.wav!=="string")
    throw new Error("Invalid none-labelled single-ayah training recording");
  const path=resolve(help,r.wav);
  if (!path.startsWith(help+"/"))throw new Error("Audio path escapes help-train");
  const [surah,ayah]=r.ayahs[0].split(":").map(Number);
  const words=index.words.filter(w=>w.surah===surah&&w.ayah===ayah);
  if(!words.length)throw new Error("Training ayah missing from the reference");
  const pronunciation=index.phonemes.slice(words[0].start,words.at(-1)!.end);
  const wav=decodeWav(new Uint8Array(readFileSync(path)));
  let audio=wav.samples;
  if(wav.sampleRate!==16000) {
    const resampler=new Resampler(wav.sampleRate,16000), head=resampler.push(audio), tail=resampler.flush();
    audio=new Float32Array(head.length+tail.length);audio.set(head);audio.set(tail,head.length);
  }
  const features=phonationFeatures(audio);
  if (features.length<pronunciation.length+2 || features.length>4096)continue;
  rows.push({id:r.id,speaker:r.speaker,kind:"none",split:"train",excluded:false,pronunciation,features,
    audioSha256:createHash("sha256").update(readFileSync(path)).digest("hex")});
}
if(new Set(rows.map(r=>r.speaker)).size<3)throw new Error("Insufficient eligible training-speaker coverage");
mkdirSync(dirname(output),{recursive:true});
writeFileSync(output,rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
writeFileSync(output+".excluded-speakers.json",JSON.stringify([...excluded].sort()));
writeFileSync(output+".provenance.json",JSON.stringify({labelSha256:createHash("sha256").update(source).digest("hex"),
  recordings:rows.length,speakers:[...new Set(rows.map(r=>r.speaker))].sort(),excludedSpeakers:[...excluded].sort()},null,2));
console.log(JSON.stringify({recordings:rows.length,speakers:new Set(rows.map(r=>r.speaker)).size,output}));
}

if(process.argv[1] && resolve(process.argv[1])===fileURLToPath(import.meta.url))main();
