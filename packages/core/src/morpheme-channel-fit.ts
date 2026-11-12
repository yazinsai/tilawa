/** Training-only estimator: pnpm --filter @tilawa/core exec tsx src/morpheme-channel-fit.ts --help-dir ... --out ... */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "./reference.js";
import { channelReference, selectMorphemeChannels } from "./morpheme-channel.js";
import type { TrainingTake } from "./morpheme-channel.js";
import type { PhonemeEvidence } from "./evidence.js";
import { trackerInput } from "./tracker.js";

const { values } = parseArgs({ options: { "help-dir": { type: "string" }, out: { type: "string" } } });
if (!values["help-dir"] || !values.out) throw new Error("Supply --help-dir and --out; scoring sets are not training input");
const directory = resolve(values["help-dir"]);
interface Label {
  id: string; kind: string; speaker: string; split: string; wav: string; ayahs: string[];
  excluded?: unknown; exclude?: unknown; events?: unknown[]; non_errors?: unknown[];
}
const labels: Label[] = readFileSync(resolve(directory, "labels.jsonl"), "utf8").trim().split("\n").map(s => JSON.parse(s));
if (labels.some(l => !l.speaker || !l.split)) throw new Error("Missing training-speaker partition provenance");
// Fail closed on inconsistent speaker partitions before reading any recording.
const speakers = new Map<string, Set<string>>();
for (const l of labels) {
  const partitions = speakers.get(l.speaker) ?? new Set<string>();
  partitions.add(l.split);
  speakers.set(l.speaker, partitions);
}
if ([...speakers.values()].some(s => s.has("train") && s.size > 1)) throw new Error("Speaker appears in both train and another partition");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const takes: TrainingTake[] = [];
const missing: string[] = [];
for (const l of labels) {
  if (l.split !== "train" || l.kind !== "none" || l.excluded || l.exclude || l.events?.length || l.non_errors?.length) continue;
  const path = resolve(directory, l.wav.replace(/\.wav$/, ".evidence.json"));
  if (!existsSync(path)) { missing.push(l.id); continue; }
  const evidence: PhonemeEvidence = JSON.parse(readFileSync(path, "utf8"));
  if (evidence.inventory !== index.inventoryId) throw new Error(`Inventory mismatch in training take ${l.id}`);
  const words = index.words.filter(w => l.ayahs.includes(`${w.surah}:${w.ayah}`));
  if (!words.length) throw new Error(`Missing training transcript ${l.id}`);
  const from = words[0].start, to = words.at(-1)!.end;
  if (index.words.filter(w => w.start >= from && w.start < to).some(w => !l.ayahs.includes(`${w.surah}:${w.ayah}`))) throw new Error("Non-contiguous training transcript");
  const heard = trackerInput(evidence).phonemes.map(p => p.p).join("");
  if (heard) takes.push({ reference: channelReference(index, from, to), heard, speaker: l.speaker });
}
console.log(JSON.stringify({ trainingTakes: takes.length, trainingSpeakers: new Set(takes.map(t => t.speaker)).size, missing: missing.length }));
if (missing.length) throw new Error("Missing training evidence; decode the shipped listener before fitting");
const fitted = selectMorphemeChannels(takes, [...index.tokens].join(""));
writeFileSync(resolve(values.out), JSON.stringify({ ...fitted, referenceVersion: index.version }, null, 2));
console.log(JSON.stringify({ ...fitted, models: undefined }));
