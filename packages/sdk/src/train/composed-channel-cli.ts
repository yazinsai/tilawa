/** Freeze the channel from real learner training speakers. No practice inputs or
 * mechanically corrected mistake transcripts are accepted as channel supervision.
 * --labels /lab/data/help-train/labels.jsonl --model <shipped model directory>
 * --excluded-speakers <JSON array> --out <training output directory>
 */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { accumulateChannel, channelFromCounts, collapseHeld, emptyCounts, loadReferenceIndex } from "@tilawa/core";
import { ComposedChannel } from "../composed-channel.js";
import { createListener } from "../listener.js";
import { loadNodeModel } from "../node.js";
import { decodeWav } from "../wav.js";

const { values } = parseArgs({ options: { labels: { type: "string" }, model: { type: "string" },
  "excluded-speakers": { type: "string" }, out: { type: "string" } } });
if (!values.labels || !values.model || !values["excluded-speakers"] || !values.out)
  throw new Error("--labels, --model, --excluded-speakers and --out are required");
if (basename(dirname(resolve(values.labels))) !== "help-train") throw new Error("Only help-train labels are authorized");
const json = (file: string) => JSON.parse(readFileSync(file, "utf8"));
const exclusionList: unknown = json(values["excluded-speakers"]);
if (!Array.isArray(exclusionList) || exclusionList.some(s => typeof s !== "string" || !s))
  throw new Error("Excluded speakers must be a JSON array of nonempty speaker ids");
const excluded = new Set<string>(exclusionList);
type Label = { id: string; kind: string; speaker: string; split: string; wav: string; ayahs: string[] };
const labels: Label[] = readFileSync(values.labels, "utf8").split(/\r?\n/).filter(Boolean).map(s => JSON.parse(s));
// Exclusion is speaker-wide, including a training take by a speaker with any held-out take.
for (const l of labels) if (l.split !== "train") excluded.add(l.speaker);
const eligible = labels.filter(l => l.kind === "none" && l.split === "train" && !excluded.has(l.speaker));
const speakers = [...new Set(eligible.map(l => l.speaker))].sort((a, b) => hash(a).localeCompare(hash(b)));
if (speakers.length < 3) throw new Error("At least three none-labelled training speakers are required");
const folds = new Map(speakers.map((s, i) => [s, i % 3]));
const index = loadReferenceIndex(json(new URL("../../../core/assets/ref-v2.json", import.meta.url).pathname));
const model = await loadNodeModel(resolve(values.model));
if (model.manifest.inventory !== index.inventoryId) throw new Error("Training model inventory mismatch");
const tokens = [...index.tokens];
const recordings: { speaker: string; written: string; heard: string; sha256: string }[] = [];
try {
  for (const l of eligible) {
    if (!l.speaker || !l.wav || !Array.isArray(l.ayahs) || !l.ayahs.length) throw new Error("Incomplete training label");
    const wanted = new Set(l.ayahs);
    const words = index.words.filter(w => wanted.has(`${w.surah}:${w.ayah}`) && w.end > w.start);
    if (!words.length) throw new Error("Training label has no reference pronunciation");
    const written = words.map(w => index.phonemes.slice(w.start, w.end)).join("");
    const bytes = readFileSync(resolve(dirname(values.labels), l.wav)), wav = decodeWav(bytes);
    const listener = createListener(model);
    const start = await listener.push(wav.samples, wav.sampleRate), end = await listener.end();
    const heard = [start, end].flatMap(e => e?.phonemes ?? []).map(p => p.p).join("");
    recordings.push({ speaker: l.speaker, written, heard, sha256: createHash("sha256").update(bytes).digest("hex") });
  }
} finally { await model.backend.release(); }
const fit = (rows: typeof recordings) => {
  const counts = emptyCounts(tokens);
  for (const r of rows) accumulateChannel(counts, r.written, r.heard);
  return counts;
};
// Construction and Laplace estimator are fixed. These disjoint-speaker predictions
// audit transfer; no practice-derived cut, transition multiplier or path limit is fitted.
const validation = [];
for (let fold = 0; fold < 3; fold++) {
  const train = recordings.filter(r => folds.get(r.speaker) !== fold);
  const held = recordings.filter(r => folds.get(r.speaker) === fold);
  const scorer = new ComposedChannel(channelFromCounts(fit(train)), tokens);
  const scores = held.map(r => {
    const written = collapseHeld([...r.written]).join(""), heard = collapseHeld([...r.heard]).join("");
    return { speaker: r.speaker, nllPerPhone: -scorer.stringLogP(written, heard)! / Math.max(1, written.length) };
  });
  if (scores.some(s => !Number.isFinite(s.nllPerPhone))) throw new Error("Training validation has unavailable channel evidence");
  validation.push({ fold, trainingRecordings: train.length, heldRecordings: held.length, scores });
}
const counts = fit(recordings);
mkdirSync(values.out, { recursive: true });
writeFileSync(resolve(values.out, "frozen-composed-channel.ts"),
  'import type { ChannelCounts } from "@tilawa/core";\n\n' +
  '/** Training-only help-train counts; generated by train/composed-channel-cli.ts. */\n' +
  `export const FROZEN_COMPOSED_CHANNEL: ChannelCounts = ${JSON.stringify(counts)};\n`);
const report = { inventory: index.inventoryId, modelSha256: model.manifest.sha256,
  labelsSha256: createHash("sha256").update(readFileSync(values.labels)).digest("hex"),
  recordings: recordings.length, speakers: speakers.length, writtenPhones: counts.nWritten,
  excludedSpeakers: [...excluded].sort(), speakerFolds: Object.fromEntries(folds), validation,
  audioHashes: recordings.map(r => r.sha256), construction: "Laplace 0.5; geometric insertions preserving mean; exact CTC/edit marginal" };
writeFileSync(resolve(values.out, "training-report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ recordings: recordings.length, speakers: speakers.length, writtenPhones: counts.nWritten }));
function hash(s: string): string { return createHash("sha256").update(s).digest("hex"); }
