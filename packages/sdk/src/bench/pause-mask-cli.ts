/** Paired full-stream diagnostic. Span selection uses only primary evidence and
 * tracker positions. This command does not read mistake or recording labels. */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { EngineEvent, PhraseAsset, ReferenceAsset, PhonemeEvidence } from "@tilawa/core";
import { createListener } from "../listener.js";
import { loadNodeModel } from "../node.js";
import { decodeWav } from "../wav.js";
import { insertionCells } from "../seam-insertion.js";
import { decodePauseMasked, pauseMaskSpans } from "../pause-mask.js";

const { values } = parseArgs({ options: {
  model: { type: "string" }, masked: { type: "string" }, wav: { type: "string" }, out: { type: "string" },
} });
if (!values.model || !values.wav || !values.out) throw new Error("--model, --wav and --out are required");
const asset = (name: string) => JSON.parse(readFileSync(new URL(`../../../core/assets/${name}`, import.meta.url), "utf8"));
const index = loadReferenceIndex(asset("ref-v2.json") as ReferenceAsset);
const phrases = asset("phrases-v1.json") as PhraseAsset;
const primary = await loadNodeModel(values.model);
const wave = decodeWav(readFileSync(values.wav));
const listener = createListener(primary, { posteriors: true });
const chunks: PhonemeEvidence[] = [];
for (let a = 0; a < wave.samples.length; a += wave.sampleRate) {
  const e = await listener.push(wave.samples.subarray(a,a+wave.sampleRate),wave.sampleRate);
  if (e) chunks.push(e);
}
const tail = await listener.end();
if (tail) chunks.push(tail);
const tracker = createTracker(index, { mode: "free", phrases });
const wi = new Map(index.words.map((w,i) => [`${w.surah}:${w.ayah}:${w.word}`,i]));
const locks: { wi: number; timeMs: number }[] = [];
const record = (events: EngineEvent[]) => {
  for (const e of events) if (e.type === "position" && e.state === "locked" && e.expected) {
    const at = wi.get(`${e.expected.surah}:${e.expected.ayah}:${e.expected.word}`);
    if (at !== undefined && locks.at(-1)?.wi !== at) locks.push({ wi: at, timeMs: e.timeMs });
  }
};
for (const e of insertionCells(chunks)) record(tracker.push(trackerInput(e)));
record(tracker.end());
const spans = pauseMaskSpans(chunks,locks,index);
let masked: PhonemeEvidence[] | undefined;
if (values.masked) {
  if (wave.sampleRate !== primary.manifest.sampleRate) throw new Error("Paired diagnostic requires native-rate audio");
  const auxiliary = await loadNodeModel(values.masked);
  if (!auxiliary.manifest.pauseMask) throw new Error("Auxiliary export has no pause-mask input");
  masked = await decodePauseMasked(auxiliary,wave.samples,spans,5);
  await auxiliary.backend.release();
}
writeFileSync(values.out,JSON.stringify({ frameMs: primary.manifest.frameMs, spans, locks, primary: chunks, masked }));
console.log(JSON.stringify({ frames: listener.decoded, spans: spans.length, masked: !!masked }));
await primary.backend.release();
