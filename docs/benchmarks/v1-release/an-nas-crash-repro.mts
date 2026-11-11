// Repro for the V1 blocker in v1-release-readiness.md: a free-recitation session dies near the end of An-Nas.
// From packages/sdk:  node --import tsx ../../docs/benchmarks/v1-release/an-nas-crash-repro.mts <model dir> <16 kHz wav>
// e.g. <main>/.scratch/models/spliced-clean-int8 <main>/.scratch/session-bench/qud/witr-muawwidhat.wav
import { readFileSync } from "node:fs";
const W = new URL("../../../packages/", import.meta.url).pathname;
const { loadReferenceIndex } = await import(`${W}core/src/reference.ts`);
const { loadNodeModel } = await import(`${W}sdk/src/node.ts`);
const { createEngine } = await import(`${W}sdk/src/session.ts`);
const { decodeWav } = await import(`${W}sdk/src/wav.ts`);
const reference = loadReferenceIndex(JSON.parse(readFileSync(`${W}core/assets/ref-v2.json`, "utf8")));
const phrases = JSON.parse(readFileSync(`${W}core/assets/phrases-v1.json`, "utf8"));
const engine = createEngine({ reference, phrases, model: await loadNodeModel(process.argv[2]) });
const wav = decodeWav(readFileSync(process.argv[3]));
const s = engine.startFreeRecitation();
const push = Math.round(0.02 * wav.sampleRate);
let rejected = 0, first: string | null = null, events = 0, eventsAfterError = 0;
for (let i = 0; i < wav.samples.length; i += push) {
  try { const ev = await s.pushAudio(wav.samples.subarray(i, i + push), wav.sampleRate); events += ev.length; if (rejected) eventsAfterError += ev.length; }
  catch (e) { rejected++; first ??= `${(i / wav.sampleRate).toFixed(1)} s: ${String(e)}`; }
}
let endError: string | null = null;
try { await s.end(); } catch (e) { endError = String(e); }
console.log(JSON.stringify({ audioS: +(wav.samples.length / wav.sampleRate).toFixed(1), rejectedPushes: rejected, first, events, eventsAfterError, endError }));
