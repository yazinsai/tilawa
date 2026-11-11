/** Generic frozen-scorer diagnostic. Inputs only select the recording to inspect;
 * candidate pronunciations and windows come from actual checker confirmations.
 * No target identities or labels participate in model selection or inference.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, PhraseAsset, TemporalEnvelopeEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { loadNodeModel, readWav } from "../packages/sdk/src/node.js";
import { Resampler } from "../packages/sdk/src/audio.js";
import { addEnvelopeEvidence } from "../packages/sdk/src/temporal-envelope.js";
import { insertionCells } from "../packages/sdk/src/seam-insertion.js";

const { values } = parseArgs({ options: { bundle: { type: "string" }, audio: { type: "string" },
  evidence: { type: "string" }, out: { type: "string" } } });
if (!values.bundle || !values.audio || !values.evidence || !values.out) throw new Error("--bundle, --audio, --evidence and --out are required");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../packages/core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const model = await loadNodeModel(values.bundle);
try {
  if (!model.temporalEnvelopes) throw new Error("The verified bundle has no trained temporal-envelope scorer");
  const input = JSON.parse(readFileSync(values.evidence, "utf8"));
  const chunks: PhonemeEvidence[] = Array.isArray(input) ? input : [{ ...input,
    ...(input.posteriors && { posteriors: { ...input.posteriors, startFrame: input.posteriors.startFrame ?? 0,
      nFrames: input.posteriors.nFrames ?? input.posteriors.ids.length / input.posteriors.k } }) }];
  const wav = await readWav(values.audio);
  let audio = wav.samples;
  if (wav.sampleRate !== 16000) {
    const resampler = new Resampler(wav.sampleRate, 16000), body = resampler.push(audio), tail = resampler.flush();
    audio = new Float32Array(body.length + tail.length); audio.set(body); audio.set(tail, body.length);
  }
  const comparisons: { candidate: SequenceEvidenceCandidate; comparison: TemporalEnvelopeEvidence | null }[] = [];
  const enriched = addEnvelopeEvidence(model.temporalEnvelopes, audio, chunks, index, phrases, {}, "free",
    (comparison, candidate) => comparisons.push({ candidate, comparison }));
  const replay = (evidence: PhonemeEvidence[]) => {
    const tracker = createTracker(index, { mode: "free", phrases, acquireMs: 1000 });
    const checker = createChecker(index, { phrases });
    const positions = [], events = [];
    for (const e of insertionCells(evidence)) {
      const p = tracker.push(trackerInput(e)); positions.push(...p); events.push(...checker.push(e, p));
    }
    const tail = tracker.end(); positions.push(...tail); events.push(...checker.end(tail));
    return { positions, events, mistakes: checker.mistakes() };
  };
  const before = replay(chunks), after = replay(enriched);
  const primary = (evidence: PhonemeEvidence[]) => evidence.map(e => ({ ...e,
    phonemes: e.phonemes.map(({ envelopeVetoes: _removed, ...p }) => p) }));
  const invariant = {
    primaryEvidence: JSON.stringify(primary(chunks)) === JSON.stringify(primary(enriched)),
    trackerEvents: JSON.stringify(before.positions) === JSON.stringify(after.positions),
    otherMistakes: JSON.stringify(before.mistakes.filter(e => e.errorClass !== "wrong_word")) ===
      JSON.stringify(after.mistakes.filter(e => e.errorClass !== "wrong_word")),
  };
  if (Object.values(invariant).some(v => !v)) throw new Error("Side evidence changed a protected primary decision");
  writeFileSync(values.out, JSON.stringify({ invariant, comparisons, before: before.mistakes, after: after.mistakes }, null, 2));
  console.log(JSON.stringify({ ...invariant, comparisons: comparisons.length, before: before.mistakes.length, after: after.mistakes.length }));
} finally { await model.backend.release(); }
