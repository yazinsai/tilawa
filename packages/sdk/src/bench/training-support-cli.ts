/** Training preparation and arbitrary audio/evidence comparison probe. No scoring or target-specific selection. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, SequenceEvidenceCandidate, TrainingSupportEvidence } from "@tilawa/core";
import { Resampler } from "../audio.js";
import { createListener } from "../listener.js";
import { loadNodeModel } from "../node.js";
import { insertionCells } from "../seam-insertion.js";
import { addTrainingSupportEvidence, supportCandidates, supportFeatures, validateTrainingSupportModel } from "../training-support.js";
import { decodeWav } from "../wav.js";

const { values } = parseArgs({ options: {
  help: { type: "string" }, model: { type: "string" }, out: { type: "string" }, "excluded-speakers": { type: "string" },
  emissions: { type: "string" }, audio: { type: "string" }, evidence: { type: "string" },
} });
if (!values.out) throw new Error("--out required");
const json = (path: string) => JSON.parse(readFileSync(path, "utf8"));
const asset = (name: string) => json(new URL(`../../../core/assets/${name}`, import.meta.url).pathname);
const reference = asset("ref-v2.json"), index = loadReferenceIndex(reference), phrases = asset("phrases-v1.json");
const tokens: string = reference.inventory.tokens;
function audio(path: string): Float32Array {
  const wav = decodeWav(readFileSync(path));
  if (wav.sampleRate === 16000) return wav.samples;
  const resampler = new Resampler(wav.sampleRate, 16000), body = resampler.push(wav.samples), tail = resampler.flush();
  const result = new Float32Array(body.length + tail.length);
  result.set(body); result.set(tail, body.length);
  return result;
}
if (values.help) {
  if (!values.model || !values["excluded-speakers"]) throw new Error("--model and --excluded-speakers required for preparation");
  const root = resolve(values.help), denied = new Set<string>(json(values["excluded-speakers"]));
  const labels = readFileSync(resolve(root, "labels.jsonl"), "utf8").split(/\r?\n/).filter(Boolean).map(s => JSON.parse(s));
  for (const r of labels) if (r.split !== "train") denied.add(r.speaker);
  const model = await loadNodeModel(resolve(values.model));
  const pronunciations = new Map<string, string>(reference.ayahs.map((a: { s: number; a: number; p: string }) => [`${a.s}:${a.a}`, a.p]));
  const rows: string[] = [], seen = new Set<string>();
  try {
    for (const r of labels) {
      if (r.split !== "train" || denied.has(r.speaker)) continue;
      const isControl = (r.events ?? []).some((e: { errorClass: string }) => e.errorClass === "wrong_word");
      if (r.kind !== "none" && !isControl) continue;
      if (!r.speaker || !r.wav || !Array.isArray(r.ayahs) || !r.ayahs.length) throw new Error("Training label lacks provenance");
      if (r.kind === "none" && (r.events.length || r.non_errors.length)) throw new Error("None-labelled recording carries a deviation");
      const path = resolve(root, r.wav);
      if (!path.startsWith(root + "/")) throw new Error("Audio path escapes authorized training root");
      const id = createHash("sha256").update(readFileSync(path)).digest("hex");
      if (seen.has(id)) throw new Error("Duplicate source recording");
      seen.add(id);
      const samples = audio(path), features = supportFeatures(samples);
      const pronunciation = r.ayahs.map((a: string) => {
        const p = pronunciations.get(a); if (!p) throw new Error("Unknown canonical training ayah"); return p;
      }).join("");
      let comparisons: SequenceEvidenceCandidate[] = [];
      if (isControl) {
        const listener = createListener(model, { posteriors: true }), chunks: PhonemeEvidence[] = [];
        const e = await listener.push(samples, 16000); if (e) chunks.push(e);
        const tail = await listener.end(); if (tail) chunks.push(tail);
        // Conservatively preserve every supported accusation in a swap-labelled recording.
        // The script is not assumed to identify the exact performed word or phone frames.
        comparisons = supportCandidates(chunks, index, phrases);
      }
      rows.push(JSON.stringify({ source: "help-train", split: "train", excluded: false, speaker: r.speaker, id,
        kind: isControl ? "substitution-control" : "none", pronunciation, features, comparisons }));
    }
  } finally { await model.backend.release(); }
  writeFileSync(values.out, rows.join("\n") + "\n");
  console.log(JSON.stringify({ recordings: rows.length, excludedSpeakers: [...denied], inventory: index.inventoryId, tokens }));
} else {
  if (!values.emissions || !values.audio || !values.evidence) throw new Error("--emissions, --audio, --evidence required for comparison");
  const model = validateTrainingSupportModel(json(values.emissions), index.inventoryId, tokens), raw = json(values.evidence);
  const chunks: PhonemeEvidence[] = Array.isArray(raw) ? raw : [{ ...raw, posteriors: raw.posteriors && {
    ...raw.posteriors, startFrame: raw.posteriors.startFrame ?? 0, nFrames: raw.posteriors.nFrames ?? raw.nFrames,
  } }];
  const comparisons: { candidate: SequenceEvidenceCandidate; comparison: TrainingSupportEvidence | null }[] = [];
  const enriched = addTrainingSupportEvidence(model, audio(values.audio), chunks, index, phrases, {}, "free",
    (comparison, candidate) => comparisons.push({ candidate, comparison }));
  const replay = (evidence: PhonemeEvidence[]) => {
    const tracker = createTracker(index, { mode: "free", phrases, acquireMs: 1000 }), checker = createChecker(index, { phrases });
    const events = [];
    for (const e of insertionCells(evidence)) { const t = tracker.push(trackerInput(e)); events.push(...t); checker.push(e, t); }
    const t = tracker.end(); events.push(...t); checker.end(t);
    return { events, mistakes: checker.mistakes() };
  };
  const before = replay(chunks), after = replay(enriched);
  if (JSON.stringify(before.events) !== JSON.stringify(after.events)) throw new Error("Support metadata changed tracker events");
  if (JSON.stringify(before.mistakes.filter(e => e.errorClass !== "wrong_word")) !==
    JSON.stringify(after.mistakes.filter(e => e.errorClass !== "wrong_word"))) throw new Error("Support metadata changed another warning class");
  writeFileSync(values.out, JSON.stringify({ comparisons, before: before.mistakes, after: after.mistakes }, null, 2));
  console.log(JSON.stringify({ comparisons: comparisons.length, vetoes: enriched.flatMap(e => e.phonemes.flatMap(p => p.trainingSupportVetoes ?? [])).length }));
}
