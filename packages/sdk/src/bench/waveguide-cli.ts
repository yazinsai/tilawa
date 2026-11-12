/** Generic actual-engine candidate export and reconstruction probe replay. */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, SequenceEvidenceCandidate } from "@tilawa/core";
import { insertionCells } from "../seam-insertion.js";
import { waveguideSupportsRival, type WaveguideComparison } from "../waveguide-evidence.js";

const { values } = parseArgs({ options: { evidence: { type: "string" }, out: { type: "string" }, scores: { type: "string" } } });
if (!values.evidence || !values.out) throw new Error("--evidence and --out required");
const json = (path: string) => JSON.parse(readFileSync(path, "utf8"));
const asset = (name: string) => json(new URL(`../../../core/assets/${name}`, import.meta.url).pathname);
const index = loadReferenceIndex(asset("ref-v2.json")), phrases = asset("phrases-v1.json");
const raw = json(values.evidence);
const chunks: PhonemeEvidence[] = Array.isArray(raw) ? raw : [{ ...raw, posteriors: raw.posteriors && {
  ...raw.posteriors, startFrame: raw.posteriors.startFrame ?? 0, nFrames: raw.posteriors.nFrames ?? raw.nFrames,
} }];
const frameMs = chunks[0].frameMs;
if (chunks.some(e => e.frameMs !== frameMs)) throw new Error("Inconsistent evidence clock");
function replay(sequenceEvidence?: (candidate: SequenceEvidenceCandidate) => boolean) {
  const tracker = createTracker(index, { mode: "free", phrases, acquireMs: 1000 });
  const checker = createChecker(index, { phrases, sequenceEvidence });
  const events = [];
  for (const chunk of insertionCells(chunks)) {
    const t = tracker.push(trackerInput(chunk)); events.push(...t); checker.push(chunk, t);
  }
  const t = tracker.end(); events.push(...t); checker.end(t);
  return { tracker: events, mistakes: checker.mistakes() };
}
const candidates: (SequenceEvidenceCandidate & { frameMs: number })[] = [];
const baseline = replay(c => { candidates.push({ ...c, frameMs }); return true; });
if (!values.scores) {
  const unique = [...new Map(candidates.map(c => [JSON.stringify(c), c])).values()];
  writeFileSync(values.out, JSON.stringify(unique, null, 2));
  console.log(JSON.stringify({ comparisons: unique.length }));
} else {
  const comparisons: WaveguideComparison[] = json(values.scores);
  const after = replay(c => waveguideSupportsRival(c, frameMs, comparisons));
  if (JSON.stringify(after.tracker) !== JSON.stringify(baseline.tracker)) throw new Error("Reconstruction changed tracker output");
  const other = (r: ReturnType<typeof replay>) => r.mistakes.filter(e => e.errorClass !== "wrong_word");
  if (JSON.stringify(other(after)) !== JSON.stringify(other(baseline))) throw new Error("Reconstruction changed another error class");
  writeFileSync(values.out, JSON.stringify({ before: baseline.mistakes, after: after.mistakes }, null, 2));
  console.log(JSON.stringify({ before: baseline.mistakes.length, after: after.mistakes.length }));
}
