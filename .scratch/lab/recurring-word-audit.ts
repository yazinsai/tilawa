/** A single-case acoustic-memory audit; uses the card's listener and does not call a scorer. */
import { readFileSync, writeFileSync } from "node:fs";
import { createChecker } from "../../packages/core/src/checker.js";
import { loadReferenceIndex } from "../../packages/core/src/reference.js";
import { createTracker, trackerInput } from "../../packages/core/src/tracker.js";
import { chunks } from "../../packages/core/src/bench/lab.js";
import type { EngineEvent, PositionEvent, WordRef } from "../../packages/core/src/events.js";
import type { OccurrenceCalibration, OccurrenceComparison, WordOccurrence } from "../../packages/core/src/recurring-word.js";

const root = new URL("../../", import.meta.url).pathname.replace(/\/$/, "");
const card = JSON.parse(readFileSync(`${root}/lab-experiment.json`, "utf8"));
const session = process.env.AUDIT_SESSION, at = Number(process.env.AUDIT_AT);
if (!session || !Number.isFinite(at) || !process.env.LAB_DATA || !process.env.LAB_OUT) throw new Error("AUDIT_SESSION, AUDIT_AT, LAB_DATA and LAB_OUT required");
const index = loadReferenceIndex(JSON.parse(readFileSync(`${root}/packages/core/assets/ref-v2.json`, "utf8")));
const phrases = JSON.parse(readFileSync(`${root}/packages/core/assets/phrases-v1.json`, "utf8"));
const evidencePath = `${process.env.LAB_DATA}/${card.testSet}/sessions/${session}.${card.listener}.phone.evidence.json`;
const evidence = JSON.parse(readFileSync(evidencePath, "utf8"));
const calibration: OccurrenceCalibration | undefined = process.env.AUDIT_CALIBRATION
  ? JSON.parse(readFileSync(process.env.AUDIT_CALIBRATION, "utf8")).model : undefined;
const apply = process.env.AUDIT_APPLY === "1";
if (apply && !calibration) throw new Error("Application requires a training-frozen calibration");
const occurrences: WordOccurrence[] = [], comparisons: OccurrenceComparison[] = [], events: EngineEvent[] = [];
const tracker = createTracker(index, { phrases, trace: true, acousticMemory: {
  calibration, audit: !apply, occurrence: x => occurrences.push(x), inspect: x => comparisons.push(x),
} });
const checker = createChecker(index, { phrases });
for (const e of chunks(evidence)) {
  const p = tracker.push(trackerInput(e));
  events.push(...p, ...checker.push(e, p));
}
const tail = tracker.end(); events.push(...tail, ...checker.end(tail));
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const near = comparisons.filter(c => Math.abs(c.toFrame * evidence.frameMs / 1000 - at) < 3);
const pronunciations = new Set(near.flatMap(c => [c.continuation, c.restart]));
const positions = events.filter((e): e is PositionEvent => e.type === "position" && Math.abs(e.timeMs / 1000 - at) < 5);
const report = { listener: card.listener, evidencePath, at, apply, calibration: calibration ?? null,
  earlierOccurrences: occurrences.filter(o => pronunciations.has(o.pronunciation) && o.toFrame * evidence.frameMs / 1000 < at)
    .map(({ trajectory, ...o }) => ({ ...o, frames: trajectory.length })),
  comparisons: near,
  positions: positions.map(p => ({ timeS: p.timeMs / 1000, state: p.state, hold: p.hold,
    candidates: p.observed.map(c => ({ word: key(c.word), origin: key(c.span.start), cost: c.cost, confidence: c.confidence })) })),
  warnings: events.filter(e => e.type === "checker" && Math.abs(e.timeMs / 1000 - at) < 30),
};
writeFileSync(`${process.env.LAB_OUT}/recurring-word-audit.json`, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
