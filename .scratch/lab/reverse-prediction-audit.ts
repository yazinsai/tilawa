/** Read-only one-session premise replay, with a model frozen before opening session evidence. */
import { readFileSync, writeFileSync } from "node:fs";
import { createTracker, trackerInput } from "../../packages/core/src/tracker.js";
import { createChecker } from "../../packages/core/src/checker.js";
import { loadReferenceIndex } from "../../packages/core/src/reference.js";
import { chunks } from "../../packages/core/src/bench/lab.js";
import type { SessionEvidence } from "../../packages/core/src/bench/sessions.js";
import type { EngineEvent } from "../../packages/core/src/events.js";
import type { LocalPredictiveOptions } from "../../packages/core/src/local.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
const session = process.env.AUDIT_SESSION, modelPath = process.env.AUDIT_MODEL;
const at = Number(process.env.AUDIT_AT);
if (!data || !out || !session || !modelPath || !Number.isFinite(at)) throw new Error("Audit inputs required");
const model = JSON.parse(readFileSync(modelPath, "utf8"));
const card = JSON.parse(readFileSync(new URL("../../lab-experiment.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../packages/core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const evidence = JSON.parse(readFileSync(`${data}/${card.testSet}/sessions/${session}.${card.listener}.phone.evidence.json`, "utf8")) as SessionEvidence;
const word = (offset: number) => {
  const w = index.words[index.wordAt(Math.max(0, offset - 1))]; return `${w.surah}:${w.ayah}:${w.word}`;
};
const reports = [];
for (const mode of ["forward", "reverse"] as const) {
  let timeS = 0;
  const comparisons: unknown[] = [];
  const options: LocalPredictiveOptions = { model, mode, onComparison(c) {
    if (Math.abs(timeS - at) <= 3) comparisons.push({ timeS, ...c,
      previousWord: word(c.previous), proposedWord: word(c.proposed), selectedWord: word(c.selected),
      candidates: c.candidates.map(x => ({ ...x, word: word(x.offset) })) });
  } };
  const tracker = createTracker(index, { phrases, localPredictive: options }), checker = createChecker(index, { phrases });
  const events: EngineEvent[] = [];
  for (const e of chunks(evidence)) {
    timeS = (e.decodedTo ?? e.phonemes.at(-1)?.end ?? 0) * e.frameMs / 1000;
    const positions = tracker.push(trackerInput(e));
    events.push(...positions, ...checker.push(e, positions));
  }
  const tail = tracker.end(); events.push(...tail, ...checker.end(tail));
  reports.push({ mode, comparisons, events: events.filter(e => e.type === "checker" || Math.abs(e.timeMs / 1000 - at) < 10) });
}
writeFileSync(`${out}/predictive-premise-audit.json`, JSON.stringify({ session, at, reports }, null, 2));
console.log(JSON.stringify(reports.map(r => ({ mode: r.mode, comparisons: r.comparisons.length,
  nearby: r.events.filter(e => e.type === "checker" && Math.abs(e.timeMs / 1000 - at) < 10) }))));
