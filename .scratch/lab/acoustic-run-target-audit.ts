/** Read-only post-training replay. A training-only fit and parity check are required before opening a case. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createTracker, trackerInput } from "../../packages/core/src/tracker.js";
import { createChecker } from "../../packages/core/src/checker.js";
import { loadReferenceIndex } from "../../packages/core/src/reference.js";
import { createAcousticRunScorer } from "../../packages/core/src/acoustic-run.js";
import type { AcousticRunModel } from "../../packages/core/src/acoustic-run.js";
import type { EngineEvent } from "../../packages/core/src/events.js";
import { chunks } from "../../packages/core/src/bench/lab.js";

const { values } = parseArgs({ options: {
  set: { type: "string" }, session: { type: "string" }, at: { type: "string" }, model: { type: "string" },
  fit: { type: "string" }, parity: { type: "string" }, mode: { type: "string", default: "audit" },
  variant: { type: "string", default: "phone" },
} });
if (!values.model || !values.fit || !values.parity || !values.set || !values.session || !values.at
  || !["audit", "apply"].includes(values.mode!)) throw new Error("model, fit, parity, set, session, at and mode audit|apply required");
const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const fitted = JSON.parse(readFileSync(values.fit, "utf8"));
if (fitted.trainingOnly !== true || !fitted.selected || !fitted.trainingSha256) throw new Error("Frozen training-only calibration required");
const raw = readFileSync(values.model, "utf8"), model = JSON.parse(raw) as AcousticRunModel;
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../packages/core/assets/ref-v2.json", import.meta.url), "utf8")));
const score = createAcousticRunScorer(model, index.inventoryId);
for (const sample of JSON.parse(readFileSync(values.parity, "utf8"))) {
  for (let i = 0; i < sample.choices.length; i++) {
    const cost = score(sample.rows, sample.choices[i]);
    if (cost === null || Math.abs(cost - sample.costs[i]) > 1e-4 * Math.max(1, Math.abs(sample.costs[i]))) {
      throw new Error("Python/TypeScript cross-encoder parity failed");
    }
  }
}
const phrases = JSON.parse(readFileSync(new URL("../../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const evidence = JSON.parse(readFileSync(join(data, values.set, "sessions",
  `${values.session}.spliced-clean-int8.${values.variant}.evidence.json`), "utf8"));
const comparisons: unknown[] = [], events: EngineEvent[] = [];
let time = 0;
const at = Number(values.at);
const word = (offset: number) => index.words[index.wordAt(Math.max(0, offset))];
const tracker = createTracker(index, { mode: "free", phrases, acousticRun: { model, audit: values.mode === "audit",
  inspect(candidates, original, selected) {
    if (Math.abs(time - at) <= 5) comparisons.push({ time, original, selected,
      candidates: candidates.map(c => ({ ...c, word: word(c.offset - 1), runStart: word(c.origin) })) });
  },
} });
const checker = createChecker(index, { phrases });
for (const e of chunks(evidence)) {
  time = (e.decodedTo ?? 0) * e.frameMs / 1000;
  const positions = tracker.push(trackerInput(e));
  events.push(...positions, ...checker.push(e, positions));
}
const final = tracker.end();
events.push(...final, ...checker.end(final));
const relevant = events.filter(e => Math.abs(e.timeMs / 1000 - at) <= 10 || e.type === "checker" && e.state === "confirmed");
writeFileSync(join(out, `acoustic-run-${values.mode}-audit.json`), JSON.stringify({
  modelSha256: createHash("sha256").update(raw).digest("hex"), comparisons, events: relevant,
}, null, 2));
console.log(JSON.stringify({ mode: values.mode, comparisons: comparisons.length, events: relevant }));
