/** Read-only single-case replay; no scorer or benchmark data is changed. */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createTracker, trackerInput } from "../../packages/core/src/tracker.js";
import { createChecker } from "../../packages/core/src/checker.js";
import { loadReferenceIndex } from "../../packages/core/src/reference.js";
import { createWordMetric } from "../../packages/core/src/word-metric.js";
import type { WordMetricModel } from "../../packages/core/src/word-metric.js";
import type { EngineEvent } from "../../packages/core/src/events.js";
import { chunks } from "../../packages/core/src/bench/lab.js";

const { values } = parseArgs({ options: {
  set: { type: "string" }, session: { type: "string" }, at: { type: "string" }, model: { type: "string" },
  mode: { type: "string", default: "audit" }, variant: { type: "string", default: "phone" },
  parity: { type: "string" },
} });
if (!values.model || !values.set || !values.session || !values.at || !["audit", "apply"].includes(values.mode!)) {
  throw new Error("--model, --set, --session, --at and --mode audit|apply required");
}
const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const model = JSON.parse(readFileSync(values.model, "utf8")) as WordMetricModel;
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../packages/core/assets/ref-v2.json", import.meta.url), "utf8")));
if (values.parity) {
  const metric = createWordMetric(model, index.inventoryId);
  for (const sample of JSON.parse(readFileSync(values.parity, "utf8"))) {
    const cost = metric.cost(sample.written, sample.observed);
    if (cost === null || Math.abs(cost - sample.cost) > 1e-4 * Math.max(1, sample.cost)) throw new Error("Python/TypeScript encoder parity failed");
  }
}
const phrases = JSON.parse(readFileSync(new URL("../../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const evidence = JSON.parse(readFileSync(join(data, values.set, "sessions", `${values.session}.spliced-clean-int8.${values.variant}.evidence.json`), "utf8"));
const comparisons: unknown[] = [], events: EngineEvent[] = [];
let time = 0;
const at = Number(values.at);
const word = (offset: number) => index.words[index.wordAt(Math.max(0, offset))];
const tracker = createTracker(index, { mode: "free", phrases, localWordMetric: {
  model, mode: values.mode as "audit" | "apply", audit(c) {
    if (Math.abs(time - at) <= 5) comparisons.push({ time, original: c.original, selected: c.selected,
      candidates: c.candidates.map(p => ({ ...p, word: word(p.offset - 1), runStart: word(p.origin) })) });
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
const relevant = events.filter(e => (e.type === "position" || e.type === "checker")
  && (Math.abs(e.timeMs / 1000 - at) <= 10 || e.type === "checker" && e.state === "confirmed"));
writeFileSync(join(out, `word-metric-${values.mode}-audit.json`), JSON.stringify({ comparisons, events: relevant }, null, 2));
console.log(JSON.stringify({ mode: values.mode, comparisons: comparisons.length, events: relevant }));
