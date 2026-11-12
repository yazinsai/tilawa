/** Read-only replay. Case, time, variant and comparison words are explicit CLI arguments. */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createTracker, trackerInput } from "../../packages/core/src/tracker.js";
import { createChecker } from "../../packages/core/src/checker.js";
import { loadReferenceIndex } from "../../packages/core/src/reference.js";
import { chunks } from "../../packages/core/src/bench/lab.js";
import type { EditOrderComparison, EditOrderModel } from "../../packages/core/src/edit-order.js";
import type { EngineEvent } from "../../packages/core/src/events.js";

const { values } = parseArgs({ options: {
  set: { type: "string" }, session: { type: "string" }, at: { type: "string" }, model: { type: "string" },
  variant: { type: "string", default: "phone" }, near: { type: "string" }, restart: { type: "string" },
} });
if (!values.set || !values.session || !values.at || !values.model) throw new Error("--set, --session, --at and --model required");
const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const model: EditOrderModel = JSON.parse(readFileSync(values.model, "utf8"));
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../packages/core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const evidence = JSON.parse(readFileSync(join(data, values.set, "sessions", `${values.session}.spliced-clean-int8.${values.variant}.evidence.json`), "utf8"));
const at = Number(values.at);
const word = (offset: number) => {
  const w = index.words[index.wordAt(Math.max(0, offset))];
  return `${w.surah}:${w.ayah}:${w.word}`;
};
const results = [];
for (const mode of ["audit", "ordered", "shuffled"] as const) {
  let time = 0;
  const comparisons: unknown[] = [], events: EngineEvent[] = [];
  const summarize = (c: EditOrderComparison) => {
    const candidates = c.candidates.map(p => ({ ...p, word: word(p.offset - 1), runStart: word(p.origin),
      orderedCost: p.cost - p.ordered / 3, shuffledCost: p.cost - p.shuffled / 3 }));
    const bestAt = (key: string | undefined) => candidates.filter(p => p.word === key).sort((a, b) => a.cost - b.cost)[0];
    const near = bestAt(values.near), restart = bestAt(values.restart);
    return { time, ...c, candidates, ...(near && restart && { preference: {
      weightedNearMinusRestart: near.cost - restart.cost,
      orderedNearMinusRestart: near.orderedCost - restart.orderedCost,
      shuffledNearMinusRestart: near.shuffledCost - restart.shuffledCost,
    } }) };
  };
  const tracker = createTracker(index, { mode: "free", phrases, localEditOrder: { model, mode, onComparison(c) {
    if (Math.abs(time - at) <= 5) comparisons.push(summarize(c));
  } } });
  const checker = createChecker(index, { phrases });
  for (const e of chunks(evidence)) {
    time = (e.decodedTo ?? 0) * e.frameMs / 1000;
    const positions = tracker.push(trackerInput(e));
    events.push(...positions, ...checker.push(e, positions));
  }
  const last = tracker.end(); events.push(...last, ...checker.end(last));
  const relevant = events.filter(e => (e.type === "position" || e.type === "checker")
    && (Math.abs(e.timeMs / 1000 - at) <= 10 || e.type === "checker" && e.state === "confirmed"));
  results.push({ mode, comparisons, events: relevant });
  console.log(JSON.stringify({ mode, comparisons: comparisons.length,
    confirmed: relevant.filter(e => e.type === "checker" && e.state === "confirmed") }));
}
writeFileSync(join(out, "edit-order-target-audit.json"), JSON.stringify({ case: values, results }, null, 2));
