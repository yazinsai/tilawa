/** Read-only, single-case replay. No scorer, label editing or benchmark-dependent engine behavior. */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { join } from "node:path";
import { createTracker, trackerInput } from "../../packages/core/src/tracker.js";
import { createChecker } from "../../packages/core/src/checker.js";
import { loadReferenceIndex } from "../../packages/core/src/reference.js";
import { chunks } from "../../packages/core/src/bench/lab.js";
import type { SessionEvidence } from "../../packages/core/src/bench/sessions.js";
import type { TokenEditModel } from "../../packages/core/src/heldout-token.js";
import type { EngineEvent } from "../../packages/core/src/events.js";

const { values } = parseArgs({ options: { set: { type: "string" }, session: { type: "string" },
  model: { type: "string" }, variant: { type: "string", default: "phone" },
  listener: { type: "string", default: "spliced-clean-int8" }, mode: { type: "string", default: "audit" } } });
if (!values.set || !values.session || !values.model || !["audit", "validate", "original"].includes(values.mode!)) throw new Error("set, session, model and valid mode required");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../packages/core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const evidence = JSON.parse(readFileSync(join(process.env.LAB_DATA!, values.set, "sessions",
  `${values.session}.${values.listener}.${values.variant}.evidence.json`), "utf8")) as SessionEvidence;
const model = JSON.parse(readFileSync(values.model, "utf8")) as TokenEditModel;
let timeMs = 0;
const comparisons: unknown[] = [];
const tracker = createTracker(index, { mode: "free", phrases, ...(values.mode === "original" ? {} : {
  localTokenValidation: { model, audit: values.mode === "audit", onComparison: c => comparisons.push({ timeMs, ...c }) },
}) });
const checker = createChecker(index, { phrases }), events: EngineEvent[] = [];
for (const e of chunks(evidence)) {
  timeMs = (e.decodedTo ?? 0) * e.frameMs;
  const positions = tracker.push(trackerInput(e));
  events.push(...positions, ...checker.push(e, positions));
}
const positions = tracker.end();
events.push(...positions, ...checker.end(positions));
const report = { mode: values.mode, comparisons, events };
writeFileSync(join(process.env.LAB_OUT!, `heldout-token-${values.mode}.json`), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ comparisons: comparisons.length, mode: values.mode,
  confirmed: events.filter(e => e.type === "checker" && e.state === "confirmed") }));
