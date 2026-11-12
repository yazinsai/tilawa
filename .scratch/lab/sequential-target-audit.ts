/** Read-only diagnostic replay through the unchanged chunker and checker; never scores or modifies evidence. */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { join } from "node:path";
import { createTracker, trackerInput } from "../../packages/core/src/tracker.js";
import { createChecker } from "../../packages/core/src/checker.js";
import { loadReferenceIndex } from "../../packages/core/src/reference.js";
import { chunks } from "../../packages/core/src/bench/lab.js";
import type { SessionEvidence } from "../../packages/core/src/bench/sessions.js";
import type { RestartCalibration } from "../../packages/core/src/sequential-restart.js";
import type { EngineEvent } from "../../packages/core/src/events.js";
const { values } = parseArgs({ options: { set: { type: "string" }, session: { type: "string" }, calibration: { type: "string" },
  mode: { type: "string", default: "sequential" }, listener: { type: "string", default: "spliced-clean-int8" }, variant: { type: "string", default: "phone" } } });
if (!values.set || !values.session || !values.calibration || !["original", "audit", "sequential"].includes(values.mode!))
  throw new Error("set, session, calibration and valid mode required");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../packages/core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const evidence = JSON.parse(readFileSync(join(process.env.LAB_DATA!, values.set, "sessions",
  `${values.session}.${values.listener}.${values.variant}.evidence.json`), "utf8")) as SessionEvidence;
const calibration = JSON.parse(readFileSync(values.calibration, "utf8")) as RestartCalibration;
const comparisons: unknown[] = []; let timeMs = 0;
const tracker = createTracker(index, { mode: "free", phrases, ...(values.mode === "original" ? {} : {
  localRestartEvidence: { calibration, audit: values.mode === "audit", onComparison: c => comparisons.push({ timeMs, ...c }) },
}) });
const checker = createChecker(index, { phrases }), events: EngineEvent[] = [];
for (const e of chunks(evidence)) {
  timeMs = (e.decodedTo ?? 0) * e.frameMs;
  const positions = tracker.push(trackerInput(e));
  events.push(...positions, ...checker.push(e, positions));
}
const positions = tracker.end(); events.push(...positions, ...checker.end(positions));
const report = { mode: values.mode, comparisons, events };
writeFileSync(join(process.env.LAB_OUT!, `sequential-${values.mode}.json`), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ comparisons: comparisons.length, mode: values.mode,
  confirmed: events.filter(e => e.type === "checker" && e.state === "confirmed") }));
