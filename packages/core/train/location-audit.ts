/** Read-only diagnostic. Frozen models are supplied before the diagnostic case is opened. */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { createChecker } from "../src/checker.js";
import { chunks } from "../src/bench/lab.js";
import { validateTrajectoryModel } from "../src/trajectory.js";
import type { TrajectoryModel } from "../src/trajectory.js";

const [evidencePath, frozenPath, seconds, ...keys] = process.argv.slice(2);
if (!process.env.LAB_OUT || !frozenPath || !Number.isFinite(Number(seconds)) || keys.length < 2) {
  throw new Error("evidence.json frozen.json seconds competing-word-keys... and LAB_OUT required");
}
const evidence = JSON.parse(readFileSync(evidencePath, "utf8"));
const frozen = JSON.parse(readFileSync(frozenPath, "utf8"));
validateTrajectoryModel(frozen.models.adversarial);
validateTrajectoryModel(frozen.models.random);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const key = (w: { surah: number; ayah: number; word: number }) => `${w.surah}:${w.ayah}:${w.word}`;
const candidates = keys.map(k => index.words.find(w => key(w) === k));
if (candidates.some(w => !w)) throw new Error("Unknown diagnostic word");
type State = { lo: number; cost: Float64Array; acoustic: Float64Array; origin: Int32Array; edits: Int32Array; best: number };
const summary = (state: State) => candidates.map(w => {
  const cells = Array.from({ length: w!.end - w!.start }, (_, j) => w!.start + j + 1)
    .filter(at => at >= state.lo && at < state.lo + state.cost.length)
    .map(at => ({ at, cost: state.cost[at - state.lo] / 10, acoustic: state.acoustic[at - state.lo] / 10,
      origin: state.origin[at - state.lo] })).filter(c => Number.isFinite(c.cost));
  return { word: key(w!), best: cells.length ? cells.reduce((a, b) => a.cost <= b.cost ? a : b) : null };
});
const core = fileURLToPath(new URL("../src/", import.meta.url));
const original = readFileSync(`${core}tracker.ts`, "utf8");
const source = original.replace(/from "\.\//g, `from "${core}`)
  .replace("local.step(ph.p);", "local.step(ph.p);\n      (globalThis as any).locationAudit?.(ph, before, local.save());");
if (!source.includes("locationAudit?.")) throw new Error("Tracker instrumentation point changed");
const instrumented = `${process.env.LAB_OUT}/location-audit-tracker.ts`;
writeFileSync(instrumented, source);
const { createTracker, trackerInput } = await import(instrumented);
let previous: State | null = null, initial: State | null = null, heard = "";
const intervals: unknown[] = [];
(globalThis as any).locationAudit = (ph: { p: string; start: number }, before: State, after: State) => {
  if (!previous || previous.cost !== before.cost) { initial = before; heard = ""; }
  heard += ph.p;
  previous = after;
  const time = ph.start * evidence.frameMs / 1000;
  if (Math.abs(time - Number(seconds)) > 2) return;
  if (Array.from(initial!.cost).filter(Number.isFinite).length !== 1) throw new Error("Comparison seed is not an initial local lock");
  const seed = { offset: initial!.best, origin: initial!.origin[initial!.best - initial!.lo], edits: initial!.edits[initial!.best - initial!.lo] };
  const rows: Record<string, ReturnType<typeof summary>> = { baseline: summary(after) };
  for (const name of ["adversarial", "random"]) {
    const local = createLocalSearch(index, seed, { model: frozen.models[name] });
    for (const p of heard) local.step(p);
    rows[name] = summary(local.save() as State);
  }
  intervals.push({ time, n: heard.length, seed, lastPhone: ph.p, scores: rows });
};
const product = (model: TrajectoryModel | null) => {
  const tracker = createTracker(index, { phrases, locationModel: model });
  const checker = createChecker(index, { phrases });
  const events = [];
  for (const e of chunks(evidence)) events.push(...checker.push(e, tracker.push(trackerInput(e))));
  events.push(...checker.end(tracker.end()));
  return events.filter(e => e.type === "checker" && e.state === "confirmed");
};
const baseline = product(null);
delete (globalThis as any).locationAudit;
const result = { trainingSha256: frozen.trainingSha256, intervals,
  confirmed: { baseline, adversarial: product(frozen.models.adversarial), random: product(frozen.models.random) } };
writeFileSync(`${process.env.LAB_OUT}/location-audit.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result));
