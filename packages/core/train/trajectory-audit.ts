/** Read-only frozen-model replay. Diagnostic word keys and times are CLI inputs, never training features. */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadReferenceIndex } from "../src/reference.js";
import { createChecker } from "../src/checker.js";
import { chunks } from "../src/bench/lab.js";
import { validateTrajectoryModel } from "../src/trajectory.js";
import { UNIT } from "../src/fuzzy.js";

const [evidencePath, frozenPath, seconds, ...keys] = process.argv.slice(2);
if (!process.env.LAB_OUT || !frozenPath || !Number.isFinite(Number(seconds)) || keys.length < 2) {
  throw new Error("evidence.json frozen.json seconds competing-word-keys... and LAB_OUT required");
}
const evidence = JSON.parse(readFileSync(evidencePath, "utf8"));
const frozen = JSON.parse(readFileSync(frozenPath, "utf8"));
validateTrajectoryModel(frozen.model);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const key = (w: { surah: number; ayah: number; word: number }) => `${w.surah}:${w.ayah}:${w.word}`;
const candidates = keys.map(k => index.words.find(w => key(w) === k));
if (candidates.some(w => !w)) throw new Error("Unknown diagnostic word");
const core = fileURLToPath(new URL("../src/", import.meta.url));
const source = readFileSync(`${core}tracker.ts`, "utf8").replace(/from "\.\//g, `from "${core}`)
  .replace("local.step(ph.p);", "local.step(ph.p);\n      (globalThis as any).trajectoryAudit?.(ph, before, local.save());");
if (!source.includes("trajectoryAudit?.")) throw new Error("Tracker instrumentation point changed");
const trackerPath = `${process.env.LAB_OUT}/trajectory-audit-tracker.ts`;
writeFileSync(trackerPath, source);
const { createTracker, trackerInput } = await import(trackerPath);
const results = [];
for (const mode of ["baseline", "conditional"]) {
  const snapshots: unknown[] = [];
  (globalThis as any).trajectoryAudit = (ph: any, before: any, after: any) => {
    const time = ph.start * evidence.frameMs / 1000;
    if (Math.abs(time - Number(seconds)) > 2) return;
    snapshots.push({ time, phone: ph.p, source: before.best, selected: after.best,
      word: key(index.words[index.wordAt(Math.max(0, after.best - 1))]),
      candidates: candidates.map(w => {
        const cells = Array.from({ length: w!.end - w!.start }, (_, j) => w!.start + j + 1)
          .filter(at => at >= after.lo && at < after.lo + after.cost.length)
          .map(at => ({ at, cost: after.cost[at - after.lo] / UNIT, acoustic: after.acoustic[at - after.lo] / UNIT }))
          .filter(c => Number.isFinite(c.cost));
        return { word: key(w!), best: cells.length ? cells.reduce((a, b) => a.cost <= b.cost ? a : b) : null };
      }) });
  };
  const tracker = createTracker(index, { phrases, trajectoryModel: mode === "baseline" ? null : frozen.model });
  const checker = createChecker(index, { phrases });
  const events = [];
  for (const e of chunks(evidence)) events.push(...checker.push(e, tracker.push(trackerInput(e))));
  events.push(...checker.end(tracker.end()));
  results.push({ mode, snapshots, confirmed: events.filter(e => e.type === "checker" && e.state === "confirmed") });
}
delete (globalThis as any).trajectoryAudit;
writeFileSync(`${process.env.LAB_OUT}/trajectory-audit.json`, JSON.stringify(results, null, 2));
console.log(JSON.stringify(results));
