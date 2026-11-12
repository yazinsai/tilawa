// Read-only case replay. Arguments select the inspection case; none enter engine inference or training.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadReferenceIndex } from "../src/reference.js";
import { createChecker } from "../src/checker.js";
import { chunks } from "../src/bench/lab.js";

const [evidencePath, nearKey, farKey, seconds, modelPath] = process.argv.slice(2);
if (!modelPath || !Number.isFinite(Number(seconds))) throw new Error("evidence.json nearWord farWord seconds model.json required");
const index = loadReferenceIndex(JSON.parse(readFileSync("packages/core/assets/ref-v2.json", "utf8")));
const phrases = JSON.parse(readFileSync("packages/core/assets/phrases-v1.json", "utf8"));
const ev = JSON.parse(readFileSync(evidencePath, "utf8"));
const model = JSON.parse(readFileSync(modelPath, "utf8")).model;
const core = fileURLToPath(new URL("../src/", import.meta.url));
const original = readFileSync(`${core}tracker.ts`, "utf8").replace(/from "\.\//g, `from "${core}`);
const key = (w: any) => `${w.surah}:${w.ayah}:${w.word}`;
const words = [nearKey, farKey].map(k => index.words.find(w => key(w) === k)!);
if (words.some(w => !w)) throw new Error("Unknown inspection word");
const results: any[] = [];
for (const [name, correction] of [["baseline", null], ["frozen", model]] as const) {
  const source = original.replace(/import \{ REFERENCE_EDIT_MODEL \} from [^;]+;/, `const REFERENCE_EDIT_MODEL = ${JSON.stringify(correction)};`)
    .replace("local.step(ph.p);", "local.step(ph.p);\n      (globalThis as any).referenceEditAudit?.(ph, before, local.save());");
  const temporary = `${process.env.LAB_OUT}/${name}-tracker.ts`;
  writeFileSync(temporary, source);
  const { createTracker, trackerInput } = await import(temporary);
  let tokens = 0;
  const snapshots: any[] = [];
  (globalThis as any).referenceEditAudit = (ph: any, before: any, after: any) => {
    tokens++;
    const time = ph.start * ev.frameMs / 1000;
    if (Math.abs(time - Number(seconds)) > 2) return;
    const candidates = words.map(w => {
      const cells = Array.from({ length: w.end - w.start + 1 }, (_, k) => {
        const at = w.start + k, i = at - after.lo;
        return { at, raw: after.cost[i], rank: after.ranking[i], origin: after.origin[i] };
      }).filter(c => Number.isFinite(c.rank));
      return { word: key(w), best: cells.length ? cells.reduce((a, b) => a.rank < b.rank ? a : b) : null };
    });
    snapshots.push({ time, token: tokens, phone: ph.p, beforeBest: before.best, best: after.best,
      word: key(index.words[index.wordAt(after.best - 1)]), candidates });
  };
  const tracker = createTracker(index, { phrases }), checker = createChecker(index, { phrases });
  const checked: any[] = [];
  for (const e of chunks(ev)) { const p = tracker.push(trackerInput(e)); checked.push(...checker.push(e, p)); }
  checked.push(...checker.end(tracker.end()));
  results.push({ name, snapshots, confirmed: checked.filter(e => e.type === "checker" && e.state === "confirmed") });
}
delete (globalThis as any).referenceEditAudit;
writeFileSync(`${process.env.LAB_OUT}/target.json`, JSON.stringify(results, null, 2));
console.log(JSON.stringify(results));
