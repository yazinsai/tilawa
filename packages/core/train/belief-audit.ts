/** Read-only diagnostic replay. Case identities are CLI inputs and never enter training or engine inference. */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadReferenceIndex } from "../src/reference.js";
import { createChecker } from "../src/checker.js";
import { chunks } from "../src/bench/lab.js";

const [evidencePath, modelPath, atSeconds, ...wordKeys] = process.argv.slice(2);
if (!modelPath || !Number.isFinite(Number(atSeconds)) || wordKeys.length < 2 || !process.env.LAB_OUT) {
  throw new Error("evidence.json frozen-model.json seconds competing-word-keys... and LAB_OUT required");
}
const ev = JSON.parse(readFileSync(evidencePath, "utf8"));
const fitted = JSON.parse(readFileSync(modelPath, "utf8"));
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const key = (w: { surah: number; ayah: number; word: number }) => `${w.surah}:${w.ayah}:${w.word}`;
const words = wordKeys.map(k => index.words.find(w => key(w) === k));
if (words.some(w => !w)) throw new Error("Unknown diagnostic word");
const core = fileURLToPath(new URL("../src/", import.meta.url));
const original = readFileSync(`${core}tracker.ts`, "utf8").replace(/from "\.\//g, `from "${core}`);
const results = [];
for (const mode of ["baseline", "recurrent", "reset", "nonrecurrent"]) {
  const source = original.replace("local.step(ph.p);", "local.step(ph.p);\n      (globalThis as any).beliefAudit?.(ph, before, local.save());")
    .replace("options.locationBelief });", `options.locationBelief, resetBelief: ${mode === "reset"}, observe() {} });`);
  const path = `${process.env.LAB_OUT}/belief-${mode}-tracker.ts`;
  writeFileSync(path, source);
  const { createTracker, trackerInput } = await import(path);
  const snapshots: unknown[] = [];
  (globalThis as any).beliefAudit = (ph: any, before: any, after: any) => {
    const time = ph.start * ev.frameMs / 1000;
    if (Math.abs(time - Number(atSeconds)) > 2) return;
    const lo = after.frame?.lo ?? after.lo;
    const candidates = words.map(w => {
      const cells = Array.from({ length: w!.end - w!.start }, (_, i) => {
        const at = w!.start + i + 1, cell = at - lo;
        return { at, score: after.ranking[cell], logit: after.frame?.logits[cell] ?? 0 };
      }).filter(c => Number.isFinite(c.score));
      return { word: key(w!), best: cells.length ? cells.reduce((a, b) => a.score < b.score ? a : b) : null };
    });
    snapshots.push({ time, phone: ph.p, beforeBest: before.best, best: after.best,
      word: key(index.words[index.wordAt(Math.max(0, after.best - 1))]), candidates });
  };
  const model = mode === "baseline" ? null : fitted.models[mode === "nonrecurrent" ? "nonrecurrent" : "recurrent"];
  const tracker = createTracker(index, { phrases, locationBelief: model });
  const checker = createChecker(index, { phrases });
  const events = [];
  for (const e of chunks(ev)) events.push(...checker.push(e, tracker.push(trackerInput(e))));
  events.push(...checker.end(tracker.end()));
  results.push({ mode, snapshots, confirmed: events.filter(e => e.type === "checker" && e.state === "confirmed") });
}
delete (globalThis as any).beliefAudit;
writeFileSync(`${process.env.LAB_OUT}/belief-audit.json`, JSON.stringify(results, null, 2));
console.log(JSON.stringify(results));
