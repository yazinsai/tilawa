// Read-only baseline replay; no model inference or scoring changes.
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { loadReferenceIndex } from "../../../packages/core/src/reference.js";
import { createTracker, trackerInput } from "../../../packages/core/src/tracker.js";
import { chunks } from "../../../packages/core/src/bench/lab.js";
process.chdir(new URL("../../../", import.meta.url).pathname);
const { createChecker } = await import("../../../packages/core/src/checker-grid-baseline.js");
const index = loadReferenceIndex(JSON.parse(readFileSync("packages/core/assets/ref-v2.json", "utf8")));
const phrases = JSON.parse(readFileSync("packages/core/assets/phrases-v1.json", "utf8"));
const root = resolve(process.env.LAB_DATA!, "lab-v1.5u");
const targets = JSON.parse(process.argv[2]) as [string, number, number, number, number][];
const results: any[] = [];
for (const [session, surah, ayah, word, at] of targets.filter((_, i) => i % Number(process.env.LAB_SHARDS) === Number(process.env.LAB_SHARD))) {
  const evidence = JSON.parse(readFileSync(join(root, "sessions", `${session}.spliced-clean-int8.phone.evidence.json`), "utf8"));
  const tracker = createTracker(index, { mode: "free", phrases }), checker = createChecker(index, { phrases });
  const events: any[] = [];
  for (const e of chunks(evidence)) { const p = tracker.push(trackerInput(e)); events.push(...checker.push(e, p)); }
  const end = tracker.end(); events.push(...checker.end(end));
  const result = { session, at, events: events.filter(e => e.type === "checker" && e.errorClass === "added_word" &&
    e.location.surah === surah && e.location.ayah === ayah && e.location.word === word && Math.abs(e.timeMs / 1000 - at) < 15) };
  results.push(result); console.log(JSON.stringify(result));
}
writeFileSync(join(process.env.LAB_OUT!, "baseline-results.json"), JSON.stringify(results, null, 2));
