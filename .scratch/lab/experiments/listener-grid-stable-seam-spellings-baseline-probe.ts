// Read-only baseline replay; no model inference or scoring changes.
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { createHash } from "node:crypto";
import { loadReferenceIndex } from "../../../packages/core/src/reference.js";
import { createTracker, trackerInput } from "../../../packages/core/src/tracker.js";
import { chunks } from "../../../packages/core/src/bench/lab.js";
process.chdir(new URL("../../../", import.meta.url).pathname);
// Remove only this experiment's additions and verify byte identity with the
// starting checker's source before executing the baseline.
let source = readFileSync("packages/core/src/checker.ts", "utf8");
source = source.replace('import { excisedNeighbourMargins, gridSpan, surplusNasal } from "./seam-excision.js";\n', "");
const cut = (first: string, next: string) => {
  const a = source.indexOf(first), b = source.indexOf(next, a);
  if (a < 0 || b < 0) throw new Error(`Missing baseline boundary ${first}`);
  source = source.slice(0, a) + source.slice(b);
};
cut("  // A completed seam remains", "  const parallels");
cut("    /** Binary identity agreement", "    for (const e of stepEdits");
cut("      for (const p of evidence.phonemes) if (p.seamGrid)", "      if (frames && evidence.posteriors)");
source = source.replace("        for (const f of seamGridRows.keys()) if (f < obs[0].start - 1) seamGridRows.delete(f);\n", "");
if (createHash("sha256").update(source).digest("hex") !== "99a8400da948e211954708b9383aa50d08b769282e98dbe676dab5e97d1610b4")
  throw new Error("Baseline checker differs from 2df59d0e");
writeFileSync("packages/core/src/checker-grid-baseline.ts", source);
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
