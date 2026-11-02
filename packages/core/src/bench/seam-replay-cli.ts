/**
 * Print ayah-seam spans the tagger would see, and the checker's confirmed mistakes.
 *   pnpm --filter @tilawa/core exec tsx src/bench/seam-replay-cli.ts --session <id> --word <surah:ayah:word>
 */
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createChecker } from "../checker.js";
import { createTracker, trackerInput } from "../tracker.js";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset } from "../reference.js";
import { scoreSeam } from "../seam-tagger.js";
import { chunks } from "./lab.js";
import type { SessionEvidence } from "./sessions.js";

const { values } = parseArgs({ options: {
  set: { type: "string", default: "lab-v1.5" },
  session: { type: "string" },
  word: { type: "string" },
  variant: { type: "string", default: "phone" },
  listener: { type: "string", default: "spliced-clean-int8" },
} });
if (!values.session) throw new Error("--session is required");
const dir = `${process.env.LAB_DATA ?? "/lab/data"}/${values.set}/sessions`;
const evidence = JSON.parse(readFileSync(`${dir}/${values.session}.${values.listener}.${values.variant}.evidence.json`, "utf8")) as SessionEvidence;
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;

const seen = new Set<string>();
const checker = createChecker(index, {
  phrases,
  seamTrace: (g) => {
    const key = `${g.rightWord}:${g.obsFrom}:${g.obsTo}`;
    if (seen.has(key)) return;
    seen.add(key);
    const w = index.words[g.rightWord];
    const span = g.tokens.slice(g.span[0], g.span[1]);
    const phones = span.map((t) => t.id >= 0 ? "ءبتثجحخدذرزسشصضطظعغفقكلمنهوياۦۥَُِ۪ـٲڇں۾ؙۜ"[t.id] : "?").join("");
    const own = span.map((t) => t.own).join("");
    const where = `${w.surah}:${w.ayah}:${w.word}`;
    if (!values.word || where === values.word) {
      console.log(`seam ${where} span ${span.length} score ${scoreSeam(g.tokens, g.span).toFixed(3)} phones ${phones} own ${own}`);
    }
  },
});
const tracker = createTracker(index, { mode: "free", phrases });
const events = [];
for (const e of chunks(evidence)) {
  const pos = tracker.push(trackerInput(e));
  events.push(...checker.push(e, pos));
}
events.push(...checker.end(tracker.end()));
const hits = events.filter((e) => e.type === "checker" && e.state === "confirmed" && (!values.word || `${e.location.surah}:${e.location.ayah}:${e.location.word}` === values.word));
console.log(`confirmed ${hits.length}` + (values.word ? ` at ${values.word}` : ""));
for (const e of hits) if (e.type === "checker") console.log(`  ${e.errorClass} ${e.location.surah}:${e.location.ayah}:${e.location.word} ${e.observed ?? ""}`);
