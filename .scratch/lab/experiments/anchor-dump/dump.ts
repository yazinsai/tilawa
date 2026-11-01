/** Oracle-phoneme replay of the six target skips: which seam the checker confirms. */
import { readFileSync } from "node:fs";
import { createChecker } from "../../../../packages/core/src/checker.ts";
import { loadReferenceIndex } from "../../../../packages/core/src/reference.ts";
import type { ReferenceAsset } from "../../../../packages/core/src/reference.ts";
import { oracleEvidence, oraclePositions } from "../../../../packages/core/src/bench/lab.ts";
import { chunk } from "../../../../packages/core/src/bench/index.ts";
import type { BenchSession } from "../../../../packages/core/src/bench/sessions.ts";
import type { PhraseAsset, WordRef } from "../../../../packages/core/src/index.ts";

const asset = JSON.parse(readFileSync(new URL("../../../../packages/core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset;
const phrases = JSON.parse(readFileSync(new URL("../../../../packages/core/assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;
const index = loadReferenceIndex(asset);
const key = (w: WordRef) => `${w.surah}:${w.ayah}:${w.word}`;
const cases = [
  ["lab-v1.5-p24", 77.25, "26:103:1"],
  ["lab-v1.5-p73", 99.92, "26:103:1"],
  ["lab-v1.5-p24", 247.21, "26:130:1"],
  ["lab-v1.4-p10", 35.63, "26:109:1"],
  ["lab-v1.5-p15", 167.88, "55:29:1"],
  ["lab-v1.5-p16", 127.68, "77:33:1"],
] as const;

for (const [id, focus, loc] of cases) {
  const s = JSON.parse(readFileSync(new URL(`./${id}.json`, import.meta.url), "utf8")) as BenchSession;
  const ev = oracleEvidence(s, index);
  const checker = createChecker(index, { phrases, policy: { localWindow: false } });
  const oracle = oraclePositions(s, index);
  const events: string[] = [];
  for (const part of chunk({ ...ev, posteriors: undefined } as never, 0.04, false)) {
    const t = (part.decodedTo ?? 0) * part.frameMs / 1000;
    for (const e of checker.push({ ...part, posteriors: undefined }, oracle(t * 1000))) {
      if (e.type === "checker" && e.state === "confirmed" && Math.abs(e.timeMs / 1000 - focus) <= 25)
        events.push(`${(e.timeMs / 1000).toFixed(2)} ${e.errorClass} ${key(e.location)}`);
    }
    if (t > focus + 30) break;
  }
  for (const e of checker.end()) {
    if (e.type === "checker" && e.state === "confirmed" && Math.abs(e.timeMs / 1000 - focus) <= 25)
      events.push(`${(e.timeMs / 1000).toFixed(2)} ${e.errorClass} ${key(e.location)}`);
  }
  console.log(`\n${id} @ ${focus} want ${loc}`);
  console.log(events.filter((e, i, a) => a.indexOf(e) === i).join("\n") || "(none)");
}
