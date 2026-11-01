import { readFileSync } from "node:fs";
import { createChecker } from "../../../packages/core/src/checker.ts";
import { chunks, oracleEvidence, oraclePositions } from "../../../packages/core/src/bench/lab.ts";
import { loadReferenceIndex } from "../../../packages/core/src/reference.ts";
import type { BenchSession } from "../../../packages/core/src/bench/sessions.ts";

const id = process.argv[2];
const at = process.argv.slice(3).map(Number);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../../packages/core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const s = JSON.parse(readFileSync(new URL(`./${id}.json`, import.meta.url), "utf8")) as BenchSession;
const ev = oracleEvidence(s, index);
const checker = createChecker(index, { phrases });
const oracle = oraclePositions(s, index);
const out = [];
for (const e of chunks(ev)) out.push(...checker.push(e, oracle((e.decodedTo ?? 0) * e.frameMs)));
out.push(...checker.end(oracle(Infinity)));
const key = (r: { surah: number; ayah: number; word: number }) => `${r.surah}:${r.ayah}:${r.word}`;
console.log("== confirmed/suspected near targets");
for (const e of out) {
  if (e.type !== "checker") continue;
  const t = e.timeMs / 1000;
  if (!at.some((x) => Math.abs(t - x) < 30) && e.state === "suspected") continue;
  if (e.state === "suspected" && !at.some((x) => Math.abs(t - x) < 25)) continue;
  console.log(e.state, e.errorClass, key(e.location), t.toFixed(2), "c="+e.confidence.toFixed(2), e.expected?.slice(0, 24));
}
