/** Export uncertain-addition queries for a frozen embedding's offline replay.
 * This diagnostic calls the engine's actual decision hook; it does not score or
 * change any saved evidence. Audio crops use decoded bounds, never label times.
 */
import { readFileSync, readdirSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createChecker, loadReferenceIndex } from "@tilawa/core";
import { chunks, oraclePositions } from "../../../core/src/bench/lab.js";
import type { BenchSession, SessionEvidence } from "../../../core/src/bench/sessions.js";

const { values } = parseArgs({ options: {
  sessions: { type: "string" }, listener: { type: "string" }, out: { type: "string" },
  variants: { type: "string", default: "phone,studio" },
} });
if (!values.sessions || !values.listener || !values.out) throw new Error("--sessions, --listener and --out required");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../../core/assets/phrases-v1.json", import.meta.url), "utf8"));
const shard = Number(process.env.LAB_SHARD ?? 0), shards = Number(process.env.LAB_SHARDS ?? 1);
const sessions = readdirSync(values.sessions).filter(f => f.endsWith(".json") && !f.endsWith(".evidence.json")).sort()
  .map(file => JSON.parse(readFileSync(resolve(values.sessions!, file), "utf8")) as BenchSession)
  .filter(session => session.version === 1 && session.words && session.events);
mkdirSync(values.out, { recursive: true });
let count = 0;
for (const [fi, session] of sessions.entries()) {
  if (fi % shards !== shard) continue;
  for (const variant of values.variants!.split(",")) {
    const path = resolve(values.sessions, `${session.id}.${values.listener}.${variant}.evidence.json`);
    if (!existsSync(path)) throw new Error(`Missing evidence ${path}`);
    const evidence = JSON.parse(readFileSync(path, "utf8")) as SessionEvidence;
    const queries: unknown[] = [];
    const checker = createChecker(index, { phrases, additionEvidence: span => {
      const neighbours = index.isolatedWordForms!.slice(Math.max(0, span.anchorWord - 3), span.anchorWord + 4);
      queries.push({ ...span, frameMs: evidence.frameMs, fromS: span.fromFrame * evidence.frameMs / 1000, toS: span.toFrame * evidence.frameMs / 1000,
        neighbours, location: index.words[span.anchorWord],
        gaps: span.gap ? [span.gap] : [] });
      return false;
    } });
    const oracle = oraclePositions(session, index);
    const events = [];
    for (const chunk of chunks(evidence)) events.push(...checker.push(chunk, oracle((chunk.decodedTo ?? 0) * chunk.frameMs)));
    events.push(...checker.end(oracle(Infinity)));
    const result = { session: session.id, variant, clean: session.events.length === 0, audio: session.audio,
      queries, events: events.filter(e => e.errorClass === "added_word"), labels: session.events };
    writeFileSync(resolve(values.out, `${session.id}.${variant}.json`), JSON.stringify(result));
    console.log(JSON.stringify({ session: session.id, variant, queries: queries.length, clean: result.clean }));
    count++;
  }
}
console.log(JSON.stringify({ completed: count }));
