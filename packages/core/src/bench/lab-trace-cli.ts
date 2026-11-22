/**
 * One case through the checker at the true position, as shipped and with each confirmation gate off, plus the
 * listener's phonemes around it (a lab drill-down aid):
 *
 *   pnpm --filter @tilawa/core exec tsx src/bench/lab-trace-cli.ts --set lab-v1.5 --session lab-v1.4-p10 --at 319.3
 */
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createChecker } from "../checker.js";
import type { CheckerPolicy } from "../checker.js";
import type { EngineEvent } from "../events.js";
import { loadReferenceIndex } from "../reference.js";
import { chunks, oraclePositions } from "./lab.js";
import type { BenchSession, SessionEvidence } from "./sessions.js";

const { values } = parseArgs({ options: { set: { type: "string" }, session: { type: "string" }, at: { type: "string" }, variant: { type: "string", default: "phone" },
  listener: { type: "string", default: "spliced-clean-int8" }, around: { type: "string", default: "4" } } });
const dir = `${process.env.LAB_DATA ?? "/lab/data"}/${values.set}/sessions`;
const s = JSON.parse(readFileSync(`${dir}/${values.session}.json`, "utf8")) as BenchSession;
const ev = JSON.parse(readFileSync(`${dir}/${values.session}.${values.listener}.${values.variant}.evidence.json`, "utf8")) as SessionEvidence;
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const at = Number(values.at), around = Number(values.around);

function run(policy: CheckerPolicy): EngineEvent[] {
  const checker = createChecker(index, { phrases, policy }), oracle = oraclePositions(s, index), out: EngineEvent[] = [];
  for (const e of chunks(ev)) { const p = oracle((e.decodedTo ?? 0) * e.frameMs); out.push(...checker.push(e, p)); }
  out.push(...checker.end(oracle(Infinity)));
  return out;
}
const near = (e: EngineEvent) => e.type === "checker" && Math.abs(e.timeMs / 1000 - at) <= around + 3;
for (const [name, policy] of [["shipped", {}], ["minConfidence 0", { minConfidence: 0 }], ["localWindow off", { localWindow: false }],
  ["both off", { minConfidence: 0, localWindow: false }]] as [string, CheckerPolicy][]) {
  const evs = run(policy).filter(near);
  console.log(`== ${name}: checker events within ${around + 3} s of ${at} s:`);
  for (const e of evs) console.log("  ", JSON.stringify(e));
}
console.log("== truth near the case:", JSON.stringify(s.events.filter(e => Math.abs(e.startS - at) <= around)));
const frame = ev.frameMs / 1000;
const ph = (ev.phonemes as any[]).filter(p => Math.abs(p.start * frame - at) <= 1.5);
console.log("== listener phonemes (t, phoneme, conf):", ph.map(p => `${(p.start * frame).toFixed(2)} ${p.p ?? p.phoneme ?? p.id} ${(p.conf ?? p.confidence ?? 0).toFixed(2)}`).join(" | "));
