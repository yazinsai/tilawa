/**
 * True-position checker replay of evidence files.
 *
 *   pnpm --filter @tilawa/core exec tsx ../../tools/replay-targets.ts \
 *     --data <sessions dir> --evidence <dir> --session <id> --at <seconds> [--session ... --at ...]
 */
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { disambiguationWords } from "../packages/core/src/disambiguation.ts";
import { loadReferenceIndex } from "../packages/core/src/reference.ts";
import { labSession } from "../packages/core/src/bench/lab.ts";
import type { BenchSession, SessionEvidence } from "../packages/core/src/bench/sessions.ts";

const { values } = parseArgs({
  options: {
    data: { type: "string" },
    evidence: { type: "string" },
    session: { type: "string", multiple: true },
    at: { type: "string", multiple: true },
  },
});
if (!values.data || !values.evidence || !values.session?.length || values.session.length !== values.at?.length) {
  throw new Error("--data, --evidence and paired --session/--at are required");
}
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../packages/core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../packages/core/assets/phrases-v1.json", import.meta.url), "utf8"));
const disamb = disambiguationWords(index);
const seen = new Map<string, ReturnType<typeof labSession>>();
for (let i = 0; i < values.session.length; i++) {
  const id = values.session[i]!;
  const at = Number(values.at![i]);
  let scored = seen.get(id);
  if (!scored) {
    const session = JSON.parse(readFileSync(`${values.data}/${id}.json`, "utf8")) as BenchSession;
    const evidence = JSON.parse(readFileSync(`${values.evidence}/${id}.evidence.json`, "utf8")) as SessionEvidence;
    const t0 = performance.now();
    scored = labSession({ session, split: "practice", evidence: { phone: evidence }, trackerOracle: false }, { index, disamb, phrases });
    seen.set(id, scored);
    console.log(`scored ${id} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  }
  const hits = scored.cases.filter((c) => c.kind === "mistake" && Math.abs(c.startS - at) < 0.05);
  console.log(`== ${id} @ ${at}`);
  if (!hits.length) console.log("(no planted mistake at this time)");
  for (const c of hits) {
    console.log(JSON.stringify({
      key: c.key, errorClass: c.errorClass, word: c.word, truePosition: c.truePosition, outcome: c.outcome, component: c.component,
    }));
  }
}
