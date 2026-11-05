/** Inspect written-word identity after a full-file replay. Inputs name cases; code is corpus-independent. */
import { readFileSync, writeFileSync, unlinkSync } from "node:fs";
import { parseArgs } from "node:util";
import { pathToFileURL } from "node:url";
import { loadReferenceIndex, createTracker, trackerInput, createChecker } from "@tilawa/core";
import type { CheckerEvent, WordRef } from "@tilawa/core";
import { chunk } from "../../../core/src/bench/index.js";
import { collapseHeld, deletionCost, insertionCost, subCost, UNIT } from "../../../core/src/fuzzy.js";

const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const strip = (s: string) => s.replace(/^[ۦۥں۾ڇؙۜ]+|[ۦۥں۾ڇؙۜ]+$/gu, "");
function hardEdits(reference: string, heard: string): number {
  const a = collapseHeld([...reference]).join(""), b = collapseHeld([...strip(heard)]).join("");
  let row = Array.from({ length: b.length + 1 }, (_, j) => j ? Infinity : 0);
  for (let j = 1; j <= b.length; j++) row[j] = row[j - 1] + Number(insertionCost(a, 0, b[j - 1]) >= UNIT);
  for (let i = 1; i <= a.length; i++) {
    const next = [row[0] + Number(deletionCost(a, i - 1) >= UNIT)];
    for (let j = 1; j <= b.length; j++) next[j] = Math.min(row[j - 1] + Number(subCost(a[i - 1], b[j - 1]) >= UNIT),
      row[j] + Number(deletionCost(a, i - 1) >= UNIT), next[j - 1] + Number(insertionCost(a, i, b[j - 1]) >= UNIT));
    row = next;
  }
  return row[b.length];
}

const { values: v } = parseArgs({ options: { baseline: { type: "string" }, replay: { type: "string" }, cases: { type: "string" }, out: { type: "string" } } });
if (!v.baseline || !v.replay || !v.cases || !v.out) throw new Error("--baseline --replay --cases --out required");
const read = (f: string) => JSON.parse(readFileSync(f, "utf8"));
const index = loadReferenceIndex(read(new URL("../../../core/assets/ref-v2.json", import.meta.url).pathname));
const phrases = read(new URL("../../../core/assets/phrases-v1.json", import.meta.url).pathname);
const baseline = read(v.baseline), replay = read(v.replay);
const cases = read(v.cases) as { location: WordRef; timeS: number }[];
const wanted = new Set(cases.map(c => key(c.location)));
const source = new URL("../../../core/src/checker.ts", import.meta.url);
const temporary = new URL("../../../core/src/checker-training-diagnostic.ts", import.meta.url);
let code = readFileSync(source, "utf8");
// Only add passive callbacks to a temporary sibling copy. The shipped checker file is untouched.
for (const [needle, injected] of [
  ["    emitted.push(e);", "    (globalThis as any).__trainingDecision?.({ ...e, from: obs[issue.obsFrom]?.start, to: (obs[issue.obsTo - 1]?.end ?? -1) + 1, heard: heardSpan(issue) });\n    emitted.push(e);"],
  ["    const judged = judge(steps);", "    const judged = judge(steps);\n    (globalThis as any).__trainingVisits?.(nowMs, (judged?.visits ?? []).map(v => ({ location: toRef(v.word), from: obs[v.obsFrom]?.start, to: (obs[v.obsTo - 1]?.end ?? -1) + 1, heard: obs.slice(v.obsFrom, v.obsTo).filter(p => !p.junk).map(p => p.p).join('') })));"],
]) {
  if (code.split(needle).length !== 2) throw new Error("Diagnostic insertion point changed");
  code = code.replace(needle, injected);
}
writeFileSync(temporary, code);
try {
  const diagnostic = await import(pathToFileURL(temporary.pathname).href);
  type Span = { location: WordRef; timeMs: number; from: number; to: number; heard: string };
  const run = (input: any, factory: typeof createChecker, hooks: boolean) => {
    const decisions: (CheckerEvent & Span)[] = [], visits: Span[] = [];
    if (hooks) {
      (globalThis as any).__trainingDecision = (e: CheckerEvent & Span) => { if (wanted.has(key(e.location))) decisions.push(e); };
      (globalThis as any).__trainingVisits = (timeMs: number, list: Span[]) => {
        for (const s of list) if (wanted.has(key(s.location)) && s.from !== undefined && s.to > s.from) visits.push({ ...s, timeMs });
      };
    }
    const rec = { ...input, inventory: index.inventoryId,
      nFrames: input.nFrames ?? Math.floor(input.audioS * 1000 / input.frameMs) };
    const tracker = createTracker(index, { mode: "free", phrases }), checker = factory(index, { phrases });
    const events: CheckerEvent[] = [];
    for (const e of chunk(rec, 0.04, true)) events.push(...checker.push(e, tracker.push(trackerInput(e))));
    events.push(...checker.end(tracker.end()));
    delete (globalThis as any).__trainingDecision; delete (globalThis as any).__trainingVisits;
    return { events, decisions, visits };
  };
  const b = run(baseline, diagnostic.createChecker, true), n = run(replay, diagnostic.createChecker, true);
  // Passive tracing must reproduce every original event byte-for-byte, on both inputs.
  for (const [input, result] of [[baseline, b], [replay, n]] as const)
    if (JSON.stringify(run(input, createChecker, false).events) !== JSON.stringify(result.events)) throw new Error("Tracing changed checker events");
  const rows = cases.map(c => {
    const known = b.decisions.filter(e => e.errorClass === "wrong_word" && key(e.location) === key(c.location) && e.state === "confirmed")
      .sort((x, y) => Math.abs(x.timeMs / 1000 - c.timeS) - Math.abs(y.timeMs / 1000 - c.timeS))[0];
    if (!known) throw new Error("Target baseline accusation absent");
    if (!Number.isFinite(known.from) || !Number.isFinite(known.to) || known.to <= known.from) throw new Error("Missing baseline frame slot");
    const wi = index.words.findIndex(w => key(w) === key(c.location));
    const w = index.words[wi];
    const forms = [index.phonemes.slice(w.start, w.end), ...(index.wordForms?.[wi] ?? [])];
    // Select by timing alone, never by which candidate span has the fewest edits.
    const nearby = n.visits.filter(s => key(s.location) === key(c.location) && s.from < known.to && s.to > known.from)
      .sort((x, y) => Math.abs(x.timeMs - known.timeMs) - Math.abs(y.timeMs - known.timeMs))[0];
    const fixed = replay.phonemes.filter((p: any) => p.start >= known.from && p.start < known.to).map((p: any) => p.p).join("");
    const span = nearby ?? { from: known.from, to: known.to, heard: fixed, timeMs: known.timeMs, location: c.location };
    const hard = Math.min(...forms.map(f => hardEdits(f, span.heard)));
    return { ...c, baseline: known, candidate: span, hardEdits: hard, spellsWritten: hard === 0,
      confirmed: n.decisions.filter(e => e.state === "confirmed" && key(e.location) === key(c.location)) };
  });
  const output = { modelSha256: replay.modelSha256, passiveTracingIdentical: true, targetsWon: rows.filter(r => r.spellsWritten).length, rows };
  writeFileSync(v.out, JSON.stringify(output, null, 1));
  console.log(JSON.stringify(output));
} finally {
  unlinkSync(temporary);
}
