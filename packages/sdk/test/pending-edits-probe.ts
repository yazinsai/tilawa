import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { loadReferenceIndex, PosteriorFrames } from "@tilawa/core";
import { chunks, runEngine } from "../../core/src/bench/lab.js";
import { discoverPendingEditWindows, jointPendingEditPosterior, addPendingEditEvidence } from "../src/pending-edits.js";

const { values } = parseArgs({ options: { session: { type: "string" }, listener: { type: "string" }, at: { type: "string" } } });
const dir = `${process.env.LAB_DATA}/lab-v1.5u/sessions`;
const session = JSON.parse(readFileSync(`${dir}/${values.session}.json`, "utf8"));
const evidence = JSON.parse(readFileSync(`${dir}/${values.session}.${values.listener}.phone.evidence.json`, "utf8"));
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../core/assets/phrases-v1.json", import.meta.url), "utf8"));
const cells = chunks(evidence);
const frames = new PosteriorFrames(index.tokens.size);
for (const e of cells) if (e.posteriors) frames.append(e.posteriors);
const events = runEngine(session, evidence, index, phrases);
const at = Number(values.at);
console.log("baseline", JSON.stringify(events.filter(e => e.type === "checker" && Math.abs(e.timeMs / 1000 - at) < 12)));
const windows = discoverPendingEditWindows(cells, index, phrases);
const selected = windows.filter(w => Math.abs(w.target.fromFrame * evidence.frameMs / 1000 - at) < 4);
const results = selected.map(window => ({ window, result: jointPendingEditPosterior(index, frames, window),
  targetWord: index.words[window.target.word], neighbours: window.neighbours.map(e => ({ ...e, location: index.words[e.word] })) }));
const enriched = addPendingEditEvidence(cells, index, phrases);
const certified = { ...evidence, phonemes: enriched.flatMap(e => e.phonemes) };
const after = runEngine(session, certified, index, phrases);
console.log("certificates", JSON.stringify(certified.phonemes.flatMap((p: any) => p.pendingEdits ?? [])));
console.log("rejudged", JSON.stringify(after.filter(e => e.type === "checker" && Math.abs(e.timeMs / 1000 - at) < 12)));
console.log("primary preserved", JSON.stringify(evidence.phonemes) === JSON.stringify(certified.phonemes.map((p: any) => {
  const { pendingEdits, ...primary } = p; return primary;
})));
console.log("windows", JSON.stringify(results));
writeFileSync(`${process.env.LAB_OUT}/pending.json`, JSON.stringify({ session: session.id, windows: results,
  certificates: certified.phonemes.flatMap((p: any) => p.pendingEdits ?? []),
  before: events, after, discovered: windows.length }, null, 2));
