// Temporary premise probe; case selectors are supplied by the caller, never engine code.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { loadReferenceIndex } from "../../src/reference.js";
import { fitEditAlignment, editAlignmentLogP } from "../../src/edit-alignment.js";
import { editTrainingReference } from "../../src/edit-alignment-training.js";
import { createTracker, trackerInput } from "../../src/tracker.js";
import { createTracker as createAuditTracker } from "./tracker.js";
import { setAudit } from "./local.js";
import type { Node } from "./local.js";
import { chunks } from "../../src/bench/lab.js";
import { createChecker } from "../../src/checker.js";
import assert from "node:assert/strict";

const data = process.env.LAB_DATA!, out = process.env.LAB_OUT!;
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const labels = readFileSync(`${data}/help-train/labels.jsonl`, "utf8").trim().split("\n").map(s => JSON.parse(s));
assert(labels.every(r => r.split === "train"));
const eligible = labels.filter(r => r.device === "phone" && editTrainingReference(index, r));
const decodePath = (id: string) => `/scratch/tracker-marginal-edit-alignments/decodes/${id}.json`;
const deadline = Date.now() + 15 * 60 * 1000;
while (!eligible.every(row => existsSync(decodePath(row.id)))) {
  if (Date.now() >= deadline) throw new Error("Training decodes incomplete");
  await new Promise(resolve => setTimeout(resolve, 5000));
}
const pairs = eligible.map(row => {
  const pair = JSON.parse(readFileSync(decodePath(row.id), "utf8"));
  assert.equal(pair.written, editTrainingReference(index, row));
  assert.equal(pair.speaker, row.speaker);
  assert(pair.heard.length > 0);
  return pair;
});
const model = fitEditAlignment([...index.tokens], pairs);
const frozen = JSON.stringify(model), hash = createHash("sha256").update(frozen).digest("hex");
writeFileSync(`${out}/fit.json`, JSON.stringify({ model, hash, clips: pairs.length,
  speakers: new Set(pairs.map(p => p.speaker)).size, writtenTokens: pairs.reduce((n, p) => n + p.written.length, 0),
  heardTokens: pairs.reduce((n, p) => n + p.heard.length, 0),
  kinds: Object.fromEntries([...new Set(pairs.map(p => p.kind))].map(k => [k, pairs.filter(p => p.kind === k).length])) }, null, 2));
console.log("TRAINING FROZEN", hash, "clips", pairs.length);
const ev = JSON.parse(readFileSync(`${data}/${process.env.AUDIT_SET}/sessions/${process.env.AUDIT_SESSION}.${process.env.AUDIT_LISTENER}.phone.evidence.json`, "utf8"));
const near = process.env.AUDIT_NEAR!.split(":").map(Number);
const far = process.env.AUDIT_FAR!.split(":").map(Number);
const nearWi = index.words.findIndex(w => w.surah === near[0] && w.ayah === near[1] && w.word === near[2]);
const farWi = index.words.findIndex(w => w.surah === far[0] && w.ayah === far[1] && w.word === far[2]);
assert(nearWi >= 0 && farWi >= 0);
const at = Number(process.env.AUDIT_AT), rows: any[] = [];
let boundary = 0, checks = 0;
const pathScore = (node: Node, heard: string, normalized: number) => {
  const nodes: Node[] = [];
  for (let cur: Node | null = node; cur; cur = cur.prev) nodes.push(cur);
  nodes.reverse();
  assert.equal(nodes.map(n => n.obs).join(""), heard);
  const original = nodes.reduce((sum, n) => sum + n.cost, 0);
  assert(Math.abs(original - normalized) < 1e-7);
  const written = nodes.map(n => n.ref).join("");
  const movement = nodes.reduce((sum, n) => sum + n.movement, 0);
  // Full edit-path marginal for the same retained reference traversal. Movement arcs
  // keep their prices. No observed prefix is dropped or freely re-started.
  const max = -editAlignmentLogP(written, heard, model, "max") * 10 / 3 + movement;
  const sum = -editAlignmentLogP(written, heard, model) * 10 / 3 + movement;
  // Also condition on the retained jump token boundaries to make the scope explicit.
  const segments: { written: string; heard: string }[] = [{ written: "", heard: "" }];
  for (const n of nodes) {
    if (n.jump) segments.push({ written: "", heard: "" });
    const s = segments.at(-1)!; s.written += n.ref; s.heard += n.obs;
  }
  const bounded = -segments.reduce((s, p) => s + editAlignmentLogP(p.written, p.heard, model), 0) * 10 / 3 + movement;
  return { original: original / 10, max: max / 10, sum: sum / 10, bounded: bounded / 10,
    movement: movement / 10, written, heard, jumps: segments.length - 1, segments };
};
setAudit(s => {
  if (Math.abs(boundary - at) > 0.25) return;
  const bestWi = index.wordAt(s.best - 1);
  if (bestWi !== farWi) return;
  const candidates = Array.from({ length: s.cost.length }, (_, i) => i)
    .filter(i => index.wordAt(Math.max(0, s.lo + i - 1)) === nearWi && Number.isFinite(s.cost[i]));
  assert(candidates.length > 0);
  const ni = candidates.reduce((a, b) => s.cost[b] < s.cost[a] ? b : a), fi = s.best - s.lo;
  const nearScore = pathScore(s.paths[ni], s.heard, s.cost[ni] + s.normBefore);
  const farScore = pathScore(s.paths[fi], s.heard, s.cost[fi] + s.normBefore);
  const farCandidates = Array.from({ length: s.cost.length }, (_, i) => i)
    .filter(i => index.wordAt(Math.max(0, s.lo + i - 1)) === farWi && Number.isFinite(s.cost[i]));
  const scoreCandidates = (cells: number[]) => cells.map(i => ({ offset: s.lo + i,
    ...pathScore(s.paths[i], s.heard, s.cost[i] + s.normBefore) }));
  const allNear = scoreCandidates(candidates), allFar = scoreCandidates(farCandidates);
  const minimum = (scores: typeof allNear, field: "sum" | "max" | "bounded") =>
    scores.reduce((a, b) => b[field] < a[field] ? b : a);
  const row = { boundary, heardTokens: s.heard.length, nearOffset: s.lo + ni, farOffset: s.best,
    nearOrigin: s.origin[ni], farOrigin: s.origin[fi], hash, near: nearScore, far: farScore,
    allNear, allFar, bestMarginalNear: minimum(allNear, "sum").offset, bestMarginalFar: minimum(allFar, "sum").offset,
    originalNearExcess: nearScore.original - farScore.original,
    retainedPathMaxNearExcess: nearScore.max - farScore.max, retainedPathMarginalNearExcess: nearScore.sum - farScore.sum,
    maxNearExcess: minimum(allNear, "max").max - minimum(allFar, "max").max,
    marginalNearExcess: minimum(allNear, "sum").sum - minimum(allFar, "sum").sum,
    boundedNearExcess: minimum(allNear, "bounded").bounded - minimum(allFar, "bounded").bounded };
  rows.push(row);
  console.log("ORDERING", JSON.stringify({ boundary, heardTokens: row.heardTokens, original: row.originalNearExcess,
    max: row.maxNearExcess, marginal: row.marginalNearExcess, bounded: row.boundedNearExcess }));
});
const baseline = createTracker(index, { phrases }), audited = createAuditTracker(index, { phrases });
const checker = createChecker(index, { phrases }), events: any[] = [];
for (const e of chunks(ev)) {
  boundary = (e.decodedTo ?? 0) * e.frameMs / 1000;
  const input = trackerInput(e), expected = baseline.push(input), actual = audited.push(input);
  assert.deepEqual(actual, expected); checks++;
  events.push(...expected, ...checker.push(e, expected));
}
const end = baseline.end(); assert.deepEqual(audited.end(), end);
events.push(...end, ...checker.end(end));
assert(rows.length > 0);
assert.equal(JSON.stringify(model), frozen);
writeFileSync(`${out}/ordering.json`, JSON.stringify({ hash, checks, rows }, null, 2));
writeFileSync(`${out}/baseline.json`, JSON.stringify(events.filter(e => e.type === "checker"), null, 2));
const variant = createTracker(index, { phrases, editAlignmentModel: model });
const variantChecker = createChecker(index, { phrases }), variantEvents: any[] = [];
for (const e of chunks(ev)) {
  const positions = variant.push(trackerInput(e));
  variantEvents.push(...positions, ...variantChecker.push(e, positions));
}
const variantEnd = variant.end();
variantEvents.push(...variantEnd, ...variantChecker.end(variantEnd));
writeFileSync(`${out}/prototype.json`, JSON.stringify(variantEvents.filter(e => e.type === "checker"), null, 2));
const warning = process.env.AUDIT_WARNING ? process.env.AUDIT_WARNING.split(":").map(Number) : [near[0], near[1], 1];
const hasTarget = (es: any[]) => es.some(e => e.type === "checker" && e.state === "confirmed" && e.errorClass === "skipped_passage"
  && e.location.surah === warning[0] && e.location.ayah === warning[1] && e.location.word === warning[2]);
const target = { baseline: hasTarget(events), prototype: hasTarget(variantEvents),
  targetsWon: Number(hasTarget(events) && !hasTarget(variantEvents)), hash };
writeFileSync(`${out}/target.json`, JSON.stringify(target, null, 2));
console.log("PRODUCT REPLAY", JSON.stringify(target));
console.log("COMPLETE", checks, "pushes", rows.length, "ordering columns");
