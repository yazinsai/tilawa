/** Diagnostic only. Run on Modal from packages/sdk after building core; all paths and cases are arguments. */
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { loadNodeModel, decodeWav } from "/app/packages/sdk/src/node.ts";
import { posteriorProductModel } from "/app/packages/sdk/src/posterior-product.ts";
import { createEngine } from "/app/packages/sdk/src/session.ts";
import { measureLatency } from "/app/packages/sdk/src/latency.ts";
import { loadReferenceIndex } from "/app/packages/core/src/reference.ts";
import { runTruePosition } from "/app/packages/core/src/bench/lab.ts";
import { scoreSessionEvents } from "/app/packages/core/src/bench/sessions.ts";

const [firstDir, secondDir, manifestPath, baselineListener, ...cases] = process.argv.slice(2);
if (!cases.length) throw new Error("first-model second-model manifest baseline-listener session-id...");
const read = (p: string) => JSON.parse(readFileSync(p, "utf8"));
const manifest = read(manifestPath);
const base = resolve(manifestPath, "..");
const index = loadReferenceIndex(read("/app/packages/core/assets/ref-v2.json"));
const phrases = read("/app/packages/core/assets/phrases-v1.json");
const first = await loadNodeModel(firstDir), second = await loadNodeModel(secondDir);
const model = posteriorProductModel(first, second);
const engine = createEngine({ reference: index, phrases, model, recordEvidence: true });
const results = [];
for (const [i, id] of cases.entries()) {
  if (i % Number(process.env.LAB_SHARDS ?? 1) !== Number(process.env.LAB_SHARD ?? 0)) continue;
  const entry = manifest.sessions.find((s: any) => s.id === id);
  if (!entry || entry.split !== "practice") throw new Error(`Missing practice entry ${id}`);
  const truth = typeof entry.session === "string" ? read(resolve(base, entry.session)) : entry.session;
  const wav = decodeWav(readFileSync(resolve(base, entry.audio.phone)));
  const session = engine.startFreeRecitation();
  await measureLatency(session, wav.samples, wav.sampleRate, 20);
  const evidence = session.evidence();
  const rows = evidence.flatMap(e => e.posteriors ? [e.posteriors] : []);
  const ev = {
    inventory: index.inventoryId, frameMs: model.manifest.frameMs,
    nFrames: rows.reduce((n, p) => n + p.nFrames, 0),
    phonemes: evidence.flatMap(e => e.phonemes),
    posteriors: { k: rows[0].k, ids: rows.flatMap(p => Array.from(p.ids)), probs: rows.flatMap(p => Array.from(p.probs)) },
  };
  const pattern = manifest.evidencePattern.replace("{id}", id).replace("{listener}", baselineListener).replace("{variant}", "phone");
  const baseline = read(resolve(base, entry.evidence?.[baselineListener]?.phone ?? pattern));
  const replay = (evidence: any) => {
    const events = runTruePosition(truth, evidence, index, phrases);
    const assignments = scoreSessionEvents(truth, events, index, { kind: "none" }).assignments;
    return { assignments, events: events.filter(e => e.type === "checker") };
  };
  const result = { id, models: [first.manifest.id, second.manifest.id], formula: "normalize(exp((log_p0+log_p1)/2))",
    additions: truth.events.map((e: any, eventIndex: number) => ({ ...e, eventIndex })).filter((e: any) => e.errorClass === "added_word"),
    baseline: replay(baseline), product: replay(ev) };
  writeFileSync(join(process.env.LAB_OUT!, `${id}.replay.json`), JSON.stringify(result, null, 2));
  results.push(result);
  console.log(JSON.stringify({ id, additions: result.additions, baseline: result.baseline.assignments, product: result.product.assignments }));
}
await model.backend.release();
