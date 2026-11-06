/** Paired SDK check. Refuses to validate a bundle without the two declared side assets. */
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput, sameWrongWord } from "@tilawa/core";
import type { EngineEvent, PhonemeEvidence } from "@tilawa/core";
import { loadNodeModel, readWav } from "../node.js";
import { createEngine } from "../session.js";
import { insertionCells } from "../seam-insertion.js";
import { assertPrimaryEvidence } from "../primary-invariant.js";

const { values } = parseArgs({ options: { model: { type: "string" }, "primary-model": { type: "string" }, wavs: { type: "string" } } });
if (!values.model || !values["primary-model"] || !values.wavs) throw new Error("--model, --primary-model and --wavs are required");
const modelDir = values.model, primaryDir = values["primary-model"];
const manifests = await Promise.all([modelDir, primaryDir].map(d => readFile(join(d, "manifest.json"))));
if (!manifests[0].equals(manifests[1])) throw new Error("Primary manifest bytes changed");
const manifest = JSON.parse(manifests[0].toString());
for (const dir of [modelDir, primaryDir]) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(join(dir, manifest.file))) hash.update(chunk);
  if (hash.digest("hex") !== manifest.sha256) throw new Error("Primary model bytes changed");
}
const asset = async (p: string) => JSON.parse(await readFile(new URL(`../../../core/assets/${p}`, import.meta.url), "utf8"));
const index = loadReferenceIndex(await asset("ref-v2.json")), phrases = await asset("phrases-v1.json");
const baseline = await loadNodeModel(modelDir), candidate = await loadNodeModel(modelDir);
if (!baseline.insertionTeacher || !candidate.insertionTeacher || !candidate.pauseModel) throw new Error("Both the folded teacher and declared pause ear are required");
baseline.pauseModel = undefined;
const replay = (chunks: PhonemeEvidence[]) => {
  const tracker = createTracker(index, { mode: "free", phrases }), checker = createChecker(index, { phrases });
  const positions: EngineEvent[] = [];
  for (const e of insertionCells(chunks)) { const events = tracker.push(trackerInput(e)); positions.push(...events); checker.push(e, events); }
  const end = tracker.end(); positions.push(...end); checker.end(end);
  return { positions, mistakes: checker.mistakes() };
};
try {
  for (const file of (await readdir(values.wavs)).filter(f => f.endsWith(".wav")).sort()) {
    const wav = await readWav(join(values.wavs, file));
    const results = [];
    for (const model of [baseline, candidate]) {
      const session = createEngine({ reference: index, phrases, model, recordEvidence: true }).startFreeRecitation();
      for (let a = 0; a < wav.samples.length; a += Math.round(wav.sampleRate * .02))
        await session.pushAudio(wav.samples.subarray(a, a + Math.round(wav.sampleRate * .02)), wav.sampleRate);
      await session.end(); results.push(session.evidence());
    }
    assertPrimaryEvidence(results[0], results[1]);
    const before = replay(results[0]), after = replay(results[1]);
    if (JSON.stringify(before.positions) !== JSON.stringify(after.positions)) throw new Error("Primary tracker changed");
    const other = (r: typeof before) => r.mistakes.filter(e => e.errorClass !== "wrong_word");
    if (JSON.stringify(other(before)) !== JSON.stringify(other(after))) throw new Error("Folded teacher wins or another primary judgment changed");
    if (after.mistakes.some(e => !before.mistakes.some(b => JSON.stringify(b) === JSON.stringify(e)))) throw new Error("Side ear created a warning");
    const agreements = results[1].flatMap(e => e.phonemes.flatMap(p => p.wrongWordAgreements ?? [])).filter(c => c.acquireMs === 0);
    console.log(JSON.stringify({ file, verified: true, agreements: agreements.map(c => ({ primary: c.primary,
      rejudged: c.rejudged, agrees: c.rejudged.some(r => sameWrongWord(c.primary, r)) })) }));
  }
} finally { await baseline.backend.release(); await candidate.backend.release(); }
