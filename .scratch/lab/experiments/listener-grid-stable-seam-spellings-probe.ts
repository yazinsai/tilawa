// Diagnostic only. Labels are inspected after full-session inference.
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { createListener } from "../../../packages/sdk/src/listener.js";
import { preserveSeamGrids } from "../../../packages/sdk/src/gap-decode.js";
import { loadNodeModel } from "../../../packages/sdk/src/node.js";
import { decodeWav } from "../../../packages/sdk/src/wav.js";
import { loadReferenceIndex } from "../../../packages/core/src/reference.js";
import { createTracker, trackerInput } from "../../../packages/core/src/tracker.js";
import { createChecker } from "../../../packages/core/src/checker.js";
import { chunks, runEngine } from "../../../packages/core/src/bench/lab.js";
import { lexicalWords } from "../../../packages/core/src/checker-lexical.js";
import { gridSpan, surplusNasal } from "../../../packages/core/src/seam-excision.js";
import type { PhonemeEvidence } from "../../../packages/core/src/evidence.js";

process.chdir(new URL("../../../", import.meta.url).pathname);
const root = resolve(process.env.LAB_DATA!, process.argv[3] ?? "lab-v1.5u");
const manifest = JSON.parse(readFileSync(join(root, "manifest-v1.json"), "utf8"));
const model = await loadNodeModel(resolve(process.env.LAB_DATA!, "models/spliced-clean-int8"));
const index = loadReferenceIndex(JSON.parse(readFileSync("packages/core/assets/ref-v2.json", "utf8")));
const phrases = JSON.parse(readFileSync("packages/core/assets/phrases-v1.json", "utf8"));
const lexical = lexicalWords(index);
const targets: [string, number[]][] = JSON.parse(process.argv[2]);
// An ungated parent control, generated only inside this diagnostic's remote run.
const controlPath = "packages/core/src/checker-grid-control.ts";
const source = readFileSync("packages/core/src/checker.ts", "utf8");
if (!source.includes("found && gridAgrees(found)")) throw new Error("Missing agreement gate");
writeFileSync(controlPath, source.replace("found && gridAgrees(found)", "found"));
const { createChecker: controlChecker } = await import("../../../packages/core/src/checker-grid-control.js");
function control(evidence: any) {
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = controlChecker(index, { phrases });
  const out: any[] = [];
  for (const e of chunks(evidence)) {
    const positions = tracker.push(trackerInput(e));
    out.push(...positions, ...checker.push(e, positions));
  }
  const end = tracker.end();
  out.push(...end, ...checker.end(end));
  return out;
}
const results: any[] = [];
for (const [session, seconds] of targets.filter((_, i) => i % Number(process.env.LAB_SHARDS ?? 1) === Number(process.env.LAB_SHARD ?? 0))) {
  const entry = manifest.sessions.find((e: any) => e.id === session);
  if (!entry || entry.split !== "practice") throw new Error(`Missing practice session ${session}`);
  const truth = typeof entry.session === "string" ? JSON.parse(readFileSync(resolve(root, entry.session), "utf8")) : entry.session;
  const saved = JSON.parse(readFileSync(join(root, "sessions", `${session}.spliced-clean-int8.phone.evidence.json`), "utf8"));
  const wav = decodeWav(readFileSync(resolve(root, entry.audio.phone)));
  const listener = createListener(model, { posteriors: true });
  const decoded: PhonemeEvidence[] = [];
  for (let i = 0; i < wav.samples.length; i += 16000) {
    const e = await listener.push(wav.samples.subarray(i, i + 16000), wav.sampleRate);
    if (e) decoded.push(e);
  }
  const tail = await listener.end(); if (tail) decoded.push(tail);
  const raw = decoded.flatMap(e => e.phonemes);
  const evidence = { ...saved, phonemes: preserveSeamGrids(raw, saved.phonemes) };
  const base = control(evidence), gated = runEngine(truth, evidence, index, phrases);
  const rows = new Map<number, number>();
  for (const p of raw) if (p.seamGrid) p.seamGrid.ids.forEach((id, f) => rows.set(p.seamGrid!.startFrame + f, id));
  for (const at of seconds) {
    const planted = truth.events.find((e: any) => e.errorClass === "added_word" && Math.abs(e.startS - at) < 0.02);
    if (!planted) throw new Error(`Missing target ${session}/${at}`);
    const wi = index.words.findIndex(w => w.surah === planted.location.surah && w.ayah === planted.location.ayah && w.word === planted.location.word);
    const matches = (e: any) => e.type === "checker" && e.errorClass === "added_word" &&
      e.location.surah === planted.location.surah && e.location.ayah === planted.location.ayah &&
      e.location.word === planted.location.word && Math.abs(e.timeMs / 1000 - at) < 15;
    const from = Math.floor(planted.startS * 1000 / saved.frameMs), to = Math.ceil(planted.endS * 1000 / saved.frameMs);
    const pp = saved.phonemes.filter((p: any) => p.start >= from - 20 && p.start < to + 20);
    const candidates: any[] = [];
    for (let i = 0; i < pp.length; i++) for (let j = i + 1; j <= pp.length; j++) {
      const said = surplusNasal(pp.slice(i, j).map((p: any) => p.p).join(""));
      const word = lexical.surplusWord(wi, said);
      if (!word) continue;
      const f0 = pp[i].start, f1 = pp[j - 1].end + 1;
      const shifted = gridSpan(rows, [...index.tokens], f0, f1);
      candidates.push({ frames: [f0, f1], said, word, shifted,
        shiftedWord: shifted === null ? null : lexical.surplusWord(wi, surplusNasal(shifted)),
      });
    }
    const result = { session, at, target: planted, base: base.filter(matches), gated: gated.filter(matches),
      span: { frames: [from, to], shipped: saved.phonemes.filter((p: any) => p.start >= from && p.start < to).map((p: any) => p.p).join(""),
        shifted: gridSpan(rows, [...index.tokens], from, to) }, candidates };
    results.push(result); console.log(JSON.stringify(result));
  }
}
writeFileSync(join(process.env.LAB_OUT!, "grid-results.json"), JSON.stringify(results, null, 2));
await model.backend.release?.();
