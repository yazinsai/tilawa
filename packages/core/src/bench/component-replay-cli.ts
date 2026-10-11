/** JSON-in / JSON-out worker for the loopback-only scoreboard server. */
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { basename, isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { loadReferenceIndex } from "../reference.js";
import type { PhraseAsset, ReferenceAsset, ReferenceIndex } from "../reference.js";
import type { PhonemeEvidence } from "../evidence.js";
import { clipWords } from "./clip-dump.js";
import type { ClipDump, DumpArm } from "./clip-dump.js";
import type { AcceptanceClip } from "./acceptance.js";
import { stitchSession } from "./jumps.js";
import type { ClipEvidence } from "./jumps.js";
import { oracleClip } from "./scoreboard.js";
import { replayComponents, replaySettings, REPLAY_DEFAULTS } from "./component-replay.js";

const ROOT = fileURLToPath(new URL("../../../../", import.meta.url));
const json = (path: string) => JSON.parse(readFileSync(path, "utf8"));
function inside(path: string): string {
  const p = resolve(ROOT, path), rel = relative(ROOT, p);
  if (rel.startsWith("..") || isAbsolute(rel)) throw new Error("Replay files must be inside the project.");
  if (existsSync(p)) {
    const real = relative(realpathSync(ROOT), realpathSync(p));
    if (real.startsWith("..") || isAbsolute(real)) throw new Error("Replay files must be inside the project.");
  }
  return p;
}
const sources: Record<string, { helpEvidenceDir: string; leadEvidenceDir?: string; modelDir?: string }> = {
  teacher: { helpEvidenceDir: ".scratch/help", leadEvidenceDir: ".scratch/leadin/evidence/teacher" },
  "spliced-clean-int8": { helpEvidenceDir: ".scratch/sdk-help/spliced-clean-int8-node-lw", leadEvidenceDir: ".scratch/leadin/evidence/spliced-clean-int8-lw", modelDir: ".scratch/models/spliced-clean-int8" },
  "spliced-clean-fp16": { helpEvidenceDir: ".scratch/sdk-help/spliced-clean-fp16-node", modelDir: ".scratch/models/spliced-clean-fp16" },
  "spliced-clean-onnx": { helpEvidenceDir: ".scratch/sdk-help/spliced-clean-onnx-node", modelDir: ".scratch/models/spliced-clean" },
  "spliced-clean-lite-int8": { helpEvidenceDir: ".scratch/sdk-help/spliced-clean-lite-int8-node", modelDir: ".scratch/models/spliced-clean-lite-int8" },
};
function readEvidence(path: string): ClipEvidence {
  const e = json(inside(path));
  if (!Number.isInteger(e.nFrames) || e.nFrames < 1 || e.nFrames > 30000 || !Array.isArray(e.phonemes)) throw new Error("Invalid or oversized sound data.");
  return e;
}
function passageFor(index: ReferenceIndex, ayahs: string[]) {
  const words = index.words.filter(w => ayahs.includes(`${w.surah}:${w.ayah}`) && w.end > w.start);
  const first = words.find(w => `${w.surah}:${w.ayah}` === ayahs[0]);
  const last = words.filter(w => `${w.surah}:${w.ayah}` === ayahs.at(-1)).at(-1);
  if (!first || !last) throw new Error("Test passage is missing from the Quran reference.");
  const ref = (w: typeof first) => ({ surah: w.surah, ayah: w.ayah, word: w.word });
  return { from: ref(first), to: ref(last) };
}
function eventKey(e: any) { return JSON.stringify([e.errorClass, e.at ?? `${e.location.surah}:${e.location.ayah}:${e.location.word}`, e.state, e.expected ?? null, e.observed ?? null, e.confidence ?? null]); }
async function execute(request: any) {
  if (!request || !["prepare", "run"].includes(request.action)) throw new Error("Choose prepare or run.");
  if (typeof request.run !== "string" || !/^[\w.-]+\.json$/.test(request.run) || request.run === "index.json") throw new Error("Invalid test run.");
  const dump: ClipDump = json(inside(`.scratch/scoreboard/clips/${request.run}`));
  if (request.recordedAt !== dump.recordedAt) throw new Error("This run has changed. Reload the scoreboard before replaying it.");
  const clip = dump.clips.find(c => c.id === request.clip);
  const setup: DumpArm = request.setup;
  if (!clip || !["free", "declared", "leadFree", "leadDeclared", "oracle"].includes(setup) || !clip.arms[setup]) throw new Error("Recording or setup is unavailable.");
  const stored = dump.replay;
  const source = { ...(sources[dump.listener] ?? {}), ...stored };
  if (!source.helpEvidenceDir) throw new Error("No replay sources are configured for this audio model.");
  const asset: ReferenceAsset = json(inside(stored?.referenceAsset ?? "packages/core/assets/ref-v2.json"));
  const index = loadReferenceIndex(asset);
  const phrases: PhraseAsset = json(inside(stored?.phraseAsset ?? "packages/core/assets/phrases-v1.json"));
  const segments = clip.arms[setup]!.segments ?? [{ source: clip.id, audio: clip.audio, ayahs: clip.ayahs, startMs: 0, endMs: clip.durationS * 1000, onsetMs: 0, planted: false }];
  const parts = segments.map((s, i) => {
    if (!/^(help|leadin)\/[\w.-]+\.wav$/.test(s.audio)) throw new Error("Invalid recording path.");
    const help = s.audio.startsWith("help/");
    const base = help ? source.helpEvidenceDir : source.leadEvidenceDir;
    if (!base) throw new Error("Sound data for the added recording is unavailable.");
    // Teacher files live next to wavs; student decodes live in a flat directory.
    const name = basename(s.audio).replace(/\.wav$/, ".evidence.json");
    const layout = stored?.helpEvidenceLayout ?? (dump.listener === "teacher" ? "wav" : "flat");
    const file = inside(`${base}/${help && layout === "wav" ? "wav/" : ""}${name}`);
    const evidence = readEvidence(file);
    if (evidence.inventory !== index.inventoryId) throw new Error("Sound data uses a different phoneme inventory.");
    const audioFile = inside(`.scratch/${help ? "help/wav" : "leadin/audio"}/${basename(s.audio)}`);
    return { ...s, gapBeforeMs: i ? s.startMs - segments[i - 1].endMs : s.startMs, evidence, audioFile, evidenceFile: relative(ROOT, file), evidenceHash: createHash("sha256").update(readFileSync(file)).digest("hex") };
  });
  const modelDir = source.modelDir ? inside(source.modelDir) : null;
  if (parts.reduce((total, p) => total + p.evidence.nFrames * p.evidence.frameMs + p.gapBeforeMs, 0) > 600000) throw new Error("Replay is limited to ten minutes.");
  const model = modelDir && existsSync(`${modelDir}/manifest.json`) ? json(`${modelDir}/manifest.json`) : null;
  const settings = replaySettings(request.settings ?? stored?.settings ?? REPLAY_DEFAULTS);
  const localSha = execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: ROOT }).toString().trim();
  const localChanges = !!execFileSync("git", ["status", "--porcelain"], { cwd: ROOT }).toString().trim();
  const originalEvidence = parts.find(p => p.source === clip.id)!.evidence;
  const acceptance = (evidence: ClipEvidence, ayahs = clip.ayahs): AcceptanceClip => ({ id: clip.id, ayahs, evidence: evidence as AcceptanceClip["evidence"], events: clip.labels, nonErrors: [] });
  const sourceMatches = JSON.stringify(clipWords(index, asset, acceptance(originalEvidence))) === JSON.stringify(clip.words);
  const meta = {
    run: request.run, recordedAt: dump.recordedAt, recordedSha: dump.sha, localSha, localChanges, listener: dump.listener, clip: clip.id, setup,
    settings, configurationRecorded: !!stored, sourceMatches, freshAvailable: !!model && existsSync(inside(`${modelDir}/${model.file}`)) && setup !== "oracle",
    model: model ? { id: model.id, format: model.format, sha256: model.sha256 } : null,
    reference: index.version, phrases: phrases.version,
    segments: parts.map(({ evidence: _, audioFile: __, ...p }) => p),
  };
  if (request.action === "prepare") return meta;
  if (!["saved", "fresh"].includes(request.listenerMode)) throw new Error("Choose saved sounds or run the listener again.");
  if (request.listenerMode === "fresh") {
    if (!meta.freshAvailable) throw new Error("This setup has no local listener model for a fresh decode.");
    // The native runtime may write its own session ID; keep that temporary file out of the checkout.
    process.chdir(tmpdir());
    const sdkUrl = new URL("../../../sdk/src/node.ts", import.meta.url).href;
    const { loadNodeModel, readWav, createListener } = await import(sdkUrl);
    const loaded = await loadNodeModel(modelDir!, { threads: 1 });
    try {
      for (const p of parts) {
        const wav = await readWav(p.audioFile);
        const listener = createListener(loaded, { posteriors: true, topK: 5 });
        const chunks: PhonemeEvidence[] = [];
        const n = Math.max(1, Math.round(wav.sampleRate * 0.02));
        for (let at = 0; at < wav.samples.length; at += n) { const e = await listener.push(wav.samples.subarray(at, at + n), wav.sampleRate); if (e) chunks.push(e); }
        const e = await listener.end(); if (e) chunks.push(e);
        const rows = chunks.flatMap(c => c.posteriors ? [c.posteriors] : []);
        p.evidence = { inventory: loaded.manifest.inventory, frameMs: loaded.manifest.frameMs, nFrames: listener.decoded, phonemes: chunks.flatMap(c => c.phonemes), ...(rows.length ? { posteriors: { k: rows[0].k, ids: rows.flatMap(r => Array.from(r.ids)), probs: rows.flatMap(r => Array.from(r.probs)) } } : {}) };
      }
    } finally { await loaded.backend.release(); }
  }
  let assembled = stitchSession(index, parts.map(p => ({ source: p.source, ayahs: p.ayahs, gapBeforeMs: p.gapBeforeMs, gapFill: "silence", evidence: p.evidence })));
  let synthetic = false;
  if (setup === "oracle") {
    const oc = oracleClip(index, asset, { id: clip.id, ayahs: clip.ayahs, events: clip.labels, non_errors: [] });
    if (!oc) throw new Error("The labels cannot be turned into simulated sounds.");
    assembled = { evidence: oc.evidence, segments: assembled.segments }; synthetic = true;
  }
  if (assembled.evidence.nFrames * assembled.evidence.frameMs > 600000) throw new Error("Replay is limited to ten minutes.");
  const ayahs = setup.startsWith("lead") ? parts.flatMap(p => p.ayahs) : clip.ayahs;
  const declared = ["declared", "leadDeclared", "oracle"].includes(setup);
  const result = replayComponents(index, phrases, assembled.evidence, declared ? passageFor(index, ayahs) : null, settings);
  const recorded = clip.arms[setup]!.engineEvents;
  const reported = result.final.map(eventKey).sort();
  const baseline = recorded?.map(eventKey).sort();
  const freshSegments = assembled.segments.map((s, i) => ({ ...s, audio: parts[i].audio }));
  return { ...meta, sources: meta.segments, listenerMode: request.listenerMode, synthetic, durationMs: assembled.evidence.nFrames * assembled.evidence.frameMs, segments: freshSegments, steps: result.steps, final: result.final, trackerOptions: result.trackerOptions, checkerOptions: result.checkerOptions,
    listenerOutput: { inventory: assembled.evidence.inventory, frameMs: assembled.evidence.frameMs, nFrames: assembled.evidence.nFrames, phonemes: assembled.evidence.phonemes, hasPosteriors: !!assembled.evidence.posteriors },
    recordedReports: recorded ?? null, matchesRecordedReports: baseline ? JSON.stringify(baseline) === JSON.stringify(reported) : null,
    changedSettings: JSON.stringify(settings) !== JSON.stringify(replaySettings(stored?.settings)),
  };
}
try {
  const input = readFileSync(0, "utf8");
  if (input.length > 16384) throw new Error("Replay request is too large.");
  process.stdout.write(JSON.stringify(await execute(JSON.parse(input))));
} catch (e) { process.stdout.write(JSON.stringify({ error: e instanceof Error ? e.message : "Replay failed." })); process.exitCode = 1; }
