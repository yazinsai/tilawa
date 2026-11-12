/** Training-only paired-origin calibration. Opens no practice labels/evidence and computes no warning scores. */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { collapseHeld, fitPhaseChannel, inferencePhase, loadReferenceIndex, phaseLogLikelihood } from "@tilawa/core";
import type { PhaseRecording, DecodedPhoneme } from "@tilawa/core";
import { loadNodeModel, decodeWav } from "../node.js";
import { Resampler } from "../audio.js";
import { createListener } from "../listener.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
if (Number(process.env.LAB_SHARDS ?? 1) !== 1) throw new Error("Run complete speaker-grouped calibration on one shard");
const cache = process.env.PHASE_CACHE ?? join(out, "paired-decodes");
mkdirSync(cache, { recursive: true });
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../../core/assets/ref-v2.json", import.meta.url), "utf8")));
interface Label { id: string; speaker: string; kind: string; split: string; scored: boolean; excluded?: unknown; ayahs: string[]; wav: string; events: unknown[]; non_errors: unknown[] }
const all = readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim().split("\n").map(x => JSON.parse(x)) as Label[];
const excluded = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
const labels = all.filter(l => l.split === "train" && !excluded.has(l.speaker) && l.kind === "none" && l.scored &&
  !l.excluded && !l.events.length && !l.non_errors.length);
const speakers = [...new Set(labels.map(l => l.speaker))].sort();
if (speakers.length < 2) throw new Error("Need at least two eligible training speakers");
const folds = Math.min(5, speakers.length), foldOf = new Map(speakers.map((s, i) => [s, i % folds]));
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const manifest = JSON.parse(readFileSync(join(data, "models/spliced-clean-int8/manifest.json"), "utf8"));
  // The inference graph's advance, native frame stride and training-time encoder geometry, never push size.
  const chunkFrames = model.manifest.shift * manifest.fbank.frameShiftMs / model.manifest.frameMs;
  if (!Number.isInteger(chunkFrames) || chunkFrames <= 0 || !(manifest.leftContext >= 0) || !(manifest.chunk > 0)) throw new Error("Missing inference geometry");
  const warmupFrames = Math.ceil(manifest.leftContext / manifest.chunk) * chunkFrames;
  const clips: { speaker: string; record: PhaseRecording }[] = [];
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i], views: { p: string; phase: number }[][] = [];
    const audio = readFileSync(join(data, "help-train", l.wav));
    const audioHash = createHash("sha256").update(audio).digest("hex");
    const wav = decodeWav(audio);
    let samples = wav.samples;
    if (wav.sampleRate !== model.manifest.sampleRate) {
      const resampler = new Resampler(wav.sampleRate, model.manifest.sampleRate);
      const a = resampler.push(samples), b = resampler.flush();
      samples = new Float32Array(a.length + b.length); samples.set(a); samples.set(b, a.length);
    }
    for (let origin = 0; origin < chunkFrames; origin++) {
      const prefixFrames = warmupFrames + origin;
      const cacheKey = `${model.manifest.sha256}-${audioHash}-${prefixFrames}`;
      const file = join(cache, `${cacheKey}.json`);
      let phonemes: DecodedPhoneme[];
      if (existsSync(file)) phonemes = JSON.parse(readFileSync(file, "utf8"));
      else {
        const prefix = prefixFrames * model.manifest.frameMs * model.manifest.sampleRate / 1000;
        const padded = new Float32Array(prefix + samples.length); padded.set(samples, prefix);
        const listener = createListener(model);
        phonemes = [...(await listener.push(padded, model.manifest.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
        writeFileSync(file, JSON.stringify(phonemes));
      }
      const view = phonemes.filter(p => p.start >= prefixFrames).map(p => ({ p: p.p, phase: inferencePhase(p)! }));
      if (view.some(p => p.phase === undefined)) throw new Error("A paired decode lost native inference coordinates");
      if (phonemes.some(p => p.inference?.chunkFrames !== chunkFrames)) throw new Error("Inference graph and declared advance disagree");
      views.push(view);
    }
    const ayahs = new Set(l.ayahs), words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length) throw new Error("Empty training reference");
    const written = collapseHeld([...reference.phonemes.slice(words[0].start, words.at(-1)!.end)]).join("");
    clips.push({ speaker: l.speaker, record: { written, views } });
    console.log(`paired training clip ${i + 1}/${labels.length}: ${views.length} origins, ${views.map(v => v.length).join(",")} tokens`);
  }
  {
    const selection: { bins: number; mass: number; logP: number; folds: number[] }[] = [];
    for (const bins of [1, 2, 4, 8].filter(b => b <= chunkFrames)) for (const mass of [16, 64, 256]) {
      const likelihoods: number[] = [];
      for (let fold = 0; fold < folds; fold++) {
        const fit = fitPhaseChannel(clips.filter(c => foldOf.get(c.speaker) !== fold).map(c => c.record), [...reference.tokens], bins, mass);
        likelihoods.push(clips.filter(c => foldOf.get(c.speaker) === fold).reduce((s, c) => s + phaseLogLikelihood(fit, c.record), 0));
      }
      selection.push({ bins, mass, logP: likelihoods.reduce((s, p) => s + p, 0), folds: likelihoods });
      console.log(JSON.stringify(selection.at(-1)));
    }
    selection.sort((a, b) => b.logP - a.logP || a.bins - b.bins || b.mass - a.mass);
    const fitted = fitPhaseChannel(clips.map(c => c.record), [...reference.tokens], selection[0].bins, selection[0].mass);
    writeFileSync(join(out, "phase-channel.json"), JSON.stringify({ ...fitted, geometry: { model: model.manifest.sha256, chunkFrames } }));
    writeFileSync(join(out, "fit-report.json"), JSON.stringify({ modelSha256: model.manifest.sha256, inventory: model.manifest.inventory,
      frameMs: model.manifest.frameMs, chunkFrames, warmupFrames, selection, speakers: speakers.length, clips: clips.length,
      labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"), speakerFolds: [...foldOf],
      trainingOnly: true, pairedOrigins: "all integer native-frame origins within one inference advance; identical audio after warmup",
      alignment: "latent transcript-constrained edits; each recording's views have total weight one" }, null, 2));
  }
} finally { await model.backend.release(); }
