/** Training preparation, export parity, or generic exact engine-window inspection.
 * Training accepts only none-labelled, training-speaker recordings. Diagnostic
 * evidence never enters model fitting or selects windows independently of the checker. */
import { createReadStream, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, relative, resolve } from "node:path";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-node";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { addWordScrubEvidence, wordScrubFeatures, readoutLogProbs, sequenceLogProbability } from "../word-scrub.js";
import type { WordScrubModel } from "../word-scrub.js";
import { loadNodeModel } from "../node.js";
import { ortBackend } from "../ort.js";
import type { OrtModule } from "../ort.js";
import { createListener } from "../listener.js";
import { insertionCells } from "../seam-insertion.js";
import { decodeWav } from "../wav.js";
import { Resampler } from "../audio.js";
import { MODEL_REGISTRY } from "../registry.js";

const { values } = parseArgs({ options: { model: { type: "string" }, extractor: { type: "string" }, labels: { type: "string" },
  out: { type: "string" }, audio: { type: "string" }, evidence: { type: "string" }, parity: { type: "string" },
  "require-auxiliaries": { type: "boolean", default: false } } });
if (!values.model || !values.out) throw new Error("--model and --out are required");
const json = (path: string) => JSON.parse(readFileSync(path, "utf8"));
const asset = (name: string) => json(new URL(`../../../core/assets/${name}`, import.meta.url).pathname);
const index = loadReferenceIndex(asset("ref-v2.json")), phrases = asset("phrases-v1.json");
const audio = (path: string) => {
  const wav = decodeWav(readFileSync(path));
  if (wav.sampleRate === 16000) return wav.samples;
  const resampler = new Resampler(wav.sampleRate, 16000), body = resampler.push(wav.samples), tail = resampler.flush();
  const samples = new Float32Array(body.length + tail.length); samples.set(body); samples.set(tail, body.length); return samples;
};
const primary = await loadNodeModel(resolve(values.model));
try {
  if (values.labels) {
    if (!values.extractor) throw new Error("--extractor is required");
    const startCpu = process.cpuUsage(), labels = readFileSync(values.labels, "utf8").split(/\r?\n/).filter(Boolean).map(s => JSON.parse(s));
    if (values["require-auxiliaries"]) {
      const auxiliaries = MODEL_REGISTRY.auxiliary?.filter(a => a.inventory === primary.manifest.inventory);
      if (!auxiliaries?.length) throw new Error("No declared inherited auxiliary inventory");
      for (const a of auxiliaries) for (const file of [a, ...a.externalData]) {
        const hash = createHash("sha256"); let bytes = 0;
        for await (const block of createReadStream(resolve(values.model!, file.file))) { hash.update(block); bytes += block.length; }
        if (bytes !== file.bytes || hash.digest("hex") !== file.sha256) throw new Error("Inherited auxiliary checksum mismatch");
      }
    }
    // Any held-out or excluded row bans its complete speaker group, including
    // otherwise train-labelled takes. Never guess missing split provenance.
    const excluded = new Set(labels.filter(r => r.split !== "train" || r.exclude || r.excluded).map(r => String(r.speaker)));
    const metadata = json(resolve(values.extractor, "word-scrub-features-metadata.json")) as WordScrubModel;
    if (metadata.primarySha256 !== primary.manifest.sha256 || metadata.tokens !== primary.manifest.tokens ||
      metadata.inventory !== primary.manifest.inventory) throw new Error("Extractor differs from the primary graph");
    const extractor = await ortBackend(ort as unknown as OrtModule, resolve(values.extractor, "word-scrub-features.onnx"), "onnxruntime-node", {
      executionProviders: ["cpu"], intraOpNumThreads: 1, graphOptimizationLevel: "disabled",
    });
    try {
      const root = realpathSync(dirname(values.labels)), rows = [], preservation = [], skipped = [];
      for (const r of labels) {
        if (r.split !== "train" || excluded.has(String(r.speaker)) || r.scored === false ||
          r.non_errors?.length || r.unsupported?.length || r.exclude || r.excluded) continue;
        const clean = r.kind === "none" && !r.events?.length;
        const substitution = r.events?.some((e: { errorClass: string; state: string }) => e.errorClass === "wrong_word" && e.state === "confirmed");
        if (!clean && !substitution) continue;
        if (!r.speaker || !r.id || !r.wav || !Array.isArray(r.ayahs) || !r.ayahs.length) throw new Error("Missing training provenance");
        const path = realpathSync(resolve(root, r.wav)), inside = relative(root, path);
        if (inside.startsWith("..") || resolve(root, inside) !== path) throw new Error("Waveform escapes authorized training root");
        const samples = audio(path);
        if (samples.length < 1600 || samples.length > 90 * 16000) { skipped.push({ id: r.id, reason: "fixed duration bound" }); continue; }
        if (!clean) {
          // Scripted labels are weak controls: never pretend they describe every
          // performed phone. Keep only accusations already made by the primary
          // engine at a labelled substitution, and retain their exact windows.
          const listener = createListener(primary, { posteriors: true, topK: 5 }), chunks: PhonemeEvidence[] = [];
          for (let at = 0; at < samples.length; at += 16000) {
            const e = await listener.push(samples.slice(at, at + 16000), 16000); if (e) chunks.push(e);
          }
          const tail = await listener.end(); if (tail) chunks.push(tail);
          const tracker = createTracker(index, { mode: "memorization", phrases, acquireMs: 1000 });
          const transcript = r.ayahs.flatMap((a: string) => index.words.filter(w => `${w.surah}:${w.ayah}` === a && w.end > w.start));
          if (!transcript.length) throw new Error("Unknown preservation transcript");
          const toRef = (w: typeof transcript[number]) => ({ surah: w.surah, ayah: w.ayah, word: w.word });
          const candidates: import("@tilawa/core").SequenceEvidenceCandidate[] = [];
          const checker = createChecker(index, { phrases, passage: { from: toRef(transcript[0]), to: toRef(transcript.at(-1)!) }, sequenceEvidence(c) {
            if (r.events.some((e: { errorClass: string; state: string; location: { surah: number; ayah: number; word: number } }) =>
              e.errorClass === "wrong_word" && e.state === "confirmed" && e.location.surah === c.word.surah && e.location.ayah === c.word.ayah && e.location.word === c.word.word)) candidates.push(c);
            return true;
          } });
          for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
          checker.end(tracker.end());
          if (candidates.length) {
            const features = await wordScrubFeatures(extractor, primary.manifest, metadata, samples);
            preservation.push({ id: r.id, speaker: String(r.speaker), source: "help-train", split: "train", kind: "scripted-control",
              candidates, features: features.observations });
          }
          continue;
        }
        const transcriptWords = r.ayahs.flatMap((a: string) => {
          const words = index.words.filter(w => `${w.surah}:${w.ayah}` === a && w.end > w.start);
          if (!words.length) throw new Error("Unknown authorized transcript");
          return words.map(w => index.phonemes.slice(w.start, w.end));
        });
        // Lexical labels are complete word pronunciations. Boundary-crossing
        // collapsed phones are ambiguous and receive no adversarial word label.
        const phones: string[] = [], wordLabels: (string | null)[] = [];
        transcriptWords.forEach(word => [...word].forEach(p => {
          if (phones.at(-1) === p) { if (wordLabels.at(-1) !== word) wordLabels[wordLabels.length - 1] = null; }
          else { phones.push(p); wordLabels.push(word); }
        }));
        const ids = phones.map(p => metadata.tokens.indexOf(p) + 1);
        if (!ids.length || ids.some(i => i < 1)) throw new Error("Unknown training phone");
        const features = await wordScrubFeatures(extractor, primary.manifest, metadata, samples);
        const minimum = ids.length + ids.slice(1).filter((p, i) => p === ids[i]).length;
        if (features.observations.length < minimum) { skipped.push({ id: r.id, reason: "impossible complete CTC transcript" }); continue; }
        rows.push({ id: r.id, speaker: String(r.speaker), source: "help-train", kind: "none", split: "train", excluded: false,
          wavSha256: createHash("sha256").update(readFileSync(path)).digest("hex"), ayahs: r.ayahs,
          features: features.observations, logProbs: features.logProbs, wordLabels, ids });
      }
      const cpu = process.cpuUsage(startCpu), outDir = dirname(values.out);
      writeFileSync(resolve(outDir, "word-scrub-preservation.jsonl"), preservation.map(r => JSON.stringify(r)).join("\n") + "\n");
      writeFileSync(values.out, rows.map(r => JSON.stringify(r)).join("\n") + "\n");
      writeFileSync(resolve(outDir, "word-scrub-excluded-speakers.json"), JSON.stringify([...excluded]));
      const report = { preservationRecordings: preservation.length, recordings: rows.length, speakers: new Set(rows.map(r => r.speaker)).size, excludedSpeakers: [...excluded], skipped,
        labelsSha256: createHash("sha256").update(readFileSync(values.labels)).digest("hex"), cpuSeconds: (cpu.user + cpu.system) / 1e6 };
      writeFileSync(resolve(outDir, "word-scrub-extraction-report.json"), JSON.stringify(report));
      console.log(JSON.stringify(report));
    } finally { await extractor.release(); }
  } else if (values.parity) {
    if (!primary.wordScrub) throw new Error("Parity requires a complete fitted bundle");
    const m = primary.wordScrub.model, fixtures = json(resolve(values.parity, "word-scrub-parity.json")), errors = [];
    for (const f of fixtures) {
      const x = f.features.map((r: number[]) => r.map((v, d) => (v - m.center[d]) / m.scale[d]));
      const head = f.arm === "original" ? m.original : f.arm === "residual" ? m.residual : m.shuffled;
      const score = sequenceLogProbability(readoutLogProbs(head, x), f.ids), error = Math.abs(score - f.score);
      if (!Number.isFinite(error) || error > .002) throw new Error("Exported CTC readout parity failed");
      errors.push(error);
    }
    if (!errors.length) throw new Error("Missing training-only parity fixtures");
    writeFileSync(values.out, JSON.stringify({ comparisons: errors.length, maxError: Math.max(...errors) }));
  } else {
    if (!primary.wordScrub || !values.audio || !values.evidence) throw new Error("--audio and --evidence require a complete fitted bundle");
    const raw = json(values.evidence), chunks: PhonemeEvidence[] = Array.isArray(raw) ? raw : [{ ...raw,
      posteriors: raw.posteriors && { ...raw.posteriors, startFrame: raw.posteriors.startFrame ?? 0, nFrames: raw.posteriors.nFrames ?? raw.nFrames } }];
    const comparisons: unknown[] = [];
    const enriched = await addWordScrubEvidence(primary.wordScrub, primary.manifest, audio(values.audio), chunks, index, phrases, {}, "free",
      (comparison, candidate) => comparisons.push({ candidate, comparison }));
    const replay = (input: PhonemeEvidence[]) => {
      const tracker = createTracker(index, { mode: "free", phrases, acquireMs: 1000 }), checker = createChecker(index, { phrases }), events = [];
      for (const e of insertionCells(input)) { const t = tracker.push(trackerInput(e)); events.push(...t); checker.push(e, t); }
      const t = tracker.end(); events.push(...t); checker.end(t); return { events, mistakes: checker.mistakes() };
    };
    const before = replay(chunks), after = replay(enriched);
    if (JSON.stringify(before.events) !== JSON.stringify(after.events)) throw new Error("Certificate changed primary tracker events");
    if (JSON.stringify(before.mistakes.filter(e => e.errorClass !== "wrong_word")) !== JSON.stringify(after.mistakes.filter(e => e.errorClass !== "wrong_word")))
      throw new Error("Certificate changed another decision class");
    writeFileSync(values.out, JSON.stringify({ comparisons, before: before.mistakes, after: after.mistakes }, null, 2));
    console.log(JSON.stringify({ comparisons: comparisons.length, certificates: enriched.flatMap(e => e.phonemes.flatMap(p => p.wordScrubVetoes ?? [])).length }));
  }
} finally { await primary.backend.release(); }
