/** Decode authorized clean training takes only. Heavy inference runs under the supervisor on Modal. */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve, relative, isAbsolute } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { createListener } from "./listener.js";
import { decodeWav, loadNodeModel } from "./node.js";
import { acousticSpeakerFolds, acousticTrainingLabels } from "./acoustic-run-training.js";
import type { AcousticTrainingLabel } from "./acoustic-run-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const root = resolve(data, "help-train");
const text = readFileSync(join(root, "labels.jsonl"), "utf8");
const labels = acousticTrainingLabels(text.trim().split("\n").map(l => JSON.parse(l)) as AcousticTrainingLabel[]);
const folds = acousticSpeakerFolds(labels);
const referenceText = readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8");
const reference = loadReferenceIndex(JSON.parse(referenceText) as ReferenceAsset);
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
const cache = process.env.ACOUSTIC_RUN_CACHE ?? join(out, "acoustic-run-decodes");
mkdirSync(cache, { recursive: true });
const clips: { speaker: string; fold: number; written: string[]; starts: number[]; evidence: unknown }[] = [];
try {
  for (const label of labels) {
    const audio = resolve(root, label.wav), rel = relative(root, audio);
    if (rel.startsWith("..") || isAbsolute(rel)) throw new Error("Training audio escapes help-train");
    const hash = createHash("sha256").update(readFileSync(audio)).digest("hex");
    const cached = join(cache, `${model.manifest.sha256}-${hash}.json`);
    let evidence: { phonemes: DecodedPhoneme[]; posteriors: { k: number; ids: number[]; probs: number[] }; frameMs: number; nFrames: number };
    if (existsSync(cached)) evidence = JSON.parse(readFileSync(cached, "utf8"));
    else {
      const wav = decodeWav(readFileSync(audio));
      const listener = createListener(model, { posteriors: true, topK: 5 });
      const chunks = [await listener.push(wav.samples, wav.sampleRate), await listener.end()].filter(c => c !== null);
      evidence = { phonemes: chunks.flatMap(c => c.phonemes), frameMs: model.manifest.frameMs, nFrames: listener.decoded,
        posteriors: { k: 5, ids: chunks.flatMap(c => Array.from(c.posteriors!.ids)), probs: chunks.flatMap(c => Array.from(c.posteriors!.probs)) } };
      writeFileSync(cached, JSON.stringify(evidence));
    }
    const ayahs = new Set(label.ayahs);
    const words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !evidence.phonemes.length || words.some((w, i) => i > 0 && words[i - 1].end !== w.start)) {
      throw new Error("Empty or non-contiguous clean training transcript");
    }
    clips.push({ speaker: label.speaker, fold: folds.get(label.speaker)!,
      written: words.map(w => reference.phonemes.slice(w.start, w.end)), starts: words.map(w => w.start), evidence });
    console.log(`Clean training decode ${clips.length}/${labels.length}`);
  }
  const document = { version: 1, trainingOnly: true, inventory: reference.inventoryId, tokens: [...reference.tokens].join(""),
    folds: Math.min(5, folds.size), clips,
    vocabulary: [...new Set(reference.words.map(w => reference.phonemes.slice(w.start, w.end)).filter(Boolean))].sort(),
    provenance: { listener: model.manifest.id, modelSha256: model.manifest.sha256,
      labelsSha256: createHash("sha256").update(text).digest("hex"), referenceSha256: createHash("sha256").update(referenceText).digest("hex"),
      eligibleLabelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
      clips: clips.length, speakers: folds.size,
      isolation: "split=train, none, scored, no mistakes/repeats/exclusion; any non-train speaker excluded; help-train only" } };
  writeFileSync(join(out, "acoustic-run-training.json"), JSON.stringify(document));
  console.log(JSON.stringify(document.provenance));
} finally { await model.backend.release(); }
