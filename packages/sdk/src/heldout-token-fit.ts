/** Training-only calibration. Never opens practice recordings or release-held-out speakers. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { fitTokenEditModel, tokenPartition, tokenRecordingLogP } from "../../core/dist/heldout-token.js";
import type { TokenRecording } from "../../core/dist/heldout-token.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";
import { tokenSpeakerFolds, tokenTrainingLabels } from "./heldout-token-training.js";
import type { TokenTrainingLabel } from "./heldout-token-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const labels = tokenTrainingLabels(readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim()
  .split("\n").map(x => JSON.parse(x)) as TokenTrainingLabel[]);
const foldOf = tokenSpeakerFolds(labels), folds = Math.min(5, foldOf.size);
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const cache = process.env.TOKEN_VALIDATION_CACHE ?? join(out, "clean-decodes");
mkdirSync(cache, { recursive: true });
const listenerModel = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips: { speaker: string; record: TokenRecording }[] = [];
  for (const l of labels) {
    const file = join(cache, `${listenerModel.manifest.sha256}-${l.id}.json`);
    let phones: DecodedPhoneme[];
    if (existsSync(file)) phones = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(readFileSync(join(data, "help-train", l.wav)));
      const listener = createListener(listenerModel);
      phones = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
      writeFileSync(file, JSON.stringify(phones));
    }
    const ayahs = new Set(l.ayahs), words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !phones.length) throw new Error("Empty eligible training clip");
    if (words.some((w, i) => i > 0 && words[i - 1].end !== w.start)) throw new Error("Non-contiguous training reference");
    clips.push({ speaker: l.speaker, record: { written: reference.phonemes.slice(words[0].start, words.at(-1)!.end), heard: phones.map(p => p.p).join("") } });
    console.log(`Decoded none clip ${clips.length}/${labels.length}`);
  }
  // Geometry and smoothing are selected together on training-speaker folds, then frozen before target replay.
  const selection: { strength: number; block: number; predictive: number; foldPredictive: number[] }[] = [];
  for (const strength of [1, 4, 16, 64]) for (const block of [1, 2, 4, 8]) {
    const foldPredictive: number[] = [];
    for (let f = 0; f < folds; f++) {
      const fitted = fitTokenEditModel(clips.filter(c => foldOf.get(c.speaker) !== f).map(c => c.record), [...reference.tokens], strength, block);
      let sum = 0, tokens = 0;
      for (const c of clips.filter(c => foldOf.get(c.speaker) === f)) {
        const full = tokenRecordingLogP(fitted, c.record);
        for (const retained of [0, 1]) {
          const mask = tokenPartition(c.record.heard.length, block, retained);
          sum += full - tokenRecordingLogP(fitted, c.record, mask);
          tokens += mask.filter(x => !x).length;
        }
      }
      foldPredictive.push(sum / tokens);
    }
    selection.push({ strength, block, foldPredictive, predictive: foldPredictive.reduce((a, b) => a + b, 0) / folds });
    console.log(JSON.stringify(selection.at(-1)));
  }
  selection.sort((a, b) => b.predictive - a.predictive || b.strength - a.strength || a.block - b.block);
  const selected = selection[0];
  const fitted = fitTokenEditModel(clips.map(c => c.record), [...reference.tokens], selected.strength, selected.block);
  const report = { trainingOnly: true, listener: listenerModel.manifest.id, modelSha256: listenerModel.manifest.sha256,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
    clips: clips.length, speakers: foldOf.size, folds, selection, selected,
    criterion: "Equal-speaker-fold held-out identity predictive log likelihood per withheld token, both complementary partitions",
    isolation: "train speakers, scored none takes, no events/repeats/exclusion; every non-train speaker excluded",
    limitations: "Training counts condition on directed minimum-edit alignments; predictive evaluation marginalizes edits. Training geometry selection uses whole single-ayah clips, runtime uses the unchanged 16-token recent-cost interval." };
  writeFileSync(join(out, "heldout-token-model.json"), JSON.stringify(fitted));
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await listenerModel.backend.release(); }
