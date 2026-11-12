/** Authorized training-only entry point; does not open practice data. */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme } from "@tilawa/core";
import { createLocalSearch } from "../../core/dist/local.js";
import { fitRestartChannel, restartRecordingLogP } from "../../core/dist/sequential-restart.js";
import { restartTrainingLabels, restartSpeakerFolds } from "./restart-training.js";
import type { RestartTrainingLabel } from "./restart-training.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const labels = restartTrainingLabels(readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim()
  .split("\n").map(x => JSON.parse(x)) as RestartTrainingLabel[]);
const foldOf = restartSpeakerFolds(labels), folds = Math.min(5, foldOf.size);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const cache = process.env.RESTART_CACHE ?? join(out, "clean-decodes");
mkdirSync(cache, { recursive: true });
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips = [];
  for (const label of labels) {
    const file = join(cache, `${model.manifest.sha256}-${label.id}.json`);
    let phones: DecodedPhoneme[];
    if (existsSync(file)) phones = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(readFileSync(join(data, "help-train", label.wav))), listener = createListener(model);
      phones = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
      writeFileSync(file, JSON.stringify(phones));
    }
    const ayahs = new Set(label.ayahs), words = index.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !phones.length || words.some((w, i) => i > 0 && words[i - 1].end !== w.start))
      throw new Error("Empty or non-contiguous eligible training recording");
    clips.push({ speaker: label.speaker, offset: words[0].start,
      record: { written: index.phonemes.slice(words[0].start, words.at(-1)!.end), heard: phones.map(p => p.p).join("") } });
    console.log(`Decoded eligible none clip ${clips.length}/${labels.length}`);
  }
  const selection = [];
  for (const strength of [1, 4, 16, 64]) {
    const predictive = [];
    for (let f = 0; f < folds; f++) {
      const channel = fitRestartChannel(clips.filter(c => foldOf.get(c.speaker) !== f).map(c => c.record), [...index.tokens], strength);
      const held = clips.filter(c => foldOf.get(c.speaker) === f);
      predictive.push(held.reduce((s, c) => s + restartRecordingLogP(channel, c.record), 0)
        / held.reduce((s, c) => s + c.record.heard.length, 0));
    }
    selection.push({ strength, predictive, mean: predictive.reduce((a, b) => a + b, 0) / folds });
  }
  selection.sort((a, b) => b.mean - a.mean || b.strength - a.strength);
  const selected = selection[0];
  const maxima: number[] = [];
  for (let f = 0; f < folds; f++) {
    const channel = fitRestartChannel(clips.filter(c => foldOf.get(c.speaker) !== f).map(c => c.record), [...index.tokens], selected.strength);
    for (const clip of clips.filter(c => foldOf.get(c.speaker) === f)) {
      let maximum = 0;
      const search = createLocalSearch(index, { offset: clip.offset, origin: clip.offset, edits: 0 }, {
        calibration: { channel, threshold: Number.MAX_VALUE },
        onComparison: c => { if (Number.isFinite(c.logOdds)) maximum = Math.max(maximum, c.logOdds); },
      });
      for (const p of clip.record.heard) search.step(p);
      maxima.push(maximum);
    }
  }
  const threshold = Math.max(0, ...maxima);
  const calibration = { channel: fitRestartChannel(clips.map(c => c.record), [...index.tokens], selected.strength), threshold };
  const report = { trainingOnly: true, listener: model.manifest.id, modelSha256: model.manifest.sha256,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
    clips: clips.length, speakers: foldOf.size, folds, selection, selected, threshold, maxima,
    criterion: "Speaker-separated clean prefix predictive likelihood; commit threshold is the maximum OOF clean-stream destination log odds, floored at zero",
    limitations: "Channel counts use directed Viterbi training alignments; single-ayah clean clips do not measure true passage-jump sensitivity or full-session family-wise error." };
  writeFileSync(join(out, "restart-calibration.json"), JSON.stringify(calibration));
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await model.backend.release(); }
