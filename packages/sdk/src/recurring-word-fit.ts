/** Training-only command. Listener inference and fitting run on Modal, never on the development Mac. */
import { createHash } from "node:crypto";
import { readFileSync, realpathSync, writeFileSync } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";
import { createTracker, fitOccurrenceCalibration, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { TrainingOccurrence } from "@tilawa/core";
import { createListener } from "./listener.js";
import { decodeWav, loadNodeModel } from "./node.js";
import { recurringWordTrainingLabels } from "./recurring-word-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const root = realpathSync(resolve(data, "help-train"));
const labelsText = readFileSync(join(root, "labels.jsonl"), "utf8");
const labels = recurringWordTrainingLabels(labelsText.trim().split("\n").filter(Boolean).map(l => JSON.parse(l)));
const referenceText = readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8");
const index = loadReferenceIndex(JSON.parse(referenceText));
const phrases = JSON.parse(readFileSync(new URL("../../core/assets/phrases-v1.json", import.meta.url), "utf8"));
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
const occurrences: TrainingOccurrence[] = [];
const audioHashes: string[] = [];
try {
  for (const label of labels) {
    const path = realpathSync(resolve(root, label.wav)), rel = relative(root, path);
    if (rel === ".." || rel.startsWith(`../`) || isAbsolute(rel)) throw new Error("Audio path escapes authorized help-train");
    const bytes = readFileSync(path), wav = decodeWav(bytes);
    const hash = createHash("sha256").update(bytes).digest("hex");
    audioHashes.push(hash);
    const tracker = createTracker(index, { phrases, acousticMemory: { audit: true, occurrence: occurrence => {
      occurrences.push({ speaker: label.speaker, recording: hash, pronunciation: occurrence.pronunciation, trajectory: occurrence.trajectory,
        fromFrame: occurrence.fromFrame, toFrame: occurrence.toFrame });
    } } });
    const listener = createListener(model, { posteriors: true, topK: 5 });
    // Feed actual finalized streaming chunks; a one-push clip must not bypass tracker admission.
    const width = Math.max(1, Math.round(wav.sampleRate * 0.04));
    for (let start = 0; start < wav.samples.length; start += width) {
      const chunk = await listener.push(wav.samples.slice(start, start + width), wav.sampleRate);
      if (chunk) tracker.push(trackerInput(chunk));
    }
    const tail = await listener.end();
    if (tail) tracker.push(trackerInput(tail));
    tracker.end();
    console.log(`Training takes ${audioHashes.length}/${labels.length}; bracketed renditions ${occurrences.length}`);
  }
  const fitted = fitOccurrenceCalibration(occurrences, { trainingOnly: true });
  writeFileSync(join(out, "recurring-word-calibration.json"), JSON.stringify({ ...fitted,
    provenance: { listener: model.manifest.id, modelSha256: model.manifest.sha256,
      labelsSha256: createHash("sha256").update(labelsText).digest("hex"),
      selectedLabelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
      referenceSha256: createHash("sha256").update(referenceText).digest("hex"),
      audioHashes, occurrences: occurrences.length,
      admission: "train speakers only; none/repeat, scored, no mistake events or exclusions; actual tracker margin and in-order bracketing",
    } }, null, 2));
  console.log(JSON.stringify(fitted));
} finally { await model.backend.release(); }
