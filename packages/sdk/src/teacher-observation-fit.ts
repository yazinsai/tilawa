/** Training-only entry: paired decodes of authorized help-train audio; no practice inputs or teacher at tracking. */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { join, relative, isAbsolute } from "node:path";
import { createTeacherObservationScorer, fitTeacherObservationModels, loadReferenceIndex } from "@tilawa/core";
import type { PairedPhoneRecording, ReferenceAsset } from "@tilawa/core";
import { createListener } from "./listener.js";
import { decodeWav, loadNodeModel } from "./node.js";
import { MODEL_REGISTRY } from "./registry.js";
import { teacherObservationFolds, teacherObservationReference, teacherObservationTrainingLabels } from "./teacher-observation-training.js";
import type { TeacherObservationLabel } from "./teacher-observation-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const root = realpathSync(join(data, "help-train"));
const labels = teacherObservationTrainingLabels(readFileSync(join(root, "labels.jsonl"), "utf8").trim().split("\n").map(s => JSON.parse(s)) as TeacherObservationLabel[]);
const groups = teacherObservationFolds(labels), folds = Math.max(...groups.values()) + 1;
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const cache = process.env.TEACHER_OBSERVATION_CACHE ?? join(out, "paired-decodes");
mkdirSync(cache, { recursive: true });
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  if (!model.insertionTeacher) throw new Error("Declared fp32 teacher is missing from the authorized listener bundle");
  const teacher = MODEL_REGISTRY.auxiliary?.find(a => a.purpose === "seam-insertion");
  if (!teacher) throw new Error("Teacher provenance missing");
  const clips: { speaker: string; record: PairedPhoneRecording; audioHash: string }[] = [];
  for (const label of labels) {
    const audioPath = realpathSync(join(root, label.wav)), rel = relative(root, audioPath);
    if (rel === ".." || rel.startsWith("../") || isAbsolute(rel)) throw new Error("Training audio escapes help-train");
    const audio = readFileSync(audioPath), audioHash = createHash("sha256").update(audio).digest("hex");
    const key = createHash("sha256").update(JSON.stringify({ audioHash, student: model.manifest,
      teacher, decoder: "whole-help-take-primary-stream-and-fp32-teacher-v1" })).digest("hex");
    const file = join(cache, `${key}.json`);
    let paired: { teacher: string; student: string };
    if (existsSync(file)) paired = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(audio);
      if (wav.sampleRate !== 16000 || model.manifest.sampleRate !== 16000) throw new Error("Paired teacher fitting requires 16 kHz training audio");
      const listener = createListener(model);
      const first = await listener.push(wav.samples, wav.sampleRate), last = await listener.end();
      const teacherEvidence = await model.insertionTeacher.decode(wav.samples, 1);
      paired = { student: [...first?.phonemes ?? [], ...last?.phonemes ?? []].map(p => p.p).join(""),
        teacher: teacherEvidence.phonemes.map(p => p.p).join("") };
      writeFileSync(file, JSON.stringify(paired));
    }
    clips.push({ speaker: label.speaker, audioHash, record: { written: teacherObservationReference(reference, label), ...paired } });
    console.log(`Paired training take ${clips.length}/${labels.length}`);
  }
  const iterations = 16, phones = [...reference.tokens];
  const selection: { mass: number; mediated: number[]; direct: number[]; combined: number }[] = [];
  for (const mass of [8, 32]) {
    const mediated: number[] = [], direct: number[] = [];
    for (let fold = 0; fold < folds; fold++) {
      const train = clips.filter(c => groups.get(c.speaker) !== fold), held = clips.filter(c => groups.get(c.speaker) === fold);
      const fitted = fitTeacherObservationModels(train.map(c => c.record), phones, mass, iterations);
      const ms = createTeacherObservationScorer(fitted.mediated), ds = createTeacherObservationScorer(fitted.direct);
      mediated.push(held.reduce((v, c) => v + ms.logP(c.record.written, c.record.student), 0));
      direct.push(held.reduce((v, c) => v + ds.logP(c.record.written, c.record.student), 0));
      if (![mediated.at(-1), direct.at(-1)].every(Number.isFinite)) throw new Error("Validation recording outside training-frozen fragment support");
      console.log(JSON.stringify({ mass, fold, mediated: mediated.at(-1), direct: direct.at(-1) }));
    }
    selection.push({ mass, mediated, direct, combined: [...mediated, ...direct].reduce((a, b) => a + b, 0) });
  }
  selection.sort((a, b) => b.combined - a.combined || b.mass - a.mass);
  const fitted = fitTeacherObservationModels(clips.map(c => c.record), phones, selection[0].mass, iterations);
  const report = { trainingOnly: true, recordings: clips.length, speakers: groups.size, folds,
    partitions: Object.fromEntries(groups), listener: model.manifest, teacher,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
    audioSha256: clips.map(c => c.audioHash), selection, selectedMass: selection[0].mass, iterations,
    parametersPerModel: fitted.parameters,
    eligibility: "scored none-labelled train takes without extra mistakes/repeats; every mixed/nontrain speaker excluded",
    control: "same fragment support, two tables, pooling, iteration budget and validation speakers; latent stages fit directly to written/student pairs",
    inference: "teacher/direct likelihood residual in existing three-nats-per-edit units; original movement and duration charges retained",
    limits: "single-ayah supervision; bounded output fragments; canonical references and both decodes remain fallible" };
  writeFileSync(join(out, "teacher-observation-models.json"), JSON.stringify({ mediated: fitted.mediated, direct: fitted.direct }));
  writeFileSync(join(out, "teacher-observation-fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await model.backend.release(); }
