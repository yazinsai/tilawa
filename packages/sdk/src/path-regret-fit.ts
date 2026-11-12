/** Training-only, deterministic uncertainty-set fit. Run before opening practice comparisons. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { EDIT_KINDS, fitAcousticCosts, trainingCostScale, trainingEditCounts } from "../../core/dist/path-regret.js";
import type { EditTrainingCounts, PathUncertaintySet } from "../../core/dist/path-regret.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";
import { regretSpeakerPartitions, regretTrainingLabels } from "./path-regret-training.js";
import type { RegretTrainingLabel } from "./path-regret-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const labels = regretTrainingLabels(readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim()
  .split("\n").map(x => JSON.parse(x)) as RegretTrainingLabel[]);
if (new Set(labels.map(l => l.id)).size !== labels.length) throw new Error("Duplicate training recording IDs");
const partitions = regretSpeakerPartitions(labels);
const partitionCount = Math.min(5, partitions.size);
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const cache = process.env.PATH_REGRET_CACHE ?? join(out, "training-decodes");
mkdirSync(cache, { recursive: true });
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips: { speaker: string; counts: EditTrainingCounts }[] = [];
  for (const l of labels) {
    // Hash IDs so paths cannot escape the cache; cached observations belong to this exact listener.
    const key = createHash("sha256").update(l.id).digest("hex");
    const file = join(cache, `${model.manifest.sha256}-${key}.json`);
    let phones: DecodedPhoneme[];
    if (existsSync(file)) phones = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(readFileSync(join(data, "help-train", l.wav)));
      const listener = createListener(model);
      phones = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
      writeFileSync(file, JSON.stringify(phones));
    }
    const ayahs = new Set(l.ayahs), words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !phones.length) throw new Error("Empty eligible training clip");
    if (words.some((w, i) => i > 0 && words[i - 1].end !== w.start)) throw new Error("Non-contiguous training transcript");
    const written = reference.phonemes.slice(words[0].start, words.at(-1)!.end);
    clips.push({ speaker: l.speaker, counts: trainingEditCounts({ written, heard: phones.map(p => p.p).join("") }) });
    console.log(`Decoded none training recording ${clips.length}/${labels.length}`);
  }
  const counts = clips.map(c => c.counts), pooled = fitAcousticCosts(counts);
  const unitsPerNat = trainingCostScale(pooled, counts);
  const partitionReports = Array.from({ length: partitionCount }, (_, f) => {
    const subset = clips.filter(c => partitions.get(c.speaker) === f);
    return { clips: subset.length, speakers: new Set(subset.map(c => c.speaker)).size,
      costs: fitAcousticCosts(subset.map(c => c.counts)).map(c => c * unitsPerNat) };
  });
  const uncertainty: PathUncertaintySet = { models: [
    { costs: pooled.map(c => c * unitsPerNat) }, ...partitionReports.map(p => ({ costs: p.costs })),
  ] };
  const report = { trainingOnly: true, listener: model.manifest.id, modelSha256: model.manifest.sha256,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
    clips: labels.length, speakers: partitions.size, kinds: EDIT_KINDS, unitsPerNat,
    coverage: "100% of disjoint speaker-partition estimates plus the pooled estimate; convex hull, no independent extremes",
    estimator: "Jeffreys-smoothed error-versus-match probabilities; one pooled training least-squares unit conversion",
    alignment: "Full-transcript minimum existing weighted edits, reference to heard; fixed conditional alignments",
    partitionReports, trainingCounts: counts.reduce((out, c) => ({ matches: out.matches + c.matches,
      counts: out.counts.map((n, j) => n + c.counts[j]) }), { matches: 0, counts: Array(EDIT_KINDS.length).fill(0) as number[] }) };
  writeFileSync(join(out, "path-uncertainty.json"), JSON.stringify(uncertainty, null, 2));
  writeFileSync(join(out, "path-uncertainty.ts"), `import type { PathUncertaintySet } from "./path-regret.js";\n\n/** Frozen training-only fit; provenance in experiment write-up. */\nexport const PATH_UNCERTAINTY: PathUncertaintySet = ${JSON.stringify(uncertainty)};\n`);
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally {
  await model.backend.release();
}
