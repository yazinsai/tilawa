/** Offline clean-text calibration. Selection and paths are supplied by the
 * caller; no evaluation identity or labelled mistake enters the asset. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { parseArgs } from "node:util";
import { createChecker } from "../../core/src/checker.js";
import { createTracker, trackerInput } from "../../core/src/tracker.js";
import { loadReferenceIndex } from "../../core/src/reference.js";
import { seamTextKey } from "../../core/src/seam-replay.js";
import { chunk, type QudRecording } from "../../core/src/bench/index.js";
import { loadNodeModel, readWav } from "./node.js";
import { createEngine } from "./session.js";

const { values } = parseArgs({ options: {
  resume: { type: "boolean" }, jobs: { type: "string" }, model: { type: "string" }, out: { type: "string" },
} });
if (!values.jobs || !values.model || !values.out) throw new Error("--jobs, --model, --out required");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../core/assets/phrases-v1.json", import.meta.url), "utf8"));
const jobs = JSON.parse(readFileSync(values.jobs, "utf8"));
const model = await loadNodeModel(values.model, { threads: 1 });
const engine = createEngine({ reference: index, phrases, model, recordEvidence: true });
const fingerprint = createHash("sha256");
fingerprint.update(readFileSync(values.jobs));
for (const path of ["../../core/src/checker.ts", "../../core/src/checker-lexical.ts", "../../core/src/ctc.ts",
  "../../core/src/tracker.ts", "../../core/src/seam-excision.ts", "../../core/src/seam-replay.ts",
  "./listener.ts", "./gap-decode.ts", "./session.ts", "./seam-replay-build.ts"]) {
  fingerprint.update(readFileSync(new URL(path, import.meta.url)));
}
const source = fingerprint.digest("hex");
const shard = Number(process.env.LAB_SHARD ?? 0), shards = Number(process.env.LAB_SHARDS ?? 1);
mkdirSync(values.out, { recursive: true });
for (let ji = shard; ji < jobs.length; ji += shards) {
  const job = jobs[ji];
  const destination = `${values.out}/${ji}.json`;
  if (values.resume && existsSync(destination)) {
    const cached = JSON.parse(readFileSync(destination, "utf8"));
    if (cached.source === source && cached.model === model.manifest.sha256 && cached.reader === job.reader &&
      cached.context === job.location) { console.log("cached", ji); continue; }
  }
  if (!existsSync(job.audio)) { console.log("unavailable", job.reader, job.location); continue; }
  const wav = await readWav(job.audio);
  const session = engine.startFreeRecitation();
  for (let at = 0; at < wav.samples.length; at += 320) await session.pushAudio(wav.samples.subarray(at, at + 320), wav.sampleRate);
  await session.end();
  const evidence = session.evidence();
  const post = evidence.flatMap(e => e.posteriors ? [e.posteriors] : []);
  const flat = { inventory: index.inventoryId, frameMs: model.manifest.frameMs,
    phonemes: evidence.flatMap(e => e.phonemes),
    nFrames: post.reduce((n, p) => n + p.nFrames, 0),
    posteriors: { k: post[0].k, ids: post.flatMap(p => Array.from(p.ids)), probs: post.flatMap(p => Array.from(p.probs)) } };
  // Exactly the ordinary free engine, on the shipped recorded gap evidence.
  // An empty veto asset suppresses only the extra path while its observer
  // records all ownership-free spellings before that veto or certification.
  const wanted = new Set<number>(job.seams);
  const seen = new Map<number, Set<string>>();
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases, seamReplays: { reference: "", inventory: "", seams: {} },
    onSeamSpelling(word, spelling) {
      if (!wanted.has(word)) return;
      if (!seen.has(word)) seen.set(word, new Set());
      if (spelling) seen.get(word)!.add(spelling);
    } });
  for (const e of chunk(flat as QudRecording, 0.04, true)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const seams = [...seen].map(([word, spellings]) => ({ word, key: seamTextKey(index, word), words: [...spellings].sort() })).filter(s => !!s.key);
  writeFileSync(destination, JSON.stringify({ source, reference: index.version, inventory: index.inventoryId,
    reader: job.reader, context: job.location, model: model.manifest.sha256, seams }));
  console.log(JSON.stringify({ job: ji, reader: job.reader, inspectedSeams: seams.length }));
}
