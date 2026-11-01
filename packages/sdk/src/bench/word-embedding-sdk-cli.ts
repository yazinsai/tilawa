/** Exercise the shipping word sidecar and free tracker against saved listener evidence.
 * This diagnostic writes decisions only. Saved evidence, audio and labels remain untouched.
 */
import { readFileSync, readdirSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { chunks, oraclePositions } from "../../../core/src/bench/lab.js";
import type { BenchSession, SessionEvidence } from "../../../core/src/bench/sessions.js";
import { confirmGapWords } from "../gap-decode.js";
import { loadNodeModel, readWav } from "../node.js";

const { values } = parseArgs({ options: {
  sessions: { type: "string" }, listener: { type: "string" }, base: { type: "string" },
  model: { type: "string" }, out: { type: "string" }, variants: { type: "string", default: "phone,studio" },
} });
if (!values.sessions || !values.listener || !values.base || !values.model || !values.out) throw new Error("--sessions, --listener, --base, --model, --out required");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../../core/assets/phrases-v1.json", import.meta.url), "utf8"));
const shard = Number(process.env.LAB_SHARD ?? 0), shards = Number(process.env.LAB_SHARDS ?? 1);
const sessions = readdirSync(values.sessions).filter(f => f.endsWith(".json") && !f.endsWith(".evidence.json")).sort()
  .map(file => JSON.parse(readFileSync(resolve(values.sessions!, file), "utf8")) as BenchSession)
  .filter(session => session.version === 1 && session.words && session.events);
const model = await loadNodeModel(values.model);
if (!model.wordEmbedding) throw new Error("Model has no word embedding sidecar");
mkdirSync(values.out, { recursive: true });
const flat = (e: SessionEvidence): PhonemeEvidence[] => [{ ...e,
  posteriors: e.posteriors && { ...e.posteriors, startFrame: 0, nFrames: e.nFrames } }];
try {
  for (const [i, session] of sessions.entries()) {
    if (i % shards !== shard) continue;
    for (const variant of values.variants!.split(",")) {
      const read = (listener: string) => JSON.parse(readFileSync(resolve(values.sessions!, `${session.id}.${listener}.${variant}.evidence.json`), "utf8")) as SessionEvidence;
      const base = read(values.base), folded = read(values.listener);
      const wave = await readWav(resolve(values.sessions, `${session.id}.${variant}.wav`));
      const annotated = await confirmGapWords(model, wave.samples, wave.sampleRate, flat(base), flat(folded), index, { phrases });
      const phonemes = annotated.flatMap(e => e.phonemes);
      const evidence = { ...folded, phonemes };
      const checker = createChecker(index, { phrases }), oracle = oraclePositions(session, index);
      const events = chunks(evidence).flatMap(chunk => checker.push(chunk, oracle((chunk.decodedTo ?? 0) * chunk.frameMs)));
      events.push(...checker.end(oracle(Infinity)));
      const freeChecker = createChecker(index, { phrases }), tracker = createTracker(index, { mode: "free", phrases });
      const freeEvents = chunks(evidence).flatMap(chunk => freeChecker.push(chunk, tracker.push({ ...chunk, posteriors: undefined })));
      freeEvents.push(...freeChecker.end(tracker.end()));
      const result = { session: session.id, variant, clean: session.events.length === 0,
        proofs: phonemes.flatMap(p => p.wholeWord ? [p.wholeWord] : []),
        events: events.filter(e => e.errorClass === "added_word"), freeEvents: freeEvents.filter(e => e.errorClass === "added_word"), labels: session.events };
      writeFileSync(resolve(values.out, `${session.id}.${variant}.json`), JSON.stringify(result));
      console.log(JSON.stringify({ session: session.id, variant, proofs: result.proofs.length, clean: result.clean }));
    }
  }
} finally { await model.backend.release(); }
