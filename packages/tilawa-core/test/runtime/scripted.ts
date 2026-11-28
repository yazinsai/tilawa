import { readFileSync } from "node:fs";
import { loadReferenceIndex } from "../../src/engine/index.js";
import type { EngineEvent, PhraseAsset, ReferenceAsset, WordRef } from "../../src/engine/index.js";
import type { InferenceBackend, LoadedModel, Tensor } from "../../src/runtime/model.js";
import { createEngine } from "../../src/runtime/session.js";
import type { SessionPolicy } from "../../src/runtime/session.js";
import { loadNodeModel } from "../../src/runtime/node.js";

const core = (p: string) => new URL(`../../assets/${p}`, import.meta.url);
export const index = loadReferenceIndex(JSON.parse(readFileSync(core("ref-v2.json"), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(core("phrases-v1.json"), "utf8")) as PhraseAsset;
const tiny = new URL("../fixtures/tiny", import.meta.url).pathname;

export const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const wordIndex = new Map(index.words.map((x, i) => [key(x), i]));
const wordAt = (r: WordRef) => wordIndex.get(key(r))!;
/** Model frames (40 ms each) in one answer of the scripted model. */
export const ANSWER_FRAMES = 8;

/**
 * A model that hears what it is told: each call returns `ANSWER_FRAMES` output frames (320 ms) of a script of token
 * ids, one id per frame (0 is blank), as log-probabilities with 0.9 on the scripted token.
 */
function scripted(model: LoadedModel, ids: number[]): LoadedModel {
  const vocab = model.manifest.tokens.length + 1;
  let t = 0;
  const backend: InferenceBackend = {
    name: "scripted",
    async run(feeds: Record<string, Tensor>) {
      const rows = ANSWER_FRAMES;
      const lp = new Float32Array(rows * vocab).fill(Math.log(0.1 / (vocab - 1)));
      for (let r = 0; r < rows; r++, t++) lp[r * vocab + (ids[t] ?? 0)] = Math.log(0.9);
      const out: Record<string, Tensor> = { log_probs: { type: "float32", data: lp, dims: [1, rows, vocab] } };
      for (const s of model.manifest.states) out[`new_${s.name}`] = feeds[s.name];
      return out;
    },
    async release() {},
  };
  return { ...model, backend };
}

export interface Shown {
  /** When the word was said (its first phoneme, ms of audio). */
  said: number;
  /** The evidence time of the first position that reached it, and the audio clock at the push that showed it. */
  evidence: number;
  shown: number;
}

export interface ScriptedTake {
  /** Audio pushed at a time (ms). */
  pushMs: number;
  policy?: SessionPolicy;
  surah: number;
  ayahs: number[];
  /** Words left out, as `surah:ayah:word`. */
  skip?: string[];
  /** Words said as other phonemes, by `surah:ayah:word`. */
  change?: Record<string, string>;
  /** Frames of silence before the first ayah (10) and after each ayah (15). */
  lead?: number;
  gap?: number;
  /** Frames of silence after words, by `surah:ayah:word` (a pause inside an ayah). */
  pauseAfter?: Record<string, number>;
}

/**
 * A memorization session of ayahs of `surah` said fast by the scripted model, a phoneme a frame (40 ms; a blank
 * between a phoneme and its repeat), so that one model answer often completes two words, with silence between ayahs;
 * pushed `pushMs` of audio at a time: when each word was said and first shown, every event, the words first shown at
 * each push, and after each push the ids of `session.mistakes()` and of the confirmations shown so far.
 */
export async function scriptedTake(o: ScriptedTake) {
  const tiny0 = await loadNodeModel(tiny);
  const tokens = [...tiny0.manifest.tokens];
  const ids: number[] = [...Array(o.lead ?? 10).fill(0)];
  const said = new Map<string, number>();
  for (const a of o.ayahs) {
    for (const x of index.words.filter((x) => x.surah === o.surah && x.ayah === a && x.end > x.start)) {
      if (o.skip?.includes(key(x))) continue;
      said.set(key(x), ids.length * 40);
      for (const p of o.change?.[key(x)] ?? index.phonemes.slice(x.start, x.end)) {
        const id = tokens.indexOf(p) + 1;
        ids.push(...(ids.at(-1) === id ? [0, id] : [id]));
      }
      ids.push(...Array(o.pauseAfter?.[key(x)] ?? 0).fill(0));
    }
    ids.push(...Array(o.gap ?? 15).fill(0));
  }
  ids.push(...Array(40).fill(0));
  const samples = new Float32Array(Math.ceil((ids.length * 40 + 400) * 16));
  const engine = createEngine({ reference: index, phrases, model: scripted(tiny0, ids) });
  const s = engine.startMemorization({ surah: o.surah, fromAyah: o.ayahs[0], toAyah: o.ayahs.at(-1)! }, o.policy);
  const chunk = o.pushMs * 16;
  const words = new Map<string, Shown>();
  const events: EngineEvent[] = [];
  const pushes: string[][] = [];
  const listed: string[][] = [];
  const confirmed: string[][] = [];
  const note = (es: EngineEvent[], clock: number) => {
    events.push(...es);
    listed.push(s.mistakes().map((e) => e.id));
    confirmed.push(events.flatMap((e) => (e.type === "checker" && e.state === "confirmed" ? [e.id] : [])));
    const fresh: string[] = [];
    for (const e of es) {
      if (e.type !== "position" || e.state !== "locked") continue;
      // Each locked position shows its run, from the run's start to its word.
      for (let i = wordAt(e.observed[0].span.start); i <= wordAt(e.observed[0].word); i++) {
        const k = key(index.words[i]);
        if (index.words[i].end === index.words[i].start || words.has(k)) continue;
        words.set(k, { said: said.get(k)!, evidence: e.timeMs, shown: clock });
        fresh.push(k);
      }
    }
    pushes.push(fresh);
  };
  for (let a = 0; a < samples.length; a += chunk) note(await s.pushAudio(samples.subarray(a, a + chunk), 16000), Math.min(a + chunk, samples.length) / 16);
  note(await s.end(), samples.length / 16);
  return { words, events, pushes, listed, confirmed };
}
