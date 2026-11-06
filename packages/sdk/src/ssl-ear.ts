import type { DecodedPhoneme, PhonemeEvidence, ReferenceIndex } from "@tilawa/core";
import { lexicalWords, minGapFrames } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { topK } from "./listener.js";
import type { InferenceBackend } from "./model.js";

export interface SslEarGeometry {
  /** Native SSL frames are 20 ms; auxiliary timestamps retain that clock. */
  chunk: number;
  left: number;
  right: number;
}

export interface SslEarSpan {
  /** Half-open bounds on the auxiliary's native 20 ms clock. */
  fromFrame: number;
  toFrame: number;
}

export interface SslSeamSpan extends SslEarSpan {
  word: number;
  pauses: SslEarSpan[];
}

/** Whole pause-bounded utterances for the folded spelling test. Never search
 * inside an utterance: a lexical suffix of a misheard word is not extra speech.
 * The native ear labels pauses by the folded blank-probability/minimum-length
 * predicate, with no threshold fitted to these recordings. */
export function sslSeamSpellings(evidence: PhonemeEvidence, spans: readonly SslSeamSpan[], index: ReferenceIndex) {
  if (evidence.frameMs !== 20 || evidence.inventory !== index.inventoryId || !evidence.posteriors)
    throw new Error("SSL spelling requires native evidence with matching inventory and posteriors");
  const post = evidence.posteriors, lexical = lexicalWords(index);
  const pause = (a: number, b: number) => {
    if (b - a < minGapFrames(evidence.frameMs) || a < post.startFrame || b > post.startFrame + post.nFrames) return false;
    for (let f = a; f < b; f++) {
      const row = (f - post.startFrame) * post.k;
      if (post.ids[row] !== 0 || post.probs[row] < 0.5) return false;
    }
    return true;
  };
  const edge = new Set("ۦۥں۾ڇؙۜ");
  const result: { word: number; said: string; fromFrame: number; toFrame: number; confidence: number }[] = [];
  for (const span of spans) {
    const phones = evidence.phonemes.filter(p => p.start >= span.fromFrame && p.end < span.toFrame);
    let from = 0;
    const inspect = (to: number) => {
      let a = from, b = to;
      while (a < b && edge.has(phones[a].p)) a++;
      while (b > a && edge.has(phones[b - 1].p)) b--;
      const kept = phones.slice(a, b), said = kept.map(p => p.p).join("");
      if (!said || !lexical.surplusWord(span.word, said) || lexical.fragmentOfSeam(span.word, said)) return;
      result.push({ word: span.word, said, fromFrame: kept[0].start, toFrame: kept.at(-1)!.end + 1,
        confidence: kept.reduce((n, p) => n + p.conf, 0) / kept.length });
    };
    for (let i = 0; i + 1 < phones.length; i++) if (pause(phones[i].end + 1, phones[i + 1].start)) {
      inspect(i + 1);
      from = i + 1;
    }
    inspect(phones.length);
  }
  return result;
}

/** Read-only seam placement from primary committed steps and labelled blank
 * pauses. A local backstep never replaces the original entry into the seam.
 * Returned bounds use the auxiliary 20 ms clock, without greedy word fits. */
export function sslSeamSpans(primary: PhonemeEvidence, locks: readonly { wi: number; timeMs: number }[],
  isSeam: (word: number) => boolean): SslSeamSpan[] {
  if (!primary.posteriors) return [];
  const post = primary.posteriors, out: SslSeamSpan[] = [];
  const clock = primary.frameMs / 20;
  const seen = new Set<string>();
  for (let k = 0; k < locks.length; k++) {
    const word = locks[k].wi - 1, previous = word - 1;
    if (previous < 0 || !isSeam(word)) continue;
    let left = -1;
    for (let j = k - 1; j >= 0; j--) {
      if (locks[j].wi < previous || locks[j].wi >= locks[k].wi) break;
      if (locks[j].wi === previous) left = j;
    }
    if (left < 0) continue;
    const from = Math.round(locks[left].timeMs / primary.frameMs);
    const to = Math.round(locks[k].timeMs / primary.frameMs);
    const pauses: SslEarSpan[] = [];
    for (let j = 0; j + 1 < primary.phonemes.length; j++) {
      const a = primary.phonemes[j].end + 1, b = primary.phonemes[j + 1].start;
      if (a < from || b > to || b - a < minGapFrames(primary.frameMs)) continue;
      let blank = a >= post.startFrame && b <= post.startFrame + post.nFrames;
      for (let f = a; blank && f < b; f++) {
        const row = (f - post.startFrame) * post.k;
        blank = post.ids[row] === 0 && post.probs[row] >= 0.5;
      }
      if (blank) pauses.push({ fromFrame: Math.round(a * clock), toFrame: Math.round(b * clock) });
    }
    if (!pauses.length) continue;
    const span = { word, fromFrame: pauses[0].fromFrame, toFrame: Math.round(to * clock), pauses };
    const key = `${word}:${span.fromFrame}:${span.toFrame}`;
    if (!seen.has(key)) { out.push(span); seen.add(key); }
  }
  return out;
}

/** The same rolling-window normalization used during plain CTC training.
 * Before-start and after-end padding is excluded from both moments. */
export function sslWindow(frames: Float32Array[], start: number, geometry: SslEarGeometry) {
  const size = geometry.left + geometry.chunk + geometry.right;
  const from = 2 * (start - geometry.left), end = 2 * (start + geometry.chunk + geometry.right);
  const data = new Float32Array(size * 160), mask = new BigInt64Array(size);
  const lo = Math.max(0, from), hi = Math.min(frames.length, end);
  const n = hi - lo;
  if (n < 2) throw new Error("SSL window requires at least two feature frames");
  for (let m = 0; m < 80; m++) {
    let mean = 0, variance = 0;
    for (let t = lo; t < hi; t++) mean += frames[t][m];
    mean /= n;
    for (let t = lo; t < hi; t++) variance += (frames[t][m] - mean) ** 2;
    const scale = Math.sqrt(variance / (n - 1) + 1e-7);
    for (let t = lo; t < hi; t++) data[(t - from) * 80 + m] = (frames[t][m] - mean) / scale;
  }
  for (let t = lo; t < hi; t += 2) mask[Math.floor((t - from) / 2)] = 1n;
  return { data, mask, size };
}

/** Full-session bounded streaming decode; no text or seam times enter inference.
 * The consumer later selects seam spans on the primary tracker's clock. */
export function createSslEar(backend: InferenceBackend, tokens: string, inventory: string, geometry: SslEarGeometry) {
  if (geometry.chunk <= 0 || geometry.left < 0 || geometry.right < 0 ||
      ![geometry.chunk, geometry.left, geometry.right].every(Number.isInteger))
    throw new Error("Invalid SSL streaming geometry");
  async function infer(samples: Float32Array, k: number, spans?: readonly SslEarSpan[]): Promise<PhonemeEvidence> {
      if (!Number.isInteger(k) || k < 1 || k > tokens.length + 1) throw new Error("Invalid SSL posterior width");
      if (spans?.some(s => !Number.isInteger(s.fromFrame) || !Number.isInteger(s.toFrame) ||
          s.fromFrame < 0 || s.toFrame <= s.fromFrame)) throw new Error("Invalid SSL seam span");
      const raw = new Fbank({ highFrequency: 8000 }).push(Float32Array.from(samples, x => x * 32768));
      if (raw.length < 2) return { inventory, frameMs: 20, phonemes: [] };
      if (raw.length % 2) raw.push(raw.at(-1)!);
      const phonemes: DecodedPhoneme[] = [], ids: number[] = [], probs: number[] = [];
      const vocab = tokens.length + 1, total = raw.length / 2;
      let previous = 0;
      for (let start = 0; start < raw.length / 2; start += geometry.chunk) {
        if (spans && !spans.some(s => s.fromFrame < start + geometry.chunk && s.toFrame > start)) {
          previous = 0;
          continue;
        }
        const w = sslWindow(raw, start, geometry);
        const output = await backend.run({
          input_features: { type: "float32", data: w.data, dims: [1, w.size, 160] },
          attention_mask: { type: "int64", data: w.mask, dims: [1, w.size] },
        });
        const lp = output.log_probs.data as Float32Array;
        if (lp.length !== geometry.chunk * vocab) throw new Error("SSL ear output differs from its declared geometry or inventory");
        for (let t = 0; t < geometry.chunk && start + t < total; t++) {
          const frame = start + t;
          if (spans && !spans.some(s => s.fromFrame <= frame && frame < s.toFrame)) {
            previous = 0;
            continue;
          }
          const top = topK(lp.subarray(t * vocab, (t + 1) * vocab), k);
          if (!spans) for (const [id, p] of top) ids.push(id), probs.push(p);
          const [id, conf] = top[0];
          if (id && id === previous) {
            phonemes.at(-1)!.end = frame;
            phonemes.at(-1)!.conf = Math.max(phonemes.at(-1)!.conf, conf);
          } else if (id) phonemes.push({ p: tokens[id - 1], start: frame, end: frame, conf });
          previous = id;
        }
      }
      return { inventory, frameMs: 20, phonemes, decodedTo: total,
        ...(!spans ? { posteriors: { k, startFrame: 0, nFrames: total, ids, probs } } : {}) };
  }
  return {
    decode(samples: Float32Array, k: number): Promise<PhonemeEvidence> {
      return infer(samples, k);
    },
    /** Retain the full-session chunk phase and real left/lookahead context.
     * Sparse tokens have global timestamps; never pretend their rows form a
     * contiguous posterior sequence. Primary tracker bounds select only which
     * chunks run and which auxiliary tokens are returned. */
    async decodeSeams(samples: Float32Array, spans: readonly SslEarSpan[]): Promise<DecodedPhoneme[]> {
      return (await infer(samples, 1, spans)).phonemes;
    },
  };
}
