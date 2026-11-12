import type { DecodedPhoneme, PhonemeEvidence } from "../engine/index.js";
import { Resampler } from "./audio.js";
import { Fbank } from "./fbank.js";
import type { LoadedModel, Tensor } from "./model.js";

export interface ListenerOptions {
  /** Attach per-frame top-k posteriors to each chunk: the checker's local-window check reads them. Default false; sessions turn it on. */
  posteriors?: boolean;
  topK?: number;
}

export interface Listener {
  /** Audio at any sample rate; returns the evidence that became final, or null when none did. */
  push(samples: Float32Array, sampleRate: number): Promise<PhonemeEvidence | null>;
  /** Pads and drains the stream: the rest of the evidence. */
  end(): Promise<PhonemeEvidence | null>;
  /** Fbank frames consumed so far; audio time is `frames * 10 ms`. */
  readonly frames: number;
  /** Output frames before this one are final: every phoneme starting earlier has been returned. */
  readonly decoded: number;
}

const round4 = (x: number) => Math.round(x * 1e4) / 1e4;

/**
 * Streaming phoneme listener: fbank, then the streaming Zipformer one window at a time, then a greedy CTC decode.
 * A chunk covers every frame decoded so far. A token run cut by a window's end is emitted with that window, its end
 * frame where the window ended.
 */
export function createListener(model: LoadedModel, options: ListenerOptions = {}): Listener {
  const m = model.manifest;
  const k = options.topK ?? 5;
  const fbank = new Fbank();
  const vocab = m.tokens.length + 1;
  let resampler: Resampler | null = null;
  let states: Record<string, Tensor> = Object.fromEntries(
    m.states.map((s) => {
      const n = s.shape.reduce((a, b) => a * b, 1);
      return [s.name, { type: s.dtype, data: s.dtype === "int64" ? new BigInt64Array(n) : new Float32Array(n), dims: s.shape }];
    }),
  );
  let feats: Float32Array[] = [];
  let consumed = 0;
  let fbankFrames = 0;
  /** Output frames decoded so far, and the first frame not yet emitted. */
  let outFrames = 0;
  let emitted = 0;
  let prev = 0;
  let open: DecodedPhoneme | null = null;
  /** The run at the end of the previous window was emitted; its continuation is the same phoneme. */
  let carried = false;
  let done: DecodedPhoneme[] = [];
  let ids: number[] = [];
  let probs: number[] = [];

  async function step(window: Float32Array[], keep: number) {
    const x = new Float32Array(m.window * m.fbank.mels);
    window.forEach((f, i) => x.set(f, i * m.fbank.mels));
    const out = await model.backend.run({ x: { type: "float32", data: x, dims: [1, m.window, m.fbank.mels] }, ...states });
    states = Object.fromEntries(m.states.map((s) => [s.name, out[`new_${s.name}`]]));
    const lp = out.log_probs.data as Float32Array;
    const n = Math.min(keep, lp.length / vocab);
    for (let t = 0; t < n; t++) {
      const row = lp.subarray(t * vocab, (t + 1) * vocab);
      const top = topK(row, k);
      if (options.posteriors) for (const [id, p] of top) ids.push(id), probs.push(round4(p));
      const [id, p] = top[0];
      const frame = outFrames + t;
      if (id !== 0 && id === prev && open) {
        open.end = frame;
        open.conf = Math.max(open.conf, round4(p));
      } else if (!(id !== 0 && id === prev && carried)) {
        if (open) done.push(open);
        open = id !== 0 ? { p: m.tokens[id - 1], start: frame, end: frame, conf: round4(p) } : null;
        carried = false;
      }
      if (id !== prev) carried = false;
      prev = id;
    }
    outFrames += n;
    // A run still open at the window's end goes out with it: continuing it in the next window can't change the token.
    if (open) {
      done.push(open);
      open = null;
      carried = true;
    }
  }

  function flush(final: boolean): PhonemeEvidence | null {
    if (final && open) {
      done.push(open);
      open = null;
    }
    const until = open ? open.start : outFrames;
    if (until <= emitted && !done.length) return null;
    const e: PhonemeEvidence = { inventory: m.inventory, frameMs: m.frameMs, phonemes: done };
    if (options.posteriors) {
      const n = until - emitted;
      e.posteriors = { k, startFrame: emitted, nFrames: n, ids: ids.slice(0, n * k), probs: probs.slice(0, n * k) };
      ids = ids.slice(n * k);
      probs = probs.slice(n * k);
    }
    done = [];
    emitted = until;
    return e;
  }

  const outputsFor = (frames: number) => Math.floor((Math.floor((frames - 7) / 2) + 1) / 2);

  async function drain() {
    while (feats.length - consumed >= m.window) {
      await step(feats.slice(consumed, consumed + m.window), Infinity);
      consumed += m.shift;
    }
    if (consumed > 4 * m.window) {
      feats = feats.slice(consumed);
      consumed = 0;
    }
  }

  return {
    get frames() {
      return fbankFrames;
    },
    get decoded() {
      return emitted;
    },
    async push(samples, sampleRate) {
      if (sampleRate !== m.sampleRate) {
        if (!resampler || resampler.from !== sampleRate) resampler = new Resampler(sampleRate, m.sampleRate);
        samples = resampler.push(samples);
      }
      const f = fbank.push(samples);
      fbankFrames += f.length;
      feats.push(...f);
      await drain();
      return flush(false);
    },
    async end() {
      if (resampler) {
        const f = fbank.push(resampler.flush());
        fbankFrames += f.length;
        feats.push(...f);
      }
      const total = outputsFor(fbankFrames);
      const pad = new Float32Array(m.fbank.mels).fill(m.fbank.padValue);
      while (outFrames < total) {
        while (feats.length - consumed < m.window) feats.push(pad);
        await step(feats.slice(consumed, consumed + m.window), total - outFrames);
        consumed += m.shift;
      }
      return flush(true);
    },
  };
}

export function topK(row: Float32Array, k: number): [number, number][] {
  const best: [number, number][] = [];
  for (let i = 0; i < row.length; i++) {
    const v = row[i];
    if (best.length < k || v > best[best.length - 1][1]) {
      let j = best.length < k ? best.length : k - 1;
      if (best.length < k) best.push([i, v]);
      while (j > 0 && best[j - 1][1] < v) {
        best[j] = best[j - 1];
        j--;
      }
      best[j] = [i, v];
    }
  }
  return best.map(([i, v]) => [i, Math.exp(v)]);
}
