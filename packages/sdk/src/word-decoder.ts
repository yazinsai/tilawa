import type { InferenceBackend } from "./model.js";

export interface HeardWord { text: string; logp: number }
export interface WordTokenizer {
  schema: 1;
  samples: number;
  prefix: number[];
  eos: number;
  maxTokens: number;
  suppressed: number[];
  vocabulary: string[];
}

export const normalizeWord = (s: string): string => s.normalize("NFKC").replace(/ٱ/g, "ا")
  .replace(/ـ/g, "").replace(/\p{M}/gu, "").replace(/[^\p{L}]/gu, "");

/** GPT-2/Whisper's reversible byte alphabet. Vocabulary pieces can split UTF-8. */
function byteAlphabet(): Map<string, number> {
  const visible = [...Array.from({ length: 94 }, (_, i) => i+33),
    ...Array.from({ length: 12 }, (_, i) => i+161), ...Array.from({ length: 82 }, (_, i) => i+174)];
  const values = new Set(visible), map = new Map<string, number>();
  let other = 256;
  for (let b = 0; b < 256; b++) map.set(String.fromCodePoint(values.has(b) ? b : other++), b);
  return map;
}

/** Whole-word confidence is the mean log-probability of its emitted BPE pieces.
 * Splitting UTF-8 or a word across pieces never creates fragment words. */
export function decodeWords(pieces: { bytes: number[]; logp: number }[]): HeardWord[] {
  const bytes: number[] = [], ranges: { from: number; to: number; logp: number }[] = [];
  for (const p of pieces) {
    const from = bytes.length;
    bytes.push(...p.bytes);
    ranges.push({ from, to: bytes.length, logp: p.logp });
  }
  const out: HeardWord[] = [];
  for (let from = 0; from < bytes.length;) {
    while (from < bytes.length && bytes[from] <= 32) from++;
    let to = from;
    while (to < bytes.length && bytes[to] > 32) to++;
    if (to === from) break;
    const text = normalizeWord(new TextDecoder().decode(Uint8Array.from(bytes.slice(from, to))));
    const scored = ranges.filter(r => r.from < to && r.to > from);
    if (text && scored.length) out.push({ text, logp: scored.reduce((s,r) => s+r.logp, 0)/scored.length });
    from = to;
  }
  return out;
}

export function createWordDecoder(encoder: InferenceBackend, decoder: InferenceBackend, tokenizer: WordTokenizer) {
  const alphabet = byteAlphabet();
  const vocabulary = tokenizer.vocabulary.map(s => [...s].map(c => alphabet.get(c)));
  const suppressed = new Set(tokenizer.suppressed);
  return {
    async decode(samples: Float32Array): Promise<HeardWord[]> {
      if (!samples.length || samples.length > tokenizer.samples) return [];
      const audio = new Float32Array(tokenizer.samples);
      audio.set(samples);
      const encoded = await encoder.run({ audio: { type: "float32", data: audio, dims: [1, audio.length] } });
      const ids = [...tokenizer.prefix], pieces: { bytes: number[]; logp: number }[] = [];
      let complete = false;
      while (ids.length < tokenizer.maxTokens) {
        const result = await decoder.run({ input_ids: { type: "int64", data: BigInt64Array.from(ids, BigInt), dims: [1, ids.length] },
          encoder_hidden_states: encoded.encoder_hidden_states });
        const logits = result.logits.data as Float32Array;
        if (logits.length !== vocabulary.length) throw new Error("Word decoder output differs from tokenizer vocabulary");
        let best = -1, peak = -Infinity;
        for (let i = 0; i < logits.length; i++) if (!suppressed.has(i) && logits[i] > peak) { best = i; peak = logits[i]; }
        if (best < 0 || !Number.isFinite(peak)) return [];
        let mass = 0;
        for (let i = 0; i < logits.length; i++) if (!suppressed.has(i)) mass += Math.exp(logits[i]-peak);
        if (best === tokenizer.eos) { complete = true; break; }
        const bytes = vocabulary[best];
        if (bytes.some(b => b === undefined)) return [];
        pieces.push({ bytes: bytes as number[], logp: -Math.log(mass) });
        ids.push(best);
      }
      return complete ? decodeWords(pieces) : [];
    },
  };
}
