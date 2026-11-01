import type { InferenceBackend } from "./model.js";
import { Fbank } from "./fbank.js";

interface AssetFile {
  file: string;
  sha256: string;
  bytes: number;
}

/** The word gallery is built from real training words; its radius is held-out calibration. */
export interface WordEmbeddingAsset {
  schema: 1;
  sampleRate: 16000;
  dimensions: number;
  keys: string[];
  radius: number;
  encoder: AssetFile;
  words: AssetFile;
  prefixes: AssetFile;
  prefixKeys: AssetFile;
}

export interface WordEmbeddingMatch {
  word: string;
  distance: number;
  neighbourDistance: number;
  radius: number;
}

export interface WordEmbeddingModel {
  /** Null means no whole word beat all repeat and restart explanations. */
  compare(audio: Float32Array, sampleRate: number, neighbours: readonly string[]): Promise<WordEmbeddingMatch | null>;
  release(): Promise<void>;
}

/** Ordinary endpointing with the same 80 ms edge margin as real-word training crops. */
export function trimWordGap(wave: Float32Array): Float32Array {
  if (wave.length < 400) return wave;
  const energy: number[] = [];
  let peak = 0;
  for (let a = 0; a + 400 <= wave.length; a += 160) {
    let power = 0;
    for (let i = a; i < a + 400; i++) power += wave[i] * wave[i];
    power /= 400;
    energy.push(power);
    peak = Math.max(peak, power);
  }
  const floor = Math.max(1e-10, peak * 1e-4);
  const first = energy.findIndex(e => e > floor);
  if (first < 0) return wave.subarray(0, 0);
  let last = energy.length - 1;
  while (energy[last] <= floor) last--;
  return wave.subarray(Math.max(0, first * 160 - 1280), Math.min(wave.length, last * 160 + 400 + 1280));
}

export function createWordEmbedding(
  asset: WordEmbeddingAsset,
  backend: InferenceBackend,
  words: Float32Array,
  prefixes: Float32Array,
  prefixKeys: Uint32Array,
): WordEmbeddingModel {
  const dim = asset.dimensions;
  if (asset.schema !== 1 || asset.sampleRate !== 16000 || !Number.isInteger(dim) || dim <= 0
    || words.length !== asset.keys.length * dim || prefixes.length !== prefixKeys.length * dim
    || !Number.isFinite(asset.radius) || asset.radius < 0 || asset.radius > 2)
    throw new Error("Invalid word embedding gallery");
  const wordIds = new Map(asset.keys.map((key, i) => [key, i]));
  if (wordIds.size !== asset.keys.length) throw new Error("Duplicate word embedding keys");
  const prefixRows = new Map<number, number[]>();
  prefixKeys.forEach((key, row) => {
    if (key >= asset.keys.length) throw new Error("Invalid prefix word id");
    const rows = prefixRows.get(key) ?? [];
    rows.push(row);
    prefixRows.set(key, rows);
  });
  const distance = (bank: Float32Array, row: number, z: Float32Array) => {
    let dot = 0;
    for (let i = 0; i < dim; i++) dot += bank[row * dim + i] * z[i];
    return Math.max(0, Math.min(2, 1 - dot));
  };
  return {
    release: () => backend.release(),
    async compare(audio, sampleRate, neighbours) {
      if (sampleRate !== asset.sampleRate || !neighbours.length) return null;
      const ids: number[] = [];
      for (const word of new Set(neighbours)) {
        const id = wordIds.get(word);
        if (id === undefined) return null;
        ids.push(id);
      }
      const crop = trimWordGap(audio);
      if (crop.length < 3840 || crop.length > 4 * sampleRate) return null;
      const frames = new Fbank().push(crop);
      const features = new Float32Array(frames.length * 80);
      frames.forEach((frame, i) => features.set(frame, i * 80));
      const outputs = await backend.run({
        x: { type: "float32", data: features, dims: [1, frames.length, 80] },
        lengths: { type: "int64", data: new BigInt64Array([BigInt(frames.length)]), dims: [1] },
        duration: { type: "float32", data: new Float32Array([crop.length / sampleRate]), dims: [1] },
      });
      const z = outputs.embedding?.data as Float32Array | undefined;
      if (!z || z.length !== dim || !z.every(Number.isFinite)) return null;
      let winner = -1, best = Infinity;
      for (let row = 0; row < asset.keys.length; row++) {
        const d = distance(words, row, z);
        if (d < best) { best = d; winner = row; }
      }
      if (winner < 0 || ids.includes(winner) || best > asset.radius) return null;
      let neighbourDistance = Infinity;
      for (const id of ids) {
        neighbourDistance = Math.min(neighbourDistance, distance(words, id, z));
        for (const row of prefixRows.get(id) ?? []) neighbourDistance = Math.min(neighbourDistance, distance(prefixes, row, z));
      }
      return best < neighbourDistance ? { word: asset.keys[winner], distance: best, neighbourDistance, radius: asset.radius } : null;
    },
  };
}
