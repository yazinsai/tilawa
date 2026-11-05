import { PosteriorFrames } from "./ctc.js";

/** A contextual decode shared by its anchored word occurrences. Probabilities retain the SDK's four decimals. */
export interface PackedWordReference {
  from: number;
  to: number;
  k: number;
  ids: string;
  /** Base64, little-endian unsigned 16-bit probabilities in units of 1/10000. */
  probs: string;
}

/** Expand a requested range through the same top-k interpretation used by the live listener. */
export function referenceRows(
  record: PackedWordReference,
  inventorySize: number,
  from: number,
  to: number,
): Float64Array[] {
  if (!Number.isInteger(from) || !Number.isInteger(to) || from < record.from || to > record.to || to <= from)
    throw new Error("Reference span is outside its contextual decode");
  const decode = (s: string) => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const ids = decode(record.ids), bytes = decode(record.probs);
  const nFrames = record.to - record.from;
  if (!Number.isInteger(record.k) || record.k < 1 || ids.length !== nFrames * record.k || bytes.length !== ids.length * 2)
    throw new Error("Malformed packed reference posteriors");
  const view = new DataView(bytes.buffer);
  const start = (from - record.from) * record.k, end = (to - record.from) * record.k;
  const probs = new Float64Array(end - start);
  for (let i = start; i < end; i++) probs[i - start] = view.getUint16(i * 2, true) / 10000;
  const frames = new PosteriorFrames(inventorySize);
  frames.append({ startFrame: from, nFrames: to - from, k: record.k, ids: ids.subarray(start, end), probs });
  return Array.from({ length: to - from }, (_, i) => Float64Array.from(frames.row(from + i), Math.exp));
}
