import { ctcForward as forward } from "@tilawa/core"

/**
 * CTC forward score for one word-sized hypothesis. Negative log-likelihood, in nats, of `labels` (token ids; blank is
 * 0 and is not listed) over frames `[from, to)`. `logProbs` is row-major frames × `vocab` natural logs, blank at
 * index 0, as the acoustic model emits them. The checker runs the same forward on the top-k evidence it stores.
 */
export function ctcForward(logProbs: ArrayLike<number>, vocab: number, labels: ArrayLike<number>, from: number, to: number): number {
  return forward((frame) => {
    const start = frame * vocab
    return ArrayBuffer.isView(logProbs)
      ? (logProbs as Float32Array).subarray(start, start + vocab)
      : Array.prototype.slice.call(logProbs, start, start + vocab)
  }, labels, from, to)
}
