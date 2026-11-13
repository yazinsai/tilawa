import type { AcousticTransitionFeatures, PhonemeEvidence } from "@tilawa/core";

/** Preserve spectral trajectories before the listener's phone classification. No transcript is involved. */
export function summarizeTransition(frames: readonly Float32Array[], width: number): number[] {
  if (!frames.length || width < 80 || frames.some(f => f.length !== 80)) throw new Error("Invalid log-mel transition window");
  const row = new Array<number>(width).fill(0);
  for (let b = 0; b < 40; b++) {
    for (const f of frames) row[b] += (f[2 * b] + f[2 * b + 1]) / (2 * frames.length);
    const first = frames[0], last = frames.at(-1)!;
    row[40 + b] = (last[2 * b] + last[2 * b + 1] - first[2 * b] - first[2 * b + 1]) / 2;
  }
  return row;
}

/** Pack original feature intervals onto primary tokens so ordinary evidence writers/rechunkers retain them.
 * Token metadata ends at the token's own last frame. No future frame is supplied to its comparison. */
export function carryAcousticTransitions(chunks: readonly PhonemeEvidence[], features: readonly AcousticTransitionFeatures[]): PhonemeEvidence[] {
  if (!features.length) return [...chunks];
  const first = features[0], width = first.width;
  const values: number[] = [];
  for (const f of features) {
    if (f.width !== width || f.startFrame !== first.startFrame + values.length / width) throw new Error("Discontinuous original features");
    for (const v of Array.from(f.values)) values.push(v);
  }
  let from = first.startFrame;
  return chunks.map(c => ({ ...c, phonemes: c.phonemes.map(p => {
    const to = p.end + 1;
    if (to <= from || to > first.startFrame + values.length / width) throw new Error("Token outside original acoustic features");
    const acousticTransitions: AcousticTransitionFeatures = { version: 1, width, startFrame: from, nFrames: to - from,
      values: values.slice((from - first.startFrame) * width, (to - first.startFrame) * width) };
    from = to;
    return { ...p, acousticTransitions };
  }) }));
}
