import type { DecodedPhoneme } from "./evidence.js";

export interface PitchContinuity { jump: number; gap: number }

/** Jumps between consecutive voiced observations, including either side of an
 * unvoiced interval. Leading/trailing silence is not an internal gap. Undefined
 * evidence and an entirely unvoiced interval cannot certify continuity. */
export function pitchContinuity(phones: readonly DecodedPhoneme[], from: number, to: number): PitchContinuity | null {
  const hz = new Map<number, number>();
  for (const p of phones) if (p.pitch) {
    const start = p.end + 1 - p.pitch.length;
    for (let i = 0; i < p.pitch.length; i++) if (start + i >= from && start + i < to) hz.set(start + i, p.pitch[i]);
  }
  let previous = -1, prevHz = 0, jump = 0, gap = 0;
  for (let f = from; f < to; f++) {
    const value = hz.get(f);
    if (value === undefined || !Number.isFinite(value) || value < 0) return null;
    if (!value) continue;
    if (previous >= 0) {
      jump = Math.max(jump, Math.abs(Math.log2(value / prevHz)));
      gap = Math.max(gap, f - previous - 1);
    }
    previous = f;
    prevHz = value;
  }
  return previous < 0 ? null : { jump, gap };
}

/** No fitted cut: both statistics must fit the observed within-word maxima. */
export function pitchFits(candidate: PitchContinuity | null, envelope: PitchContinuity | null): boolean {
  return !!candidate && !!envelope && candidate.jump <= envelope.jump && candidate.gap <= envelope.gap;
}
