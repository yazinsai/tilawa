import { PosteriorFrames } from "@tilawa/core";

/** A maximal exact plateau in the input PCM, with half-open sample bounds. */
export interface ConstantAudioInterval {
  from: number;
  to: number;
  value: number;
  entryJump: number | null;
  exitJump: number | null;
}

/** Candidate enumeration only: equality alone does not certify a dropout. */
export function constantAudioIntervals(samples: Float32Array): ConstantAudioInterval[] {
  const intervals: ConstantAudioInterval[] = [];
  let from = 0;
  const finish = (to: number) => {
    if (to - from < 2 || !Number.isFinite(samples[from])) return;
    intervals.push({ from, to, value: samples[from],
      entryJump: from ? Math.abs(samples[from] - samples[from - 1]) : null,
      exitJump: to < samples.length ? Math.abs(samples[to] - samples[to - 1]) : null });
  };
  for (let i = 1; i < samples.length; i++) {
    if (samples[i] !== samples[i - 1]) { finish(i); from = i; }
  }
  if (samples.length) finish(samples.length);
  return intervals;
}

/**
 * A certified unavailable observation has unit emission likelihood for every
 * class, including blank. The CTC clock and transitions remain intact. This
 * differs from excising frames: phones can occupy elapsed but unobserved time.
 * No stored posterior, greedy token or replacement waveform is changed.
 */
export class CensoredPosteriorFrames extends PosteriorFrames {
  private readonly unobserved: Float64Array;

  constructor(private readonly source: PosteriorFrames, size: number,
    private readonly unavailable: ReadonlyArray<readonly [number, number]>) {
    super(size);
    this.unobserved = new Float64Array(size + 1);
    for (const [from, to] of unavailable) {
      if (!Number.isInteger(from) || !Number.isInteger(to) || from < source.startFrame ||
        to <= from || to > source.endFrame) throw new RangeError("Unavailable frames lie outside the evidence");
    }
  }

  override get startFrame() { return this.source.startFrame; }
  override get endFrame() { return this.source.endFrame; }
  override row(frame: number): Float64Array {
    return this.unavailable.some(([from, to]) => frame >= from && frame < to)
      ? this.unobserved : this.source.row(frame);
  }
}

/** Both hypotheses must be finite and change their strict preference. */
export function omissionDependsOnMissingAudio(frames: PosteriorFrames, size: number,
  read: ArrayLike<number>, omit: ArrayLike<number>, from: number, to: number,
  unavailable: ReadonlyArray<readonly [number, number]>): boolean {
  if (!unavailable.some(([a, b]) => a < to && b > from)) return false;
  const readNll = frames.nll(read, from, to), omitNll = frames.nll(omit, from, to);
  if (!Number.isFinite(readNll) || !Number.isFinite(omitNll) || readNll <= omitNll) return false;
  const censored = new CensoredPosteriorFrames(frames, size, unavailable);
  const missingRead = censored.nll(read, from, to), missingOmit = censored.nll(omit, from, to);
  return Number.isFinite(missingRead) && Number.isFinite(missingOmit) && missingRead < missingOmit;
}
