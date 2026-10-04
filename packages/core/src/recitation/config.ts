export interface EngineConfig {
  jumpCost: number;
  repeatCost: number;
  commitDwell: number;
  okDistance: number;
  unsureDistance: number;
  minHeardFraction: number;
  minMargin: number;
  lostWindow: number;
  lostRate: number;
  holdWindow: number;
  holdRate: number;
  searchMinChars: number;
  searchQueryChars: number;
  searchDecisiveDistance: number;
  searchDecisiveMargin: number;
  searchEveryFrames: number;
  searchEveryChars: number;
  locateFailedFrames: number;
  relocateEveryFrames: number;
  relocateQueryChars: number;
  relocateMaxDistance: number;
  relocateRateMargin: number;
  idleFrames: number;
  maxStruggles: number;
  settleFrames: number;
  /** Correction mode: on lock, start the tracker at the hit ayah's first word
   * and replay up to this many heard chars per expected char of the words
   * before the hit (0 = off). */
  backfillRatio: number;
  /** Correction mode: when the take ends before any lock, align the buffered
   * chars to the best search hit at or under this distance for the final word
   * check (0 = off). */
  stopAlignDistance: number;
  /** Correction mode with an expected passage: extra cost of a tracker jump
   * to a word outside the passage (similar passages elsewhere in the surah). */
  outsideJumpCost: number;
  /** Expected passage, interior ayah A the tracker sits in: A counts as
   * skipped when at least `skipMinChars` heard chars aligned to A fit A+1 with
   * distance at most `skipMaxDistance`, and better than A by `skipMargin`.
   * `skipMinChars` 0 disables. */
  skipMinChars: number;
  skipMaxDistance: number;
  skipMargin: number;
  /** Max fraction of those chars the A+1 reading may leave unaligned at its head. */
  skipMaxHead: number;
  /** Correction mode: see `VerdictTracer.anchorAyahEnd` (0 = off). */
  anchorAyahEnd: number;
}

export const DEFAULT_CONFIG: EngineConfig = {
  jumpCost: 12,
  repeatCost: 10,
  commitDwell: 6,
  okDistance: 0.15,
  unsureDistance: 0.4,
  minHeardFraction: 0.34,
  minMargin: 0.35,
  lostWindow: 120,
  lostRate: 0.35,
  holdWindow: 30,
  holdRate: 0.45,
  searchMinChars: 12,
  searchQueryChars: 250,
  searchDecisiveDistance: 0.35,
  searchDecisiveMargin: 0.1,
  searchEveryFrames: 25,
  searchEveryChars: 12,
  locateFailedFrames: 375,
  relocateEveryFrames: 37,
  relocateQueryChars: 100,
  relocateMaxDistance: 0.3,
  relocateRateMargin: 0.12,
  idleFrames: 200,
  maxStruggles: 3,
  settleFrames: 25,
  backfillRatio: 1.5,
  stopAlignDistance: 0.35,
  outsideJumpCost: 0,
  skipMinChars: 10,
  skipMaxDistance: 0.35,
  skipMargin: 0.25,
  skipMaxHead: 0.1,
  anchorAyahEnd: 2,
};

export const BUFFER_CAP = 1000;

export const SAMPLE_RATE = 16000;
export const FBANK_BINS = 80;
export const FRAME_LENGTH = 400;
export const FRAME_SHIFT = 160;
export const ZIPFORMER_T = 61;
export const ZIPFORMER_HOP = 48;
export const ZIPFORMER_VOCAB = 251;
export const CTC_HZ = 25;
