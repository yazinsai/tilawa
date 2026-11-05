/** Raw-audio autocorrelation pitch, independent of phoneme posteriors. Detector
 * parameters describe the human voice (60–500 Hz, 50 ms window); acceptance is
 * calibrated separately from the session's own words, never from these values.
 * Zero denotes an aperiodic frame. The box filter and its phase carry across pushes. */
export class PitchTrack {
  private readonly stride: number;
  private readonly rate: number;
  private sum = 0;
  private count = 0;
  private samples: number[] = [];
  private offset = 0;
  private hz: number[] = [];
  constructor(sampleRate: number, private readonly frameMs: number) {
    this.stride = Math.max(1, Math.round(sampleRate / 4000));
    this.rate = sampleRate / this.stride;
  }
  push(pcm: Float32Array) {
    for (const x of pcm) {
      this.sum += x;
      if (++this.count === this.stride) {
        this.samples.push(this.sum / this.count);
        this.sum = this.count = 0;
      }
    }
    this.drain(false);
  }
  end() {
    if (this.count) this.samples.push(this.sum / this.count);
    this.sum = this.count = 0;
    this.drain(true);
  }
  slice(from: number, to: number): number[] {
    return this.hz.slice(from, to);
  }
  private drain(final: boolean) {
    const window = Math.round(this.rate * 0.05);
    const hop = this.rate * this.frameMs / 1000;
    const total = this.offset + this.samples.length;
    while (true) {
      const from = Math.round(this.hz.length * hop);
      if (final ? from >= total : from + window > total) break;
      const x = new Float64Array(window);
      for (let j = 0; j < window; j++) x[j] = this.samples[from - this.offset + j] ?? 0;
      this.hz.push(autocorrelationPitch(x, this.rate));
    }
    const drop = Math.max(0, Math.round(this.hz.length * hop) - this.offset);
    this.samples = this.samples.slice(drop);
    this.offset += drop;
  }
}

function autocorrelationPitch(x: Float64Array, rate: number): number {
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  for (let i = 0; i < x.length; i++) x[i] -= mean;
  const lo = Math.ceil(rate / 500), hi = Math.floor(rate / 60);
  const corr = new Float64Array(hi + 2);
  for (let lag = lo - 1; lag <= hi + 1; lag++) {
    let xy = 0, xx = 0, yy = 0;
    for (let j = lag; j < x.length; j++) {
      xy += x[j] * x[j - lag];
      xx += x[j] ** 2;
      yy += x[j - lag] ** 2;
    }
    corr[lag] = xx * yy > 0 ? xy / Math.sqrt(xx * yy) : 0;
  }
  const peaks: number[] = [];
  for (let lag = lo; lag <= hi; lag++) if (corr[lag] >= corr[lag - 1] && corr[lag] > corr[lag + 1]) peaks.push(lag);
  const best = Math.max(0, ...peaks.map(lag => corr[lag]));
  if (best < 0.6) return 0;
  // Earliest strong peak avoids selecting an integer multiple of the period.
  const lag = peaks.find(lag => corr[lag] >= 0.9 * best)!;
  const curvature = corr[lag - 1] - 2 * corr[lag] + corr[lag + 1];
  const delta = curvature ? (corr[lag - 1] - corr[lag + 1]) / (2 * curvature) : 0;
  return rate / (lag + delta);
}
