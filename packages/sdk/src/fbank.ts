/**
 * Streaming 80-dim log-mel fbank at 16 kHz, identical to the student's training features:
 * `torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, energy_floor=1e-10, high_freq=-400)` on float
 * samples in [-1, 1]: 25 ms povey window, 10 ms shift, snip_edges, per-frame DC removal, pre-emphasis 0.97.
 */
export const SAMPLE_RATE = 16000;
const WIN = 400;
const HOP = 160;
const NFFT = 512;
const MELS = 80;
const EPS = 1.1920928955078125e-7;

/** Positive normalized filters with a fixed energy scale, exported with the encoder. */
export interface FrequencyPooling {
  schema: 1;
  fftSize: 512;
  filters: { start: number; weights: number[]; scale: number }[];
}

function poolingBank(pooling: FrequencyPooling): typeof melBank {
  if (pooling.schema !== 1 || pooling.fftSize !== NFFT || pooling.filters.length !== MELS) {
    throw new Error("Frequency pooling differs from the listener's analysis geometry");
  }
  return pooling.filters.map(({ start, weights, scale }) => {
    if (!Number.isInteger(start) || start < 0 || weights.length === 0 || start + weights.length > NFFT / 2 ||
        !Number.isFinite(scale) || scale <= 0 || weights.some(w => !Number.isFinite(w) || w <= 0) ||
        Math.abs(weights.reduce((a, b) => a + b, 0) - 1) > 1e-5) {
      throw new Error("Frequency pooling requires normalized positive contiguous filters");
    }
    // Snapshot the exported data: later manifest mutations cannot change a stream.
    return { start, weights: Float64Array.from(weights, w => w * scale) };
  });
}

const mel = (hz: number) => 1127 * Math.log(1 + hz / 700);

const povey = Float64Array.from({ length: WIN }, (_, i) => Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1)), 0.85));

const makeMelBank = (highFrequency: number) => {
  const lo = mel(20);
  const hi = mel(highFrequency);
  const step = (hi - lo) / (MELS + 1);
  return Array.from({ length: MELS }, (_, m) => {
    const [l, c, r] = [lo + m * step, lo + (m + 1) * step, lo + (m + 2) * step];
    const bins: number[] = [];
    const weights: number[] = [];
    for (let k = 0; k < NFFT / 2; k++) {
      const f = mel((k * SAMPLE_RATE) / NFFT);
      const v = Math.min((f - l) / (c - l), (r - f) / (r - c));
      if (v > 0) {
        bins.push(k);
        weights.push(v);
      }
    }
    return { start: bins[0] ?? 0, weights: Float64Array.from(weights) };
  });
};
const melBank = makeMelBank(SAMPLE_RATE / 2 - 400);

const cos = Float64Array.from({ length: NFFT / 2 }, (_, k) => Math.cos((-2 * Math.PI * k) / NFFT));
const sin = Float64Array.from({ length: NFFT / 2 }, (_, k) => Math.sin((-2 * Math.PI * k) / NFFT));
const reversed = Uint16Array.from({ length: NFFT }, (_, i) => {
  let r = 0;
  for (let b = 1, x = i; b < NFFT; b <<= 1, x >>= 1) r = (r << 1) | (x & 1);
  return r;
});

function fft(re: Float64Array, im: Float64Array) {
  for (let i = 0; i < NFFT; i++) {
    const j = reversed[i];
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= NFFT; len <<= 1) {
    const half = len >> 1;
    const stride = NFFT / len;
    for (let i = 0; i < NFFT; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * stride];
        const wi = sin[k * stride];
        const a = i + k;
        const b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
}

export class Fbank {
  private readonly bank: typeof melBank;
  constructor(options: { highFrequency?: number; frequencyPooling?: FrequencyPooling } = {}) {
    if (options.highFrequency !== undefined && options.frequencyPooling !== undefined) {
      throw new Error("Frequency pooling already defines the frequency regions");
    }
    this.bank = options.frequencyPooling ? poolingBank(options.frequencyPooling) :
      options.highFrequency === undefined ? melBank : makeMelBank(options.highFrequency);
  }
  private pending = new Float32Array(0);
  private readonly re = new Float64Array(NFFT);
  private readonly im = new Float64Array(NFFT);

  /** Accepts 16 kHz float samples in [-1, 1]; returns every newly complete frame. */
  push(samples: Float32Array): Float32Array[] {
    const buf = new Float32Array(this.pending.length + samples.length);
    buf.set(this.pending);
    buf.set(samples, this.pending.length);
    const { re, im } = this;
    const out: Float32Array[] = [];
    let off = 0;
    for (; off + WIN <= buf.length; off += HOP) {
      re.fill(0);
      im.fill(0);
      let mean = 0;
      for (let i = 0; i < WIN; i++) mean += buf[off + i];
      mean /= WIN;
      for (let i = WIN - 1; i >= 0; i--) {
        const x = buf[off + i] - mean;
        const prev = i > 0 ? buf[off + i - 1] - mean : x;
        re[i] = (x - 0.97 * prev) * povey[i];
      }
      fft(re, im);
      const frame = new Float32Array(MELS);
      for (let m = 0; m < MELS; m++) {
        const { start, weights } = this.bank[m];
        let e = 0;
        for (let j = 0; j < weights.length; j++) {
          const k = start + j;
          e += weights[j] * (re[k] * re[k] + im[k] * im[k]);
        }
        frame[m] = Math.log(Math.max(e, EPS));
      }
      out.push(frame);
    }
    this.pending = buf.slice(off);
    return out;
  }
}
