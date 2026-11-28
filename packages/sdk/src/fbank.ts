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

/**
 * In-place radix-2 FFT. Only bins 0 to NFFT / 2 - 1 are read afterwards (the mel bank's), so the last stage leaves
 * the upper half as it was. Every value read is computed exactly as the textbook loop computes it, butterfly for
 * butterfly; only the loop order (each twiddle read once a stage) and the twiddle 1 + 0i (k = 0, a plain sum and
 * difference: the same numbers, a zero's sign aside, which the power spectrum squares away) and the real input's zero
 * imaginary parts in the first two stages are cheaper. The app runs
 * this in JavaScriptCore without a JIT, where each array read and multiplication in the inner loop is paid in full.
 */
/** `im` must be all +0 on the way in (a real input). */
function fft(re: Float64Array, im: Float64Array) {
  for (let i = 0; i < NFFT; i++) {
    const j = reversed[i];
    if (i < j) {
      // Plain temporaries: a destructuring swap allocates two arrays a call in an interpreter without a JIT.
      // The imaginary parts are all +0 here: nothing to swap.
      const r = re[i];
      re[i] = re[j];
      re[j] = r;
    }
  }
  // The input is real: the imaginary parts are +0 and stay +0 through the first stage (sums and differences of +0),
  // and the second stage's odd twiddle (k = 1) multiplies a real value.
  for (let a = 0; a < NFFT; a += 2) {
    const xr = re[a + 1];
    re[a + 1] = re[a] - xr;
    re[a] += xr;
  }
  {
    const wr = cos[NFFT / 4];
    const wi = sin[NFFT / 4];
    for (let a = 0; a < NFFT; a += 4) {
      const b = a + 2;
      const xr = re[b];
      re[b] = re[a] - xr;
      re[a] += xr;
      const c = a + 1, e = a + 3;
      const yr = re[e] * wr;
      const yi = re[e] * wi;
      re[e] = re[c] - yr;
      im[e] = -yi;
      re[c] += yr;
      im[c] = yi;
    }
  }
  for (let len = 8; len < NFFT; len <<= 1) {
    const half = len >> 1;
    const stride = NFFT / len;
    for (let a = 0; a < NFFT; a += len) {
      const b = a + half;
      const xr = re[b];
      const xi = im[b];
      re[b] = re[a] - xr;
      im[b] = im[a] - xi;
      re[a] += xr;
      im[a] += xi;
    }
    for (let k = 1; k < half; k++) {
      const wr = cos[k * stride];
      const wi = sin[k * stride];
      for (let a = k; a < NFFT; a += len) {
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
  // The last stage (len = NFFT): one butterfly per bin, whose lower output alone is read.
  const half = NFFT >> 1;
  re[0] += re[half];
  im[0] += im[half];
  for (let a = 1; a < half; a++) {
    const b = a + half;
    const wr = cos[a];
    const wi = sin[a];
    re[a] += re[b] * wr - im[b] * wi;
    im[a] += re[b] * wi + im[b] * wr;
  }
}

export class Fbank {
  private readonly bank: typeof melBank;
  /** The bank's weights end to end, and each filter's first and last-plus-one bin: one array read a weight. */
  private readonly flat: { starts: Int32Array; ends: Int32Array; weights: Float64Array };
  constructor(options: { highFrequency?: number } = {}) {
    this.bank = options.highFrequency === undefined ? melBank : makeMelBank(options.highFrequency);
    const starts = Int32Array.from(this.bank, (b) => b.start);
    const ends = Int32Array.from(this.bank, (b) => b.start + b.weights.length);
    const weights = new Float64Array(this.bank.reduce((n, b) => n + b.weights.length, 0));
    let at = 0;
    for (const b of this.bank) {
      weights.set(b.weights, at);
      at += b.weights.length;
    }
    this.flat = { starts, ends, weights };
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
      // Samples 0 to WIN - 1 are written below; the rest of the FFT's input is zero.
      re.fill(0, WIN);
      im.fill(0);
      let mean = 0;
      for (let i = 0; i < WIN; i++) mean += buf[off + i];
      mean /= WIN;
      // Each centred sample once, as the one before the next (the first is its own).
      let prev = buf[off] - mean;
      for (let i = 0; i < WIN; i++) {
        const x = buf[off + i] - mean;
        re[i] = (x - 0.97 * prev) * povey[i];
        prev = x;
      }
      fft(re, im);
      const frame = new Float32Array(MELS);
      const { starts, ends, weights } = this.flat;
      for (let m = 0, w = 0; m < MELS; m++) {
        let e = 0;
        for (let k = starts[m], end = ends[m]; k < end; k++, w++) e += weights[w] * (re[k] * re[k] + im[k] * im[k]);
        frame[m] = Math.log(Math.max(e, EPS));
      }
      out.push(frame);
    }
    this.pending = buf.slice(off);
    return out;
  }
}
