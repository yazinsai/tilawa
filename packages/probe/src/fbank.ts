/** Streaming Kaldi-style 80-dim log-mel fbank at 16 kHz (25 ms window, 10 ms shift), as sherpa-onnx feeds Zipformer. */
const SR = 16000;
const WIN = 400;
const HOP = 160;
const NFFT = 512;
const MELS = 80;

const mel = (hz: number) => 1127 * Math.log(1 + hz / 700);

const povey = Float32Array.from({ length: WIN }, (_, i) => Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1)), 0.85));

const melBank: { start: number; weights: Float32Array }[] = (() => {
  const lo = mel(20);
  const hi = mel(SR / 2 - 400);
  const step = (hi - lo) / (MELS + 1);
  const bins = NFFT / 2;
  return Array.from({ length: MELS }, (_, m) => {
    const [l, c, r] = [lo + m * step, lo + (m + 1) * step, lo + (m + 2) * step];
    const w = new Float32Array(bins);
    let start = -1;
    for (let k = 0; k < bins; k++) {
      const f = mel((k * SR) / NFFT);
      const v = f > l && f < r ? (f <= c ? (f - l) / (c - l) : (r - f) / (r - c)) : 0;
      if (v > 0 && start < 0) start = k;
      w[k] = v;
    }
    return { start: Math.max(0, start), weights: w };
  });
})();

const fft = (re: Float32Array, im: Float32Array) => {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k);
        const wi = Math.sin(ang * k);
        const a = i + k;
        const b = a + len / 2;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
};

export class Fbank {
  private pending = new Float32Array(0);

  /** Accepts float samples in [-1, 1]; returns every complete frame. */
  push(samples: Float32Array): Float32Array[] {
    const buf = new Float32Array(this.pending.length + samples.length);
    buf.set(this.pending);
    buf.set(samples, this.pending.length);
    const out: Float32Array[] = [];
    let off = 0;
    const re = new Float32Array(NFFT);
    const im = new Float32Array(NFFT);
    for (; off + WIN <= buf.length; off += HOP) {
      re.fill(0);
      im.fill(0);
      let mean = 0;
      for (let i = 0; i < WIN; i++) mean += buf[off + i];
      mean /= WIN;
      for (let i = WIN - 1; i >= 0; i--) {
        const x = (buf[off + i] - mean);
        const prev = i > 0 ? (buf[off + i - 1] - mean) : x;
        re[i] = (x - 0.97 * prev) * povey[i];
      }
      fft(re, im);
      const frame = new Float32Array(MELS);
      for (let m = 0; m < MELS; m++) {
        const { start, weights } = melBank[m];
        let e = 0;
        for (let k = start; k < weights.length && (weights[k] > 0 || k === start); k++) e += weights[k] * (re[k] * re[k] + im[k] * im[k]);
        frame[m] = Math.log(Math.max(e, 1.1920929e-7));
      }
      out.push(frame);
    }
    this.pending = buf.slice(off);
    return out;
  }
}
