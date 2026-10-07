/** Streaming windowed-sinc resampler, low-passed just under the output Nyquist. */
export class Resampler {
  private readonly ratio: number;
  private readonly cutoff: number;
  private history = new Float32Array(0);
  /** Input position of the next output sample, relative to `history[0]`. */
  private pos = 0;
  private static readonly TAPS = 16;

  constructor(readonly from: number, readonly to: number) {
    this.ratio = from / to;
    this.cutoff = Math.min(1, to / from) * 0.95;
  }

  push(input: Float32Array): Float32Array {
    if (this.from === this.to) return input;
    const buf = new Float32Array(this.history.length + input.length);
    buf.set(this.history);
    buf.set(input, this.history.length);
    const taps = Math.ceil(Resampler.TAPS / this.cutoff);
    const out: number[] = [];
    for (; this.pos + taps < buf.length; this.pos += this.ratio) {
      const c = Math.floor(this.pos);
      let s = 0;
      let norm = 0;
      for (let k = c - taps + 1; k <= c + taps; k++) {
        const x = (k - this.pos) * this.cutoff;
        const w = 0.5 + 0.5 * Math.cos((Math.PI * (k - this.pos)) / (taps + 1));
        const h = (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)) * w;
        norm += h;
        if (k >= 0) s += buf[k] * h;
      }
      out.push(s / norm);
    }
    const keep = Math.max(0, Math.floor(this.pos) - taps);
    this.history = buf.slice(keep);
    this.pos -= keep;
    return Float32Array.from(out);
  }

  /** The samples still held back for the filter's right side. */
  flush(): Float32Array {
    const taps = Math.ceil(Resampler.TAPS / this.cutoff);
    return this.from === this.to ? new Float32Array(0) : this.push(new Float32Array(taps + 1));
  }
}

export interface VadOptions {
  /** Speech starts above this RMS level in dBFS once the noise floor is learnt; default 12 dB above the floor. */
  marginDb?: number;
  /** Keep this much audio after speech ends before closing, and this much before it starts. */
  hangMs?: number;
  preRollMs?: number;
}

/**
 * Energy VAD with an adaptive noise floor, on 16 kHz audio in 20 ms frames. It gates what reaches the listener, so the
 * model doesn't run on long silences. Dropped audio compresses the session's timeline, so the hangover must outlast the
 * checker's confirm delay (2.5 s by default): pending errors confirm on the blanks decoded during the hangover.
 */
export class Vad {
  private floorDb = -60;
  private speaking = false;
  private silentMs = 0;
  private pre: Float32Array[] = [];
  private rest = new Float32Array(0);
  private readonly margin: number;
  private readonly hang: number;
  private readonly preRoll: number;
  private static readonly FRAME = 320;

  constructor(options: VadOptions = {}) {
    this.margin = options.marginDb ?? 12;
    this.hang = options.hangMs ?? 4000;
    this.preRoll = options.preRollMs ?? 300;
  }

  get active() {
    return this.speaking;
  }

  /** Returns the audio to pass on: speech plus pre-roll and hangover, in order. */
  push(samples: Float32Array): Float32Array {
    const buf = new Float32Array(this.rest.length + samples.length);
    buf.set(this.rest);
    buf.set(samples, this.rest.length);
    const out: Float32Array[] = [];
    let off = 0;
    for (; off + Vad.FRAME <= buf.length; off += Vad.FRAME) {
      const f = buf.subarray(off, off + Vad.FRAME);
      let e = 0;
      for (let i = 0; i < f.length; i++) e += f[i] * f[i];
      const db = 10 * Math.log10(e / f.length + 1e-12);
      const loud = db > this.floorDb + this.margin;
      if (!loud) this.floorDb += (db - this.floorDb) * (db < this.floorDb ? 0.2 : 0.01);
      if (loud) {
        if (!this.speaking) out.push(...this.pre);
        this.pre = [];
        this.speaking = true;
        this.silentMs = 0;
      } else if (this.speaking) {
        this.silentMs += 20;
        if (this.silentMs > this.hang) this.speaking = false;
      }
      if (this.speaking) out.push(f.slice());
      else {
        this.pre.push(f.slice());
        if (this.pre.length * 20 > this.preRoll) this.pre.shift();
      }
    }
    this.rest = buf.slice(off);
    const n = out.reduce((a, b) => a + b.length, 0);
    const res = new Float32Array(n);
    let o = 0;
    for (const f of out) {
      res.set(f, o);
      o += f.length;
    }
    return res;
  }
}
