/**
 * Int16 audio retained on the session clock. Sample `s` is session ms `s / 16`.
 * - ring: a fixed buffer of the last `capacity` samples, allocated once.
 * - take: everything from the first sample, grown on demand and hard-capped at `capacity` samples; later samples
 *   are dropped and `overflowed` is set.
 */
export class PcmStore {
  private buf: Int16Array;
  /** Session sample index of the first sample ever written; null before any. */
  private first: number | null = null;
  /** One past the last written session sample index. */
  private end = 0;
  overflowed = false;

  constructor(readonly mode: "ring" | "take", readonly capacity: number) {
    this.buf = new Int16Array(mode === "ring" ? capacity : Math.min(capacity, 16_000 * 30));
  }

  get empty(): boolean {
    return this.first === null;
  }

  /** Expected session sample index of the next chunk, or null before the first one. */
  get next(): number | null {
    return this.first === null ? null : this.end;
  }

  /** Retained `[start, end)` in session samples. */
  get range(): { start: number; end: number } {
    if (this.first === null) return { start: 0, end: 0 };
    if (this.mode === "take") return { start: this.first, end: Math.min(this.end, this.first + this.capacity) };
    return { start: Math.max(this.first, this.end - this.capacity), end: this.end };
  }

  /** True once take mode has filled its cap: no later audio will be kept. */
  get full(): boolean {
    return this.mode === "take" && this.first !== null && this.end >= this.first + this.capacity;
  }

  /** Allocated bytes (the memory bound tests read this). */
  get bytes(): number {
    return this.buf.byteLength;
  }

  /** Appends contiguous samples starting at session sample `at` (the caller checks contiguity). */
  write(samples: Float32Array, at: number): void {
    if (this.first === null) {
      this.first = at;
      this.end = at;
    }
    const n = samples.length;
    if (this.mode === "take") {
      const room = this.first + this.capacity - this.end;
      const keep = Math.max(0, Math.min(n, room));
      if (keep < n) this.overflowed = true;
      const offset = this.end - this.first;
      if (offset + keep > this.buf.length) {
        let size = this.buf.length;
        while (size < offset + keep) size *= 2;
        const grown = new Int16Array(Math.min(size, this.capacity));
        grown.set(this.buf.subarray(0, offset));
        this.buf = grown;
      }
      for (let i = 0; i < keep; i++) this.buf[offset + i] = toInt16(samples[i]);
      this.end += n;
      return;
    }
    // Only the last `capacity` samples of a long chunk survive.
    const skip = Math.max(0, n - this.capacity);
    for (let i = skip; i < n; i++) this.buf[(this.end + i) % this.capacity] = toInt16(samples[i]);
    this.end += n;
  }

  /** Copies session samples `[from, to)`, which must lie inside `range`. */
  read(from: number, to: number): Int16Array {
    const out = new Int16Array(Math.max(0, to - from));
    if (this.first === null) return out;
    for (let s = from, i = 0; s < to; s++, i++)
      out[i] = this.mode === "take" ? this.buf[s - this.first] : this.buf[s % this.capacity];
    return out;
  }

  release(): void {
    this.buf = new Int16Array(0);
    this.first = null;
    this.end = 0;
  }
}

function toInt16(x: number): number {
  if (!(x === x)) return 0; // NaN
  const c = x < -1 ? -1 : x > 1 ? 1 : x;
  return c < 0 ? Math.round(c * 32768) : Math.round(c * 32767);
}
