/** Cuts a stream of feature frames into encoder windows of `size` frames, advancing by `shift`. */
export class FrameChunker {
  private buf: Float32Array[] = [];

  constructor(
    readonly size: number,
    readonly shift: number,
  ) {}

  push(frames: Float32Array[]): Float32Array[][] {
    this.buf.push(...frames);
    const out: Float32Array[][] = [];
    while (this.buf.length >= this.size) {
      out.push(this.buf.slice(0, this.size));
      this.buf = this.buf.slice(this.shift);
    }
    return out;
  }

  /** Zero-pads the remainder into one last window if it holds frames no window has covered yet. */
  flush(): Float32Array[][] {
    if (this.buf.length <= this.size - this.shift) return [];
    const dim = this.buf[0].length;
    const w = [...this.buf];
    while (w.length < this.size) w.push(new Float32Array(dim));
    this.buf = [];
    return [w];
  }
}
