import type { CtcToken, HeardChar, VowelProbs } from "./types.js";

export const SHORT_VOWEL_ORDER = ["\u064e", "\u064f", "\u0650"] as const; // fatha, damma, kasra

interface Run {
  id: number;
  frame: number;
  p1: number;
  p2: number;
  vowels: VowelProbs | null;
}

export class GreedyCtcDecoder {
  readonly symbols: readonly string[];
  readonly blank: number;
  /** For tokens ending in a short vowel: ids of the same token with each of the three vowels. */
  private readonly siblings: Array<Int32Array | null>;
  private previousBest: number;
  private frameIndex = 0;
  private run: Run | null = null;

  constructor(symbols: readonly string[], blank: number) {
    this.symbols = symbols;
    this.blank = blank;
    this.previousBest = blank;
    const byText = new Map(symbols.map((s, i) => [s, i] as const));
    this.siblings = symbols.map((sym) => {
      const last = sym[sym.length - 1] ?? "";
      if (!(SHORT_VOWEL_ORDER as readonly string[]).includes(last)) return null;
      const stem = sym.slice(0, -1);
      const ids = new Int32Array(3);
      for (let k = 0; k < 3; k++) ids[k] = byText.get(stem + SHORT_VOWEL_ORDER[k]!) ?? -1;
      return ids;
    });
  }

  get framesDecoded(): number {
    return this.frameIndex;
  }

  /** Start frame of the token still being decoded (not yet emitted), if any. */
  get pendingFrame(): number | null {
    return this.run ? this.run.frame : null;
  }

  reset(): void {
    this.previousBest = this.blank;
    this.frameIndex = 0;
    this.run = null;
  }

  consume(logProbs: ArrayLike<number>, frames: number, classes: number): CtcToken[] {
    const out: CtcToken[] = [];
    for (let t = 0; t < frames; t++) {
      const row = t * classes;
      let best = 0;
      let p1 = logProbs[row]!;
      let p2 = -Infinity;
      for (let c = 1; c < classes; c++) {
        const p = logProbs[row + c]!;
        if (p > p1) {
          p2 = p1;
          p1 = p;
          best = c;
        } else if (p > p2) {
          p2 = p;
        }
      }
      const sib = best !== this.blank ? this.siblings[best] : null;
      let vowels: VowelProbs | null = null;
      if (sib) {
        vowels = [0, 0, 0];
        for (let k = 0; k < 3; k++) {
          const id = sib[k]!;
          vowels[k] = id >= 0 ? Math.exp(logProbs[row + id]!) : 0;
        }
      }
      this.step(best, p1, p2, vowels, out);
    }
    return out;
  }

  flush(): CtcToken[] {
    const out: CtcToken[] = [];
    if (this.run) {
      out.push(this.emit(this.run));
      this.run = null;
    }
    this.previousBest = this.blank;
    return out;
  }

  private step(best: number, p1: number, p2: number, vowels: VowelProbs | null, out: CtcToken[]): void {
    const blank = this.blank;
    if (best !== blank && best !== this.previousBest) {
      if (this.run) out.push(this.emit(this.run));
      this.run = { id: best, frame: this.frameIndex, p1, p2, vowels };
    } else if (
      best !== blank &&
      best === this.previousBest &&
      this.run &&
      p1 > this.run.p1
    ) {
      this.run.p1 = p1;
      this.run.p2 = p2;
      this.run.vowels = vowels;
    } else if (best === blank && this.run) {
      out.push(this.emit(this.run));
      this.run = null;
    }
    this.previousBest = best;
    this.frameIndex++;
  }

  private emit(run: Run): CtcToken {
    const token: CtcToken = {
      sym: this.symbols[run.id] ?? "",
      frame: run.frame,
      margin: Math.exp(run.p1) - Math.exp(run.p2),
    };
    if (run.vowels) token.vowels = run.vowels;
    return token;
  }
}

export function expandTokens(tokens: readonly CtcToken[]): HeardChar[] {
  const out: HeardChar[] = [];
  for (const t of tokens) {
    const chars = [...t.sym];
    for (let i = 0; i < chars.length; i++) {
      const hc: HeardChar = { ch: chars[i]!, frame: t.frame, margin: t.margin };
      // Vowel alternatives belong to the token's final (vowel) char only.
      if (t.vowels && i === chars.length - 1) hc.vowels = t.vowels;
      out.push(hc);
    }
  }
  return out;
}
