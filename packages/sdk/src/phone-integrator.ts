import type { DecodedPhoneme } from "@tilawa/core";

/** Rebase an isolated decode without rounding its original contribution intervals to the frame grid. */
export function shiftPhoneEvent(p: DecodedPhoneme, frameShift: number, fromSample: number, sampleRate: number): DecodedPhoneme {
  const integration = p.integration;
  if (!integration) return { ...p, start: p.start + frameShift, end: p.end + frameShift };
  const offset = Math.round(fromSample * integration.sampleRate / sampleRate);
  return {
    ...p, start: p.start + frameShift, end: p.end + frameShift,
    integration: { ...integration, contributions: integration.contributions.map((c) => ({
      ...c, fromSample: c.fromSample + offset, toSample: c.toSample + offset,
    })) },
  };
}

/** Continuous acoustic integration. No transcript, target count, or text history enters this decoder. */
export class PhoneIntegrator {
  private mass = 0;
  private scores: number[];
  private contributions: { fromSample: number; toSample: number; mass: number }[] = [];
  private end = 0;

  constructor(private tokens: string, private sampleRate: number, private frameMs: number) {
    this.scores = new Array(tokens.length + 1).fill(0);
  }

  /** Hold one posterior row for a possible terminal tail; original source intervals travel separately. */
  finalizedFrame(nextFrame: number): number {
    return this.mass > 0 ? Math.max(0, nextFrame - 1) : nextFrame;
  }

  push(mass: number, scores: ArrayLike<number>, frame: number): DecodedPhoneme[] {
    if (!Number.isFinite(mass) || mass < 0 || mass > 1 || scores.length !== this.scores.length)
      throw new Error("Invalid CIF acoustic output");
    for (let i = 0; i < scores.length; i++) if (!Number.isFinite(scores[i])) throw new Error("Nonfinite CIF phone score");
    const out: DecodedPhoneme[] = [];
    while (mass > 0) {
      this.end = frame;
      const take = Math.min(mass, 1 - this.mass);
      this.contributions.push({
        fromSample: Math.round(frame * this.frameMs * this.sampleRate / 1000),
        toSample: Math.round((frame + 1) * this.frameMs * this.sampleRate / 1000),
        mass: take,
      });
      for (let i = 0; i < scores.length; i++) this.scores[i] += take * scores[i];
      this.mass += take;
      mass -= take;
      if (this.mass >= 1) {
        const token = this.fire();
        if (token) out.push(token);
      }
    }
    return out;
  }

  /** Ordinary CIF tail rounding, only at end of audio; it never rescales the stream to a text length. */
  endStream(finalFrame = this.end): DecodedPhoneme[] {
    this.end = finalFrame;
    const token = this.mass >= 0.5 ? this.fire() : null;
    this.clear();
    return token ? [token] : [];
  }

  private fire(): DecodedPhoneme | null {
    // Normalizing the partial terminal vector preserves the affine classifier's bias.
    const logits = this.scores.map((s) => s / this.mass);
    const best = logits.indexOf(Math.max(...logits));
    const conf = 1 / logits.reduce((s, l) => s + Math.exp(l - logits[best]), 0);
    const token: DecodedPhoneme | null = best === 0 ? null : {
      p: this.tokens[best - 1], start: this.end, end: this.end, conf,
      integration: { sampleRate: this.sampleRate, mass: this.mass, contributions: this.contributions },
    };
    this.clear();
    return token;
  }

  private clear() {
    this.mass = 0;
    this.scores.fill(0);
    this.contributions = [];
  }
}
