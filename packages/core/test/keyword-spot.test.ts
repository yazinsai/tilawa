import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "../src/ctc.js";
import { withPosteriors } from "../src/fixtures/index.js";
import type { DecodedPhoneme, PhonemeEvidence } from "../src/evidence.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { ReferenceAsset } from "../src/reference.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const id = new Map([...index.tokens].map((t, i) => [t, i + 1]));
const ids = (s: string) => [...s].map((p) => id.get(p) ?? 0);

function framesOf(phonemes: DecodedPhoneme[], override?: (j: number) => [string, number][] | undefined): PosteriorFrames {
  const evidence: PhonemeEvidence = { inventory: index.inventoryId, frameMs: 20, phonemes };
  const chunk = withPosteriors(index, evidence, override ?? (() => undefined));
  const frames = new PosteriorFrames(index.tokens.size);
  frames.append(chunk.posteriors!);
  return frames;
}

describe("whole-sequence keyword spot", () => {
  const said = "بَيت";
  const phonemes: DecodedPhoneme[] = [...said].map((p, j) => ({ p, start: 3 * j, end: 3 * j + 1, conf: 0.9 }));

  it("costs nothing when the frames are that sequence, and a lot when they are not", () => {
    const frames = framesOf(phonemes);
    const present = frames.keywordExtra(ids(said), frames.startFrame, frames.endFrame);
    const absent = frames.keywordExtra(ids("شمشمشمشمشم"), frames.startFrame, frames.endFrame);
    expect(present).toBeLessThan(1);
    expect(absent).toBeGreaterThan(40);
  });

  it("can enter the sequence after leading filler", () => {
    const lead: DecodedPhoneme[] = [
      { p: "م", start: 0, end: 1, conf: 0.9 },
      { p: "م", start: 3, end: 4, conf: 0.9 },
      ...phonemes.map((p) => ({ ...p, start: p.start + 9, end: p.end + 9 })),
    ];
    const frames = framesOf(lead);
    expect(frames.keywordExtra(ids(said), frames.startFrame, frames.endFrame)).toBeLessThan(1);
  });
});
