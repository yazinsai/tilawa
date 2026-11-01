import { readFileSync } from "node:fs";
import { expect, it, vi } from "vitest";
import { loadReferenceIndex } from "@tilawa/core";
import { syntheticFixture, withPosteriors } from "../../core/src/fixtures/index.js";
import { confirmGapWords } from "../src/gap-decode.js";
import { loadNodeModel } from "../src/node.js";

it("consults the word model only for uncertain additions in changed gaps and preserves the decoded evidence", async () => {
  const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")));
  const from = { surah: 2, ayah: 255, word: 1 }, to = { surah: 2, ayah: 256, word: 10 };
  const first = index.words.find(w => w.surah === 2 && w.ayah === 255 && w.word === 1)!;
  const target = index.words.find(w => w.surah === 2 && w.ayah === 255 && w.word === 23)!;
  const alien = index.words.find(w => w.surah === 108 && w.ayah === 1 && w.word === 3)!;
  const word = index.phonemes.slice(alien.start, alien.end), at = target.start - first.start;
  const f = syntheticFixture(index, { id: "word-evidence", from, to, chunkSize: 8, edits: [{ kind: "ins", at, text: word }] });
  const flat = withPosteriors(index, { ...f.evidence[0], phonemes: f.evidence.flatMap(e => e.phonemes) },
    j => j >= at && j < at + word.length ? [[f.evidence.flatMap(e => e.phonemes)[j].p, .45], ["", .5]] : undefined);
  const audio = new Float32Array((flat.posteriors!.nFrames + 1) * 640);
  const model = await loadNodeModel(new URL("./fixtures/tiny", import.meta.url).pathname);
  const compare = vi.fn(async (_audio: Float32Array, _rate: number, neighbours: readonly string[]) => ({
    word: index.isolatedWordForms!.find(w => !neighbours.includes(w))!, distance: .1, neighbourDistance: .5, radius: .3,
  }));
  model.wordEmbedding = { compare, release: async () => {} };
  try {
    expect(await confirmGapWords(model, audio, 16000, [flat], [flat], index)).toEqual([flat]);
    expect(compare).not.toHaveBeenCalled();
    const baseline = structuredClone(flat);
    const frame = flat.phonemes[at].start;
    (baseline.posteriors!.probs as number[])[frame * flat.posteriors!.k] += .001;
    const annotated = await confirmGapWords(model, audio, 16000, [baseline], [flat], index);
    expect(compare).toHaveBeenCalledTimes(1);
    expect(annotated[0].phonemes.filter(p => p.wholeWord)).toHaveLength(1);
    expect(annotated[0].posteriors).toEqual(flat.posteriors);
    expect(annotated[0].phonemes.map(({ wholeWord, ...p }) => p)).toEqual(flat.phonemes);
  } finally { await model.backend.release(); }
});
