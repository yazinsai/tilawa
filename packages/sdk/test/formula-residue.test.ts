import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { completeRecitationFormula, createChecker, lexicalWords, loadReferenceIndex, recitationFormulas } from "@tilawa/core";
import type { PhonemeEvidence, SeamInsertionCandidate } from "@tilawa/core";
import { syntheticFixture, withPosteriors } from "../../core/src/fixtures/index.js";
import { formulaResidues } from "../src/formula-residue.js";
import { addInsertionEvidence } from "../src/seam-insertion.js";
import type { LoadedModel } from "../src/model.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../core/assets/phrases-v1.json", import.meta.url), "utf8"));
const formulas = recitationFormulas(index, phrases);
const formula = phrases.phrases.find((p: { name: string }) => p.name === "takbir").p as string;
const word = "فَءَسقَينَاكُمُۥۥهُ";
const isWord = (s: string) => s === word;

describe("complete formula and residual word discovery", () => {
  it("uses the existing formula costs but pays for both observed boundaries", () => {
    expect(completeRecitationFormula(formula, formula)).toBe(true);
    expect(completeRecitationFormula(formula.replace("ُ", "ِ"), formula)).toBe(true);
    expect(completeRecitationFormula("غغغغغ" + formula, formula)).toBe(false);
    expect(completeRecitationFormula(formula + "غغغغغ", formula)).toBe(false);
  });
  it("finds either order and formulas on both sides without reusing tokens", () => {
    for (const [raw, from] of [[formula + word, formula.length], [word + formula, 0],
      [formula + word + formula, formula.length]] as const) {
      expect(formulaResidues(raw, formulas, isWord)).toContainEqual({ from, to: from + word.length });
    }
  });

  it("accounts for repeated formulas while keeping formula-only utterances exempt", () => {
    expect(formulaResidues(formula + formula + word, formulas, isWord))
      .toContainEqual({ from: formula.length * 2, to: formula.length * 2 + word.length });
    expect(formulaResidues(formula, formulas, () => true)).toEqual([]);
    expect(formulaResidues(formula + formula, formulas, () => true)).toEqual([]);
    // The unchanged tolerance can explain short adjoining speech as a formula.
    expect(formulaResidues(formula + "وَءِںںں", formulas, s => s === "وَءِںںں")).toEqual([]);
    expect(formulaResidues("", formulas, () => true)).toEqual([]);
    expect(formulaResidues(word, [], isWord)).toEqual([]);
  });

  it("rejects unaccounted speech, partial words and two residual words", () => {
    expect(formulaResidues(formula + word + "غغغغغغغغ", formulas, isWord)).toEqual([]);
    expect(formulaResidues(formula + word.slice(1), formulas, isWord)).toEqual([]);
    expect(formulaResidues(word + formula + word, formulas, isWord)).toEqual([]);
    // Finding a lexical part inside the formula cannot leave the rest unconsumed.
    expect(formulaResidues(formula, formulas, s => s === "ءَكبَر")).toEqual([]);
  });

  it("uses the same complete lexical and local-text exclusions supplied by the checker", () => {
    const lexical = lexicalWords(index);
    const at = index.words.findIndex((_, i) => lexical.surplusWord(i, word) && !lexical.fragmentOfSeam(i, word));
    expect(at).toBeGreaterThanOrEqual(0);
    const allowed = (s: string) => !!lexical.surplusWord(at, s) && !lexical.fragmentOfSeam(at, s);
    expect(formulaResidues(formula + word, formulas, allowed).some(r =>
      r.from === formula.length && r.to === formula.length + word.length)).toBe(true);
    const own = index.phonemes.slice(index.words[at].start, index.words[at].end);
    expect(formulaResidues(formula + own, formulas, allowed)).toEqual([]);
  });
});

// Same synthetic blurred-onset seam as the core excision tests. The formula and
// word share an utterance; only the unchanged outer pauses bound discovery.
const from = { surah: 44, ayah: 9, word: 1 }, to = { surah: 44, ayah: 11, word: 5 };
const seam = { surah: 44, ayah: 10, word: 1 };
const start = index.words.find(w => w.surah === from.surah && w.ayah === from.ayah && w.word === from.word)!.start;
const right = index.words.find(w => w.surah === seam.surah && w.ayah === seam.ayah && w.word === seam.word)!;
const options = { passage: { from, to }, phrases, policy: { localWindow: { nats: Infinity } } };
function evidenceFor(extra: string, startPause = true, endPause = true): PhonemeEvidence {
  const edits = [{ kind: "ins" as const, at: right.start - start, text: extra },
    ...(startPause ? [{ kind: "gap" as const, at: right.start - start, frames: 20 }] : [])];
  const evidence = syntheticFixture(index, { id: "formula-residue", from, to, edits }).evidence[0];
  const at = right.start - start + extra.length;
  if (endPause) for (let i = at; i < evidence.phonemes.length; i++) {
    const p = evidence.phonemes[i];
    evidence.phonemes[i] = { ...p, start: p.start + 20, end: p.end + 20 };
  }
  const written = evidence.phonemes[at].p;
  evidence.phonemes[at] = { ...evidence.phonemes[at], p: "غ", conf: 0.5 };
  return withPosteriors(index, evidence, j => j === at ? [["غ", 0.5], [written, 0.42], ["", 0.06]] : undefined);
}
function discover(evidence: PhonemeEvidence, composite: boolean) {
  const candidates: SeamInsertionCandidate[] = [];
  const checker = createChecker(index, { ...options,
    ...(composite && { seamUtterance: (raw: string, allowed: (s: string) => boolean) => formulaResidues(raw, formulas, allowed) }),
    seamInsertion(c) { candidates.push(c); return true; } });
  checker.push(evidence, []); checker.end();
  return candidates;
}
function replay(evidence: PhonemeEvidence) {
  const checker = createChecker(index, options);
  checker.push(evidence, []); checker.end();
  return checker.mistakes().filter(e => e.errorClass === "added_word" &&
    e.location.surah === seam.surah && e.location.ayah === seam.ayah && e.location.word === seam.word);
}

describe("formula residues through ordinary seam certification", () => {
  it("discovers a whole residual word where whole-utterance discovery fails", () => {
    for (const extra of [formula + word, word + formula, formula + word + formula]) {
      const evidence = evidenceFor(extra), original = structuredClone(evidence);
      expect(discover(evidence, false)).toHaveLength(0);
      expect(discover(evidence, true)).toMatchObject([{ said: word }]);
      expect(evidence).toEqual(original);
    }
  });

  it("replays only a matching passing teacher certificate without a discovery callback", () => {
    const evidence = evidenceFor(formula + word);
    const candidate = discover(evidence, true).find(c => c.said === word)!;
    expect(candidate).toMatchObject({ said: word });
    for (const [marginal, shift, observed, expected] of [
      [0.5001, 0, candidate.said, 1], [0.5, 0, candidate.said, 0], [0.49, 0, candidate.said, 0],
      [1, 1, candidate.said, 0], [1, 0, "كَتَبنَا", 0],
    ] as const) {
      const certified = structuredClone(evidence);
      certified.phonemes[0].seamInsertions = [{ word: seam, observed,
        fromFrame: candidate.fromFrame + shift, toFrame: candidate.toFrame, marginal }];
      // Exercise the product's JSON transport as well as the callback-free path.
      expect(replay(JSON.parse(JSON.stringify(certified)))).toHaveLength(expected);
    }
    expect(replay(evidence)).toHaveLength(0);
  });

  it("requires complete formula accounting during certificate replay", () => {
    const evidence = evidenceFor("غغغغغغغغ" + word);
    const at = right.start - start + 8;
    evidence.phonemes[0].seamInsertions = [{ word: seam, observed: word,
      fromFrame: evidence.phonemes[at].start, toFrame: evidence.phonemes[at + word.length - 1].end + 1, marginal: 1 }];
    expect(replay(evidence)).toHaveLength(0);
  });

  it("keeps formula-only utterances, unaccounted speech and missing outer pauses quiet", () => {
    for (const extra of [formula, formula + word + "غغغغغغغغ"])
      expect(discover(evidenceFor(extra), true)).toHaveLength(0);
    expect(discover(evidenceFor(formula + word, false), true)).toHaveLength(0);
    expect(discover(evidenceFor(formula + word, true, false), true)).toHaveLength(0);
  });

  it("routes SDK discovery to the existing teacher on original audio and preserves primary evidence", async () => {
    const primary = evidenceFor(formula + word), original = structuredClone(primary);
    const teacherPhones = structuredClone(primary);
    const onset = right.start - start + formula.length + word.length;
    teacherPhones.phonemes[onset] = { ...teacherPhones.phonemes[onset], p: index.phonemes[right.start], conf: 1 };
    const teacherRows = withPosteriors(index, teacherPhones).posteriors!;
    const samplesPerFrame = primary.frameMs * 16;
    const audio = Float32Array.from({ length: (teacherRows.nFrames + 20) * samplesPerFrame }, (_, i) => (i % 31) / 31);
    const audioBefore = audio.slice(), calls: number[] = [];
    const model = { manifest: { sampleRate: 16000 }, insertionTeacher: {
      async decode(samples: Float32Array): Promise<PhonemeEvidence> {
        expect(samples.buffer).toBe(audio.buffer);
        const fromFrame = (samples.byteOffset - audio.byteOffset) / Float32Array.BYTES_PER_ELEMENT / samplesPerFrame;
        calls.push(fromFrame);
        const nFrames = Math.min(samples.length / samplesPerFrame, teacherRows.nFrames - fromFrame);
        return { inventory: primary.inventory, frameMs: primary.frameMs, phonemes: [], posteriors: {
          ...teacherRows, startFrame: 0, nFrames,
          ids: Array.from(teacherRows.ids).slice(fromFrame * teacherRows.k, (fromFrame + nFrames) * teacherRows.k),
          probs: Array.from(teacherRows.probs).slice(fromFrame * teacherRows.k, (fromFrame + nFrames) * teacherRows.k),
        } };
      },
    } } as unknown as LoadedModel;
    const enriched = await addInsertionEvidence(model, audio, [primary], index, phrases);
    expect(calls.length).toBeGreaterThan(0);
    const certificate = enriched.flatMap(e => e.phonemes).flatMap(p => p.seamInsertions ?? []).find(c => c.observed === word)!;
    expect(certificate).toMatchObject({ word: seam, observed: word });
    expect(certificate.marginal).toBeGreaterThan(0.5);
    expect(enriched[0].posteriors).toBe(primary.posteriors);
    expect(enriched[0].phonemes.map(({ seamInsertions: _, ...p }) => p)).toEqual(primary.phonemes);
    expect(primary).toEqual(original);
    expect(audio).toEqual(audioBefore);
  });
});
