import { PHRASE_SWITCH, PosteriorFrames, createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { PendingEdit, PendingEditWindow, PendingEditEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";

const add = (a: number, b: number): number => a === -Infinity ? b : b === -Infinity ? a
  : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

export interface JointEditPosterior {
  replacement: number;
  correctWithNeighbour: number;
  unchanged: number;
  independentReplacement: number;
  paths: number;
  veto: boolean;
}

/** Closed union of complete CTC chains. Sum over edit identity and all acoustic
 * boundaries together. Each chain consumes every frame once; its edit priors
 * are applied outside the frame recurrence, once per event (including repeats).
 * Too many alternatives yield unavailable evidence, never a pruned certificate.
 */
export function jointPendingEditPosterior(index: ReferenceIndex, frames: PosteriorFrames,
  window: PendingEditWindow, maxPaths = 128): JointEditPosterior | null {
  const { target, wordFrom, wordTo, fromFrame, toFrame } = window;
  if (target.kind !== "replace" || wordFrom < 0 || wordTo > index.words.length || wordTo <= wordFrom ||
    ![wordFrom, wordTo, fromFrame, toFrame, target.word].every(Number.isInteger) ||
    target.word < wordFrom || target.word >= wordTo || fromFrame < frames.startFrame ||
    toFrame > frames.endFrame || toFrame <= fromFrame || !Number.isInteger(maxPaths) || maxPaths < 2) return null;
  const first = index.words[wordFrom], last = index.words[wordTo - 1];
  if (first.surah !== last.surah || last.ayah < first.ayah || last.ayah > first.ayah + 1) return null;
  const ids = new Map([...index.tokens].map((p, i) => [p, i + 1]));
  const key = (e: PendingEdit) => `${e.kind}:${e.word}:${e.observed}`;
  const neighbours = [...new Map(window.neighbours.map(e => [key(e), e])).values()];
  if (neighbours.some(e => e.word === target.word && e.kind === "replace")) return null;
  const edits = [target, ...neighbours];
  if (edits.some(e => e.word < wordFrom || e.word >= wordTo || !e.observed ||
    (e.kind !== "replace" && e.kind !== "insert") || !Number.isInteger(e.word) ||
    !Number.isInteger(e.fromFrame) || !Number.isInteger(e.toFrame) ||
    e.fromFrame < fromFrame || e.toFrame > toFrame || e.toFrame <= e.fromFrame ||
    [...e.observed].some(p => !ids.has(p)))) return null;
  if ([...index.phonemes.slice(index.words[wordFrom].start, index.words[wordTo - 1].end)].some(p => !ids.has(p))) return null;
  // Alternatives for one event are mutually exclusive. An insertion and a
  // replacement at its following word remain compatible, even if the greedy
  // acoustic spans overlap: inference is precisely what decides their timing.
  const compatible = (chosen: PendingEdit[], e: PendingEdit) =>
    !chosen.some(c => c.word === e.word && c.kind === e.kind);
  const subsets: PendingEdit[][] = [[]];
  for (const e of edits) {
    const extra = subsets.filter(s => compatible(s, e)).map(s => [...s, e]);
    if (subsets.length + extra.length > maxPaths) return null;
    subsets.push(...extra);
  }
  const acoustic = new Map<string, number>();
  const score = (chosen: PendingEdit[]) => {
    const labels: number[] = [], repeats: [number, number][] = [];
    let ayahStart = 0, previousAyah = "";
    const append = (s: string) => {
      const start = labels.length;
      labels.push(...[...s].map(p => ids.get(p)!));
      if (labels.length > start) repeats.push([start, labels.length - 1]);
    };
    for (let w = wordFrom; w < wordTo; w++) {
      const word = index.words[w];
      const ayah = `${word.surah}:${word.ayah}`;
      if (ayah !== previousAyah) {
        if (labels.length > ayahStart) repeats.push([ayahStart, labels.length - 1]);
        ayahStart = labels.length; previousAyah = ayah;
      }
      const insert = chosen.find(e => e.word === w && e.kind === "insert");
      if (insert) append(insert.observed);
      const replace = chosen.find(e => e.word === w && e.kind === "replace");
      append(replace?.observed ?? index.phonemes.slice(word.start, word.end));
    }
    if (labels.length > ayahStart) repeats.push([ayahStart, labels.length - 1]);
    const k = JSON.stringify([labels, repeats]);
    let likelihood = acoustic.get(k);
    if (likelihood === undefined) { likelihood = -frames.nll(labels, fromFrame, toFrame, repeats); acoustic.set(k, likelihood); }
    return likelihood - chosen.length * PHRASE_SWITCH;
  };
  let replacement = -Infinity, correct = -Infinity, unchanged = -Infinity, z = -Infinity;
  for (const chosen of subsets) {
    const s = score(chosen);
    z = add(z, s);
    if (chosen.includes(target)) replacement = add(replacement, s);
    else if (chosen.length) correct = add(correct, s);
    else unchanged = s;
  }
  if (!Number.isFinite(z)) return null;
  const control = score([target]);
  const independentTotal = add(control, unchanged);
  if (!Number.isFinite(independentTotal)) return null;
  const independentReplacement = Math.exp(control - independentTotal);
  return { replacement: Math.exp(replacement - z), correctWithNeighbour: Math.exp(correct - z),
    unchanged: Math.exp(unchanged - z), independentReplacement, paths: subsets.length,
    veto: independentReplacement > 0.5 && correct > replacement };
}

/** Discover from the primary engine only. Neighbours are its existing lexical
 * substitutions and additions, including candidates not yet independently
 * confirmed. The largest available two-ayah snapshot retains their uncertainty.
 */
export function discoverPendingEditWindows(chunks: PhonemeEvidence[], index: ReferenceIndex,
  phrases?: PhraseAsset): PendingEditWindow[] {
  const windows = new Map<string, PendingEditWindow>();
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases, jointPendingEdit() { return false; }, pendingEdits(window) {
    const key = `${window.target.word}:${window.target.fromFrame}:${window.target.observed}`;
    const previous = windows.get(key);
    if (!previous || window.toFrame - window.fromFrame > previous.toFrame - previous.fromFrame) windows.set(key, window);
  } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  return [...windows.values()];
}

/** Replays the primary evidence with the SDK scorer; exports only successful horizon comparisons.
 * Tokens and posterior rows retain their identity. No side model, labels or free-phone decoding is used. */
export function addPendingEditEvidence(chunks: PhonemeEvidence[], index: ReferenceIndex,
  phrases?: PhraseAsset): PhonemeEvidence[] {
  const certificates: PendingEditEvidence[] = [];
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases, jointPendingEdit(window, frames, availableFrame) {
    const result = jointPendingEditPosterior(index, frames, window);
    if (result?.veto) {
      const w = index.words[window.target.word];
      certificates.push({ word: { surah: w.surah, ayah: w.ayah, word: w.word },
        observed: window.target.observed, fromFrame: window.target.fromFrame, toFrame: window.target.toFrame,
        availableFrame, replacement: result.replacement, correctWithNeighbour: result.correctWithNeighbour,
        independentReplacement: result.independentReplacement });
    }
    // Discovery cannot remove a primary accusation or change another edit's confirmation procedure.
    return false;
  } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const first = chunks.findIndex(c => c.phonemes.length);
  if (!certificates.length || first < 0) return chunks;
  return chunks.map((c, i) => i !== first ? c : { ...c, phonemes: c.phonemes.map((p, j) => j ? p :
    { ...p, pendingEdits: certificates }) });
}
