import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, EngineEvent, PhonemeEvidence, ReferenceIndex, WrongWordAccusation, WrongWordAgreement } from "@tilawa/core";
import type { LoadedModel } from "./model.js";
import type { LockPoint } from "./gap-decode.js";
import { decodePauseMasked, pauseMaskSpans } from "./pause-mask.js";
import { assertPrimaryEvidence } from "./primary-invariant.js";
import { insertionCells } from "./seam-insertion.js";

/** Rejudge with the primary tracker's positions. The side ear never locates the reciter. */
export function rejudgeWrongWords(primary: PhonemeEvidence[], side: PhonemeEvidence[], index: ReferenceIndex,
  options: Omit<CheckerOptions, "wrongWord"> = {}, mode: "free" | "memorization" = "free", acquireMs?: number): WrongWordAgreement[] {
  const strip = (chunks: PhonemeEvidence[]) => chunks.map(e => ({ ...e,
    phonemes: e.phonemes.map(({ wrongWordAgreements: _, ...p }) => p) }));
  const cells = insertionCells(strip(primary)), other = insertionCells(strip(side));
  if (!cells.length || !other.length) return [];
  if (other.some(e => e.inventory !== index.inventoryId || e.frameMs !== cells[0].frameMs))
    throw new Error("Wrong-word side ear inventory or clock differs from the primary");
  const proposed: WrongWordAccusation[] = [], judged: WrongWordAccusation[] = [];
  const tracker = createTracker(index, { mode, phrases: options.phrases, ...(acquireMs === undefined ? {} : { acquireMs }) });
  const checker = createChecker(index, { ...options, agreementClock: acquireMs ?? 0,
    wrongWord: (c, _id, graph) => { if (!graph) proposed.push(c); } });
  const review = createChecker(index, { ...options, agreementClock: acquireMs ?? 0, wrongWord: c => judged.push(c) });
  const timeline: { until: number; events: EngineEvent[] }[] = [];
  for (const e of cells) {
    const events = tracker.push(trackerInput(e));
    timeline.push({ until: e.posteriors!.startFrame + e.posteriors!.nFrames, events });
    checker.push(e, events);
  }
  const ending = tracker.end();
  checker.end(ending);
  let at = 0;
  for (const e of other) {
    const until = e.posteriors!.startFrame + e.posteriors!.nFrames;
    const events: EngineEvent[] = [];
    while (at < timeline.length && timeline[at].until <= until) events.push(...timeline[at++].events);
    review.push(e, events);
  }
  review.end([...timeline.slice(at).flatMap(t => t.events), ...ending]);
  return proposed.map(primary => {
    // Keep the entire side slot, including any independently drawn boundary.
    const rejudged = judged.filter(c => c.word.surah === primary.word.surah && c.word.ayah === primary.word.ayah &&
      c.word.word === primary.word.word && c.fromFrame < primary.toFrame && primary.fromFrame < c.toFrame);
    const from = Math.min(primary.fromFrame, ...rejudged.map(c => c.fromFrame));
    const to = Math.max(primary.toFrame, ...rejudged.map(c => c.toFrame));
    return { acquireMs: acquireMs ?? 0, primary, rejudged, evidence: other.filter(e => e.posteriors!.startFrame < to &&
      e.posteriors!.startFrame + e.posteriors!.nFrames > from) };
  });
}

/** Same certificate carrier as the folded teacher gate; no primary field is replaced. */
export function carryWrongWordAgreements(primary: PhonemeEvidence[], agreements: WrongWordAgreement[]): PhonemeEvidence[] {
  const at = primary.findIndex(e => e.phonemes.length);
  if (at < 0 || !agreements.length) return primary;
  return primary.map((e, i) => i !== at ? e : { ...e,
    phonemes: e.phonemes.map((p, j) => j ? p : { ...p, wrongWordAgreements: agreements }) });
}

/** Decode-time gate: stored certificates cannot change folded teacher wins or create another call. */
export function assertAgreementOnlyVetoes(primary: PhonemeEvidence[], enriched: PhonemeEvidence[], index: ReferenceIndex,
  options: Omit<CheckerOptions, "wrongWord">, mode: "free" | "memorization", acquireMs?: number): void {
  assertPrimaryEvidence(primary, enriched);
  const tracker = createTracker(index, { mode, phrases: options.phrases, ...(acquireMs === undefined ? {} : { acquireMs }) });
  const settings = { ...options, wrongWord: undefined, agreementClock: acquireMs ?? 0 };
  const before = createChecker(index, settings), after = createChecker(index, settings);
  const a = insertionCells(primary), b = insertionCells(enriched);
  for (let i = 0; i < a.length; i++) {
    const events = tracker.push(trackerInput(a[i]));
    before.push(a[i], events); after.push(b[i], events);
  }
  const ending = tracker.end(); before.end(ending); after.end(ending);
  const unchanged = (c: ReturnType<typeof createChecker>) => c.mistakes().filter(e => e.errorClass !== "wrong_word");
  if (JSON.stringify(unchanged(before)) !== JSON.stringify(unchanged(after)))
    throw new Error("Pause-ear agreement changed folded teacher wins or another primary decision");
  const original = new Set(before.mistakes().map(e => JSON.stringify(e)));
  if (after.mistakes().some(e => !original.has(JSON.stringify(e)))) throw new Error("Pause-ear agreement created a warning");
}

/** The primary teacher certificates are complete before this read-only side pass begins. */
export async function addWrongWordAgreements(model: LoadedModel, audio: Float32Array, primary: PhonemeEvidence[],
  source: PhonemeEvidence[], locks: LockPoint[], index: ReferenceIndex,
  options: Omit<CheckerOptions, "wrongWord">, mode: "free" | "memorization", acquireMs: number): Promise<PhonemeEvidence[]> {
  if (!model.pauseModel || !primary.length || primary.some(e => !e.posteriors)) return primary;
  const manifestBefore = JSON.stringify(model.manifest), evidenceBefore = JSON.stringify(primary);
  const spans = pauseMaskSpans(source, locks, index);
  // Full trained stream, including unmasked words: this ear must agree at every string-path accusation.
  const side = await decodePauseMasked(model.pauseModel, audio, spans, primary[0].posteriors!.k);
  // Stored evidence is replayed on the core's native-cell acquisition clock. Keep both clocks explicit;
  // live SDK replay below uses the SDK's one-second acquisition clock.
  const enriched = carryWrongWordAgreements(primary, [
    ...rejudgeWrongWords(primary, side, index, options, mode),
    ...rejudgeWrongWords(primary, side, index, options, mode, acquireMs),
  ]);
  if (JSON.stringify(model.manifest) !== manifestBefore || JSON.stringify(primary) !== evidenceBefore)
    throw new Error("Pause-ear agreement mutated the primary manifest or evidence");
  // These checks run on every fresh decode, before its evidence can be saved and scored.
  assertAgreementOnlyVetoes(primary, enriched, index, options, mode);
  assertAgreementOnlyVetoes(primary, enriched, index, options, mode, acquireMs);
  return enriched;
}
