/** Inspect a benchmark's component boundaries without changing its data or scores. */
import { createChecker } from "../checker.js";
import type { CheckerPolicy } from "../checker.js";
import { createTracker } from "../tracker.js";
import type { TrackerOptions } from "../tracker.js";
import type { CheckerEvent, EngineEvent, WordRef } from "../events.js";
import type { PhonemeEvidence } from "../evidence.js";
import type { PhraseAsset, ReferenceIndex } from "../reference.js";
import { chunk } from "./index.js";
import type { QudRecording } from "./index.js";
import type { ClipEvidence } from "./jumps.js";

export interface ReplaySettings {
  chunkMs: number;
  acquireMs: number;
  minConfidence: number;
  confirmMs: number;
  localWindow: boolean;
  finaliseFree: boolean;
}
export const REPLAY_DEFAULTS: ReplaySettings = { chunkMs: 320, acquireMs: 1000, minConfidence: 0.6, confirmMs: 2500, localWindow: true, finaliseFree: true };
export interface ReplayStep {
  index: number;
  fromMs: number;
  toMs: number;
  phase: "push" | "end";
  listenerOutput: PhonemeEvidence | null;
  trackerInput: PhonemeEvidence | null;
  trackerOutput: EngineEvent[];
  checkerInput: { evidence: PhonemeEvidence | null; trackerEvents: EngineEvent[] };
  checkerOutput: CheckerEvent[];
}
export function replaySettings(value: unknown): ReplaySettings {
  if (value === undefined) return { ...REPLAY_DEFAULTS };
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Test settings must be an object.");
  const v = value as Record<string, unknown>;
  if (Object.keys(v).some(k => !Object.hasOwn(REPLAY_DEFAULTS, k))) throw new Error("Unknown test setting.");
  const out = { ...REPLAY_DEFAULTS, ...v } as ReplaySettings;
  for (const [key, min, max] of [["chunkMs", 40, 2000], ["acquireMs", 40, 5000], ["minConfidence", 0, 1], ["confirmMs", 0, 10000]] as const) {
    if (typeof out[key] !== "number" || !Number.isFinite(out[key]) || out[key] < min || out[key] > max) throw new Error(`Invalid ${key}: use ${min}–${max}.`);
  }
  if (typeof out.localWindow !== "boolean" || typeof out.finaliseFree !== "boolean") throw new Error("Confirmation settings must be true or false.");
  return out;
}
export function replayComponents(index: ReferenceIndex, phrases: PhraseAsset, evidence: ClipEvidence, passage: { from: WordRef; to: WordRef } | null, settings: ReplaySettings) {
  const trackerOptions: TrackerOptions = { mode: "memorization", phrases, acquireMs: settings.acquireMs, trace: true };
  const policy: CheckerPolicy = { minConfidence: settings.minConfidence, confirm: { after: "delay", ms: settings.confirmMs }, localWindow: settings.localWindow, finaliseFree: settings.finaliseFree };
  const tracker = createTracker(index, trackerOptions);
  const checkerOptions = { phrases, policy, ...(passage ? { passage } : {}) };
  const checker = createChecker(index, checkerOptions);
  const steps: ReplayStep[] = [];
  let fromMs = 0;
  for (const e of chunk(evidence as QudRecording, settings.chunkMs / 1000, !!evidence.posteriors?.ids.length)) {
    // This is the same boundary as scoreClip: tracker sees the greedy decode; checker also sees posteriors.
    const { posteriors: _, ...trackerInput } = e;
    const trackerOutput = tracker.push(trackerInput);
    const checkerOutput = checker.push(e, trackerOutput);
    const toMs = (e.decodedTo ?? 0) * e.frameMs;
    steps.push({ index: steps.length, fromMs, toMs, phase: "push", listenerOutput: e, trackerInput, trackerOutput, checkerInput: { evidence: e, trackerEvents: trackerOutput }, checkerOutput });
    fromMs = toMs;
  }
  const trackerOutput = tracker.end();
  steps.push({ index: steps.length, fromMs, toMs: fromMs, phase: "end", listenerOutput: null, trackerInput: null, trackerOutput, checkerInput: { evidence: null, trackerEvents: trackerOutput }, checkerOutput: checker.end(trackerOutput) });
  const final = [...new Map(steps.flatMap(s => s.checkerOutput).map(e => [e.id, e])).values()];
  return { settings, trackerOptions: { ...trackerOptions, phrases: phrases.version }, checkerOptions: { passage, policy, phrases: phrases.version }, steps, final };
}
