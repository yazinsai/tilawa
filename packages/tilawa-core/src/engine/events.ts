import type { AlignmentOp } from "./align.js";

export const EVENT_SCHEMA_VERSION = 1;

export interface WordRef {
  surah: number;
  ayah: number;
  word: number;
}

export interface WordSpan {
  start: WordRef;
  end: WordRef;
}

export interface PositionCandidate {
  /** The most recently recited word. */
  word: WordRef;
  /** Words covered by the evidence so far; may cross ayahs. */
  span: WordSpan;
  confidence: number;
  /** Weighted alignment cost per observed phoneme, used only for ranking. 0 is an exact match. */
  cost: number;
  /** Unweighted phoneme error rate against the matched span, from the same alignment path. */
  per: number;
  /** Set when a leading basmala was recited before the span; it is excluded from `span` and `per`. */
  prefix?: "basmala";
}

export type TrackerState = "acquiring" | "locked" | "ambiguous" | "lost";

export interface TraceCandidate {
  span: WordSpan;
  /** Half-open phoneme range in the reference index. */
  offset: number;
  length: number;
  votes: number;
  cost: number;
  per: number;
}

export interface TrackerTrace {
  shortlist: TraceCandidate[];
  transition: { from: TrackerState | null; to: TrackerState };
  /** Alignment of the observed phonemes against the top candidate; null without a candidate. */
  path: AlignmentOp[] | null;
  /** The top fitting candidate's confidence terms; null when nothing fits. `confidence = fit * separation`. */
  decision: { fit: number; separation: number; leadEdits: number; requiredEdits: number } | null;
}

export interface PositionEvent {
  type: "position";
  v: typeof EVENT_SCHEMA_VERSION;
  timeMs: number;
  referenceVersion: string;
  expected: WordRef | null;
  /** Ranked best first; deduplicated so each entry is a distinct location. */
  observed: PositionCandidate[];
  state: TrackerState;
  /**
   * Set on the locked event that starts a misfit hold. An open issue stays open only when its own edit makes the
   * heard words after it fit the anchor again; every other open issue ends `uncertain`, as on a lock drop. The
   * checker's window stays so tokens heard after this are still judged.
   */
  hold?: "start";
  /**
   * A slot-differing parallel span resolved. `from` and `to` are the member span starts. Equal means the
   * continuation confirmed the locked copy, and held slot accusations are judged against it. Different means the
   * heard word fit only the other member: the span is re-read against that copy.
   */
  relocated?: { from: WordRef; to: WordRef };
  /**
   * The session's first lock has not yet heard a word outside every slot-differing span. Slot accusations wait
   * for the continuation. Absent once that word arrives, and on every later lock.
   */
  spanHold?: true;
  /** Present only when the tracker was created with `trace: true`. */
  trace?: TrackerTrace;
}

export type MemorizationErrorClass = "wrong_word" | "omitted_word" | "added_word" | "skipped_passage";

export type PronunciationErrorClass =
  | "short_vowel"
  | "consonant_h_ha"
  | "consonant_ain_hamza"
  | "consonant_sad_sin"
  | "consonant_dad_dal"
  | "consonant_ta_ta"
  | "consonant_qaf_kaf"
  | "consonant_dhal_zay"
  | "consonant_tha_sin"
  | "missing_shadda"
  | "madd_natural";

export type ErrorClass = MemorizationErrorClass | PronunciationErrorClass;

export type CheckerState = "suspected" | "confirmed" | "resolved" | "uncertain";

export interface CheckerEvent {
  type: "checker";
  v: typeof EVENT_SCHEMA_VERSION;
  id: string;
  timeMs: number;
  state: CheckerState;
  errorClass: ErrorClass;
  location: WordRef;
  expected?: string;
  observed?: string;
  confidence: number;
}

export interface CapabilityEvent {
  type: "capabilities";
  v: typeof EVENT_SCHEMA_VERSION;
  modelVersion: string;
  referenceVersion: string;
  supported: ErrorClass[];
  unsupported: ErrorClass[];
}

export type EngineEvent = PositionEvent | CheckerEvent | CapabilityEvent;
