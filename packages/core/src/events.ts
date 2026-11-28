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
  /**
   * Words shown earlier that the reader only began and then left (`TrackerRefinements.withdrawBegun`), or said as
   * another place, when a take's start moves from the declared opening to the ayah the reader started at
   * (`TrackerRefinements.declaredMove`): show them dark again. Set on the locked event that first shows the run the
   * reader moved to, which never covers them. The positions to show (`Tracker.shown`) name only words they showed, each
   * once. A later locked event whose run covers one of them (from its span's start to its word) shows it again.
   */
  withdrawn?: WordRef[];
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

/**
 * A mistake is `suspected` first, then ends in one of the other three, which are final: `confirmed`, `resolved` (the
 * reader corrected themselves, or a skip was recovered) or `uncertain` (too unclear to judge). A suspected mistake can
 * be sent `suspected` again with {@link CheckerEvent.due} before it ends.
 *
 * `suspected` is not for display. It is the checker's working state: raised as soon as the alignment sees a possible
 * mistake and withdrawn (`resolved`, `uncertain`) as often as not, while the reciter reads on, corrects themselves or
 * moves, and the alignment window can raise one outside the passage. Show a mistake once it is `confirmed`, or once it
 * is `due` (an omission as a word still owed), taken down if it then ends `resolved` or `uncertain`.
 */
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
  /**
   * Set on a `suspected` mistake in a declared passage that is likely to be confirmed: an omission (`omitted_word`,
   * `skipped_passage`) once the reciter has read on past it (two words after it heard right, both begun after the
   * suspicion; `refinements.omissionDue`), a wrong or added word once its evidence has stood unchanged and is sure
   * (`refinements.markWhenSure`). The same id was sent `suspected` before; confirmation still waits for the usual rule
   * and the mistake can still end `resolved` (the reciter goes back and says it, or the reading that follows explains
   * it). Provisional: an app may show it (an omission as a word still owed) and must take it down when the same id ends
   * `resolved` or `uncertain`. Absent on every other event.
   */
  due?: true;
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
