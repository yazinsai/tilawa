import type { AlignmentOp } from "./align.js";
import { PosteriorFrames } from "./ctc.js";
import { CONSONANTS, MID_WORD_START, UNIT, alignSubstring, collapseHeld, fitBounds, growingFit, insertionCost, isConfusable, isHeld } from "./fuzzy.js";
import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { EngineEvent, PositionCandidate, PositionEvent, TraceCandidate, TrackerState, TrackerTrace, WordRef } from "./events.js";
import { EvidenceError } from "./evidence.js";
import { minGapFrames } from "./gap-decode.js";
import type { DecodedPhoneme, PhonemeEvidence } from "./evidence.js";
import { createLocalSearch } from "./local.js";
import type { LocalPosition, LocalSearch } from "./local.js";
import type { AyahRef, PhraseAsset, ReferenceIndex } from "./reference.js";
import { letterSkeleton } from "./parallel-phrases.js";
import { continuationDecision, longestSpanAt, spanId, slotSpans } from "./slot-spans.js";
import type { SlotSpan } from "./slot-spans.js";

export interface TrackerOptions {
  mode?: "free" | "memorization";
  /** Grams occurring more often than this carry too little location information to vote. */
  maxGramOccurrences?: number;
  maxCandidates?: number;
  /** Fit: a candidate fits only with PER at most this. Default 0.35. */
  maxPer?: number;
  /** Fit: and weighted cost per observed phoneme at most this. Default {@link MISFIT_MAX_COST}. */
  maxCost?: number;
  /** Evidence: `locked` needs at least this many observed phonemes. Default 10. */
  minPhonemes?: number;
  /** Evidence: and a span of at least this many words. Default 2. */
  minWords?: number;
  /**
   * Separation: `locked` needs the best distinct candidate to trail by `max(minSeparation, separationScale / n)`
   * plain edits for `n` observed phonemes, so short clips need a wider margin. Defaults {@link COMMIT_MIN_SEPARATION}
   * and {@link COMMIT_SEPARATION_SCALE}.
   */
  minSeparation?: number;
  separationScale?: number;
  /** `locked` also needs the top candidate's confidence at least this; a unique but poor fit stays `acquiring`. Default 0.08. */
  minLockConfidence?: number;
  /** Bonus for the intended or filename ayah (full) and its neighbours (half). Bounded so it can't override fit or separation. */
  prior?: { surah: number; ayah: number; weight: number };
  /** Full prior bonus, uniformly, for any ayah inside one of the inclusive ranges (a surah or a page); `ayah: Infinity` ends at the surah's last ayah. Same cap as `prior`. */
  priorRange?: { ranges: { from: AyahRef; to: AyahRef }[]; weight: number };
  /**
   * Once locked, a distant location must hold a global lock for this many consecutive observed phonemes before the
   * tracker jumps there. Default 8.
   */
  jumpPhonemes?: number;
  /**
   * Non-Quran phrases to recognise and skip in a stream. A basmala outside 1:1 and 27:30 is skipped too, and then
   * favours surah openings. Skipping only acts on earlier pushes, so a single-push clip is unaffected.
   */
  phrases?: PhraseAsset;
  /** Attach shortlist, state transition and alignment path to every position event. */
  trace?: boolean;
  /**
   * Streaming: before the first lock, run global search once per this much audio on a fixed grid of evidence time
   * rather than once per push. How often a stream is searched decides how short a prefix can lock, so with this set
   * the events don't depend on how the caller chunks evidence. Default: once per push (offline locate of a whole clip).
   */
  acquireMs?: number;
  /**
   * Memorization: the declared passage, from its first word (where the reciter should begin) to its last. With
   * `refinements.declaredStart` the first lock can come from its opening rather than from global search.
   */
  passage?: { from: WordRef; to: WordRef };
  /**
   * The tracker's refinements of rc.1's rules ({@link TrackerRefinements}): each on unless set false, but
   * `declaredStart` and `withdrawBegun`, off unless set true.
   */
  refinements?: TrackerRefinements;
}

/**
 * Refinements of the 1.0.0-rc.1 tracker's rules. Each is on unless set `false`, but `declaredStart` and
 * `withdrawBegun`, which are off unless set `true` (each says why); off, a field keeps the rc.1 rule it replaces, so its effect can be measured alone.
 */
export interface TrackerRefinements {
  /**
   * No advance on a guess. A later word entered on a risky step shows only once at least two of its phonemes (all of
   * a shorter word's) were heard as written, as a visit needs to count as reciting a word. A step is risky when its
   * phoneme was not heard as written, or when it is the first phoneme after a pause (a frame gap the checker's pause
   * test labels, `GAP_PAD_S`) and the decoder heard it below `BREATH_CONFIDENCE`: maybe a breath or a stray sound.
   * A word entered on a clean step shows on its first phoneme, as in rc.1, after a pause too, so a clean ayah opening
   * shows when rc.1 shows it. The same rule picks the word a lock shows (global search, a relock, the declared
   * opening), and the local search a lock starts counts the matches its path already heard in that word. Risk: a
   * correct word whose first phoneme was heard not as written, or faintly after a pause, shows a phoneme later.
   */
  confirmAdvance?: boolean;
  /**
   * A doubled onset's first half belongs to the word before. A word whose written start is a run of one sound (a
   * shadda: a sun letter, idgham, ٱللَّه) shows once a phoneme of the run's second half is heard: the second of a run
   * of two, the third of a run of three or four. In speech the run's first half closes the word before, which stays
   * shown until then. A word that is only such a run shows as before. Risk: such a word shows a phoneme or two later
   * than in rc.1; a held doubled sound the decoder hears as more phonemes than written does not delay it further.
   */
  onsetRun?: boolean;
  /**
   * Memorization: the take's first words, from the declared passage (`passage`). Before the first lock, the phonemes
   * heard from each onset (the take's first phoneme, or the first after a pause; leading phrases set aside) are also
   * aligned against the passage's opening, which may lock once they fit it as a landing must (`CONFIRM_COST` per
   * phoneme) over `DECLARED_MIN` phonemes (its whole first word when that is shorter) and no phrase still being said
   * fits as well. Every ayah's opening in the mushaf is a rival, judged by how well the speech heard so far fits it: a
   * rival is in play while its fit is less than half an edit worse than the declared opening's (`DISTINCT`), and told
   * apart once it is worse by more. While rivals are in play, the declared opening's words show only as far as every
   * rival in play shares them: the same word as written (the orthography's marks set aside: تَبَٰرَكَ is تَبَارَكَ),
   * the same sounds, or the same sounds split into other words that end together (وَإِن مَّا and وَإِمَّا). So of
   * ayahs that open with the same words, the declared one, the one next in order, lights as they are said, and the page
   * moves at the first word they do not share; while a rival whose first word differs, however alike it sounds, is in
   * play, nothing locks, and the opening shows once the speech tells them apart. The lock shows the word the local
   * search would show there (`onsetRun`, `confirmAdvance`), or the word before it when the phoneme that told them apart
   * is the first of the next word, and each later word shows as it is said. With `declaredLeadIn`, every word of the
   * lead-in tail (the ayahs just before the declared one) is a rival too. When a rival fits better by an edit, or the
   * speech no longer fits the declared opening as a landing must, the reciter started that other ayah: the lock is let
   * go and global search finds them, as after any wrong lock, or, for a rival near the passage, the lock moves there
   * (`declaredMove`). Sounds that are no phrase, said before a pause, do not hold the lock back: the opening is tried
   * from the onset after them, and not with them when they do not fit it. Off, a take's first words show once global
   * search finds the reciter, as in rc.1.
   *
   * Risks: once the lock is let go for a rival far from the passage (another surah's ayah that opens alike), global
   * search finds the reciter only from what it heard after the lock, as with `declaredMove` off; an opening that a
   * rival's differs from only in a held length (إِنَّ and إِنَّآ), or that is the start of a longer rival (الٓمٓ and
   * الٓمٓر, طسٓ and طسٓمٓ), shows its first word with the first sound of the next word, so a pause after it delays it
   * by the pause; a declared passage of one word whose pausal form opens longer ayahs (55:1 alone) is not shown by its
   * opening at all, but only once global search finds it, as in rc.1; a reciter who says an ayah that opens with the
   * same words as the declared one is shown the declared words they share while saying them, and the page moves when
   * the words part; a false start of the declared opening lights the words said in it; a declared opening that also
   * opens a phrase (ٱللَّهُ, the takbir's) locks once the speech leaves the phrase; a reciter who starts inside an ayah
   * elsewhere, not at its opening, with words that fit the declared opening, is shown on it until the local search
   * loses them; sounds said straight into the opening, with no pause, still delay the lock as in rc.1.
   */
  declaredStart?: boolean;
  /**
   * Memorization, with `declaredStart`: a reciter who leads in from just before the passage is not shown its first word
   * before saying it. Every word of the lead-in tail (the declared ayah's words before the declared word, and every word
   * of the `LEAD_AYAHS` (2) ayahs before it) is a rival of the declared opening as an ayah's opening is, fitted from its
   * first phoneme: while one whose first word differs is in play nothing locks, and once one fits the speech better by
   * an edit the lock is let go (or moves to it, `declaredMove`). Off, only ayah openings are rivals: a lead-in from
   * inside the ayah before (its last words, or going on after a breath inside it) whose sounds fit the declared opening
   * as a landing must is shown the declared first word before it is said. A tail word inside its ayah shares no word
   * with the declared opening: the declared ayah is the one next in order only to an ayah that opens alike, while a
   * reciter who leads in says the tail's words first. Risks: a reciter who leads in from further back is shown as with
   * the field off; a declared opening whose first words recur inside the tail shows only once the speech parts from
   * the tail, a word or two later.
   */
  declaredLeadIn?: boolean;
  /**
   * Memorization, with `declaredStart`: when the lock on the declared opening is let go because another ayah's opening
   * (or a word of the lead-in tail, `declaredLeadIn`) fits the speech better by an edit, and that rival lies near the
   * passage (from the lead-in tail to `NEAR_AYAHS` (2) ayahs past the passage's last, in its surah), the reciter
   * started there: the lock moves to it, on all the speech heard since the onset the declared lock was made on, as that
   * lock was made, and the words said there show. A declared opening that no longer fits as a landing must yields to a
   * near rival that does, the same way. While two or more near rivals fit alike (within half an edit), the speech has
   * not told them apart and the mushaf's order does not either (a lead-in from before the passage and a later start
   * open alike): the contest stays open, the declared words showing only as far as every rival in play shares them,
   * until one stands apart; past what a fit reads, the last before the declared ayah, else the first after it. The
   * declared words the page showed that the new run does not cover are named in its `withdrawn`. A move is made on one
   * onset's speech, so the first onset after a pause that follows it (or that ends a contest held open) is judged again
   * against the declared opening and its rivals, as the take's first is: when the declared opening locks there and the
   * run shown does not already cover its first word, the lock goes back to it, and the declared ayah lights from its
   * first word. A rival far from the passage lets go to global search, as with the field off. Off, every let-go goes to
   * global search, which reads only the speech heard after the lock: words said can stay dark, a lost-place prompt can
   * come while the reciter recites correctly, and words of another place can light. Risks: the declared words shown
   * before the move show, then go dark (an app that shows `withdrawn`), as for any lock let go; while near rivals fit
   * alike, the reciter's words past what they share wait for the words to part (or for a pause), and a pause that ends
   * the contest held open leaves the place to the declared run and the tracker's search.
   */
  declaredMove?: boolean;
  /**
   * Memorization: no `lost` before the place is first found. Until the first lock an assessment that fits nowhere is
   * reported `acquiring`; the tracker's own state is unchanged. The take's final event still reports `lost` when the
   * place was never found.
   */
  noLostBeforeLock?: boolean;
  /**
   * One syllable heard on weak evidence is not yet the word. A later word of the run entered on weak evidence (one of
   * the last `CONFIRM` phonemes up to its first not heard as written, or a risky step as `confirmAdvance` reads one),
   * whose phonemes heard so far another word's opening that a skip forward lands on (a later word of its ayah, or the
   * next ayah's first word; none past the declared passage's end with `passageWindow`) fits as well as the word itself
   * (`fitsElsewhere`), shows once two of its consonants (all of a word with fewer) were heard as written, a doubled
   * consonant (a shadda: a sun letter, idgham) counting once, or once its last phoneme is reached: the reciter may have
   * left it out and be saying that one. A word entered on phonemes heard as written shows on its first sound, as in
   * rc.1. Never an ayah's first word, which shows as before, and never a word the reference gives no consonant (a nasal
   * alone, ںںں). A held word is never kept dark, nor shown with the next: when the run goes on into the next word
   * before it has shown, it shows then, and the next word waits for its own consonants. A lock shows its last word by
   * the same rule (the word before it while that word is held), so the word shown does not depend on where the pushes
   * fall; a new run's first word keeps its own rule (`CONFIRM`). Risk: a correct word entered on a sound the decoder
   * did not hear as written, whose opening a word nearby shares, shows when its second consonant is heard, a phoneme or
   * a few later than in rc.1, and one whose consonants the decoder mishears shows at its last phoneme, or as the next
   * word begins.
   */
  consonantConfirm?: boolean;
  /**
   * A misheard opening that fits a word further on does not advance the run. While what was heard of a later word of
   * the run is not all as written, and the opening of another word that a skip forward lands on (a later word of its
   * ayah, or the next ayah's first word; none past the declared passage's end with `passageWindow`) fits it as well as
   * the word itself or better (`fitsElsewhere`), the word waits, until `RIVAL_MATCHES` of its phonemes (all of a
   * shorter word's) were heard as written or its last phoneme is reached: the reciter may have left it out and be
   * saying that one. A word whose phonemes are heard as written shows on its first sound, as in rc.1, even where
   * another word opens alike: by the owner's rule for look-alikes the order the reciter reads in decides, and the
   * tracker moves if they were saying the other. Never an ayah's first word, for the same reason. A held word shows as
   * `consonantConfirm` shows one, alone, as the run goes on past it, and a lock judges its last word the same way, from
   * what it heard of it. Risks: a correct word whose first sound the decoder misheard as a word nearby opens shows a
   * few phonemes later than in rc.1; a word left out whose next word opens with its sound still shows on that sound,
   * as in rc.1.
   */
  rivalHold?: boolean;
  /**
   * Memorization: no jump out of the declared passage (`passage`). While the local search's position is inside the
   * passage, none of its jumps lands past the passage's end, so a look-alike after it (a phrase of the next ayahs that
   * fits a sound or two of the passage's) takes no lock while the reciter is still reading the passage, and a global
   * lock past the end is not near it, however close (`NEAR_AYAHS`): one the local search cannot follow, which commits
   * as any jump does. A reciter who reads on past the passage's last word is followed there in order, as without it,
   * and is then not bounded; a return before the passage's start is followed as without it. Risk: a reciter who leaves
   * the passage by a skip forward, not by reading on, is followed there only once global search commits the jump
   * (`jumpPhonemes`), a few phonemes later than without it.
   */
  passageWindow?: boolean;
  /**
   * A restart is not a surah entry. A jump back to the first word of the surah the local search is in costs a plain
   * jump: only entering a surah from outside it (reading on from the one before, or jumping in from another) pays the
   * surah entry, which keeps what ends a surah from sliding into the next one's opening. Without it, a reciter who
   * starts the surah again pays for an entry they never made, and a look-alike before the surah's start (17:111
   * against 18:1) can take the restart. Risk: none known; entering a surah in order costs what it did.
   */
  restartNoEntry?: boolean;
  /**
   * A word only begun goes dark again once the reciter has clearly moved past it, so a word left out never looks said.
   * When the run that shows a word leaves it by a jump forward (the run the reciter moved to starts after it) while that
   * run had only begun it (not said by the tracker's test of a word said, `begunByConsonants`), the locked `position`
   * event that first shows the new run names it in `withdrawn`, and an app darkens it. The word is judged as the run's
   * path left it, just before the first sound the new run reads as its own (`LocalPosition.since`), not after the path
   * went on to take those sounds as more of it; a later word the old run showed on those sounds is withdrawn too
   * (`leftOnNewAudio`). So too an ayah's first word shown on the onset of a restart, once the run goes back before it (a
   * jump back to the run's own first word starts a run too). Only a word the run that leaves it showed first is
   * withdrawn: a word an earlier run showed (the reciter said it before going back over it) stays shown. A run is the
   * run's own: a lock after a position that held no place (a lock let go, a lost place) starts a run, even on the words
   * the run before it showed, and a word the run showed with no step of its path in it (shown by the lock that started
   * it, or passed over) is kept, since nothing says it was only begun. A word the new run covers (from its first word to
   * its word) is never withdrawn, so the run that withdraws a word never shows it, and a later run that covers it (the
   * reciter says it after all) shows it again. Words the path read on past in order, and words heard in full (the shared
   * words of ayahs that open alike among them, shown by the order the reciter reads in), stay shown. Off, a word stays
   * shown once shown, as in rc.1. Risks: a short word said whose consonant the decoder misheard, after which the run
   * jumps forward, goes dark although said, until a run covers it again; an ayah's first word guessed on two clean consonants
   * of a restart stays shown; a word begun at the take's end, with no run after it, stays shown; a word begun, gone
   * back over and begun again before the skip stays shown, as an earlier run showed it.
   */
  withdrawBegun?: boolean;
  /**
   * With `withdrawBegun`: a later word the run that leaves it showed, which its path reached only after the first sound
   * the new run reads as its own, goes dark whatever was heard of it: what lit it was the new run's speech read against
   * the old run's words. Off, such a word goes dark only when only begun there, as the word the path was in. Risk: a
   * word said just before a jump, whose sounds the new run's search also takes as its own, goes dark until a run covers
   * it again.
   */
  leftOnNewAudio?: boolean;
  /**
   * With `withdrawBegun`: the word a jump leaves was only begun unless the tracker's test of a word said
   * (`showSaid`'s) passes: the path has heard it through its last consonant, every consonant (a doubled one once) or all
   * but one in a word of three or more. Judging a word the run has left, it gives the doubt to the reciter: a consonant
   * heard as its engine-confusable partner counts as heard, and the one a longer word may miss may be its last (the
   * path then stands just before it). So a word left after its first syllable goes dark, and a word said in full with
   * a consonant misheard or dropped stays shown. Off, only a word's onset counts as begun: no more than
   * `ADVANCE_MATCHES` (2) of its phonemes and fewer than `CONFIRM_CONSONANTS` (2) of its consonants (all of a word with
   * fewer) heard as written, so a word whose first syllable and next consonant were said before the skip stays shown.
   * Risks: a word of one or two consonants said with one misheard as other than its confusable partner goes dark when
   * the run then jumps forward, until a run covers it again; a longer word left just before its last consonant, or with
   * all but one of its consonants said, stays shown.
   */
  begunByConsonants?: boolean;
  /**
   * A word shows once the reciter has said it, not on its first sound. The positions to show ({@link Tracker.shown})
   * light a run's words through the last one said. A word inside an ayah is said once the next word is under way:
   * `NEXT_BEGUN` (3) of its phonemes heard, its whole length when shorter. The decoder places a phoneme at its first
   * sound, so a word's last phoneme heard does not mean the word is finished, and its last sounds can run into the next
   * word's first (a doubled onset, a nasal merged into the next consonant, a joined hamzat al-wasl); the next word's
   * first syllable heard means it is. An ayah's last word, which a pause follows, is said once heard to its end: its
   * phonemes heard as written through its last one (of a final held vowel, its first phoneme: the hold need not be
   * waited for), or the next ayah begun. At a pause (no phoneme for 0.48 s, longer than the decoder's gap since a held
   * sound decodes as its start and then quiet; shown when decoding reaches it: a `push()` whose `decodedTo` is that
   * far, or {@link Tracker.silence}, by which a session tells the silence its listener decodes), the word the reciter
   * stopped on shows once heard as written to its end (a reciter who stops on a word drops its final short vowel); at
   * the take's end and when the page leaves the run (a jump, a lost place), once all its consonants have been heard.
   * Otherwise the word before it, left, shows if the path heard it as written (a word left out, or read from other
   * sounds, is not said). A run's
   * first word shows once it is said; until then the page goes on showing what it showed. A word only begun, then left
   * for a later one, never lights. A run that moves to a twin copy of the same words keeps what was said of the first.
   * Every other rule of what shows (`onsetRun`, `confirmAdvance`, `consonantConfirm`, `rivalHold`, `declaredStart`)
   * still holds: a word shows once said and once they would show it. `push()` and `end()` still return the run's own
   * positions (where the reciter is, the word being said included), which the checker reads; only what shows changes.
   * Off, the positions to show are those positions, as in rc.1. Risks: a word shows about one syllable after it ends,
   * a little later than its end (before a pause, 0.48 s after it; a word before a pause heard otherwise than as written
   * shows only once the next word is under way); a reciter who stops in the middle of a word after its last consonant
   * (a cut final vowel) is shown the word at the pause; a word said with two of its sounds misheard, then left by a jump, shows only
   * once a later word of its run is said.
   */
  showSaid?: boolean;
}

export interface Tracker {
  push(evidence: PhonemeEvidence): EngineEvent[];
  end(): EngineEvent[];
  /**
   * The positions to show for the last `push()` or `end()`, in order: what an app lights. With
   * `refinements.showSaid` (on by default), a word lights once it has been said, and a position is added when a word is
   * finished; without it, they are the positions `push()` and `end()` returned.
   */
  shown(): PositionEvent[];
  /**
   * Decoding has passed frame `decodedTo` with no phoneme since the last `push()`: time went by in silence, which a
   * caller that pushes only phonemes tells this way (a session does, as its listener decodes). It changes no position
   * and returns none for the checker. With `refinements.showSaid` (on by default) it returns the positions to show that
   * the silence settles: a pause after a word heard as written shows the word once the quiet after its last sound is a
   * pause's (0.48 s). Without it, nothing.
   */
  silence(decodedTo: number): PositionEvent[];
}

/** An aligned phoneme's frames and decoder confidence, by its index in the aligned string (undefined outside it). */
type HeardAt = (at: number) => { start: number; end: number; conf: number } | undefined;

interface Match {
  offset: number;
  length: number;
  /** Weighted cost in `UNIT`s. */
  cost: number;
  per: number;
  votes: number;
  path: AlignmentOp[];
  basmala?: boolean;
  /** The decoded alignment cost, kept when posteriors replace `cost` for ranking. */
  decodedCost?: number;
  /** With `basmala`: observed phonemes before this index are the prefix, which costs `prefixCost`. */
  prefixEnd?: number;
  prefixCost?: number;
}

const MAX_BASMALA_PER = 0.25;
/** The largest prior bonus, in `UNIT`s; below `minSeparation` so a prior alone never separates. */
const MAX_PRIOR_BONUS = UNIT / 2;
/** Nats of CTC likelihood, relative to the decoded sequence, that cost as much as one plain edit. */
const NATS_PER_EDIT = 3;
/** Posteriors are rejected when fewer decoded phonemes than this appear in the top-k over their frames. */
const MIN_LISTED = 0.5;
/**
 * While acquiring, at most this many earlier phonemes are kept when a new chunk arrives. The chunk itself is always
 * searched whole, so a single-push clip is unaffected; streamed acquisitions longer than this depend on chunking.
 */
const MAX_HISTORY = 96;
/** Local recent cost, in plain edits per phoneme, above which global search runs alongside local tracking. */
const WEAK_COST = 0.3;
/** Phonemes before the weakness was noticed that the global search also sees: recent cost lags the deviation. */
const LOOKBACK = 6;
/** Global search while weak or after `lost` looks at only the latest this many phonemes. */
const WINDOW = 48;
/** While weak: hold the anchor once the window (lookback included) holds this many phonemes, nothing fits globally and local cost exceeds `maxCost`. `lost` follows only when that search locks off the anchor. */
const LOST_AFTER = 24;
/** A global lock within this many ayahs of the local bookmark is the local search's to follow, not a jump. */
const NEAR_AYAHS = 2;
/**
 * refinements.declaredLeadIn, refinements.declaredMove: the ayahs before the declared ayah a reciter may lead in from,
 * reading into the passage: as many as `NEAR_AYAHS`, the ayahs past the passage a rival near it may open.
 */
const LEAD_AYAHS = NEAR_AYAHS;
/** Phonemes by which recent cost lags a deviation: half the local search's recent-cost window. */
const WEAK_LAG = 8;
/** A new local run shows once its last this many phonemes cost at most `CONFIRM_COST` plain edits each on average. */
const CONFIRM = 3;
const CONFIRM_COST = 0.34;
/**
 * refinements.declaredStart: the declared opening locks once this many phonemes were heard (leading phrases set aside),
 * or its whole first word when that is shorter, and never on fewer than `CONFIRM`, the evidence a new local run shows
 * on. Nine ayahs in ten open on a first word of five phonemes or more (median 8), so the first word shows while it is
 * still being said; the fifth phoneme is heard past the first syllable, so the lock does not rest on the last sound
 * heard (where a sound cut off or held blurs it).
 */
const DECLARED_MIN = 5;
/** refinements.declaredStart: the reference a fit of an opening reads, from the opening on (as `fitAt`). */
const FIT_REACH = 2 * WINDOW;
/**
 * refinements.declaredStart: a rival opening is in play while its fit of the speech is less than this much worse than
 * the declared opening's: half an edit, so an opening a held length away is in play, and one an edit away is told
 * apart.
 */
const DISTINCT = UNIT / 2;
/** refinements.declaredMove: `rivalAhead`'s answer while near rivals fit the speech alike: the contest stays open. */
const HELD = -2;
/**
 * refinements.declaredStart: every ayah's opening (the first phoneme of its first spoken word), sorted by the reference
 * text a fit of it reads, as `fitBounds` walks them: the rivals of a declared opening. Built once per reference.
 */
const openingsByText = new WeakMap<ReferenceIndex, Int32Array>();
function ayahOpenings(index: ReferenceIndex): Int32Array {
  let sorted = openingsByText.get(index);
  if (sorted) return sorted;
  const openings: { offset: number; text: string }[] = [];
  let last: WordRef | null = null;
  for (const w of index.words) {
    if (w.end <= w.start || (last && last.surah === w.surah && last.ayah === w.ayah)) continue;
    last = w;
    openings.push({ offset: w.start, text: index.phonemes.slice(w.start, w.start + FIT_REACH) });
  }
  openings.sort((a, b) => (a.text < b.text ? -1 : a.text > b.text ? 1 : a.offset - b.offset));
  openingsByText.set(index, (sorted = Int32Array.from(openings, (o) => o.offset)));
  return sorted;
}
/**
 * refinements.confirmAdvance: phonemes of a later word heard as written before the run shows it (all of a shorter
 * word), after a risky step. Two, as the checker counts a visit as reciting a word (`counts` in checker.ts: two of its
 * phonemes consumed, all of a shorter word's): a word shows on the evidence that would count it as said.
 */
const ADVANCE_MATCHES = 2;
/**
 * refinements.consonantConfirm: consonants of a later word of the run heard as written before it shows (all of a word
 * with fewer). Two, as `ADVANCE_MATCHES` counts two sounds for a visit to count as reciting a word, but consonants:
 * every consonant carries a vowel, so the first consonant and its vowel are two sounds that many words share.
 */
const CONFIRM_CONSONANTS = ADVANCE_MATCHES;
/** The short vowels (fatha, damma, kasra): a word's last one is dropped at a stop. */
const SHORT_VOWELS = new Set(["\u064E", "\u064F", "\u0650"]);
/**
 * refinements.showSaid: the quiet after a word's last phoneme that settles it as a pause, ms. Longer than the
 * decoder's gap (`minGapFrames`, 0.3 s): the decoder hears a held sound (a ghunna, a madd) at its start and then often
 * nothing for as long, so a shorter quiet can be the hold itself. On recorded recitations, a quiet of 320 ms showed
 * 15 words before their end (where an aligner placed it) that no pause rule shows early, 400 ms 11 and 480 ms 9.
 */
const PAUSE_QUIET_MS = 480;
/**
 * refinements.withdrawBegun: steps of the shown run's path kept, to read the word a jump left as the path was when the
 * jump began. More than a new run waits before it shows (`CONFIRM`, `jumpPhonemes`, a weak stretch's `WINDOW`).
 */
const TRAIL = 256;
/**
 * refinements.rivalHold: phonemes of the word heard as written after which the run's order wins over a rival that
 * still fits as well (all of a shorter word's). Four: two syllables, each a consonant and its vowel, the evidence
 * `consonantConfirm` asks of a word (`CONFIRM_CONSONANTS`) counted in sounds heard as written. A rival that still fits
 * as well after them is the same text (a refrain, a phrase said twice), which no wait tells apart, so the order the
 * reciter is reading in decides.
 */
const RIVAL_MATCHES = 2 * CONFIRM_CONSONANTS;
/**
 * refinements.confirmAdvance: below this decoder confidence, the first phoneme heard after a pause may be a breath or
 * a stray sound rather than speech, and a later word entered on it shows only after `ADVANCE_MATCHES`. The tracker's
 * own cut, apart from the checker's `minConfidence` (changing what the checker judges leaves the flow alone). Measured
 * on Tilawa's own fixtures (lab/benchmark/test_corpus, _v2 and _v3: 33 multi-ayah clips, 122 ayah openings heard after
 * a pause, each aligned to its text): a correct opening's first phoneme falls below 0.6 in 3 (2.5%; below 0.5 in
 * none, below 0.7 in 7), and those show a phoneme later. The sounds after a pause that the alignment left out of the
 * text were few there (9) and mostly heard surely (median 0.85, one below 0.6): the cut catches only faint ones, and
 * a sound not heard as the word's first phoneme is a risky step whatever its confidence.
 */
const BREATH_CONFIDENCE = 0.6;
/**
 * After a skipped takbir, al-Fatiha's opening gets this bonus: in prayer each rak'ah restarts with it, and its al-hamd
 * opening is shared with five other passages, so the half-edit `prior` bonus couldn't separate it.
 */
const FATIHA_BONUS = 2 * UNIT;
/** A phrase is heard when it aligns to the latest phonemes at most this weighted cost per phrase phoneme. */
const PHRASE_MAX_COST = 0.3;
/** While locked, a heard phrase is skipped only if local search fit it this badly, per phoneme: else it's Quran text. */
const PHRASE_LOCAL_COST = 0.25;
/** Phonemes kept for phrase detection; longer than any phrase as recited. */
const HEARD = 64;
/**
 * A phrase in progress holds off locking once this many of its phonemes, and this share of it, are heard. Shorter
 * prefixes ("Allahu" of the takbir) are everyday Quran text.
 */
const PHRASE_PREFIX = 12;
/** A restart in progress holds off locking once this many phonemes restate the window's start. */
const RESTART_PREFIX = 6;
/** Repeated blocks this long, at most this weighted cost per phoneme apart, collapse to one copy in a stream. */
const MIN_REPEAT = 8;
const MAX_REPEAT = 40;
const REPEAT_MAX_COST = 0.15;
/**
 * refinements.showSaid: phonemes of the next word heard (its whole length when shorter) before a word inside an ayah,
 * left for it, counts as said. The decoder places a phoneme at its first sound, and a word's last sounds run into the
 * next word's first (a doubled onset, a nasal merged, a joined hamzat al-wasl), so a word is surely finished once the
 * next word's first syllable has been heard.
 */
const NEXT_BEGUN = 3;

const ayahCompare = (a: AyahRef, b: AyahRef) => a.surah - b.surah || a.ayah - b.ayah;
/** Mushaf order of two words. */
const refOrder = (a: WordRef, b: WordRef) => ayahCompare(a, b) || a.word - b.word;

/** Local recent cost above this, in plain edits per phoneme, is the misfit that starts a hold. */
export const MISFIT_MAX_COST = 0.35;
/** Commit separation, in plain edits: `max(COMMIT_MIN_SEPARATION, COMMIT_SEPARATION_SCALE / n)`. */
export const COMMIT_MIN_SEPARATION = 1;
export const COMMIT_SEPARATION_SCALE = 30;

/** The lock's commit margin for `n` observed phonemes, in plain edits. */
export function commitMargin(n: number, minSeparation = COMMIT_MIN_SEPARATION, separationScale = COMMIT_SEPARATION_SCALE): number {
  return Math.max(minSeparation, separationScale / Math.max(1, n));
}

export interface IssueDropFit {
  n: number;
  /** Weighted cost per observed phoneme, in plain edits. */
  unedited: number;
  edited: number;
  /** Plain edits by which the edited anchor beats the written one. */
  gain: number;
  /** {@link commitMargin} for this span. */
  margin: number;
  explains: boolean;
}

/**
 * Cost of aligning every heard phoneme to a substring of `reference`, starting at its first phoneme, in `UNIT`s.
 * The same weighted word costs as local tracking (`subCost`, `insertionCost`, `deletionCost`).
 */
export function anchorFitCost(reference: string, heard: string): number | null {
  if (!reference || !heard) return null;
  const fit = alignSubstring(reference, 0, reference.length, heard, (offset) => offset === 0, { strictStart: true });
  return fit ? fit.cost : null;
}

/**
 * Whether applying an issue's edit to the anchor text explains a misfit. `written` is the anchor from the issue
 * through the drop; `edited` is that text with the issue's own edit applied; `heard` is the phonemes between them.
 * The edited fit must beat the unedited fit by the commit margin and fall below the misfit cost. No new threshold.
 */
export function issueDropFit(
  written: string,
  edited: string,
  heard: string,
  maxCost = MISFIT_MAX_COST,
  minSeparation = COMMIT_MIN_SEPARATION,
  separationScale = COMMIT_SEPARATION_SCALE,
): IssueDropFit | null {
  const uneditedCost = anchorFitCost(written, heard);
  const editedCost = anchorFitCost(edited, heard);
  if (uneditedCost === null || editedCost === null) return null;
  const n = heard.length;
  const gain = (uneditedCost - editedCost) / UNIT;
  const margin = commitMargin(n, minSeparation, separationScale);
  const unedited = uneditedCost / UNIT / n;
  const editedPer = editedCost / UNIT / n;
  return { n, unedited, edited: editedPer, gain, margin, explains: gain >= margin && editedPer < maxCost };
}

/**
 * An added word's heard phones deleted, the phones after them kept. `heard` is the issue through the drop;
 * `kept` is `heard` without the inserted span. The margin is the commit margin of the full span, so a short
 * tail is not asked to clear the margin a longer one would. Same word costs as {@link issueDropFit}.
 */
export function insertionDropFit(
  reference: string,
  heard: string,
  kept: string,
  maxCost = MISFIT_MAX_COST,
  minSeparation = COMMIT_MIN_SEPARATION,
  separationScale = COMMIT_SEPARATION_SCALE,
): IssueDropFit | null {
  if (!reference || !heard || !kept) return null;
  const uneditedCost = anchorFitCost(reference, heard);
  const editedCost = anchorFitCost(reference, kept);
  if (uneditedCost === null || editedCost === null) return null;
  const n = heard.length;
  const gain = (uneditedCost - editedCost) / UNIT;
  const margin = commitMargin(n, minSeparation, separationScale);
  const unedited = uneditedCost / UNIT / n;
  const editedPer = editedCost / UNIT / n;
  return { n, unedited, edited: editedPer, gain, margin, explains: gain >= margin && editedPer < maxCost };
}

export interface DestinationWord {
  /** Phonemes of the destination word this span aligned to. */
  destination: string;
  /** Heard phonemes the destination alignment gave that word. */
  heard: string;
}

/**
 * Heard words that fit a relocation's destination better than the anchor continuation with exactly one
 * inserted heard word, by the commit margin. A word the insertion hypothesis also fits, such as the anchor's
 * own next word, counts zero. `total` is the winning hypothesis's cost, in `UNIT`s, so callers can prefer the
 * anchor start the hypothesis actually explains.
 */
export function relocationSupport(
  anchorWords: readonly string[],
  heard: readonly DestinationWord[],
  minSeparation = COMMIT_MIN_SEPARATION,
  separationScale = COMMIT_SEPARATION_SCALE,
): { count: number; total: number } | null {
  const words = heard.filter((h) => h.heard.length > 0 && h.destination.length > 0);
  if (!words.length || !anchorWords.length) return null;
  let best: { count: number; total: number } | null = null;
  for (let ins = 0; ins < words.length; ins++) {
    let ai = 0;
    let total = 0;
    let count = 0;
    let placed = true;
    for (let i = 0; i < words.length; i++) {
      const h = words[i];
      const dest = anchorFitCost(h.destination, h.heard);
      let hyp: number | null = null;
      if (i === ins) {
        let c = 0;
        for (const ch of h.heard) c += insertionCost("", 0, ch);
        hyp = c;
      } else if (ai < anchorWords.length) {
        hyp = anchorFitCost(anchorWords[ai], h.heard);
        ai++;
      }
      if (dest === null || hyp === null) {
        placed = false;
        break;
      }
      // The commit margin widens on a short span. A word the destination reads exactly is already that sure, so it
      // counts once it leads by the margin's floor; a partial or imperfect reading has to clear the wide margin, which
      // is what keeps a shared word heard in part from voting.
      const gain = (hyp - dest) / UNIT;
      const margin = commitMargin(h.heard.length, minSeparation, separationScale);
      if (gain >= margin || (dest === 0 && gain >= minSeparation)) count++;
      total += hyp;
    }
    if (placed && (!best || total < best.total)) best = { count, total };
  }
  return best;
}

/** What the tracker is fed from one chunk of listener evidence, in the product and in every bench alike: phonemes only. */
export function trackerInput(evidence: PhonemeEvidence): PhonemeEvidence {
  return { ...evidence, posteriors: undefined };
}

export function createTracker(index: ReferenceIndex, options: TrackerOptions = {}): Tracker {
  const maxGramOccurrences = options.maxGramOccurrences ?? 5000;
  const maxCandidates = options.maxCandidates ?? 64;
  const maxPer = options.maxPer ?? 0.35;
  const maxCost = options.maxCost ?? MISFIT_MAX_COST;
  const minPhonemes = options.minPhonemes ?? 10;
  const minWords = options.minWords ?? 2;
  const minSeparation = options.minSeparation ?? COMMIT_MIN_SEPARATION;
  const separationScale = options.separationScale ?? COMMIT_SEPARATION_SCALE;
  const minLockConfidence = options.minLockConfidence ?? 0.08;
  const prior = options.prior;
  const priorRange = options.priorRange;
  const jumpPhonemes = options.jumpPhonemes ?? 8;
  const memorization = options.mode === "memorization";
  const refine = options.refinements ?? {};
  const confirmAdvance = refine.confirmAdvance !== false;
  const onsetRun = refine.onsetRun !== false;
  const declaredStart = memorization && refine.declaredStart !== false;
  const declaredLeadIn = declaredStart && refine.declaredLeadIn !== false;
  const declaredMove = declaredStart && refine.declaredMove !== false;
  const noLostBeforeLock = memorization && refine.noLostBeforeLock !== false;
  const consonantConfirm = refine.consonantConfirm !== false;
  const rivalHold = refine.rivalHold !== false;
  const passageWindow = memorization && refine.passageWindow !== false;
  const restartNoEntry = refine.restartNoEntry !== false;
  const withdrawBegun = refine.withdrawBegun !== false;
  const leftOnNewAudio = withdrawBegun && refine.leftOnNewAudio !== false;
  const begunByConsonants = withdrawBegun && refine.begunByConsonants !== false;
  const showSaid = refine.showSaid !== false;
  /** refinements.passageWindow: the declared passage's first and last words, and its phonemes; null without one. */
  const bounds = (() => {
    const p = options.passage;
    if (!passageWindow || !p) return null;
    const at = (r: WordRef) => index.words.findIndex((w) => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word);
    const [from, to] = [at(p.from), at(p.to)];
    return from >= 0 && to >= from ? { from, to, phonemes: { from: index.words[from].start, to: index.words[to].end } } : null;
  })();
  /**
   * refinements.passageWindow: the local search at `offset` is inside the declared passage
   * (`LocalSearchOptions.passage`).
   */
  const boundedAt = (offset: number) => !!bounds && offset >= bounds.phonemes.from && offset <= bounds.phonemes.to;
  const wordStarts = new Set(index.words.map((w) => w.start));
  const isWordStart = (offset: number) => wordStarts.has(offset);
  const fatiha = index.words.filter((w) => w.surah === 1 && w.ayah === 1);
  const basmala = index.phonemes.slice(fatiha[0].start, fatiha.at(-1)!.end);
  const fatihaAyah2 = index.words.find((w) => w.surah === 1 && w.ayah === 2)!.start;
  const fatihaRabbi = index.words.findIndex((w) => w.surah === 1 && w.ayah === 2 && w.word === 3);

  let observed = "";
  let observedConf: number[] = [];
  const observedFrames: { start: number; end: number }[] = [];
  const tokenIds = new Map([...index.tokens].map((t, i) => [t, i + 1]));
  /** Only the first acquisition reranks with posteriors; dropped at lock, so later global searches are decoded-only. */
  let posteriors: PosteriorFrames | null = null;
  let decodedOnly = false;
  let withPosteriors = false;
  let frameMs: number | null = null;
  let lastFrame = 0;
  let last: PositionEvent | null = null;
  /**
   * refinements.withdrawBegun: the last locked position's run: its first word and its word (word indices), the
   * furthest word it showed (`reach`), its number (`id`, counting the runs shown), and, once known, the local search it
   * is on and where in that search it began (`LocalPosition.since`): a jump back to the run's own first word starts
   * another run there.
   */
  let shownRun: { id: number; start: number; word: number; reach: number; search: number; since: number | undefined } | null = null;
  /** refinements.withdrawBegun: the runs shown so far. */
  let runs = 0;
  /**
   * refinements.withdrawBegun: each word shown and not withdrawn since, with the run that showed it first. A run
   * withdraws only words it showed first: a word an earlier run showed was heard then, and stays shown.
   */
  const litBy = new Map<number, number>();
  /**
   * refinements.withdrawBegun: the path of the run last shown, step by step (`noteBegun`): the local search it is on
   * (`search`), the step, the path's word, and whether only that word's onset was heard.
   */
  let trail: { search: number; step: number; at: number; begun: boolean }[] = [];
  /** refinements.withdrawBegun: counts the local searches started, so a step is read against its own search. */
  let search = 0;
  /** The positions to show for the current `push()` or `end()` (`shown`). */
  let shownOut: PositionEvent[] = [];
  /** refinements.showSaid: the last position shown. */
  let shownLast: PositionEvent | null = null;
  /** refinements.showSaid: the last word said (word index, -1 for none yet) in the run whose first word is `start`. */
  let said = { start: -1, wi: -1 };
  /** refinements.showSaid: words withdrawn by positions not shown yet (a run whose first word is not said yet). */
  let carried: WordRef[] = [];
  /** refinements.showSaid: the words the page shows: a withdrawal names only these, each once. */
  const pageShown = new Set<number>();
  /**
   * refinements.showSaid: counts the positions sent (`emit`), and the count when the last one sent was first shown: its
   * one-time news (a hold's start, a relocation) is shown once.
   */
  let sent = 0;
  let toldAt = 0;
  /**
   * refinements.showSaid: the local search's last position on the run shown (its first word `start`) whose phoneme was
   * heard as written.
   */
  let heardOn: { start: number; pos: LocalPosition } | null = null;
  /** Set once locked: from then on the tracker follows the reciter locally, phoneme by phoneme. */
  let local: LocalSearch | null = null;
  let lockedOn: Match | null = null;
  /** The event that reported the lock, kept with its trace until a local event replaces it. */
  let lockEvent: PositionEvent | null = null;
  let emittedSinceLock = false;
  /** Local fit has degraded: global search over the latest phonemes runs alongside the local search. */
  let weak = false;
  /** A distant global lock seen while weak, and for how many consecutive phonemes it has held. */
  let pending: { m: Match; held: number } | null = null;
  /** After a misfit: global search over a rolling window, one phoneme at a time, until it locks again. */
  let reacquiring = false;
  /**
   * Lock kept while that search runs. A misfit used to emit `lost` before any destination had committed, and the
   * checker then dropped the anchor and the tokens heard in the gap. `lost` is emitted only when a lock commits off
   * this anchor, or the session ends still uncommitted. The held event carries `hold: "start"` once. The checker
   * keeps an open issue only when that issue's edit explains the drop, and downgrades the rest; tokens after the
   * signal still enter the window.
   */
  let holding: PositionEvent | null = null;
  /** Set at the first lock. The onset veto applies only to that lock. */
  let everLocked = false;
  /** Slot-differing span the lock is inside, until its continuation picks a member. */
  let twin: { span: SlotSpan; memberStart: number; endPhone: number; buf: string } | null = null;
  /**
   * A lock onto the other member of a parallel span (or an exact copy of the next ayah). The phones heard until
   * the in-order continuation commits are what a retro rebind is scored on.
   */
  let excursion: { home: number; dest: number; length: number; entry: number; phones: string; trimmed: boolean } | null = null;
  /**
   * Holding and continuation resolution run only on the first lock, and only until that lock's position leaves
   * every slot-differing span. A word outside those spans is unique to its location. Any later `startLocal`
   * (a jump, a reacquisition, the re-anchor after a copy change) closes it.
   */
  let arm = false;
  /** Spans already resolved or given up at the lost timeout. Not watched again. */
  const twinSettled = new Set<string>();
  const allSpans = slotSpans(index);
  /**
   * refinements.confirmAdvance: the end frame of the latest phoneme heard, and whether it may be a breath (heard after
   * a pause, below `BREATH_CONFIDENCE`); the word the local search's run is in, and whether it was entered on a risky
   * step (a phoneme not heard as written, or one that may be a breath).
   */
  let heardEnd = -1;
  let breath = false;
  /** A pause between a phoneme that ends at frame `end` and one that starts at `start`: the checker's labelled gap. */
  const isPause = (end: number, start: number) => frameMs !== null && frameMs > 0 && start - end - 1 >= minGapFrames(frameMs);
  let runWord = -1;
  let riskyEntry = false;
  /** refinements.consonantConfirm: the run's word was entered on weak evidence (`weakEntry` in `localEvent`). */
  let weakEntry = false;
  /**
   * refinements.rivalHold: the word the local search's best path is in, and the phonemes heard since it entered it (from
   * a lock, those its path placed there).
   */
  let wordHeard = { word: -1, heard: "" };
  /** Every phoneme until the first lock, untrimmed, so a first-lock veto can see back to the speech onset. */
  let raw = "";
  let rawConf: number[] = [];
  /** Phones of `raw` consumed by a leading non-Quran phrase the streaming matcher already cut. */
  let onsetCut = 0;
  /** Memorization mode: the expected bookmark held in place after the reciter jumped away from it. */
  let frozen: WordRef | null = null;
  /** The expected bookmark from before local fit last turned weak. */
  let steady: WordRef | null = null;
  /**
   * The expected bookmark after each recent phoneme. Recent cost lags a deviation by several phonemes, during which
   * local skips can already have moved the bookmark; the one from before that lag is what weakness should freeze.
   */
  const expectedHistory: (WordRef | null)[] = [];
  /** The local search's cost for each of the last `CONFIRM` phonemes. */
  const stepCosts: number[] = [];

  const phraseList = options.phrases ? [...options.phrases.phrases, { name: "basmala", text: "", p: basmala }] : [];
  /** The latest phonemes, with the local search's cost for each and its state before each, for phrase detection. */
  let heard = "";
  let heardCost: number[] = [];
  let heardState: unknown[] = [];
  /** While acquiring: observed phonemes up to here end with a phrase, to drop at the next push. */
  let phraseCut = 0;
  /** A basmala was skipped in an earlier push (`surahStart`) or this one (`basmalaHeard`): favour surah openings. */
  let surahStart = false;
  let basmalaHeard = false;
  /** A takbir was skipped: in prayer al-Fatiha comes next (`fatihaNext`), promoted from `takbirHeard` like the basmala. */
  let fatihaNext = false;
  /** Re-acquiring after a basmala ended a locked passage: where it ended, and phonemes since. */
  let afterBasmala: { from: WordRef; held: number } | null = null;
  let takbirHeard = false;
  /** A phrase was just skipped while locked: a boundary, so the next distant lock needn't wait for persistence. */
  let afterPhrase = false;
  let pushes = 0;
  const validate = (e: PhonemeEvidence) => {
    if (e.inventory !== index.inventoryId) {
      throw new EvidenceError(`Evidence inventory "${e.inventory}" does not match index inventory "${index.inventoryId}"`);
    }
    if (frameMs !== null && e.frameMs !== frameMs) {
      throw new EvidenceError(`Frame stride changed from ${frameMs} ms to ${e.frameMs} ms within a session`);
    }
    for (const ph of e.phonemes) {
      if (!index.tokens.has(ph.p)) throw new EvidenceError(`Phoneme "${ph.p}" is not in inventory "${index.inventoryId}"`);
    }
    if (e.posteriors ? decodedOnly : withPosteriors) throw new EvidenceError("Every chunk in a session must carry posteriors, or none");
    const p = e.posteriors;
    if (!p) return;
    if (p.ids.length !== p.nFrames * p.k || p.probs.length !== p.nFrames * p.k) {
      throw new EvidenceError(`Posterior rows hold ${p.ids.length} ids and ${p.probs.length} probs, expected nFrames x k = ${p.nFrames * p.k}`);
    }
    let listed = 0;
    for (const ph of e.phonemes) {
      if (ph.start < p.startFrame || ph.end >= p.startFrame + p.nFrames) {
        throw new EvidenceError(`Phoneme frames ${ph.start}..${ph.end} fall outside posterior frames ${p.startFrame}..${p.startFrame + p.nFrames - 1}; check the frame stride`);
      }
      const id = tokenIds.get(ph.p)!;
      let found = false;
      for (let f = ph.start; f <= ph.end && !found; f++) {
        for (let c = 0; c < p.k; c++) if (p.ids[(f - p.startFrame) * p.k + c] === id) found = true;
      }
      if (found) listed++;
    }
    if (e.phonemes.length && listed < MIN_LISTED * e.phonemes.length) {
      throw new EvidenceError(`Only ${listed} of ${e.phonemes.length} decoded phonemes appear in their frames' top-k; posterior ids don't follow the inventory order`);
    }
  };

  const exactScan = (query: string): Match[] => {
    const found: Match[] = [];
    for (let at = index.phonemes.indexOf(query); at !== -1 && found.length < maxCandidates; at = index.phonemes.indexOf(query, at + 1)) {
      const path: AlignmentOp[] = [...query].map((p, k) => ({ op: "match", ref: p, refAt: k, obsAt: k }));
      found.push({ offset: at, length: query.length, cost: 0, per: 0, votes: 0, path });
    }
    return found;
  };

  const fuzzyMatches = (query: string, queryConf: number[]): Match[] => {
    if (query.length < index.gramSize) return exactScan(query);
    const votes = new Map<number, number>();
    for (let q = 0; q + index.gramSize <= query.length; q++) {
      const hits = index.grams.get(query.slice(q, q + index.gramSize));
      if (!hits || hits.length > maxGramOccurrences) continue;
      for (const pos of hits) {
        const diagonal = Math.max(0, pos - q);
        votes.set(diagonal, (votes.get(diagonal) ?? 0) + 1);
      }
    }
    // Every gram was too common to vote; fall back to a direct scan.
    if (votes.size === 0) return exactScan(query);

    // Learner insertions and deletions shift the diagonal, so nearby diagonals vote for one location.
    const slack = Math.max(4, Math.ceil(query.length * 0.25));
    // Bins are centred on vote peaks with a fixed width: chaining nearby diagonals lets common grams join one bin
    // across the whole Quran on long clips.
    const diagonals = [...votes.keys()].sort((a, b) => a - b);
    const prefix = [0];
    for (const d of diagonals) prefix.push(prefix.at(-1)! + votes.get(d)!);
    const firstAtLeast = (x: number) => {
      let a = 0;
      let b = diagonals.length;
      while (a < b) {
        const mid = (a + b) >> 1;
        if (diagonals[mid] < x) a = mid + 1;
        else b = mid;
      }
      return a;
    };
    const windowed = diagonals.map((d, k) => {
      const a = firstAtLeast(d - slack);
      const b = firstAtLeast(d + slack + 1);
      return { k, lo: diagonals[a], hi: diagonals[b - 1], votes: prefix[b] - prefix[a] };
    });
    windowed.sort((x, y) => y.votes - x.votes || diagonals[x.k] - diagonals[y.k]);
    const bins: { lo: number; hi: number; votes: number; centre: number }[] = [];
    for (const c of windowed) {
      if (bins.length >= maxCandidates) break;
      const centre = diagonals[c.k];
      if (bins.some((b) => Math.abs(b.centre - centre) <= slack)) continue;
      bins.push({ lo: c.lo, hi: c.hi, votes: c.votes, centre });
    }

    const byStart = new Map<number, Match>();
    for (const bin of bins) {
      const lo = Math.max(0, bin.lo - slack);
      const hi = Math.min(index.phonemes.length, bin.hi + query.length + slack);
      const a = alignSubstring(index.phonemes, lo, hi, query, isWordStart, { conf: queryConf });
      if (!a || a.per > maxPer) continue;
      const prev = byStart.get(a.start);
      if (prev && prev.cost <= a.cost) continue;
      byStart.set(a.start, { offset: a.start, length: a.end - a.start, cost: a.cost, per: a.per, votes: bin.votes, path: a.path });
    }
    return [...byStart.values()].sort((a, b) => a.cost - b.cost || a.offset - b.offset);
  };

  /** Matches for the evidence, plus matches for what follows a leading basmala, which the index only holds as 1:1. */
  const locate = (): Match[] => {
    const plain = fuzzyMatches(observed, observedConf);
    const prefix = alignSubstring(observed, 0, observed.length, basmala, (o) => o === 0, { strictStart: true });
    if (!prefix || prefix.per > MAX_BASMALA_PER || observed.length - prefix.end < index.gramSize) return plain;
    const k = prefix.end;
    const after = fuzzyMatches(observed.slice(k), observedConf.slice(k))
      // Basmala then 1:2 is just 1:1–1:2, which the plain search already found.
      .filter((m) => m.offset !== fatihaAyah2)
      .map((m) => ({
        ...m,
        cost: m.cost + prefix.cost,
        basmala: true,
        prefixEnd: k,
        prefixCost: prefix.cost,
        path: m.path.map((o) => (o.op === "del" ? o : { ...o, obsAt: o.obsAt + k })),
      }));
    const byStart = new Map<number, Match>();
    for (const m of [...plain, ...after]) if (!byStart.has(m.offset) || byStart.get(m.offset)!.cost > m.cost) byStart.set(m.offset, m);
    return [...byStart.values()].sort((a, b) => a.cost - b.cost || a.offset - b.offset);
  };

  const toIds = (tokens: string[]) => collapseHeld(tokens).map((t) => tokenIds.get(t)!);
  /**
   * Reranking cost: the CTC likelihood, over every frame of the evidence, of the candidate's alignment read as labels
   * (reference tokens where it matches, substitutes or deletes; observed tokens where it inserts), relative to the
   * decoded sequence. Insertions, a mid-word start and a basmala prefix add their decoded costs on top.
   */
  const posteriorCost = (frames: PosteriorFrames, m: Match, self: number): number => {
    const from = observedFrames[0].start;
    const to = observedFrames.at(-1)!.end + 1;
    const labels: string[] = [];
    const prefixEnd = m.basmala ? m.prefixEnd! : 0;
    for (let j = 0; j < prefixEnd; j++) labels.push(observed[j]);
    let edits = (m.basmala ? m.prefixCost! : 0) + (isWordStart(m.offset) ? 0 : MID_WORD_START);
    const token = (o: AlignmentOp | undefined) => (!o ? undefined : o.op === "ins" ? o.obs : o.ref);
    m.path.forEach((o, i) => {
      if (o.op === "ins") {
        // An inserted held sound beside the same sound only lengthens it; the collapse drops it, so don't charge it.
        const lengthens = isHeld(o.obs) && (labels.at(-1) === o.obs || token(m.path[i + 1]) === o.obs);
        if (!lengthens) edits += UNIT * observedConf[o.obsAt];
        labels.push(o.obs);
      } else labels.push(o.ref);
    });
    const nll = frames.nll(toIds(labels), from, to);
    if (!Number.isFinite(nll) || !Number.isFinite(self)) return m.cost;
    return Math.max(0, edits + ((nll - self) * UNIT) / NATS_PER_EDIT);
  };

  const ref = (i: number): WordRef => {
    const w = index.words[i];
    return { surah: w.surah, ayah: w.ayah, word: w.word };
  };

  const normCost = (m: Match) => m.cost / UNIT / observed.length;
  /** Fit takes the better of the two costs: posteriors can rescue a mangled decode, but CTC charges held-sound length in full. */
  const fitCost = (m: Match) => Math.min(m.cost, m.decodedCost ?? m.cost) / UNIT / observed.length;

  const firstWord = (m: Match) => index.wordAt(m.offset);
  const lastWord = (m: Match) => index.wordAt(m.offset + Math.max(0, m.length - 1));

  const toCandidate = (m: Match, confidence: number): PositionCandidate => {
    const end = ref(lockWord(m, observedAt));
    return { word: end, span: { start: ref(firstWord(m)), end }, confidence, cost: normCost(m), per: m.per, ...(m.basmala ? { prefix: "basmala" as const } : {}) };
  };

  const priorBonus = (m: Match): number => {
    const first = index.words[firstWord(m)];
    const opening = first.word === 1 && (first.ayah === 1 || (first.surah === 1 && first.ayah === 2));
    const nextSurah = afterBasmala && opening && first.ayah === 1 && first.surah === afterBasmala.from.surah + 1 ? MAX_PRIOR_BONUS : 0;
    const openingBonus = (surahStart && opening ? MAX_PRIOR_BONUS : 0) + nextSurah;
    // Only once the span reaches "rabbi": "al-hamdu lillahi" alone is also how 6:1, 18:1, 34:1 and 35:1 open.
    if (fatihaNext && first.surah === 1 && first.word === 1 && first.ayah <= 2 && lastWord(m) >= fatihaRabbi) return FATIHA_BONUS;
    let bonus = openingBonus;
    if (priorRange) {
      const inside = priorRange.ranges.some(({ from, to }) => {
        for (let i = firstWord(m); i <= lastWord(m); i++) {
          const w = index.words[i];
          if (ayahCompare(w, from) >= 0 && ayahCompare(w, to) <= 0) return true;
        }
        return false;
      });
      if (inside) bonus = Math.max(bonus, MAX_PRIOR_BONUS * Math.min(1, Math.max(0, priorRange.weight)));
    }
    if (!prior) return bonus;
    let best = 0;
    for (let i = firstWord(m); i <= lastWord(m); i++) {
      const w = index.words[i];
      if (w.surah !== prior.surah) continue;
      const d = Math.abs(w.ayah - prior.ayah);
      best = Math.max(best, d === 0 ? 1 : d === 1 ? 0.5 : 0);
    }
    return Math.max(bonus, best * MAX_PRIOR_BONUS * Math.min(1, Math.max(0, prior.weight)));
  };

  /** Overlapping or touching ranges describe one location, e.g. shifted spans of the same ayah. */
  const sameLocation = (a: Match, b: Match) => a.offset <= b.offset + b.length && b.offset <= a.offset + a.length;

  /** Phoneme offset where the ayah containing `offset` starts. */
  const ayahOrigin = (offset: number): number => {
    let i = index.wordAt(Math.max(0, Math.min(offset, index.phonemes.length - 1)));
    const w = index.words[i];
    while (i > 0) {
      const prev = index.words[i - 1];
      if (prev.surah !== w.surah || prev.ayah !== w.ayah) break;
      i--;
    }
    return index.words[i].start;
  };
  /** Ayah start immediately before an ayah start, or -1 at the beginning of the mushaf. */
  const previousAyahOrigin = (origin: number): number => {
    const i = index.wordAt(origin);
    if (i <= 0) return -1;
    return ayahOrigin(index.words[i - 1].start);
  };

  /**
   * Heard phonemes from the first Quran onset. Phrases the streaming matcher already cut are dropped via
   * `onsetCut`; anything still leading is stripped with the same phrase alignment.
   */
  const onsetSpeech = (): { text: string; conf: number[] } => {
    const cut = leadingPhrases(raw.slice(onsetCut), phraseList);
    return { text: raw.slice(onsetCut + cut), conf: rawConf.slice(onsetCut + cut) };
  };
  /**
   * Where each phrase ends when it leads the speech `rest[0, hi)` (0 when it does not): `leadingPhrases` asks again with
   * the same speech on every phoneme heard before the first lock, so the answers are kept (a bounded memo, cleared
   * at each lock: `startLocal`).
   */
  const phraseLeads = new Map<string, number>();
  const leadEnd = (phrase: string, rest: string, hi: number): number => {
    const key = `${phrase}\u0000${rest.slice(0, hi)}`;
    const known = phraseLeads.get(key);
    if (known !== undefined) return known;
    // Same alignment as `hear`: the phrase is the query, any reference offset may start it, and the cost cap is
    // PHRASE_MAX_COST. Only a match that begins at the onset is a leading phrase.
    const n = phrase.length;
    const a = alignSubstring(rest, 0, hi, phrase, () => true);
    const end = !a || a.start !== 0 || a.cost / UNIT / n > PHRASE_MAX_COST ||
      a.path.filter((o) => o.op === "match" && o.obsAt >= n - 2).length < 2 ? 0 : a.end;
    if (phraseLeads.size >= 4096) phraseLeads.clear();
    phraseLeads.set(key, end);
    return end;
  };
  /** How many leading phonemes of `text` are finished phrases of `list`, said one after another. */
  const leadingPhrases = (text: string, list: typeof phraseList): number => {
    let total = 0;
    for (let guard = 0; guard < 6 && total < text.length; guard++) {
      const rest = text.slice(total);
      let cut = 0;
      for (const phrase of list) {
        const n = phrase.p.length;
        if (n < 2 || rest.length < 2) continue;
        const end = leadEnd(phrase.p, rest, Math.min(rest.length, Math.ceil(n * 1.6) + 4));
        if (end > cut) cut = end;
      }
      if (!cut) break;
      total += cut;
    }
    return total;
  };

  /**
   * Alignment of every onset phone against the contiguous ayahs ending at this candidate. The run walks back an ayah
   * at a time until it covers that speech, so a look-alike tail has to match the heard opening with its own
   * preceding ayah.
   */
  const contiguousRunCost = (m: Match, speech: string, conf: number[]): number => {
    const refEnd = Math.min(index.phonemes.length, m.offset + Math.max(m.length, 0));
    if (refEnd <= 0 || !speech) return Infinity;
    let lo = ayahOrigin(Math.min(m.offset, refEnd - 1));
    for (let guard = 0; refEnd - lo < speech.length && guard < 30; guard++) {
      const prev = previousAyahOrigin(lo);
      if (prev < 0 || prev >= lo) break;
      lo = prev;
    }
    const a = alignSubstring(index.phonemes, lo, refEnd, speech, isWordStart, {
      strictStart: true,
      ...(conf.length === speech.length ? { conf } : {}),
    });
    return a ? a.cost : Infinity;
  };

  /** Global search over the observed phonemes: the event it supports, the lock if any, and its trace. */
  const assess = () => {
    const frames = posteriors;
    const self = frames ? frames.nll(toIds([...observed]), observedFrames[0].start, observedFrames.at(-1)!.end + 1) : 0;
    const matches = frames ? locate().map((m) => ({ ...m, cost: posteriorCost(frames, m, self), decodedCost: m.cost })) : locate();
    const scored = matches.map((m) => ({ m, score: m.cost - priorBonus(m) })).sort((a, b) => a.score - b.score || a.m.offset - b.m.offset);
    const distinct: typeof scored = [];
    for (const c of scored) if (!distinct.some((k) => sameLocation(k.m, c.m))) distinct.push(c);
    const fitting = distinct.filter((c) => c.m.per <= maxPer && fitCost(c.m) <= maxCost);

    const n = observed.length;
    const required = UNIT * Math.max(minSeparation, separationScale / n);
    const decide = (pool: typeof fitting) => {
      // How far each candidate leads the best other fitting candidate; negative when it trails.
      const lead = (c: (typeof pool)[number]) => Math.min(...pool.filter((o) => o !== c).map((o) => o.score - c.score), Infinity);
      const top = pool[0];
      const separated = top !== undefined && lead(top) >= required;
      const terms = (c: (typeof pool)[number]) => ({
        fit: Math.max(0, 1 - c.m.per / maxPer) * Math.max(0, 1 - fitCost(c.m) / maxCost),
        separation: Math.min(1, Math.max(0, 0.5 + lead(c) / (2 * required))),
      });
      const confidence = (c: (typeof pool)[number]) => {
        const t = terms(c);
        return t.fit * t.separation;
      };
      const enough =
        n >= minPhonemes && top !== undefined && lastWord(top.m) - firstWord(top.m) + 1 >= minWords && confidence(top) >= minLockConfidence;
      const state: TrackerState = !top ? "lost" : !separated ? "ambiguous" : !enough ? "acquiring" : "locked";
      // Ambiguous lists every candidate within the margin of the best; otherwise only the best.
      const shown = !top ? [] : separated ? [top] : pool.filter((c) => c.score - top.score < required);
      const candidates = shown.map((c) => toCandidate(c.m, confidence(c)));
      const event: PositionEvent = {
        type: "position",
        v: EVENT_SCHEMA_VERSION,
        timeMs: lastFrame * (frameMs ?? 0),
        referenceVersion: index.version,
        expected: state === "locked" ? candidates[0].word : null,
        observed: candidates,
        state,
      };
      const trace = (from: TrackerState | null): TrackerTrace => ({
        shortlist: scored.map(({ m }) => ({
          span: toCandidate(m, 0).span,
          offset: m.offset,
          length: m.length,
          votes: m.votes,
          cost: normCost(m),
          per: m.per,
        })),
        transition: { from, to: state },
        path: scored[0]?.m.path ?? null,
        decision: top ? { ...terms(top), leadEdits: lead(top) / UNIT, requiredEdits: required / UNIT } : null,
      });
      return { event, lock: state === "locked" ? top.m : null, trace, top: top && n >= minPhonemes && confidence(top) > 0 ? top.m : null, state };
    };

    let decision = decide(fitting);
    // First lock only. A suffix window can prefer a look-alike whose own ayah matches the tail. Score each
    // candidate on the contiguous ayahs back to the speech onset, and drop the commit when that winner is beaten
    // by the same margin that would commit a lock. A winner that is not beaten is left exactly as ranked.
    // An opening that already starts on an ayah's first word is the clean case: the heard onset is that ayah, and
    // the candidate stays. The veto is for a tail match, whose own preceding ayah has to cover the onset.
    const ayahStart = decision.state === "locked" && index.words[firstWord(fitting[0].m)].word === 1;
    if (decision.state === "locked" && !everLocked && fitting.length > 1 && !ayahStart) {
      const speech = onsetSpeech();
      if (speech.text.length >= minPhonemes) {
        const costs = fitting.map((c) => contiguousRunCost(c.m, speech.text, speech.conf));
        const best = Math.min(...costs);
        if (costs[0] - best >= required) {
          const kept = fitting.filter((_, i) => costs[i] - best < required);
          if (kept.length) decision = decide(kept);
        }
      }
    }
    return { event: decision.event, lock: decision.lock, trace: decision.trace, top: decision.top };
  };

  const marked = (event: PositionEvent): PositionEvent => (arm ? { ...event, spanHold: true } : event);
  /** Word index of `r` (the words are in the mushaf's order). */
  const wordOf = (r: WordRef): number => {
    let [lo, hi] = [0, index.words.length - 1];
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (refOrder(index.words[mid], r) < 0) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  /**
   * refinements.withdrawBegun: a locked `event` that starts a run names, in `withdrawn`, the words the run shown before
   * it left only begun. A lock after a position that held no place (a lock let go, a lost place) starts a run, even
   * on the words the run before it showed: what that run's path heard is not this one's. The old run's path is taken
   * where the reciter left it: just before the new run's first phoneme when the local search's jump gives it
   * (`since`), else its last step. The word it was in then is withdrawn when only begun (`begun`); a later word the old
   * run showed, which its path reached only on sounds the new run reads as its own, when only begun there too, or
   * whatever was heard of it (`leftOnNewAudio`). A word the old run showed with no step of its path in it is kept:
   * nothing says it was only begun. Words before, which the path read on past, are never withdrawn, nor words the new
   * run covers, nor words an earlier run showed first (`litBy`). A word withdrawn lies before the new run (a jump
   * forward), or after its word when it is an ayah's first word (a restart before it). Words `event` already names
   * stay named.
   */
  const withdrawing = (event: PositionEvent, since?: number): PositionEvent => {
    const c = event.state === "locked" ? event.observed[0] : undefined;
    if (!c) return event;
    const [start, word] = [wordOf(c.span.start), wordOf(c.word)];
    const old = shownRun;
    const jumped = !old || last?.state !== "locked" || start !== old.start ||
      (since !== undefined && old.since !== undefined && old.search === search && since !== old.since);
    shownRun = jumped ? { id: ++runs, start, word, reach: word, search, since } : { ...old!, word, reach: Math.max(old!.reach, word) };
    const run = shownRun;
    // The page shows the run's words from its first word to its word.
    for (let w = jumped ? start : old!.reach + 1; w <= word; w++) if (!litBy.has(w)) litBy.set(w, run.id);
    if (!jumped) return event;
    const path = trail;
    trail = [];
    if (!old || !path.length) return event;
    const latest = (keep: (e: (typeof path)[number]) => boolean) => {
      for (let k = path.length - 1; k >= 0; k--) if (keep(path[k])) return path[k];
      return undefined;
    };
    const left = since !== undefined && path.at(-1)!.search === search ? latest((e) => e.search === search && e.step < since) : path.at(-1);
    if (!left) return event;
    const named = event.withdrawn ?? [];
    const withdrawn: WordRef[] = [];
    for (let w = left.at; w <= old.word; w++) {
      if (index.words[w].end === index.words[w].start || (w >= run.start && w <= run.word)) continue;
      if (!(run.start > w || (run.word < w && opensAyah(w))) || litBy.get(w) !== old.id) continue;
      const heard = w === left.at ? left : latest((e) => e.at === w);
      if (!heard || (!heard.begun && !(leftOnNewAudio && w > left.at))) continue;
      litBy.delete(w);
      if (!named.some((r) => wordOf(r) === w)) withdrawn.push(ref(w));
    }
    return withdrawn.length ? { ...event, withdrawn: [...named, ...withdrawn] } : event;
  };
  /**
   * refinements.withdrawBegun: after a step of the local search on the run last shown, its path's word and whether
   * that word was only begun: fewer than all its consonants heard as written (`begunByConsonants`; a doubled one counts
   * once); or, with that off, only its onset: no more than `ADVANCE_MATCHES` of its phonemes (a consonant and its vowel)
   * and fewer than `CONFIRM_CONSONANTS` of its consonants (all of a word with fewer) heard as written. A step on another
   * run (a jump not yet shown) is not kept.
   */
  const noteBegun = () => {
    if (!withdrawBegun || !local || !shownRun) return;
    const pos = local.position();
    if (index.wordAt(pos.origin) !== shownRun.start) return;
    // A run a global lock showed is the run its search starts on.
    if (shownRun.since === undefined) shownRun = { ...shownRun, search, since: pos.since };
    else if (shownRun.search !== search || shownRun.since !== pos.since) return;
    const at = index.wordAt(Math.max(0, pos.offset - 1));
    const step = pos.steps - 1;
    // A search rewound (a phrase set aside) takes its steps again.
    while (trail.length && trail.at(-1)!.search === search && trail.at(-1)!.step >= step) trail.pop();
    const begun = begunByConsonants ? !consonantsHeard(at, pos, true)
      : pos.wordMatches <= ADVANCE_MATCHES && pos.consonantMatches < Math.min(CONFIRM_CONSONANTS, consonantsOf(at));
    trail.push({ search, step, at, begun });
    if (trail.length > TRAIL) trail.shift();
  };
  /** An event repeated (a hold, the take's end): what it withdrew was withdrawn when it was first sent. */
  const settled = (event: PositionEvent): PositionEvent => {
    if (!event.withdrawn) return event;
    const { withdrawn: _, ...rest } = event;
    return rest;
  };
  /**
   * refinements.showSaid: the phoneme offset past which word `wi` counts as heard to its end: its end, or, when it ends
   * in a held vowel (a run of one vowel), just past that run's first phoneme.
   */
  const doneAt = (wi: number) => {
    const w = index.words[wi];
    const lastPh = index.phonemes[w.end - 1];
    let k = w.end - 1;
    while (k > w.start && index.phonemes[k - 1] === lastPh) k--;
    return w.end - k >= 2 && !CONSONANTS.has(lastPh) ? k + 1 : w.end;
  };
  /** refinements.showSaid: the phoneme offset just past word `wi`'s last consonant (its start when it has none). */
  const consonantsDone = (wi: number) => {
    const w = index.words[wi];
    for (let k = w.end; k > w.start; k--) if (CONSONANTS.has(index.phonemes[k - 1])) return k;
    return w.start;
  };
  /**
   * refinements.showSaid, refinements.begunByConsonants: the tracker's one test of a word said. The path, stopped in
   * word `at`, has heard it through its last consonant, every consonant (a doubled one once) or all but one in a word of
   * three or more: a word only begun has not. showSaid counts the consonants heard as written (`consonantMatches`).
   * begunByConsonants (`near`) judges a word the run has left, so it gives the doubt to the reciter, since a said word
   * kept dark is worse than one shown a little late: a consonant heard as its engine-confusable partner counts as heard
   * (`consonantNear`), and the one consonant a word of three or more may miss may be its last, which leaves the path
   * just before it.
   */
  const consonantsHeard = (at: number, pos: LocalPosition, near = false) => {
    const w = index.words[at];
    let n = 0;
    for (let k = w.start; k < w.end; k++) if (CONSONANTS.has(index.phonemes[k]) && (k === w.start || index.phonemes[k - 1] !== index.phonemes[k])) n++;
    const miss = n >= 3 ? 1 : 0;
    const heard = near ? pos.consonantNear : pos.consonantMatches;
    return pos.offset >= consonantsDone(at) - (near ? miss : 0) && pos.offset > w.start && heard >= n - miss;
  };
  /**
   * refinements.showSaid: the path, in word `at`, heard the word before it as written, every phoneme or all but one in a
   * word of three or more: a word the path left out whole, or read from other sounds, has not been.
   */
  const previousHeard = (at: number, pos: LocalPosition) => {
    const prev = spokenStep(at, -1);
    if (prev < 0) return false;
    const n = index.words[prev].end - index.words[prev].start;
    return pos.previousMatches >= (n >= 3 ? n - 1 : n);
  };
  /**
   * refinements.showSaid: the path, stopped in word `at` at a pause, heard all of it a stop keeps, every phoneme as
   * written: to its end, but a last short vowel, which a stop drops. A pause before more of it than that (a long vowel
   * not begun or not held to its length) falls inside the word: the reciter goes on with it. A word heard otherwise
   * waits for the next word, as without a pause: shown a little late rather than on a guess.
   */
  const heardToStop = (at: number, pos: LocalPosition) => {
    const w = index.words[at];
    const kept = SHORT_VOWELS.has(index.phonemes[w.end - 1]) ? w.end - 1 : w.end;
    return pos.offset >= kept && pos.wordMatches >= kept - w.start;
  };
  /** refinements.showSaid: word `wi` is its ayah's last spoken word (a pause follows it, not another word). */
  const endsAyah = (wi: number) => {
    const next = spokenStep(wi, 1);
    return next < 0 || index.words[next].surah !== index.words[wi].surah || index.words[next].ayah !== index.words[wi].ayah;
  };
  /**
   * refinements.showSaid: the last word said when the path has heard through phoneme offset `heard`, which falls in word
   * `at`: the word before `at`, once `at` is under way (`NEXT_BEGUN` of its phonemes heard, or its first when the word
   * before ends an ayah), else the word before that, which `at`'s predecessor has left.
   */
  const saidBefore = (at: number, heard: number) => {
    const prev = spokenStep(at, -1);
    const need = prev >= 0 && endsAyah(prev) ? 1 : Math.min(NEXT_BEGUN, index.words[at].end - index.words[at].start);
    return heard - index.words[at].start >= need ? prev : spokenStep(prev, -1);
  };
  /**
   * refinements.showSaid: the last word said on the local search's path. When `settled` (a pause, the take's end, the
   * run left), the word the path is in once every consonant of it was heard (at a pause, `"pause"`, also all of it a
   * stop keeps, `heardToStop`), else the word before it, which the path has left, if the path heard it as written
   * (`previousHeard`; -1, nothing new, otherwise).
   * Otherwise an ayah's last word once heard to its end (the last phoneme heard as written, `stepCost` 0 on the step
   * that reached it), and any other word as `saidBefore` reads the path's word.
   */
  const saidOn = (pos: LocalPosition, settled: boolean | "pause"): number => {
    const at = index.wordAt(Math.max(0, pos.offset - 1));
    if (settled) {
      if (consonantsHeard(at, pos) && (settled !== "pause" || heardToStop(at, pos))) return at;
      return previousHeard(at, pos) ? spokenStep(at, -1) : -1;
    }
    if (endsAyah(at) && pos.offset >= doneAt(at) && pos.steps > 0 && pos.stepCost === 0) return at;
    return saidBefore(at, pos.offset);
  };
  /**
   * refinements.showSaid: the last word said when a lock on `m` is made: its last word if that ends an ayah and was heard
   * to its end (its last phoneme heard as written), else as `saidBefore` reads it.
   */
  const saidAtLock = (m: Match): number => {
    const lastWi = lastWord(m);
    if (m.length === 0) return spokenStep(lastWi, -1);
    let k = m.path.length - 1;
    while (k >= 0 && m.path[k].op === "ins") k--;
    if (endsAyah(lastWi) && m.offset + m.length >= doneAt(lastWi) && m.path[k]?.op === "match") return lastWi;
    return saidBefore(lastWi, m.offset + m.length);
  };
  /** refinements.showSaid: word `wi` is said in the run whose first word is `start`. */
  const sayUpTo = (start: number, wi: number) => {
    if (said.start !== start) said = { start, wi: -1 };
    if (wi > said.wi) said = { start, wi };
  };
  const put = (event: PositionEvent) => {
    let out = event;
    if (carried.length && event.state === "locked") {
      const named = new Set<number>();
      const withdrawn = carried.filter((r) => {
        const wi = wordOf(r);
        if (named.has(wi) || !pageShown.has(wi)) return false;
        named.add(wi);
        return true;
      });
      carried = [];
      if (withdrawn.length) out = { ...event, withdrawn };
    }
    for (const r of out.withdrawn ?? []) pageShown.delete(wordOf(r));
    const c = out.state === "locked" ? out.observed[0] : undefined;
    if (c) for (let wi = wordOf(c.span.start); wi >= 0 && wi <= wordOf(c.word); wi = spokenStep(wi, 1)) pageShown.add(wi);
    shownOut.push(out);
    shownLast = out;
    if (openingContest) pageLit(openingContest, out);
  };
  /**
   * refinements.showSaid: show the locked `event`'s run through its last word said (`said`), at `timeMs`: nothing
   * while the run's first word is not said, and nothing new when that is what shows already, unless `event` carries
   * what an app is told once (a hold's start, a relocation, a span hold, a withdrawal).
   */
  const showUpTo = (event: PositionEvent, timeMs: number) => {
    const c = event.observed[0];
    const start = wordOf(c.span.start);
    const wi = Math.min(wordOf(c.word), said.start === start ? said.wi : -1);
    if (wi < start) return;
    const shown = shownLast?.state === "locked" ? shownLast.observed[0] : undefined;
    const fresh = toldAt !== sent;
    const told = (fresh && (!!event.hold || !!event.relocated)) || carried.length > 0 || event.spanHold !== shownLast?.spanHold;
    if (!told && shown && wordOf(shown.word) === wi && wordOf(shown.span.start) === start) return;
    toldAt = sent;
    if (fresh && wi === wordOf(c.word) && event.timeMs === timeMs) return put(event);
    const word = ref(wi);
    const { trace, hold, relocated, ...rest } = event;
    const news = fresh ? { ...(hold && { hold }), ...(relocated && { relocated }), ...(trace && event.timeMs === timeMs && { trace }) } : {};
    put({ ...rest, ...news, timeMs, observed: [{ ...c, word, span: { ...c.span, end: word } }, ...event.observed.slice(1)] });
  };
  /** Off (refinements.showSaid), the positions to show are the positions `push()` and `end()` return. */
  const positionsOf = (events: EngineEvent[]) => events.filter((e): e is PositionEvent => e.type === "position");
  /** refinements.showSaid: the position `event`, which `emit` sends after `prior`, as it shows (`shown`). */
  const show = (event: PositionEvent, prior: PositionEvent | null) => {
    if (!showSaid) return;
    const c = event.state === "locked" ? event.observed[0] : undefined;
    if (!c) return put(event);
    // A twin copy of the same words: what was said of the copy left is said of this one.
    if (event.relocated) {
      const before = shownLast?.state === "locked" ? shownLast.observed[0] : undefined;
      const all = !!before && prior?.state === "locked" && !!prior.observed[0] && wordOf(before.word) === wordOf(prior.observed[0].word);
      sayUpTo(wordOf(c.span.start), all ? wordOf(c.word) : spokenStep(wordOf(c.word), -1));
    }
    if (!event.withdrawn) return showUpTo(event, event.timeMs);
    carried.push(...event.withdrawn);
    const { withdrawn: _, ...plain } = event;
    showUpTo(plain, event.timeMs);
  };
  /**
   * refinements.showSaid: the lock just sent, shown once its run's words said are known. A lock sent before its local
   * search starts (`startLocal` reads what the lock heard) showed nothing of its new run when it was sent.
   */
  const showLock = () => {
    if (!showSaid || last?.state !== "locked" || !last.observed[0]) return;
    const { withdrawn: _, ...plain } = last;
    showUpTo(plain, last.timeMs);
  };
  /**
   * refinements.showSaid: a position repeated (a hold's start, the take's end) as it shows: the last one shown, at its
   * time.
   */
  const showRepeat = (event: PositionEvent) => {
    if (!showSaid) return;
    if (event.state !== "locked" || shownLast?.state !== "locked") return event.state === "locked" ? undefined : put(event);
    const { withdrawn: _, trace: __, ...rest } = shownLast;
    put({ ...rest, timeMs: event.timeMs, ...(event.hold && { hold: event.hold }), ...(event.trace && { trace: event.trace }) });
  };
  /**
   * refinements.showSaid: after a step of the local search (or settled, as `saidOn` reads it: at a pause, `"pause"`),
   * the run shown as far as it has been said, at `timeMs`. Only the run the last position shows: a jump not shown yet
   * waits for its own position. `leaving`: the page is about to leave the run, which is settled where its path last
   * heard a phoneme as written.
   */
  const noteSaid = (settled: boolean | "pause" = false, leaving = false, timeMs = lastFrame * (frameMs ?? 0)) => {
    if (!showSaid || !local || last?.state !== "locked" || !last.observed[0]) return;
    const start = wordOf(last.observed[0].span.start);
    let pos = local.position();
    const onRun = index.wordAt(pos.origin) === start;
    if (onRun && pos.steps > 0 && pos.stepCost === 0) heardOn = { start, pos: { ...pos } };
    // The page leaves the run: settle it where its path last heard a phoneme as written. The search may already have
    // gone on to the speech that follows (another run's, read against this one) or jumped to it.
    if (leaving) {
      if (heardOn?.start !== start) return;
      pos = heardOn.pos;
    } else if (!onRun) return;
    sayUpTo(start, saidOn(pos, settled));
    const { withdrawn: _, ...plain } = last;
    showUpTo(plain, timeMs);
  };
  /**
   * refinements.showSaid: decoding has reached `decodedTo` with nothing heard after the last phoneme for
   * `PAUSE_QUIET_MS`: the pause settles the word the reciter stopped on, shown at the time the quiet reached that.
   */
  const settlePause = (decodedTo: number | undefined) => {
    if (!showSaid || !local || heardEnd < 0 || !frameMs || decodedTo === undefined) return;
    const quiet = Math.max(minGapFrames(frameMs), Math.round(PAUSE_QUIET_MS / frameMs));
    if (decodedTo - heardEnd - 1 >= quiet) noteSaid("pause", false, Math.max(lastFrame, heardEnd + 1 + quiet) * frameMs);
  };
  const emit = (event: PositionEvent, trace: (from: TrackerState | null) => TrackerTrace, since?: number): PositionEvent => {
    const from = last?.state ?? null;
    const out = marked(withdrawBegun ? withdrawing(event, since) : event);
    // refinements.showSaid: a position that leaves the run shown (another run, or no lock) settles it first, as a pause
    // does: the word the reciter left is said once its consonants were heard.
    const c = out.state === "locked" ? out.observed[0] : undefined;
    if (showSaid && last?.state === "locked" && last.observed[0] && (!c || wordOf(c.span.start) !== wordOf(last.observed[0].span.start))) noteSaid(true, true);
    // refinements.declaredMove: without showSaid the positions are what the page shows.
    if (openingContest && !showSaid) pageLit(openingContest, out);
    const prior = last;
    last = out;
    sent++;
    const traced = options.trace ? { ...out, trace: trace(from) } : out;
    show(traced, prior);
    return traced;
  };
  const bare = (event: PositionEvent) => {
    const { spanHold: _, timeMs: __, trace: ___, withdrawn: ____, ...rest } = event;
    return JSON.stringify(rest);
  };
  const unchanged = (event: PositionEvent) => !!last && bare(last) === bare(event);

  const evaluate = (final: boolean): PositionEvent[] => {
    if (!observed) return [];
    const a = assessOutsidePhrases();
    if (a.lock && !everLocked && longestSpanAt(allSpans, lastWord(a.lock)) !== null) arm = true;
    if (unchanged(a.event) && !final) return [];
    if (a.lock && !final) lockedOn = a.lock;
    const out = emit(a.event, a.trace);
    // refinements.noLostBeforeLock: a memorization session has not lost its place before it first found it. Only the
    // report changes: `last` keeps the assessment, so trimming and re-emission work as before.
    if (noLostBeforeLock && !everLocked && out.state === "lost" && !final) {
      const reported: PositionEvent = { ...out, state: "acquiring", ...(out.trace && { trace: { ...out.trace, transition: { ...out.trace.transition, to: "acquiring" as const } } }) };
      // refinements.showSaid: the page is shown the same report.
      const k = shownOut.lastIndexOf(out);
      if (k >= 0) shownOut[k] = reported;
      if (shownLast === out) shownLast = reported;
      return [reported];
    }
    return [out];
  };

  const quietTrace = (from: TrackerState | null, to: TrackerState, shortlist: TraceCandidate[] = []): TrackerTrace => ({
    shortlist,
    transition: { from, to },
    path: null,
    decision: null,
  });

  const observe = (ph: DecodedPhoneme) => {
    breath = heardEnd >= 0 && isPause(heardEnd, ph.start) && ph.conf < BREATH_CONFIDENCE;
    heardEnd = ph.end;
    observedFrames.push({ start: ph.start, end: ph.end });
    observed += ph.p;
    observedConf.push(ph.conf);
    if (!everLocked) {
      raw += ph.p;
      for (let i = 0; i < ph.p.length; i++) rawConf.push(ph.conf);
    }
    lastFrame = Math.max(lastFrame, ph.end + 1);
    if (excursion) {
      excursion.phones += ph.p;
      if (excursion.phones.length > 480) {
        excursion.phones = excursion.phones.slice(-480);
        excursion.trimmed = true;
      }
    }
  };
  /** Keep only the latest `n` observed phonemes. */
  const keepLast = (n: number) => {
    const drop = observed.length - n;
    if (drop <= 0) return;
    observed = observed.slice(drop);
    observedConf = observedConf.slice(drop);
    observedFrames.splice(0, drop);
  };

  /**
   * Record one phoneme for phrase detection, with the local search's state before it and its cost; returns how many
   * of the latest phonemes form a whole phrase that ends here, and which.
   */
  const hear = (p: string, state: unknown, cost: number): { name: string; length: number } | null => {
    if (!phraseList.length) return null;
    heard += p;
    heardCost.push(cost);
    heardState.push(state);
    if (heard.length > HEARD) {
      heard = heard.slice(1);
      heardCost = heardCost.slice(1);
      heardState = heardState.slice(1);
    }
    for (const phrase of phraseList) {
      const n = phrase.p.length;
      const lo = Math.max(0, heard.length - Math.ceil(n * 1.6) - 4);
      const a = alignSubstring(heard, lo, heard.length, phrase.p, () => true);
      if (!a || a.end < heard.length - 1 || a.cost / UNIT / n > PHRASE_MAX_COST) continue;
      // The phrase must be finished: its last two phonemes heard, not just cheap to leave out.
      const heardEnd = a.path.filter((o) => o.op === "match" && o.obsAt >= n - 2).length;
      if (heardEnd < 2) continue;
      return { name: phrase.name, length: heard.length - a.start };
    }
    return null;
  };
  /** The latest phonemes are a phrase still being said: hold off locking, a prefix like "a'udhu billahi" is Quran text too. */
  const phraseInProgress = (): boolean => {
    for (const phrase of phraseList) {
      const n = phrase.p.length;
      for (let k = Math.min(n - 2, heard.length); k >= PHRASE_PREFIX; k--) {
        const a = alignSubstring(phrase.p, 0, n - 2, heard.slice(-k), (o) => o === 0, { strictStart: true });
        if (a && a.cost / UNIT / k <= PHRASE_MAX_COST / 2) return true;
      }
    }
    return false;
  };
  /** The window's latest phonemes restate its start: a restart still being said, not yet a repeat to collapse. */
  const repeatInProgress = (): boolean => {
    for (let k = Math.min(MAX_REPEAT, observed.length >> 1); k >= RESTART_PREFIX; k--) {
      const a = alignSubstring(observed, 0, observed.length - k, observed.slice(-k), (o) => o === 0, { strictStart: true });
      if (a && a.cost / UNIT / k <= REPEAT_MAX_COST) return true;
    }
    return false;
  };
  /** Assess, but never lock while a phrase or a repeat is in progress, or right after a phrase ended before new speech. */
  const assessOutsidePhrases = () => {
    const a = assess();
    const phraseJustEnded = phraseCut > 0 && observed.length - phraseCut < minPhonemes;
    if (!a.lock || !phraseList.length || (!phraseJustEnded && !phraseInProgress() && !repeatInProgress())) return a;
    const event: PositionEvent = { ...a.event, state: "acquiring", expected: null };
    return { ...a, event, lock: null, top: null, trace: (from: TrackerState | null) => ({ ...a.trace(from), transition: { from, to: "acquiring" as const } }) };
  };
  /**
   * An immediately repeated block at the end of the window ("Alif Lam Mim" twice) is a repetition, not two ayahs: keep
   * one copy, so global search doesn't prefer passages that happen to continue like the repeat.
   */
  const collapseRepeat = () => {
    for (let n = Math.min(MAX_REPEAT, observed.length >> 1); n >= MIN_REPEAT; n--) {
      const tail = observed.slice(-n);
      const before = observed.slice(-2 * n, -n);
      const a = alignSubstring(before, 0, n, tail, (o) => o === 0, { strictStart: true });
      if (a && a.end >= n - 1 && a.cost / UNIT / n <= REPEAT_MAX_COST) {
        const at = observed.length - 2 * n;
        const collapsed = observed.slice(0, at) + tail;
        const conf = [...observedConf.slice(0, at), ...observedConf.slice(-n)];
        // Quran text can itself contain a near-repeat (من أجر إن أجري). Removing it would manufacture evidence
        // for a different passage. Keep the whole phrase whenever it fits the reference at least as well.
        const cost = (text: string, confidence: number[]) => Math.min(...fuzzyMatches(text, confidence).map((m) => m.cost));
        if (cost(observed, observedConf) <= cost(collapsed, conf)) continue;
        observed = collapsed;
        observedConf = [...observedConf.slice(0, at), ...observedConf.slice(-n)];
        observedFrames.splice(at, n);
        return;
      }
    }
  };
  const forget = () => {
    heard = "";
    heardCost = [];
    heardState = [];
  };

  /** Drop the latest `n` observed phonemes. */
  const dropLast = (n: number) => {
    const keep = Math.max(0, observed.length - n);
    observed = observed.slice(0, keep);
    observedConf = observedConf.slice(0, keep);
    observedFrames.length = keep;
  };

  /**
   * How a lock's match entered its last word: on a risky step (its first phoneme there deleted or not heard as
   * written, or one that may be a breath: heard after a pause, below `BREATH_CONFIDENCE`; `heardAt` gives an observed
   * phoneme's frames and confidence), and how many of that word's phonemes it heard as written.
   */
  const lockEntry = (m: Match, heardAt: HeardAt) => {
    const w = index.words[lastWord(m)];
    const inWord = m.path.filter((o) => o.op !== "ins" && m.offset + o.refAt >= w.start && m.offset + o.refAt < w.end);
    const first = inWord[0];
    let risky = !first || first.op !== "match";
    if (first?.op === "match") {
      const [at, before] = [heardAt(first.obsAt), heardAt(first.obsAt - 1)];
      risky = !!at && !!before && isPause(before.end, at.start) && at.conf < BREATH_CONFIDENCE;
    }
    const once = (refAt: number, r: string) => m.offset + refAt === w.start || index.phonemes[m.offset + refAt - 1] !== r;
    const consonants = inWord.filter((o) => o.op === "match" && CONSONANTS.has(o.ref) && once(o.refAt, o.ref)).length;
    // refinements.begunByConsonants: a consonant heard as its engine-confusable partner counts as heard.
    const near = inWord.filter((o) => (o.op === "match" || (o.op === "sub" && isConfusable(o.ref, o.obs))) && CONSONANTS.has(o.ref) && once(o.refAt, o.ref)).length;
    // refinements.consonantConfirm: the path's last `CONFIRM` heard phonemes up to the word's first, as the local
    // search's step costs read them: not all heard as written.
    let weak = !first;
    if (first) {
      let heard = 0;
      for (let k = m.path.indexOf(first); k >= 0 && heard < CONFIRM; k--) {
        const o = m.path[k];
        if (o.op !== "del") heard++;
        if (o.op !== "match") weak = true;
      }
    }
    return { risky, matches: inWord.filter((o) => o.op === "match").length, consonants, near, weak: weak || risky };
  };
  /** An observed phoneme's frames and decoder confidence, by its index in the observed string. */
  const observedAt: HeardAt = (j) => (observedFrames[j] ? { ...observedFrames[j], conf: observedConf[j] } : undefined);
  /**
   * refinements.consonantConfirm, refinements.rivalHold: phonemes `heard` in word `wi` fit the opening of a word a
   * skip forward from it lands on (a later word of its ayah, or the next ayah's first word; with `passageWindow`, none
   * past the declared passage's end while the search is inside it) at no more cost than they fit the word itself: the
   * reciter may have left the word out and be saying that one. With `misheard`, only when the word itself does not fit
   * them as written: phonemes heard as written are the word's, the run's order decides, and the word shows at once. It
   * reads only what was heard in the word, so a lock and the run that follows it judge a word alike, wherever the
   * pushes fall.
   */
  const fitsElsewhere = (wi: number, heard: string, misheard: boolean) => {
    if (!heard) return false;
    const fit = (start: number) => {
      const f = growingFit(index.phonemes, start, Math.min(index.phonemes.length, start + 2 * heard.length + 2));
      for (const p of heard) f.push(p);
      return f.cost();
    };
    const own = fit(index.words[wi].start);
    if (own === null || (misheard && own === 0)) return false;
    const end = ayahEndOf(wi);
    let last = end < index.words.length && index.words[end].surah === index.words[wi].surah ? spoken(end) : -1;
    if (last < 0) last = end - 1;
    if (boundedAt(index.words[wi].start)) last = Math.min(last, bounds!.to);
    for (let k = wi + 1; k <= last; k++) {
      if (index.words[k].end <= index.words[k].start) continue;
      const cost = fit(index.words[k].start);
      if (cost !== null && cost <= own) return true;
    }
    return false;
  };
  /** The phonemes a lock on `m` heard in word `wi`, from the first one its path placed there. */
  const heardIn = (m: Match, wi: number) => {
    const w = index.words[wi];
    let heard = "";
    for (const op of m.path) {
      if (op.op === "del" || (op.op !== "ins" ? m.offset + op.refAt < w.start : !heard)) continue;
      heard += op.op === "match" ? op.ref : op.obs;
    }
    return heard;
  };
  /**
   * The word a lock on `m` shows, on the terms the local search shows a word by once it takes over: not inside a
   * doubled onset (`onsetRun`), a later word entered on a risky step only once two of its phonemes (all of a shorter
   * word's) were heard as written (`confirmAdvance`), and, unless it opens an ayah, a later word the lock is still
   * inside, while another word's opening fits what was heard of it as well (`fitsElsewhere`), only once two of its
   * consonants were heard as written if it was entered on weak evidence (`consonantConfirm`), and four of its phonemes
   * if what was heard of it is not all as written (`rivalHold`).
   */
  const lockWord = (m: Match, heardAt: HeardAt): number => {
    const last = lastWord(m);
    if (onsetRun) {
      const shown = beforeOnset(last, m.offset + m.length, m.offset);
      if (shown !== last) return shown;
    }
    const before = spokenStep(last, -1);
    if (confirmAdvance && last > firstWord(m)) {
      const { risky, matches } = lockEntry(m, heardAt);
      if (risky && matches < Math.min(ADVANCE_MATCHES, index.words[last].end - index.words[last].start) && before >= firstWord(m)) return before;
    }
    if ((consonantConfirm || rivalHold) && before >= firstWord(m) && !opensAyah(last) && m.offset + m.length < index.words[last].end) {
      const { matches, consonants, weak } = lockEntry(m, heardAt);
      const length = index.words[last].end - index.words[last].start;
      const heard = heardIn(m, last);
      if (consonantConfirm && weak && consonants < Math.min(CONFIRM_CONSONANTS, consonantsOf(last)) && fitsElsewhere(last, heard, false)) return before;
      if (rivalHold && matches < Math.min(RIVAL_MATCHES, length) && fitsElsewhere(last, heard, true)) return before;
    }
    return last;
  };

  const startLocal = (m: Match, heardAt: HeardAt = observedAt) => {
    // refinements.showSaid: a lock on another run, sent after this, settles the run shown first (as `emit` does).
    if (showSaid && last?.state === "locked" && last.observed[0] && wordOf(last.observed[0].span.start) !== firstWord(m)) noteSaid(true, true);
    // refinements.declaredStart: a new lock (a relocation, a reacquisition) ends the take's opening contest, and the
    // second look a move waits for (refinements.declaredMove).
    openingContest = null;
    rejudge = null;
    // refinements.confirmAdvance: the run goes on counting the matches the lock heard in its last word, from how it entered it
    // (a relocation to a twin passage starts after a whole word, heard).
    const entry = (confirmAdvance || consonantConfirm || rivalHold || withdrawBegun || showSaid) && m.length > 0 ? lockEntry(m, heardAt) : { risky: false, matches: 0, consonants: 0, near: 0, weak: false };
    runWord = m.length > 0 ? lastWord(m) : index.wordAt(Math.max(0, m.offset - 1));
    riskyEntry = confirmAdvance && entry.risky;
    weakEntry = consonantConfirm && entry.weak;
    wordHeard = { word: runWord, heard: (consonantConfirm || rivalHold) && m.length > 0 ? heardIn(m, runWord) : "" };
    search++;
    if (showSaid) sayUpTo(firstWord(m), saidAtLock(m));
    local = createLocalSearch(
      index,
      { offset: m.offset + m.length, origin: m.offset, edits: Math.round(m.per * m.length), wordMatches: entry.matches, consonantMatches: entry.consonants, consonantNear: entry.near },
      {
        countMatches: confirmAdvance || consonantConfirm || rivalHold || withdrawBegun || showSaid,
        countConsonants: consonantConfirm || rivalHold || withdrawBegun || showSaid,
        countSince: withdrawBegun,
        countNear: begunByConsonants,
        restartNoEntry,
        ...(bounds && { passage: bounds.phonemes }),
      },
    );
    weak = false;
    pending = null;
    reacquiring = false;
    holding = null;
    if (everLocked) arm = false;
    everLocked = true;
    raw = "";
    rawConf = [];
    phraseLeads.clear();
    onsetCut = 0;
    surahStart = false;
    basmalaHeard = false;
    fatihaNext = false;
    takbirHeard = false;
    afterBasmala = null;
    afterPhrase = false;
    twin = null;
    keepLast(0);
  };

  /** Memorization holds the expected bookmark where a jump left it; free mode follows the reciter. */
  const expectedFor = (word: WordRef): WordRef => frozen ?? word;
  /** Neither a jump nor a passage change happened, so the bookmark stays where local tracking last trusted it. */
  const expectedWhileLost = () => frozen ?? steady;

  const near = (m: Match, offset: number) => {
    // refinements.passageWindow: the local search, inside the passage, does not jump past its end.
    if (boundedAt(offset) && lastWord(m) > bounds!.to) return false;
    const a = index.words[lastWord(m)];
    const b = index.words[index.wordAt(Math.max(0, offset - 1))];
    return a.surah === b.surah && Math.abs(a.ayah - b.ayah) <= NEAR_AYAHS;
  };
  /** Same passage the local search would keep: not a relocation. */
  const nearWord = (a: WordRef, b: WordRef) => a.surah === b.surah && Math.abs(a.ayah - b.ayah) <= NEAR_AYAHS;

  const spokenStep = (wi: number, dir: -1 | 1): number => {
    if (wi < 0) return -1;
    for (let k = wi + dir; k >= 0 && k < index.words.length; k += dir) {
      if (index.words[k].end > index.words[k].start) return k;
    }
    return -1;
  };
  /** Heard words of a destination match: reference word, and the phonemes the alignment heard for it. */
  const destinationWords = (m: Match): DestinationWord[] => {
    const groups: { wi: number; heard: string }[] = [];
    let cur: { wi: number; heard: string } | null = null;
    let lead = "";
    for (const op of m.path) {
      if (op.op === "ins") {
        if (cur) cur.heard += op.obs;
        else lead += op.obs;
        continue;
      }
      const wi = index.wordAt(Math.max(0, Math.min(index.phonemes.length - 1, m.offset + op.refAt)));
      if (!cur || cur.wi !== wi) {
        cur = { wi, heard: lead };
        lead = "";
        groups.push(cur);
      }
      if (op.op === "match") cur.heard += op.ref;
      else if (op.op === "sub") cur.heard += op.obs;
    }
    if (lead) {
      if (groups.length) groups[0].heard = lead + groups[0].heard;
      else groups.push({ wi: -1, heard: lead });
    }
    return groups.filter((g) => g.heard.length > 0 && g.wi >= 0).map((g) => ({
      destination: index.phonemes.slice(index.words[g.wi].start, index.words[g.wi].end),
      heard: g.heard,
    }));
  };
  const anchorWordsFrom = (wi: number, n: number): string[] => {
    const out: string[] = [];
    for (let k = wi; k < index.words.length && out.length < n; k++) {
      const w = index.words[k];
      if (w.end > w.start) out.push(index.phonemes.slice(w.start, w.end));
    }
    return out;
  };
  /**
   * A distant lock relocates only when enough heard words fit it better than the anchor plus one inserted word.
   * The anchor start is whichever of the cursor and its neighbours that hypothesis explains cheapest: the cursor
   * sits on the word before a shared word as often as on the shared word itself.
   */
  const relocationCommits = (m: Match, anchorWi: number): boolean => {
    const heard = destinationWords(m);
    if (heard.length < minWords) return false;
    const seeds = [anchorWi, spokenStep(anchorWi, -1), spokenStep(anchorWi, 1), spokenStep(spokenStep(anchorWi, 1), 1)].filter((i) => i >= 0);
    let best: { count: number; total: number } | null = null;
    for (const seed of seeds) {
      const fit = relocationSupport(anchorWordsFrom(seed, heard.length + 1), heard, minSeparation, separationScale);
      if (fit && (!best || fit.total < best.total)) best = fit;
    }
    return !!best && best.count >= minWords;
  };

  /**
   * The anchor stays locked and relocation search continues. One event tells the checker to downgrade issues open
   * now; `last` stays the plain lock, so later re-emits of the anchor compare unchanged.
   */
  const beginHold = (): PositionEvent => {
    if (twin) {
      twinSettled.add(spanId(twin.span));
      twin = null;
    }
    if (memorization) frozen ??= steady;
    const anchor = settled(last!);
    holding = anchor;
    local = null;
    weak = false;
    pending = null;
    reacquiring = true;
    const event: PositionEvent = { ...anchor, timeMs: lastFrame * (frameMs ?? 0), hold: "start" };
    const out = options.trace ? { ...event, trace: quietTrace(anchor.state, "locked") } : event;
    showRepeat(out);
    return out;
  };

  const lostEvent = (timeMs: number, trace: (from: TrackerState | null) => TrackerTrace): PositionEvent =>
    emit(
      { type: "position", v: EVENT_SCHEMA_VERSION, timeMs, referenceVersion: index.version, expected: expectedWhileLost(), observed: [], state: "lost" },
      trace,
    );

  /** Next word that has its own phonemes. Assimilated words are skipped. */
  const spoken = (wi: number) => {
    let k = wi;
    while (k < index.words.length && index.words[k].end === index.words[k].start) k++;
    return k;
  };

  const ayahStartOf = (wi: number) => {
    let k = wi;
    const w = index.words[wi];
    while (k > 0 && index.words[k - 1].surah === w.surah && index.words[k - 1].ayah === w.ayah) k--;
    return k;
  };
  const ayahEndOf = (wi: number) => {
    let k = wi + 1;
    const w = index.words[wi];
    while (k < index.words.length && index.words[k].surah === w.surah && index.words[k].ayah === w.ayah) k++;
    return k;
  };
  const phonesBetween = (from: number, to: number) => {
    if (from < 0 || to <= from || to > index.words.length) return "";
    return index.phonemes.slice(index.words[from].start, index.words[to - 1].end);
  };
  const skeleton = (from: number, to: number) => {
    let s = "";
    for (let i = from; i < to; i++) s += `${letterSkeleton(index.wordText?.[i] ?? "")}\0`;
    return s;
  };
  /**
   * A jump off the in-order ayah onto a parallel copy. Closed when a later lock is that copy's in-order
   * continuation and the skipped-over text fits the excursion's phones no worse than the copy does.
   * Exact copies tie. A one-word variant pays that word's edits, so the heard copy is the cheaper one.
   */
  const beginExcursion = (fromWi: number, m: Match) => {
    const destWi = firstWord(m);
    if (fromWi < 0 || destWi < 0) return;
    if (excursion && destWi >= excursion.dest && destWi < excursion.dest + excursion.length) return;
    const at = lastWord(m);
    const span = longestSpanAt(allSpans, destWi);
    if (span) {
      const dest = span.starts.find((st) => destWi >= st && destWi < st + span.length);
      const home = span.starts.find((st) => fromWi >= st && fromWi < st + span.length);
      if (dest !== undefined && home !== undefined && dest !== home) {
        excursion = { home, dest, length: span.length, entry: Math.max(0, Math.min(span.length - 1, at - dest)), phones: "", trimmed: false };
        return;
      }
    }
    const homeLo = fromWi < index.words.length ? ayahEndOf(fromWi) : -1;
    const destLo = destWi < index.words.length ? ayahStartOf(destWi) : -1;
    if (homeLo >= 0 && homeLo < index.words.length && destLo >= 0 && destLo !== homeLo) {
      const homeHi = ayahEndOf(homeLo);
      const destHi = ayahEndOf(destLo);
      const length = homeHi - homeLo;
      if (length > 0 && destHi - destLo === length && skeleton(homeLo, homeHi) === skeleton(destLo, destHi)) {
        excursion = { home: homeLo, dest: destLo, length, entry: Math.max(0, Math.min(length - 1, at - destLo)), phones: "", trimmed: false };
        return;
      }
    }
    excursion = null;
  };
  /** Retro locked event, or null when this lock does not close the excursion. */
  const finishExcursion = (destWi: number): PositionEvent | null => {
    const ex = excursion;
    if (!ex || ex.trimmed || destWi < 0) return null;
    if (destWi >= ex.dest && destWi < ex.dest + ex.length) return null;
    const cont = spoken(ex.home + ex.length);
    // Landing here is the commit: the search already preferred this continuation by the commit margin.
    if (destWi < cont || destWi >= cont + 4) return null;
    const from = ex.entry;
    const homeText = phonesBetween(ex.home + from, ex.home + ex.length);
    const destText = phonesBetween(ex.dest + from, ex.dest + ex.length);
    const homeCost = anchorFitCost(homeText, ex.phones);
    const destCost = anchorFitCost(destText, ex.phones);
    // Exact copies tie. A one-word variant pays that word, so the copy that was heard stays cheaper.
    if (homeCost === null || destCost === null || homeCost > destCost) return null;
    const end = ex.home + ex.length - 1;
    const event: PositionEvent = {
      type: "position",
      v: EVENT_SCHEMA_VERSION,
      timeMs: lastFrame * (frameMs ?? 0),
      referenceVersion: index.version,
      expected: expectedFor(ref(ex.home)),
      observed: [{ word: ref(end), span: { start: ref(ex.home), end: ref(end) }, confidence: 1, cost: 0, per: 0 }],
      state: "locked",
      relocated: { from: ref(ex.dest), to: ref(ex.home) },
    };
    return emit(event, (fromState) => quietTrace(fromState, "locked"));
  };
  const takeJump = (fromWi: number, m: Match): PositionEvent | null => {
    const closing = finishExcursion(firstWord(m));
    if (closing) {
      excursion = null;
      return closing;
    }
    beginExcursion(fromWi, m);
    return null;
  };

  /**
   * While the lock sits in a slot-differing span, score heard phones against each member's next word. The current
   * copy wins when its own continuation fits. Another copy wins only when the heard word fits that continuation and
   * beats every other member by the commit margin. Held slot accusations then follow that copy.
   */
  const considerTwin = (p: string, pos: LocalPosition): PositionEvent | null => {
    const wi = index.wordAt(Math.max(0, pos.offset - 1));
    const outside = longestSpanAt(allSpans, wi) === null;
    if (!twin && !arm) return null;
    if (!twin) {
      if (outside) {
        arm = false;
        return null;
      }
      const span = longestSpanAt(allSpans, wi);
      if (!span || twinSettled.has(spanId(span))) return null;
      const memberStart = span.starts.find((st) => wi >= st && wi < st + span.length);
      if (memberStart === undefined) return null;
      const endWi = memberStart + span.length;
      if (endWi >= index.words.length) return null;
      twin = { span, memberStart, endPhone: index.words[endWi].start, buf: "" };
    }
    // A clean phone still inside the span belongs to the shared text, not to the continuation.
    if (pos.offset <= twin.endPhone && pos.stepCost < 1) twin.buf = "";
    else {
      twin.buf += p;
      if (twin.buf.length > 96) twin.buf = twin.buf.slice(-96);
    }
    const span = twin.span;
    const candidates = span.starts.map((st) => {
      const w = spoken(st + span.length);
      const word = index.words[w];
      return { id: st, phonemes: word ? index.phonemes.slice(word.start, word.end) : "" };
    });
    const pick = continuationDecision(twin.buf, candidates, twin.memberStart, (n) => commitMargin(n, minSeparation, separationScale), maxCost);
    // The continuation settled the copy. A later global return must not rebind it again.
    if (pick) excursion = null;
    // Past the first continuation word with no unique other copy: keep the locked one and let the window move on.
    // The continuation word itself sits outside the span; the open span still gets to judge it.
    if (!pick && wi < twin.memberStart + twin.span.length + 4) return null;
    if (!pick) {
      const fromWi = twin.memberStart;
      twinSettled.add(spanId(twin.span));
      twin = null;
      if (outside) arm = false;
      const word = ref(wi);
      emittedSinceLock = true;
      return emit({
        type: "position",
        v: EVENT_SCHEMA_VERSION,
        timeMs: lastFrame * (frameMs ?? 0),
        referenceVersion: index.version,
        expected: expectedFor(word),
        observed: [{ word, span: { start: ref(index.wordAt(pos.origin)), end: word }, confidence: 1, cost: pos.recentCost, per: 0 }],
        state: "locked",
        relocated: { from: ref(fromWi), to: ref(fromWi) },
      }, (from) => quietTrace(from, "locked"));
    }
    const fromWi = twin.memberStart;
    const toWi = pick.id;
    twinSettled.add(spanId(span));
    twin = null;
    if (outside) arm = false;
    const at = toWi === fromWi ? wi : spoken(toWi + span.length);
    const origin = toWi === fromWi ? index.wordAt(pos.origin) : toWi;
    const word = ref(at);
    const event: PositionEvent = {
      type: "position",
      v: EVENT_SCHEMA_VERSION,
      timeMs: lastFrame * (frameMs ?? 0),
      referenceVersion: index.version,
      expected: expectedFor(word),
      observed: [{
        word,
        span: { start: ref(origin), end: word },
        confidence: 1,
        cost: pos.recentCost,
        per: 0,
      }],
      state: "locked",
      relocated: { from: ref(fromWi), to: ref(toWi) },
    };
    emittedSinceLock = true;
    if (toWi !== fromWi) {
      const dest = index.words[at];
      if (!dest) return null;
      stepCosts.length = 0;
      startLocal({ offset: dest.end, length: 0, cost: 0, per: 0, votes: 0, path: [] });
    }
    return emit(event, (from) => quietTrace(from, "locked"));
  };

  /**
   * refinements.onsetRun: the word to show for the local search's word `wi`. While the search is in the first half of
   * `wi`'s leading run of one sound (a doubled onset: a sun letter, idgham, ٱللَّه), that sound still closes the word
   * before, which stays shown when the run reached it; `wi` shows once the search reaches a phoneme of the run's second
   * half (more than half the run, rounded up, heard). The second half, not the run's end: the rule needs only to keep
   * the closing half off the next word, and waiting for the end would hold the word as long as the reciter holds the
   * doubled sound. A word that is nothing but the run (the reference gives a few words only a nasal, ںںں, and reads
   * their other sounds into the next word) has no phoneme of its own after it: the next word's would show both at
   * once, so such a word shows as rc.1 shows it.
   */
  const beforeOnset = (wi: number, offset: number, origin: number): number => {
    const w = index.words[wi];
    let run = 1;
    while (w.start + run < w.end && index.phonemes[w.start + run] === index.phonemes[w.start]) run++;
    if (run < 2 || run === w.end - w.start || offset - w.start > Math.ceil(run / 2)) return wi;
    const before = spokenStep(wi, -1);
    return before >= 0 && before >= index.wordAt(origin) ? before : wi;
  };

  /** Word `wi` is its ayah's first spoken word. */
  const opensAyah = (wi: number) => spoken(ayahStartOf(wi)) === wi;
  /** refinements.consonantConfirm: the consonants of word `wi`, a doubled one once (as the local search counts them). */
  const consonantsOf = (wi: number) => {
    const w = index.words[wi];
    let n = 0;
    for (let k = w.start; k < w.end; k++) if (CONSONANTS.has(index.phonemes[k]) && (k === w.start || index.phonemes[k - 1] !== index.phonemes[k])) n++;
    return n;
  };
  /**
   * refinements.consonantConfirm: the run's word `wi`, entered on weak evidence, has fewer than `CONFIRM_CONSONANTS` of
   * its consonants heard as written, and what was heard of it fits another word's opening as well (`fitsElsewhere`).
   */
  const unconfirmed = (wi: number, pos: LocalPosition) =>
    consonantConfirm && weakEntry && pos.consonantMatches < Math.min(CONFIRM_CONSONANTS, consonantsOf(wi)) && wordHeard.word === wi &&
    fitsElsewhere(wi, wordHeard.heard, false);
  /**
   * refinements.rivalHold: what was heard in word `wi`, not all as written, fits another word's opening in the window
   * as well (`fitsElsewhere`), and fewer than `RIVAL_MATCHES` of its phonemes (all of a shorter word's) were heard as
   * written.
   */
  const contested = (wi: number, pos: LocalPosition) =>
    rivalHold && pos.wordMatches < Math.min(RIVAL_MATCHES, index.words[wi].end - index.words[wi].start) && wordHeard.word === wi &&
    fitsElsewhere(wi, wordHeard.heard, true);

  /** The local search's position as an event, or null when its word and run start haven't changed. */
  const localEvent = (shortlist: TraceCandidate[]): PositionEvent | null => {
    const pos = local!.position();
    const at = index.wordAt(pos.offset - 1);
    // refinements.declaredStart: while the take's speech may still be another ayah's opening, the declared opening's
    // words show only as far as every rival in play shares them.
    const shown = Math.min(onsetRun ? beforeOnset(at, pos.offset, pos.origin) : at, openingContest ? openingContest.shows : Infinity);
    const word = ref(shown);
    const length = pos.offset - pos.origin;
    const per = length ? pos.edits / length : 0;
    const confidence = Math.max(0, 1 - per / maxPer) * Math.max(0, 1 - pos.recentCost / maxCost);
    const candidate: PositionCandidate = { word, span: { start: ref(index.wordAt(pos.origin)), end: word }, confidence, cost: pos.recentCost, per };
    const prev = last!.observed[0];
    if (last!.state === "locked" && JSON.stringify([prev.word, prev.span.start]) === JSON.stringify([word, candidate.span.start])) return null;
    const newRun = JSON.stringify(prev.span.start) !== JSON.stringify(candidate.span.start);
    // A weak local search jumps around chasing evidence that may belong elsewhere; only its current run may advance. So
    // too while near rivals hold the take's opening contest open (refinements.declaredMove): the speech is one of theirs.
    if ((weak || openingContest?.held) && newRun) return null;
    // A skip or restart shows once its landing fits: at a jump's onset, local skips chase audio that belongs elsewhere.
    if (newRun && stepCosts.length >= CONFIRM && stepCosts.slice(-CONFIRM).reduce((a, b) => a + b, 0) / CONFIRM > CONFIRM_COST) return null;
    // refinements.confirmAdvance: a step onto a later word of the run, entered on a risky step (a phoneme not heard as
    // written, or one that may be a breath), shows once that word is heard, not on its first sound: one misheard sound
    // after a pause otherwise lights the next word (often the next ayah's first). A clean entry shows at once.
    if (at !== runWord) {
      runWord = at;
      riskyEntry = breath || pos.stepCost > 0;
      // refinements.consonantConfirm: one of the run's last `CONFIRM` phonemes, this word's first among them, was not
      // heard as written.
      weakEntry = consonantConfirm && (riskyEntry || stepCosts.some((c) => c > 0));
    }
    const later = !newRun && last!.state === "locked" && shown === at && refOrder(word, prev.word) > 0;
    if (confirmAdvance && riskyEntry && later && pos.wordMatches < Math.min(ADVANCE_MATCHES, index.words[at].end - index.words[at].start)) return null;
    let observed = candidate;
    // refinements.consonantConfirm, refinements.rivalHold: a later word of the run that is not an ayah's first word
    // waits for its own evidence while the search is still inside it. Meanwhile the word before it, if the run has gone
    // past it without showing it and heard it as a visit counts a word read (`ADVANCE_MATCHES`), shows on its own: no
    // word that was heard is kept dark, and none shows with the next one. A word the path left out whole stays dark.
    if (later && !opensAyah(at) && pos.offset < index.words[at].end && (unconfirmed(at, pos) || contested(at, pos))) {
      const before = spokenStep(at, -1);
      if (before < 0 || refOrder(ref(before), prev.word) <= 0) return null;
      if (pos.previousMatches < Math.min(ADVANCE_MATCHES, index.words[before].end - index.words[before].start)) return null;
      observed = { ...candidate, word: ref(before), span: { ...candidate.span, end: ref(before) } };
    }
    const event: PositionEvent = {
      type: "position",
      v: EVENT_SCHEMA_VERSION,
      timeMs: lastFrame * (frameMs ?? 0),
      referenceVersion: index.version,
      expected: expectedFor(observed.word),
      observed: [observed],
      state: "locked",
    };
    emittedSinceLock = true;
    return emit(event, (from) => quietTrace(from, "locked", shortlist), pos.since);
  };

  /**
   * Global search over the latest phonemes, until it locks somewhere. While `holding`, the anchor stays locked:
   * non-locked assessments are not emitted. A lock off the anchor emits `lost` and then the new place; a lock on
   * the same passage emits only that lock, so tokens heard in the gap stay in the checker's window.
   */
  const reacquire = (): PositionEvent[] => {
    keepLast(WINDOW);
    if (phraseList.length) collapseRepeat();
    const a = assessOutsidePhrases();
    // After a basmala ends a surah, the next surah is the expected continuation: its opening on top is enough.
    // Al-Fatiha excepted: in prayer any surah follows it.
    if (!a.lock && afterBasmala && afterBasmala.from.surah !== 1 && a.top) {
      const first = index.words[firstWord(a.top)];
      if (first.surah === afterBasmala.from.surah + 1 && first.ayah === 1) {
        a.lock = a.top;
        const c = toCandidate(a.top, a.event.observed[0]?.confidence ?? 0);
        a.event = { ...a.event, state: "locked", observed: [c], expected: c.word };
      }
    }
    if (a.lock) {
      const anchorWord = holding?.observed[0]?.word;
      const dest = a.event.observed[0]?.word;
      const release = !!(holding && anchorWord && dest && !nearWord(anchorWord, dest));
      const anchorWi = anchorWord ? index.words.findIndex((w) => w.surah === anchorWord.surah && w.ayah === anchorWord.ayah && w.word === anchorWord.word) : -1;
      // A lookalike of the anchor plus one inserted word has not earned the move. Keep searching.
      if (!(release && anchorWi >= 0 && !relocationCommits(a.lock, anchorWi))) {
        if (memorization && afterBasmala) {
          const first = index.words[firstWord(a.lock)];
          if (!(first.surah === afterBasmala.from.surah + 1 && first.ayah === 1)) frozen ??= steady;
        }
        afterBasmala = null;
        const released = release ? [lostEvent(a.event.timeMs, (from) => quietTrace(from, "lost"))] : [];
        const fromWi = anchorWi >= 0 ? anchorWi : -1;
        const rebind = fromWi >= 0 ? takeJump(fromWi, a.lock) : null;
        startLocal(a.lock);
        return [...released, ...(rebind ? [rebind] : []), emit({ ...a.event, expected: expectedFor(a.event.observed[0].word) }, a.trace)];
      }
    }
    // Between surahs the last position stays on screen, unless nothing locks for a while.
    if (afterBasmala && ++afterBasmala.held < WINDOW) return [];
    // Junk that fits nowhere would otherwise favour passages with text before the recited phrase to absorb it.
    if (a.event.state === "lost") keepLast(LOOKBACK);
    if (holding) {
      const event: PositionEvent = { ...holding, timeMs: lastFrame * (frameMs ?? 0) };
      return unchanged(event) ? [] : [emit(event, (from) => quietTrace(from, "locked"))];
    }
    const event = { ...a.event, expected: expectedWhileLost() };
    return unchanged(event) ? [] : [emit(event, a.trace)];
  };

  /**
   * Phoneme by phoneme after the first lock. Local tracking emits one event per change of word or run start. When its
   * fit degrades, global search over the latest phonemes runs alongside: a distant global lock that holds for
   * `jumpPhonemes` commits a jump, and nothing fitting anywhere for long enough holds the anchor while the search
   * continues. `lost` is emitted when that search locks off the anchor, or when the session ends still there.
   */
  const follow = (evidence: PhonemeEvidence): EngineEvent[] => {
    const events: EngineEvent[] = [];
    for (const ph of evidence.phonemes) {
      const onset = heardEnd >= 0 && isPause(heardEnd, ph.start);
      observe(ph);
      // refinements.declaredMove: after a move, the first onset after a pause is judged again against the declared
      // opening and its rivals (`rejudge`); so is the onset after a pause while near rivals hold the contest open.
      if (onset && (rejudge === "after" || (openingContest?.held && declaredMove))) {
        if (openingContest?.held) openingContest = null;
        rejudge = "on";
        opening = "";
        openingFrames = [];
        onsets = [];
        onsetFits = [];
        openingEnd = -1;
      }
      // refinements.declaredStart: the reciter started another ayah that opens like the declared one: the lock on the
      // declared opening is let go, and global search finds where they are; or, when that ayah is near the passage
      // (refinements.declaredMove), the lock moves there on the speech since the onset.
      if (openingContest) {
        const contest = openingContest;
        const rival = hearContest(ph);
        if (rival !== null) {
          local = null;
          weak = false;
          pending = null;
          reacquiring = true;
          const moved = rival >= 0 ? rivalLock(rival, contest) : null;
          if (moved) {
            events.push(moved);
            continue;
          }
        }
      }
      if (rejudge === "on" && local) {
        const moved = rejudgeOnset(ph);
        if (moved) {
          events.push(moved);
          continue;
        }
      }
      if (!local) {
        const heardPhrase = hear(ph.p, null, NaN);
        if (heardPhrase) {
          forget();
          dropLast(heardPhrase.length);
          if (heardPhrase.name === "basmala") surahStart = true;
          if (heardPhrase.name === "takbir") fatihaNext = true;
          continue;
        }
        events.push(...reacquire());
        continue;
      }
      const before = local.save();
      local.step(ph.p);
      if (consonantConfirm || rivalHold) {
        const w = index.wordAt(Math.max(0, local.position().offset - 1));
        wordHeard = w === wordHeard.word ? { word: w, heard: wordHeard.heard + ph.p } : { word: w, heard: ph.p };
      }
      stepCosts.push(local.position().stepCost);
      if (stepCosts.length > CONFIRM) stepCosts.shift();
      const heardPhrase = hear(ph.p, before, local.position().stepCost);
      if (heardPhrase) {
        const from = heardCost.length - heardPhrase.length;
        const cost = heardCost.slice(from).reduce((a, b) => a + b, 0) / heardPhrase.length;
        if (cost >= PHRASE_LOCAL_COST) {
          // Not Quran text where we are: rewind the local search to before the phrase, as if it was never heard.
          if (heardPhrase.name === "basmala") {
            // A surah starts, not necessarily the next one: hand over to global search, favouring openings.
            afterBasmala = { from: last!.observed[0].word, held: 0 };
            if (memorization) steady = last!.expected;
            local = null;
            reacquiring = true;
            surahStart = true;
            forget();
            keepLast(0);
            continue;
          }
          local.restore(heardState[from]);
          wordHeard = { word: -1, heard: "" };
          if (heardPhrase.name === "takbir") fatihaNext = true;
          afterPhrase = true;
          forget();
          // The phrase is a boundary: global search restarts after it.
          keepLast(0);
          const e = localEvent([]);
          if (e) events.push(e);
          noteBegun();
          noteSaid();
          continue;
        }
      }
      const pos = local.position();
      const relocated = considerTwin(ph.p, pos);
      if (relocated) {
        events.push(relocated);
        continue;
      }
      if (!weak && pos.recentCost > WEAK_COST) {
        weak = true;
        steady = expectedHistory[0] ?? last!.expected;
      } else if (weak && pos.recentCost <= WEAK_COST) {
        weak = false;
        pending = null;
        afterPhrase = false;
      }
      let shortlist: TraceCandidate[] = [];
      if (!weak) keepLast(LOOKBACK);
      else {
        keepLast(WINDOW);
        const a = assessOutsidePhrases();
        if (options.trace) shortlist = a.trace(null).shortlist;
        if (a.lock && !near(a.lock, pos.offset)) {
          pending = { m: a.lock, held: pending && sameLocation(pending.m, a.lock) ? pending.held + 1 : 1 };
        } else pending = null;
        if (pending && pending.held >= (afterPhrase ? 1 : jumpPhonemes) && relocationCommits(pending.m, index.wordAt(Math.max(0, pos.offset - 1)))) {
          if (memorization) frozen ??= steady;
          const fromWi = index.wordAt(Math.max(0, pos.offset - 1));
          const rebind = takeJump(fromWi, pending.m);
          if (rebind) events.push(rebind);
          startLocal(pending.m);
        } else if (!pending && a.event.state === "lost" && pos.recentCost > maxCost && observed.length >= LOST_AFTER) {
          // Nothing has committed elsewhere. Keep this anchor until the search locks off it, or the session ends.
          events.push(beginHold());
          continue;
        }
      }
      const e = localEvent(shortlist);
      if (e) events.push(e);
      noteBegun();
      noteSaid();
      expectedHistory.push(last!.expected);
      if (expectedHistory.length > WEAK_LAG) expectedHistory.shift();
    }
    return events;
  };

  /**
   * Before a new chunk: drop all but `MAX_HISTORY` phonemes of earlier chunks, or all but `LOOKBACK` after `lost`, and
   * the posterior rows before what's kept. Done here rather than after evaluating, so `end` sees what the last push saw.
   */
  const trimHistory = () => {
    if (phraseCut) {
      if (!everLocked) onsetCut += phraseCut;
      keepLast(observed.length - phraseCut);
      phraseCut = 0;
    }
    if (basmalaHeard) surahStart = true;
    if (takbirHeard) fatihaNext = true;
    const keep = last?.state === "lost" ? LOOKBACK : MAX_HISTORY;
    if (observed.length <= keep) return;
    keepLast(keep);
    if (observedFrames.length) posteriors?.dropBefore(observedFrames[0].start);
  };

  /**
   * Leading junk (a takbir, a cough) costs every candidate the same insertions, which can hold a stream in
   * `acquiring` long after the recitation itself is clear. So when the history doesn't lock, try its latest phonemes.
   */
  const evaluateSuffix = (fresh: number): PositionEvent[] => {
    const keep = Math.max(WINDOW, fresh);
    if (observed.length <= keep) return [];
    const saved = { observed, observedConf, frames: [...observedFrames] };
    keepLast(keep);
    const a = assessOutsidePhrases();
    if (a.lock) {
      lockedOn = a.lock;
      if (!everLocked && longestSpanAt(allSpans, lastWord(a.lock)) !== null) arm = true;
      return [emit(a.event, a.trace)];
    }
    observed = saved.observed;
    observedConf = saved.observedConf;
    observedFrames.splice(0, observedFrames.length, ...saved.frames);
    return [];
  };

  /** One acquisition step: a push's phonemes, or one `acquireMs` cell's, searched globally. */
  const acquire = (phonemes: DecodedPhoneme[]): EngineEvent[] => {
    trimHistory();
    for (const ph of phonemes) {
      observe(ph);
      const heardPhrase = hear(ph.p, null, NaN);
      if (heardPhrase) {
        forget();
        phraseCut = observed.length;
        basmalaHeard ||= heardPhrase.name === "basmala";
        takbirHeard ||= heardPhrase.name === "takbir";
      }
    }
    if (pushes++ > 0 && phraseList.length) collapseRepeat();
    const events: EngineEvent[] = evaluate(false);
    if (!lockedOn) events.push(...evaluateSuffix(phonemes.length));
    if (lockedOn) {
      lockEvent = events.at(-1) as PositionEvent;
      startLocal(lockedOn);
      showLock();
      posteriors = null;
    }
    return events;
  };

  const endLocal = (): PositionEvent[] => {
    const timeMs = lastFrame * (frameMs ?? 0);
    // The search never committed. The anchor was held; give it up now.
    if (holding) {
      holding = null;
      return [lostEvent(timeMs, (from) => quietTrace(from, "lost"))];
    }
    // refinements.showSaid: the take's end settles the word the reciter stopped on.
    noteSaid(true);
    // Nothing followed the lock: its event, with the global search's trace, is still the answer.
    if (local && !emittedSinceLock) {
      const e = { ...settled(lockEvent!), timeMs };
      const out = e.trace ? { ...e, trace: { ...e.trace, transition: { from: "locked" as const, to: "locked" as const } } } : e;
      showRepeat(out);
      return [out];
    }
    const final = { ...settled(last!), timeMs };
    const out = options.trace ? { ...final, trace: quietTrace(final.state, final.state) } : final;
    showRepeat(out);
    return [out];
  };

  /** Phonemes of the current acquisition cell, held until a phoneme from a later cell shows the cell is complete. */
  let cell: DecodedPhoneme[] = [];
  let cellAt = 0;

  /** refinements.declaredStart: phoneme offset of the declared passage's first word, or -1 when the rule does not apply. */
  const declaredOffset = (() => {
    const p = options.passage?.from;
    if (!declaredStart || !p) return -1;
    const w = index.words.find((x) => x.surah === p.surah && x.ayah === p.ayah && x.word === p.word);
    return w && w.end > w.start ? w.start : -1;
  })();
  /** Phrases a reciter may say before the passage; a passage that opens al-Fatiha opens with the basmala itself. */
  const openingPhrases = declaredOffset === fatiha[0].start ? phraseList.filter((ph) => ph.name !== "basmala") : phraseList;
  /** The longest of those phrases: a finished phrase can set at most this much speech aside. */
  const longestPhrase = Math.max(0, ...openingPhrases.map((ph) => ph.p.length));
  /** refinements.declaredStart: the opening of the declared passage's first ayah, which is no rival of its own. */
  const declaredAyah = (() => {
    const p = options.passage?.from;
    if (declaredOffset < 0 || !p) return -1;
    return index.words.find((x) => x.surah === p.surah && x.ayah === p.ayah && x.end > x.start)?.start ?? -1;
  })();
  /** refinements.declaredStart: the declared opening's first spoken word, or -1. */
  const declaredFirst = declaredOffset < 0 ? -1 : index.wordAt(declaredOffset);
  /**
   * refinements.declaredStart: the phonemes the declared opening locks on (`DECLARED_MIN`, or its whole first word when
   * that is shorter, never fewer than `CONFIRM`).
   */
  const declaredFloor = declaredFirst < 0 ? DECLARED_MIN
    : Math.max(CONFIRM, Math.min(DECLARED_MIN, index.words[declaredFirst].end - index.words[declaredFirst].start));
  /**
   * refinements.declaredStart: after a lock on the declared opening while rivals are in play (`contestLimit`), the fits
   * and the speech since its onset, grown with each phoneme heard; null otherwise. While it lasts, the declared
   * opening's words show only as far as every rival in play shares them (`shows`). It ends when no rival is in play
   * (the declared opening was said), when the speech has fitted both as far as a fit reads (`FIT_REACH`: the same
   * text, and the order the reciter was asked for decides), or when another lock is made. When the speech no longer
   * fits the declared opening as a landing must, or a rival fits it better by an edit, the reciter started that other
   * ayah: the lock is let go and the tracker's search finds them, as after any wrong lock.
   */
  let openingContest: OpeningContest | null = null;
  /**
   * refinements.declaredMove: after a move, `"after"` until the first onset after a pause, `"on"` while that onset is
   * judged again (`rejudgeOnset`); null otherwise.
   */
  let rejudge: "after" | "on" | null = null;
  /** refinements.declaredMove: the run the last move started, from its first word to the word it locked on. */
  let moved = { start: -1, word: -1 };
  /**
   * refinements.declaredStart: the take's opening contest hears phoneme `ph`. Null while the lock holds; when it is to be
   * let go, the offset of the rival opening the reciter is saying, if the lock is to move there (`declaredMove`: a
   * rival near the passage, `nearDeclared`), else -1: global search finds them.
   */
  const hearContest = (ph: DecodedPhoneme): number | null => {
    const c = openingContest!;
    c.text += ph.p;
    for (let i = 0; i < ph.p.length; i++) c.frames.push({ start: ph.start, end: ph.end, conf: ph.conf });
    const n = c.text.length;
    const grow: Grow = (fit, bar = Infinity) => {
      for (let cost = fit.cost(); fit.heard < n; cost = fit.cost()) {
        if (cost === null || cost >= bar) return cost;
        fit.push(c.text[fit.heard]);
      }
      return fit.cost();
    };
    const cost = grow(c.fits.declared);
    const fails = cost === null || cost / UNIT / n > CONFIRM_COST;
    // With declaredMove, a declared opening that no longer fits still yields to a near rival that fits as a landing must.
    const rival = !fails ? rivalAhead(c.fits, grow, cost, c.text)
      : declaredMove ? rivalAhead(c.fits, grow, Math.min(cost ?? Infinity, CONFIRM_COST * UNIT * n + minSeparation * UNIT), c.text) : -1;
    // Near rivals that fit alike: the contest stays open, the declared words shown as far as they all share them.
    c.held = rival === HELD;
    if (c.held) return null;
    if (fails || rival >= 0) {
      openingContest = null;
      return declaredMove && rival >= 0 && nearDeclared(rival) ? rival : -1;
    }
    const shared = n >= FIT_REACH ? Infinity : contestLimit(c.fits, grow, cost, c.text, -1);
    if (shared === Infinity) openingContest = null;
    else c.shows = Math.max(c.shows, declaredWord(shared - 1));
    return null;
  };
  /** The reference span a fit of an opening at `offset` reads (as `fitAt`). */
  const reachEnd = (offset: number) => Math.min(index.phonemes.length, offset + FIT_REACH);
  /**
   * refinements.declaredStart: a word as written, without the marks the mushaf's orthography adds to it (the tatweel,
   * the madd sign, the small high and low letters and the pause signs), with the dagger alif written as an alif: so
   * تَبَٰرَكَ and تَبَارَكَ, or ٱلَّذِىٓ before a hamza and ٱلَّذِى, are one word. An index without the text compares
   * phonemes alone.
   */
  const writtenAs = (wi: number) => index.wordText?.[wi]?.replace(/[\u0640\u0653\u06D6-\u06ED]/g, "").replace(/\u0670/g, "\u0627");
  const soundsOf = (wi: number) => index.phonemes.slice(index.words[wi].start, index.words[wi].end);
  /** refinements.declaredStart: how many of the declared opening's first words each rival opening shares, by offset. */
  const sharedWords = new Map<number, number>();
  /**
   * refinements.declaredStart: how many of the declared opening's first spoken words the ayah opening at `offset`
   * shares with it, as far as a fit reads (`FIT_REACH`): word for word, the same word as written (`writtenAs`) or the
   * same sounds; or the same sounds split into other words, ending together at a word's end (وَإِن مَّا and وَإِمَّا).
   * 0 when their first words differ.
   */
  const sharedWith = (offset: number): number => {
    let n = sharedWords.get(offset);
    if (n !== undefined) return n;
    const reach = reachEnd(declaredOffset);
    n = 0;
    for (let a = declaredFirst, b = index.wordAt(offset); a >= 0 && b >= 0 && index.words[a].start < reach; a = spokenStep(a, 1), b = spokenStep(b, 1)) {
      const text = writtenAs(a);
      if ((text === undefined || text !== writtenAs(b)) && soundsOf(a) !== soundsOf(b)) break;
      n++;
    }
    let ours = "";
    let theirs = "";
    for (let a = declaredFirst, b = index.wordAt(offset), k = 0; a >= 0 && index.words[a].start < reach;) {
      if (ours.length <= theirs.length) {
        ours += soundsOf(a);
        a = spokenStep(a, 1);
        k++;
      } else if (b >= 0) {
        theirs += soundsOf(b);
        b = spokenStep(b, 1);
      } else break;
      const length = Math.min(ours.length, theirs.length);
      if (ours.slice(0, length) !== theirs.slice(0, length)) break;
      if (ours.length === theirs.length) n = Math.max(n, k);
    }
    sharedWords.set(offset, n);
    return n;
  };
  /** refinements.declaredStart: the declared opening's spoken word `k` words after its first (its first for k <= 0). */
  const declaredWord = (k: number) => {
    let wi = declaredFirst;
    for (let i = 0; i < k && spokenStep(wi, 1) >= 0; i++) wi = spokenStep(wi, 1);
    return wi;
  };
  /**
   * refinements.declaredStart: the fewest of the declared opening's words that a rival in play shares with it
   * (`sharedWith`), or Infinity when no rival is in play. A rival is in play while its fit of an onset's speech `text`
   * is less than `DISTINCT` worse than the declared opening's fit (cost `cost`), or better. Every ayah's opening in the
   * mushaf is a rival (`ayahOpenings`): `fitBounds` finds the ones whose fit of the speech may come within the bar,
   * cheapest first, and each that shares fewer words than the fewest so far is then fitted as the declared opening is.
   * It stops at the first one in play that shares `floor` words or fewer. With `declaredLeadIn`, every word of the
   * lead-in tail inside its ayah (`leadStarts`) is a rival too, fitted from its first phoneme as an opening is, that
   * shares no word: the declared ayah is the one next in order only to an ayah that opens alike, while a reciter who
   * leads in says the tail's words first.
   *
   * Half an edit, not the commit margin global search needs for its own lock (30/n edits, six at the five phonemes a
   * declared lock needs at least): the passage is declared, so its opening needs only to fit the speech better than any
   * other ayah the reciter may be starting at, not to stand out from the whole Qur'an. A fit's cost only grows as it
   * hears more, so a rival whose fit of part of the speech already reaches the bar is grown no further.
   */
  const contestLimit = (fits: OnsetFits, grow: Grow, cost: number, text: string, floor: number): number => {
    const bar = cost + DISTINCT;
    let fewest = Infinity;
    const inPlay = (start: number, shared: number) => {
      if (start === declaredAyah || shared >= fewest) return false;
      const rival = grow(rivalFit(fits, start), bar);
      if (rival === null || rival >= bar) return false;
      fewest = shared;
      return fewest <= floor;
    };
    for (const { start } of fitBounds(index.phonemes, ayahOpenings(index), reachEnd, text, bar)) if (inPlay(start, sharedWith(start))) return fewest;
    for (const start of leadStarts) if (inPlay(start, 0)) return fewest;
    return fewest;
  };
  /** refinements.declaredStart: an onset's fit of the rival opening at `start`, made the first time it is sought. */
  const rivalFit = (fits: OnsetFits, start: number) => {
    let fit = fits.rivals.get(start);
    if (!fit) fits.rivals.set(start, (fit = growingFit(index.phonemes, start, reachEnd(start))));
    return fit;
  };
  /**
   * refinements.declaredStart: the lead-in tail: the declared ayah's spoken words before the declared word, and every
   * spoken word of the `LEAD_AYAHS` ayahs before it, as word indices (`from` to the declared word), or null. A reciter
   * may begin a little before the passage, reading into it.
   */
  const leadTail = (() => {
    if (declaredFirst < 0) return null;
    const d = index.words[declaredFirst];
    let from = declaredFirst;
    let ayahs = 0;
    let ayah = `${d.surah}:${d.ayah}`;
    for (let wi = declaredFirst - 1; wi >= 0; wi--) {
      const w = index.words[wi];
      if (w.end <= w.start) continue;
      const at = `${w.surah}:${w.ayah}`;
      if (at !== ayah && ++ayahs > LEAD_AYAHS) break;
      ayah = at;
      from = wi;
    }
    return { from, to: declaredFirst };
  })();
  /**
   * refinements.declaredLeadIn: the first phoneme of every word of the lead-in tail (`leadTail`) inside its ayah: rivals
   * of the declared opening besides the ayah openings, which are rivals already.
   */
  const leadStarts: number[] = [];
  if (declaredLeadIn && leadTail) {
    for (let wi = leadTail.from; wi < leadTail.to; wi++) if (index.words[wi].end > index.words[wi].start && !opensAyah(wi)) leadStarts.push(index.words[wi].start);
  }
  /**
   * refinements.declaredMove: the last word of the passage's surroundings: `NEAR_AYAHS` ayahs past the passage's last
   * ayah, in its surah.
   */
  const nearEnd = (() => {
    const to = options.passage?.to;
    if (!declaredMove || !to) return -1;
    let end = index.words.findIndex((w) => w.surah === to.surah && w.ayah === to.ayah && w.word === to.word);
    if (end < 0) return -1;
    while (end + 1 < index.words.length && index.words[end + 1].surah === to.surah && index.words[end + 1].ayah <= to.ayah + NEAR_AYAHS) end++;
    return end;
  })();
  /**
   * refinements.declaredMove: the rival opening at phoneme offset `offset` is near the passage: from the lead-in tail
   * (`leadTail`) to `NEAR_AYAHS` ayahs past the passage's end (`nearEnd`).
   */
  const nearDeclared = (offset: number) => {
    const wi = index.wordAt(offset);
    return !!leadTail && wi >= leadTail.from && wi <= nearEnd;
  };
  /**
   * refinements.declaredStart: another ayah's opening (or, with `declaredLeadIn`, a word of the lead-in tail) fits the
   * onset's speech `text` better than the declared opening's fit (`cost`) by at least an edit: the reciter is saying
   * that ayah. Gives its offset, or -1 for none. With `declaredMove`, a rival near the passage (`nearDeclared`) when
   * there is one, once it stands apart: the one near rival within `DISTINCT` of the best. While two or more fit alike,
   * the speech has not told them apart, and the order of the mushaf does not tell which one the reciter is saying (a
   * lead-in from before the passage, or a later start, opens alike): `HELD`, the contest stays open. Past what a fit
   * reads (`FIT_REACH`), the last before the declared ayah, else the first after it.
   */
  const rivalAhead = (fits: OnsetFits, grow: Grow, cost: number, text: string): number => {
    const bar = cost - minSeparation * UNIT;
    if (bar <= 0) return -1;
    let found = -1;
    const near: { start: number; cost: number }[] = [];
    const ahead = (start: number) => {
      if (start === declaredAyah) return false;
      const rival = grow(rivalFit(fits, start), bar);
      if (rival === null || rival >= bar) return false;
      if (!declaredMove) return true;
      if (found < 0) found = start;
      if (nearDeclared(start)) near.push({ start, cost: rival });
      return false;
    };
    for (const { start } of fitBounds(index.phonemes, ayahOpenings(index), reachEnd, text, bar)) if (ahead(start)) return start;
    for (const start of leadStarts) if (ahead(start)) return start;
    if (!near.length) return found;
    const best = Math.min(...near.map((r) => r.cost));
    const alike = near.filter((r) => r.cost < best + DISTINCT).map((r) => r.start);
    if (alike.length === 1) return alike[0];
    if (text.length < FIT_REACH) return HELD;
    const before = alike.filter((start) => start < declaredOffset);
    return before.length ? Math.max(...before) : Math.min(...alike);
  };
  /**
   * refinements.declaredStart: phonemes heard before the first lock, from the oldest onset that can still fit the opening,
   * with their frames; and where each onset (the take's first phoneme, or the first after a pause) begins in them.
   */
  let opening = "";
  let openingFrames: { start: number; end: number; conf: number }[] = [];
  let onsets: number[] = [];
  /**
   * refinements.declaredStart: for each onset, the fits its speech (leading phrases set aside, from `cut`) grows a phoneme at
   * a time: the declared opening, each phrase, and each rival opening tried so far. They start over when the leading
   * phrases set aside change.
   */
  /**
   * Grows a fit with the onset's speech and gives its cost, stopping early once the cost reaches `bar`: the cost of
   * all the speech is then at least that.
   */
  type Grow = (fit: ReturnType<typeof growingFit>, bar?: number) => number | null;
  interface OnsetFits {
    cut: number;
    declared: ReturnType<typeof growingFit>;
    phrases: ReturnType<typeof growingFit>[];
    rivals: Map<number, ReturnType<typeof growingFit>>;
    /** The speech before the next onset is not the opening (decided once that onset exists). */
    other?: boolean;
  }
  let onsetFits: (OnsetFits | undefined)[] = [];
  /** refinements.declaredStart: an onset's fits and its speech, while rivals are in play. */
  interface OpeningContest {
    fits: OnsetFits;
    text: string;
    /** The frames each phoneme of `text` was heard in, with its confidence. */
    frames: { start: number; end: number; conf: number }[];
    /** The last of the declared opening's words that may show: as far as every rival in play shares them. */
    shows: number;
    /** The words the page showed since the lock (refinements.declaredMove): a move withdraws only these. */
    lit: Set<number>;
    /** Near rivals fit the speech alike (refinements.declaredMove): the contest is held open. */
    held: boolean;
  }
  /** The end frame of the latest phoneme heard before the first lock, for the pause test. */
  let openingEnd = -1;
  /** Fits the declared opening from its first phoneme, strictly, with the speech `text`. */
  const fitAt = (offset: number, text: string) =>
    alignSubstring(index.phonemes, offset, reachEnd(offset), text, (o) => o === offset, { strictStart: true });
  /**
   * refinements.declaredStart: the declared opening, once the speech from some onset (leading phrases set aside) fits it from
   * its first phoneme as a landing must (`CONFIRM_COST` per phoneme), no phrase still being said fits it as well, and
   * every rival opening in play shares at least its first word (`contestLimit`). Onsets are tried oldest first: the
   * most speech that fits.
   */
  const declaredLock = (): { m: Match; n: number; base: number; contest: OpeningContest | null } | null => {
    for (let k = 0; k < onsets.length; k++) {
      const said = opening.slice(onsets[k]);
      const cut = leadingPhrases(said, openingPhrases);
      const text = said.slice(cut);
      const n = text.length;
      if (n < declaredFloor || n > WINDOW) continue;
      // The onset's speech only grows while the leading phrases set aside stay the same.
      let fits = onsetFits[k];
      if (!fits || fits.cut !== cut) {
        fits = onsetFits[k] = {
          cut, rivals: new Map(),
          declared: growingFit(index.phonemes, declaredOffset, reachEnd(declaredOffset)),
          phrases: openingPhrases.map((phrase) => growingFit(phrase.p, 0, phrase.p.length)),
        };
      }
      const grow: Grow = (fit, bar = Infinity) => {
        for (let c = fit.cost(); fit.heard < n; c = fit.cost()) {
          if (c === null || c >= bar) return c;
          fit.push(text[fit.heard]);
        }
        return fit.cost();
      };
      // Speech said before a pause that does not fit the opening as a landing must (`DECLARED_MIN` phonemes or more) was
      // not the opening: if the reciter has begun it, they began after the pause, at an onset tried in its turn. Fitted
      // with it, that speech would count as insertions, and every rival within an edit of so poor a fit would be sought.
      if (k + 1 < onsets.length && fits.other === undefined) {
        const before = text.slice(0, Math.max(0, onsets[k + 1] - onsets[k] - cut));
        const lead = growingFit(index.phonemes, declaredOffset, reachEnd(declaredOffset));
        for (const p of before) lead.push(p);
        const c = lead.cost();
        fits.other = before.length >= DECLARED_MIN && (c === null || c / UNIT / before.length > CONFIRM_COST);
      }
      if (fits.other) continue;
      const cost = grow(fits.declared);
      if (cost === null || cost / UNIT / n > CONFIRM_COST) continue;
      if (fits.phrases.some((fit) => {
        const b = grow(fit);
        return b !== null && b <= cost;
      })) continue;
      // The whole alignment, for the path a lock reports: only once the speech is about to lock.
      const a = fitAt(declaredOffset, text);
      if (!a || a.start !== declaredOffset || a.per > maxPer) continue;
      // A rival in play whose first word differs: the first word is not yet told apart.
      const shared = contestLimit(fits, grow, cost, text, 0);
      if (shared === 0) continue;
      const contest = shared === Infinity ? null
        : { fits, text, frames: openingFrames.slice(onsets[k] + cut), shows: declaredWord(shared - 1), lit: new Set<number>(), held: false };
      return { m: { offset: a.start, length: a.end - a.start, cost: a.cost, per: a.per, votes: 0, path: a.path }, n, base: onsets[k] + cut, contest };
    }
    return null;
  };
  /**
   * refinements.declaredStart: one phoneme heard before the first lock. A pause before it opens a new onset. Onsets whose
   * speech (leading phrases set aside) can no longer fit within a window are dropped, with the speech before the
   * oldest one left, so the search before the first lock stays bounded however long the reciter speaks elsewhere.
   */
  const hearOpening = (ph: DecodedPhoneme) => {
    if (openingEnd < 0 || isPause(openingEnd, ph.start)) {
      onsets.push(opening.length);
      onsetFits.push(undefined);
    }
    openingEnd = ph.end;
    if (!onsets.length) return;
    opening += ph.p;
    for (let i = 0; i < ph.p.length; i++) openingFrames.push({ start: ph.start, end: ph.end, conf: ph.conf });
    while (onsets.length && opening.length - onsets[0] > WINDOW + longestPhrase) {
      onsets.shift();
      onsetFits.shift();
    }
    const keep = onsets[0] ?? opening.length;
    if (keep > 0) {
      opening = opening.slice(keep);
      openingFrames = openingFrames.slice(keep);
      onsets = onsets.map((o) => o - keep);
    }
  };
  /** refinements.declaredStart: lock on the declared opening, as an acquisition lock would, and hand over to local search. */
  const openingLock = ({ m, n, base, contest }: { m: Match; n: number; base: number; contest: OpeningContest | null }, midTake = false): PositionEvent => {
    const pageFrom = shownOut.length;
    const heardAt: HeardAt = (j) => openingFrames[base + j];
    let word = lockWord(m, heardAt);
    // The phoneme that told the declared opening apart may be the first of its next word: the lock shows the word that
    // was heard, and the next one shows as the local search hears it, so each shows as it is said.
    if (word > declaredFirst && word === lastWord(m) && lockEntry(m, heardAt).matches < ADVANCE_MATCHES && spokenStep(word, -1) >= firstWord(m)) {
      word = spokenStep(word, -1);
    }
    const shown = contest ? Math.min(word, contest.shows) : word;
    const end = ref(shown);
    const cost = m.cost / UNIT / n;
    const confidence = Math.max(0, 1 - m.per / maxPer) * Math.max(0, 1 - cost / maxCost);
    const event: PositionEvent = {
      type: "position",
      v: EVENT_SCHEMA_VERSION,
      timeMs: lastFrame * (frameMs ?? 0),
      referenceVersion: index.version,
      expected: expectedFor(end),
      observed: [{ word: end, span: { start: ref(firstWord(m)), end }, confidence, cost, per: m.per }],
      state: "locked",
    };
    if (longestSpanAt(allSpans, lastWord(m)) !== null) arm = true;
    const out = emit(event, (from) => quietTrace(from, "locked"));
    lockEvent = out;
    startLocal(m, heardAt);
    showLock();
    openingContest = contest;
    // What the lock put on the page, sent before the contest was the tracker's.
    if (contest) for (const e of showSaid ? shownOut.slice(pageFrom) : [out]) pageLit(contest, e);
    if (!midTake) {
      // The opening's phonemes never went through `observe`: the next phoneme's pause test is against the last of them.
      heardEnd = openingEnd;
      posteriors = null;
    }
    cell = [];
    opening = "";
    openingFrames = [];
    onsets = [];
    onsetFits = [];
    return out;
  };
  /**
   * refinements.declaredMove: the reciter is saying the rival opening at `offset`, near the passage, that the take's
   * opening `contest` let go for: lock there on the speech heard since the contest's onset, as `openingLock` locks the
   * declared opening, and hand over to local search. The lock names in `withdrawn` the declared words shown since the
   * declared lock that its run does not cover. Null when that speech does not fit the rival as a lock must: global
   * search finds the reciter, as after any wrong lock. The move withdraws only declared words the page showed, and the
   * first onset after a pause is judged again (`rejudgeOnset`).
   */
  const rivalLock = (offset: number, contest: OpeningContest): PositionEvent | null => {
    const { text, frames } = contest;
    const a = fitAt(offset, text);
    if (!a || a.start !== offset || a.per > maxPer) return null;
    const m: Match = { offset: a.start, length: a.end - a.start, cost: a.cost, per: a.per, votes: 0, path: a.path };
    const heardAt: HeardAt = (j) => frames[j];
    const word = lockWord(m, heardAt);
    const end = ref(word);
    const cost = m.cost / UNIT / text.length;
    const confidence = Math.max(0, 1 - m.per / maxPer) * Math.max(0, 1 - cost / maxCost);
    const withdrawn: WordRef[] = [];
    for (const wi of [...contest.lit].sort((a, b) => a - b)) {
      if (wi >= firstWord(m) && wi <= word) continue;
      litBy.delete(wi);
      withdrawn.push(ref(wi));
    }
    const event: PositionEvent = {
      type: "position",
      v: EVENT_SCHEMA_VERSION,
      timeMs: lastFrame * (frameMs ?? 0),
      referenceVersion: index.version,
      expected: expectedFor(end),
      observed: [{ word: end, span: { start: ref(firstWord(m)), end }, confidence, cost, per: m.per }],
      state: "locked",
      ...(withdrawn.length && { withdrawn }),
    };
    const out = emit(event, (from) => quietTrace(from, "locked"));
    lockEvent = out;
    // The speech heard is the rival's: none of it is a phrase to set aside from the run that reads it.
    forget();
    startLocal(m, heardAt);
    showLock();
    rejudge = "after";
    moved = { start: firstWord(m), word };
    return out;
  };
  /**
   * refinements.declaredMove: a move is made on one onset's speech, so the first onset after a pause that follows it is
   * judged again against the declared opening and its rivals, as the take's first is (`hearOpening`, `declaredLock`),
   * while the local search goes on from the move. When the declared opening locks on that onset, the run shown does not
   * already cover its first word, and the run the move started has not read on in order into that speech (an ayah the
   * same as the declared one may follow the moved-to one: 109:5 after 109:4, with 109:3 declared), the lock goes there
   * (`openingLock`): the reciter's run lights the declared ayah from its first word. The look ends with that lock, with
   * any other, or once the onset's speech is longer than a lock reads (`WINDOW`).
   */
  const rejudgeOnset = (ph: DecodedPhoneme): PositionEvent | null => {
    hearOpening(ph);
    const lock = declaredLock();
    if (!lock && opening.length <= WINDOW) return null;
    rejudge = null;
    const shown = last?.state === "locked" ? last.observed[0] : undefined;
    const covers = !!shown && wordOf(shown.span.start) <= declaredFirst && wordOf(shown.word) >= declaredFirst;
    const readOn = !!shown && wordOf(shown.span.start) === moved.start && wordOf(shown.word) > moved.word;
    if (lock && !covers && !readOn) {
      local = null;
      weak = false;
      pending = null;
      return openingLock(lock, true);
    }
    opening = "";
    openingFrames = [];
    onsets = [];
    onsetFits = [];
    return null;
  };
  /** refinements.declaredMove: the words a locked page `event` shows join the contest's (`OpeningContest.lit`). */
  function pageLit(contest: OpeningContest, event: PositionEvent) {
    const c = event.state === "locked" ? event.observed[0] : undefined;
    if (!c) return;
    for (let wi = wordOf(c.span.start); wi >= 0 && wi <= wordOf(c.word); wi = spokenStep(wi, 1)) contest.lit.add(wi);
  }

  return {
    push(evidence) {
      shownOut = [];
      const out = pushPhonemes(evidence);
      if (!showSaid) shownOut = positionsOf(out);
      // refinements.showSaid: a pause settles the word the reciter stopped on.
      settlePause(evidence.decodedTo);
      return out;
    },
    end() {
      shownOut = [];
      const out = endTake();
      if (!showSaid) shownOut = positionsOf(out);
      return out;
    },
    shown() {
      const out = shownOut;
      shownOut = [];
      return out;
    },
    silence(decodedTo) {
      shownOut = [];
      settlePause(decodedTo);
      const out = shownOut;
      shownOut = [];
      return out;
    },
  };

  function pushPhonemes(evidence: PhonemeEvidence): EngineEvent[] {
    validate(evidence);
    frameMs = evidence.frameMs;
    if (evidence.posteriors) withPosteriors = true;
    else decodedOnly = true;
    if (local || reacquiring) return follow(evidence);
    if (evidence.posteriors) (posteriors ??= new PosteriorFrames(index.tokens.size)).append(evidence.posteriors);
    const opens = declaredOffset >= 0 && !everLocked;
    const events: EngineEvent[] = [];
    const phs = evidence.phonemes;
    // Phonemes in time order: the acquisition cells a phoneme closes are searched before the declared opening hears
    // it, so the events don't depend on how the caller chunks evidence.
    for (let i = 0; i < phs.length; i++) {
      if (options.acquireMs) {
        const at = Math.floor((phs[i].start * frameMs) / options.acquireMs);
        while (at > cellAt && !local) {
          events.push(...acquire(cell));
          cell = [];
          cellAt++;
        }
        if (local) return [...events, ...follow({ ...evidence, phonemes: phs.slice(i), posteriors: undefined })];
      }
      if (opens && !everLocked) {
        hearOpening(phs[i]);
        const lock = declaredLock();
        if (lock) {
          lastFrame = Math.max(lastFrame, phs[i].end + 1);
          const event = openingLock(lock);
          return [...events, event, ...follow({ ...evidence, phonemes: phs.slice(i + 1), posteriors: undefined })];
        }
      }
      if (options.acquireMs) cell.push(phs[i]);
    }
    if (!options.acquireMs) return acquire(evidence.phonemes);
    const complete = evidence.decodedTo === undefined ? 0 : Math.floor((evidence.decodedTo * frameMs) / options.acquireMs);
    while (complete > cellAt && !local) {
      events.push(...acquire(cell));
      cell = [];
      cellAt++;
    }
    return events;
  }

  function endTake(): EngineEvent[] {
    if (!local && !reacquiring) {
      const events = cell.length ? acquire(cell) : [];
      cell = [];
      return [...events, ...(local ? endLocal() : evaluate(true))];
    }
    return endLocal();
  }
}
