import { align as editPath } from "./align.js";
import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { CheckerEvent, CheckerState, EngineEvent, MemorizationErrorClass, PositionEvent, WordRef } from "./events.js";
import type { DecodedPhoneme, PhonemeEvidence } from "./evidence.js";
import { PosteriorFrames } from "./ctc.js";
import { CONSONANTS, UNIT, alignSubstring, collapseHeld, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import { channelFromCounts, spanLogP, type Channel, type ChannelCounts } from "./noisy-channel.js";
import { LISTENER_CHANNEL } from "./listener-channel.js";
import { insertionDropFit, issueDropFit } from "./tracker.js";
import type { PhraseAsset, ReferenceIndex } from "./reference.js";
import { lexicalWords } from "./checker-lexical.js";
import { swapTailUnion } from "./checker-swap-tail.js";
import { minGapFrames } from "./gap-decode.js";
import { procliticOnset, waqfMarked } from "./mushaf-marks.js";
import { absenceMargin } from "./ayah-ledger.js";
import { excisedNeighbourMargins, surplusNasal } from "./seam-excision.js";
import { PHRASE_PEAK, PHRASE_SWITCH, letterSkeleton, parallelPaths } from "./parallel-phrases.js";
import { isMaddLetter, maddSites, stemConsonant } from "./madd-drops.js";
import { formulaSlots } from "./formula-slots.js";
import { spanId, slotSpans } from "./slot-spans.js";

export interface CheckerPolicy {
  /** When a suspected wrong or added word is confirmed: after `ms` (2–3 s), or once the next word has begun. Default 2500 ms. */
  confirm?: { after: "delay"; ms: number } | { after: "next-word" };
  /** Whether resolved errors (self-corrections, recovered skips) count as mistakes. Default false. */
  countResolved?: boolean;
  /** Mean decoder confidence over an error's phonemes below which it ends `uncertain`, never `confirmed`. Default 0.6. */
  minConfidence?: number;
  /**
   * Before confirming a wrong, omitted or added word, compare how well the chunk posteriors around it (the word and one
   * either side) support the written words versus what was decoded, over all CTC timings. If the written words fit
   * nearly as well, the error ends `uncertain`. Default on; it needs posteriors on every chunk, and without them nothing
   * changes. `nats` overrides the margin the decode must win by (`SUPPORT`, doubled for a wrong word heard `UNSURE`).
   */
  localWindow?: boolean | { nats: number };
  /**
   * Free recitation, while latched: at session end, confirm a suspected wrong or added word once a later word was
   * recited right, as a declared session does. Default true.
   */
  finaliseFree?: boolean;
  /**
   * The checker's refinements of rc.1's rules ({@link CheckerRefinements}): each on unless set false.
   */
  refinements?: CheckerRefinements;
}

/**
 * Refinements of the 1.0.0-rc.1 checker's rules. Each is on unless set `false`; off, a field keeps the rc.1 rule it
 * replaces, so its effect can be measured alone.
 */
export interface CheckerRefinements {
  /**
   * A repeat that rejoins is not a skip. A word counts as read once a visit heard it right and the alignment went on
   * to its next word (an ayah's last word: or went back, to repeat). An omission is not raised when every word of it
   * was read earlier in the current pass and the reading after it rejoins: it resumes where reading on from the
   * furthest word read before it goes (the next word; from an ayah's last two words, the next ayah's first word), or
   * it begins an ayah's first word after the reciter broke off (a pause, or a phrase that is not the Quran's, between
   * the speech either side of it, or no speech before it in the alignment). A pause alone excuses nothing: a
   * resumption inside an ayah stays an omission (forget, hesitate, skip), in a first reading or a re-reading, and so
   * does a resumption at an ayah's first word in continuous speech. Nor is an ayah's end a break: every ayah ends
   * with a pause, so whole ayahs left out after one stay omissions, a re-reading's as a first reading's, unless the
   * reading goes on where the reciter had read to (in the ayah that reading on from the furthest word read goes into:
   * the next one after an ayah read to its end, or the ayah they broke off in, started again).
   *
   * A pass is set by the recitation, the same way in a declared passage and in free recitation. One starts at a takbir
   * or a basmala, at a jump (off the passage, back onto it, or ahead of the window; in free recitation the passage is
   * what was read since the place was found, from its first word) or after the place was lost, and where the reciter
   * goes back to an earlier ayah `PASS_BACK_WORDS` words or more behind the furthest word read (or back to a surah's
   * first ayah) and reads the ayah they went back to through to its last word: the passage read again (the memorization
   * drill, a second rak'ah) has its own omissions, from the return on. A shorter return is a repeat inside the pass,
   * and so is a return that leaves the ayah it went back to before its last word (a few words to find the place).
   * Coming forward again to the furthest ayah read before going back (`reanchorBack`'s rejoin, or a jump in the window)
   * goes on with the pass it left. Words never read in the pass stay omissions. Risks: a reciter who goes back fewer
   * than `PASS_BACK_WORDS` words, reads an ayah again and breaks off inside it (a pause, then the next ayah's first
   * word) is not marked for the rest of that ayah, read in the first reading; one who broke off inside an ayah, went
   * back and starts that ayah again is not marked for the ayahs the run-up left out.
   */
  readNotSkipped?: boolean;
  /**
   * Follow a reciter who goes back further than the alignment window, and forward again. A tracker lock of
   * `JUMP_WORDS` words or more, inside the passage and behind the window, starts the alignment over there, as a jump
   * off the passage does, unless the window's own text holds the same words (a refrain's later copy: the window
   * follows it). Shorter locks stay ignored, so a word that sounds like an earlier one moves nothing. In free
   * recitation the passage is the stretch read so far (from the first lock on), so a reciter who starts again from an
   * earlier ayah is followed there too.
   *
   * Once the window has followed a reciter back in a declared passage, a lock of `JUMP_WORDS` words or more inside the
   * passage and past the window's far end starts the alignment over there in turn:
   * - landing on the furthest ayah read before going back, up to where reading on from the furthest word goes (the
   *   word after it, or the next ayah's first word when it was the ayah's last word or the one before), is a rejoin:
   *   nothing is marked, and the pass left goes on (`readNotSkipped`);
   * - landing further on, the words from there to the landing were never read: an omission, or a skipped passage
   *   from three words;
   * - landing short of the furthest ayah, after a return that started a new pass (`readNotSkipped`: the passage read
   *   again), the words from where that reading stood to the landing are left out of that pass; with no new pass,
   *   nothing is marked.
   * Nothing is marked on the words read right before or during the excursion. A slide into a look-alike ayah that the
   * reciter leaves on their own, coming back to where they had been, is a self-correction and marks nothing: the audio
   * and the text cannot tell it from a deliberate revision that goes back to the same place.
   *
   * What the window held open when it moves ends `uncertain`, as at a jump. Risks: a lock behind the window on words
   * the reciter did not go back to say ends the window's open suspicions `uncertain`; a slide into a look-alike ayah
   * that the reciter corrects alone is not reported.
   */
  reanchorBack?: boolean;
  /**
   * A stutter is not a wrong word: a word heard with a block of sounds said twice in a row, which with one copy
   * removed has the written word's consonants in order, is the written word restarted. Risk: it compares consonants,
   * so a restarted word whose vowels were also changed (short vowels the checker does not judge) is let through.
   */
  stutter?: boolean;
  /**
   * The ayah ledger raises no whole-ayah skip for an ayah the alignment read in part: the word-level omission covers
   * its missing words. Risk: an ayah left out whose words the alignment matched in part by chance (a phrase it shares
   * with its neighbours) is left to the word-level omissions, which can mark a shorter skip or none.
   */
  ledgerPartial?: boolean;
  /**
   * Say an omission is due once the reciter has read on past it, ahead of its confirmation: its `suspected` event is
   * sent again with `due: true` ({@link CheckerEvent.due}). In a declared passage, when it is sure, the confirmation
   * delay has passed since it was suspected, the alignment ends after it, and two words after it were heard right,
   * both begun after the suspicion. Free recitation sends none: it follows the tracker, whose place can still move to
   * the other copy of a repeated passage. Confirmation keeps rc.1's rule (two ayahs on, or the end), so `confirmed`
   * stays final: a reader who goes back and says the word before then corrected themselves, and it ends `resolved`.
   * `false` sends no `due` event.
   */
  omissionDue?: boolean;
  /**
   * A commit keeps nothing its new window cannot place. When the window grows past `MAX_WINDOW`, its older part is
   * committed at a word boundary and the latest phonemes are aligned again from there, in a window that starts an ayah
   * before the cut word. If the path kept after the cut goes back further than that (the reciter restarted an ayah or
   * more back, within the kept phonemes), the restart would be read against the words inside the window and charged to
   * them as wrong words and omissions. The cut moves instead to the first word boundary after the restart's landing
   * whose kept path stays inside its window, keeping at least `AFTER_DROP` phonemes; while there is none, the commit
   * waits, up to twice `MAX_WINDOW`, and past that cuts as before. A restart reads again the text it lands on: when the
   * first speech the path places from its landing does not fit that text as a lock's words fit a passage
   * (`FITS_PASSAGE`, judged on `REREAD_MIN` phonemes or more), the path reached behind only to place speech from
   * elsewhere (the reciter gone back further than the window, onto a refrain's earlier copy), and the cut stays. A wrong
   * word that the restart then reads right is a self-correction, and one said wrong again after the restart is still a
   * wrong word. Risk: while it waits, one alignment covers up to twice as many phonemes.
   */
  commitAfterJump?: boolean;
  /**
   * Memorization: starting later than the declared passage leaves its first ayahs out. While the alignment still runs
   * from the passage's first word (before any commit, jump or return), a first reading that begins in a later ayah of
   * the passage, with three words or more of the passage before it (a `skipped_passage`'s floor), marks those words
   * left out, as one skip. A first reading the window cannot reach (the take's first lock inside the passage, past the
   * two ayahs the window reads ahead) starts the window over there, as a jump does, with the passage's words before it
   * left out, and nothing is marked on the words between. A reading that begins inside the first ayah, or two words or
   * fewer in (a microphone that opened late, a word cut off), or outside the passage (another surah, before or after
   * it), marks no skip. Speech before the passage that is not the Quran's stays set aside, as before. Risk: a reciter
   * who means to start later than the passage they declared is marked for the ayahs before.
   */
  firstAyahOut?: boolean;
  /**
   * Memorization: no word mark outside the declared passage. The alignment window reaches past the passage's ends (an
   * ayah before the anchor, two after the furthest word read), so it can raise a mark on a word that was not asked
   * for, on a look-alike or on speech after the passage's end; such a mark is dropped. One that reaches into the
   * passage stays. A free recitation, whose passage is what was read so far, keeps its marks. Risk: a reciter who reads
   * on past the passage's end is not marked there; the tracker's lost-place prompt tells them they left it.
   */
  marksInPassage?: boolean;
  /**
   * A restart is not a wrong word. The alignment can charge a restart (the reciter going back a little to take a run at
   * a word again) to words after it whose first reading a commit has taken out of the window, as one long wrong word
   * whose heard text is the stretch read again (an ayah read whole and started again two or three times before reading
   * on, with a commit falling inside the restarts). A wrong word heard as at least `REPEAT_MIN` phonemes and twice its
   * own length, that fits the text before it (from two ayahs back, not before a declared passage) as a lock's words fit
   * a passage (`FITS_PASSAGE`), charged to words read right earlier in the current pass, where the reading after them
   * rejoins as `readNotSkipped` judges a repeat's, was that restart: nothing was said wrong and nothing left out, so it
   * is not marked. Any other wrong word stays one: words never read, or a reading that goes on inside the ayah (earlier
   * text said in place of a word, then reading on). It uses `readNotSkipped`'s record of the words read, so it acts
   * only with that field on. Risk: a reciter who reads a passage, goes back fewer than `PASS_BACK_WORDS` words, says
   * earlier text in place of words they had read right and goes on where they had read to is not marked for them.
   */
  repeatNotWrong?: boolean;
  /**
   * **Off by default.** Memorization: a mark as soon as the checker is sure of it, and none lost at the take's end. It is
   * off because a reciter who leaves the declared passage for another surah can be marked on the word where they left,
   * before the lost-place prompt that should be all they see: the other surah's opening read past that word, and its
   * next word begun, pass every test a confirmation asks but the wait. A suspected wrong or
   * added word that passes all its confirmation asks but the wait (read past, the next word begun, heard clearly, and
   * the posteriors there and favouring what was heard over the written words by the confirmation's margin, twice
   * `SUPPORT` for a change heard unclearly or spelling no word), with its evidence standing `STAND_MS` unchanged (the
   * same class, words and heard text) in place of the wait, is sent `suspected` again with `due: true`
   * ({@link CheckerEvent.due}) ahead of its confirmation, which keeps rc.1's time and confirms it on that evidence.
   * Standing filters out suspicions the next words explain; it says nothing of how clearly a word was heard, so it
   * replaces only the wait. When the session ends, a mark already due is decided on the evidence it was due on, and a
   * mistake where the reading stopped (a wrong word with the word before it recited right and nothing heard after it,
   * or an added word between two words recited right) on its evidence, `confirmed` when it passes the same test (no
   * more speech will change it, so it need not stand), instead of always `uncertain`: a slip on the last word before a
   * stop, with no word after it to wait for, is marked. Speech after the words recited right that the alignment could
   * not place on the passage (other speech, a hold, several words heard wrong in a row) ends `uncertain`, as in rc.1.
   * And a take that ends one or two words (`ENDING_LEFT`) short of the declared passage's end, inside its last ayah,
   * by the alignment and the tracker's place both, left those words out (one `omitted_word`, decided as any
   * omission); a take that ends further back stopped, and nothing after it is marked. Free recitation keeps rc.1's
   * rule: it follows the tracker, whose place can still move to the other copy of a repeated passage. Risks: a `due`
   * mark can still end `resolved` when the reading that follows explains the word after all (the next word's first
   * sound read as the end of this one), which an app shows as a mark taken down; a slip of two words or more on the
   * last words before a stop is not marked; a reciter who means to say a word again after a stop, or who stops a word
   * or two before the passage's end on purpose, is marked.
   */
  markWhenSure?: boolean;
}

export interface SeamInsertionCandidate {
  word: number;
  said: string;
  fromFrame: number;
  toFrame: number;
  seamFromFrame: number;
  seamToFrame: number;
  prefixFrame?: number;
}

export interface CheckerOptions {
  /** Auxiliary listener evidence may veto only posterior-certified seam spellings. */
  seamInsertion?: (candidate: SeamInsertionCandidate) => boolean;
  policy?: CheckerPolicy;
  /** Non-Quran phrases: an inserted block that is one of these (or a basmala) is not an added word. */
  phrases?: PhraseAsset;
  /**
   * The passage the session declared (memorization): the checker aligns against it rather than the tracker's lock, so
   * a wrong word that forms another ayah is still wrong. A tracker lock off the passage spanning `JUMP_WORDS` words is
   * a deliberate jump, followed until the tracker locks back inside the passage.
   */
  passage?: { from: WordRef; to: WordRef };
  /**
   * Listener confusion counts for closed omission placement and ayah-final formula choices. The scored
   * build uses {@link LISTENER_CHANNEL}. An empty phone list leaves the channel decisions off.
   */
  channel?: ChannelCounts;
}

export interface Checker {
  /** One evidence chunk and the events the tracker returned for it. */
  push(evidence: PhonemeEvidence, events: EngineEvent[]): CheckerEvent[];
  /** Decides every open error: the session is over, so pending skips are not coming back. */
  end(events?: EngineEvent[]): CheckerEvent[];
  /** Confirmed errors, plus resolved ones when the policy counts them. */
  mistakes(): CheckerEvent[];
}

/** Leaving the current place for any word start in the window: a restart or repeat (back) or a skip (ahead). */
const JUMP = 2 * UNIT;
/** Per phoneme of evidence before the lock span that the alignment may drop as leading junk. */
const LEAD = 1;
/**
 * Per phoneme of a non-Quran phrase, far below any edit: where the written words fit the same phonemes as well, they
 * are the words. A waqf ending like ٱلْعَـٰلَمِينَ passes for آمين, and a free phrase there lets a skip take the word.
 */
const PHRASE = 1 / 64;
/** A word is wrong above this weighted cost per reference phoneme it covers, with at least `MIN_WRONG_EDITS` plain edits. */
const WRONG = 0.3;
/**
 * Memorization costs: a different word changes consonants (`CONSONANTS`). Vowels, madd and the marks for qalqalah and
 * ghunna cost `SOFT` to change; a sound held longer or shorter (beside the same sound) costs `HELD`. Consonant edits
 * are hard.
 */
const SOFT = 3;
const HELD = 2;
/** A wrong word needs this many hard edits, or one in a word with at most one consonant. */
const MIN_HARD = 2;
/** Inserted phonemes between two words that make an added word. */
const MIN_ADDED = 3;
/** A visit counts as reciting the word once it covers this share of it; less is a cut-off attempt. */
const MIN_COVERAGE = 0.5;
/** Matched phonemes past an error's word that make the next word's onset. */
const ONSET = 3;
/** Phonemes kept while not anchored, to align from the lock span's start. */
const HISTORY = 160;
/** Phonemes kept when the tracker leaves the window: about what a jump holds before it commits. */
const AFTER_DROP = 24;
/** Above this many phonemes the window commits its older part, keeping the latest `KEEP`. */
const MAX_WINDOW = 240;
const KEEP = 120;
/** The window spans one ayah before the anchor and this many after the furthest recited word. */
const AHEAD_AYAHS = 2;
const PHRASE_MAX_COST = 0.3;
/** Consecutive words a lock off the declared passage spans before it counts as a deliberate jump. */
const JUMP_WORDS = 4;
/**
 * refinements.readNotSkipped: a return to an earlier ayah this many words or more behind the furthest word read (or to
 * a surah's first ayah), read through, is the passage read again: a new pass. A shorter one is a repeat inside the
 * pass. Three ayahs of the reference's median length: half of its 6236 ayahs have 10 words or fewer. Counted in words,
 * a return weighs the same where ayahs are short (Ar-Rahman's) as where they are long (Al-Baqarah's).
 */
const PASS_BACK_WORDS = 30;
/**
 * refinements.repeatNotWrong: the fewest heard phonemes a wrong word's text needs to be read as a repeat. A heard text
 * that short, fitting the text before the word within `FITS_PASSAGE` per phoneme, is no accident: the tracker locks a
 * place on 10 phonemes (`minPhonemes`), and this holds a margin over that. Shorter heard text is a word's own slot.
 */
const REPEAT_MIN = 12;
/**
 * refinements.commitAfterJump: the fewest phonemes of a restart's speech its fit is judged on, as `REPEAT_MIN` (the
 * tracker locks a place on 10, and this holds a margin over that). On fewer it is taken as a restart.
 */
const REREAD_MIN = REPEAT_MIN;
/**
 * Below this cost per phoneme the lock's words also fit the declared passage: a wrong word that forms another ayah
 * shares most of its words with the passage, a deliberate jump doesn't.
 */
const FITS_PASSAGE = 0.35;
/** Nats by which the decoded words must beat the written ones over the local window for the error to stand. */
const SUPPORT = 10;
/**
 * refinements.markWhenSure: words of a declared passage's end, inside its last ayah, that a take ending short of them
 * left out; a take that ends further back stopped.
 */
const ENDING_LEFT = 2;
/**
 * refinements.markWhenSure: how long a suspected wrong or added word's evidence (class, words, heard text) must stand
 * unchanged, while speech goes on, before it is due. Most suspicions the reading after them explains are withdrawn
 * within about 0.6 s.
 */
const STAND_MS = 800;
/**
 * A wrong word whose changed sounds the decoder heard below this mean confidence must win by twice the margin: on
 * degraded audio a confusable sound (ن for م, ف for ك) is heard at even odds with blank, yet a few of them sum past it.
 */
const UNSURE = 0.8;
/**
 * Manner that a held pausal final still separates on a phone band. Nasal hum, liquid (trill or lateral),
 * obstruent (stop or fricative) and glide. ن/م are one nasal; ف/ك are one obstruent, so those channel
 * pairs are not a contradiction. Ghunna marks take the nasal of ن/م; qalqala is not a manner.
 */
const MANNER: Readonly<Record<string, "nasal" | "liquid" | "obstruent" | "glide">> = {
  "ن": "nasal", "م": "nasal", "ں": "nasal", "۾": "nasal",
  "ر": "liquid", "ل": "liquid",
  "و": "glide", "ي": "glide",
  "ء": "obstruent", "ب": "obstruent", "ت": "obstruent", "ث": "obstruent", "ج": "obstruent", "ح": "obstruent",
  "خ": "obstruent", "د": "obstruent", "ذ": "obstruent", "ز": "obstruent", "س": "obstruent", "ش": "obstruent",
  "ص": "obstruent", "ض": "obstruent", "ط": "obstruent", "ظ": "obstruent", "ع": "obstruent", "غ": "obstruent",
  "ف": "obstruent", "ق": "obstruent", "ك": "obstruent", "ه": "obstruent",
};
/** Saktah and waqf tokens the listener can emit. A frame gap the pause test labels is the other pause. */
const PAUSE_TOKEN = new Set("ؙۜ");
/**
 * A dropped consonant-bearing ending takes a consonant plus a long vowel. A brisk murattal phoneme is two frames of
 * the 40 ms stride, and a long vowel is a madd of two counts, so both need 240 ms. A shorter hole between the
 * remaining word and the next onset was not spoken; a longer one can be a phone channel that blanked speech.
 */
const SUFFIX_HOLD_MS = (1 + 2) * 2 * 40;

const INF = Infinity;
const OP_START = 0;
const OP_INS = 1;
const OP_DIAG = 2;
const OP_DEL = 3;
const OP_JUMP = 4;
const OP_LEAD = 5;

type Step =
  | { op: "diag"; ref: number; obs: number }
  | { op: "del"; ref: number }
  | { op: "ins"; at: number; obs: number }
  | { op: "lead"; obs: number }
  | { op: "jump" };

/** An evidence phoneme; `junk` once it is known to be part of a non-Quran phrase. */
type Heard = DecodedPhoneme & { junk?: boolean };

interface Visit {
  word: number;
  covered: number;
  consumed: number;
  cost: number;
  hard: number;
  /** Consonants heard as themselves. */
  kept: number;
  obsFrom: number;
  obsTo: number;
  stepFrom?: number;
  stepTo?: number;
}

interface Issue {
  cls: MemorizationErrorClass;
  /** Word index of the error; for omissions the first omitted word, with `last` the last one. */
  word: number;
  last: number;
  obsFrom: number;
  obsTo: number;
  observed?: string;
  /** Absorbed speech reported as one word. The original span still owns this accusation's confirmation gates. */
  heardUnion?: { obsFrom: number; obsTo: number; observed: string };
  /** A lexical candidate's exact neighboring visits; other occurrences must not widen its acoustic window. */
  supportRange?: [number, number];
  /** Score this word's own frames. A short ending disappears when the next word's labels slide onto its tail. */
  wordLocal?: boolean;
  /**
   * Decoded hypothesis over `supportRange`: a shorter attested word plus the next written word. The next word's
   * consonant has to be inside this comparison, or it is free to stand in for the missing ending.
   */
  pair?: string;
  /**
   * The joint fit matched the next word, but the main alignment lent its onset to the suffix, so that word has no
   * visit and `onset` cannot see it. Confirmation may use this instead of a later diagonal.
   */
  pairOnset?: boolean;
  /**
   * Attested parallel phrase scored by CTC over `supportRange`. The written path and this one share the
   * entry and exit words; only the interior word is substituted. No insertion or restart arc. `need` is the
   * margin it must win by.
   */
  phrase?: { written: string; alt: string; need: number };
  /**
   * Surplus speech before an ayah's word 1 already spells a whole mushaf word. That identity replaces the
   * posterior support margin: the span is not needed to read the text.
   */
  lexicon?: boolean;
  /**
   * A word-final long vowel before a hamza, judged by its duration against this reciter's earlier madd
   * munfasil. The forward sum prices one vowel as nothing, so the length is the whole decision.
   */
  madd?: boolean;
  /**
   * Substitution taken from the per-word alignment, not a parallel phrase or a lexical candidate. A span that
   * spells another mushaf word, or that carries more consonants than the written word, keeps today's margin.
   * Anything shorter or the same length that spells no such word needs twice the support margin.
   */
  stringPath?: boolean;
  /**
   * Ayah-final slot before a labelled pause whose last consonant manner contradicts the written pausal
   * final. Stands in for the lexical gate and the 0.6 confidence cut. `SUPPORT` and the weak-evidence
   * gate are unchanged.
   */
  manner?: boolean;
}

interface Open {
  id: string;
  issue: Issue;
  sinceMs: number;
  /**
   * refinements.omissionDue: the reciter read on past this omission; refinements.markWhenSure: its evidence stood and
   * is sure. Its `suspected` events carry `due`.
   */
  due?: true;
  /** refinements.markWhenSure: when the suspicion's evidence (class, words, heard text) was last set, ms. */
  stableMs: number;
  /** refinements.markWhenSure: said due on its evidence, which stood (`due` is set too). */
  stood?: true;
}

export function createChecker(index: ReferenceIndex, options: CheckerOptions = {}): Checker {
  const policy = options.policy ?? {};
  const confirm = policy.confirm ?? { after: "delay", ms: 2500 };
  const refine = policy.refinements ?? {};
  const readNotSkipped = refine.readNotSkipped !== false;
  const reanchorBack = refine.reanchorBack !== false;
  const stutterRule = refine.stutter !== false;
  const ledgerPartial = refine.ledgerPartial !== false;
  const omissionDue = refine.omissionDue !== false;
  const commitAfterJump = refine.commitAfterJump !== false;
  const firstAyahOut = refine.firstAyahOut !== false;
  const marksInPassage = refine.marksInPassage !== false;
  const repeatNotWrong = refine.repeatNotWrong !== false;
  const markWhenSure = refine.markWhenSure === true;
  /** refinements.firstAyahOut: the alignment runs from the declared passage's first word, as the take began. */
  let opening = false;
  /**
   * refinements.firstAyahOut: where the tracker's first lock of the take that it kept began (its span's first word), or
   * -1.
   */
  let firstLock = -1;
  const minConfidence = policy.minConfidence ?? 0.6;
  const frames = (policy.localWindow ?? true) ? new PosteriorFrames(index.tokens.size) : null;
  const support = typeof policy.localWindow === "object" ? policy.localWindow.nats : SUPPORT;
  const tokenId = new Map([...index.tokens].map((t, i) => [t, i + 1]));
  const omissionChannel: Channel = channelFromCounts(options.channel ?? LISTENER_CHANNEL);
  const rankOmission = omissionChannel.phones.length > 0;
  const ref = index.phonemes;
  const words = index.words;
  const lexical = lexicalWords(index);
  const insertionEvidence = new Map<string, number>();
  const insertionKey = (word: WordRef, observed: string, from: number, to: number) =>
    JSON.stringify([word.surah, word.ayah, word.word, observed, from, to]);
  // A completed seam remains in the streaming window over many pushes. Its
  // immutable rows and tokens need only be certified once. Coordinates in a
  // cached result are relative to that window, so commits may trim obs safely.
  const seamExcisionCache = new Map<string, {
    until: number; span: { from: number; to: number; said: string } | null;
  }>();
  const parallels = parallelPaths(index);
  const madd = maddSites(index);
  const formulas = formulaSlots(index);
  const allSpans = slotSpans(index);
  const spanById = new Map(allSpans.map((s) => [spanId(s), s]));
  /** Spans whose continuation already picked a copy. Their slots are no longer held. */
  const resolvedSpans = new Set<string>();
  /** Spans the lock has sat in and not yet resolved. */
  const watching = new Set<string>();
  /** Latest locked word. A span that the cursor has left falls back to the copy it is on. */
  let cursor = -1;
  /**
   * refinements.reanchorBack: the furthest word the tracker has locked on. A cursor in an earlier ayah than it is the
   * tracker following a return; one that only holds or lags stays in its ayah or comes forward.
   */
  let cursorFar = -1;
  /**
   * Ayahs the alignment has actually read, and ayahs the tracker has locked, so a never-read ayah between
   * two committed locks can be judged by the closed read-versus-skip forward sum. Cleared when the lock drops.
   */
  interface AyahSpan { wordFrom: number; wordTo: number; frameFrom: number; frameTo: number }
  const readAyah = new Map<number, AyahSpan>();
  const lockedAyah = new Map<number, AyahSpan>();
  /** Last ayah the lock finished, or the alignment read through. A flash of the next ayah does not move it. */
  let settled = -1;
  let lockSurah = -1;
  let pendingGap: { left: number; right: number } | null = null;
  const ledgerDone = new Set<number>();
  /** Skips the fit already accepted. Re-asserted each window so a later alignment does not resolve them. */
  const ledgerIssues = new Map<number, Issue>();
  /**
   * Ayah span of each accepted ledger skip. The seam can open in a later window, after this one has confirmed
   * and the anchor has moved, so the span outlives `ledgerIssues`.
   */
  const ledgerSpans = new Map<number, number>();
  /** Obs index where the current push's phonemes start, so a new lock can own that chunk's frames. */
  let chunkFrom = 0;
  /** The current push's phonemes while its tracker events are read, before they join `obs`. */
  let pending: DecodedPhoneme[] = [];
  /**
   * Held slot accusations whose phones commit already trimmed. A locked step would have decided them; they wait
   * for the copy. The same-copy release confirms them, a move to the other copy drops them.
   */
  const frozen = new Map<number, { o: Open; sure: boolean }>();
  const wordStarts = new Set(words.map((w) => w.start));
  const ayahOf: number[] = [];
  const ayahFirst: number[] = [];
  const ayahLast: number[] = [];
  words.forEach((w, i) => {
    const prev = words[i - 1];
    if (!prev || prev.surah !== w.surah || prev.ayah !== w.ayah) ayahFirst.push(i);
    ayahOf.push(ayahFirst.length - 1);
    ayahLast[ayahFirst.length - 1] = i;
  });
  const fatiha = words.filter((w) => w.surah === 1 && w.ayah === 1);
  const phrases = [...(options.phrases?.phrases.map((p) => p.p) ?? []), ref.slice(fatiha[0].start, fatiha.at(-1)!.end)];
  /** refinements.readNotSkipped: the phrases that open a new recitation (a takbir begins a rak'ah, a basmala a surah). */
  const passPhrases = new Set([...(options.phrases?.phrases.filter((p) => p.name === "takbir").map((p) => p.p) ?? []), phrases.at(-1)!]);
  const nextWord = (wi: number) => {
    let k = wi + 1;
    while (k < words.length && words[k].end === words[k].start) k++;
    return k;
  };
  const prevWord = (wi: number) => {
    let k = wi - 1;
    while (k > 0 && words[k].end === words[k].start) k--;
    return k;
  };
  const wordIndex = (r: WordRef) => words.findIndex((w) => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word);
  const toRef = (i: number): WordRef => ({ surah: words[i].surah, ayah: words[i].ayah, word: words[i].word });
  /**
   * Free recitation latches onto the tracker's first lock as if it were declared, growing it as the reciter goes on,
   * so a wrong word that forms another ayah is still wrong; only a deliberate jump moves it.
   */
  const latched = !options.passage;
  let passage = options.passage && { from: wordIndex(options.passage.from), to: wordIndex(options.passage.to) };
  /**
   * refinements.readNotSkipped: in free recitation, the first word of the stretch read since the tracker latched (the
   * passage's first word before it moves on with the window), or -1.
   */
  let readFrom = -1;
  /**
   * Off the passage by more than an ayah either side. In free recitation, with refinements.readNotSkipped, the passage
   * reaches back to the start of the stretch read, as a declared one does: a return there is a return, not a jump.
   */
  const offPassage = (wi: number) => {
    if (!passage) return false;
    const from = latched && readNotSkipped && readFrom >= 0 ? Math.min(readFrom, passage.from) : passage.from;
    return ayahOf[wi] < ayahOf[from] - 1 || ayahOf[wi] > ayahOf[passage.to] + 1;
  };
  const fitsPassage = (from: number, to: number) => {
    const text = ref.slice(words[from].start, words[to].end);
    const a = alignSubstring(ref, words[passage!.from].start, words[passage!.to].end, text, (o) => wordStarts.has(o));
    return !!a && a.cost / UNIT / text.length < FITS_PASSAGE;
  };

  /** refinements.reanchorBack: the words `from`–`to` also fit the alignment window's own text, as a passage fits its copy. */
  const fitsWindow = (from: number, to: number) => {
    const text = ref.slice(words[from].start, words[to].end);
    const a = hi > lo ? alignSubstring(ref, lo, hi, text, (o) => wordStarts.has(o)) : null;
    return !!a && a.cost / UNIT / text.length < FITS_PASSAGE;
  };

  let frameMs = 0;
  let nowMs = 0;
  let trackerLocked = false;
  /** Aligning against the declared passage; false while following a deliberate jump off it. */
  let declared = !!passage;
  /** Phonemes since the last anchor (or the latest `HISTORY` while not anchored). */
  let obs: Heard[] = [];
  /**
   * Matched madd-munfasil lengths already scrolled out of the window, in frames from the stem consonant to the
   * next hamza. The threshold for a vowel drop is the shortest of these, plus any still in the window.
   */
  const pastMadd: { frame: number; span: number }[] = [];
  /** The same silence the inner-waqf path labels: a blank of `GAP_PAD_S` between two phones. */
  const labelledPause = (prev: number): boolean => {
    if (frameMs <= 0 || prev < 0 || prev + 1 >= obs.length) return false;
    let a = obs[prev]!.end + 1;
    const b = obs[prev + 1]!.start;
    if (frames && frames.startFrame >= 0 && a >= frames.startFrame && b <= frames.endFrame) {
      const id = tokenId.get(obs[prev]!.p);
      if (id) while (a < b) {
        const row = frames.row(a);
        let best = 0;
        for (let i = 1; i < row.length; i++) if (row[i]! > row[best]!) best = i;
        if (best !== id) break;
        a++;
      }
    }
    if (b - a < minGapFrames(frameMs)) return false;
    if (!frames || frames.startFrame < 0 || a < frames.startFrame || b > frames.endFrame) return true;
    const owns = Math.log(0.5);
    for (let f = a; f < b; f++) if (frames.row(f)[0]! < owns) return false;
    return true;
  };
  interface MaddHit {
    word: number;
    frame: number;
    span: number;
    consObs: number;
    hamzaObs: number;
    pause: boolean;
    /** A madd letter of the written word was heard between the consonant and the hamza. */
    matched: boolean;
  }
  /** Connected madds the alignment has both edges for. A pause between the words is not one of them. */
  const measureMadd = (aligned: Step[]): MaddHit[] => {
    const exact = new Map<number, number>();
    const seen = new Set<number>();
    for (const s of aligned) if (s.op === "diag" && ref[s.ref] === obs[s.obs]!.p) {
      exact.set(s.ref, s.obs);
      seen.add(index.wordAt(s.ref));
    }
    const out: MaddHit[] = [];
    for (const wi of seen) {
      const next = madd.munfasil.get(wi);
      if (next === undefined) continue;
      const cons = stemConsonant(ref, words[wi]!.start, words[wi]!.end);
      const hamza = words[next]!.start;
      const consObs = exact.get(cons);
      const hamzaObs = exact.get(hamza);
      if (cons < words[wi]!.start || consObs === undefined || hamzaObs === undefined || hamzaObs <= consObs) continue;
      const span = obs[hamzaObs]!.start - obs[consObs]!.start;
      if (span <= 0) continue;
      let pause = false;
      for (let k = consObs; k < hamzaObs && !pause; k++) pause = labelledPause(k);
      let matched = false;
      if (!pause) for (const s of aligned) {
        if (s.op !== "diag" || s.obs <= consObs || s.obs >= hamzaObs) continue;
        if (s.ref < words[wi]!.start || s.ref >= words[wi]!.end) continue;
        if (ref[s.ref] === obs[s.obs]!.p && isMaddLetter(ref[s.ref]!)) matched = true;
      }
      out.push({ word: wi, frame: obs[consObs]!.start, span, consObs, hamzaObs, pause, matched });
    }
    return out;
  };
  /**
   * Vowel-drop candidates shorter than every earlier matched madd in this session. A matched madd that is not
   * that short joins the baseline; a flagged drop does not, so one miss cannot lower the bar for the next.
   */
  const maddNow = (aligned: Step[]) => {
    const hits = measureMadd(aligned).sort((a, b) => a.frame - b.frame);
    const baseline = pastMadd.map((p) => ({ ...p }));
    const flags: MaddHit[] = [];
    for (const m of hits) {
      const earlier = baseline.filter((b) => b.frame < m.frame);
      const min = earlier.reduce((n, b) => Math.min(n, b.span), Infinity);
      if (madd.drops.has(m.word) && !m.pause && earlier.length > 0 && m.span < min) flags.push(m);
      else if (m.matched && !m.pause && !baseline.some((b) => b.frame === m.frame)) baseline.push({ frame: m.frame, span: m.span });
    }
    return { flags, baseline };
  };
  /** Word the window's alignment starts from; null while not anchored. */
  let anchor: number | null = null;
  let lead = 0;
  /** The anchor came from a tracker lock rather than a commit, so its first word is an edge. */
  let fromLock = false;
  let lo = 0;
  let hi = 0;
  let furthest = -1;
  /** The furthest word the tracker locked on since the window last started over. */
  let lockReach = -1;
  /** Words the window's latest alignment read right, with the frame each visit ended on. */
  let readEnds: { word: number; end: number }[] = [];
  const open = new Map<number, Open>();
  /** Words whose error was confirmed or ended uncertain; never reopened. */
  const decided = new Set<number>();
  /** Word spans of already-confirmed skipped_passage warnings, so a broader later realignment does not duplicate them. */
  const confirmedSkips: { word: number; last: number }[] = [];
  const overlapsConfirmedSkip = (word: number, last: number) => confirmedSkips.some((s) => word <= s.last && s.word <= last);
  /**
   * refinements.readNotSkipped: when each word was last read right (ms), kept across commits, so a repeat that rejoins (or
   * stops short) is not read as leaving out what the current pass already said.
   */
  const readRight = new Map<number, number>();
  /**
   * refinements.readNotSkipped: when each word was first read right in the pass it was last read in (ms), so where the
   * reciter had read to before a given moment (`readTo`) stays put when a later reading says the word again.
   */
  const firstRead = new Map<number, number>();
  /**
   * refinements.readNotSkipped: when the current pass began (ms). A word read before it is not read in this pass: reading
   * the passage again is a new recitation, and what it leaves out is left out.
   */
  let passFromMs = 0;
  /**
   * refinements.readNotSkipped: a return that may open a pass (`PASS_BACK_WORDS`), not yet read through: the ayah gone
   * back to, and when (ms). It opens a pass from then once the path reaches the ayah's last word, and is dropped if the
   * path leaves the ayah first.
   */
  let returning: { ayah: number; at: number } | null = null;
  /** refinements.readNotSkipped: a new pass from `ms` on (a takbir or basmala, a re-anchor, a return read through). */
  const newPass = (ms: number) => {
    if (ms > passFromMs) passFromMs = ms;
  };
  /**
   * The reciter went back (the path stepping back to an earlier ayah, or `reanchorBack` following a lock behind the
   * window) and has not yet come back: the furthest word read before going back, the pass then current (ms), when
   * they went back (ms), and the ayah gone back to. Coming back to the furthest ayah, up to where reading on from the
   * furthest word goes (`rejoinAt`), is a rejoin: the pass then current goes on, and nothing between is left out.
   * Reading on past the furthest word, a takbir or basmala, a jump elsewhere or a lost place ends it.
   */
  let wentBack: { furthest: number; pass: number; at: number; ayah: number } | null = null;
  /**
   * refinements.reanchorBack: words a forward follow found unread (past the furthest word, or skipped in the passage read
   * again), by first word. The alignment, re-anchored at the landing, no longer covers them.
   */
  const followGaps = new Map<number, Issue>();
  const emitted: CheckerEvent[] = [];
  let seq = 0;

  const event = (o: Open, state: CheckerState, confidence?: number): CheckerEvent => {
    const { issue } = o;
    const conf = confidence ?? clarity(issue);
    const e: CheckerEvent = {
      type: "checker",
      v: EVENT_SCHEMA_VERSION,
      id: o.id,
      timeMs: nowMs,
      state,
      errorClass: issue.cls,
      location: toRef(issue.word),
      expected: ref.slice(words[issue.word].start, words[issue.last].end),
      ...(issue.heardUnion ? { observed: issue.heardUnion.observed } : issue.observed !== undefined ? { observed: issue.observed } : {}),
      confidence: conf,
      ...(o.due && state === "suspected" && { due: true as const }),
    };
    emitted.push(e);
    return e;
  };

  /** Mean decoder confidence over an error's phonemes; for an omission, the phonemes either side of the gap. */
  const clarity = (issue: Issue): number => {
    const from = Math.max(0, Math.min(issue.obsFrom, obs.length - 1));
    const to = Math.min(obs.length, Math.max(issue.obsTo, from + 1));
    const span = issue.cls === "omitted_word" || issue.cls === "skipped_passage" ? obs.slice(Math.max(0, from - 3), to + 3) : obs.slice(from, to);
    return span.length ? span.reduce((a, p) => a + p.conf, 0) / span.length : 0;
  };

  const decide = (o: Open, out: CheckerEvent[], state?: "uncertain") => {
    open.delete(o.issue.word);
    decided.add(o.issue.word);
    // refinements.markWhenSure: a mistake said due on evidence that stood is decided on it.
    const sure = o.stood || (((declared && !latched) || trackerLocked) && heardSure(o.issue) && supported(o.issue) !== false);
    const finalState = state ?? (sure ? "confirmed" : "uncertain");
    if (o.issue.cls === "skipped_passage" && finalState === "confirmed") {
      confirmedSkips.push({ word: o.issue.word, last: o.issue.last });
    }
    out.push(event(o, finalState));
  };

  /** The steps of the window's latest alignment. */
  let steps: Step[] = [];
  /**
   * Whether the posteriors over the error's word and a recited word either side favour what was decoded over the
   * written words by `SUPPORT`; null when there's no telling (no posteriors there, or a skipped passage).
   */
  const supported = (issue: Issue, requireFinite = false, margin?: number): boolean | null => {
    // Spelled surplus is structural. A likelihood contest over the neighbours is what loses the word.
    if (issue.lexicon) return null;
    // The madd's length was already compared with this reciter's own. One vowel does not move the forward sum.
    if (issue.madd) return null;
    if (!frames || frames.startFrame < 0 || issue.cls === "skipped_passage") return null;
    const heardAt = (wi: number) => steps.filter((s): s is Extract<Step, { op: "diag" }> => s.op === "diag" && index.wordAt(s.ref) === wi).map((s) => s.obs);
    const before = prevWord(issue.word);
    const after = issue.cls === "added_word" ? issue.word : nextWord(issue.last);
    const left = before < issue.word ? heardAt(before) : [];
    const right = after < words.length ? heardAt(after) : [];
    const from = left.length ? before : issue.word;
    const to = right.length ? after : issue.cls === "added_word" ? issue.word : issue.last;
    const at = (issue.supportRange ? [issue.supportRange[0], issue.supportRange[1] - 1] :
      [...left, ...right, ...(issue.obsTo > issue.obsFrom ? [issue.obsFrom, issue.obsTo - 1] : [])]).filter((k) => k >= 0 && k < obs.length);
    if (!at.length && !issue.wordLocal) return null;
    const ids = (s: string) => [...s].map((p) => tokenId.get(p) ?? 0);
    // The ending is a few phones. Over the word plus its neighbours the next word's labels take those frames and
    // the written word looks almost as likely; on the word's own frames the consonant has to be accounted for.
    if (issue.wordLocal && issue.obsTo > issue.obsFrom) {
      const a0 = obs[issue.obsFrom].start, b0 = obs[issue.obsTo - 1].end + 1;
      if (a0 < frames.startFrame || b0 > frames.endFrame) return null;
      const word = ref.slice(words[issue.word].start, words[issue.last].end);
      const need = margin ?? supportNeed(issue);
      const expectedNll = frames.nll(ids(word), a0, b0), observedNll = frames.nll(ids(issue.observed ?? ""), a0, b0);
      if (requireFinite && (!Number.isFinite(expectedNll) || !Number.isFinite(observedNll))) return null;
      return expectedNll - observedNll >= need;
    }
    // Same joint span as the edit comparison. A wide window that leaves the next word out lets its onset consonant
    // explain the ending a second time.
    if (issue.pair && issue.supportRange && issue.supportRange[1] > issue.supportRange[0]) {
      const a0 = obs[issue.supportRange[0]].start, b0 = obs[issue.supportRange[1] - 1].end + 1;
      if (a0 < frames.startFrame || b0 > frames.endFrame) return null;
      const through = nextWord(issue.last);
      const writtenPair = ref.slice(words[issue.word].start, words[through].end);
      const need = margin ?? supportNeed(issue);
      const expectedNll = frames.nll(ids(writtenPair), a0, b0), observedNll = frames.nll(ids(issue.pair), a0, b0);
      if (requireFinite && (!Number.isFinite(expectedNll) || !Number.isFinite(observedNll))) return null;
      return expectedNll - observedNll >= need;
    }
    if (!at.length) return null;
    const f0 = obs[Math.min(...at)].start;
    const f1 = obs[Math.max(...at)].end + 1;
    if (f0 < frames.startFrame || f1 > frames.endFrame) return null;
    if (issue.phrase && issue.supportRange) {
      const collapse = (t: string) => collapseHeld([...t]).join("");
      const altNll = frames.nll(ids(collapse(issue.phrase.alt)), f0, f1);
      // A written phrase the frames cannot carry at all still loses to a finite parallel. NaN (both impossible)
      // is not evidence either way.
      if (!Number.isFinite(altNll)) return requireFinite ? null : false;
      return frames.nll(ids(collapse(issue.phrase.written)), f0, f1) - altNll >= issue.phrase.need;
    }
    const span = (a: number, b: number) => (a < b ? ref.slice(words[a].start, words[b - 1].end) : "");
    const written = ref.slice(words[from].start, words[to].end);
    const observed = issue.observed ?? "";
    const decoded =
      issue.cls === "added_word"
        ? span(from, issue.word) + observed + span(issue.word, to + 1)
        : span(from, issue.word) + (issue.cls === "wrong_word" ? observed : "") + span(issue.last + 1, to + 1);
    const need = margin ?? supportNeed(issue);
    const expectedNll = frames.nll(ids(written), f0, f1), observedNll = frames.nll(ids(decoded), f0, f1);
    if (requireFinite && (!Number.isFinite(expectedNll) || !Number.isFinite(observedNll))) return null;
    return expectedNll - observedNll >= need;
  };

  /** Waqf and pause marks at the edges of a heard span. They are not part of the word. */
  const waqfEdge = new Set("ۦۥں۾ڇؙۜ");
  const stripEdge = (s: string) => {
    let a = 0, b = s.length;
    while (a < b && waqfEdge.has(s[a]!)) a++;
    while (b > a && waqfEdge.has(s[b - 1]!)) b--;
    return s.slice(a, b);
  };
  /** The heard word with phrase-labelled pauses removed and waqf tails stripped. */
  const heardSpan = (issue: Issue): string => {
    const from = Math.max(0, issue.obsFrom);
    const to = Math.min(obs.length, issue.obsTo);
    const live = from < to ? obs.slice(from, to).filter((p) => !p.junk).map((p) => p.p).join("") : "";
    return stripEdge(live || issue.observed || "");
  };
  const consonantCount = (s: string) => collapseHeld([...s]).filter((p) => CONSONANTS.has(p)).length;
  /** Manner of the last consonant run after waqf tails are stripped. Null when no consonant remains. */
  const lastManner = (phones: string) => {
    const bare = stripEdge(phones);
    for (let i = bare.length - 1; i >= 0; i--) {
      const m = MANNER[bare[i]!];
      if (m) return m;
    }
    return null;
  };
  /** The isolated pausal form's final consonant. Connected text when the index has no pause form. */
  const pausalManner = (wi: number) =>
    lastManner(index.isolatedWordForms?.[wi] || ref.slice(words[wi].start, words[wi].end));
  /** A manner mismatch is sure without the 0.6 cut. The weak-evidence gate still reads the changed phones. */
  const heardSure = (issue: Issue) => issue.manner === true || clarity(issue) >= minConfidence;
  /**
   * Nats the decoded hypothesis must win by. A per-word substitution that does not spell another mushaf word
   * (not the written word, not a word or prefix within ±3) clears twice `SUPPORT`, unless the span has more
   * consonants than the written word: that extra speech is the next word's, and today's margin still owns it.
   * A manner mismatch replaces that lexical doubling. The weak-evidence gate still asks for twice `SUPPORT`.
   * A spelled word keeps today's margin.
   */
  const supportNeed = (issue: Issue): number => {
    const weak = issue.cls === "wrong_word" && changedClarity(issue) < UNSURE;
    // The lexical doubling is the gate a manner mismatch replaces. Weak evidence still doubles.
    if (issue.manner) return weak ? 2 * support : support;
    if (issue.stringPath) {
      const heard = heardSpan(issue);
      const written = ref.slice(words[issue.word].start, words[issue.last].end);
      if (consonantCount(heard) <= consonantCount(written) && lexical.surplusWord(issue.word, heard) === null) return 2 * support;
    }
    return weak ? 2 * support : support;
  };

  /** Mean decoder confidence over a wrong word's decoded phonemes that differ from the written word. */
  const changedClarity = (issue: Issue): number => {
    const heard = obs.slice(Math.max(0, issue.obsFrom), issue.obsTo);
    const changed = editPath(ref.slice(words[issue.word].start, words[issue.last].end), heard.map((p) => p.p).join(""))
      .flatMap((o) => (o.op === "sub" || o.op === "ins" ? [heard[o.obsAt].conf] : []));
    return changed.length ? changed.reduce((a, c) => a + c, 0) / changed.length : 1;
  };

  /** Heard phonemes after the issue, up to `to`, against the anchor with the word at `cutAt` removed. */
  const continuationFit = (issue: Issue, cutAt: number, to: number) => {
    const from = words[issue.word]?.start;
    if (from === undefined || cutAt <= from || from >= hi || cutAt > hi) return null;
    const heard = obs.slice(issue.obsTo, to).map((p) => p.p).join("");
    return issueDropFit(ref.slice(from, hi), ref.slice(cutAt, hi), heard);
  };

  /** An added word: delete its heard phones and align what follows to the anchor, which still contains that word. */
  const insertionFit = (issue: Issue, to: number) => {
    const from = words[issue.word]?.start;
    if (from === undefined || issue.obsTo <= issue.obsFrom || to <= issue.obsTo || from >= hi) return null;
    const heard = obs.slice(issue.obsFrom, to).map((p) => p.p).join("");
    const kept = obs.slice(issue.obsTo, to).map((p) => p.p).join("");
    return insertionDropFit(ref.slice(from, hi), heard, kept);
  };

  /** Every issue open now ends `uncertain` and is never reopened. A real lock loss still does this to all of them. */
  const sealAll = (out: CheckerEvent[]) => {
    for (const o of [...open.values()]) {
      open.delete(o.issue.word);
      decided.add(o.issue.word);
      out.push(event(o, "uncertain"));
    }
  };

  /**
   * A misfit hold. An issue stays open when its own edit makes the heard words up to the next open issue fit the
   * anchor again: the edited fit beats the unedited fit by the commit margin and falls below the misfit cost.
   * An omission or a wrong word deletes that written word. An added word deletes its heard phones instead and
   * keeps the phones that follow, against the anchor text those phones are. Anything else is downgraded as on a
   * lock loss.
   */
  const sealOpen = (out: CheckerEvent[], skip?: Set<number>) => {
    const ordered = [...open.values()].sort((a, b) => a.issue.obsFrom - b.issue.obsFrom);
    const keep = new Set<number>();
    for (const o of ordered) {
      const next = ordered.find((x) => x.issue.obsFrom > o.issue.obsTo);
      const to = next ? next.issue.obsFrom : obs.length;
      const fit = continuationFit(o.issue, words[o.issue.word].end, to);
      const inserted = o.issue.cls === "added_word" ? insertionFit(o.issue, to) : null;
      if (fit?.explains || inserted?.explains) keep.add(o.issue.word);
    }
    for (const o of [...open.values()]) {
      if (keep.has(o.issue.word) || skip?.has(o.issue.word)) continue;
      open.delete(o.issue.word);
      decided.add(o.issue.word);
      out.push(event(o, "uncertain"));
    }
  };

  const noteSpan = (wi: number) => {
    if (wi < 0) return;
    for (const s of allSpans) {
      const id = spanId(s);
      if (resolvedSpans.has(id) || watching.has(id)) continue;
      if (s.starts.some((st) => wi >= st && wi < st + s.length)) watching.add(id);
    }
  };

  /** A differing slot of a span the lock is still inside. Accusations there wait for the continuation. */
  const heldSlot = (wi: number) => {
    for (const id of watching) {
      if (resolvedSpans.has(id)) continue;
      const s = spanById.get(id);
      if (!s) continue;
      for (const st of s.starts) for (const off of s.slots) if (st + off === wi) return true;
    }
    return false;
  };

  const slotWords = () => {
    const out = new Set<number>();
    for (const id of watching) {
      if (resolvedSpans.has(id)) continue;
      const s = spanById.get(id);
      if (!s) continue;
      for (const st of s.starts) for (const off of s.slots) out.add(st + off);
    }
    return out;
  };

  const resolveWatching = () => {
    for (const id of watching) resolvedSpans.add(id);
    watching.clear();
  };

  const sureOf = (o: Open) =>
    ((declared && !latched) || trackerLocked) && heardSure(o.issue) && supported(o.issue) !== false;
  /** refinements.markWhenSure: the same suspicion, the same words and the same text heard. */
  const sameEvidence = (a: Issue, b: Issue) =>
    a.cls === b.cls && a.word === b.word && a.last === b.last && a.observed === b.observed;
  /**
   * refinements.markWhenSure: sure enough to show before its confirmation, or to decide at the take's end: all a
   * confirmation asks of the same evidence (`sureOf`: heard clearly, and the posteriors favouring what was heard over
   * the written words by `supportNeed`'s margin, twice `SUPPORT` for a change heard unclearly or spelling no word), and
   * the posteriors there to tell (a mistake they cannot judge waits for its confirmation).
   */
  const standsSure = (o: Open) => sureOf(o) && supported(o.issue) === true;
  /**
   * refinements.markWhenSure: at the take's end, a mistake where the reading stopped: a wrong word with the word before
   * it recited right and nothing heard after it, or an added word between two words recited right. Speech after the
   * words recited right that the alignment could not place on the passage (other speech, a hold, several words heard
   * wrong in a row) is not a reading that stopped there, and is not decided as wrong words.
   */
  const endsReading = (issue: Issue, right: Set<number>) =>
    issue.word > 0 && right.has(prevWord(issue.word)) &&
    (issue.cls === "wrong_word" ? issue.obsTo >= obs.length : issue.cls === "added_word" && right.has(issue.word));

  /** Confirm frozen slot accusations, or drop the ones that named the copy the continuation rejected. */
  const releaseFrozen = (out: CheckerEvent[], drop: boolean, from = -1, end = Infinity) => {
    for (const [wi, f] of [...frozen]) {
      if (wi < from || wi >= end) continue;
      frozen.delete(wi);
      decided.add(f.o.issue.word);
      out.push(event(f.o, drop ? "resolved" : (f.sure ? "confirmed" : "uncertain")));
    }
  };

  const freezeSlot = (wi: number, o: Open) => {
    if (frozen.has(wi)) return;
    frozen.set(wi, { o, sure: sureOf(o) });
    open.delete(wi);
  };

  /** True while the cursor is still inside a watched span or on its first few continuation words. */
  const spanStillOpen = (wi: number) => {
    for (const id of watching) {
      if (resolvedSpans.has(id)) continue;
      const s = spanById.get(id);
      if (!s) continue;
      if (s.starts.some((st) => wi >= st && wi < st + s.length + 4)) return true;
    }
    return false;
  };

  /**
   * The continuation picked the other member. Slot accusations named the old copy, so they drop. Every other
   * decision stays, and the phones stay in the window a locked step already committed: no fresh-lock `lead`, no
   * slice of the word-1 gap. The anchor slides onto the parallel text and the next step keeps appending.
   */
  const rebase = (from: number, to: number, out: CheckerEvent[]) => {
    const span = allSpans.find((s) => s.starts.includes(from) && s.starts.includes(to));
    if (!span || anchor === null) return;
    opening = false;
    const delta = to - from;
    const end = from + span.length;
    const slotAt = new Set(span.slots.map((off) => from + off));
    for (const wi of [...open.keys()]) {
      if (wi < from || wi >= end) continue;
      const o = open.get(wi)!;
      open.delete(wi);
      if (slotAt.has(wi)) out.push(event(o, "resolved"));
      else {
        const moved = o.issue.word + delta;
        // The in-order word was already accused (the real skip). Close the copy's duplicate instead of reopening it.
        if (decided.has(moved)) {
          decided.add(o.issue.word);
          out.push(event(o, "resolved"));
        } else {
          const issue = { ...o.issue, word: moved, last: o.issue.last + delta };
          open.set(issue.word, { ...o, issue });
        }
      }
    }
    releaseFrozen(out, true, from, end);
    for (const w of [...decided]) if (w >= from && w < end && !slotAt.has(w)) decided.add(w + delta);
    for (const [w, last] of [...ledgerSpans]) {
      if (w < from || w >= end) continue;
      ledgerSpans.delete(w);
      // A whole ayah sits inside the span, so the end slides with the start. A slot was the other copy.
      if (!slotAt.has(w)) ledgerSpans.set(w + delta, last + delta);
    }
    for (const sk of [...confirmedSkips]) {
      if (sk.word >= from && sk.word < end && !slotAt.has(sk.word)) {
        confirmedSkips.push({ word: sk.word + delta, last: sk.last + delta });
      }
    }
    if (anchor >= from && anchor <= end) anchor += delta;
    else if (anchor < from) anchor = to;
    if (furthest >= from && furthest < end) furthest += delta;
    if (furthest < anchor) furthest = anchor;
    if (passage) {
      passage.from += delta;
      passage.to = Math.max(passage.from, passage.to + delta);
    }
    clearLedger();
    // The next lock in this same batch is the continuation. Score it against the slid window, not the copy's.
    const ayah = ayahOf[anchor!];
    lo = words[ayahFirst[Math.max(0, ayah - 1)]].start;
    const far = ayahOf[Math.max(anchor!, furthest)];
    hi = words[ayahLast[Math.min(ayahLast.length - 1, far + AHEAD_AYAHS)]].end;
  };

  /**
   * The reciter went back to word `back` at `ms`, having read up to word `reach`. A return while already away is the
   * same excursion: it keeps the pass and the furthest word from before the first.
   */
  const goneBack = (back: number, reach: number, ms: number) => {
    const far = Math.max(reach, wentBack?.furthest ?? -1);
    const ayah = ayahOf[back];
    if (far < 0 || ayah >= ayahOf[far]) return;
    wentBack = { furthest: far, pass: wentBack?.pass ?? passFromMs, at: ms, ayah };
    if (readNotSkipped && ms > passFromMs && (far - back >= PASS_BACK_WORDS || words[back].ayah === 1)) {
      if (!returning || ms > returning.at) returning = { ayah, at: ms };
    }
  };

  /**
   * Where the reciter had read to before the speech of the lock on words `start`–`end`, made at `atMs`, which went
   * back. The window and the tracker both read the first words of a return as words ahead until the tracker finds it:
   * the furthest word the alignment read right before that speech, and no further than the tracker had locked
   * (`lockedTo`) before. The lock's speech is the phonemes heard up to the lock, this push's included, so where it
   * begins does not depend on how the audio was chunked.
   */
  const readBefore = (start: number, end: number, lockedTo: number, atMs: number) => {
    const heard = pending.length ? [...obs, ...pending] : obs;
    let k = heard.length;
    while (k > 0 && (heard[k - 1].end + 1) * frameMs > atMs) k--;
    const cut = heard[Math.max(0, k - (words[end].end - words[start].start))]?.start ?? Infinity;
    let reach = anchor ?? -1;
    for (const r of readEnds) if (r.end < cut) reach = Math.max(reach, r.word);
    return lockedTo < 0 ? reach : Math.min(reach, lockedTo);
  };

  /**
   * Start over where the tracker went; `back` is the word it went back to, behind the window, the reciter having read
   * to word `reach` (`readBefore`). A forward follow (`followForward`) keeps the pass and the return as it decided
   * them.
   */
  const dropAnchor = (out: CheckerEvent[], back?: number, follow = false, reach = -1) => {
    opening = false;
    lockReach = -1;
    readEnds = [];
    // A fit that already accepted a skip has its evidence. The lock leaving is the same as the passage ending.
    for (const wi of [...ledgerIssues.keys()]) {
      const o = open.get(wi);
      ledgerIssues.delete(wi);
      if (!o) continue;
      open.delete(wi);
      decided.add(wi);
      const finalState = sureOf(o) ? "confirmed" : "uncertain";
      if (o.issue.cls === "skipped_passage" && finalState === "confirmed") {
        confirmedSkips.push({ word: o.issue.word, last: o.issue.last });
      }
      out.push(event(o, finalState));
    }
    sealAll(out);
    anchor = null;
    furthest = -1;
    const ms = obs.length ? obs[Math.max(0, obs.length - AFTER_DROP)].start * frameMs : 0;
    // A forward follow (`followForward`) has decided the pass and the return itself.
    if (!follow && back !== undefined) goneBack(back, reach, ms);
    else if (!follow) {
      wentBack = null;
      // A jump, or the place lost: where the tracker locks next is not a return.
      cursorFar = -1;
      // refinements.readNotSkipped: what the alignment reads from the new anchor on is a new pass.
      if (readNotSkipped) newPass(ms);
    }
    obs = obs.slice(-AFTER_DROP);
    // Ayahs already read stay read, so the next lock can still close a gap this jump opened.
  };

  /**
   * refinements.reanchorBack: the reciter, followed back, came forward again past the window's far end, to word `landing`.
   * The window starts over there. Landing on the furthest ayah read before going back, up to where reading on from
   * the furthest word goes (`rejoinAt`), is a rejoin and marks nothing; a lock's first word can be a word late, so
   * the word after that is still a rejoin. Past it, the words between were never read: an omission, or a skipped
   * passage from three words. A landing short of the furthest ayah, after a return that opened a pass (the passage
   * read again), skipped the words from where that reading stood: they are left out of this pass. Nothing is marked
   * on the words read before or during the excursion.
   */
  const followForward = (out: CheckerEvent[], landing: number) => {
    const back = wentBack!;
    const reach = Math.max(furthest, anchor ?? -1);
    let gapFrom = -1;
    if (comesBack(landing)) {
      const at = rejoinAt(back.furthest);
      if (landing > nextWord(at)) gapFrom = at;
      cameBack();
    } else if (readNotSkipped && passFromMs > back.pass && reach >= 0) gapFrom = nextWord(reach);
    const gapTo = prevWord(landing);
    dropAnchor(out, undefined, true);
    if (gapFrom >= 0 && gapFrom <= gapTo) {
      const n = words.slice(gapFrom, gapTo + 1).filter((w) => w.end > w.start).length;
      // A new reading of these words: what the window decided there before (while it read the excursion as the
      // passage) does not stand for it.
      if (n) {
        decided.delete(gapFrom);
        followGaps.set(gapFrom, { cls: n >= 3 ? "skipped_passage" : "omitted_word", word: gapFrom, last: gapTo, obsFrom: -1, obsTo: -1 });
      }
    }
  };

  /**
   * refinements.firstAyahOut: the take's first reading began at word `start` of the declared passage, past the window's
   * reach. The window starts over there, as after a jump, and the passage's words before it, three or more, are one
   * skip; what the window read of that speech against the passage's opening is set aside.
   */
  const startLate = (out: CheckerEvent[], start: number) => {
    const [from, to] = [passage!.from, prevWord(start)];
    dropAnchor(out, undefined, true);
    const n = words.slice(from, to + 1).filter((w) => w.end > w.start).length;
    if (n >= 3) {
      decided.delete(from);
      followGaps.set(from, { cls: "skipped_passage", word: from, last: to, obsFrom: -1, obsTo: -1 });
    }
  };

  const clearLedger = () => {
    readAyah.clear();
    lockedAyah.clear();
    settled = -1;
    lockSurah = -1;
    pendingGap = null;
    ledgerIssues.clear();
  };

  const growSpan = (map: Map<number, AyahSpan>, ayah: number, word: number, frameFrom: number, frameTo: number) => {
    const h = map.get(ayah);
    if (!h) map.set(ayah, { wordFrom: word, wordTo: word, frameFrom, frameTo });
    else {
      h.wordFrom = Math.min(h.wordFrom, word);
      h.wordTo = Math.max(h.wordTo, word);
      h.frameFrom = Math.min(h.frameFrom, frameFrom);
      h.frameTo = Math.max(h.frameTo, frameTo);
    }
  };

  /** Letter identity of an ayah. A refrain copy occurs among its neighbours; a skipped unique ayah does not. */
  const ayahText = (ayah: number) => {
    let s = "";
    for (let i = ayahFirst[ayah]; i <= ayahLast[ayah]; i++) s += `${letterSkeleton(index.wordText?.[i] ?? "")}\0`;
    return s || ref.slice(words[ayahFirst[ayah]].start, words[ayahLast[ayah]].end);
  };

  /**
   * Refrain words (an ayah whose letter text occurs again) and the shared words of a near-duplicate span.
   * The word a span's members disagree on is the slot, not duplicate text.
   */
  const copyText = new Set<number>();
  {
    const skel: string[] = [];
    const copies = new Map<string, number>();
    for (let a = 0; a < ayahFirst.length; a++) {
      const t = ayahText(a);
      skel.push(t);
      copies.set(t, (copies.get(t) ?? 0) + 1);
    }
    for (let a = 0; a < skel.length; a++) if ((copies.get(skel[a]) ?? 0) > 1) {
      for (let w = ayahFirst[a]; w <= ayahLast[a]; w++) copyText.add(w);
    }
    for (const s of allSpans) {
      const slot = new Set(s.slots);
      for (const st of s.starts) for (let off = 0; off < s.length; off++) if (!slot.has(off)) copyText.add(st + off);
    }
  }

  const wordPhones = (from: number, to: number) => {
    const out: string[] = [];
    for (let i = from; i <= to; i++) out.push(ref.slice(words[i].start, words[i].end));
    return out;
  };

  const spanFrames = (ayah: number) => readAyah.get(ayah) ?? lockedAyah.get(ayah);

  /**
   * Frames still needed for an open gap: the left neighbour's speech, until the skip-versus-read sum runs.
   * Null when nothing is waiting, so the window drops frames as before.
   */
  const ledgerFloor = (): number | null => {
    if (!pendingGap || !frames) return null;
    const span = spanFrames(pendingGap.left);
    return span ? span.frameFrom : null;
  };

  /** The tracker locked on `right` having come from `left`. Score once the right ayah is in, or the passage moves on. */
  const closeGap = (out: CheckerEvent[], issues: Map<number, Issue>, left: number, right: number, force: boolean) => {
    if (!frames || frames.startFrame < 0 || words[ayahFirst[left]].surah !== words[ayahFirst[right]].surah) return;
    const rightLock = lockedAyah.get(right);
    const rightRead = readAyah.get(right);
    const wordTo = Math.max(rightLock?.wordTo ?? -1, rightRead?.wordTo ?? -1);
    if (!force && wordTo < ayahLast[right] && settled <= right) return;
    if (pendingGap && pendingGap.left === left && pendingGap.right === right) pendingGap = null;
    const leftSpan = spanFrames(left);
    const rightTo = Math.max(rightLock?.frameTo ?? 0, rightRead?.frameTo ?? 0);
    if (!leftSpan || wordTo < ayahFirst[right] || rightTo <= leftSpan.frameFrom) return;
    if (leftSpan.frameFrom < frames.startFrame || rightTo > frames.endFrame) return;
    const leftTo = Math.max(leftSpan.wordTo, lockedAyah.get(left)?.wordTo ?? -1);
    const leftWords = wordPhones(Math.max(ayahFirst[left], leftSpan.wordFrom), Math.min(ayahLast[left], leftTo));
    const rightWords = wordPhones(ayahFirst[right], Math.min(ayahLast[right], wordTo));
    const idOf = (p: string) => tokenId.get(p) ?? 0;
    let at = 0;
    const boundary = rightLock?.frameFrom ?? rightRead?.frameFrom ?? rightTo;
    for (let i = 0; i < obs.length; i++) if (obs[i].start < boundary) at = i;
    for (let a = left + 1; a < right; a++) {
      if (ayahFinished(a) || ledgerDone.has(a)) continue;
      // refinements.ledgerPartial: an ayah the alignment read in part was not skipped whole. Its unread words are the
      // alignment's to judge, word by word, beside the part that was read.
      if (ledgerPartial && readAyah.has(a)) continue;
      if (ayahText(a) === ayahText(left) || ayahText(a) === ayahText(right)) continue;
      const n = words.slice(ayahFirst[a], ayahLast[a] + 1).filter((w) => w.end > w.start).length;
      if (n < 3) continue;
      const margin = absenceMargin(frames, idOf, [leftWords, wordPhones(ayahFirst[a], ayahLast[a]), rightWords], 1, leftSpan.frameFrom, rightTo);
      ledgerDone.add(a);
      if (margin === null || margin < PHRASE_SWITCH) continue;
      const wi = ayahFirst[a];
      if (decided.has(wi) || heldSlot(wi) || overlapsConfirmedSkip(wi, ayahLast[a])) continue;
      const openIssue = open.get(wi);
      if (openIssue && openIssue.issue.cls !== "wrong_word" && openIssue.issue.cls !== "omitted_word") {
        // The seam already keyed this ayah. Remember the ayah so a later seam on the same run can be dropped.
        if (openIssue.issue.cls === "skipped_passage") ledgerSpans.set(wi, ayahLast[a]);
        continue;
      }
      const existing = issues.get(wi);
      if (existing && existing.cls !== "wrong_word" && existing.cls !== "omitted_word") {
        if (existing.cls === "skipped_passage") ledgerSpans.set(wi, ayahLast[a]);
        continue;
      }
      if (openIssue) {
        open.delete(wi);
        out.push(event(openIssue, "resolved"));
      }
      const issue: Issue = { cls: "skipped_passage", word: wi, last: ayahLast[a], obsFrom: at, obsTo: at };
      ledgerIssues.set(wi, issue);
      ledgerSpans.set(wi, issue.last);
      issues.set(wi, issue);
    }
  };

  const keepLedger = (issues: Map<number, Issue>) => {
    for (const [wi, issue] of [...ledgerIssues]) {
      if (decided.has(wi) || ayahFinished(ayahOf[wi])) {
        ledgerIssues.delete(wi);
        // Confirmed skips stay. A finished ayah was read, so the skip did not stand.
        if (!decided.has(wi)) ledgerSpans.delete(wi);
        continue;
      }
      // A wrong or omitted word on this ayah start is the skip misfiled. The fit already chose.
      const now = issues.get(wi);
      if (!now || now.cls === "wrong_word" || now.cls === "omitted_word") issues.set(wi, issue);
    }
  };

  const ayahFinished = (ayah: number) =>
    (lockedAyah.get(ayah)?.wordTo ?? -1) >= ayahLast[ayah] || (readAyah.get(ayah)?.wordTo ?? -1) >= ayahLast[ayah];

  const noteLock = () => {
    if (cursor < 0 || !obs.length) return;
    const ayah = ayahOf[cursor];
    const surah = words[ayahFirst[ayah]].surah;
    const from = obs[Math.min(chunkFrom, obs.length - 1)]!;
    const to = obs.at(-1)!;
    growSpan(lockedAyah, ayah, cursor, from.start, to.end + 1);
    if (surah !== lockSurah) {
      settled = -1;
      lockSurah = surah;
      pendingGap = null;
    }
    if (settled < 0) {
      if (ayahFinished(ayah)) settled = ayah;
      return;
    }
    if (ayah > settled + 1 && pendingGap?.right !== ayah && words[ayahFirst[settled]].surah === surah
      && ayah - settled <= AHEAD_AYAHS + 1) {
      pendingGap = { left: settled, right: ayah };
    }
    if (ayahFinished(ayah) && ayah > settled) settled = ayah;
  };

  const setWindow = () => {
    const a = ayahOf[anchor!];
    lo = words[ayahFirst[Math.max(0, a - 1)]].start;
    const f = ayahOf[Math.max(anchor!, furthest)];
    hi = words[ayahLast[Math.min(ayahLast.length - 1, f + AHEAD_AYAHS)]].end;
  };

  /** Per-window cost tables: every cell cost depends only on the window position and the heard phoneme. */
  let costLo = -1;
  let costHi = -1;
  let delCost = new Float64Array(0);
  const rowCosts = new Map<string, { sub: Float64Array; ins: Float64Array }>();
  /** `align`'s table as the last call left it: the rows for its first `rows` heard phonemes (`heard`, `junk`). */
  const dp = {
    lo: -1, hi: -1, a0: -1, lead: -1, rows: 0,
    d: new Float64Array(0), op: new Uint8Array(0), jumpFrom: new Int32Array(0),
    heard: [] as Heard[], junk: [] as boolean[],
  };
  const costsFor = (p: string) => {
    let c = rowCosts.get(p);
    if (!c) {
      const n = hi - lo;
      c = { sub: new Float64Array(n), ins: new Float64Array(n + 1) };
      for (let i = 0; i < n; i++) c.sub[i] = subCost(ref[lo + i], p);
      for (let i = 0; i <= n; i++) c.ins[i] = insertionCost(ref, lo + i, p, hi);
      rowCosts.set(p, c);
    }
    return c;
  };

  /** Weighted alignment of the window's phonemes from the anchor, with jump arcs to every word start in the window. */
  const align = (): Step[] => {
    if (lo !== costLo || hi !== costHi) {
      costLo = lo;
      costHi = hi;
      rowCosts.clear();
      delCost = new Float64Array(hi - lo);
      for (let i = 0; i < hi - lo; i++) {
        const at = lo + i;
        // The listener can shorten any doubled sound, including a shadda, without dropping the word.
        delCost[i] = ref[at] === ref[at - 1] || ref[at] === ref[at + 1] ? HELD : deletionCost(ref, at);
      }
    }
    const a0 = words[anchor!].start - lo;
    const n = hi - lo;
    const m = obs.length;
    const w = n + 1;
    // Row j of the table depends only on the window (lo, hi), the anchor, `lead` and the heard phonemes up to j (each
    // phoneme's sound and junk mark, and the sound before it). The last alignment's rows are kept, and a call computes
    // only the rows after the longest prefix that is unchanged, the same values in the same order: the window's
    // phonemes grow by a cell at a time, so most rows are the previous call's.
    if (dp.lo !== lo || dp.hi !== hi || dp.a0 !== a0 || dp.lead !== lead) {
      dp.lo = lo;
      dp.hi = hi;
      dp.a0 = a0;
      dp.lead = lead;
      dp.rows = 0;
    }
    let kept = Math.min(dp.rows, m);
    for (let j = 0; j < kept; j++) {
      const h = obs[j];
      if (dp.heard[j] !== h || dp.junk[j] !== !!h.junk) {
        kept = j;
        break;
      }
    }
    if ((m + 1) * w > dp.d.length) {
      const size = 2 * (m + 1) * w;
      const d = new Float64Array(size);
      const op = new Uint8Array(size);
      d.set(dp.d.subarray(0, (kept + 1) * w));
      op.set(dp.op.subarray(0, (kept + 1) * w));
      dp.d = d;
      dp.op = op;
    }
    if (m + 1 > dp.jumpFrom.length) {
      const jumpFrom = new Int32Array(2 * (m + 1));
      jumpFrom.set(dp.jumpFrom.subarray(0, kept + 1));
      dp.jumpFrom = jumpFrom;
    }
    const { d, op, jumpFrom } = dp;
    // Indexed loops over typed arrays, and the per-row facts read once: the app runs this in JavaScriptCore
    // without a JIT, where every iterator, property read and call in the inner loop is paid in full.
    let startCount = 0;
    const starts = new Int32Array(n + 1);
    for (let i = 0; i <= n; i++) if (wordStarts.has(lo + i)) starts[startCount++] = i;
    const jumps = (j: number) => {
      const row = j * w;
      let k = 0;
      let low = d[row];
      for (let i = 1; i <= n; i++) {
        const v = d[row + i];
        if (v < low) {
          low = v;
          k = i;
        }
      }
      jumpFrom[j] = k;
      const via = low + JUMP;
      for (let x = 0; x < startCount; x++) {
        const at = row + starts[x];
        if (via < d[at]) {
          d[at] = via;
          op[at] = OP_JUMP;
        }
      }
    };
    if (dp.rows === 0 || kept === 0) {
      d.fill(INF, 0, w);
      op.fill(OP_START, 0, w);
      d[a0] = 0;
      op[a0] = OP_START;
      for (let i = a0 + 1; i <= n; i++) {
        d[i] = d[i - 1] + delCost[i - 1];
        op[i] = OP_DEL;
      }
      jumps(0);
    }
    for (let j = kept + 1; j <= m; j++) {
      const p = obs[j - 1].p;
      const { sub, ins: insCost } = costsFor(p);
      // A repeated decoded sound can be lengthening even at a word edge, where the reference has no matching run.
      // Use the same held-sound charge as judging; keep ordinary insertion costs for unrelated speech.
      const noise = p === obs[j - 2]?.p ? HELD : UNIT;
      const row = j * w;
      const prev = row - w;
      const junk = obs[j - 1].junk;
      const leadAt = j <= lead ? a0 : -1;
      // Column 0: no diagonal or deletion into it.
      {
        const ins = d[prev] + (junk ? PHRASE : leadAt === 0 ? LEAD : insCost[0] < noise ? insCost[0] : noise);
        d[row] = ins < INF ? ins : INF;
        op[row] = ins < INF && leadAt === 0 ? OP_LEAD : OP_INS;
      }
      let left = d[row];
      for (let i = 1; i <= n; i++) {
        let best = d[prev + i - 1] + sub[i - 1];
        let o = OP_DIAG;
        let extra: number;
        if (junk) extra = PHRASE;
        else if (i === leadAt) extra = LEAD;
        else {
          const c = insCost[i];
          extra = c < noise ? c : noise;
        }
        const ins = d[prev + i] + extra;
        if (ins < best) {
          best = ins;
          o = i === leadAt ? OP_LEAD : OP_INS;
        }
        const del = left + delCost[i - 1];
        if (del < best) {
          best = del;
          o = OP_DEL;
        }
        d[row + i] = best;
        op[row + i] = o;
        left = best;
      }
      jumps(j);
    }
    dp.heard.length = m;
    dp.junk.length = m;
    for (let j = kept; j < m; j++) {
      dp.heard[j] = obs[j];
      dp.junk[j] = !!obs[j].junk;
    }
    dp.rows = m;

    let i = 0;
    for (let k = 1; k <= n; k++) if (d[m * w + k] < d[m * w + i]) i = k;
    const steps: Step[] = [];
    let j = m;
    for (;;) {
      const o = op[j * w + i];
      if (o === OP_START) break;
      if (o === OP_INS) steps.push({ op: "ins", at: lo + i, obs: --j });
      else if (o === OP_LEAD) steps.push({ op: "lead", obs: --j });
      else if (o === OP_DIAG) steps.push({ op: "diag", ref: lo + --i, obs: --j });
      else if (o === OP_DEL) steps.push({ op: "del", ref: lo + --i });
      else {
        steps.push({ op: "jump" });
        i = jumpFrom[j];
      }
    }
    return steps.reverse();
  };

  /**
   * Mark phrases that end at each phoneme from `from` on: the alignment consumes them for `PHRASE`. Once anchored, a run
   * the window's own text fits at least as well is that text (ءَامِنِينَ, not آمين).
   */
  const markPhrases = (from: number) => {
    for (let end = from + 1; end <= obs.length; end++) {
      for (const p of phrases) {
        const start = Math.max(0, end - Math.ceil(p.length * 1.6) - 4);
        const tail = obs.slice(start, end).map((x) => x.p).join("");
        const a = alignSubstring(tail, 0, tail.length, p, () => true);
        if (!a || a.end < tail.length - 1 || a.cost / UNIT / p.length > PHRASE_MAX_COST) continue;
        if (anchor !== null) {
          const text = alignSubstring(ref, lo, hi, tail.slice(a.start), () => true);
          if (text && text.cost <= a.cost) continue;
        }
        for (let k = start + a.start; k < end; k++) obs[k].junk = true;
        // refinements.readNotSkipped: what is read after a takbir or a basmala is a new pass; a return before it is over.
        if (passPhrases.has(p)) {
          if (readNotSkipped) newPass(obs[start + a.start].start * frameMs);
          wentBack = null;
        }
      }
    }
  };

  const charge = (v: Visit, cost: number) => {
    v.cost += cost;
    if (cost >= UNIT) v.hard++;
  };
  const subCharge = (r: string, p: string) => (r === p ? 0 : CONSONANTS.has(r) && CONSONANTS.has(p) ? subCost(r, p) : SOFT);
  const delCharge = (at: number) => (ref[at] === ref[at - 1] || ref[at] === ref[at + 1] ? HELD : CONSONANTS.has(ref[at]) ? UNIT : SOFT);
  /** Inserting `p` before `ref[at]`, after the heard `prev`. */
  const insCharge = (at: number, p: string, prev?: string) =>
    p === ref[at - 1] || p === ref[at] || p === prev ? HELD : CONSONANTS.has(p) ? UNIT : SOFT;

  /** Min-charge alignment of `obs[from, to)` onto word `wi`. The path is that alignment. */
  const chargeAlign = (wi: number, from: number, to: number): Step[] => {
    const start = words[wi].start;
    const n = words[wi].end - start;
    const m = Math.max(0, to - from);
    const W = m + 1;
    const cost = new Float64Array((n + 1) * W);
    const back = new Int32Array((n + 1) * W);
    const kind = new Uint8Array((n + 1) * W);
    for (let j = 1; j <= m; j++) {
      cost[j] = cost[j - 1] + insCharge(start, obs[from + j - 1].p, obs[from + j - 2]?.p);
      back[j] = j - 1;
      kind[j] = OP_INS;
    }
    for (let i = 1; i <= n; i++) {
      const at = start + i - 1;
      const del = delCharge(at);
      const row = i * W;
      const above = row - W;
      cost[row] = cost[above] + del;
      back[row] = above;
      kind[row] = OP_DEL;
      for (let j = 1; j <= m; j++) {
        const p = obs[from + j - 1].p;
        const sub = subCharge(ref[at], p);
        let best = cost[above + j - 1] + sub;
        let b = above + j - 1;
        let k = OP_DIAG;
        const dcost = cost[above + j] + del;
        if (dcost < best) { best = dcost; b = above + j; k = OP_DEL; }
        const icost = cost[row + j - 1] + insCharge(at + 1, p, obs[from + j - 2]?.p);
        if (icost < best) { best = icost; b = row + j - 1; k = OP_INS; }
        cost[row + j] = best;
        back[row + j] = b;
        kind[row + j] = k;
      }
    }
    const path: Step[] = [];
    for (let cell = n * W + m; cell > 0;) {
      const i = Math.floor(cell / W);
      const j = cell % W;
      const how = kind[cell];
      if (how === OP_DIAG) path.push({ op: "diag", ref: start + i - 1, obs: from + j - 1 });
      else if (how === OP_DEL) path.push({ op: "del", ref: start + i - 1 });
      else path.push({ op: "ins", at: start + i, obs: from + j - 1 });
      cell = back[cell];
    }
    path.reverse();
    return path;
  };

  /** Whether a visit's cost makes the words `[v.word, last]` it covers a wrong word rather than a mispronounced one. */
  const wrong = (v: Visit, last = v.word) => {
    let weight = 0;
    let consonants = 0;
    for (let k = words[v.word].start; k < words[last].end; k++) {
      const hard = CONSONANTS.has(ref[k]);
      weight += hard ? UNIT : SOFT;
      if (hard) consonants++;
    }
    const length = words[last].end - words[v.word].start;
    return v.cost / ((weight * v.covered) / length) > WRONG && v.hard >= Math.max(1, Math.min(MIN_HARD, consonants));
  };

  const skeleton = (wi: number) => [...ref.slice(words[wi].start, words[wi].end)].filter((p) => CONSONANTS.has(p)).join("");
  const hasConsonant = (wi: number) => [...ref.slice(words[wi].start, words[wi].end)].some((p) => CONSONANTS.has(p));

  /** Score phonemes heard where the alignment skipped words `[from, to]`, as one visit over them. */
  const scoreGap = (from: number, to: number, said: Heard[]): Visit => {
    const start = words[from].start;
    const end = words[to].end;
    const v: Visit = { word: from, covered: end - start, consumed: said.length, cost: 0, hard: 0, kept: 0, obsFrom: 0, obsTo: 0 };
    const a = alignSubstring(ref, start, end, said.map((p) => p.p).join(""), (o) => o === start, { strictStart: true });
    if (!a) return { ...v, cost: Infinity, hard: Infinity };
    let at = start;
    for (const o of a.path) {
      if (o.op === "ins") charge(v, insCharge(at, o.obs, said[o.obsAt - 1]?.p));
      else {
        at = a.start + o.refAt + 1;
        charge(v, o.op === "del" ? delCharge(at - 1) : subCharge(o.ref, o.op === "sub" ? o.obs : o.ref));
      }
    }
    for (let k = a.end; k < end; k++) charge(v, delCharge(k));
    return v;
  };

  /**
   * Costs of word `wi` read from its far edge (its start when `after`, else its end) against the first `j` of `heard`,
   * read the same way, for every `j`.
   */
  const edgeFit = (wi: number, heard: string[], after: boolean, noise = false): Float64Array => {
    const w = words[wi];
    const n = w.end - w.start;
    const r = (i: number) => (after ? w.start + i : w.end - 1 - i);
    const pos = (i: number) => (after ? w.start + i : w.end - i);
    const ins = (i: number, j: number) => {
      const cost = insertionCost(ref, pos(i), heard[j - 1], w.end);
      return noise ? Math.min(cost, heard[j - 1] === heard[j - 2] ? HELD : UNIT) : cost;
    };
    const del = (i: number) => noise && (ref[r(i)] === ref[r(i) - 1] || ref[r(i)] === ref[r(i) + 1]) ? HELD : deletionCost(ref, r(i));
    let row = new Float64Array(heard.length + 1);
    for (let j = 1; j <= heard.length; j++) row[j] = row[j - 1] + ins(0, j);
    for (let i = 1; i <= n; i++) {
      const next = new Float64Array(heard.length + 1);
      next[0] = row[0] + del(i - 1);
      for (let j = 1; j <= heard.length; j++) {
        next[j] = Math.min(
          row[j - 1] + subCost(ref[r(i - 1)], heard[j - 1]),
          row[j] + del(i - 1),
          next[j - 1] + ins(i, j),
        );
      }
      row = next;
    }
    return row;
  };

  /**
   * An added word found inside a word can take the word's edge with it: وبنين كريم aligned as وبنِ[ينكر]يم, held sounds
   * being cheap to insert beside the word's own. Unless the word fits worse on its own, split the visit's heard phonemes
   * again where it fits best, the added word keeping the rest on its side: it was said between the words.
   */
  const resplit = (issue: Issue, v: Visit, after: boolean) => {
    if (v.obsFrom < 0) return;
    const at: number[] = [];
    for (let k = v.obsFrom; k < v.obsTo; k++) if (!obs[k].junk) at.push(k);
    if (!after) at.reverse();
    const heard = at.map((k) => obs[k].p);
    const keep = at.filter((k) => k < issue.obsFrom || k >= issue.obsTo).map((k) => obs[k].p);
    const now = edgeFit(v.word, keep, after)[keep.length];
    const fit = edgeFit(v.word, heard, after);
    let best = -1;
    for (let s = 0; s + MIN_ADDED + 1 <= heard.length; s++) {
      if (heard.slice(s).filter((p) => CONSONANTS.has(p)).length < 2) continue;
      if (best < 0 || fit[s] < fit[best]) best = s;
    }
    if (best < 0 || fit[best] > now) return;
    const rest = at.slice(best).sort((a, b) => a - b);
    issue.obsFrom = rest[0];
    issue.obsTo = rest.at(-1)! + 1;
    issue.observed = rest.map((k) => obs[k].p).join("");
  };

  /** Rebuild a whole-word fit after choosing an insertion between two intact neighboring words. */
  const wordSteps = (wi: number, from: number, to: number): Step[] => {
    const word = words[wi], n = word.end - word.start, m = to - from, width = m + 1;
    const d = new Float64Array((n + 1) * width);
    const op = new Uint8Array(d.length);
    const ins = (at: number, j: number) => Math.min(insertionCost(ref, at, obs[from + j - 1].p, word.end), obs[from + j - 1].p === obs[from + j - 2]?.p ? HELD : UNIT);
    const del = (at: number) => ref[at] === ref[at - 1] || ref[at] === ref[at + 1] ? HELD : deletionCost(ref, at);
    for (let j = 1; j <= m; j++) { d[j] = d[j - 1] + ins(word.start, j); op[j] = OP_INS; }
    for (let i = 1; i <= n; i++) {
      const row = i * width, prev = row - width, at = word.start + i - 1;
      d[row] = d[prev] + del(at); op[row] = OP_DEL;
      for (let j = 1; j <= m; j++) {
        const p = obs[from + j - 1].p;
        d[row + j] = d[prev + j - 1] + subCost(ref[at], p); op[row + j] = OP_DIAG;
        const deletion = d[prev + j] + del(at);
        if (deletion < d[row + j]) { d[row + j] = deletion; op[row + j] = OP_DEL; }
        const insertion = d[row + j - 1] + ins(at + 1, j);
        if (insertion < d[row + j]) { d[row + j] = insertion; op[row + j] = OP_INS; }
      }
    }
    const path: Step[] = [];
    let i = n, j = m;
    while (i || j) {
      const o = op[i * width + j];
      if (o === OP_DIAG) path.push({ op: "diag", ref: word.start + --i, obs: from + --j });
      else if (o === OP_DEL) path.push({ op: "del", ref: word.start + --i });
      else path.push({ op: "ins", at: word.start + i, obs: from + --j });
    }
    return path.reverse();
  };

  /**
   * One insertion/deletion alignment of `reference` against `obs[from, to)`, the same DP as `wordSteps`. Each heard
   * phoneme is consumed once, so a consonant on the next word cannot also satisfy a suffix.
   */
  const fitSpan = (reference: string, from: number, to: number): { cost: number; matches: { refAt: number; obs: number }[] } | null => {
    const n = reference.length, m = to - from;
    if (!n || m < 0 || from < 0 || to > obs.length) return null;
    const width = m + 1;
    const d = new Float64Array((n + 1) * width);
    const op = new Uint8Array(d.length);
    // The heard phonemes' facts, read once: their sounds, the held-sound charge, and the insertion charge away from a
    // reference run of the same sound (where `insertionCost` is the plain charge: `insertionCost("", 0, p, 0)`). Each
    // cell reads the same costs as `Math.min(insertionCost(reference, at, p, n), noise)` and `subCost`, from tables:
    // the app runs this in JavaScriptCore without a JIT, where a call and a string key per cell are paid in full.
    const ps: string[] = new Array(m + 1);
    const noise = new Float64Array(m + 1);
    const plain = new Float64Array(m + 1);
    for (let j = 1; j <= m; j++) {
      const p = obs[from + j - 1].p;
      ps[j] = p;
      noise[j] = p === obs[from + j - 2]?.p ? HELD : UNIT;
      const c = insertionCost("", 0, p, 0);
      plain[j] = c < noise[j] ? c : noise[j];
    }
    const ins = (at: number, j: number) => {
      const p = ps[j];
      if (!((at > 0 && reference[at - 1] === p) || (at < n && reference[at] === p))) return plain[j];
      return Math.min(insertionCost(reference, at, p, n), noise[j]);
    };
    const del = (at: number) => reference[at] === reference[at - 1] || reference[at] === reference[at + 1] ? HELD : deletionCost(reference, at);
    const subRows = new Map<string, Float64Array>();
    for (let j = 1; j <= m; j++) { d[j] = d[j - 1] + ins(0, j); op[j] = OP_INS; }
    for (let i = 1; i <= n; i++) {
      const row = i * width, prev = row - width, at = i - 1;
      const r = reference[at];
      let sub = subRows.get(r);
      if (!sub) {
        sub = new Float64Array(m + 1);
        for (let j = 1; j <= m; j++) sub[j] = subCost(r, ps[j]);
        subRows.set(r, sub);
      }
      const removal = del(at);
      d[row] = d[prev] + removal; op[row] = OP_DEL;
      for (let j = 1; j <= m; j++) {
        let best = d[prev + j - 1] + sub[j];
        let o = OP_DIAG;
        const deletion = d[prev + j] + removal;
        if (deletion < best) { best = deletion; o = OP_DEL; }
        const insertion = d[row + j - 1] + ins(at + 1, j);
        if (insertion < best) { best = insertion; o = OP_INS; }
        d[row + j] = best; op[row + j] = o;
      }
    }
    const matches: { refAt: number; obs: number }[] = [];
    let i = n, j = m;
    while (i || j) {
      const o = op[i * width + j];
      if (o === OP_DIAG) matches.push({ refAt: --i, obs: from + --j });
      else if (o === OP_DEL) i--;
      else if (j) j--;
      else break;
    }
    return { cost: d[n * width + m], matches };
  };

  /**
   * The hole between the remaining word's last confident phone and the next word's confident onset cannot hold the
   * dropped consonant and a long vowel. No confident edge on either side is not evidence the ending was unspoken.
   */
  const suffixUnspoken = (formLength: number, matches: { refAt: number; obs: number }[]): boolean => {
    if (frameMs <= 0) return false;
    let stemEnd = -1, onsetStart = Infinity;
    for (const m of matches) {
      const ph = obs[m.obs];
      if (ph.conf < minConfidence) continue;
      if (m.refAt < formLength) stemEnd = Math.max(stemEnd, ph.end);
      else onsetStart = Math.min(onsetStart, ph.start);
    }
    return stemEnd >= 0 && onsetStart < Infinity && (onsetStart - stemEnd - 1) * frameMs < SUFFIX_HOLD_MS;
  };

  /** The errors the current alignment shows, by word; and where it stands. `bound` is the words a counting visit holds. */
  const judge = (steps: Step[], recoverAdditions = true): { issues: Map<number, Issue>; lexical: Issue[]; head: number; furthest: number; steps: Step[]; right: Set<number>; bound: Set<number>; visits: Visit[] } | null => {
    const visits: Visit[] = [];
    const added: Issue[] = [];
    let cur: Visit | null = null;
    let block: number[] = [];
    /** Insertions inside the current word, with where they sit. */
    let inner: { at: number; obs: number }[] = [];
    let head = -1;
    /** Added words found inside a visit, re-split once the visit is complete. */
    const within: { issue: Issue; v: Visit; after: boolean }[] = [];
    // A long run inside a word is another word said there, before it or after it by which half it sits in.
    const flushInner = () => {
      const v = cur;
      if (!v || !inner.length) return void (inner = []);
      if (inner.length >= MIN_ADDED + 1 && inner.filter((k) => CONSONANTS.has(obs[k.obs].p)).length >= 2) {
        const w = words[v.word];
        const after = inner[0].at - w.start >= (w.end - w.start) / 2;
        const word = after ? nextWord(v.word) : v.word;
        const text = inner.map((k) => obs[k.obs].p).join("");
        const issue: Issue = { cls: "added_word", word, last: word, obsFrom: inner[0].obs, obsTo: inner.at(-1)!.obs + 1, observed: text };
        added.push(issue);
        within.push({ issue, v, after });
      } else for (const k of inner) charge(v, insCharge(k.at, obs[k.obs].p, obs[k.obs - 1]?.p));
      inner = [];
    };
    let stepAt = 0;
    const close = () => {
      flushInner();
      if (cur) visits.push(cur);
      cur = null;
    };
    for (; stepAt < steps.length; stepAt++) {
      const s = steps[stepAt];
      if (s.op === "jump") {
        close();
        block = [];
        continue;
      }
      if (s.op === "lead") continue;
      if (s.op === "ins") {
        const inside = cur && !wordStarts.has(s.at) && s.at > words[cur.word].start && s.at < words[cur.word].end;
        if (obs[s.obs].junk) continue;
        if (inside) {
          inner.push({ at: s.at, obs: s.obs });
          cur!.obsTo = s.obs + 1;
          cur!.stepTo = stepAt + 1;
        } else if (cur || visits.length) block.push(s.obs);
        continue;
      }
      const wi = index.wordAt(s.ref);
      if (!cur || cur.word !== wi) {
        const prev: Visit | null = cur;
        close();
        // Only between two consecutive words of one run: elsewhere an insertion is junk around a jump or the lock edge.
        if (block.length >= MIN_ADDED && block.filter((k) => CONSONANTS.has(obs[k].p)).length >= 2 && prev && nextWord(prev.word) === wi) {
          const text = block.map((k) => obs[k].p).join("");
          added.push({ cls: "added_word", word: wi, last: wi, obsFrom: block[0], obsTo: block.at(-1)! + 1, observed: text });
        }
        block = [];
        cur = { word: wi, covered: 0, consumed: 0, cost: 0, hard: 0, kept: 0, obsFrom: s.op === "diag" ? s.obs : -1, obsTo: -1, stepFrom: stepAt };
      }
      flushInner();
      const v: Visit = cur;
      v.stepTo = stepAt + 1;
      v.covered++;
      if (s.op === "diag") {
        if (v.obsFrom < 0) v.obsFrom = s.obs;
        v.obsTo = s.obs + 1;
        v.consumed++;
        charge(v, subCharge(ref[s.ref], obs[s.obs].p));
        if (ref[s.ref] === obs[s.obs].p && CONSONANTS.has(ref[s.ref])) v.kept++;
        head = wi;
      } else charge(v, delCharge(s.ref));
    }
    close();
    for (const x of within) resplit(x.issue, x.v, x.after);

    const length = (wi: number) => words[wi].end - words[wi].start;
    // Vowels alone or one isolated consonant aren't a multi-phoneme word: a neighbour's noisy onset can land on it.
    const counts = (v: Visit) => v.consumed > 0 && v.covered / length(v.word) >= MIN_COVERAGE &&
      (hasConsonant(v.word) ? v.kept > 0 && v.consumed >= Math.min(2, length(v.word)) : true);
    /** Each word's last counting visit, and whether any counting visit got it right. */
    const last = new Map<number, Visit>();
    const right = new Set<number>();
    for (const v of visits) {
      if (!counts(v) || v.word < anchor!) continue;
      last.set(v.word, v);
      if (!wrong(v)) right.add(v.word);
    }
    for (const v of visits) {
      if (!counts(v) || wrong(v) || v.word < anchor! || v.obsFrom < 0 || v.obsTo <= v.obsFrom) continue;
      growSpan(readAyah, ayahOf[v.word], v.word, obs[v.obsFrom].start, obs[v.obsTo - 1].end + 1);
    }
    if (recoverAdditions) {
      // Use whole-word pronunciations here: a cheap one-consonant substitution is tolerated by the
      // ordinary checker, but cannot prove that an intervening word was actually repeated.
      const original = visits.filter(v => counts(v) && !wrong(v));
      const intact = original.filter(v => lexical.matches(v.word, obs.slice(v.obsFrom, v.obsTo).map(p => p.p).join("")));
      const repairs: { from: number; to: number; path: Step[] }[] = [];
      const weight = (wi: number) => [...ref.slice(words[wi].start, words[wi].end)].reduce((n, p) => n + (CONSONANTS.has(p) ? UNIT : SOFT), 0);
      // Preserve the existing repairs first; stricter extra candidates cannot replace their evidence.
      const pairs: [Visit, Visit, boolean][] = original.slice(1).map((v, i) => [original[i], v, false]);
      for (let i = 1; i < intact.length; i++) {
        pairs.push([intact[i - 1], intact[i], true]);
        // An isolated forward detour cannot be a repeat of earlier speech. Backward detours
        // remain protected, as do multiword excursions and attempts at either neighbor.
        if (i >= 2) {
          const left = intact[i - 2], detour = intact[i - 1], right = intact[i];
          if (nextWord(left.word) === right.word && detour.word > nextWord(right.word) &&
            !visits.some(v => v.word === detour.word && v.obsTo <= left.obsFrom && v.obsFrom >= 0 &&
              lexical.matches(v.word, obs.slice(v.obsFrom, v.obsTo).map(p => p.p).join("")))) {
            pairs.push([left, right, true]);
          }
        }
      }
      for (const [left, right, strict] of pairs) {
        if (nextWord(left.word) !== right.word || left.obsTo > right.obsFrom || left.stepFrom === undefined || right.stepTo === undefined) continue;
        if (repairs.some(r => left.stepFrom! < r.to && right.stepTo! > r.from)) continue;
        // New segmentation candidates must establish an addition while its following word
        // is still current or just completed. Later realignment cannot open a stale warning;
        // an already suspected addition keeps its evidence until confirmation.
        if (strict && open.get(right.word)?.issue.cls !== "added_word" &&
          intact.filter(v => v.obsFrom >= right.obsTo).length > 1) continue;
        // A tolerated repeat of the preceding word stays a repeat even if its pronunciation
        // is not an exact lexical form. Only the new strict candidates need this protection.
        if (strict && visits.some(v => v.word === left.word && v.obsFrom >= left.obsTo &&
          v.obsTo <= right.obsFrom && counts(v) && !wrong(v))) continue;
        // A wrong attempt at the following word, then that word read correctly, is a self-correction.
        if (open.get(right.word)?.issue.cls === "wrong_word" || visits.some(v => v.word === right.word && v.obsFrom >= left.obsTo && v.obsFrom < right.obsFrom)) continue;
        if (added.some(a => a.word === right.word && a.obsFrom >= left.obsFrom && a.obsTo <= right.obsTo)) continue;
        const from = left.obsFrom, to = right.obsTo;
        // A visit can start with deletions and insertions before its first diagonal match. Its obsFrom then
        // excludes that leading evidence; only replace a path whose observed indices are covered exactly.
        const represented = steps.slice(left.stepFrom, right.stepTo).flatMap(s => "obs" in s ? [s.obs] : []);
        if (represented.length !== to - from || represented.some((oi, j) => oi !== from + j)) continue;
        const heard = obs.slice(from, to);
        if (heard.some(p => p.junk)) continue;
        const phones = heard.map(p => p.p), n = phones.length;
        const leftLength = words[left.word].end - words[left.word].start, rightLength = words[right.word].end - words[right.word].start;
        if (n < leftLength + rightLength + MIN_ADDED) continue;
        const leftNow = phones.slice(0, left.obsTo - from), rightNow = phones.slice(right.obsFrom - from).reverse();
        const leftLimit = Math.min(WRONG * weight(left.word), edgeFit(left.word, leftNow, true, true)[leftNow.length]);
        const rightLimit = Math.min(WRONG * weight(right.word), edgeFit(right.word, rightNow, false, true)[rightNow.length]);
        const l = edgeFit(left.word, phones, true, true), r = edgeFit(right.word, [...phones].reverse(), false, true);
        let joined = Infinity;
        for (let a = 0; a <= n; a++) joined = Math.min(joined, l[a] + r[n - a]);
        const consonants = [0];
        for (const p of phones) consonants.push(consonants.at(-1)! + Number(CONSONANTS.has(p)));
        let best: { a: number; b: number; cost: number } | null = null;
        for (let a = 1; a + MIN_ADDED < n; a++) {
          if (l[a] > leftLimit) continue;
          for (let b = a + MIN_ADDED; b < n; b++) {
            if (consonants[b] - consonants[a] < 2 || r[n - b] > rightLimit) continue;
            const cost = l[a] + r[n - b];
            if (joined - cost < MIN_HARD * UNIT) continue;
            if (!best || cost < best.cost || (cost === best.cost && b - a < best.b - best.a)) best = { a, b, cost };
          }
        }
        if (!best || new Set(phones.slice(best.a, best.b).filter(p => CONSONANTS.has(p))).size < 2) continue;
        const a = from + best.a, b = from + best.b;
        const insertion: Step[] = Array.from({ length: b - a }, (_, j) => ({ op: "ins", at: words[right.word].start, obs: a + j }));
        repairs.push({ from: left.stepFrom, to: right.stepTo, path: [...wordSteps(left.word, from, a), ...insertion, ...wordSteps(right.word, b, to)] });
      }
      if (repairs.length) {
        const repaired = [...steps];
        for (const r of repairs.sort((a, b) => b.from - a.from)) repaired.splice(r.from, r.to - r.from, ...r.path);
        return judge(repaired, false);
      }
    }
    const stepEdits: { lo: number; hi: number; path: Step[] }[] = [];
    const seen = [...last.keys()];
    const issues = new Map<number, Issue>();
    if (!seen.length) return null;
    // The lock span's first word can be a cut-off tail or leading junk: span edges are only good to a word.
    let first = Math.min(...seen) + (fromLock ? 1 : 0);
    // refinements.firstAyahOut: aligning from the declared passage's first word, a first reading that begins in a later
    // ayah, three words or more in, left those words out: the gap scan starts at the passage's first word.
    // A first lock outside the passage (another surah, before or after it) is no later start of it.
    if (firstAyahOut && opening && declared && passage && anchor === passage.from && !fromLock && (firstLock < 0 || (firstLock >= passage.from && firstLock <= passage.to))) {
      const begun = Math.min(...seen);
      if (ayahOf[begun] > ayahOf[passage.from] && begun <= passage.to && words.slice(passage.from, begun).filter((w) => w.end > w.start).length >= 3) first = passage.from;
    }
    // A wrong visit far ahead can be junk aligned to a short word; only correct visits show how far the reciter got.
    const reached = Math.max(...seen.filter((wi) => right.has(wi)), first);
    let gap: Issue | null = null;
    const flush = (after?: Visit) => {
      if (!gap) return;
      // Evidence inside the gap means the words were said, differently: a wrong word, not an omission. Phrase marks
      // don't count against that, a garbled word can look like "amin".
      const said = after ? obs.slice(gap.obsFrom, after.obsFrom) : [];
      const expected = words[gap.last].end - words[gap.word].start;
      if (said.length >= Math.max(MIN_ADDED, expected / 2) && said.some((p) => CONSONANTS.has(p.p))) {
        if (wrong(scoreGap(gap.word, gap.last, said), gap.last)) {
          issues.set(gap.word, { ...gap, cls: "wrong_word", obsTo: after!.obsFrom, observed: said.map((p) => p.p).join(""), stringPath: true });
        }
        gap = null;
        return;
      }
      // Words past the head may still be coming: a gap is an omission once the reciter is past it, or already suspected.
      const was = open.get(gap.word)?.issue.cls;
      if (!(gap.last < head || was === "omitted_word" || was === "skipped_passage")) return void (gap = null);
      // One heard stretch, two adjacent words, the outer neighbour already accepted. The alignment kept the
      // stretch on the word its hand costs prefer. Rank the two survivors by the listener channel instead,
      // and move the omission onto the other word when that survivor explains the stretch. The omission
      // count does not change. A tie, or an empty channel, leaves the alignment's placement.
      if (rankOmission && gap.word === gap.last && after && !said.length && !issues.has(prevWord(gap.word))) {
        const y = gap.word;
        const x = prevWord(y);
        const left = prevWord(x);
        const seg = last.get(x);
        const leftV = last.get(left);
        if (ayahOf[x] === ayahOf[y] && seg && right.has(x) && seg.obsTo === gap.obsFrom && seg.obsFrom >= 0
          && seg.stepFrom !== undefined && seg.stepTo !== undefined
          && leftV && right.has(left) && leftV.obsTo >= 0
          && after.word === nextWord(y)) {
          const from = leftV.obsTo;
          const to = seg.obsTo;
          // More phones than the two words is other speech, not one segment shared by them.
          const budget = words[x].end - words[x].start + (words[y].end - words[y].start);
          const heard = to > from && to - from <= budget ? collapseHeld(obs.slice(from, to).filter((p) => !p.junk).map((p) => p.p)) : [];
          const written = (wi: number) => collapseHeld([...ref.slice(words[wi].start, words[wi].end)]);
          if (heard.length && spanLogP(heard, written(y), omissionChannel) > spanLogP(heard, written(x), omissionChannel)) {
            const dels: Step[] = [];
            for (let k = words[x].start; k < words[x].end; k++) dels.push({ op: "del", ref: k });
            let lo = seg.stepFrom;
            let hi = seg.stepTo;
            for (let k = 0; k < steps.length; k++) {
              const s = steps[k];
              const o = s.op === "diag" || s.op === "ins" || s.op === "lead" ? s.obs : -1;
              const at = s.op === "diag" || s.op === "del" ? s.ref : s.op === "ins" ? s.at : -1;
              const w = at >= 0 ? index.wordAt(at) : -1;
              if ((o >= from && o < to) || w === x || w === y) {
                lo = Math.min(lo, k);
                hi = Math.max(hi, k + 1);
              }
            }
            if (hi < steps.length && steps[hi].op === "jump") {
              let k = hi + 1;
              while (k < steps.length && (steps[k].op === "jump" || steps[k].op === "lead")) k++;
              const s = steps[k];
              const at = s && (s.op === "diag" || s.op === "del") ? s.ref : s && s.op === "ins" ? s.at : -1;
              if (at >= 0 && index.wordAt(at) === after.word) hi++;
            }
            stepEdits.push({ lo, hi, path: [...dels, ...chargeAlign(y, from, to)] });
            right.delete(x);
            issues.set(x, { cls: "omitted_word", word: x, last: x, obsFrom: from, obsTo: from });
            gap = null;
            return;
          }
        }
      }
      // A word either side of a skip can recur on the other side (برب الناس [ملك الناس] إله الناس), so the alignment
      // can place the gap a word off. Of the equal placements, take the one on an ayah's start: reciters skip ayahs.
      // A look-alike ending (one leading consonant, the article heard erased) is the same slide, but only when that
      // jump sits on a blank run the pause test already labels, or on a cut-off visit. Nothing anchored: the lexical
      // placement stands, so a skip is never created or removed.
      const onStart = (i: Issue) => ayahFirst[ayahOf[i.word]] === i.word;
      const alike = (a: number, b: number) => {
        const x = skeleton(a), y = skeleton(b);
        if (!x || !y || x === y) return false;
        const [longer, shorter] = x.length >= y.length ? [x, y] : [y, x];
        return longer.length === shorter.length + 1 && longer.endsWith(shorter);
      };
      const slide = (match: (a: number, b: number) => boolean): Issue[] => {
        const out: Issue[] = [];
        let r: Issue = gap!;
        for (let v = after; v && nextWord(r.last) === v.word && match(r.word, v.word); v = last.get(nextWord(v.word))) {
          r = { ...r, word: nextWord(r.word), last: v.word, obsFrom: v.obsTo, obsTo: v.obsTo };
          out.push(r);
        }
        let l: Issue = gap!;
        for (let v = last.get(prevWord(l.word)); v && match(v.word, l.last); v = last.get(prevWord(v.word))) {
          l = { ...l, word: v.word, last: prevWord(l.last), obsFrom: v.obsFrom, obsTo: v.obsFrom };
          out.push(l);
        }
        return out;
      };
      // Frames between two phonemes that the blank already owns. Missing posteriors leave the greedy blank run.
      const blankOwned = (from: number, to: number) => {
        if (!frames || frames.startFrame < 0 || from < frames.startFrame || to > frames.endFrame) return true;
        const owns = Math.log(0.5);
        for (let f = from; f < to; f++) if (frames.row(f)[0] < owns) return false;
        return true;
      };
      const anchored = (i: Issue) => {
        const at = i.obsFrom;
        if (at > 0 && at < obs.length && frameMs > 0) {
          const a = obs[at - 1].end + 1, b = obs[at].start;
          if (b - a >= minGapFrames(frameMs) && blankOwned(a, b)) return true;
        }
        return visits.some((v) => v.consumed > 0 && v.obsFrom >= 0 && v.covered / length(v.word) < MIN_COVERAGE && v.obsFrom <= at && at <= v.obsTo);
      };
      if (!onStart(gap)) {
        const today = slide((a, b) => skeleton(a) === skeleton(b)).find(onStart) ?? gap;
        const pauseSeam = onStart(today) ? undefined : slide(alike).find((i) => onStart(i) && anchored(i));
        gap = pauseSeam ?? today;
      }
      const n = words.slice(gap.word, gap.last + 1).filter((x) => x.end > x.start).length;
      issues.set(gap.word, { ...gap, cls: n >= 3 ? "skipped_passage" : "omitted_word" });
      gap = null;
    };
    for (let wi = first, top = Math.max(...seen); wi <= top; wi++) {
      if (length(wi) === 0) continue;
      const v = last.get(wi);
      if (!v) {
        // The phonemes either side of the gap: the last one before it in the evidence is where the skip happened.
        const before = [...last.values()].filter((x) => x.word < wi).reduce((a, x) => Math.max(a, x.obsTo), 0);
        if (gap) gap.last = wi;
        else gap = { cls: "omitted_word", word: wi, last: wi, obsFrom: before, obsTo: before };
        continue;
      }
      flush(v);
      // The word being recited is judged only once it reaches the word's end: before that it may still come right.
      if (wi === head && v.covered < length(wi)) continue;
      // Once recited right, a later wrong visit is a repeat that went astray; a wrong one then a right one is a correction.
      if (!right.has(wi)) {
        issues.set(wi, { cls: "wrong_word", word: wi, last: wi, obsFrom: v.obsFrom, obsTo: v.obsTo, observed: obs.slice(v.obsFrom, v.obsTo).map((p) => p.p).join(""), stringPath: true });
      }
    }
    flush();
    for (const a of added) if (a.word >= first && !issues.has(a.word)) issues.set(a.word, a);
    const candidates: Issue[] = [];
    const heard = (v: Visit) => obs.slice(v.obsFrom, v.obsTo).map(p => p.p).join("");
    const complete = (v: Visit) => v.covered === length(v.word) && v.obsFrom >= 0 &&
      v.stepFrom !== undefined && v.stepTo !== undefined && lexical.matches(v.word, heard(v));
    const intact = visits.filter(v => counts(v) && !wrong(v) && complete(v));
    for (let i = 1; i < intact.length; i++) {
      const left = intact[i - 1], right = intact[i], wi = nextWord(left.word);
      if (wi < first || nextWord(wi) !== right.word || left.obsTo >= right.obsFrom || issues.has(wi)) continue;
      // Equal sounds straddling a boundary admit another split, especially madd and ghunna.
      if ([left.obsTo, right.obsFrom].some(at => obs[at - 1]?.p === obs[at]?.p)) continue;
      // A genuinely correct visit elsewhere protects repeats and self-corrections. The old soft `right` set alone
      // cannot establish that: it includes the consonant changes this separate candidate path is meant to catch.
      if (visits.some(v => v.word === wi && complete(v))) continue;
      if (added.some(a => a.obsFrom < right.obsTo && a.obsTo > left.obsFrom)) continue;
      const path = steps.slice(left.stepFrom!, right.stepTo!);
      if (path.some(s => s.op === "jump" || s.op === "lead" ||
        ("ref" in s && (index.wordAt(s.ref) < left.word || index.wordAt(s.ref) > right.word)))) continue;
      const represented = path.flatMap(s => "obs" in s ? [s.obs] : []);
      if (represented.length !== right.obsTo - left.obsFrom || represented.some((oi, j) => oi !== left.obsFrom + j)) continue;
      if (obs.slice(left.obsFrom, right.obsTo).some(p => p.junk)) continue;
      const observed = obs.slice(left.obsTo, right.obsFrom).map(p => p.p).join("");
      if (lexical.alternative(wi, observed, {
        before: ayahOf[left.word] === ayahOf[wi] ? heard(left) : undefined,
        after: ayahOf[right.word] === ayahOf[wi] ? heard(right) : undefined,
      })) candidates.push({
        cls: "wrong_word", word: wi, last: wi, obsFrom: left.obsTo, obsTo: right.obsFrom, observed,
        supportRange: [left.obsFrom, right.obsTo],
      });
    }
    // A pronoun or dual ending rides in the tail, where a pause or a held vowel is cheap and no consonant was
    // substituted. The insertion gap before the next visited word supplies the complete heard form.
    const settled = visits.filter(v => counts(v) && !wrong(v) && v.word >= first && v.obsFrom >= 0 &&
      v.stepFrom !== undefined && v.stepTo !== undefined);
    for (let i = 0; i < settled.length - 1; i++) {
      const left = settled[i], right = settled[i + 1];
      if (nextWord(left.word) !== right.word || left.obsTo > right.obsFrom || right.stepFrom! < left.stepTo!) continue;
      if (issues.has(left.word)) continue;
      // Another visit that is this word keeps a repeat or a self-correction. The tail may sit past the last matched
      // phone, so this visit's own slice can still be the written form.
      if (settled.some(v => v !== left && v.word === left.word && lexical.matches(v.word, heard(v)))) continue;
      const between = steps.slice(left.stepTo!, right.stepFrom!);
      if (between.some(s => s.op !== "ins")) continue;
      const gap = between.flatMap(s => s.op === "ins" ? [s.obs] : []);
      if (gap.length !== right.obsFrom - left.obsTo || gap.some((oi, j) => oi !== left.obsTo + j)) continue;
      if (obs.slice(left.obsFrom, right.obsFrom).some(p => p.junk)) continue;
      if (added.some(a => a.obsFrom < right.obsFrom && a.obsTo > left.obsFrom)) continue;
      const observed = obs.slice(left.obsFrom, right.obsFrom).map(p => p.p).join("");
      const ending = lexical.ending(left.word, observed);
      // An observed suffix has decoded consonant frames whose confidence and posteriors can corroborate it. A missing
      // suffix has no such local anchor; the ordinary omission path remains responsible for that evidence.
      if (!ending?.observedConsonant) continue;
      candidates.push({
        cls: "wrong_word", word: left.word, last: left.word, obsFrom: left.obsFrom, obsTo: right.obsFrom, observed,
        wordLocal: true,
      });
    }
    // The other direction: the ending consonant was not said. A jump to the next word is cheaper than deleting it,
    // and held-sound tolerance can lend that word's onset to the suffix, so the two words are fit as one span.
    for (let i = 0; i < settled.length - 1; i++) {
      const left = settled[i], right = settled[i + 1];
      if (nextWord(left.word) !== right.word || left.obsTo > right.obsFrom || right.stepFrom! < left.stepTo!) continue;
      if (issues.has(left.word) || candidates.some(c => c.word === left.word)) continue;
      if (settled.some(v => v !== left && v.word === left.word && lexical.matches(v.word, heard(v)))) continue;
      const alts = lexical.droppedSuffixes(left.word);
      if (!alts.length) continue;
      const from = left.obsFrom, to = right.obsTo;
      if (to <= from || obs.slice(from, to).some(p => p.junk)) continue;
      if (visits.some(v => v !== left && v !== right && v.obsFrom < to && v.obsTo > from)) continue;
      if (added.some(a => a.obsFrom < to && a.obsTo > from)) continue;
      const next = ref.slice(words[right.word].start, words[right.word].end);
      const written = fitSpan(ref.slice(words[left.word].start, words[left.word].end) + next, from, to);
      if (!written) continue;
      let best: { form: string; cost: number; matches: { refAt: number; obs: number }[] } | null = null;
      for (const form of alts) {
        const fit = fitSpan(form + next, from, to);
        if (fit && (!best || fit.cost < best.cost)) best = { form, cost: fit.cost, matches: fit.matches };
      }
      // The same margin as an added ending: the shorter pair beats the written pair by a consonant.
      if (!best || written.cost - best.cost < UNIT) continue;
      if (!suffixUnspoken(best.form.length, best.matches)) continue;
      candidates.push({
        cls: "wrong_word", word: left.word, last: left.word, obsFrom: from,
        obsTo: right.obsFrom > from ? right.obsFrom : left.obsTo, observed: best.form,
        supportRange: [from, to], pair: best.form + next,
      });
    }
    // No visit of the next word: its phones were matched as the ending. The span is this visit plus whatever was
    // heard before the following settled word, so the onset consonant is inside the fit and cannot also be the suffix.
    for (const left of settled) {
      if (left.covered !== length(left.word)) continue;
      if (issues.has(left.word) || candidates.some(c => c.word === left.word)) continue;
      if (settled.some(v => v !== left && v.word === left.word && lexical.matches(v.word, heard(v)))) continue;
      const partner = nextWord(left.word);
      if (partner >= words.length || settled.some(v => v.word === partner)) continue;
      const alts = lexical.droppedSuffixes(left.word);
      if (!alts.length) continue;
      const later = settled.find(v => v.obsFrom >= left.obsTo && v.word > left.word);
      const from = left.obsFrom, to = later ? later.obsFrom : left.obsTo;
      if (to <= from || obs.slice(from, to).some(p => p.junk)) continue;
      if (visits.some(v => v !== left && v.word !== partner && v.obsFrom < to && v.obsTo > from && counts(v))) continue;
      if (added.some(a => a.obsFrom < to && a.obsTo > from)) continue;
      const next = ref.slice(words[partner].start, words[partner].end);
      if (!next) continue;
      const written = fitSpan(ref.slice(words[left.word].start, words[left.word].end) + next, from, to);
      if (!written) continue;
      let best: { form: string; cost: number; matches: { refAt: number; obs: number }[] } | null = null;
      for (const form of alts) {
        const fit = fitSpan(form + next, from, to);
        if (fit && (!best || fit.cost < best.cost)) best = { form, cost: fit.cost, matches: fit.matches };
      }
      if (!best || written.cost - best.cost < UNIT) continue;
      const seq = best.form + next;
      let need = 0, got = 0;
      for (let k = 0; k < next.length; k++) if (CONSONANTS.has(next[k])) {
        need++;
        const at = best.form.length + k;
        if (best.matches.some(m => m.refAt === at && obs[m.obs].p === seq[at])) got++;
      }
      if (!need || got < need) continue;
      if (!suffixUnspoken(best.form.length, best.matches)) continue;
      let onsetObs = to;
      for (const m of best.matches) if (m.refAt >= best.form.length) onsetObs = Math.min(onsetObs, m.obs);
      candidates.push({
        cls: "wrong_word", word: left.word, last: left.word, obsFrom: from,
        obsTo: onsetObs > from ? onsetObs : left.obsTo, observed: best.form,
        supportRange: [from, to], pair: seq, pairOnset: true,
      });
    }
    // An added word at an ayah start sits in the greedy phones before word 1, after both seam words already
    // have a complete in-order visit. Pauses and waqf tails are not part of that word. Neighbours, their
    // prefixes and non-Quran phrases are not an addition.
    const phraseSpan = (s: string) => phrases.some((p) => {
      if (Math.abs(p.length - s.length) > Math.ceil(p.length * 0.6) + 4) return false;
      const hit = alignSubstring(s, 0, s.length, p, () => true);
      return !!hit && hit.start === 0 && hit.end >= s.length - 1 && hit.cost / UNIT / p.length <= PHRASE_MAX_COST;
    });
    for (const w1 of new Set(visits.map((v) => v.word))) {
      if (w1 < first || words[w1].end === words[w1].start || ayahFirst[ayahOf[w1]] !== w1) continue;
      const prev = prevWord(w1);
      if (prev < 0 || ayahOf[prev] === ayahOf[w1]) continue;
      const lefts = visits.filter((v) => v.word === prev && complete(v));
      const rights = visits.filter((v) => v.word === w1 && complete(v));
      let best: { left: Visit; right: Visit } | null = null;
      for (const right of rights) for (const left of lefts) {
        if (left.obsTo > right.obsFrom) continue;
        if (left.stepTo !== undefined && right.stepFrom !== undefined && right.stepFrom < left.stepTo) continue;
        if (!best || right.obsFrom < best.right.obsFrom || (right.obsFrom === best.right.obsFrom && left.obsTo > best.left.obsTo)) best = { left, right };
      }
      if (!best || best.right.obsFrom <= best.left.obsTo) continue;
      const raw = obs.slice(best.left.obsTo, best.right.obsFrom).filter((p) => !p.junk).map((p) => p.p).join("");
      const said = stripEdge(raw);
      if (!said || phraseSpan(said)) continue;
      const spelled = lexical.surplusWord(w1, said);
      if (!spelled) continue;
      const existing = issues.get(w1);
      if (existing && existing.cls !== "added_word") continue;
      issues.set(w1, {
        cls: "added_word", word: w1, last: w1, obsFrom: best.left.obsTo, obsTo: best.right.obsFrom, observed: said, lexicon: true,
      });
    }
    // A mushaf waqf sign inside an ayah, where the reciter actually stops. The previous word is read in its
    // pausal form and the pause is measured from the end of that token's own run, so one held frame of the
    // same class does not eat the silence. Ayah seams stay on the loop above.
    const formed = (s: string) => collapseHeld([...s]).join("");
    const vowel = new Set("َُِاۦۥ۪");
    const residueConsonant = new Set([...CONSONANTS, "ڇ", "ں", "۾"]);
    const inventory = [...index.tokens];
    const gapBlank = (from: number, to: number) => {
      if (!frames || frames.startFrame < 0 || from < frames.startFrame || to > frames.endFrame) return true;
      const owns = Math.log(0.5);
      for (let f = from; f < to; f++) if (frames.row(f)[0] < owns) return false;
      return true;
    };
    const argmax = (frame: number) => {
      const row = frames!.row(frame);
      let best = 0;
      for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
      return best;
    };
    /** The silence after `prev`, starting once that token's class is no longer the argmax. */
    const runEndPause = (prev: number) => {
      if (frameMs <= 0 || prev < 0 || prev + 1 >= obs.length) return false;
      let a = obs[prev].end + 1;
      const b = obs[prev + 1].start;
      if (frames && frames.startFrame >= 0 && a >= frames.startFrame && b <= frames.endFrame) {
        const id = tokenId.get(obs[prev].p);
        if (id) while (a < b && argmax(a) === id) a++;
      }
      return b - a >= minGapFrames(frameMs) && gapBlank(a, b);
    };
    const pausalVisit = (v: Visit) => {
      const iso = index.isolatedWordForms?.[v.word];
      if (!iso || v.obsFrom < 0 || v.obsTo <= v.obsFrom || v.stepFrom === undefined || v.stepTo === undefined) return false;
      return formed(heard(v)) === formed(iso) || formed(stripEdge(heard(v))) === formed(stripEdge(iso));
    };
    // Obs index where the stem (and its proclitic vowel, if heard) begins. Phones before that in the
    // visit were glued on from the surplus. Null when the visit is not this word minus its proclitic.
    const residueClear = (at: number) => {
      if (!frames || frames.startFrame < 0) return true;
      const a = obs[at].start, b = obs[at].end + 1;
      if (a < frames.startFrame || b > frames.endFrame) return true;
      for (let f = a; f < b; f++) {
        const id = argmax(f);
        if (id !== 0 && residueConsonant.has(inventory[id - 1] ?? "")) return false;
      }
      return true;
    };
    const procliticCut = (v: Visit): number | null => {
      if (!procliticOnset(v.word) || v.obsFrom < 0 || v.obsTo <= v.obsFrom || v.stepFrom === undefined || v.stepTo === undefined) return null;
      const heardRaw = heard(v);
      const tails: string[] = [];
      for (const full of [ref.slice(words[v.word].start, words[v.word].end), index.isolatedWordForms?.[v.word] ?? ""]) {
        if (full && "وفبكل".includes(full[0]!) && formed(heardRaw) !== formed(full)) tails.push(full.slice(1));
      }
      for (const tail of tails) {
        const stem = vowel.has(tail[0] ?? "") ? tail.slice(1) : tail;
        const wantTail = formed(tail), wantStem = formed(stem);
        if (!wantStem) continue;
        for (let i = 0; i < heardRaw.length; i++) {
          const suf = formed(heardRaw.slice(i));
          if (suf !== wantTail && suf !== wantStem) continue;
          const keepsVowel = suf === wantTail && vowel.has(tail[0] ?? "");
          if (keepsVowel && !residueClear(v.obsFrom + i)) return null;
          return v.obsFrom + i;
        }
      }
      return null;
    };
    for (const rightWord of new Set(visits.map((v) => v.word))) {
      if (rightWord < first || words[rightWord].end === words[rightWord].start || ayahFirst[ayahOf[rightWord]] === rightWord) continue;
      const prev = prevWord(rightWord);
      if (prev < 0 || ayahOf[prev] !== ayahOf[rightWord] || !waqfMarked(prev)) continue;
      const lefts = visits.filter((v) => v.word === prev && (complete(v) || pausalVisit(v)));
      const rights = visits.filter((v) => v.word === rightWord && (complete(v) || procliticCut(v) !== null));
      let best: { left: Visit; right: Visit } | null = null;
      for (const right of rights) for (const left of lefts) {
        if (left.obsTo > right.obsFrom) continue;
        if (left.stepTo !== undefined && right.stepFrom !== undefined && right.stepFrom < left.stepTo) continue;
        if (!best || right.obsFrom < best.right.obsFrom || (right.obsFrom === best.right.obsFrom && left.obsTo > best.left.obsTo)) best = { left, right };
      }
      if (!best) continue;
      // A waqf-tail phone can sit just outside the pausal visit. The silence is after that phone.
      const cut = complete(best.right) ? best.right.obsFrom : procliticCut(best.right)!;
      let pauseAt = best.left.obsTo - 1;
      while (pauseAt + 1 < cut && waqfEdge.has(obs[pauseAt + 1]!.p)) pauseAt++;
      if (cut <= best.left.obsTo || !runEndPause(pauseAt)) continue;
      const raw = obs.slice(best.left.obsTo, cut).filter((p) => !p.junk).map((p) => p.p).join("");
      const said = stripEdge(raw);
      if (!said || phraseSpan(said)) continue;
      const spelled = lexical.surplusWord(rightWord, said);
      if (!spelled) continue;
      const existing = issues.get(rightWord);
      if (existing && existing.cls !== "added_word") continue;
      issues.set(rightWord, {
        cls: "added_word", word: rightWord, last: rightWord, obsFrom: best.left.obsTo, obsTo: cut, observed: said, lexicon: true,
      });
    }
    // A spelled span can be lent to either neighbour by the greedy alignment.
    // Use visits only to bound the seam, then certify presence on the posteriors
    // left after excision. No complete visit or initial ownership is required.
    // Without certificates or a discovery callback, this path cannot emit.
    const inspectSeams = !!options.seamInsertion || insertionEvidence.size > 0;
    const seamPauses = [0], seamJunk = [0];
    const seamUtterances: { from: number; to: number }[] = [];
    let utteranceFrom = -1;
    for (const [key, value] of seamExcisionCache) if (value.until < obs[0].start) seamExcisionCache.delete(key);
    if (inspectSeams && frames && frames.startFrame >= 0) for (let i = 0; i < obs.length; i++) {
      seamPauses.push(seamPauses[i] + Number(runEndPause(i)));
      seamJunk.push(seamJunk[i] + Number(!!obs[i].junk));
      if (seamPauses[i + 1] > seamPauses[i]) {
        if (utteranceFrom >= 0) seamUtterances.push({ from: utteranceFrom, to: i + 1 });
        utteranceFrom = i + 1;
      }
    }
    seamUtterances.sort((a, b) => (b.to - b.from) - (a.to - a.from) || a.from - b.from);
    if (inspectSeams && frames && frames.startFrame >= 0) for (const w1 of new Set(visits.map(v => v.word))) {
      if (w1 < first || !length(w1) || decided.has(w1) || issues.get(w1)?.lexicon) continue;
      const prev = prevWord(w1), w2 = nextWord(w1);
      if (prev < 0 || w2 >= words.length || words[prev].surah !== words[w1].surah) continue;
      const inner = ayahOf[prev] === ayahOf[w1];
      if (inner ? !waqfMarked(prev) : ayahFirst[ayahOf[w1]] !== w1) continue;
      const lefts = visits.filter(v => v.word === prev && v.obsFrom >= 0 && v.obsTo > v.obsFrom);
      const ends = visits.filter(v => v.word === w2 && v.obsFrom >= 0 && v.obsTo > v.obsFrom);
      let found: Issue | undefined;
      for (const left of lefts) for (const end of ends) {
        if (found || left.obsTo > end.obsFrom) continue;
        const from = left.obsFrom, to = end.obsTo;
        const f0 = obs[from].start, f1 = obs[to - 1].end + 1;
        if (f0 < frames.startFrame || f1 > frames.endFrame) continue;
        if (inner && seamPauses[end.obsFrom] === seamPauses[from]) continue;
        if (!seamUtterances.some(s => s.from >= from && s.to <= end.obsFrom)) continue;
        const key = JSON.stringify([w1, f0, f1, obs.slice(from, end.obsFrom).map(p => [p.p, p.start, p.end, !!p.junk])]);
        const cached = seamExcisionCache.get(key);
        if (cached) {
          if (cached.span) found = {
            cls: "added_word", word: w1, last: w1, obsFrom: from + cached.span.from,
            obsTo: from + cached.span.to, observed: cached.span.said, lexicon: true,
          };
          continue;
        }
        const forms = [prev, w1, w2].map(wi => [...new Set([
          ref.slice(words[wi].start, words[wi].end), index.isolatedWordForms?.[wi] ?? "",
        ].filter(Boolean))]);
        // Longer spans first: a whole word takes precedence over its lexical tail.
        for (const { from: a, to: b } of seamUtterances) {
          if (found) break;
          if (a < from || b > end.obsFrom) continue;
          if (seamJunk[b] !== seamJunk[a]) continue;
          // Excision must take an entire pause-bounded utterance, not a fragment
          // of a word or a held run that the remaining frames can still explain.
          const said = stripEdge(surplusNasal(obs.slice(a, b).map(p => p.p).join("")));
          if (!said || !lexical.surplusWord(w1, said) || phraseSpan(said)) continue;
          if (lexical.fragmentOfSeam(w1, said)) continue;
          const margins = excisedNeighbourMargins(frames, p => tokenId.get(p) ?? 0, forms, f0, f1,
            [obs[a].start, obs[b - 1].end + 1]);
          if (!margins || margins.some(m => m < PHRASE_SWITCH)) continue;
          const preceding = prevWord(prev);
          const prefix = visits.filter(v => v.word === preceding && ayahOf[preceding] === ayahOf[prev] &&
            v.obsFrom >= 0 && v.obsTo <= left.obsFrom).sort((a, b) => b.obsTo - a.obsTo)[0];
          const candidate: SeamInsertionCandidate = { word: w1, said,
            fromFrame: obs[a].start, toFrame: obs[b - 1].end + 1, seamFromFrame: f0, seamToFrame: f1,
            prefixFrame: prefix && obs[prefix.obsFrom].start };
          const permitted = options.seamInsertion ? options.seamInsertion(candidate) :
            (insertionEvidence.get(insertionKey(words[w1], said, candidate.fromFrame, candidate.toFrame)) ?? 0) > 0.5;
          if (!permitted) continue;
          found = { cls: "added_word", word: w1, last: w1, obsFrom: a, obsTo: b, observed: said, lexicon: true };
          break;
        }
        seamExcisionCache.set(key, { until: f1, span: found ? {
          from: found.obsFrom - from, to: found.obsTo - from, said: found.observed!,
        } : null });
      }
      if (found) issues.set(w1, found);
    }
    for (const e of stepEdits.sort((a, b) => b.lo - a.lo)) steps.splice(e.lo, e.hi - e.lo, ...e.path);
    // The identical two-word lead plus a locked ayah end closes this choice. Score only the greedy
    // phones before its labelled pause, treating waqf tails and held sounds identically on both sides.
    // These candidates still need affirmative posterior support and the unchanged weak-evidence gate.
    if (rankOmission && trackerLocked) for (const slot of visits) {
      const wi = slot.word;
      const fillers = formulas.get(wi);
      if (!fillers || slot.obsFrom < 0 || slot.obsTo <= slot.obsFrom || slot.stepFrom === undefined) continue;
      if (wi < first || decided.has(wi)) continue;
      if (cursor !== wi && (lockedAyah.get(ayahOf[wi])?.wordTo ?? -1) < wi) continue;
      const left = visits.find((v) => v.word === wi - 1 && v.obsTo <= slot.obsFrom && counts(v) && !wrong(v));
      const lead = left && visits.find((v) => v.word === wi - 2 && v.obsTo <= left.obsFrom && counts(v) && !wrong(v));
      if (!left || !lead) continue;
      const from = slot.obsFrom;
      const next = visits.find((v) => v.obsFrom >= slot.obsTo && v.word > wi);
      const limit = next?.obsFrom ?? obs.length;
      let to = slot.obsTo;
      while (to < limit && !labelledPause(to - 1)) to++;
      if (!labelledPause(to - 1)) continue;
      if (obs.slice(from, to).some((p) => p.junk)) continue;
      const path = steps.slice(slot.stepFrom).filter((s) => "obs" in s && s.obs >= from && s.obs < to);
      if (path.length !== to - from || path.some((s) =>
        (s.op !== "diag" && s.op !== "ins") || index.wordAt(s.op === "ins" ? s.at : s.ref) !== wi)) continue;
      const said = stripEdge(obs.slice(from, to).map((p) => p.p).join(""));
      const phones = (s: string) => collapseHeld([...stripEdge(s)]);
      if (!said) continue;
      const written = ref.slice(words[wi].start, words[wi].end);
      let best = spanLogP(phones(said), phones(written), omissionChannel);
      let alternative: string | null = null;
      for (const filler of fillers) {
        const score = spanLogP(phones(said), phones(filler), omissionChannel);
        if (score > best) { best = score; alternative = filler; }
      }
      if (!alternative) continue;
      candidates.push({
        cls: "wrong_word", word: wi, last: wi, obsFrom: from, obsTo: to, observed: said,
      });
    }
    // A twin that only drops the madd before a hamza. Waqf still removes that vowel, so a labelled pause stays
    // quiet. Everywhere else the span has to be shorter than this reciter's own matched madd munfasil.
    for (const m of maddNow(steps).flags) {
      if (issues.has(m.word) || decided.has(m.word)) continue;
      const drop = madd.drops.get(m.word)!;
      const visit = visits.find((v) => v.word === m.word && v.obsFrom >= 0 && v.obsFrom <= m.consObs && v.obsTo > m.consObs);
      issues.set(m.word, {
        cls: "wrong_word", word: m.word, last: m.word, obsFrom: visit?.obsFrom ?? m.consObs, obsTo: m.hamzaObs,
        observed: drop.alt, madd: true,
      });
      right.delete(m.word);
    }
    // Phones from the previous word's last in-order visit up to the first labelled pause.
    // A gap inside that previous word is already before `start`.
    const closingSlot = (wi: number, fallbackFrom: number): { from: number; to: number } | null => {
      if (ayahLast[ayahOf[wi]] !== wi) return null;
      const ayah = ayahOf[wi];
      const prev = prevWord(wi);
      let anchored = -1;
      for (const v of visits) {
        if (v.word !== prev || v.obsFrom < 0 || v.obsTo <= anchored) continue;
        // A repeat after a later word is not the in-order visit.
        if (visits.some((u) => u.obsFrom >= 0 && u.word > prev && u.obsTo > 0 && u.obsTo <= v.obsFrom)) continue;
        anchored = v.obsTo;
      }
      const start = anchored >= 0 ? anchored : fallbackFrom;
      if (start < 0 || start > obs.length) return null;
      let limit = obs.length;
      for (const v of visits) {
        if (v.obsFrom < start || ayahOf[v.word] === ayah) continue;
        if (v.obsFrom < limit) limit = v.obsFrom;
      }
      if (limit <= start) return null;
      // The first labelled pause after the previous word. Speech before it is the slot; a pause
      // earlier than that word is already outside `start`. Unowned phones after the pause (the
      // next ayah's onset, not yet a visit) stay out.
      let to = -1;
      for (let i = start; i < limit; i++) {
        if (PAUSE_TOKEN.has(obs[i]!.p)) { to = i; break; }
        if (i + 1 < obs.length && runEndPause(i)) { to = i + 1; break; }
      }
      return to > start ? { from: start, to } : null;
    };
    const slotText = (from: number, to: number) =>
      obs.slice(from, to).filter((p) => !p.junk && !PAUSE_TOKEN.has(p.p)).map((p) => p.p);
    const mannerContradicts = (wi: number, phones: string[]) => {
      const heard = lastManner(phones.join(""));
      const written = pausalManner(wi);
      return !!heard && !!written && heard !== written;
    };
    for (const [wi, issue] of [...issues]) {
      if (issue.phrase || issue.word !== wi) continue;
      if (issue.cls !== "wrong_word" && issue.cls !== "omitted_word") continue;
      const slot = closingSlot(wi, issue.obsFrom);
      if (!slot) continue;
      const phones = slotText(slot.from, slot.to);
      if (!mannerContradicts(wi, phones)) continue;
      if (issue.cls === "wrong_word" && issue.last === wi) {
        issue.manner = true;
        continue;
      }
      // An omission already raised on this word. Speech in the slot (two non-blank tokens, one a
      // consonant) that contradicts the pausal final becomes that same accusation as a wrong word.
      if (issue.cls !== "omitted_word" || issue.word !== wi) continue;
      const consonant = phones.some((p) => CONSONANTS.has(p) || MANNER[p] === "nasal");
      if (phones.length < 2 || !consonant) continue;
      issues.set(wi, {
        cls: "wrong_word", word: wi, last: wi, obsFrom: slot.from, obsTo: slot.to,
        observed: phones.join(""), stringPath: true, manner: true,
      });
    }
    return { issues, lexical: candidates, head, furthest: reached, steps, right, bound: new Set(last.keys()), visits: visits.filter(counts) };
  };

  /** Matched phonemes on the word after an error, heard after it: the next word has begun. */
  const onset = (steps: Step[], issue: Issue) => {
    // A parallel phrase is only raised once its shared exit word has matched, which is already past the
    // interior word. The immediate next word can be the suffix the two phrases disagree on, so exact
    // phoneme hits there are the wrong test.
    if (issue.phrase) return true;
    const next = nextWord(issue.last);
    // After the Quran's last word there is no next word to begin: decide at the passage end instead.
    if (next >= words.length) return false;
    let n = 0;
    for (const s of steps) if (s.op === "diag" && s.obs >= issue.obsTo && index.wordAt(s.ref) === next && ref[s.ref] === obs[s.obs].p) n++;
    return n >= Math.min(ONSET, words[next].end - words[next].start);
  };

  /**
   * After going back, a jump forward that lands on word `landing` came back if it lands on the furthest ayah read
   * before going back or later. Up to where reading on from the furthest word goes (`rejoinAt`) it is a rejoin: the
   * reciter went on from where they had been, and nothing between is left out. Past it, the words between were never
   * read. Either way the pass then current goes on: a revision read through does not make the words read before it
   * unread.
   */
  const comesBack = (landing: number) => !!wentBack && landing >= ayahFirst[ayahOf[wentBack.furthest]];
  /**
   * Where reading on from word `far` goes: the word after it, or the next ayah's first word when `far` is within a
   * word of its ayah's end (the last word's lock or visit can lag behind the reciter).
   */
  const rejoinAt = (far: number) => {
    const last = ayahLast[ayahOf[far]];
    return far >= prevWord(last) && last + 1 < words.length ? nextWord(last) : nextWord(far);
  };
  const cameBack = () => {
    if (!wentBack) return;
    if (readNotSkipped && passFromMs > wentBack.pass) passFromMs = wentBack.pass;
    returning = null;
    wentBack = null;
  };
  /**
   * refinements.readNotSkipped: the words this alignment read, and where the reciter went back and came forward again.
   * A visit that heard its word right reads it when the path goes on to the next word, and an ayah's last word also
   * when the path goes back from it (a repeat). A right visit followed by anything else is the alignment's edge before
   * a jump, not reading. A word now accused of being said wrong was not read right. A path that steps back to an
   * earlier ayah is a return (`goneBack`); one `PASS_BACK_WORDS` words or more behind the furthest word read, or to a
   * surah's first ayah, read through (its last word read, as above), starts a new pass where it went back. A jump
   * forward after a return comes back (`cameBack`); reading on past the furthest word ends the return. Visits are
   * scanned again as the alignment grows: each step is taken in the order heard, and the state it leaves is the same on
   * a second scan.
   */
  const noteRead = (visits: Visit[], issues: Map<number, Issue>) => {
    const heardAt = (v: Visit) => (v.obsFrom >= 0 && v.obsFrom < obs.length ? obs[v.obsFrom].start * frameMs : -1);
    /** Visit `k` read its word: heard right, and the path went on to the next word (or back, from an ayah's last). */
    const reads = (k: number) => {
      const [v, n] = [visits[k], visits[k + 1]];
      if (!n || wrong(v) || issues.get(v.word)?.cls === "wrong_word") return false;
      return n.word === nextWord(v.word) || (v.word === ayahLast[ayahOf[v.word]] && n.word <= v.word);
    };
    let reach = -1;
    for (let k = 1; k < visits.length; k++) {
      const [b, v] = [visits[k - 1], visits[k]];
      if (!wrong(b)) reach = Math.max(reach, b.word);
      const at = heardAt(v);
      if (at < 0) continue;
      if (ayahOf[b.word] > ayahOf[v.word] && words[b.word].surah === words[v.word].surah) {
        if (at > (wentBack?.at ?? passFromMs)) goneBack(v.word, reach, at);
      } else if (wentBack && at > wentBack.at) {
        if (v.word > nextWord(b.word) && comesBack(v.word)) cameBack();
        else if (v.word > wentBack.furthest) wentBack = null;
      }
      if (at <= passFromMs) continue;
      if (returning && at >= returning.at && ayahOf[v.word] !== returning.ayah) returning = null;
      if (returning && at >= returning.at && v.word === ayahLast[returning.ayah] && reads(k)) {
        newPass(returning.at);
        returning = null;
      }
    }
    for (let k = 0; k + 1 < visits.length; k++) {
      const v = visits[k];
      if (heardAt(v) < passFromMs || !reads(k)) continue;
      const at = heardAt(v);
      readRight.set(v.word, at);
      const first = firstRead.get(v.word);
      if (first === undefined || first < passFromMs || at < first) firstRead.set(v.word, at);
    }
  };
  /**
   * refinements.readNotSkipped: the furthest word read right in the current pass before `ms`. A word whose first
   * reading in the pass was noted before the pass moved on counts by its latest reading.
   */
  const readTo = (ms: number) => {
    let far = -1;
    for (const [w, first] of firstRead) {
      const at = first >= passFromMs ? first : readRight.get(w)!;
      if (w > far && at >= passFromMs && at < ms) far = w;
    }
    return far;
  };
  /** refinements.readNotSkipped: every spoken word of `[from, to]` was read in the current pass. */
  const allRead = (from: number, to: number) => {
    for (let w = from; w <= to; w++) if (words[w].end > words[w].start && (readRight.get(w) ?? -1) < passFromMs) return false;
    return true;
  };
  /**
   * refinements.readNotSkipped: the reading after omission `issue` rejoins. It resumes where reading on from the
   * furthest word read before it in the pass goes (`rejoinAt`), so the words between were read already; or it begins an
   * ayah's first word after the reciter broke off at the omission: a pause or a phrase that is not the Quran's between
   * the speech either side of it, or no speech before it in the window. A resumption inside an ayah, after a pause or
   * not, does not rejoin (a hesitation, then a word left out), nor one at an ayah's first word in continuous speech.
   * Nor does one that leaves whole ayahs out short of where the reciter had read to: an omission that begins at an
   * ayah's first word follows an ayah read to its end and the pause every ayah ends with, not a reciter breaking off, so
   * going on at an ayah's first word before the ayah that reading on from the furthest word goes into (that ayah again
   * when it was read to its end) leaves the ayahs between out of the re-reading.
   */
  const rejoins = (issue: Issue, visits: Visit[]) => {
    const from = Math.max(0, issue.obsFrom);
    const right = visits.find((v) => v.word > issue.last && v.obsFrom >= from && v.obsFrom < obs.length);
    if (!right || frameMs <= 0) return false;
    const far = readTo(obs[right.obsFrom].start * frameMs);
    if (far >= issue.last && right.word === rejoinAt(far)) return true;
    if (right.word !== ayahFirst[ayahOf[right.word]]) return false;
    if (issue.word === ayahFirst[ayahOf[issue.word]]) {
      if (far < 0) return false;
      // The ayah that reading on from the furthest word goes into (past the Quran's last word, the one after it).
      const on = rejoinAt(far);
      if (ayahOf[right.word] < (on < words.length ? ayahOf[on] : ayahOf[far] + 1)) return false;
    }
    if (issue.obsFrom <= 0) return true;
    for (let k = issue.obsFrom - 1; k < right.obsFrom && k + 1 < obs.length; k++) {
      if (obs[k].junk || obs[k + 1].junk || obs[k + 1].start - obs[k].end - 1 >= minGapFrames(frameMs)) return true;
    }
    return false;
  };

  /** refinements.stutter: consonants in order, a doubled one once, a ghunna mark as its nasal. */
  const consonantSkeleton = (phones: string) => {
    let s = "";
    for (const p of phones) {
      const c = p === "ں" ? "ن" : p === "۾" ? "م" : p;
      if (CONSONANTS.has(c) && s.at(-1) !== c) s += c;
    }
    return s;
  };
  /**
   * refinements.stutter: `heard` is `written` with one block of two or more sounds, a consonant among them, said twice in
   * a row (a restart inside the word): without one copy it has the written word's consonants exactly, in order. A
   * different word changes those consonants, so this cannot pass one off as the written word. A block said twice
   * makes the word at most twice as long as written; anything longer is not a restart of it, and is not scanned (the
   * scan is cubic in the heard length).
   */
  const stutterOf = (heard: string, written: string): boolean => {
    const target = consonantSkeleton(written);
    if (!target || heard.length > 2 * written.length) return false;
    for (let k = 2; 2 * k <= heard.length; k++) {
      for (let i = 0; i + 2 * k <= heard.length; i++) {
        const block = heard.slice(i, i + k);
        if (block !== heard.slice(i + k, i + 2 * k) || ![...block].some((p) => CONSONANTS.has(p))) continue;
        if (consonantSkeleton(heard.slice(0, i) + heard.slice(i + k)) === target) return true;
      }
    }
    return false;
  };

  /**
   * refinements.omissionDue: the reciter has read on past a suspected omission. The confirmation delay has passed since it
   * was suspected, the alignment ends after the gap, two words after it were heard right and both began after the
   * suspicion (a pause at the gap can still end in going back to say what was left out), and the omission is sure.
   */
  const readOnPast = (o: Open, visits: Visit[]): boolean => {
    if (confirm.after === "delay" && nowMs - o.sinceMs < confirm.ms) return false;
    const tail = visits.at(-1);
    if (!tail || tail.word <= o.issue.last) return false;
    const after = new Set<number>();
    for (const v of visits) {
      if (v.word <= o.issue.last || v.obsFrom < 0 || obs[v.obsFrom].start * frameMs < o.sinceMs || wrong(v)) continue;
      after.add(v.word);
    }
    return after.size >= 2 && sureOf(o);
  };

  /**
   * Parallel phrases the alignment has entered and left: the written phonemes against each attested substitute,
   * over the frames from the shared entry word to the shared exit word. Suffix frames sit in that one span, so
   * they cannot be lent to a neighbour and counted again. The best path that clears `PHRASE_SWITCH` is a wrong word.
   */
  const parallelCandidates = (aligned: Step[]): Issue[] => {
    if (!frames || frames.startFrame < 0 || !parallels.size) return [];
    let loW = Infinity;
    let hiW = -1;
    for (const s of aligned) if (s.op === "diag") {
      const wi = index.wordAt(s.ref);
      if (wi < loW) loW = wi;
      if (wi > hiW) hiW = wi;
    }
    if (hiW < loW) return [];
    const ids = (s: string) => [...s].map((p) => tokenId.get(p) ?? 0);
    const covered = (start: number, length: number): { from: number; to: number } | null => {
      const w0 = words[start].start;
      const w1 = words[start + length - 1].end;
      const entry = start;
      const exit = start + length - 1;
      let from = -1;
      let to = -1;
      let entryDiag = false;
      let exitDiag = false;
      let run = false;
      const reset = () => { from = -1; to = -1; entryDiag = false; exitDiag = false; run = false; };
      for (const s of aligned) {
        if (s.op === "jump" || s.op === "lead") {
          if (exitDiag) break;
          reset();
          continue;
        }
        const refAt = s.op === "ins" ? s.at : s.ref;
        if (refAt < w0 || refAt >= w1) {
          if (exitDiag) break;
          if (run) reset();
          continue;
        }
        run = true;
        const wi = index.wordAt(Math.min(refAt, w1 - 1));
        if (s.op === "diag" && wi === entry) entryDiag = true;
        if (s.op === "diag" && wi === exit) exitDiag = true;
        if ("obs" in s && entryDiag) {
          if (from < 0) from = s.obs;
          to = s.obs + 1;
        }
      }
      return entryDiag && exitDiag && to > from ? { from, to } : null;
    };
    const collapse = (t: string) => collapseHeld([...t]).join("");
    /** Mean decoder confidence over the heard phones that miss the written phrase and match the parallel. */
    const switchedClarity = (path: { written: string; alt: string }, span: { from: number; to: number }) => {
      const heard = obs.slice(span.from, span.to);
      const said = heard.map((p) => p.p).join("");
      const wCol = collapse(path.written), aCol = collapse(path.alt), sCol = collapse(said);
      const off = new Set(editPath(path.written, said).flatMap((o) => (o.op === "sub" || o.op === "ins" ? [o.obsAt] : [])));
      const on = editPath(path.alt, said).flatMap((o) => (o.op === "match" && off.has(o.obsAt) ? [heard[o.obsAt].conf] : []));
      if (on.length) return on.reduce((a, c) => a + c, 0) / on.length;
      // Pure-deletion parallel (e.g. alayhima -> alayhi) has no substituted phones; require the decode to prefer alt.
      const edits = (a: string, b: string) => editPath(a, b).filter((o) => o.op !== "match").length;
      return edits(aCol, sCol) < edits(wCol, sCol) ? 1 : 0;
    };
    const out: Issue[] = [];
    const taken = new Set<number>();
    for (let w = loW; w <= hiW; w++) {
      const paths = parallels.get(w);
      if (!paths || taken.has(w) || decided.has(w)) continue;
      let best: { path: (typeof paths)[number]; span: { from: number; to: number }; margin: number; need: number } | null = null;
      for (const path of paths) {
        const span = covered(path.start, 4);
        if (!span) continue;
        const f0 = obs[span.from].start;
        const f1 = obs[span.to - 1].end + 1;
        if (f1 - f0 < 4 || f0 < frames.startFrame || f1 > frames.endFrame) continue;
        const altNll = frames.nll(ids(collapse(path.alt)), f0, f1);
        if (!Number.isFinite(altNll)) continue;
        // Blank-owned frames say nothing about which phrase was said. A flat phone frame (the whole inventory
        // sharing the mass) is not a heard word either.
        let speech = 0;
        let heard = 0;
        const blankOwns = Math.log(0.5);
        for (let f = f0; f < f1; f++) {
          const row = frames.row(f);
          if (row[0] >= blankOwns) continue;
          let top = row[1];
          for (let i = 2; i < row.length; i++) if (row[i] > top) top = row[i];
          speech += Math.exp(top);
          heard++;
        }
        if (heard < 4 || speech / heard < PHRASE_PEAK) continue;
        const margin = frames.nll(ids(collapse(path.written)), f0, f1) - altNll;
        // The attested-parallel discount is for a substitute heard surely. Where the phones that leave the written
        // phrase for this one were decoded below `UNSURE` (ل heard as ن on a phone band), it is the confusable
        // sound `supportNeed` doubles for, and it takes that same bar.
        const need = switchedClarity(path, span) < UNSURE ? 2 * support : PHRASE_SWITCH;
        if (!(margin >= need) || (best && margin <= best.margin)) continue;
        best = { path, span, margin, need };
      }
      if (!best) continue;
      taken.add(w);
      out.push({
        cls: "wrong_word", word: w, last: w, obsFrom: best.span.from, obsTo: best.span.to, observed: best.path.altWord,
        supportRange: [best.span.from, best.span.to], phrase: { written: best.path.written, alt: best.path.alt, need: best.need },
      });
    }
    return out;
  };

  const evaluate = (out: CheckerEvent[], final: boolean) => {
    // The session ended, or the cursor left the span, before a unique continuation arrived.
    // Frozen slot accusations were judged on the committed window; this copy is the one they name.
    if (watching.size && (final || (cursor >= 0 && !spanStillOpen(cursor)))) {
      resolveWatching();
      releaseFrozen(out, false);
    }
    if (anchor === null || !obs.length) return;
    setWindow();
    steps = align();
    const judged = judge(steps);
    const beforeGap = pendingGap;
    noteLock();
    const issues = judged?.issues ?? new Map<number, Issue>();
    if (beforeGap && (!pendingGap || beforeGap.left !== pendingGap.left || beforeGap.right !== pendingGap.right)) {
      closeGap(out, issues, beforeGap.left, beforeGap.right, true);
    }
    if (pendingGap) closeGap(out, issues, pendingGap.left, pendingGap.right, final);
    keepLedger(issues);
    for (const [wi, issue] of [...followGaps]) {
      if (decided.has(wi)) followGaps.delete(wi);
      else if (!issues.has(wi)) issues.set(wi, issue);
    }
    // One unread run, two emitters. The seam starts after the ledger ayah and the words between them are bound
    // refrain or shared duplicate-span text (or there are none). The ledger key is the earlier one, so the seam
    // is dropped. A unique word between them keeps both. The span is the one the fit already accepted, including
    // a skip confirmed before this window's anchor.
    const bound = judged?.bound ?? new Set<number>();
    if (ledgerSpans.size) for (const [wi, issue] of [...issues]) {
      if (issue.cls !== "skipped_passage" || ledgerSpans.has(wi)) continue;
      for (const [, last] of ledgerSpans) {
        if (issue.word <= last) continue;
        let same = true;
        for (let w = last + 1; w < issue.word; w++) {
          if (words[w].end === words[w].start) continue;
          if (!bound.has(w) || !copyText.has(w)) { same = false; break; }
        }
        if (same) { issues.delete(wi); break; }
      }
    }
    if (!judged && !issues.size) return;
    if (judged) steps = judged.steps;
    const head = judged?.head ?? cursor;
    const reached = judged?.furthest ?? cursor;
    const right = judged?.right ?? new Set<number>();
    // Use the final (possibly repaired) path and identical frames/context for both acoustic hypotheses. Unlike the
    // legacy path, a lexical candidate needs affirmative support; absent or indeterminate posteriors mean abstain.
    if (judged) for (const candidate of judged.lexical) if (supported(candidate, true) === true) {
      issues.set(candidate.word, candidate);
      right.delete(candidate.word);
    }
    for (const candidate of parallelCandidates(steps)) {
      const existing = [...issues.values()].find((i) => i.word <= candidate.word && candidate.word <= i.last);
      if (existing && !(existing.cls === "omitted_word" && existing.word === candidate.word && existing.last === candidate.word)) continue;
      if (supported(candidate, true) !== true) continue;
      issues.set(candidate.word, candidate);
      right.delete(candidate.word);
    }
    // Only the support-path addition is removed. Keep the wrong word's decision evidence and report the
    // combined heard span, so absorbing a tail cannot promote an uncertain wrong word into a warning.
    if (judged) for (const [wi, added] of [...issues]) {
      if (added.cls !== "added_word" || added.lexicon || decided.has(wi)) continue;
      const wrong = issues.get(prevWord(wi));
      if (!wrong || wrong.cls !== "wrong_word") continue;
      const location = toRef(wrong.word);
      const confirmed = decided.has(wrong.word) ? emitted.find(e => e.state === "confirmed" && e.errorClass === "wrong_word" &&
        e.location.surah === location.surah && e.location.ayah === location.ayah && e.location.word === location.word) : undefined;
      // An uncertain decision is not an accusation that can own another word's speech.
      if (decided.has(wrong.word) && !confirmed) continue;
      const union = swapTailUnion(wrong, added, {
        nextWord, visits: judged.visits, labelledPause,
        heard: span => heardSpan({ ...wrong, ...span }), spellsWord: lexical.surplusWord,
      });
      if (!union) continue;
      issues.set(wrong.word, { ...wrong, heardUnion: union });
      issues.delete(wi);
      if (confirmed) confirmed.observed = union.observed;
    }
    // An addition after the Quran's last word (past the end of An-Nas) has no following word to sit before, so it
    // cannot be located or reported; without this its index is out of range and the session dies.
    for (const wi of [...issues.keys()]) if (wi >= words.length) issues.delete(wi);
    // refinements.stutter: the written word restarted inside itself is the written word, not a wrong one.
    if (stutterRule) for (const [wi, issue] of [...issues]) {
      if (issue.cls !== "wrong_word" || !issue.stringPath || issue.word !== issue.last) continue;
      if (stutterOf(heardSpan(issue), ref.slice(words[wi].start, words[wi].end))) issues.delete(wi);
    }
    // refinements.readNotSkipped: words the reciter already read right in this pass, where the reading after them
    // rejoins, were not left out: a repeat rejoined after them.
    if (readNotSkipped) {
      if (judged) noteRead(judged.visits, issues);
      // refinements.repeatNotWrong: a long wrong word whose heard text reads the text just before it again, charged to
      // words read right earlier in this pass where the reading after them rejoins, was a restart: nothing was said
      // wrong and nothing left out. Any other stays a wrong word.
      if (repeatNotWrong) for (const [wi, issue] of [...issues]) {
        if (issue.cls !== "wrong_word" || decided.has(wi) || followGaps.has(wi) || !allRead(issue.word, issue.last)) continue;
        const heard = heardSpan(issue);
        if (heard.length < Math.max(REPEAT_MIN, 2 * (words[issue.last].end - words[issue.word].start))) continue;
        const back = words[ayahFirst[Math.max(0, ayahOf[issue.word] - 2)]].start;
        const from = declared && passage && !latched ? Math.max(back, words[passage.from].start) : back;
        const to = words[issue.word].start;
        if (to <= from) continue;
        const fit = alignSubstring(ref, from, to, heard, (o) => wordStarts.has(o));
        if (!fit || fit.cost / UNIT / heard.length >= FITS_PASSAGE) continue;
        // Read as words left out (the heard text is the restart's, before them): they rejoin as a repeat's would.
        if (rejoins({ ...issue, cls: "omitted_word", obsFrom: issue.obsTo }, judged?.visits ?? [])) issues.delete(wi);
      }
      for (const [wi, issue] of [...issues]) {
        if ((issue.cls === "omitted_word" || issue.cls === "skipped_passage") && !followGaps.has(wi) && allRead(issue.word, issue.last) &&
          rejoins(issue, judged?.visits ?? [])) issues.delete(wi);
      }
    }
    // refinements.marksInPassage: a mark wholly outside the declared passage is on a word that was not asked for.
    if (marksInPassage && declared && passage && !latched) {
      for (const [wi, issue] of [...issues]) if (issue.word > passage.to || issue.last < passage.from) issues.delete(wi);
    }
    // refinements.markWhenSure: the take ended a word or two short of the declared passage's end, inside its last ayah,
    // by the alignment and the tracker's place both: those words were left out (a stop further back is a stop, and
    // what follows it is not marked).
    if (final && markWhenSure && declared && passage && !latched && judged?.visits.length && cursor >= 0) {
      const said = Math.max(cursor, ...judged.visits.map((v) => v.word), ...[...issues.values()].map((x) => x.last));
      const left: number[] = [];
      for (let k = said + 1; k <= passage.to; k++) if (words[k].end > words[k].start) left.push(k);
      if (said >= passage.from && left.length >= 1 && left.length <= ENDING_LEFT && ayahOf[said] === ayahOf[passage.to] &&
        !left.some((k) => open.has(k) || decided.has(k) || issues.has(k))) {
        issues.set(left[0], { cls: "omitted_word", word: left[0], last: left.at(-1)!, obsFrom: obs.length, obsTo: obs.length });
      }
    }
    furthest = Math.max(furthest, reached);
    readEnds = (judged?.visits ?? []).filter((v) => !wrong(v) && v.obsTo > v.obsFrom && v.obsFrom >= 0).map((v) => ({ word: v.word, end: obs[v.obsTo - 1].end }));
    // Read on past where the reciter had been before going back: the return is over.
    if (wentBack && reached > wentBack.furthest) wentBack = null;
    for (const [wi, o] of [...open]) {
      const now = issues.get(wi);
      if (!now) {
        // Commit moved the anchor past a held slot. The accusation was made on the locked-step window;
        // keep it until the continuation, instead of dropping it because this window no longer covers it.
        if (heldSlot(wi) && wi < anchor!) {
          freezeSlot(wi, o);
          continue;
        }
        open.delete(wi);
        out.push(event(o, "resolved"));
        ledgerSpans.delete(wi);
      } else {
        if (!sameEvidence(o.issue, now)) [o.stableMs, o.stood] = [nowMs, undefined];
        const moved = now.cls !== o.issue.cls;
        o.issue = now;
        if (moved) out.push(event(o, "suspected"));
      }
    }
    for (const [wi, issue] of issues) {
      if (open.has(wi) || decided.has(wi)) continue;
      if (issue.cls === "skipped_passage" && overlapsConfirmedSkip(issue.word, issue.last)) continue;
      const o: Open = { id: `${words[wi].surah}:${words[wi].ayah}:${words[wi].word}#${++seq}`, issue, sinceMs: nowMs, stableMs: nowMs };
      open.set(wi, o);
      out.push(event(o, "suspected"));
    }
    for (const o of [...open.values()]) {
      const { issue } = o;
      const omission = issue.cls === "omitted_word" || issue.cls === "skipped_passage";
      // The swallowed next word never advances `head`. The joint fit already heard it; wait until a later word or
      // the passage end so earlier audio can still resolve the suspicion.
      const followed = issue.pairOnset || onset(steps, issue);
      const past = head !== issue.word || (issue.pairOnset === true && final);
      // refinements.reanchorBack: not while the tracker, gone back to an earlier ayah, is still before the omission.
      // Until the window follows a return, it can read the return's first words as words two ayahs ahead. A tracker
      // that only holds or lags behind a skip does not wait it.
      const trackerBack = cursor >= 0 && cursor < issue.word && ayahOf[cursor] < ayahOf[cursorFar];
      const due = omission
        ? final || (ayahOf[reached] >= ayahOf[issue.last] + 2 && !(reanchorBack && trackerBack))
        : past &&
          followed &&
          (confirm.after === "next-word" || nowMs - o.sinceMs >= confirm.ms || (final && declared && (!latched || (policy.finaliseFree ?? true)) && [...right].some((wi) => wi > issue.last)));
      // A slot the two copies disagree on stays open until the continuation picks a copy.
      if (heldSlot(issue.word)) continue;
      if (due) decide(o, out);
      // The session ended first. A declared passage (or a latched free one, with `finaliseFree`) confirms once a later
      // word was recited right; otherwise it ends uncertain. refinements.markWhenSure: in a declared passage, a mark
      // already due is decided on the evidence it was due on, and a mistake where the reading stopped (`endsReading`)
      // on its evidence: no more speech will change it, so it need not stand, but it must be sure (`standsSure`).
      else if (final && markWhenSure && declared && !latched && (o.stood || (endsReading(issue, right) && standsSure(o)))) decide(o, out);
      else if (final) decide(o, out, "uncertain");
      // refinements.omissionDue: the reciter has read on past it; say so now, ahead of the two-ayah confirmation.
      else if (omission && omissionDue && !o.due && declared && !latched && judged && readOnPast(o, judged.visits)) {
        o.due = true;
        out.push(event(o, "suspected"));
      }
      // refinements.markWhenSure: all its confirmation asks but the wait (read past, its next word begun, sure), with its
      // evidence standing `STAND_MS` in place of the wait; say so now, ahead of the confirmation.
      else if (!omission && markWhenSure && !o.due && declared && !latched && past && followed && nowMs - o.stableMs >= STAND_MS && standsSure(o)) {
        o.due = true;
        o.stood = true;
        out.push(event(o, "suspected"));
      }
    }
    if (latched && passage) {
      passage.from = Math.max(passage.from, ayahFirst[Math.max(0, ayahOf[anchor] - 1)]);
      passage.to = Math.max(passage.to, furthest, anchor);
    }
    // Same commit a locked step uses, including while a span is held. Skipping it left `lead` set and the
    // whole opening in one alignment, so the word-1 gap the spelled-surplus rule reads was no longer that gap.
    if (!final && obs.length > MAX_WINDOW) commit(steps, out);
  };

  /** Freeze the window's older part at a word boundary, deciding errors left behind it. */
  const commit = (steps: Step[], out: CheckerEvent[]) => {
    const limit = obs.length - KEEP;
    let cut = -1;
    let at = -1;
    let cutStep = -1;
    const boundary = (k: number) => {
      const [s, next] = [steps[k], steps[k + 1]];
      return s.op === "diag" && wordStarts.has(s.ref + 1) && next?.op === "diag" && next.ref === s.ref + 1;
    };
    for (let k = 0; k < steps.length; k++) {
      const s = steps[k];
      if (s.op !== "diag" || s.obs >= limit) continue;
      if (boundary(k)) {
        cut = s.obs + 1;
        at = s.ref + 1;
        cutStep = k;
      }
    }
    if (cut < 0) return;
    // refinements.commitAfterJump: the kept path, aligned again from the cut word, must stay inside the window that
    // starts an ayah before it. One that goes back behind it (a restart) moves the cut past the restart's landing.
    if (commitAfterJump) {
      const windowFrom = (offset: number) => words[ayahFirst[Math.max(0, ayahOf[index.wordAt(offset)] - 1)]].start;
      // The lowest reference phoneme the path reads from each step on.
      const keptFrom = new Array<number>(steps.length + 1).fill(Infinity);
      for (let k = steps.length - 1; k >= 0; k--) {
        const s = steps[k];
        keptFrom[k] = s.op === "diag" || s.op === "del" ? Math.min(s.ref, keptFrom[k + 1]) : keptFrom[k + 1];
      }
      if (keptFrom[cutStep + 1] < windowFrom(at)) {
        // The restart's landing: the first step after the cut that reads behind the new window.
        const behind = windowFrom(at);
        let landing = cutStep + 1;
        for (; landing < steps.length; landing++) {
          const s = steps[landing];
          if ((s.op === "diag" || s.op === "del") && s.ref < behind) break;
        }
        // A restart reads again the text it lands on: the first speech the path places from its landing on fits that
        // text, as a lock's words fit a passage. A path that reaches behind only to place speech from elsewhere (the
        // reciter gone back further than the window, onto a refrain's earlier copy) is no restart, and the cut stays
        // where it was. Too little speech to tell is taken as a restart.
        const restart = (() => {
          const land = (steps[landing] as { ref: number } | undefined)?.ref;
          const from = steps.slice(landing).find((s): s is Extract<Step, { obs: number }> => s.op === "diag" || s.op === "ins")?.obs;
          if (land === undefined || from === undefined || obs.length - from < REREAD_MIN) return true;
          const heard = obs.slice(from, from + 2 * REREAD_MIN).map((p) => p.p).join("");
          const fit = alignSubstring(ref, land, Math.min(ref.length, land + 2 * heard.length), heard, (o) => o === land);
          return !!fit && fit.cost / UNIT / heard.length < FITS_PASSAGE;
        })();
        let moved = false;
        for (let k = cutStep + 1; restart && k < steps.length && !moved; k++) {
          const s = steps[k];
          if (s.op !== "diag") continue;
          if (obs.length - (s.obs + 1) < AFTER_DROP) break;
          if (!boundary(k) || keptFrom[k + 1] < windowFrom(s.ref + 1)) continue;
          [cut, at, moved] = [s.obs + 1, s.ref + 1, true];
        }
        if (!moved && restart && obs.length < 2 * MAX_WINDOW) return;
      }
    }
    const floor = obs[cut]!.start;
    for (const b of maddNow(steps).baseline) if (b.frame < floor && !pastMadd.some((p) => p.frame === b.frame)) pastMadd.push(b);
    const word = index.wordAt(at);
    for (const o of [...open.values()]) if (o.issue.word < word && heldSlot(o.issue.word)) freezeSlot(o.issue.word, o);
    for (const o of [...open.values()]) if (o.issue.word < word && !heldSlot(o.issue.word)) decide(o, out);
    for (const o of open.values()) {
      const range = o.issue.supportRange;
      const union = o.issue.heardUnion;
      o.issue = { ...o.issue, obsFrom: Math.max(0, o.issue.obsFrom - cut), obsTo: Math.max(0, o.issue.obsTo - cut),
        ...(union ? { heardUnion: { ...union, obsFrom: Math.max(0, union.obsFrom - cut), obsTo: Math.max(0, union.obsTo - cut) } } : {}),
        ...(range ? { supportRange: [Math.max(0, range[0] - cut), Math.max(0, range[1] - cut)] as [number, number] } : {}) };
    }
    obs = obs.slice(cut);
    anchor = word;
    lead = 0;
    fromLock = false;
    opening = false;
  };

  const track = (events: EngineEvent[], out: CheckerEvent[]): number | null => {
    let next: number | null = null;
    for (const e of events) {
      if (e.type !== "position") continue;
      const p = e as PositionEvent;
      // The continuation picked a copy. Held slots are judged against that member, not sealed here.
      if (p.relocated) {
        trackerLocked = true;
        const from = wordIndex(p.relocated.from);
        const to = wordIndex(p.relocated.to);
        resolveWatching();
        if (from >= 0 && to >= 0 && from !== to) rebase(from, to, out);
        else releaseFrozen(out, false);
        continue;
      }
      if (p.state === "locked" && p.observed[0]) {
        cursor = wordIndex(p.observed[0].word);
        if (reanchorBack) cursorFar = Math.max(cursorFar, cursor);
        // Only the first lock, before it has heard a location-unique word, holds differing slots.
        if (p.spanHold) {
          noteSpan(cursor);
          noteSpan(wordIndex(p.observed[0].span.start));
        }
      }
      // A misfit hold keeps this lock and keeps delivering tokens. An open issue stays only when its edit
      // explains the drop; the rest end uncertain. Slot accusations fall back to this copy instead of being sealed.
      if (p.hold === "start") {
        const skip = slotWords();
        resolveWatching();
        releaseFrozen(out, false);
        sealOpen(out, skip);
      }
      if (p.state !== "locked") {
        trackerLocked = false;
        // refinements.firstAyahOut: a lock the tracker let go (a look-alike opening's) is not where the take began.
        firstLock = -1;
        // Lost, it latches again on the next lock.
        if (latched && passage) {
          passage = undefined;
          readFrom = -1;
          declared = false;
        }
        if (declared) continue;
        if (anchor !== null) dropAnchor(out);
        next = null;
        continue;
      }
      trackerLocked = true;
      const start = wordIndex(p.observed[0].span.start);
      const end = wordIndex(p.observed[0].word);
      if (firstLock < 0) firstLock = start;
      const lockedTo = lockReach;
      lockReach = Math.max(lockReach, end);
      if (latched && !passage) {
        passage = { from: start, to: end };
        readFrom = start;
        declared = true;
        next = start;
        continue;
      }
      if (declared) {
        // refinements.firstAyahOut: the take's first lock, inside the passage past the window's reach while the
        // alignment still runs from the passage's first word, is a first reading that began there.
        if (firstAyahOut && lockedTo < 0 && opening && passage && anchor === passage.from && !fromLock && start > passage.from &&
          start <= passage.to && words[start].start >= hi) {
          startLate(out, start);
          next = start;
          continue;
        }
        if (end - start + 1 < JUMP_WORDS || !offPassage(start) || !offPassage(end) || fitsPassage(start, end)) {
          // refinements.reanchorBack: a lock of `JUMP_WORDS` words inside the passage and behind the window is the reciter
          // going back (restarting the passage, revising an earlier ayah). The window can't follow it there, and what
          // it holds ahead would read as left out: start over where the tracker went, as after a jump off the passage.
          // Words the window also holds (a refrain's later copy) are the window's to follow.
          // Once followed back, a lock past the window's far end is the reciter coming forward again (`followForward`).
          if (reanchorBack && anchor !== null && end - start + 1 >= JUMP_WORDS && !offPassage(start) && !offPassage(end)) {
            if (words[end].start < lo && !fitsWindow(start, end)) {
              dropAnchor(out, start, false, readBefore(start, end, lockedTo, p.timeMs));
              next = start;
            } else if (wentBack && words[start].start >= hi && !fitsWindow(start, end)) {
              followForward(out, start);
              next = start;
            }
          }
          continue;
        }
        if (latched) [passage, readFrom] = [{ from: start, to: end }, start];
        else declared = false;
        if (anchor !== null) dropAnchor(out);
        next = start;
        continue;
      }
      if (passage && !offPassage(end)) {
        // Back on the declared passage after a jump.
        declared = true;
        if (anchor !== null) dropAnchor(out);
        next = start;
        continue;
      }
      if (anchor === null) next ??= start;
      else if (words[end].start < lo || words[end].start >= hi) {
        // A jump the window can't follow: start over where the tracker went. Behind the window, with
        // refinements.readNotSkipped or reanchorBack, it is the reciter going back (`goneBack`).
        const back = (readNotSkipped || reanchorBack) && words[end].start < lo;
        dropAnchor(out, back ? start : undefined, false, back ? readBefore(start, end, lockedTo, p.timeMs) : -1);
        next = start;
      }
    }
    return next;
  };

  /** Start aligning: from the declared passage, or where the tracker locked. */
  const anchorOn = (next: number | null, events: EngineEvent[]) => {
    if (anchor === null && next === null && declared && obs.length) {
      anchor = passage!.from;
      lead = obs.length;
      fromLock = false;
      furthest = -1;
      opening = true;
    } else if (anchor === null && next !== null) {
      anchor = next;
      const end = events.filter((e): e is PositionEvent => e.type === "position").at(-1)!.observed[0].word;
      const span = words[wordIndex(end)].end - words[next].start;
      lead = Math.max(0, obs.length - span - 4);
      fromLock = true;
      furthest = -1;
      // refinements.reanchorBack: a gap a forward follow found sits where the speech after it begins.
      for (const issue of followGaps.values()) if (issue.obsFrom < 0) issue.obsFrom = issue.obsTo = Math.min(lead, obs.length - 1);
    }
  };

  return {
    push(evidence, events) {
      for (const p of evidence.phonemes) for (const m of p.seamInsertions ?? []) {
        if (Number.isFinite(m.marginal) && m.marginal >= 0 && m.marginal <= 1 &&
          Number.isInteger(m.fromFrame) && m.toFrame > m.fromFrame) {
          const key = insertionKey(m.word, m.observed, m.fromFrame, m.toFrame);
          insertionEvidence.set(key, Math.max(insertionEvidence.get(key) ?? 0, m.marginal));
        }
      }
      const out: CheckerEvent[] = [];
      frameMs = evidence.frameMs;
      if (frames && evidence.posteriors) frames.append(evidence.posteriors);
      pending = evidence.phonemes;
      const next = track(events, out);
      pending = [];
      const before = obs.length;
      obs.push(...evidence.phonemes.map((p) => ({ ...p })));
      chunkFrom = before;
      markPhrases(before);
      const lastPh = obs.at(-1);
      if (lastPh) nowMs = Math.max(nowMs, (lastPh.end + 1) * frameMs);
      anchorOn(next, events);
      if (anchor === null) {
        if (obs.length > HISTORY) obs = obs.slice(-HISTORY);
      } else evaluate(out, false);
      if (frames && obs.length) {
        const floor = ledgerFloor();
        frames.dropBefore(floor === null ? obs[0].start : Math.min(obs[0].start, floor));
      }
      return out;
    },
    end(events = []) {
      const out: CheckerEvent[] = [];
      const next = track(events, out);
      anchorOn(next, events);
      evaluate(out, true);
      return out;
    },
    mistakes() {
      return emitted.filter((e) => e.state === "confirmed" || (policy.countResolved && e.state === "resolved"));
    },
  };
}
