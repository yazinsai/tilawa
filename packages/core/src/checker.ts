import { align as editPath } from "./align.js";
import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { CheckerEvent, CheckerState, EngineEvent, MemorizationErrorClass, PositionEvent, WordRef } from "./events.js";
import type { DecodedPhoneme, PhonemeEvidence } from "./evidence.js";
import { PosteriorFrames } from "./ctc.js";
import { UNIT, alignSubstring, collapseHeld, deletionCost, insertionCost, subCost } from "./fuzzy.js";
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
 * Memorization costs: a different word changes consonants. Vowels, madd and the marks for qalqalah and ghunna cost
 * `SOFT` to change; a sound held longer or shorter (beside the same sound) costs `HELD`. Consonant edits are hard.
 */
const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
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
 * Below this cost per phoneme the lock's words also fit the declared passage: a wrong word that forms another ayah
 * shares most of its words with the passage, a deliberate jump doesn't.
 */
const FITS_PASSAGE = 0.35;
/** Nats by which the decoded words must beat the written ones over the local window for the error to stand. */
const SUPPORT = 10;
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
}

export function createChecker(index: ReferenceIndex, options: CheckerOptions = {}): Checker {
  const policy = options.policy ?? {};
  const confirm = policy.confirm ?? { after: "delay", ms: 2500 };
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
  /** Off the passage by more than an ayah either side. */
  const offPassage = (wi: number) => !!passage && (ayahOf[wi] < ayahOf[passage.from] - 1 || ayahOf[wi] > ayahOf[passage.to] + 1);
  const fitsPassage = (from: number, to: number) => {
    const text = ref.slice(words[from].start, words[to].end);
    const a = alignSubstring(ref, words[passage!.from].start, words[passage!.to].end, text, (o) => wordStarts.has(o));
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
  const open = new Map<number, Open>();
  /** Words whose error was confirmed or ended uncertain; never reopened. */
  const decided = new Set<number>();
  const emitted: CheckerEvent[] = [];
  let seq = 0;

  const event = (o: Open, state: CheckerState): CheckerEvent => {
    const { issue } = o;
    const conf = clarity(issue);
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
    const sure = ((declared && !latched) || trackerLocked) && heardSure(o.issue) && supported(o.issue) !== false;
    out.push(event(o, state ?? (sure ? "confirmed" : "uncertain")));
  };

  /** The steps of the window's latest alignment. */
  let steps: Step[] = [];
  /**
   * Whether the posteriors over the error's word and a recited word either side favour what was decoded over the
   * written words by `SUPPORT`; null when there's no telling (no posteriors there, or a skipped passage).
   */
  const supported = (issue: Issue, requireFinite = false): boolean | null => {
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
      const need = supportNeed(issue);
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
      const need = supportNeed(issue);
      const expectedNll = frames.nll(ids(writtenPair), a0, b0), observedNll = frames.nll(ids(issue.pair), a0, b0);
      if (requireFinite && (!Number.isFinite(expectedNll) || !Number.isFinite(observedNll))) return null;
      return expectedNll - observedNll >= need;
    }
    if (!at.length) return null;
    const f0 = obs[Math.min(...at)].start;
    const f1 = obs[Math.max(...at)].end + 1;
    if (f0 < frames.startFrame || f1 > frames.endFrame) return null;
    if (issue.phrase && issue.supportRange) {
      const altNll = frames.nll(ids(issue.phrase.alt), f0, f1);
      // A written phrase the frames cannot carry at all still loses to a finite parallel. NaN (both impossible)
      // is not evidence either way.
      if (!Number.isFinite(altNll)) return requireFinite ? null : false;
      return frames.nll(ids(issue.phrase.written), f0, f1) - altNll >= issue.phrase.need;
    }
    const span = (a: number, b: number) => (a < b ? ref.slice(words[a].start, words[b - 1].end) : "");
    const written = ref.slice(words[from].start, words[to].end);
    const observed = issue.observed ?? "";
    const decoded =
      issue.cls === "added_word"
        ? span(from, issue.word) + observed + span(issue.word, to + 1)
        : span(from, issue.word) + (issue.cls === "wrong_word" ? observed : "") + span(issue.last + 1, to + 1);
    const need = supportNeed(issue);
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

  const dropAnchor = (out: CheckerEvent[]) => {
    // A fit that already accepted a skip has its evidence. The lock leaving is the same as the passage ending.
    for (const wi of [...ledgerIssues.keys()]) {
      const o = open.get(wi);
      ledgerIssues.delete(wi);
      if (!o) continue;
      open.delete(wi);
      decided.add(wi);
      out.push(event(o, sureOf(o) ? "confirmed" : "uncertain"));
    }
    sealAll(out);
    anchor = null;
    furthest = -1;
    obs = obs.slice(-AFTER_DROP);
    // Ayahs already read stay read, so the next lock can still close a gap this jump opened.
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
      if (ayahText(a) === ayahText(left) || ayahText(a) === ayahText(right)) continue;
      const n = words.slice(ayahFirst[a], ayahLast[a] + 1).filter((w) => w.end > w.start).length;
      if (n < 3) continue;
      const margin = absenceMargin(frames, idOf, [leftWords, wordPhones(ayahFirst[a], ayahLast[a]), rightWords], 1, leftSpan.frameFrom, rightTo);
      ledgerDone.add(a);
      if (margin === null || margin < PHRASE_SWITCH) continue;
      const wi = ayahFirst[a];
      if (decided.has(wi) || heldSlot(wi)) continue;
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
    const d = new Float64Array((m + 1) * w).fill(INF);
    const op = new Uint8Array((m + 1) * w);
    const jumpFrom = new Int32Array(m + 1);
    const starts: number[] = [];
    for (let i = 0; i <= n; i++) if (wordStarts.has(lo + i)) starts.push(i);
    const jumps = (j: number) => {
      const row = j * w;
      let k = 0;
      for (let i = 1; i <= n; i++) if (d[row + i] < d[row + k]) k = i;
      jumpFrom[j] = k;
      for (const s of starts) {
        if (d[row + k] + JUMP < d[row + s]) {
          d[row + s] = d[row + k] + JUMP;
          op[row + s] = OP_JUMP;
        }
      }
    };
    d[a0] = 0;
    op[a0] = OP_START;
    for (let i = a0 + 1; i <= n; i++) {
      d[i] = d[i - 1] + delCost[i - 1];
      op[i] = OP_DEL;
    }
    jumps(0);
    for (let j = 1; j <= m; j++) {
      const p = obs[j - 1].p;
      const { sub, ins: insCost } = costsFor(p);
      // A repeated decoded sound can be lengthening even at a word edge, where the reference has no matching run.
      // Use the same held-sound charge as judging; keep ordinary insertion costs for unrelated speech.
      const noise = p === obs[j - 2]?.p ? HELD : UNIT;
      const row = j * w;
      const prev = row - w;
      for (let i = 0; i <= n; i++) {
        let best = INF;
        let o = OP_INS;
        if (i > 0) {
          best = d[prev + i - 1] + sub[i - 1];
          o = OP_DIAG;
        }
        const ins = d[prev + i] + (obs[j - 1].junk ? PHRASE : j <= lead && i === a0 ? LEAD : Math.min(insCost[i], noise));
        if (ins < best) {
          best = ins;
          o = j <= lead && i === a0 ? OP_LEAD : OP_INS;
        }
        if (i > 0) {
          const del = d[row + i - 1] + delCost[i - 1];
          if (del < best) {
            best = del;
            o = OP_DEL;
          }
        }
        d[row + i] = best;
        op[row + i] = o;
      }
      jumps(j);
    }

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
    const ins = (at: number, j: number) => Math.min(
      insertionCost(reference, at, obs[from + j - 1].p, n),
      obs[from + j - 1].p === obs[from + j - 2]?.p ? HELD : UNIT,
    );
    const del = (at: number) => reference[at] === reference[at - 1] || reference[at] === reference[at + 1] ? HELD : deletionCost(reference, at);
    for (let j = 1; j <= m; j++) { d[j] = d[j - 1] + ins(0, j); op[j] = OP_INS; }
    for (let i = 1; i <= n; i++) {
      const row = i * width, prev = row - width, at = i - 1;
      d[row] = d[prev] + del(at); op[row] = OP_DEL;
      for (let j = 1; j <= m; j++) {
        const p = obs[from + j - 1].p;
        d[row + j] = d[prev + j - 1] + subCost(reference[at], p); op[row + j] = OP_DIAG;
        const deletion = d[prev + j] + del(at);
        if (deletion < d[row + j]) { d[row + j] = deletion; op[row + j] = OP_DEL; }
        const insertion = d[row + j - 1] + ins(at + 1, j);
        if (insertion < d[row + j]) { d[row + j] = insertion; op[row + j] = OP_INS; }
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
    const first = Math.min(...seen) + (fromLock ? 1 : 0);
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
    /** Mean decoder confidence over the heard phones that miss the written phrase and match the parallel. */
    const switchedClarity = (path: { written: string; alt: string }, span: { from: number; to: number }) => {
      const heard = obs.slice(span.from, span.to);
      const said = heard.map((p) => p.p).join("");
      const off = new Set(editPath(path.written, said).flatMap((o) => (o.op === "sub" || o.op === "ins" ? [o.obsAt] : [])));
      const on = editPath(path.alt, said).flatMap((o) => (o.op === "match" && off.has(o.obsAt) ? [heard[o.obsAt].conf] : []));
      return on.length ? on.reduce((a, c) => a + c, 0) / on.length : 1;
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
        const altNll = frames.nll(ids(path.alt), f0, f1);
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
        const margin = frames.nll(ids(path.written), f0, f1) - altNll;
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
    furthest = Math.max(furthest, reached);
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
      } else if (now.cls !== o.issue.cls) {
        o.issue = now;
        out.push(event(o, "suspected"));
      } else o.issue = now;
    }
    for (const [wi, issue] of issues) {
      if (open.has(wi) || decided.has(wi)) continue;
      const o: Open = { id: `${words[wi].surah}:${words[wi].ayah}:${words[wi].word}#${++seq}`, issue, sinceMs: nowMs };
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
      const due = omission
        ? final || ayahOf[reached] >= ayahOf[issue.last] + 2
        : past &&
          followed &&
          (confirm.after === "next-word" || nowMs - o.sinceMs >= confirm.ms || (final && declared && (!latched || (policy.finaliseFree ?? true)) && [...right].some((wi) => wi > issue.last)));
      // A slot the two copies disagree on stays open until the continuation picks a copy.
      if (heldSlot(issue.word)) continue;
      if (due) decide(o, out);
      // The session ended first. A declared passage (or a latched free one, with `finaliseFree`) confirms once a later
      // word was recited right; otherwise it ends uncertain.
      else if (final) decide(o, out, "uncertain");
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
    for (let k = 0; k < steps.length; k++) {
      const s = steps[k];
      if (s.op !== "diag" || s.obs >= limit) continue;
      const next = steps[k + 1];
      if (wordStarts.has(s.ref + 1) && next?.op === "diag" && next.ref === s.ref + 1) {
        cut = s.obs + 1;
        at = s.ref + 1;
      }
    }
    if (cut < 0) return;
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
        // Lost, it latches again on the next lock.
        if (latched && passage) {
          passage = undefined;
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
      if (latched && !passage) {
        passage = { from: start, to: end };
        declared = true;
        next = start;
        continue;
      }
      if (declared) {
        if (end - start + 1 < JUMP_WORDS || !offPassage(start) || !offPassage(end) || fitsPassage(start, end)) continue;
        if (latched) passage = { from: start, to: end };
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
        // A jump the window can't follow: start over where the tracker went.
        dropAnchor(out);
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
    } else if (anchor === null && next !== null) {
      anchor = next;
      const end = events.filter((e): e is PositionEvent => e.type === "position").at(-1)!.observed[0].word;
      const span = words[wordIndex(end)].end - words[next].start;
      lead = Math.max(0, obs.length - span - 4);
      fromLock = true;
      furthest = -1;
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
      const next = track(events, out);
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
      anchorOn(track(events, out), events);
      evaluate(out, true);
      return out;
    },
    mistakes() {
      return emitted.filter((e) => e.state === "confirmed" || (policy.countResolved && e.state === "resolved"));
    },
  };
}
