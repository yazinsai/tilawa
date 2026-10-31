import { align as editPath } from "./align.js";
import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { CheckerEvent, CheckerState, EngineEvent, MemorizationErrorClass, PositionEvent, WordRef } from "./events.js";
import type { DecodedPhoneme, PhonemeEvidence } from "./evidence.js";
import { PosteriorFrames } from "./ctc.js";
import { UNIT, alignSubstring, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { PhraseAsset, ReferenceIndex } from "./reference.js";

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

export interface CheckerOptions {
  policy?: CheckerPolicy;
  /** Non-Quran phrases: an inserted block that is one of these (or a basmala) is not an added word. */
  phrases?: PhraseAsset;
  /**
   * The passage the session declared (memorization): the checker aligns against it rather than the tracker's lock, so
   * a wrong word that forms another ayah is still wrong. A tracker lock off the passage spanning `JUMP_WORDS` words is
   * a deliberate jump, followed until the tracker locks back inside the passage.
   */
  passage?: { from: WordRef; to: WordRef };
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
  const gapScores = new Map<string, number>();
  const ref = index.phonemes;
  const words = index.words;
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
      ...(issue.observed !== undefined ? { observed: issue.observed } : {}),
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
    const sure = ((declared && !latched) || trackerLocked) && clarity(o.issue) >= minConfidence && supported(o.issue) !== false;
    out.push(event(o, state ?? (sure ? "confirmed" : "uncertain")));
  };

  /** The steps of the window's latest alignment. */
  let steps: Step[] = [];
  /**
   * Whether the posteriors over the error's word and a recited word either side favour what was decoded over the
   * written words by `SUPPORT`; null when there's no telling (no posteriors there, or a skipped passage).
   */
  const supported = (issue: Issue): boolean | null => {
    if (!frames || frames.startFrame < 0 || issue.cls === "skipped_passage") return null;
    const heardAt = (wi: number) => steps.filter((s): s is Extract<Step, { op: "diag" }> => s.op === "diag" && index.wordAt(s.ref) === wi).map((s) => s.obs);
    const before = prevWord(issue.word);
    const after = issue.cls === "added_word" ? issue.word : nextWord(issue.last);
    const left = before < issue.word ? heardAt(before) : [];
    const right = after < words.length ? heardAt(after) : [];
    const from = left.length ? before : issue.word;
    const to = right.length ? after : issue.cls === "added_word" ? issue.word : issue.last;
    const at = [...left, ...right, ...(issue.obsTo > issue.obsFrom ? [issue.obsFrom, issue.obsTo - 1] : [])].filter((k) => k >= 0 && k < obs.length);
    if (!at.length) return null;
    const f0 = obs[Math.min(...at)].start;
    const f1 = obs[Math.max(...at)].end + 1;
    if (f0 < frames.startFrame || f1 > frames.endFrame) return null;
    const ids = (s: string) => [...s].map((p) => tokenId.get(p) ?? 0);
    const span = (a: number, b: number) => (a < b ? ref.slice(words[a].start, words[b - 1].end) : "");
    const written = ref.slice(words[from].start, words[to].end);
    const observed = issue.observed ?? "";
    const decoded =
      issue.cls === "added_word"
        ? span(from, issue.word) + observed + span(issue.word, to + 1)
        : span(from, issue.word) + (issue.cls === "wrong_word" ? observed : "") + span(issue.last + 1, to + 1);
    const need = issue.cls === "wrong_word" && changedClarity(issue) < UNSURE ? 2 * support : support;
    return frames.nll(ids(written), f0, f1) - frames.nll(ids(decoded), f0, f1) >= need;
  };

  /** Mean decoder confidence over a wrong word's decoded phonemes that differ from the written word. */
  const changedClarity = (issue: Issue): number => {
    const heard = obs.slice(Math.max(0, issue.obsFrom), issue.obsTo);
    const changed = editPath(ref.slice(words[issue.word].start, words[issue.last].end), heard.map((p) => p.p).join(""))
      .flatMap((o) => (o.op === "sub" || o.op === "ins" ? [heard[o.obsAt].conf] : []));
    return changed.length ? changed.reduce((a, c) => a + c, 0) / changed.length : 1;
  };

  const dropAnchor = (out: CheckerEvent[]) => {
    for (const o of [...open.values()]) {
      open.delete(o.issue.word);
      decided.add(o.issue.word);
      out.push(event(o, "uncertain"));
    }
    anchor = null;
    furthest = -1;
    obs = obs.slice(-AFTER_DROP);
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

  /** The errors the current alignment shows, by word; and where it stands. */
  const judge = (steps: Step[], recoverAdditions = true, recoverGaps = true): { issues: Map<number, Issue>; head: number; furthest: number; steps: Step[]; right: Set<number> } | null => {
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
    if (recoverAdditions) {
      // Consecutive correct visits preserve real repeats and excursions: a correctly recited intervening word
      // remains a visit, while a fragment forced through a wrong word does not conceal an insertion.
      const intact = visits.filter(v => counts(v) && !wrong(v));
      const repairs: { from: number; to: number; path: Step[] }[] = [];
      const weight = (wi: number) => [...ref.slice(words[wi].start, words[wi].end)].reduce((n, p) => n + (CONSONANTS.has(p) ? UNIT : SOFT), 0);
      for (let i = 1; i < intact.length; i++) {
        const left = intact[i - 1], right = intact[i];
        if (nextWord(left.word) !== right.word || left.obsTo > right.obsFrom || left.stepFrom === undefined || right.stepTo === undefined) continue;
        if (repairs.length && left.stepFrom < repairs.at(-1)!.to) continue;
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
        for (const r of repairs.reverse()) repaired.splice(r.from, r.to - r.from, ...r.path);
        return judge(repaired, false, recoverGaps);
      }
    }
    if (recoverGaps && frames && frames.startFrame >= 0) {
      // A noisy word can lend its ending to a short preceding word, leaving a gap on the noisy word instead.
      // Compare the two complete spoken sequences over the same fixed flanks; only acoustic evidence can choose
      // between them. Then move the borrowed visit itself, rather than merely renaming the resulting warning.
      const ordered = [...last.values()].sort((a, b) => a.word - b.word);
      const repairs: { from: number; to: number; path: Step[] }[] = [];
      for (let i = 1; i < ordered.length; i++) {
        const left = ordered[i - 1], rightVisit = ordered[i];
        const gapFrom = nextWord(left.word), gapTo = prevWord(rightVisit.word);
        if (gapFrom > gapTo || left.obsTo > rightVisit.obsFrom) continue;
        type Move = { from: number; to: number; target: number; first: number; last: number; borrowed: Visit; a: Visit; b: Visit };
        const moves: Move[] = [];
        const before = last.get(prevWord(left.word));
        if (before && !wrong(before) && !wrong(left) && !wrong(rightVisit) && left.stepFrom !== undefined && rightVisit.stepFrom !== undefined) {
          moves.push({ from: left.stepFrom, to: rightVisit.stepFrom, target: gapTo, first: left.word, last: prevWord(gapTo), borrowed: left, a: before, b: rightVisit });
        }
        const after = last.get(nextWord(rightVisit.word));
        if (after && !wrong(left) && !wrong(rightVisit) && !wrong(after) && left.stepTo !== undefined && rightVisit.stepTo !== undefined) {
          moves.push({ from: left.stepTo, to: rightVisit.stepTo, target: gapFrom, first: nextWord(gapFrom), last: rightVisit.word, borrowed: rightVisit, a: left, b: after });
        }
        let best: { move: Move; gain: number } | undefined;
        for (const move of moves) {
          const { borrowed, a, b } = move;
          if (a.obsFrom < 0 || a.obsTo > borrowed.obsFrom || borrowed.obsTo > b.obsFrom || b.obsTo <= b.obsFrom) continue;
          if (repairs.some(r => move.from < r.to && move.to > r.from)) continue;
          const represented = steps.slice(move.from, move.to).flatMap(s => "obs" in s ? [s.obs] : []);
          if (represented.length !== borrowed.obsTo - borrowed.obsFrom || represented.some((oi, j) => oi !== borrowed.obsFrom + j)) continue;
          // A repeat or another real edit needs its own explanation, outside this one-visit relocation.
          const positions = steps.slice(move.from, move.to).flatMap(s => "ref" in s ? [s.ref] : []);
          if (positions.some((ri, j) => j > 0 && ri < positions[j - 1]) || obs.slice(a.obsFrom, b.obsTo).some(p => p.junk)) continue;
          if (added.some(x => x.obsFrom < b.obsTo && x.obsTo > a.obsFrom)) continue;
          const f0 = obs[a.obsFrom].start, f1 = obs[b.obsTo - 1].end + 1;
          if (f0 < frames.startFrame || f1 > frames.endFrame) continue;
          const score = (first: number, lastWord: number) => {
            const cacheKey = `${a.word}:${b.word}:${first}:${lastWord}:${f0}:${f1}`;
            let value = gapScores.get(cacheKey);
            if (value === undefined) {
              const written = ref.slice(words[a.word].start, words[first].start) + ref.slice(words[lastWord].end, words[b.word].end);
              value = frames.nll([...written].map(p => tokenId.get(p) ?? 0), f0, f1);
              if (gapScores.size >= 512) gapScores.clear();
              gapScores.set(cacheKey, value);
            }
            return value;
          };
          const gain = score(gapFrom, gapTo) - score(move.first, move.last);
          if (gain >= support && (!best || gain > best.gain)) best = { move, gain };
        }
        if (!best) continue;
        const m = best.move;
        const deletion: Step[] = [];
        for (let wi = m.first; wi <= m.last; wi++) for (let ri = words[wi].start; ri < words[wi].end; ri++) deletion.push({ op: "del", ref: ri });
        const spoken = wordSteps(m.target, m.borrowed.obsFrom, m.borrowed.obsTo);
        repairs.push({ from: m.from, to: m.to, path: m.target < m.first ? [...spoken, ...deletion] : [...deletion, ...spoken] });
      }
      if (repairs.length) {
        const repaired = [...steps];
        for (const r of repairs.sort((a, b) => b.from - a.from)) repaired.splice(r.from, r.to - r.from, ...r.path);
        return judge(repaired, recoverAdditions, false);
      }
    }
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
          issues.set(gap.word, { ...gap, cls: "wrong_word", obsTo: after!.obsFrom, observed: said.map((p) => p.p).join("") });
        }
        gap = null;
        return;
      }
      // Words past the head may still be coming: a gap is an omission once the reciter is past it, or already suspected.
      const was = open.get(gap.word)?.issue.cls;
      if (!(gap.last < head || was === "omitted_word" || was === "skipped_passage")) return void (gap = null);
      // A word either side of a skip can recur on the other side (برب الناس [ملك الناس] إله الناس), so the alignment
      // can place the gap a word off. Of the equal placements, take the one on an ayah's start: reciters skip ayahs.
      const onStart = (i: Issue) => ayahFirst[ayahOf[i.word]] === i.word;
      if (!onStart(gap)) {
        const placements: Issue[] = [];
        let r: Issue = gap;
        for (let v = after; v && nextWord(r.last) === v.word && skeleton(r.word) === skeleton(v.word); v = last.get(nextWord(v.word))) {
          r = { ...r, word: nextWord(r.word), last: v.word, obsFrom: v.obsTo, obsTo: v.obsTo };
          placements.push(r);
        }
        let l: Issue = gap;
        for (let v = last.get(prevWord(l.word)); v && skeleton(v.word) === skeleton(l.last); v = last.get(prevWord(v.word))) {
          l = { ...l, word: v.word, last: prevWord(l.last), obsFrom: v.obsFrom, obsTo: v.obsFrom };
          placements.push(l);
        }
        gap = placements.find(onStart) ?? gap;
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
        issues.set(wi, { cls: "wrong_word", word: wi, last: wi, obsFrom: v.obsFrom, obsTo: v.obsTo, observed: obs.slice(v.obsFrom, v.obsTo).map((p) => p.p).join("") });
      }
    }
    flush();
    for (const a of added) if (a.word >= first && !issues.has(a.word)) issues.set(a.word, a);
    return { issues, head, furthest: reached, steps, right };
  };

  /** Matched phonemes on the word after an error, heard after it: the next word has begun. */
  const onset = (steps: Step[], issue: Issue) => {
    const next = nextWord(issue.last);
    let n = 0;
    for (const s of steps) if (s.op === "diag" && s.obs >= issue.obsTo && index.wordAt(s.ref) === next && ref[s.ref] === obs[s.obs].p) n++;
    return n >= Math.min(ONSET, words[next].end - words[next].start);
  };

  const evaluate = (out: CheckerEvent[], final: boolean) => {
    if (anchor === null || !obs.length) return;
    setWindow();
    steps = align();
    const judged = judge(steps);
    if (!judged) return;
    steps = judged.steps;
    const { issues, head, furthest: reached, right } = judged;
    furthest = Math.max(furthest, reached);
    for (const [wi, o] of [...open]) {
      const now = issues.get(wi);
      if (!now) {
        open.delete(wi);
        out.push(event(o, "resolved"));
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
      const due = omission
        ? final || ayahOf[reached] >= ayahOf[issue.last] + 2
        : head !== issue.word &&
          onset(steps, issue) &&
          (confirm.after === "next-word" || nowMs - o.sinceMs >= confirm.ms || (final && declared && (!latched || (policy.finaliseFree ?? true)) && [...right].some((wi) => wi > issue.last)));
      if (due) decide(o, out);
      // The session ended first. A declared passage (or a latched free one, with `finaliseFree`) confirms once a later
      // word was recited right; otherwise it ends uncertain.
      else if (final) decide(o, out, "uncertain");
    }
    if (latched && passage) {
      passage.from = Math.max(passage.from, ayahFirst[Math.max(0, ayahOf[anchor] - 1)]);
      passage.to = Math.max(passage.to, furthest, anchor);
    }
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
    const word = index.wordAt(at);
    for (const o of [...open.values()]) if (o.issue.word < word) decide(o, out);
    for (const o of open.values()) {
      o.issue = { ...o.issue, obsFrom: Math.max(0, o.issue.obsFrom - cut), obsTo: Math.max(0, o.issue.obsTo - cut) };
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
      const out: CheckerEvent[] = [];
      frameMs = evidence.frameMs;
      if (frames && evidence.posteriors) frames.append(evidence.posteriors);
      const next = track(events, out);
      const before = obs.length;
      obs.push(...evidence.phonemes.map((p) => ({ ...p })));
      markPhrases(before);
      const lastPh = obs.at(-1);
      if (lastPh) nowMs = Math.max(nowMs, (lastPh.end + 1) * frameMs);
      anchorOn(next, events);
      if (anchor === null) {
        if (obs.length > HISTORY) obs = obs.slice(-HISTORY);
      } else evaluate(out, false);
      if (frames && obs.length) frames.dropBefore(obs[0].start);
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
