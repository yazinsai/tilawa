import { EVENT_SCHEMA_VERSION } from "./events.js";
import type { CheckerEvent, CheckerState, EngineEvent, MemorizationErrorClass, PositionEvent, WordRef } from "./events.js";
import type { DecodedPhoneme, PhonemeEvidence } from "./evidence.js";
import { UNIT, alignSubstring, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { PhraseAsset, ReferenceIndex } from "./reference.js";

export interface CheckerPolicy {
  /** When a suspected wrong or added word is confirmed: after `ms` (2–3 s), or once the next word has begun. Default 2500 ms. */
  confirm?: { after: "delay"; ms: number } | { after: "next-word" };
  /** Whether resolved errors (self-corrections, recovered skips) count as mistakes. Default false. */
  countResolved?: boolean;
  /** Mean decoder confidence over an error's phonemes below which it ends `uncertain`, never `confirmed`. Default 0.6. */
  minConfidence?: number;
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
  const passage = options.passage && { from: wordIndex(options.passage.from), to: wordIndex(options.passage.to) };
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
    out.push(event(o, state ?? ((declared || trackerLocked) && clarity(o.issue) >= minConfidence ? "confirmed" : "uncertain")));
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

  /** Weighted alignment of the window's phonemes from the anchor, with jump arcs to every word start in the window. */
  const align = (): Step[] => {
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
      d[i] = d[i - 1] + deletionCost(ref, lo + i - 1);
      op[i] = OP_DEL;
    }
    jumps(0);
    for (let j = 1; j <= m; j++) {
      const p = obs[j - 1].p;
      const row = j * w;
      const prev = row - w;
      for (let i = 0; i <= n; i++) {
        let best = INF;
        let o = OP_INS;
        if (i > 0) {
          best = d[prev + i - 1] + subCost(ref[lo + i - 1], p);
          o = OP_DIAG;
        }
        const ins = d[prev + i] + (obs[j - 1].junk ? 0 : j <= lead && i === a0 ? LEAD : insertionCost(ref, lo + i, p, hi));
        if (ins < best) {
          best = ins;
          o = j <= lead && i === a0 ? OP_LEAD : OP_INS;
        }
        if (i > 0) {
          const del = d[row + i - 1] + deletionCost(ref, lo + i - 1);
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

  /** Mark phrases that end at each phoneme from `from` on: the alignment consumes them for free. */
  const markPhrases = (from: number) => {
    for (let end = from + 1; end <= obs.length; end++) {
      for (const p of phrases) {
        const start = Math.max(0, end - Math.ceil(p.length * 1.6) - 4);
        const tail = obs.slice(start, end).map((x) => x.p).join("");
        const a = alignSubstring(tail, 0, tail.length, p, () => true);
        if (!a || a.end < tail.length - 1 || a.cost / UNIT / p.length > PHRASE_MAX_COST) continue;
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

  /** The errors the current alignment shows, by word; and where it stands. */
  const judge = (steps: Step[]) => {
    const visits: Visit[] = [];
    const added: Issue[] = [];
    let cur: Visit | null = null;
    let block: number[] = [];
    /** Insertions inside the current word, with where they sit. */
    let inner: { at: number; obs: number }[] = [];
    let head = -1;
    // A long run inside a word is another word said there, before it or after it by which half it sits in.
    const flushInner = () => {
      const v = cur;
      if (!v || !inner.length) return void (inner = []);
      if (inner.length >= MIN_ADDED + 1 && inner.filter((k) => CONSONANTS.has(obs[k.obs].p)).length >= 2) {
        const w = words[v.word];
        const word = inner[0].at - w.start < (w.end - w.start) / 2 ? v.word : nextWord(v.word);
        const text = inner.map((k) => obs[k.obs].p).join("");
        added.push({ cls: "added_word", word, last: word, obsFrom: inner[0].obs, obsTo: inner.at(-1)!.obs + 1, observed: text });
      } else for (const k of inner) charge(v, insCharge(k.at, obs[k.obs].p, obs[k.obs - 1]?.p));
      inner = [];
    };
    const close = () => {
      flushInner();
      if (cur) visits.push(cur);
      cur = null;
    };
    for (const s of steps) {
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
        cur = { word: wi, covered: 0, consumed: 0, cost: 0, hard: 0, kept: 0, obsFrom: s.op === "diag" ? s.obs : -1, obsTo: -1 };
      }
      flushInner();
      const v: Visit = cur;
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

    const length = (wi: number) => words[wi].end - words[wi].start;
    // Vowels and madd alone, every consonant lost, aren't the word: a decoder's held sound can align onto a dropped one.
    const counts = (v: Visit) => v.consumed > 0 && v.covered / length(v.word) >= MIN_COVERAGE && (v.kept > 0 || !hasConsonant(v.word));
    /** Each word's last counting visit, and whether any counting visit got it right. */
    const last = new Map<number, Visit>();
    const right = new Set<number>();
    for (const v of visits) {
      if (!counts(v) || v.word < anchor!) continue;
      last.set(v.word, v);
      if (!wrong(v)) right.add(v.word);
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
      if (said.length >= Math.max(MIN_ADDED, expected / 2)) {
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
    const judged = judge(align());
    if (!judged) return;
    const { issues, head, furthest: reached, steps, right } = judged;
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
          (confirm.after === "next-word" || nowMs - o.sinceMs >= confirm.ms || (final && declared && [...right].some((wi) => wi > issue.last)));
      if (due) decide(o, out);
      // The session ended first. A declared passage confirms once a later word was recited right; free recitation can't
      // tell a wrong word from the tracker's next guess, so it ends uncertain.
      else if (final) decide(o, out, "uncertain");
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
        if (declared) continue;
        if (anchor !== null) dropAnchor(out);
        next = null;
        continue;
      }
      trackerLocked = true;
      const start = wordIndex(p.observed[0].span.start);
      const end = wordIndex(p.observed[0].word);
      if (declared) {
        if (end - start + 1 < JUMP_WORDS || !offPassage(start) || !offPassage(end) || fitsPassage(start, end)) continue;
        declared = false;
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

  return {
    push(evidence, events) {
      const out: CheckerEvent[] = [];
      frameMs = evidence.frameMs;
      const next = track(events, out);
      const before = obs.length;
      obs.push(...evidence.phonemes.map((p) => ({ ...p })));
      markPhrases(before);
      const lastPh = obs.at(-1);
      if (lastPh) nowMs = Math.max(nowMs, (lastPh.end + 1) * frameMs);
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
      if (anchor === null) {
        if (obs.length > HISTORY) obs = obs.slice(-HISTORY);
        return out;
      }
      evaluate(out, false);
      return out;
    },
    end(events = []) {
      const out: CheckerEvent[] = [];
      track(events, out);
      evaluate(out, true);
      return out;
    },
    mistakes() {
      return emitted.filter((e) => e.state === "confirmed" || (policy.countResolved && e.state === "resolved"));
    },
  };
}
