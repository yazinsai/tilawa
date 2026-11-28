import { readFileSync } from "node:fs";
import { createChecker } from "../../src/engine/checker.js";
import type { CheckerRefinements } from "../../src/engine/checker.js";
import { createTracker } from "../../src/engine/tracker.js";
import type { TrackerRefinements } from "../../src/engine/tracker.js";
import { loadReferenceIndex } from "../../src/engine/reference.js";
import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../../src/engine/events.js";
import type { DecodedPhoneme, PhonemeEvidence } from "../../src/engine/evidence.js";
import type { PhraseAsset, ReferenceAsset } from "../../src/engine/reference.js";
import { DEFAULTS } from "../../src/defaults.js";

/**
 * Helpers for the refinements' tests (`refinements` in the tracker's and checker's options, `policy.refinements` in a
 * session).
 * The takes are synthetic phoneme streams built from the reference: a phoneme every 120 ms (three 40 ms frames),
 * pushed three at a time (about one model window) unless a test says otherwise, with posteriors that favour the
 * phoneme said.
 */

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
export const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
export const index = loadReferenceIndex(asset);
export const FRAME_MS = 40;

export type Fields = CheckerRefinements & TrackerRefinements;
/**
 * Every correction on, `markWhenSure` included (off by default): what a test of a correction runs, unless it says
 * otherwise. A take recited with no `refinements` runs the defaults ({@link DEFAULT}).
 */
export const ALL: Required<Fields> = {
  readNotSkipped: true, reanchorBack: true, stutter: true, ledgerPartial: true, omissionDue: true,
  commitAfterJump: true, firstAyahOut: true, marksInPassage: true, repeatNotWrong: true, markWhenSure: true,
  confirmAdvance: true, onsetRun: true, declaredStart: true, declaredLeadIn: true, declaredMove: true, noLostBeforeLock: true,
  consonantConfirm: true, rivalHold: true, passageWindow: true, restartNoEntry: true, withdrawBegun: true, leftOnNewAudio: true,
  begunByConsonants: true, showSaid: true,
};
/**
 * The defaults, as a session resolves them (`DEFAULTS.policy.refinements`): every correction on but markWhenSure.
 */
export const DEFAULT: Fields = DEFAULTS.policy.refinements;
/** Every correction on with one switched off: rc.1's rule for that one. */
export const without = (name: keyof Fields): Fields => ({ ...ALL, [name]: false });
/** rc.1: every correction off. */
export const RC1: Required<Fields> = {
  readNotSkipped: false, reanchorBack: false, stutter: false, ledgerPartial: false, omissionDue: false,
  commitAfterJump: false, firstAyahOut: false, marksInPassage: false, repeatNotWrong: false, markWhenSure: false,
  confirmAdvance: false, onsetRun: false, declaredStart: false, declaredLeadIn: false, declaredMove: false, noLostBeforeLock: false,
  consonantConfirm: false, rivalHold: false, passageWindow: false, restartNoEntry: false, withdrawBegun: false, leftOnNewAudio: false,
  begunByConsonants: false, showSaid: false,
};
/**
 * The version before the corrections that follow the tracker's run and keep marks in the passage: `readNotSkipped` and
 * `declaredStart` off, the other corrections of its time on, and the later ones off.
 */
export const EARLIER: Required<Fields> = {
  ...RC1, reanchorBack: true, stutter: true, ledgerPartial: true, omissionDue: true, confirmAdvance: true, onsetRun: true, noLostBeforeLock: true,
};
/**
 * `EARLIER` with `readNotSkipped` and `declaredStart` on: every correction of that version on, the set an older test of
 * one of them compares against. The later corrections mend some of the same cases (a restart is also
 * `commitAfterJump`'s) and hold some words longer by their own rules (`consonantConfirm`, `rivalHold`), each tested on
 * its own.
 */
export const EARLIER_ON: Required<Fields> = { ...EARLIER, readNotSkipped: true, declaredStart: true };
/** `EARLIER_ON` with one correction switched off: rc.1's rule for that one. */
export const earlierWithout = (name: keyof Fields): Fields => ({ ...EARLIER_ON, [name]: false });
/** rc.1 with one correction on. */
export const only = (name: keyof Fields): Fields => ({ ...RC1, [name]: true });

export const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const wordIndex = new Map(index.words.map((x, i) => [key(x), i]));
export const at = (r: WordRef) => wordIndex.get(key(r))!;
export const ref = (i: number): WordRef => ({ surah: index.words[i].surah, ayah: index.words[i].ayah, word: index.words[i].word });
export const spoken = (i: number) => index.words[i].end > index.words[i].start;
/** The spoken words of an ayah, as word indices. */
export const ayah = (s: number, a: number) => index.words.flatMap((x, i) => (x.surah === s && x.ayah === a && spoken(i) ? [i] : []));
export const passage = (s: number, a: number, b: number): [WordRef, WordRef] => [ref(ayah(s, a)[0]), ref(ayah(s, b).at(-1)!)];

/** Words said as written, or other speech (`said`), at a decoder confidence; or a pause of that many frames. */
export type Piece = { words: number[] } | { said: string; conf?: number } | { pause: number };
/** An ayah's spoken words `from`–`to` (1-based, inclusive; the whole ayah by default), said as written. */
export const say = (s: number, a: number, from = 1, to?: number): Piece => {
  const ws = ayah(s, a);
  return { words: ws.slice(from - 1, to ?? ws.length) };
};
export const noise = (said: string, conf = 0.96): Piece => ({ said, conf });
export const pause = (frames: number): Piece => ({ pause: frames });

export interface Speech {
  phonemes: DecodedPhoneme[];
  /** When each word was first said (its first phoneme's start, ms). */
  saidAt: Map<string, number>;
}
/** The word indices of each ayah's first spoken word. */
const opensAyah = new Set(index.words.flatMap((x, i) => {
  if (!spoken(i)) return [];
  let j = i - 1;
  while (j >= 0 && !spoken(j)) j--;
  return j < 0 || index.words[j].surah !== x.surah || index.words[j].ayah !== x.ayah ? [i] : [];
}));
/**
 * How speech is delivered: each phoneme said for `ms(k)` milliseconds (`k` counts the phonemes said), and an ayah's
 * first phoneme heard at confidence `opening(k)` (`k` counts the ayah openings said) rather than 0.96.
 */
export interface Delivery {
  ms: (k: number) => number;
  opening: (k: number) => number;
}
/**
 * The pieces as decoded phonemes from frame 10: 3 frames apart (one frame between them), or, with a `delivery`, each
 * over the frames its time covers, back to back.
 */
export function speech(pieces: Piece[], delivery?: Delivery): Speech {
  const phonemes: DecodedPhoneme[] = [];
  const saidAt = new Map<string, number>();
  let frame = 10;
  let ms = frame * FRAME_MS;
  let [said, openings] = [0, 0];
  const add = (text: string, conf: number, opening = false) => {
    [...text].forEach((p, i) => {
      if (!delivery) {
        phonemes.push({ p, start: frame, end: frame + 1, conf });
        frame += 3;
        return;
      }
      const start = Math.round(ms / FRAME_MS);
      ms += delivery.ms(said++);
      const first = opening && i === 0;
      phonemes.push({ p, start, end: Math.max(start, Math.round(ms / FRAME_MS) - 1), conf: first ? delivery.opening(openings++) : conf });
    });
  };
  for (const piece of pieces) {
    if ("pause" in piece) [frame, ms] = [frame + piece.pause, ms + piece.pause * FRAME_MS];
    else if ("said" in piece) add(piece.said, piece.conf ?? 0.96);
    else for (const wi of piece.words) {
      if (!saidAt.has(key(ref(wi)))) saidAt.set(key(ref(wi)), (delivery ? Math.round(ms / FRAME_MS) : frame) * FRAME_MS);
      const w = index.words[wi];
      add(index.phonemes.slice(w.start, w.end), 0.96, opensAyah.has(wi));
    }
  }
  return { phonemes, saidAt };
}
/** A repeatable sequence of numbers in [lo, hi), from `seed`. */
export const spread = (seed: number, lo: number, hi: number) => {
  let x = seed >>> 0 || 1;
  return () => {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    return lo + (x / 2 ** 32) * (hi - lo);
  };
};

/**
 * Top-5 posteriors from frame `from` to just past the chunk (no further than `until`, where the next chunk's speech
 * starts): the phoneme said on its frames, blank between.
 */
function withPosteriors(evidence: PhonemeEvidence, from: number, until = Infinity): PhonemeEvidence {
  const ids = [...index.tokens];
  const id = (t: string) => ids.indexOf(t) + 1;
  const to = Math.min(evidence.phonemes.at(-1)!.end + 2, until);
  const rows: [number, number][][] = Array.from({ length: to - from }, () => [[0, 0.95]]);
  for (const ph of evidence.phonemes) {
    for (let f = ph.start; f <= ph.end; f++) rows[f - from] = [[id(ph.p), ph.conf > 0.5 ? 0.9 : 0.45], [0, 0.05]];
  }
  const flatIds: number[] = [];
  const flatProbs: number[] = [];
  for (const row of rows) {
    for (let t = 1; row.length < 5; t++) if (!row.some(([i]) => i === t)) row.push([t, 0]);
    for (const [i, p] of row) {
      flatIds.push(i);
      flatProbs.push(p);
    }
  }
  return { ...evidence, decodedTo: to, posteriors: { k: 5, startFrame: from, nFrames: rows.length, ids: flatIds, probs: flatProbs } };
}

export interface Take {
  /** The tracker's positions: where the reciter is, the word being said included (what the checker reads). */
  positions: PositionEvent[];
  /** The positions to show (`Tracker.shown`): with refinements.showSaid, each word once said. */
  shown: PositionEvent[];
  /** Every checker event, in order. */
  checker: CheckerEvent[];
  /** The confirmed mistakes, as "class word". */
  mistakes: string[];
}
/** Frames a listener decodes at a time (one model window, 320 ms): how often time passing in silence is told. */
const WINDOW_FRAMES = 8;
/**
 * A memorization take of `passage` (free recitation when it is `null`) through the tracker and the checker. `step`
 * phonemes a push (3 by default); between pushes, the silence before the next phoneme is told (`Tracker.silence`) at
 * each listener window's end, as a session does. `inject` may add tracker events before the checker sees a push (it
 * gets the count of phonemes pushed so far).
 */
export function recite(
  s: Speech,
  o: { passage: [WordRef, WordRef] | null; refinements?: Fields; checker?: boolean; step?: number; inject?: (pushed: number, events: EngineEvent[]) => EngineEvent[] },
): Take {
  const declared = o.passage && { from: o.passage[0], to: o.passage[1] };
  const tracker = createTracker(index, declared
    ? { mode: "memorization", phrases, acquireMs: 1000, passage: declared, refinements: o.refinements }
    : { mode: "free", phrases, acquireMs: 1000, refinements: o.refinements });
  const checker = o.checker === false ? null : createChecker(index, { phrases, ...(declared && { passage: declared }), policy: { refinements: o.refinements } });
  const positions: PositionEvent[] = [];
  const shown: PositionEvent[] = [];
  const out: CheckerEvent[] = [];
  const step = o.step ?? 3;
  let frame = 0;
  for (let i = 0; i < s.phonemes.length; i += step) {
    const part = s.phonemes.slice(i, i + step);
    const evidence = withPosteriors({ inventory: index.inventoryId, frameMs: FRAME_MS, phonemes: part }, frame, s.phonemes[i + step]?.start);
    frame = evidence.decodedTo!;
    let events = tracker.push(evidence);
    if (o.inject) events = o.inject(i + part.length, events);
    for (const e of events) if (e.type === "position") positions.push(e);
    shown.push(...tracker.shown());
    if (checker) out.push(...checker.push(evidence, events));
    const next = s.phonemes[i + step]?.start;
    if (next !== undefined) {
      for (let f = (Math.floor(frame / WINDOW_FRAMES) + 1) * WINDOW_FRAMES; f <= next; f += WINDOW_FRAMES) shown.push(...tracker.silence(f));
    }
  }
  const last = tracker.end();
  for (const e of last) if (e.type === "position") positions.push(e);
  shown.push(...tracker.shown());
  if (checker) out.push(...checker.end(last));
  const mistakes = (checker?.mistakes() ?? []).map((e) => `${e.errorClass} ${key(e.location)}`);
  return { positions, shown, checker: out, mistakes };
}

/** When each word first shows: every spoken word of a locked position's run, from the run's start to its word. */
export function shownAt(positions: PositionEvent[]): Map<string, number> {
  const shown = new Map<string, number>();
  for (const p of positions) {
    if (p.state !== "locked" || !p.observed[0]) continue;
    for (let i = at(p.observed[0].span.start); i <= at(p.observed[0].word); i++) {
      if (spoken(i) && !shown.has(key(ref(i)))) shown.set(key(ref(i)), p.timeMs);
    }
  }
  return shown;
}
/**
 * The page at the take's end, as an app draws it: every spoken word of a locked position's run lit, from the run's
 * start to its word, and a word a position withdraws (`withdrawn`, after that position's own run) dark until a later
 * run covers it again. Each word lit at the end, with when it was first lit.
 */
export function drawn(positions: PositionEvent[]): Map<string, number> {
  const lit = new Map<string, number>();
  const dark = new Set<string>();
  for (const p of positions) {
    if (p.state === "locked" && p.observed[0]) {
      for (let i = at(p.observed[0].span.start); i <= at(p.observed[0].word); i++) {
        if (!spoken(i)) continue;
        if (!lit.has(key(ref(i)))) lit.set(key(ref(i)), p.timeMs);
        dark.delete(key(ref(i)));
      }
    }
    for (const w of p.withdrawn ?? []) if (lit.has(key(w))) dark.add(key(w));
  }
  return new Map([...lit].filter(([k]) => !dark.has(k)));
}
/** Every word a position withdrew, as "key@ms", in order. */
export const withdrawals = (positions: PositionEvent[]) => positions.flatMap((p) => (p.withdrawn ?? []).map((w) => `${key(w)}@${p.timeMs}`));
/**
 * `s` with a pause of `frames` after the first `n` phonemes of `word` ("surah:ayah:word"): the reciter stops inside
 * the word and goes on with the rest of it.
 */
export function breakAt(s: Speech, word: string, n: number, frames: number): Speech {
  const first = s.phonemes.findIndex((p) => p.start * FRAME_MS === s.saidAt.get(word));
  const cut = s.phonemes[first + n].start;
  return {
    phonemes: s.phonemes.map((p, i) => (i >= first + n ? { ...p, start: p.start + frames, end: p.end + frames } : p)),
    saidAt: new Map([...s.saidAt].map(([k, ms]) => [k, ms >= cut * FRAME_MS ? ms + frames * FRAME_MS : ms])),
  };
}
/** The first `n` phonemes of the `w`th spoken word of ayah `s:a`, said as written: a word begun. */
export const begin = (s: number, a: number, w: number, n: number): Piece => {
  const x = index.words[ayah(s, a)[w - 1]];
  return { said: index.phonemes.slice(x.start, x.start + n) };
};
/** A word as written, without the orthography's marks (tatweel, madd sign, small Quranic marks), the dagger alif as alif. */
const written = (wi: number) => index.wordText?.[wi]?.replace(/[\u0640\u0653\u06D6-\u06ED]/g, "").replace(/\u0670/g, "\u0627");
const sounds = (wi: number) => index.phonemes.slice(index.words[wi].start, index.words[wi].end);
/** Two words are the same word: the same text as written, or the same sounds. */
export const sameWord = (a: number, b: number) => (written(a) !== undefined && written(a) === written(b)) || sounds(a) === sounds(b);
/**
 * The words of `shown` shown before they were said, or never said, but for those an ayah said by then reads alike: by
 * the rule for ayahs that open with the same words, an ayah's word may show while another ayah's word at its place is
 * said, when the two ayahs open with the same words up to it (`sameWord`), or with the same sounds up to its end,
 * split into other words. Each as "key(ms shown minus ms said)", or "key(never said)".
 */
export function guessed(s: Speech, shown: Map<string, number>): string[] {
  const said = [...s.saidAt].map(([k, ms]) => ({ ms, wi: wordIndex.get(k)! }));
  const alike = (wi: number, t: number) => {
    const own = ayah(index.words[wi].surah, index.words[wi].ayah);
    const place = own.indexOf(wi);
    return said.some(({ ms, wi: w }) => {
      const x = index.words[w];
      if (ms > t || (x.surah === index.words[wi].surah && x.ayah === index.words[wi].ayah)) return false;
      const other = ayah(x.surah, x.ayah);
      const j = other.indexOf(w);
      if (j === place && own.slice(0, place + 1).every((v, i) => sameWord(v, other[i]))) return true;
      return own.slice(0, place + 1).map(sounds).join("") === other.slice(0, j + 1).map(sounds).join("");
    });
  };
  const out: string[] = [];
  for (const [k, t] of shown) {
    const ms = s.saidAt.get(k);
    if (ms !== undefined && t >= ms) continue;
    if (!alike(wordIndex.get(k)!, t)) out.push(ms === undefined ? `${k}(never said)` : `${k}(${t - ms})`);
  }
  return out;
}
/** The first locked position that reached `word` as its current word. */
export const reached = (t: Take, word: string) => t.positions.find((p) => p.state === "locked" && key(p.observed[0].word) === word);
export const decided = (t: Take, cls: string, word: string) =>
  t.checker.find((e) => e.state !== "suspected" && e.errorClass === cls && key(e.location) === word);
/** Ayahs `from`–`to` of surah `s`, each followed by a short pause, leaving out the ayahs in `skip`. */
export const ayahs = (s: number, from: number, to: number, skip: number[] = []): Piece[] => {
  const out: Piece[] = [];
  for (let a = from; a <= to; a++) if (!skip.includes(a)) out.push(say(s, a), pause(15));
  return out;
};
