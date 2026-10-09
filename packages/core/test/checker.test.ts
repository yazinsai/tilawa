import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, createTracker, loadReferenceIndex } from "../src/index.js";
import type { CheckerEvent, CheckerOptions, DecodedPhoneme, PhonemeEvidence, PhraseAsset, ReferenceAsset, WordRef } from "../src/index.js";
import { isAssertable, replayChecked, syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit, Fixture } from "../src/fixtures/index.js";
import { alignSubstring } from "../src/fuzzy.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
/** A word that occurs nowhere near 2:255, so it can't be read as a repeat or a skip. */
const ALIEN = text(w(108, 1, 3));

const from = w(2, 255, 1);
const at = (word: WordRef) => find(word).start - find(from).start;
const target = w(2, 255, 23);
const replaceWord = (word: WordRef, by: string): Edit[] => [
  { kind: "del", at: at(word), count: text(word).length },
  { kind: "ins", at: at(word), text: by },
];

const run = (edits: Edit[], o: { to?: WordRef; chunkSize?: number; checker?: CheckerOptions } = {}) =>
  replayChecked(index, syntheticFixture(index, { id: "c", from, to: o.to ?? w(2, 256, 10), edits, chunkSize: o.chunkSize ?? 8 }), {}, o.checker);

const summary = (events: CheckerEvent[]) => events.map((e) => `${e.state} ${e.errorClass} ${key(e.location)}`);
const confirmed = (events: CheckerEvent[]) => events.filter((e) => e.state === "confirmed");

describe("event schema", () => {
  it("emits versioned checker events with class, location and the expected text", () => {
    const events = run(replaceWord(target, ALIEN));
    expect(events.length).toBeGreaterThan(0);
    for (const e of events) {
      expect(e).toMatchObject({ type: "checker", v: 1, errorClass: "wrong_word", location: target, expected: text(target) });
      expect(ALIEN).toContain(e.observed);
      expect(e.id).toBe(events[0].id);
      expect(e.confidence).toBeGreaterThan(0.9);
    }
  });
});

describe("clean recitation", () => {
  it("emits nothing", () => {
    expect(run([])).toEqual([]);
  });

  it("ignores a single mispronounced phoneme and a confusable pair", () => {
    expect(run([{ kind: "sub", at: at(target) + 2, to: "س" }])).toEqual([]);
    expect(run([{ kind: "sub", at: at(w(2, 255, 16)), to: "ه" }])).toEqual([]);
  });
});

describe("words without consonants", () => {
  it("are never wrong words: a held ghunna heard as ن is how it's said", () => {
    const start = w(67, 16, 1);
    const ghunna = find(w(67, 16, 3));
    expect(index.phonemes.slice(ghunna.start, ghunna.end)).toBe("ںںں");
    const f = syntheticFixture(index, { id: "c", from: start, to: w(67, 17, 5), edits: [{ kind: "del", at: ghunna.start - find(start).start, count: 3 }, { kind: "ins", at: ghunna.start - find(start).start, text: "ن" }], chunkSize: 8 });
    expect(confirmed(replayChecked(index, f, {}, { passage: { from: start, to: w(67, 17, 5) } }))).toEqual([]);
  });
});

describe("memorization errors", () => {
  it("confirms a wrong word", () => {
    expect(summary(run(replaceWord(target, ALIEN)))).toEqual(["suspected wrong_word 2:255:23", "confirmed wrong_word 2:255:23"]);
  });

  it("confirms an omitted word once the next ayah ends", () => {
    const events = run([{ kind: "del", at: at(target), count: text(target).length }], { to: w(2, 257, 5) });
    expect(summary(events)).toEqual(["suspected omitted_word 2:255:23", "confirmed omitted_word 2:255:23"]);
    // Confirmed at 2:257, not at the end of the session.
    const ended = syntheticFixture(index, { id: "c", from, to: w(2, 257, 5) }).evidence[0].phonemes.at(-1)!.end;
    expect(events[1].timeMs).toBeLessThan(ended * 20);
  });

  it("doesn't count vowels and a glottal stop between words as an added word", () => {
    expect(confirmed(run([{ kind: "ins", at: at(target), text: "ااءِ" }]))).toEqual([]);
  });

  it("confirms an added word", () => {
    expect(summary(run([{ kind: "ins", at: at(target), text: ALIEN }]))).toEqual(["suspected added_word 2:255:23", "confirmed added_word 2:255:23"]);
  });
});

describe("self-correction", () => {
  // A wrong word, then a restart on the right one: يسمع for يشفع.
  const attempt = text(target).replace("ش", "س").replace("ف", "م");

  it("goes suspected, then resolved", () => {
    expect(summary(run([{ kind: "ins", at: at(target), text: attempt }], { chunkSize: 2 }))).toEqual(["suspected wrong_word 2:255:23", "resolved wrong_word 2:255:23"]);
  });

  it("resolves a restart from an earlier word too", () => {
    const edits: Edit[] = [
      { kind: "repeat", from: at(w(2, 255, 22)), to: at(target) },
      { kind: "ins", at: at(target), text: attempt },
    ];
    expect(summary(run(edits, { chunkSize: 2 }))).toEqual(["suspected wrong_word 2:255:23", "resolved wrong_word 2:255:23"]);
  });

  it("counts as a mistake only when the policy says so", { timeout: 30_000 }, () => {
    const fixture = syntheticFixture(index, { id: "c", from, to: w(2, 256, 10), edits: [{ kind: "ins", at: at(target), text: attempt }], chunkSize: 2 });
    const mistakes = (countResolved?: boolean) => {
      const tracker = createTracker(index);
      const checker = createChecker(index, { policy: { countResolved } });
      for (const e of fixture.evidence) checker.push(e, tracker.push(e));
      checker.end(tracker.end());
      return checker.mistakes();
    };
    expect(mistakes()).toEqual([]);
    expect(summary(mistakes(true))).toEqual(["resolved wrong_word 2:255:23"]);
  });
});

describe("repeats", () => {
  it("emit nothing, short or long", () => {
    expect(run([{ kind: "repeat", from: at(w(2, 255, 10)), to: at(w(2, 255, 13)) }])).toEqual([]);
    expect(run([{ kind: "repeat", from: at(w(2, 255, 22)), to: at(w(2, 255, 24)) }])).toEqual([]);
    expect(run([{ kind: "repeat", from: at(w(2, 255, 1)), to: at(w(2, 255, 20)) }])).toEqual([]);
  });
});

describe("skips", () => {
  it("resolve when the skipped words are recited before the next ayah ends", () => {
    // 2:255:22–24 skipped, then the reciter goes back for them and carries on.
    const edits: Edit[] = [
      { kind: "repeat", from: at(w(2, 255, 22)), to: at(w(2, 255, 27)) },
      { kind: "del", at: at(w(2, 255, 22)), count: at(w(2, 255, 25)) - at(w(2, 255, 22)) },
    ];
    expect(summary(run(edits))).toEqual(["suspected skipped_passage 2:255:22", "resolved skipped_passage 2:255:22"]);
  });

  it("are placed on the skipped words when the word before them recurs after them", () => {
    // قل أعوذ برب الناس [ملك الناس] إله الناس: the first الناس is 114:1:4, not 114:2:2.
    const nas = w(114, 1, 1);
    const skipped = [find(w(114, 2, 1)), find(w(114, 2, 2))];
    const f = syntheticFixture(index, { id: "c", from: nas, to: w(114, 3, 2), edits: [{ kind: "del", at: skipped[0].start - find(nas).start, count: skipped[1].end - skipped[0].start }], chunkSize: 8 });
    const events = replayChecked(index, f, {}, { passage: { from: nas, to: w(114, 3, 2) } });
    expect(new Set(events.map((e) => `${e.errorClass} ${key(e.location)}`))).toEqual(new Set(["omitted_word 114:2:1"]));
  });

  it("are placed on the skipped ayah when its first word recurs after it", () => {
    // ما تعبدون [ولا أنتم عابدون ما أعبد] ولا أنا: the ولا heard is 109:4:1.
    const from109 = w(109, 2, 1);
    const skipped = [find(w(109, 3, 1)), find(w(109, 3, 5))];
    const f = syntheticFixture(index, { id: "c", from: from109, to: w(109, 5, 5), edits: [{ kind: "del", at: skipped[0].start - find(from109).start, count: skipped[1].end - skipped[0].start }], chunkSize: 8 });
    const events = replayChecked(index, f, {}, { passage: { from: from109, to: w(109, 5, 5) } });
    expect(new Set(events.map((e) => `${e.errorClass} ${key(e.location)}`))).toEqual(new Set(["skipped_passage 109:3:1"]));
  });

  it("confirm when they are not", () => {
    const edits: Edit[] = [{ kind: "del", at: at(w(2, 255, 30)), count: at(w(2, 255, 40)) - at(w(2, 255, 30)) }];
    expect(summary(run(edits, { to: w(2, 257, 5) }))).toEqual(["suspected skipped_passage 2:255:30", "confirmed skipped_passage 2:255:30"]);
  });
});

describe("confirm delay", () => {
  const edits = replaceWord(target, ALIEN);
  const timing = (checker: CheckerOptions) => {
    const events = run(edits, { chunkSize: 2, checker });
    const s = events.find((e) => e.state === "suspected")!;
    const c = events.find((e) => e.state === "confirmed")!;
    return c.timeMs - s.timeMs;
  };

  it("waits the configured delay", { timeout: 30_000 }, () => {
    expect(timing({ policy: { confirm: { after: "delay", ms: 2000 } } })).toBeGreaterThanOrEqual(2000);
    expect(timing({ policy: { confirm: { after: "delay", ms: 3000 } } })).toBeGreaterThanOrEqual(3000);
    expect(timing({ policy: { confirm: { after: "delay", ms: 3000 } } })).toBeLessThan(3500);
  });

  it("or confirms at the next word's onset", () => {
    // Three phonemes into the next word, at 60 ms each.
    expect(timing({ policy: { confirm: { after: "next-word" } } })).toBeLessThan(500);
  });
});

describe("unclear evidence", () => {
  it("ends uncertain on low decoder confidence", () => {
    const f = syntheticFixture(index, { id: "c", from, to: w(2, 256, 10), edits: replaceWord(target, ALIEN), chunkSize: 8 });
    const start = at(target);
    let k = 0;
    const evidence = f.evidence.map((e) => ({
      ...e,
      phonemes: e.phonemes.map((p) => (k++ >= start && k <= start + ALIEN.length ? { ...p, conf: 0.3 } : p)),
    }));
    const events = replayChecked(index, { ...f, evidence });
    expect(summary(events)).toEqual(["suspected wrong_word 2:255:23", "uncertain wrong_word 2:255:23"]);
  });

  it("ends uncertain when the tracker stops being locked", () => {
    // The wrong word, then noise long enough for the tracker to lose its lock.
    const edits: Edit[] = [...replaceWord(target, ALIEN), { kind: "noise", at: at(w(2, 255, 24)), length: 60, seed: 7 }];
    const events = run(edits, { checker: { policy: { confirm: { after: "delay", ms: 3000 } } } });
    expect(confirmed(events)).toEqual([]);
    expect(events.find((e) => key(e.location) === "2:255:23")?.state).toBe("suspected");
    expect(events.some((e) => e.state === "uncertain")).toBe(true);
  });

  it("leaves a wrong word uncertain when the session ends inside the confirm delay", () => {
    const events = run(replaceWord(w(2, 255, 48), ALIEN), { to: w(2, 255, 50) });
    expect(confirmed(events)).toEqual([]);
  });

  it("never confirms a wrong word the session ended on", () => {
    // لعظيم said as لعزيز, the last word.
    const last = at(w(2, 255, 50));
    const events = run([{ kind: "sub", at: last + 2, to: "ز" }, { kind: "sub", at: last + 6, to: "ز" }], { to: w(2, 255, 50) });
    expect(confirmed(events)).toEqual([]);
  });
});

describe("local-window confirmation", () => {
  // يسمع for يشفع: two consonants decoded wrong.
  const misheard = text(target).replace("ش", "س").replace("ف", "م");
  const edits: Edit[] = [
    { kind: "sub", at: at(target) + text(target).indexOf("ش"), to: "س" },
    { kind: "sub", at: at(target) + text(target).indexOf("ف"), to: "م" },
  ];
  /** The fixture's chunks with posteriors: `alt(j)` gives decoded phoneme `j`'s distribution, otherwise the decode at 0.9. */
  const heard = (e: Edit[], alt: (j: number, p: string) => [string, number][] | undefined) => {
    const f = syntheticFixture(index, { id: "c", from, to: w(2, 256, 10), edits: e, chunkSize: 8 });
    const all = f.evidence.flatMap((x) => x.phonemes);
    const whole = withPosteriors(index, { ...f.evidence[0], phonemes: all }, (j) => alt(j, all[j].p));
    const { k, startFrame, ids, probs } = whole.posteriors!;
    let a = startFrame;
    const evidence = f.evidence.map((x, i) => {
      const b = i + 1 < f.evidence.length ? f.evidence[i + 1].phonemes[0].start : startFrame + whole.posteriors!.nFrames;
      const rows = { k, startFrame: a, nFrames: b - a, ids: Array.from(ids).slice((a - startFrame) * k, (b - startFrame) * k), probs: Array.from(probs).slice((a - startFrame) * k, (b - startFrame) * k) };
      a = b;
      return { ...x, posteriors: rows };
    });
    return { ...f, evidence };
  };
  const word = (j: number) => j >= at(target) && j < at(target) + text(target).length;
  const check = (f: Fixture, localWindow?: boolean) => confirmed(replayChecked(index, f, {}, { policy: { localWindow } }));

  it("doesn't confirm a wrong word when the audio supports the intended word almost as well", () => {
    // The decoder's runner-up at each misheard sound is the written one.
    const f = heard(edits, (j, p) => (word(j) && (p === "س" || p === "م") ? [[p, 0.5], [p === "س" ? "ش" : "ف", 0.45], ["", 0.03]] : undefined));
    expect(misheard).not.toBe(text(target));
    expect(summary(check(f, false))).toEqual(["confirmed wrong_word 2:255:23"]);
    expect(check(f, true)).toEqual([]);
  });

  it("still confirms a wrong word the audio doesn't support as written", () => {
    expect(summary(check(heard(replaceWord(target, ALIEN), () => undefined), true))).toEqual(["confirmed wrong_word 2:255:23"]);
    expect(summary(check(heard(edits, () => undefined), true))).toEqual(["confirmed wrong_word 2:255:23"]);
  });
});

describe("declared passage", () => {
  const passage = { from, to: w(2, 255, 50) };
  /** The checker alone, with the tracker's events optionally withheld. */
  const declared = (edits: Edit[], o: { to?: WordRef; withhold?: boolean; policy?: CheckerOptions["policy"] } = {}) => {
    const f = syntheticFixture(index, { id: "c", from, to: o.to ?? w(2, 255, 50), edits, chunkSize: 8 });
    const tracker = createTracker(index, { mode: "memorization", phrases });
    const checker = createChecker(index, { phrases, passage, policy: o.policy });
    const out: CheckerEvent[] = [];
    for (const e of f.evidence) {
      const events = tracker.push(e);
      out.push(...checker.push(e, o.withhold ? [] : events));
    }
    out.push(...checker.end(o.withhold ? [] : tracker.end()));
    return out;
  };

  it("aligns against the passage without a tracker lock", () => {
    expect(summary(confirmed(declared(replaceWord(target, ALIEN), { withhold: true })))).toEqual(["confirmed wrong_word 2:255:23"]);
  });

  it("emits nothing on clean recitation", () => {
    expect(confirmed(declared([]))).toEqual([]);
  });

  it("confirms at session end a wrong word a later word followed, inside the confirm delay", () => {
    const events = declared(replaceWord(w(2, 255, 48), ALIEN), { policy: { confirm: { after: "delay", ms: 60_000 } } });
    expect(summary(confirmed(events))).toEqual(["confirmed wrong_word 2:255:48"]);
  });

  it("follows a deliberate jump out of the passage and back without flagging it", () => {
    const away = [w(112, 1, 1), w(112, 4, 5)].map(find);
    const jump = index.phonemes.slice(away[0].start, away[1].end);
    expect(confirmed(declared([{ kind: "ins", at: at(w(2, 255, 10)), text: jump }]))).toEqual([]);
  });
});

const tlogDir = new URL("./fixtures/tlog/", import.meta.url);
const tlog: Fixture[] = readdirSync(tlogDir)
  .filter((f) => f.endsWith(".json"))
  .sort()
  .map((f) => JSON.parse(readFileSync(new URL(f, tlogDir), "utf8")));

describe("tlog fixtures", () => {
  it("emit no confirmed errors", { timeout: 60_000 }, () => {
    for (const f of tlog) expect(confirmed(replayChecked(index, f, { phrases }, { phrases })), f.id).toEqual([]);
  });
});

/**
 * Real teacher evidence with one word corrupted: the clip's phonemes are aligned to its reviewed span to find the word,
 * which is then replaced, dropped, or preceded by an alien word.
 */
describe("corrupted teacher evidence", () => {
  type Corruption = "wrong_word" | "omitted_word" | "added_word";
  const corrupt = (f: Fixture, word: number, kind: Corruption): PhonemeEvidence[] => {
    const ph = f.evidence.flatMap((e) => e.phonemes);
    const span = f.expect!.span!;
    const lo = find(span.start).start;
    const a = alignSubstring(index.phonemes, lo, find(span.end).end, ph.map((p) => p.p).join(""), () => true)!;
    const w0 = index.words[word];
    const inWord = a.path.filter((o) => o.op !== "ins" && o.op !== "del" && a.start + o.refAt >= w0.start && a.start + o.refAt < w0.end);
    const first = (inWord[0] as { obsAt: number }).obsAt;
    const last = (inWord.at(-1) as { obsAt: number }).obsAt;
    const alien = (frame: number): DecodedPhoneme[] => [...ALIEN].map((p) => ({ p, start: frame, end: frame, conf: 0.99 }));
    const out =
      kind === "wrong_word"
        ? [...ph.slice(0, first), ...alien(ph[first].start), ...ph.slice(last + 1)]
        : kind === "omitted_word"
          ? [...ph.slice(0, first), ...ph.slice(last + 1)]
          : [...ph.slice(0, first), ...alien(ph[first].start), ...ph.slice(first)];
    const chunks: PhonemeEvidence[] = [];
    for (let i = 0; i < out.length; i += 10) chunks.push({ inventory: f.evidence[0].inventory, frameMs: f.evidence[0].frameMs, phonemes: out.slice(i, i + 10) });
    return chunks;
  };

  /** Audio-verified clips spanning at least 5 words; the corrupted word is in the middle, two words clear of both edges. */
  const cases = tlog
    .filter((f) => isAssertable(f) && f.expect?.span)
    .map((f) => {
      const s = index.words.indexOf(find(f.expect!.span!.start));
      const e = index.words.indexOf(find(f.expect!.span!.end));
      let mid = Math.floor((s + e) / 2);
      while (mid < e - 2 && index.words[mid].end - index.words[mid].start < 4) mid++;
      return { f, s, e, mid };
    })
    .filter(({ s, e, mid }) => e - s + 1 >= 5 && mid >= s + 2 && mid <= e - 2 && index.words[mid].end - index.words[mid].start >= 4);

  it("has enough clips", () => {
    expect(cases.length).toBeGreaterThanOrEqual(8);
  });

  /** Where the tracker first locked, if it then stayed locked: the checker judges only words while locked. */
  const firstLock = (evidence: PhonemeEvidence[]) => {
    const t = createTracker(index, { phrases });
    const states = [...evidence.flatMap((c) => t.push(c)), ...t.end()].filter((e) => e.type === "position");
    const at = states.findIndex((e) => e.state === "locked");
    if (at === -1 || states.slice(at).some((e) => e.state !== "locked")) return null;
    return index.words.indexOf(find(states[at].observed[0].word));
  };

  for (const kind of ["wrong_word", "omitted_word", "added_word"] as const) {
    it(`confirms a ${kind.replace("_", " ")} at its location and nothing else`, { timeout: 60_000 }, () => {
      let eligible = 0;
      const misses: string[] = [];
      for (const { f, s, mid } of cases) {
        const evidence = corrupt(f, mid, kind);
        const events = confirmed(replayChecked(index, { ...f, evidence }, { phrases }, { phrases, policy: { confirm: { after: "next-word" } } }));
        const where = key(index.words[mid]);
        // Nothing confirmed anywhere else, for any corruption.
        expect(events.filter((e) => key(e.location) !== where).map((e) => `${e.errorClass} ${key(e.location)}`), f.id).toEqual([]);
        // Short clips often lock only after the corrupted word, lose the lock over it, or never lock: the tracker's part.
        const lock = firstLock(evidence);
        if (lock === null || lock < s || lock >= mid) continue;
        eligible++;
        if (!events.some((e) => key(e.location) === where && e.errorClass === kind)) misses.push(`${f.id} ${where}`);
      }
      expect(eligible).toBeGreaterThanOrEqual(5);
      expect(1 - misses.length / eligible, misses.join(", ")).toBeGreaterThanOrEqual(0.8);
    });
  }
});

describe("free recitation", () => {
  const free = (o: { from: WordRef; to: WordRef; edits?: Edit[] }) =>
    confirmed(replayChecked(index, syntheticFixture(index, { id: "c", from: o.from, to: o.to, edits: o.edits ?? [], chunkSize: 8 }), { phrases }, { phrases }));
  const offset = (start: WordRef, word: WordRef) => find(word).start - find(start).start;

  it("checks a wrong word that sends the tracker to another ayah against where it latched", () => {
    // 26:124 and 26:142 differ in their fifth word only; the tracker follows the wrong word there for one chunk.
    const start = w(26, 123, 1);
    const wrong = w(26, 124, 5);
    const edits: Edit[] = [
      { kind: "del", at: offset(start, wrong), count: text(wrong).length },
      { kind: "ins", at: offset(start, wrong), text: text(w(26, 142, 5)) },
    ];
    const f = syntheticFixture(index, { id: "c", from: start, to: w(26, 127, 4), edits, chunkSize: 8 });
    const tracker = createTracker(index, { phrases });
    const checker = createChecker(index, { phrases });
    const out: CheckerEvent[] = [];
    let sent = 0;
    let injected = false;
    for (const e of f.evidence) {
      const events = tracker.push(e);
      sent += e.phonemes.length;
      const inject = !injected && sent >= offset(start, w(26, 125, 2));
      if (inject) injected = true;
      const elsewhere = { word: w(26, 142, 5), span: { start: w(26, 142, 1), end: w(26, 142, 5) }, confidence: 0.5, cost: 0.1, per: 0.1 };
      const lock = { type: "position" as const, v: 1, timeMs: (e.phonemes.at(-1)!.end + 1) * e.frameMs, referenceVersion: index.version, state: "locked" as const, expected: elsewhere.word, observed: [elsewhere] };
      out.push(...checker.push(e, inject ? [lock] : events));
    }
    out.push(...checker.end(tracker.end()));
    expect(summary(confirmed(out))).toEqual(["confirmed wrong_word 26:124:5"]);
  });

  it("follows a sustained jump elsewhere without flagging it", () => {
    const away = [w(112, 1, 1), w(112, 4, 5)].map(find);
    const jump = index.phonemes.slice(away[0].start, away[1].end);
    expect(free({ from, to: w(2, 256, 10), edits: [{ kind: "ins", at: at(w(2, 255, 10)), text: jump }] })).toEqual([]);
  });

  it("follows on across a surah boundary", () => {
    expect(free({ from: w(113, 4, 1), to: w(114, 3, 2) })).toEqual([]);
  });
});

describe("a lock the tracker reaches only at the end of the session", () => {
  it("still gets the recitation checked", () => {
    const start = w(94, 5, 1);
    const skip = find(w(94, 5, 2));
    const f = syntheticFixture(index, { id: "end-lock", from: start, to: w(94, 5, 4), edits: [{ kind: "del", at: skip.start - find(start).start, count: skip.end - skip.start }], chunkSize: 4 });
    const checker = createChecker(index, { phrases });
    const events: CheckerEvent[] = [];
    for (const e of f.evidence) events.push(...checker.push(e, []));
    const end = f.evidence.at(-1)!.phonemes.at(-1)!.end;
    const lock = {
      type: "position" as const, v: 1, timeMs: (end + 1) * 20, referenceVersion: index.version, state: "locked" as const, expected: w(94, 5, 4),
      observed: [{ word: w(94, 5, 4), span: { start, end: w(94, 5, 4) }, confidence: 0.4, cost: 0.1, per: 0.1 }],
    };
    events.push(...checker.end([lock]));
    expect(summary(confirmed(events))).toEqual(["confirmed omitted_word 94:5:2"]);
  });
});
