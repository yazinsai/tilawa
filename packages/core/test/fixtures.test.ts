import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { EngineEvent, PositionEvent, ReferenceAsset } from "../src/index.js";
import { diffExpectation, finalPosition, isAssertable, replay, syntheticFixture } from "../src/fixtures/index.js";
import type { Fixture } from "../src/fixtures/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);

const tlogDir = new URL("./fixtures/tlog/", import.meta.url);
const tlogFixtures: Fixture[] = readdirSync(tlogDir)
  .filter((f) => f.endsWith(".json"))
  .sort()
  .map((f) => JSON.parse(readFileSync(new URL(f, tlogDir), "utf8")));

const w = (surah: number, ayah: number, word: number) => ({ surah, ayah, word });

const synthetic: Fixture[] = [
  syntheticFixture(index, {
    id: "exact-1-2",
    from: w(1, 2, 1),
    to: w(1, 2, 4),
    expect: { state: "locked", span: { start: w(1, 2, 1), end: w(1, 2, 4) } },
  }),
  syntheticFixture(index, {
    id: "exact-2-255-chunked",
    from: w(2, 255, 3),
    to: w(2, 255, 12),
    chunkSize: 7,
    expect: { state: "locked", span: { start: w(2, 255, 3), end: w(2, 255, 12) } },
  }),
  syntheticFixture(index, { id: "gap-1-2", from: w(1, 2, 1), to: w(1, 2, 4), edits: [{ kind: "gap", at: 10, frames: 200 }] }),
  syntheticFixture(index, { id: "sub-1-2", from: w(1, 2, 1), to: w(1, 2, 4), edits: [{ kind: "sub", at: 3, to: "ه" }] }),
  ...(
    [
      ["fuzzy-sub", [{ kind: "sub", at: 12, to: "ق" }]],
      [
        "fuzzy-multi-sub",
        [
          { kind: "sub", at: 5, to: "ق" },
          { kind: "sub", at: 20, to: "ك" },
          { kind: "sub", at: 33, to: "ز" },
        ],
      ],
      ["fuzzy-del", [{ kind: "del", at: 10, count: 2 }]],
      ["fuzzy-ins", [{ kind: "ins", at: 15, text: "كم" }]],
      [
        "fuzzy-mixed",
        [
          { kind: "sub", at: 8, to: "ط" },
          { kind: "del", at: 25, count: 1 },
          { kind: "ins", at: 40, text: "ي" },
        ],
      ],
    ] as const
  ).map(([id, edits]) =>
    syntheticFixture(index, {
      id: `${id}-2-255`,
      from: w(2, 255, 3),
      to: w(2, 255, 12),
      edits: [...edits],
      expect: { state: "locked", span: { start: w(2, 255, 3), end: w(2, 255, 12) } },
    }),
  ),
  syntheticFixture(index, {
    id: "fuzzy-cross-ayah-100-6-7",
    from: w(100, 6, 1),
    to: w(100, 7, 3),
    edits: [{ kind: "sub", at: 7, to: "م" }],
    expect: { state: "locked", span: { start: w(100, 6, 1), end: w(100, 7, 3) } },
  }),
];

const allFixtures = [...tlogFixtures, ...synthetic];
const stripTrace = (events: EngineEvent[]) =>
  events.map((e) => (e.type === "position" ? (({ trace: _, ...rest }: PositionEvent) => rest)(e) : e));

const decodedOnly = (f: Fixture): Fixture => ({ ...f, evidence: f.evidence.map(({ posteriors: _, ...e }) => e) });

describe("fixture harness", () => {
  it("has real tlog fixtures", () => {
    expect(tlogFixtures.length).toBeGreaterThanOrEqual(4);
    for (const f of tlogFixtures) expect(f.provenance.kind).toBe("tlog");
  });

  it.each(allFixtures.map((f) => [f.id, f] as const))("%s replays deterministically through a JSON round trip", (_, f) => {
    const roundTripped: Fixture = JSON.parse(JSON.stringify(f));
    expect(replay(index, roundTripped)).toEqual(replay(index, f));
  });

  it.each(allFixtures.map((f) => [f.id, f] as const))("%s: tracing does not change non-trace event content", (_, f) => {
    expect(stripTrace(replay(index, f, { trace: true }))).toEqual(replay(index, f));
  });

  it.each(synthetic.filter(isAssertable).map((f) => [f.id, f] as const))("%s meets its expectation", (_, f) => {
    expect(diffExpectation(f, replay(index, f), index)).toEqual([]);
  });

  // Verified clips the tracker still gets wrong; each must move out of this list once fixed.
  const pending = new Set<string>([
    // Mid-word starts on short clips: the true ayah fits but isn't separated from look-alikes elsewhere.
    "tlog-clean-00000-of-00411-450",
    "tlog-clean-00000-of-00411-680",
    // Blind review (ticket 07): the teacher dropped 2+ words at a clip edge, so the span is short.
    ...[20, 28, 805, 925].map((r) => `tlog-clean-00000-of-00411-${r}`),
    // Blind review: a repeated ayah is stretched over the next one instead of staying put.
    "tlog-clean-00000-of-00411-597",
    // Blind review: the span runs 2 words past the reviewer's end.
    "tlog-clean-00000-of-00411-323",
    // Blind review: clean recitation the tracker won't lock (acquiring, ambiguous or lost).
    ...[7, 18, 53, 126, 201, 246, 536, 913].map((r) => `tlog-clean-00000-of-00411-${r}`),
  ]);
  const verified = tlogFixtures.filter(isAssertable);

  it.each(verified.filter((f) => !pending.has(f.id)).map((f) => [f.id, f] as const))("%s (audio-verified) meets its expectation", (_, f) => {
    expect(diffExpectation(f, replay(index, f), index)).toEqual([]);
  });

  it.fails.each(verified.filter((f) => pending.has(f.id)).map((f) => [f.id, f] as const))(
    "%s (audio-verified) meets its expectation — known failure",
    (_, f) => {
      expect(diffExpectation(f, replay(index, f), index)).toEqual([]);
    },
  );

  it("locks the low-volume 875 clip clear of the confidence floor despite its shortened ghunna", () => {
    const final = finalPosition(replay(index, tlogFixtures.find((f) => f.id === "tlog-clean-00000-of-00411-875")!))!;
    expect(final.state).toBe("locked");
    expect(final.observed[0].confidence).toBeGreaterThan(0.2);
  });

  it("reports the learner errors of the verified 100:1 clip in its PER and edit path", () => {
    const final = finalPosition(replay(index, tlogFixtures.find((f) => f.id === "tlog-clean-00000-of-00411-303")!, { trace: true }))!;
    expect(final.observed[0].per).toBeGreaterThan(0);
    const subs = final.trace!.path!.filter((o) => o.op === "sub").map((o) => o.op === "sub" && o.ref + o.obs);
    expect(subs).toEqual(expect.arrayContaining(["ضد", "دل"]));
  });

  it.each(tlogFixtures.filter((f) => !isAssertable(f)).map((f) => [f.id, f] as const))(
    "%s (unverified, decoded only) matches its regression snapshot",
    (_, f) => {
      const final = finalPosition(replay(index, decodedOnly(f)));
      expect({ state: final?.state, spans: final?.observed.map((c) => c.span) }).toMatchSnapshot();
    },
  );

  it("a silence gap does not change the located span", () => {
    const gap = finalPosition(replay(index, synthetic.find((f) => f.id === "gap-1-2")!));
    expect(gap?.state).toBe("locked");
    expect(gap?.observed[0].span).toEqual({ start: w(1, 2, 1), end: w(1, 2, 4) });
  });
});

describe("tracing", () => {
  it("records the shortlist, transition and an all-match path for an exact recitation", () => {
    const events = replay(index, synthetic[0], { trace: true }) as PositionEvent[];
    expect(events[0].trace?.transition).toEqual({ from: null, to: "locked" });
    const final = events.at(-1)!;
    expect(final.observed).toHaveLength(1);
    expect(final.trace?.shortlist[0]).toMatchObject({ span: final.observed[0].span, cost: 0, per: 0 });
    expect(final.trace?.path?.every((o) => o.op === "match")).toBe(true);
  });

  it("records state transitions across chunks", () => {
    const events = replay(index, synthetic[1], { trace: true }) as PositionEvent[];
    const transitions = events.map((e) => e.trace!.transition);
    expect(transitions[0].from).toBeNull();
    for (let i = 1; i < transitions.length; i++) expect(transitions[i].from).toBe(transitions[i - 1].to);
    expect(transitions.at(-1)?.to).toBe("locked");
  });

  it("shows a synthetic substitution in the traced alignment path", () => {
    const f = synthetic.find((s) => s.id === "fuzzy-sub-2-255")!;
    const final = finalPosition(replay(index, f, { trace: true }))!;
    const subs = final.trace!.path!.filter((o) => o.op !== "match");
    expect(subs).toEqual([expect.objectContaining({ op: "sub", obs: "ق", refAt: 12, obsAt: 12 })]);
  });
});
