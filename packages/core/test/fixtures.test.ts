import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { EngineEvent, PositionEvent, ReferenceAsset } from "../src/index.js";
import { diffExpectation, finalPosition, isAssertable, replay, syntheticFixture } from "../src/fixtures/index.js";
import type { Fixture } from "../src/fixtures/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v1.json", import.meta.url), "utf8"));
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
];

const allFixtures = [...tlogFixtures, ...synthetic];
const stripTrace = (events: EngineEvent[]) =>
  events.map((e) => (e.type === "position" ? (({ trace: _, ...rest }: PositionEvent) => rest)(e) : e));

describe("fixture harness", () => {
  it("has real tlog fixtures, none of them audio-verified yet", () => {
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
    expect(diffExpectation(f, replay(index, f))).toEqual([]);
  });

  it.each(tlogFixtures.filter((f) => !isAssertable(f)).map((f) => [f.id, f] as const))(
    "%s (unverified) matches its regression snapshot",
    (_, f) => {
      const final = finalPosition(replay(index, f));
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
    expect(final.trace?.shortlist).toHaveLength(1);
    expect(final.trace?.path?.every((o) => o.op === "match")).toBe(true);
  });

  it("records state transitions across chunks", () => {
    const events = replay(index, synthetic[1], { trace: true }) as PositionEvent[];
    const transitions = events.map((e) => e.trace!.transition);
    expect(transitions[0].from).toBeNull();
    for (let i = 1; i < transitions.length; i++) expect(transitions[i].from).toBe(transitions[i - 1].to);
    expect(transitions.at(-1)?.to).toBe("locked");
  });

  it.todo("shows a synthetic substitution in the traced alignment path (needs fuzzy locate, ticket 03)");
});
