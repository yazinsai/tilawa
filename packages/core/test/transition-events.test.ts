import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { AcousticTokenIdentity, CheckerState, DiscontinuityIdentity, MemorizationErrorClass } from "../src/events.js";
import { createTransitionExclusivity, discontinuityIdentity, sameDiscontinuity, transitionForEdit, transitionsBetween } from "../src/transition-events.js";

const source = { p: "ب", start: 10, end: 12 };
const resume = { p: "ت", start: 15, end: 17 };
const jump = (id = 1, parent: DiscontinuityIdentity | null = null, a: AcousticTokenIdentity = source, b: AcousticTokenIdentity = resume) =>
  discontinuityIdentity(1, id, a, b, 20, 60, parent);

describe("exclusive descriptions of an accepted discontinuity", () => {
  it.each<MemorizationErrorClass>(["wrong_word", "omitted_word", "added_word"])("retains a confirmed %s instead of its singleton ledger description", (cls) => {
    const events = createTransitionExclusivity();
    const identity = jump();
    events.record("confirmed", cls, 4, 4, identity);
    // A transported snapshot represents the same event even though it is a different object.
    expect(events.alternative(1, 8, [JSON.parse(JSON.stringify(identity))])).toBe(true);
  });

  it.each<CheckerState>(["suspected", "uncertain", "resolved"])("does not let a %s word issue replace a ledger report", (state) => {
    const events = createTransitionExclusivity();
    events.record(state, "omitted_word", 4, 4, jump());
    expect(events.alternative(1, 8, [jump()])).toBe(false);
  });

  it("keeps adjacent mistakes whose source or resume boundaries differ, even by one frame", () => {
    const events = createTransitionExclusivity();
    events.record("confirmed", "wrong_word", 4, 4, jump());
    for (const boundary of ["start", "end"] as const) {
      expect(events.alternative(1, 8, [jump(1, null, { ...source, [boundary]: source[boundary] + 1 })])).toBe(false);
      expect(events.alternative(1, 8, [jump(1, null, source, { ...resume, [boundary]: resume[boundary] + 1 })])).toBe(false);
    }
    expect(events.alternative(1, 8, [jump(1, null, { ...source, p: "ج" })])).toBe(false);
  });

  it("keeps a ledger skip independently supported by a second accepted transition", () => {
    const events = createTransitionExclusivity();
    const first = jump();
    const second = jump(2, first, resume, { p: "ج", start: 20, end: 22 });
    events.record("confirmed", "omitted_word", 4, 4, first);
    expect(events.alternative(1, 8, [first, second])).toBe(false);
    expect(events.alternative(1, 8, [second])).toBe(false);
  });

  it("requires identical ancestry and a word edit contained in the ledger span", () => {
    const events = createTransitionExclusivity();
    const original = jump(3, jump());
    events.record("confirmed", "omitted_word", 4, 5, original);
    expect(events.alternative(1, 8, [jump(3, jump(2))])).toBe(false);
    expect(events.alternative(5, 8, [original])).toBe(false);
    expect(events.alternative(1, 4, [original])).toBe(false);
  });

  it("withholds exclusivity when acoustic or transition support is missing", () => {
    const events = createTransitionExclusivity();
    events.record("confirmed", "omitted_word", 4, 4, jump());
    for (const support of [undefined, null, []]) expect(events.alternative(1, 8, support)).toBe(false);
    const unbounded = discontinuityIdentity(1, 2, null, resume, 20, 60, null);
    events.record("confirmed", "wrong_word", 4, 4, unbounded);
    expect(events.alternative(1, 8, [unbounded])).toBe(false);
    events.record("confirmed", "skipped_passage", 4, 4, jump(3));
    expect(events.alternative(1, 8, [jump(3)])).toBe(false);
  });
});

describe("acoustic transition ancestry", () => {
  it("copies and freezes acoustic identities and retains immutable ancestors", () => {
    const token = { ...source };
    const parent = jump();
    const identity = jump(2, parent, token);
    token.start = 99;
    expect(identity.source).toEqual(source);
    expect(identity.parent).toBe(parent);
    expect([identity, identity.source, identity.resume, parent].every(Object.isFrozen)).toBe(true);
  });

  it("projects all new accepted transitions between anchors, including after JSON transport", () => {
    const first = jump();
    const second = jump(2, first);
    const third = jump(3, second);
    expect(transitionsBetween({ generation: 1, head: first }, { generation: 1, head: third })).toEqual([second, third]);
    expect(transitionsBetween({ generation: 1, head: first }, JSON.parse(JSON.stringify({ generation: 1, head: third })))).toEqual([second, third]);
    expect(transitionsBetween({ generation: 1, head: null }, { generation: 1, head: third })).toHaveLength(3);
    expect(transitionsBetween({ generation: 1, head: first }, { generation: 1, head: first })).toEqual([]);
    expect(transitionsBetween(undefined, { generation: 1, head: first })).toBeNull();
    expect(transitionsBetween({ generation: 1, head: first }, { generation: 2, head: null })).toBeNull();
    expect(transitionsBetween({ generation: 1, head: third }, { generation: 1, head: second })).toBeNull();
  });

  it("binds word edits to exact acoustic boundaries and refuses duplicate boundary support", () => {
    const first = jump();
    expect(transitionForEdit({ generation: 1, head: first }, source, resume)).toBe(first);
    expect(transitionForEdit({ generation: 1, head: first }, source, { ...resume, start: 16 })).toBeUndefined();
    expect(transitionForEdit({ generation: 1, head: first }, undefined, resume)).toBeUndefined();
    expect(transitionForEdit({ generation: 1, head: jump(2, first) }, source, resume)).toBeUndefined();
    expect(sameDiscontinuity(first, jump(2))).toBe(false);
  });

  const index = loadReferenceIndex({ version: "synthetic", text: "synthetic", profile: {}, inventory: { id: "synthetic", tokens: "بتجدرزسش" },
    ayahs: [{ s: 1, a: 1, w: ["بت", "جدر", "زسش"], p: "بتجدرزسش", o: [0, 2, 5], iso: ["بت", "جدر", "زسش"] }] });

  it("preserves DP positions and costs while tracing jumps, and restores the original token frontier", () => {
    const start = { offset: 2, origin: 0, edits: 0 };
    const plain = createLocalSearch(index, start);
    let id = 0;
    const traced = createLocalSearch(index, start, { generation: 1, source, nextId: () => ++id });
    const tokens = [..."زسشبت"].map((p, i) => ({ p, start: 15 + i * 3, end: 17 + i * 3 }));
    const saved = traced.save();
    for (const token of tokens) {
      plain.step(token.p);
      traced.step(token.p, token);
      const { transition: _, ...position } = traced.position();
      const { transition: __, ...expected } = plain.position();
      expect(position).toEqual(expected);
    }
    const history = traced.position().transition;
    expect(history).not.toBeNull();
    const original = JSON.stringify(history);
    const boundaries = (identity: DiscontinuityIdentity | null): unknown => identity
      ? { source: identity.source, resume: identity.resume, from: identity.from, to: identity.to, parent: boundaries(identity.parent) }
      : null;
    traced.restore(saved);
    for (const token of tokens) traced.step(token.p, token);
    expect(boundaries(traced.position().transition)).toEqual(boundaries(history));
    expect(JSON.stringify(history)).toBe(original);
  });
});
