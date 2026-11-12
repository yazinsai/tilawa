import { describe, expect, test } from "vitest";
import { buildIncidents, type MomentAnswer } from "../../help/src/claims";
import type { Moment } from "../../help/src/moments";
import { bookmark, word } from "./fixtures";

let n = 0;
const uuid = () => `00000000-0000-4000-8000-${String(++n).padStart(12, "0")}`;
const at = { sessionMs: 60_000, wallMs: 1_700_000_100_000 };
const ans = (question: MomentAnswer["question"], answer: MomentAnswer["answer"], extra: Partial<MomentAnswer> = {}): MomentAnswer => ({ question, answer, answeredAt: at, ...extra });
const moment = (id: string, kind: Moment["kind"], question: Moment["question"], startMs: number, extra: Partial<Moment> = {}): Moment => ({
  id,
  kind,
  question,
  atMs: startMs + 6000,
  clip: { startMs, endMs: startMs + 8000 },
  location: word(1, 1),
  eventIds: [],
  bookmarkIds: [],
  ...extra,
});

describe("buildIncidents", () => {
  test("each moment kind maps to its incident kind with the reader's claim", () => {
    const b = bookmark("11111111-1111-4111-8111-111111111111", 26_000);
    const moments = [
      moment("m1", "warning", "made_mistake", 0, { eventIds: ["w1"], confirmed: true }),
      moment("m2", "stumble", "made_mistake", 20_000, { bookmarkIds: [b.id] }),
      moment("m3", "position", "wrong_ayah", 30_000),
      moment("m4", "sample", "reading_correct", 40_000),
      moment("x1", "marked", "made_mistake", 50_000),
    ];
    const out = buildIncidents(
      {
        moments,
        answers: { m1: ans("made_mistake", "no"), m2: ans("made_mistake", "yes"), m3: ans("wrong_ayah", "unsure"), m4: ans("reading_correct", "no"), x1: ans("made_mistake", "skipped") },
        bookmarks: [b],
        durationMs: 60_000,
      },
      uuid,
    );
    expect(out.map((i) => [i.kind, i.claim.question, i.claim.answer])).toEqual([
      ["warning", "made_mistake", "no"],
      ["stumble", "made_mistake", "yes"],
      ["wrong_position", "wrong_ayah", "unsure"],
      ["unflagged_sample", "reading_correct", "no"],
      ["missed_mistake", "made_mistake", "skipped"],
    ]);
    expect(out[0].eventIds).toEqual(["w1"]);
    // The bookmark keeps its own id and tap time.
    expect(out[1]).toMatchObject({ id: b.id, markedAt: b.markedAt, eventIds: [] });
    expect(out[0].markedAt).toEqual(at);
    expect(out[0]).toMatchObject({ startMs: 0, endMs: 8000 });
  });

  test("a warning the reader agrees with is still only their claim", () => {
    const out = buildIncidents({ moments: [moment("m1", "warning", "made_mistake", 0, { eventIds: ["w"] })], answers: { m1: ans("made_mistake", "yes") }, bookmarks: [], durationMs: 10_000 }, uuid);
    expect(out).toHaveLength(1);
    expect(out[0].claim).toEqual({ question: "made_mistake", answer: "yes" });
    expect(Object.keys(out[0]).sort()).toEqual(["claim", "endMs", "eventIds", "id", "kind", "markedAt", "startMs"]);
  });

  test("moments never reached produce nothing; unflagged audio is never labelled", () => {
    const out = buildIncidents({ moments: [moment("m1", "sample", "reading_correct", 0)], answers: {}, bookmarks: [], durationMs: 10_000 }, uuid);
    expect(out).toEqual([]);
  });

  test("unasked bookmarks are sent unanswered", () => {
    const b = bookmark("22222222-2222-4222-8222-222222222222", 3000);
    const out = buildIncidents({ moments: [], answers: {}, bookmarks: [b], durationMs: 10_000 }, uuid);
    expect(out).toEqual([{ ...b, startMs: 0, endMs: 5000, claim: { answer: null } }]);
  });

  test("a bookmark folded into a warning shares the answer and stays a stumble", () => {
    const b = bookmark("33333333-3333-4333-8333-333333333333", 7000);
    const out = buildIncidents(
      { moments: [moment("m1", "warning", "made_mistake", 0, { eventIds: ["w"], bookmarkIds: [b.id] })], answers: { m1: ans("made_mistake", "unsure") }, bookmarks: [b], durationMs: 10_000 },
      uuid,
    );
    expect(out.map((i) => [i.kind, i.claim.answer])).toEqual([
      ["warning", "unsure"],
      ["stumble", "unsure"],
    ]);
    expect(out[1].id).toBe(b.id);
  });

  test("'we showed the wrong ayah' adds a wrong_position claim", () => {
    const wrong = { sessionMs: 60_000, wallMs: 1_700_000_200_000 };
    const out = buildIncidents(
      { moments: [moment("m1", "sample", "reading_correct", 0)], answers: { m1: ans("reading_correct", "yes") }, wrongAyah: { m1: wrong }, bookmarks: [], durationMs: 10_000 },
      uuid,
    );
    expect(out.map((i) => i.kind).sort()).toEqual(["unflagged_sample", "wrong_position"]);
    expect(out.find((i) => i.kind === "wrong_position")).toMatchObject({ claim: { question: "wrong_ayah", answer: "yes" }, markedAt: wrong });
  });

  test("advanced text and notes are trimmed, capped and only present when given", () => {
    const long = "ا".repeat(2500);
    const out = buildIncidents(
      { moments: [moment("m1", "stumble", "made_mistake", 0, { bookmarkIds: ["b"] })], answers: { m1: ans("made_mistake", "yes", { text: `  ${long} `, note: "   " }) }, bookmarks: [bookmark("b", 5000)], durationMs: 10_000 },
      uuid,
    );
    expect(out[0].claim.text).toHaveLength(2000);
    expect(out[0].claim).not.toHaveProperty("note");
  });

  test("spans are clamped into the take", () => {
    const out = buildIncidents({ moments: [moment("m1", "marked", "made_mistake", 5000)], answers: { m1: ans("made_mistake", "yes") }, bookmarks: [], durationMs: 9000 }, uuid);
    expect(out[0]).toMatchObject({ startMs: 5000, endMs: 9000 });
  });
});
