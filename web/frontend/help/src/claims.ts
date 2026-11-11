// Turns the reader's answers into report incidents. Pure and deterministic given `uuid`.
//
// What an incident's kind means here (docs/help-collection.md, "What /help sends"):
//   warning          a warning we raised, put to the reader ("made_mistake"); answer "no" = they say it was false
//   stumble          a bookmark the reader tapped while reciting; answer null when they were never asked about it
//   wrong_position   "Did we show the wrong ayah?", asked or volunteered
//   unflagged_sample a random moment with no warning ("reading_correct"); never assumed clean
//   missed_mistake   a moment the reader marked while listening back
// Only the reader's own answers are stored. Nothing here infers a label.
import type { ClaimAnswer, Incident, IncidentClaim, IncidentKind } from "@tilawa/core/report";
import { clipAround, type Moment, type QuestionId } from "./moments";

export type Answer = Exclude<ClaimAnswer, null>;

export interface MomentAnswer {
  question: QuestionId;
  answer: Answer;
  answeredAt: { sessionMs: number; wallMs: number };
  /** Optional "what I actually recited" text from the advanced box. */
  text?: string;
  note?: string;
}

export interface ReviewResult {
  moments: readonly Moment[];
  /** Answers by moment id. Moments the reader never reached have none and produce no incident. */
  answers: Readonly<Record<string, MomentAnswer>>;
  /** Moment ids where the reader pressed "We showed the wrong ayah", with when. */
  wrongAyah?: Readonly<Record<string, { sessionMs: number; wallMs: number }>>;
  /** Every bookmark from the take (stumble incidents from `recorder.mark`). */
  bookmarks: readonly Incident[];
  durationMs: number;
}

const KIND: Record<Moment["kind"], IncidentKind> = {
  warning: "warning",
  stumble: "stumble",
  position: "wrong_position",
  sample: "unflagged_sample",
  marked: "missed_mistake",
};

const MAX_TEXT = 2000;

function claimOf(question: QuestionId, a: MomentAnswer | null): IncidentClaim {
  const claim: IncidentClaim = { question, answer: a ? a.answer : null };
  const text = a?.text?.trim();
  const note = a?.note?.trim();
  if (text) claim.text = text.slice(0, MAX_TEXT);
  if (note) claim.note = note.slice(0, MAX_TEXT);
  return claim;
}

function span(clip: { startMs: number; endMs: number }, durationMs: number) {
  const startMs = Math.max(0, Math.min(durationMs, Math.floor(clip.startMs)));
  const endMs = Math.max(startMs, Math.min(durationMs, Math.floor(clip.endMs)));
  return { startMs, endMs };
}

export function buildIncidents(review: ReviewResult, uuid: () => string = () => crypto.randomUUID()): Incident[] {
  const { durationMs } = review;
  const out: Incident[] = [];
  const asked = new Set<string>();
  const bookmarkById = new Map(review.bookmarks.map((b) => [b.id, b]));

  for (const m of review.moments) {
    const a = review.answers[m.id] ?? null;
    if (a) {
      const s = span(m.clip, durationMs);
      if (m.kind === "stumble") {
        // The reader's own bookmark: keep its id and tap time, add their answer.
        for (const id of m.bookmarkIds) {
          const b = bookmarkById.get(id);
          if (!b) continue;
          asked.add(id);
          out.push({ ...b, ...s, eventIds: [], markedAt: { ...b.markedAt }, claim: claimOf(m.question, a) });
        }
      } else {
        out.push({ id: uuid(), kind: KIND[m.kind], ...s, eventIds: [...m.eventIds], markedAt: { ...a.answeredAt }, claim: claimOf(m.question, a) });
        // Bookmarks folded into a warning or position moment share its answer but stay stumbles.
        for (const id of m.bookmarkIds) {
          const b = bookmarkById.get(id);
          if (!b) continue;
          asked.add(id);
          out.push({ ...b, ...s, eventIds: [], markedAt: { ...b.markedAt }, claim: claimOf(m.question, a) });
        }
      }
    }
    const wrong = review.wrongAyah?.[m.id];
    if (wrong && m.question !== "wrong_ayah") {
      out.push({
        id: uuid(),
        kind: "wrong_position",
        ...span(m.clip, durationMs),
        eventIds: [],
        markedAt: { ...wrong },
        claim: { question: "wrong_ayah", answer: "yes" },
      });
    }
  }

  // Bookmarks the review never reached still say "something happened here", unanswered.
  for (const b of review.bookmarks) {
    if (asked.has(b.id) || b.kind !== "stumble") continue;
    out.push({ ...b, ...span(clipAround(b.markedAt.sessionMs, durationMs), durationMs), eventIds: [], markedAt: { ...b.markedAt }, claim: { answer: null } });
  }
  return out.sort((x, y) => x.startMs - y.startMs).slice(0, 100); // stable: ties keep build order
}
