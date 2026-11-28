import { describe, expect, it } from "vitest";
import { DEFAULT, ayah, ayahs, guessed, passage, recite, shownAt, speech } from "./takes.js";

/**
 * A take that starts one ayah later than declared, with the default refinements: the passage is y−1..y+1 (y−1..y at a
 * surah's end) and the reciter starts at y, an ayah that opens with y−1's first word or words. `declaredStart` locks on
 * the shared words and lets the lock go where the two ayahs part. With `declaredMove` off, global search then has only
 * what was heard after the lock, so words said stay dark (most of them), a lost-place prompt comes during correct
 * recitation (12:74, 12:78, 13:29, 21:62, 23:59, 38:61, 53:50), or words of another place light (36:25). With the
 * defaults the lock moves to y on all the speech since the onset (of ayahs after y−1 that open alike, the first: 11:9
 * before 11:10), so every word said shows, nothing unsaid lights, no lost-place prompt comes, and the left-out ayah is
 * marked. Each take runs as the session's resolved policy (`DEFAULT`) at 1 and 3 phonemes a push. The takes are
 * synthetic phoneme streams built from the reference (`./takes.ts`).
 */
const LATER =
  "2:14 2:55 2:70 2:150 2:206 2:248 2:275 3:41 3:91 3:173 4:111 4:132 4:168 5:66 5:106 6:47 6:90 6:99 6:162 7:15 7:204 " +
  "8:65 8:75 9:85 9:115 10:35 10:43 10:75 11:9 12:49 12:72 12:74 12:78 13:29 14:11 15:60 16:47 16:71 16:81 16:95 17:33 " +
  "17:37 18:76 21:60 21:62 22:13 22:58 22:62 23:16 23:59 23:80 25:48 25:65 27:64 28:17 28:69 30:9 30:22 30:24 32:27 " +
  "33:57 36:16 36:25 37:14 37:68 37:166 38:61 40:68 41:41 49:2 49:12 52:39 53:45 53:50 54:38 54:53 57:13 58:12 59:24 " +
  "61:6 66:7 67:21 69:51 70:33 71:9 72:9 72:11 72:13 72:21 76:3 77:30 81:13 85:11 93:8 100:8 105:2 109:4 113:4";

describe("the default refinements, at a take that starts one ayah later than declared", () => {
  for (const [s, y] of LATER.split(" ").map((k) => k.split(":").map(Number))) {
    const b = ayah(s, y + 1).length ? y + 1 : y;
    it(`find the reciter where they are: ${s}:${y - 1}–${b} started at ${s}:${y}`, () => {
      const take = speech(ayahs(s, y, b));
      for (const step of [1, 3]) {
        const t = recite(take, { passage: passage(s, y - 1, b), step, refinements: DEFAULT });
        const shown = shownAt(t.positions);
        expect(t.positions.filter((e) => e.state === "lost").map((e) => e.timeMs), `step ${step}: lost at`).toEqual([]);
        expect(guessed(take, shown), `step ${step}: lit, never said`).toEqual([]);
        expect([...take.saidAt.keys()].filter((w) => !shown.has(w)), `step ${step}: said, never shown`).toEqual([]);
        expect(t.mistakes, `step ${step}`).toContain(`skipped_passage ${s}:${y - 1}:1`);
      }
    });
  }
});
