# 09: Reacquisition and jumps

**What to build:** When local evidence weakens, the tracker broadens to global search, and it tells a mistake apart from a genuine jump. It keeps the expected and observed bookmarks distinct, according to the session mode.

- **Weak evidence:** when local fit degrades, the tracker keeps the local hypothesis alongside global candidates rather than dropping it.
- **Competing hypotheses:** a pronunciation or wrong-word error at the current location, a repetition or repair, or a jump. A jump commits only after evidence persists across a configurable span.
- **Bookmarks:**
  - The **observed** bookmark moves when a jump commits.
  - The **expected** bookmark moves only on an explicit passage change, or in free mode once a jump commits.
  - In memorization mode, a committed jump leaves the expected bookmark in place, so the gap stays visible to the future checker.
- **After `lost`:** the tracker re-acquires globally without a session restart.

**Blocked by:** 08.

**Status:** done

- [x] A single wrong word that matches another verse doesn't commit a jump.
- [x] A sustained distant jump commits after the persistence span, in both modes, with the bookmark behaviour each mode requires.
- [x] Noise, then `lost`, then resumed recitation elsewhere re-acquires the correct location.
- [x] Jump commit latency and false-jump count are reported per fixture.

## Comments

- Implemented in `packages/core/src/tracker.ts`, with tests in `packages/core/test/reacquisition.test.ts`. `jumpReport` in `src/fixtures` reports commit latency, false jumps and local flashes from the events alone.
- **Weak:** local tracking turns weak when its recent cost goes above 0.3 edits per phoneme. A global search, decoded-only, then runs on every phoneme over the latest 48 phonemes, starting 6 phonemes before the weakness was noticed. The local search keeps running. While weak, local events may only advance the current run. A local jump arc that starts a new run is held back, because those arcs chase evidence that may belong elsewhere. Weakness clears when the recent cost drops back to 0.3 or below.
- **Jump:** a global `locked` result more than 2 ayahs from the local bookmark, or in another surah, has to hold its location for `jumpPhonemes` consecutive phonemes (default 8). Local search then restarts there. Lock results within 2 ayahs are left to the local search's skip and restart arcs.
- **Lost:** `lost` is emitted while weak, once the window holds 24 or more phonemes, nothing fits globally, and the local cost is above `maxCost`. After that, global search runs on every phoneme over a rolling 48-phoneme window until it locks, and local tracking resumes there.
- **Expected bookmark:** free mode moves it with a committed jump or a re-acquisition. Memorization freezes it at the bookmark from before the weakness, and never moves it again: there's no passage-change input yet, because that belongs to the session layer. During `lost`, both modes keep the last trusted bookmark.
- Latencies on the fixtures: 2:255 to 36:1 commits in 1.42 s, and 18:1 to 67:1 in 1.6 s. There are 0 false jumps. The 2:255 to 36:1 case has 1 local flash: 2:254:1 shows for 3 phonemes at the jump onset, through the existing local skip arc, before the weak phase starts.
- Posteriors now rerank only the first acquisition. They're dropped at lock, so later global searches use decoded output only.
- Known limits:
  - One phoneme of good local fit resets the jump persistence count.
  - Nearby skips of 1 to 2 ayahs with errors are never committed as jumps.
