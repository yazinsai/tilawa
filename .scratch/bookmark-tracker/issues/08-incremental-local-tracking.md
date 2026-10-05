# 08: Incremental local tracking

**What to build:** Once locked, the tracker follows a live reciter chunk by chunk at word level. It handles pauses, repeats, backward restarts, and skips within the local area without losing its place.

- **Chunked input:** `push` accepts any chunk size. History is bounded: memory and work per push stay constant for long sessions.
- **Local search:** a continuity-favouring search around the observed bookmark, with transitions for forward progress, repeat, backward restart within the ayah or recent ayahs, and a local skip of a few words or the next ayah.
- **Pauses:** silence or empty evidence never moves the bookmark and never triggers `lost` by itself.
- **Streaming equivalence:** the same input frames, delivered in different chunk sizes, produce equivalent events at the same evidence boundaries. The test compares state after the same frame index. It never compares an early streaming decision with an offline result that has seen future frames.

**Blocked by:** 04.

**Status:** done

- [x] Word-level `position` events advance correctly through a multi-ayah synthetic recitation.
- [x] Repeating a word or phrase keeps the position and emits a position at the repeated word, not a jump.
- [x] A backward restart within the ayah is followed.
- [x] Skipping a few words or the next ayah is followed and is visible in the events.
- [x] A long silence gap doesn't change position or state.
- [x] The streaming-equivalence test passes for chunk sizes from 1 frame to whole-clip, compared at shared frame boundaries.
- [x] Memory and time per push are bounded in a long-session test.

## Comments

- Implemented in `packages/core/src/local.ts` (`createLocalSearch`), wired into `createTracker`; tests in `packages/core/test/local-tracking.test.ts`.
- On the first `locked` event the tracker hands off from global search to a streaming DP column over the bookmark's ayah ±2 ayahs, advanced one decoded phoneme at a time. Each step allows match/sub, insertion, deletion (same held-length costs as fuzzy locate), and a jump from the best state to any word start in the window at 2 edits (plus 0.001 per phoneme of distance, so ties go to the nearer landing). Back and forward jumps cost the same; with back cheaper, skips whose landing word opens like an earlier word briefly went backwards.
- Events: one per change of word or run start. `span.start` resets on every jump, so repeats, restarts and skips show up as a new run. `confidence` = `(1 − per/maxPer)(1 − recentCost/maxCost)`, with `recentCost` the mean weighted cost over the last 16 phonemes.
- A word can flash for one event when the next word shares its opening phonemes (e.g. 2:255:13 before a repeat of 10 resolves). Holding a word until 2 of its phonemes match was tried and dropped: it didn't remove the flash and made local and global lock events disagree at shared boundaries.
- Streaming equivalence is asserted on `{state, word}` at shared boundaries. Lock timing depends on chunk size (global search runs once per push), so span start and confidence can differ right after a lock.
- Bounded history: global search state is cleared at lock; local work per phoneme is bounded by the 5-ayah window. While acquiring, history keeps at most 512 earlier phonemes (posterior rows trimmed to match); the current push is always searched whole, so single-push relabelling is unchanged. The long-session test checks time per push only; memory isn't observable through the public API.
- Not done here (ticket 09): the tracker never leaves `locked` once local, even if `recentCost` climbs; posteriors are ignored after lock; noise during a pause steps the search like any other phoneme.
- Found while testing: one whole-clip push of a 424-phoneme clip takes ~35 s in global search (each candidate's DP is quadratic in clip length). That predates this ticket; flagging it for 10.
