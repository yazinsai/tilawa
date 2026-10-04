# 08: Incremental local tracking

**What to build:** Once locked, the tracker follows a live reciter chunk by chunk at word level. It handles pauses, repeats, backward restarts, and skips within the local area without losing its place.

- **Chunked input:** `push` accepts any chunk size. History is bounded: memory and work per push stay constant for long sessions.
- **Local search:** a continuity-favouring search around the observed bookmark, with transitions for forward progress, repeat, backward restart within the ayah or recent ayahs, and a local skip of a few words or the next ayah.
- **Pauses:** silence or empty evidence never moves the bookmark and never triggers `lost` by itself.
- **Streaming equivalence:** the same input frames, delivered in different chunk sizes, produce equivalent events at the same evidence boundaries. The test compares state after the same frame index. It never compares an early streaming decision with an offline result that has seen future frames.

**Blocked by:** 04.

**Status:** ready-for-agent

- [ ] Word-level `position` events advance correctly through a multi-ayah synthetic recitation.
- [ ] Repeating a word or phrase keeps the position and emits a position at the repeated word, not a jump.
- [ ] A backward restart within the ayah is followed.
- [ ] Skipping a few words or the next ayah is followed and is visible in the events.
- [ ] A long silence gap doesn't change position or state.
- [ ] The streaming-equivalence test passes for chunk sizes from 1 frame to whole-clip, compared at shared frame boundaries.
- [ ] Memory and time per push are bounded in a long-session test.
