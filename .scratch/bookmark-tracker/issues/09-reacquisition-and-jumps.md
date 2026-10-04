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

**Status:** ready-for-agent

- [ ] A single wrong word that matches another verse doesn't commit a jump.
- [ ] A sustained distant jump commits after the persistence span, in both modes, with the bookmark behaviour each mode requires.
- [ ] Noise, then `lost`, then resumed recitation elsewhere re-acquires the correct location.
- [ ] Jump commit latency and false-jump count are reported per fixture.
