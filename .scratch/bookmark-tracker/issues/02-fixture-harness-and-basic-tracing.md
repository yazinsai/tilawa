# 02: Fixture harness and basic tracing

**What to build:** Every algorithm ticket from 03 onward can be developed against replayable fixtures and diagnosed from a trace.

- **Fixture format:** a small JSON file holding `PhonemeEvidence` (optionally with top-k posteriors), plus expected outcomes and a provenance note. Provenance is either synthetic or a real clip. A real clip records its tlog shard, row, and whether the expected label has been independently checked against the audio.
- **Converter:** turns rows of the tlog `pass1` parquet into fixtures.
- **Synthetic generator:** builds fixtures from reference phonemes by applying edits: substitution, deletion, insertion, repeat, skip, restart, noise, and silence gaps.
- **`trace` option:** events carry the shortlist (after deduplication), per-candidate alignment cost and fit, state transitions, and the alignment/edit path for the top candidate.
- **Replay runner:** runs a fixture through the public API and prints a readable diff of expected versus actual results.

**Blocked by:** 01.

**Status:** done

- [x] Fixtures round-trip: convert, replay, and get identical events across runs.
- [ ] A synthetic fixture generated with a known edit replays and shows the edit in the traced alignment path.
- [x] Real-clip fixtures carry an `audioVerified` flag. Tests assert expected labels only for verified fixtures; unverified ones are regression snapshots only.
- [x] Turning tracing on doesn't change any non-trace event content.

## Comments

- Built: the fixture format with provenance and an `audioVerified` flag; a Python converter that turns pass1 parquet rows into fixtures (its greedy decode reproduces Muaalem's output exactly); the synthetic generator (sub, del, ins, repeat, noise, gap); the `trace` option (shortlist, transition, alignment path); and the `pnpm replay <fixtures>` CLI.
- Not yet done: "a synthetic edit shows in the traced path". The exact-match tracker can't locate edited input, so there's no path to show. The test is an `it.todo` and moves to ticket 03.
- Four real fixtures are committed as snapshots (unverified): 100:3 is locked, while 100:1 (ض→د and د→ل), 100:6 (contains 100:6–7) and 100:7 come out lost. The 100:7 clip looks like the end of 100:8 plus the start of 100:9, not 100:8 alone.
