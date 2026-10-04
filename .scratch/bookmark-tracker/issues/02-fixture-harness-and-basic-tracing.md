# 02: Fixture harness and basic tracing

**What to build:** Every algorithm ticket from 03 onward can be developed against replayable fixtures and diagnosed from a trace.

- **Fixture format:** a small JSON file holding `PhonemeEvidence` (optionally with top-k posteriors), plus expected outcomes and a provenance note. Provenance is either synthetic or a real clip. A real clip records its tlog shard, row, and whether the expected label has been independently checked against the audio.
- **Converter:** turns rows of the tlog `pass1` parquet into fixtures.
- **Synthetic generator:** builds fixtures from reference phonemes by applying edits: substitution, deletion, insertion, repeat, skip, restart, noise, and silence gaps.
- **`trace` option:** events carry the shortlist (after deduplication), per-candidate alignment cost and fit, state transitions, and the alignment/edit path for the top candidate.
- **Replay runner:** runs a fixture through the public API and prints a readable diff of expected versus actual results.

**Blocked by:** 01.

**Status:** ready-for-agent

- [ ] Fixtures round-trip: convert, replay, and get identical events across runs.
- [ ] A synthetic fixture generated with a known edit replays and shows the edit in the traced alignment path.
- [ ] Real-clip fixtures carry an `audioVerified` flag. Tests assert expected labels only for verified fixtures; unverified ones are regression snapshots only.
- [ ] Turning tracing on doesn't change any non-trace event content.
