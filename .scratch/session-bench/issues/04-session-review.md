# 04 Session items in the scoreboard review page

Status: resolved
Type: task
Blocked by: none

Spec: `../spec.md` (Scoreboard: "review page opens it at the event, with a few seconds of audio either side"). Code: `docs/benchmarks/scoreboard-review*.js`, `docs/benchmarks/scoreboard.html`, `tools/scoreboard_serve.py`, clip dump in `scoreboard-cli.ts`; guide `docs/benchmarks/scoreboard-review.md`.

## Acceptance

- `sessions-cli` (or scoreboard `--sessions`) writes a session dump into the clip-dump dir: per session, its audio path, spans/words, labelled events, non-errors, and every checker warning (all states with times) plus its assignment; position events are optional but the wrong-position events must be included.
- `scoreboard_serve.py` serves session audio read-only (new prefix, 127.0.0.1 only, like existing ones).
- Review page lists session incidents (false warnings, misses, wrong positions, non-error warnings) and plays from `max(0, t-3s)` to `t+3s` by default, not from the start; Quran words around the event are shown.
- Existing review storage/identity rules extended so session incidents have stable keys; v1/v2 dumps still load.
- `node --test tools/scoreboard-review.test.cjs` gains session-item tests and passes.
- Review-only change: no scoreboard row.

## Comments

Implemented on branch `session-review` (2026-10-13).

- `packages/core/src/bench/session-dump.ts` builds the dump. `sessions-cli` writes `sessions-<sha>-<listener>.json` and adds a `kind: "sessions"` entry to the clip-dump `index.json`. Pass `--clip-dump none` to skip it. Use `--audio-root` (default `.scratch/session-bench`) to set where audio paths are resolved from.
- Each session carries its audio path, spans, Quran words with text, labelled events, non-errors and context-keyed detail. That detail includes the score, every checker warning with all its states (times and confidence) and its assignment, and the wrong position events. Correct position events are omitted.
- The dump replays the engine in its own copy of the `runSession` loop, so `sessions.ts` is untouched. The CLI throws if the replayed hits, false warnings, non-error warnings or wrong positions differ from the `runSession` row. If ticket 03 changes `runSession` (the context adapter is shared, the loop is not), `sessionEvents` must follow.
- `scoreboard_serve.py` serves `/audio/sessions/` from `.scratch/session-bench`, read-only on 127.0.0.1. Symlinks pointing outside it are denied.
- On the review page, a session run lists four item types: false warnings, misses, wrong positions (consecutive ones within 3 s are merged) and non-error warnings. Playback runs from max(0, t−3 s) to t+3 s by default, where t is the first checker state, the labelled error start or the first wrong position. The labelled Quran words within that window are shown, and tapping one plays it. Keys use the content and the time, not checker ids (their `#seq` shifts between runs). Reviews use the existing v2 store and identity. Clip dumps (v1 and v2) load unchanged.
- The page was smoke-tested on the three QUD sessions: 31 items, a bounded window that stops at t+3 s, and switching between session and clip runs.
- Open: no tracker candidate trace for sessions; no run comparison between a session run and a clip run (it is blocked); `scoreboard-cli --sessions` was not added (`sessions-cli` writes the dump).
