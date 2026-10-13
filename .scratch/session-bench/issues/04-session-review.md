# 04 Session items in the scoreboard review page

Status: ready-for-agent
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
