# Tilawa

Offline, streaming Quran recitation engine and SDK: word-level position tracking and memorization-mistake detection, fully on-device (web / node / React Native). See `PLAN.md`.

This repo is public. It holds the research engine and its lab, and ships two things built from it:

- **`@tilawa/core`** (`packages/tilawa-core/`), the published SDK. Its `src/engine/` and `src/runtime/` are copies of the research engine (`packages/core/`, `packages/sdk/`) at the commit in `packages/tilawa-core/scripts/pin.mjs`. ONNX Runtime is an optional peer dependency, one per platform subpath.
- **Web demo** (`web/`), tilawa.dev and /help, deployed with `git push dokku main:main`.

Never commit user audio, /help server data or anything from `.scratch/help-v2/`.

## Layout

```
packages/core/            # research engine (tracker, checker, events, reference); private workspace pkg
packages/sdk/             # research runtime (listener, fbank, models, ort, web/node/RN loaders)
packages/sample/          # research sample / SDK harness app
packages/probe/           # research device probe
packages/tilawa-core/     # @tilawa/core SDK (the shipped product, npm, own lockfile, outside the pnpm workspace)
  src/engine/             #   COPY of packages/core/src at the pin. Never hand-edit.
  src/runtime/            #   COPY of packages/sdk/src at the pin. Never hand-edit.
  src/index.ts            #   curated root barrel (named exports only)
  src/session.ts          #   session lifecycle wrapper + onAudio tap (contract §3)
  src/tilawa.ts           #   shared createTilawa plumbing; src/web.ts, node.ts, react-native.ts per platform
  src/report/             #   @tilawa/core/report
  assets/                 #   ref-v2.json, phrases-v1.json (exported as @tilawa/core/assets/*)
  scripts/pin.mjs         #   the engine commit the SDK ships
  scripts/sync-engine.mjs #   copies src/engine + src/runtime from the pinned commit; --check verifies
  scripts/parity-golden.mjs  # regenerates test/fixtures/parity/golden.json from the pinned commit
  scripts/pack-smoke.mjs  #   npm pack -> install in a temp project -> import every subpath -> Node session
web/frontend/             # demo + /help (Vite + node server)
modal/  tools/            # lab, autolab and the Modal jobs
docs/                     # SDK docs; docs/benchmarks (scoreboard, lab results), docs/agents
.scratch/                 # research notes, lab cards, issues
```

`README.md` at the root is the SDK's published README. `packages/tilawa-core/README.md`, `LICENSE` and `NOTICE.md` are gitignored copies made by `npm run sync-docs` at pack time; edit the root files.

## Release scoreboard

The scoreboard (`docs/benchmarks/scoreboard.md`) holds the release gates, checked before a release rather than per change. Day-to-day changes are lab experiments (below).

## Lab experiments

Any change that can move a lab number (listener, tracker, checker) is an experiment with a recorded result (`.scratch/lab/spec.md`): `python3 tools/lab.py new <id> --component ... --hypothesis '...' --expect-component N --expect-overall N` makes its worktree and card, and `python3 tools/lab.py run <worktree>` records its practice result and verdict. For hand-run experiments the user decides `lab fold` or `lab discard`; pass `fold --yes` only when the user has approved that fold. Autolab's experiments (`tools/autolab.py`) are folded or discarded by its gate (`tools/lab_gate.py`) straight onto `main`, tagged `autolab/fold/<id>`; the user can take one back out with `python3 tools/autolab.py revert <id>`. Fold-in updates the main line (`main`). See `tools/lab.py` for every command. Autolab is paused: don't start or resume it unless the user asks.

## Compute gotchas

- Heavy work runs on Modal, not the Mac.
- A job backgrounded inside a shell call (`nohup ... &`) dies when the call returns, taking its Modal app with it: the run stalls silently with partial outputs. Use the shell tool's own background mode for long `modal run`s.
- Module-level code in a Modal file also runs inside the container, where local paths don't exist.

## Agent skills

### Issue tracker

Issues and specs live as local markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) recorded as a `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.

### Audio failure reviews

When reviewing recitation labels, missed mistakes or false warnings, follow `docs/agents/audio-review.md`: automate incident and component triage first, use a calibrated independent audio opinion when available, and ask the user only for unresolved listening judgments and spot checks.

## SDK rules

- Algorithm changes happen in `packages/core` / `packages/sdk` (usually as lab experiments). They reach the SDK only through a deliberate promotion, one commit: bump `COMMIT` in `scripts/pin.mjs` to a `main` SHA, `npm run sync-engine`, `npm run parity:golden`, and a CHANGELOG line. Folds on `main` don't change the SDK by themselves. Local behaviour goes in wrapper modules outside `src/engine` and `src/runtime`.
- The root entry stays runtime-neutral: no `node:` or `onnxruntime-*` imports outside `src/web.ts`, `src/node.ts`, `src/react-native.ts` and their vendored loaders.
- The public contract is `@tilawa/core` 1.x: additive changes only.

## Build & test the SDK

```bash
cd packages/tilawa-core
npm ci
npm run typecheck
npm run build
npx vitest run        # deterministic, no network (tiny fixture model)
npm run test:pack     # packed-tarball smoke test
npm run sync-engine -- --check   # src/engine + src/runtime match the pin
```

All of these must be green before merge. CI (`.github/workflows/ci.yml`) runs them plus the demo build.

## Run the web demo

The demo is the SDK's regression guard: if it still recognizes recitation against `@tilawa/core`, the SDK is correct. It uses the built package (`file:../../packages/tilawa-core`), so build core first.

```bash
cd web/frontend
npm ci
npm run build                                   # tsc && vite build (/ and /help)
npm run assets:v1 -- --models-dir <exports>     # dist/v1: models, reference, ORT runtime (after the build)
npm run build:server && npm run start           # bundled node server (dist-server/index.mjs)
npx vitest run                                  # deterministic tests
TILAWA_MODELS_DIR=<exports> npx playwright test # opt-in: recites a real clip into the built demo
```

`Dockerfile` at the root builds and serves this demo (`--build-arg TILAWA_ASSETS_URL=...`, see its header).

## Making changes

- **SDK engine change** → change `packages/core` / `packages/sdk` (a lab experiment if it can move a lab number), then promote it (see SDK rules). **Wrapper/API change** → edit `packages/tilawa-core/src/` outside the vendored dirs, add a deterministic test, keep `vitest` and `test:pack` green.
- **Demo change** → edit `web/frontend/src/`. Verify `npm run build` typechecks and the demo still recognizes recitation.
- **New model / matching strategy / training** → lab work (see Lab experiments). Research code never imports from `packages/tilawa-core` or `web/`.

## Worktree + merge discipline

Lab experiments get their worktrees from `tools/lab.py`. Develop every other change in a worktree under `./.worktrees/`, then merge back with `--no-ff`.

```bash
git worktree add .worktrees/<name> -b <name>
cd .worktrees/<name>
# ... implement, test (vitest + test:pack + demo build) ...
git commit                 # subject: "<area>: <what changed>" (≤72 chars); body: the why + before/after
git merge <name> --no-ff -m "Merge branch '<name>': ..."
git worktree remove .worktrees/<name>
```

Never skip hooks or bypass signing.
