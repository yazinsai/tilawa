# Tilawa

Offline, streaming Quran recitation engine: word-level position tracking and memorization-mistake detection, fully on-device (web / node / React Native).

This repo ships two things:

- **`@tilawa/core`** (`packages/core/`), the SDK. Its engine is vendored from the private research repo (`yazinsai/tilawa-reborn`) at the commit in `packages/core/src/engine/SOURCE.json`. ONNX Runtime is an optional peer dependency, one per platform subpath.
- **Web demo** (`web/`), the browser demo at tilawa.dev.

The older Python research/training harness lives under `lab/` and has its own `lab/AGENTS.md`.

## Layout

```
packages/core/            # @tilawa/core SDK (the shipped product)
  src/engine/             #   VENDORED research core (tracker, checker, events, reference). Never hand-edit.
  src/runtime/            #   VENDORED research sdk (listener, fbank, models, ort, web/node/RN loaders). Never hand-edit.
  src/index.ts            #   curated root barrel (named exports only)
  src/session.ts          #   session lifecycle wrapper + onAudio tap (contract §3)
  src/tilawa.ts           #   shared createTilawa plumbing; src/web.ts, node.ts, react-native.ts per platform
  src/report/             #   @tilawa/core/report
  assets/                 #   ref-v2.json, phrases-v1.json (exported as @tilawa/core/assets/*)
  scripts/sync-engine.mjs #   re-vendors src/engine + src/runtime from the pinned commit (scripts/pin.mjs)
  scripts/parity-golden.mjs  # regenerates test/fixtures/parity/golden.json from the research commit
  scripts/pack-smoke.mjs  #   npm pack -> install in a temp project -> import every subpath -> Node session
  test/                   #   vitest; test/engine + test/runtime are ported research tests
web/                      # demo (Vite)
lab/                      # Python research harness, see lab/AGENTS.md
```

`README.md` at the root is the SDK's published README. `packages/core/README.md`, `LICENSE` and `NOTICE.md` are gitignored copies made by `npm run sync-docs` at pack time; edit the root files.

## SDK rules

- Algorithm changes happen in the research repo, then come here with `node scripts/sync-engine.mjs --repo <research checkout> --commit <sha>` (bump `scripts/pin.mjs`), `npm run parity:golden -- --repo <checkout>`, and a CHANGELOG line. Local behaviour goes in wrapper modules outside `src/engine` and `src/runtime`.
- The root entry stays runtime-neutral: no `node:` or `onnxruntime-*` imports outside `src/web.ts`, `src/node.ts`, `src/react-native.ts` and their vendored loaders.
- The public contract is `@tilawa/core` 1.x: additive changes only.

## Build & test the SDK

```bash
cd packages/core
npm ci
npm run typecheck
npm run build
npx vitest run        # deterministic, no network (tiny fixture model)
npm run test:pack     # packed-tarball smoke test
```

All of these must be green before merge. CI (`.github/workflows/ci.yml`) runs them plus the demo build.

## Run the web demo

The demo is the SDK's regression guard: if it still recognizes recitation against `@tilawa/core`, the SDK is correct. It uses the built package (`file:../../packages/core`), so build core first.

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

- **SDK engine change** → make it in the research repo, then re-vendor (see SDK rules). **Wrapper/API change** → edit `packages/core/src/` outside the vendored dirs, add a deterministic test, keep `vitest` and `test:pack` green.
- **Demo change** → edit `web/frontend/src/`. Verify `npm run build` typechecks and the demo still recognizes recitation.
- **New model / matching strategy / training** → that's lab work. See `lab/AGENTS.md`. Nothing in `lab/` may import from `packages/` or `web/`, and vice versa.

## Worktree + merge discipline

Develop every change in a worktree under `./.worktrees/`, then merge back with `--no-ff`.

```bash
git worktree add .worktrees/<name> -b <name>
cd .worktrees/<name>
# ... implement, test (vitest + test:pack + demo build) ...
git commit                 # subject: "<area>: <what changed>" (≤72 chars); body: the why + before/after
git merge <name> --no-ff -m "Merge branch '<name>': ..."
git worktree remove .worktrees/<name>
```

Never skip hooks or bypass signing.
