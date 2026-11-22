# Tilawa V1: execution handoff for Opus

Status: draft execution plan
Prepared: 2026-11-11
Lead: Claude Opus, using delegates for bounded implementation and verification

## Start here

Read this plan, inspect both repositories, then lead the work through the acceptance criteria below. The user requested this document before implementation; producing it does not mean implementation or publication has happened. When handed this document with an instruction to execute, start with workstream 00 and maintain a durable task ledger.

The objective is to replace the old public Tilawa engine with the successful experimental engine, ship the breaking change as V1 of the existing SDK, update tilawa.dev, and turn tilawa.dev/help into an accessible natural-recitation contribution flow backed by a reusable SDK reporting API and a reviewed dataset pipeline.

Do the integration and resolve routine engineering choices autonomously. Delegate implementation against agreed interfaces, review the results, and verify the integrated product. Bring the user concrete choices only where product policy, permissions, release readiness, or listening judgment genuinely requires them.

## Product decisions from the user

- Public website, demo, documentation and model distribution belong under **tilawa.dev**.
- **tilawa.dev/help** is the main place where people help report problems and contribute useful recordings.
- People should recite naturally, anywhere in the Quran. The engine finds their place. Collect naturally occurring mistakes and incorrect model feedback.
- The interface must work well for elderly people and people with little technical experience: large readable controls, clear language, one task at a time.
- Expose report capture through the SDK so our website and third-party apps can reuse it.
- Replace the previous engine in the existing public GitHub/npm product. This is a breaking V1 release, not a competing new product.
- Prepare a launch video and an X/Twitter post with the demo and SDK links.

Recommended initial scope: tracking plus memorization correction (wrong, omitted, added words and skipped passages). Pronunciation/tajweed classes remain explicitly unsupported unless their existing release requirements are met. Version the SDK, model, reference assets and reporting schema independently.

## Repositories and current state

Paths are observations from the preparation machine; resolve them again if executing elsewhere.

| Role | Local path | Public identity / observation |
| --- | --- | --- |
| Proven engine and research | `/Users/rock/ai/projects/tilawa` | Remote points to `yazinsai/tilawa-reborn` |
| Current handoff checkout | `/Users/rock/.t3/worktrees/tilawa/t3-b6edfd4d` | Branch `t3/b6edfd4d`; contains uncommitted simplified harness changes |
| Existing public product | `/Users/rock/ai/projects/offline-tarteel` | User identifies public repo as `github.com/yazinsai/tilawa`; local origin still says `yazinsai/offline-tarteel` |
| Existing npm SDK | `offline-tarteel/packages/core` | Package metadata says `@tilawa/core`, version `0.4.0`, MIT |
| Website and server | `offline-tarteel/web/frontend` | Vite frontend, Hono server; root Dockerfile; Dokku remote app `tilawa` |

The engine source checkout and its main branch were both at `441dddc3665537b3ef87aab79779277cbc44087e` when prepared. Treat this as a candidate to verify, not a request to silently replace it with whichever experiment is newest.

The new architecture has `packages/core` for tracking/checking and `packages/sdk` for runtime/session adapters. The old public package already uses the name `@tilawa/core`. Resolve this naming collision deliberately: preserve the public `@tilawa/core@1.0.0` identity, then internalize or rename implementation packages as appropriate. Avoid publishing a second public SDK that users must discover separately unless the user chooses that strategy.

The simplified harness changes are in `packages/sample/index.html`, `src/main.ts`, `src/harness.css`, `vite.config.ts` and `README.md`. They are uncommitted and therefore absent from a delegate checkout based only on HEAD. Capture them in a reviewed commit or patch before assigning migration work. Preserve unrelated untracked files in the old checkout.

### Evidence and existing mechanisms

- New browser `/live` uses the full `spliced-clean-int8` student model, automatic WebGPU/WASM fallback, one thread, free recitation, and VAD off. It does not load the auxiliary teacher.
- The simple local harness automatically prepares the engine on Start recording; tuning and diagnostics are collapsed. It exports exact input PCM as float WAV plus events/evidence JSON.
- Internal `/live` already flags moments and saves a full take with events and audio on Modal. Read `docs/benchmarks/live-review.md` and `tools/live_review_core.py`. Its validator expects mono 16 kHz **PCM16** WAV, which differs from the harness float WAV. Choose and document a common ingestion contract; do not assume the formats are interchangeable.
- Existing `/help` is a standalone static recording page with Hono APIs under `/help/api`. The current server accepts only v4 correct-recitation prompts; earlier deliberate-mistake prompts are retired. Preserve natural recitation as the policy.
- Existing `/help` consent version is `2026-09-30`; wording includes CC BY 4.0 publication and training/evaluation. Keep historical consent attached to historical submissions. New private reporting consent is a separate versioned policy.
- Existing `/help` storage includes audio plus a JSONL submission index, pseudonymous speaker IDs and a separate subscriber list. Keep subscription emails unlinkable to recordings.
- The old Dockerfile fetches old model/corpus GitHub release assets. Updating TypeScript alone will leave the deployed product on old assets.
- The old server deliberately exempts `/help` from COOP/COEP because it previously needed no inference. Revisit headers and cross-origin fonts/assets when adding the new SDK.
- The release scoreboard is dated October 13 and reports failed/unmeasured gates. Local harness build/typecheck and 25 SDK tests passed during preparation; these are not evidence that release gates pass.

## Execution rules

Read each repository's `AGENTS.md` before work, and nested instructions when entering their scope. In the research repo, also read `.scratch/lab/spec.md` before any listener/tracker/checker change and `docs/agents/audio-review.md` before reviewing audio failures. In the public repo, follow its worktree, hooks, signing and required browser-test discipline.

- Keep the proven algorithm unchanged during packaging and UI migration wherever possible. Any change that can move a lab score needs its recorded experiment. Hand-run folds need the user's approval; autolab uses its existing gate. Porting to the public repo needs parity evidence even when the algorithm is unchanged.
- Heavy inference, benchmark and training jobs run on Modal. Use the execution tool's persistent/background mode for long jobs; `nohup ... &` inside a returning shell loses the Modal app. Local paths at module scope also execute inside Modal containers.
- Use the runtime's delegate tools and bind any child worktree through those tools. Do not create a child's checkout in a shell or its task prompt. Respect the local worktree conventions when preparing the lead's own checkout.
- Read the delegate model-routing skill if available. The user wants Opus to lead; use enabled Claude models for workers appropriate to their tasks. At preparation, standalone Claude was disabled and an OpenCode Claude delegate failed with `OAuth access token is invalid`. Resolve provider access before scheduling delegates; report unavailable providers instead of silently substituting.
- Limit parallel work to the available slots. Give each delegate a narrow owner area, dependencies, acceptance tests and expected return format. Have the lead own shared API/schema decisions. Avoid overlapping file edits.
- Each delegate returns: files changed, commit/PR if applicable, tests and their results, remaining concerns, and integration instructions. The lead reviews every shipped change and runs integration checks.
- Keep credentials out of prompts, logs, reports and commits. Use available private credential tools when needed. Audit existing public/admin routes rather than assuming stored audio is protected.
- Prepare staging, package dry runs and rollback before publication. Respect existing user authorization and actual tool approval requirements; ask about unresolved changes to release bars, consent terms or publication permissions with concrete proposals.
- Prepare social drafts and a video artifact; posting from the user's account requires explicit authorization to send the post.

## Task tracking and delegation

Create `.scratch/tilawa-v1-release/spec.md` and individual numbered issues under `.scratch/tilawa-v1-release/issues/` in the chosen integration checkout. Record status, dependencies, owner, acceptance criteria and evidence links per issue. Follow the repository's triage vocabulary; track execution progress separately from triage state. Keep this PLAN linked from the spec.

Suggested delegation lanes after the contract is agreed:

| Lane | Responsibility | Shared boundary owned by Opus |
| --- | --- | --- |
| SDK integration | Public API, assets, runtime adapters, migration | Package identity and event contract |
| SDK reporting | Opt-in capture, bounded audio, report export | Report schema and timing semantics |
| Website/help | Main demo, contributor interface, accessibility | SDK reporting calls and consent payload |
| Collection/data | Ingestion, private review, dataset exporter | Schema, consent, label provenance |
| Validation/release | Parity, devices, package smoke tests, deployment | Frozen candidate and launch decision |
| Launch content | Documentation, real demo recording, social drafts | Verified claims and final URLs |

Start with SDK and reporting work after contract definition. Backend and UI can proceed against that frozen schema. Integrate one complete report round trip before expanding the remaining surface.

## Workstreams and completion criteria

### 00 — Inventory, candidate freeze and migration map

Inspect both repos, public GitHub identity, npm package ownership/version, live deployment and persistent storage. Record the actual branch SHAs, model/reference hashes, source changes to import, public entry points to replace, historical data locations and deployment routes. Compare actual runtime behavior to documentation.

Build a migration matrix: old public call/event → new call/event → compatibility or intentional breaking change. Identify existing users' model URLs and service-worker/cache behavior. Record code, model and training-data licensing separately; remove old model dependencies from the proposed release.

**Done:** the lead has a pinned candidate, verified public destination, preserved local work, data backup plan, migration matrix, and an explicit list of open decisions. No destructive repo replacement or model swap has occurred.

### 01 — V1 public contract

Agree the package layout and platform entry points while preserving `@tilawa/core` as the public SDK. Design a default setup that loads the reference and model without making beginners supply corpus files or choose runtimes. Retain self-hosting and injected runtime/storage options for advanced integrations.

Specify session lifetime, asynchronous serialization, stop/drain/reset, cleanup, position/checker events, supported capabilities, cancellation, load progress and errors. Define how SDK defaults stay stable while models version independently. Include an example for free recitation and one for a declared passage.

Define reporting as an optional capability with a portable report artifact and app-owned submission seam. Keep default inference offline and free of uploads or full-session recording retention. Proposed API names are draft: finalize them in the spec before delegation.

**Done:** reviewed TypeScript interface sketches and runnable target examples exist; package naming, event mapping, version policy and reporting seam are unambiguous.

### 02 — Port the proven engine and package it

Integrate core, session and runtime adapters into the public product without reviving the old tracker/checker. Include reference and phrase assets through the public SDK; avoid reliance on source aliases or research-directory symlinks. Keep existing repositories' history and research assets intact.

Verify worker placement and UI responsiveness on the web, native adapter boundaries, runtime dependency requirements and cleanup. Ensure optional reporting does not alter inference events or chunking semantics. Cover old API migration rather than maintaining two active correction engines accidentally.

**Done:** public package builds; deterministic tests pass; selected identical audio/evidence produces equivalent source/public engine events; clean external web and Node apps can import and run it; React Native adapter validation is recorded before advertising RN support.

### 03 — SDK report capture

Implement bounded opt-in retention and the agreed report artifact. Capture enough audio before and after an incident to review acquisition, ambiguity, corrections and skips. Allow reports during a take and after stopping. Reports referring to a checker event retain that event ID; missed mistakes/wrong positions can be reported without an existing checker event.

Specify one audio clock, mapping to wall-clock arrival and session timestamps, sample rate/channel metadata, resampling and VAD implications, crop offsets and truncation indicators. Handle a requested interval that has left the buffer, delayed inference, early/late flags and disposal explicitly.

A report includes schema version, report/session IDs, SDK/model/reference identity and model hash, session mode/passage/policy, incident kind and time span, user claim/note, linked events, relevant evidence, audio integrity metadata, and versioned consent. Include an app-provided source tag and optional pseudonymous contributor identifier; avoid unnecessary device/personal data. Separate user claims from machine observations and later review labels.

Make raw PCM retention bounded and diagnostic posterior capture optional/bounded. Offer local export and a transport adapter/callback; no credentials in the SDK. Local report creation still works offline. Third-party apps can own their backend or use documented Tilawa ingestion with scoped credentials/restrictions appropriate to the platform.

**Done:** tests cover timing/cropping, resampling, missing context, multiple reports, stop/drain, bounded memory and opt-out; reporting-enabled/disabled inference parity passes; report/audio round-trip validation works outside the repo.

### 04 — tilawa.dev and `/help` interface

Use the same published SDK for the main demo and contributor page. Keep runtime/model jargon out of the normal experience. `/help` starts with a short explanation and consent, then one obvious Start action, natural reading, Stop and a small review/send flow. Choosing a surah is optional; the engine locates the recitation.

During recording offer an optional **I stumbled here** bookmark without interrupting reading. Afterward prioritize a few suspected/problematic moments, with the ayah and an immediately playable short clip. Ask one plain question at a time: Was your reading correct? Did you make a mistake here? Did we show the wrong ayah? Include Not sure and Skip; allow another moment to be marked from playback. A warning cannot establish its own truth, and unflagged audio cannot be labelled clean automatically.

Accept clean contributions as useful data with consent, including a small review sample of unflagged regions to reduce feedback-selection bias. People who cannot identify the exact word can still send a timestamp and audio. Advanced text correction stays optional.

Support Arabic and English, RTL Quran text, large text and touch targets, visible focus, keyboard/screen-reader access, plain microphone permission recovery, playback, upload progress, retry and clear success. Preserve a take during network failures; clarify whether retry survives refresh and make durable local storage opt-in with expiry/cleanup. Confirm before discarding an unsent take.

**Done:** main demo and `/help` use V1; complete recording/review/send works on iOS Safari and Android Chrome; no model/provider choices are required; accessibility checks pass; an elderly/nontechnical participant can complete the flow with minimal guidance, or the remaining human usability check is explicitly pending.

### 05 — Collection service and historical data

Adapt the existing Hono/Dokku service and `/help/api` namespace. Preserve old submission formats and consent provenance; add a versioned ingestion path instead of reinterpreting old JSONL rows. Reuse storage where suitable with a deliberate migration and backup.

Validate actual audio format/duration/size, crop/event time ranges, schema, consent and identifiers. Make writes atomic and report IDs idempotent so retries do not duplicate contributions. Bound request sizes and rate-limit submissions. Protect audio, report listings and review routes with actual authorization; keep admin tokens server-side. Document third-party CORS/credentials without distributing a privileged secret to browsers.

Store uploads privately, separate consent to send from optional training/public-dataset permissions, and provide a documented retention/deletion process. Keep old contributor attribution and subscriber separation. Decide the new consent wording and record its version before collection goes live.

Revisit `/help` headers, font origins, WASM assets and caching; verify inference works under the deployed isolation configuration. Return a receipt useful for retry, support and deletion.

**Done:** one SDK report can be sent, retried, viewed privately and deleted on staging; invalid uploads are rejected; anonymous requests cannot list/download reports; old submissions remain readable under their original consent; restart/redeploy preserves data.

### 06 — Review to training/evaluation pipeline

Ingest reports into an automated triage queue using original audio, source versions, event/evidence traces and contributor claims. Group incidents and reuse existing human findings. Follow `docs/agents/audio-review.md`: a teacher is correlated evidence; an independent audio service must be calibrated before treating it as a useful second opinion.

Keep statuses for reported, triaged, needs listening, validated, excluded and exported, separate from training labels. Store reviewer judgments and uncertainty alongside the untouched original report. Surface only residual listening questions and spot checks to the user.

Export versioned datasets with audio hashes, incident spans, corrected labels/transcripts where supported, consent eligibility, label provenance and pseudonymous speaker grouping. Deduplicate crops/recordings and separate speakers across training/practice/exam. Mark test material held out before fine-tuning; review consent eligibility before either training or public redistribution.

**Done:** an end-to-end staging contribution becomes a reviewed example in the existing dataset layout; withheld-consent examples are excluded; duplicates and speaker leakage are caught; re-running export is reproducible. A production retraining run is a later lab experiment, not required to launch reporting.

### 07 — Evaluation and release readiness

Re-run the current candidate's release scoreboard and appropriate lab practice/exam evaluation on Modal. Use the declared release scope and the authoritative existing bars, not stale harness success. Record missed mistakes, false warnings, tracking acquisition/false jumps, and supported-class coverage.

Verify device latency, RTF, memory, sustained sessions, repeats/restarts/self-corrections, ambiguous passages, permission failures and offline behavior on the supported device floor. Verify source-to-public packaging parity, reporting overhead and model/cache identity. Run the public repo's required deterministic, browser and demo checks; update them to exercise the new SDK.

If gates fail or lack valid data, produce a concrete blocker list. Propose a scoped prerelease where appropriate; do not relabel an unready build stable or lower thresholds without an explicit user decision.

**Done:** a fresh candidate report accounts for every required release gate; supported claims match measured behavior; outstanding failures have a user-visible disposition before stable release.

### 08 — Hosting, documentation and release mechanics

Serve immutable versioned model/reference/runtime assets under tilawa.dev with hashes, manifests, correct MIME types, CORS/CORP and cache headers. An immutable GitHub artifact may be the build source, but developer-facing defaults resolve through tilawa.dev. Verify clean download, warm cache, checksum failures and documented self-hosted offline use. Set an operational model-download cost budget and monitor it.

Update Docker, dependency locks, model URLs and asset preparation to the frozen candidate. Keep raw user data outside public assets and npm archives. Test service-worker upgrades from the current live site so old assets/events do not mix with V1. Prepare a rollback that does not destroy new submissions.

Publish clear quickstarts, API/migration docs, support matrix, reporting integration, privacy/contribution policy, model card and notices. Keep the ordinary path simple; advanced hosting/runtime tuning belongs in separate documentation.

Run package dry runs and inspect the tarball: public exports/types resolve, runtime dependencies are intentional, assets are accessible, and source/test/private data do not leak. Install the tarball or prerelease in clean external projects and exercise recording and reporting.

**Done:** staging serves the correct candidate; clean package consumers work; old-client upgrade and rollback are tested; release notes, package identity/ownership, licenses and artifacts are ready.

### 09 — Launch assets and publication

Create a 30–60 second real demo recording: start reciting, see the location and a genuine correction, review/report a moment, then show the developer quickstart. Use permissioned audio; label any scripted demonstration accurately. Avoid invented performance claims. Produce a shareable video file, captions and thumbnail if useful.

Draft an X post linking tilawa.dev, `/help`, SDK docs, npm and the public repo. Explain on-device recitation and the contribution loop in plain language. Prepare release notes and a short contributor invitation.

Once readiness and authorization are satisfied, publish the prerelease/stable package and versioned artifacts, tag the public repository, deploy the website, and smoke-test the actual published package and production URLs. Keep assets compatible with old releases and ensure rollback is available. Post social content only with explicit account-posting authorization.

**Done:** package/artifact versions and source SHAs are recorded; production demo/docs/help and reporting receipt work; monitoring and private triage have owners/runbooks; launch artifacts are delivered and publication state is explicit.

## Dependency order

00 → 01 → 02 and 03.
03's schema unlocks 04 and 05; integrate one report round trip early.
05 unlocks 06. 02–05 feed 07. Packaging/hosting in 08 can proceed alongside evaluation, then waits for the frozen passing candidate. 07 and 08 unlock publication in 09. Launch content can be drafted earlier, but claims and screenshots must use the final candidate.

## Decisions to resolve with concrete proposals

- Confirm actual GitHub/npm ownership and deployment identity; the observed origin URLs differ from the user's public repo name.
- Final public package/subpath layout and old API migration strategy.
- Default contribution duration/context window and whether contributors share entire takes or selected clips; propose bounds based on review utility and memory measurements.
- New consent wording, training/public-redistribution choices, retention/deletion policy and any treatment of minors. Keep the basic UI readable without bundling permissions silently.
- Whether available evidence supports stable V1 or a labelled prerelease, and how pending human usability checks will be completed.
- Any credential/provider limitation, budget increase, policy exception or destructive data operation that remains genuinely unresolved.

## Final acceptance checklist

- [ ] One public V1 SDK identity replaces the old engine, with a migration guide.
- [ ] The frozen model/reference/SDK versions and supported claims are reproducible.
- [ ] Fresh release evaluation and supported-platform package tests are recorded.
- [ ] tilawa.dev hosts the demo, docs and versioned assets using V1.
- [ ] `/help` supports natural recitation and simple review of false warnings, missed mistakes and wrong locations.
- [ ] Reporting is opt-in, bounded, portable and reusable by third-party apps.
- [ ] Private ingestion, retry, deletion and historical consent preservation work.
- [ ] Validated reports export to a consent-aware, speaker-separated dataset.
- [ ] Old-site upgrades, persistent storage and rollback are verified.
- [ ] Launch video, captions, social draft and release notes are delivered.
- [ ] Published artifacts/URLs are verified; anything still awaiting authorization is identified precisely.

The handback should link the code/PRs, package versions, deployed URLs, evaluation evidence, launch files and any remaining decisions. Finishing a delegate or a green unit-test run is not completion of this plan.
