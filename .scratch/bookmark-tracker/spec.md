Status: ready-for-agent

# Bookmark tracker

## Problem Statement

Apps using the Tilawa SDK need to know where in the Quran a person is reciting, word by word, without being told in advance. They need to know when the person repeats, skips, restarts, or jumps to another passage. Today nothing turns the listener's phoneme output into a Quran position.

The same gap blocks the data pipeline. The tlog filename labels are frequently wrong: in the first shard, only 26% of clips are within 10% phoneme error of their filename ayah. Clips are often offset by one ayah, or contain two ayahs. Without a tracker that can find the true passage, about half of tlog can't be used for distillation.

## Solution

Add a deterministic, replayable tracker to `@tilawa/core`.

It takes in listener phoneme evidence, either streamed chunk by chunk or as one whole clip. It emits versioned `position` events. Each event carries the expected bookmark, a ranked list of observed candidates with confidence, and a tracker state: `acquiring`, `locked`, `ambiguous`, or `lost`. When a passage is genuinely ambiguous, such as a shared phrase, the tracker reports that instead of inventing a unique location.

The same API serves two consumers:

- **The live SDK:** a streaming session.
- **tlog relabelling:** one push of a whole clip, with the filename ayah as a weak prior. A confident `locked` result relabels the clip; `ambiguous` or `lost` excludes it.

## User Stories

1. As an app developer, I want to start a tracker with no known passage, so that users can begin reciting anywhere in the Quran.
2. As an app developer, I want to start a tracker with an intended passage, so that a memorization exercise is located faster and more reliably.
3. As an app developer, I want word-level `position` events while the user recites, so that I can highlight the current word.
4. As an app developer, I want each position event to include confidence, so that I can avoid highlighting when the tracker is unsure.
5. As an app developer, I want an explicit `ambiguous` state with several candidates, so that I can show "still listening" instead of a wrong location when phrases repeat across surahs.
6. As an app developer, I want separate expected and observed bookmarks, so that a deviation is visible rather than silently absorbed.
7. As a reciter, I want the tracker to follow me when I repeat a word or phrase, so that repetition doesn't lose my place.
8. As a reciter, I want the tracker to follow me when I restart from earlier in the ayah, so that self-correction works naturally.
9. As a reciter, I want the tracker to notice when I skip ahead, so that a skipped passage can be reported in memorization mode.
10. As a reciter in free recitation, I want the tracker to follow an intentional jump to another passage once enough evidence arrives, so that I can change passages freely.
11. As a reciter, I want a single wrong word not to cause a jump, so that a mistake is treated as a mistake and not a passage change.
12. As a reciter, I want long pauses not to lose my position, so that I can breathe or think.
13. As a reciter, I want the tracker to start correctly from the middle of an ayah, so that I can begin anywhere.
14. As a reciter, I want my position recovered after background noise or non-recitation speech, so that interruptions don't break the session.
15. As an app developer, I want a `lost` state when the tracker can't locate the recitation, so that I can prompt the user.
16. As an app developer, I want the tracker to re-acquire globally after `lost`, so that the session recovers without a restart.
17. As an app developer, I want the same events from the same input on every platform, so that web, Node, and React Native behave identically.
18. As an app developer, I want events tagged with the schema, reference, and model versions, so that I can handle upgrades safely.
19. As a data engineer, I want to locate a whole tlog clip in one call, using its filename ayah as a prior, so that I can relabel clips at scale.
20. As a data engineer, I want clips where the tracker isn't confident excluded rather than guessed, so that distillation labels stay trustworthy.
21. As a data engineer, I want the matched passage span reported (start and end word, possibly crossing ayahs), so that clips containing two ayahs are labelled correctly.
22. As a data engineer, I want a phoneme error rate against the matched passage, so that I can sort clips into the clean-match, deviant, ambiguous, and reject buckets.
23. As a data engineer, I want the tracker to run in Node against stored pass-one output, so that relabelling needs no GPU.
24. As a researcher, I want a replayable trace of candidate changes and alignment scores, so that I can tell why the tracker chose a location.
25. As a researcher, I want shortlist recall measurable, so that I know whether the first stage dropped the true location.
26. As a researcher, I want the optional top-k posteriors to rerank candidates, so that I can measure posterior-aware scoring against decoded-only scoring.
27. As a maintainer, I want the reference index built from a pinned Quran text and phonemizer version, so that positions are reproducible.
28. As a maintainer, I want the tracker to run in real time on the device floor in pure TypeScript, so that no native module is required in v1.

## Implementation Decisions

- **Module.** The tracker lives in `@tilawa/core` and is written in TypeScript. It has no runtime dependencies. It runs in browsers, Node, and Hermes.
- **One public entry point:**
  - `createTracker(referenceIndex, options)` returns an object with `push(evidence)` and `end()`. Both return `EngineEvent[]`.
  - `options` contains `mode` (`free` or `memorization`), an optional `prior` (an intended passage or a filename ayah, plus its weight), and an optional `trace` flag.
  - `push` emits events as soon as evidence supports them. `end` flushes any pending decisions.
- **Input: `PhonemeEvidence`.** This is the listener's output for a chunk:
  - decoded phonemes, each with start and end frame and confidence;
  - frame duration;
  - optionally, per-frame top-k phoneme ids and probabilities.
  
  It matches what the Modal teacher pass already stores (`hyp`, `hyp_conf`, `top5_ids`, `top5_p`). Phoneme symbols are `quran-transcript` phonemes.
- **Output.** Output uses the existing `EngineEvent` schema, version 1:
  - The tracker emits only `position` events. Checker events are out of scope.
  - The `position` event gains an optional matched span (start and end word reference) and an optional `per` field, both used by relabelling.
  - These additions don't break schema version 1, because they're optional fields.
- **Reference index.** The reference index is a versioned, precomputed asset loaded by the tracker:
  - It's built offline from a pinned Uthmani text plus `quran-transcript` phonemes using a fixed `MoshafAttributes` profile.
  - Each word maps to its phoneme span, with stable surah/ayah/word identifiers.
  - It contains an n-gram inverted index over phonemes for retrieval.
  - Context-dependent pronunciation is handled by indexing a word's phonemes both in continuous context and in its stop form.
- **Stage 1: retrieval.** This stage follows the QUD aligner pattern:
  - n-gram anchor voting over the index produces a candidate shortlist.
  - A substring Levenshtein DP with word-boundary constraints then scores each candidate.
  - Substitution costs are phoneme-class-aware, so confusable pairs and short-vowel swaps cost less than unrelated substitutions.
- **Stage 2: reranking.** When top-k posteriors are present, CTC posterior scoring reranks the shortlist. Without them, the tracker uses decoded output only. Both paths must produce valid events.
- **States:**
  - `acquiring`: global search.
  - `locked`: continuity-favouring local search around the expected bookmark, with transitions for forward, repeat, backward restart, and skip.
  - `ambiguous`: two or more candidates within a margin.
  - `lost`: the best score is below threshold. The tracker returns to global search.
  - A jump needs evidence that persists for a configurable span before the observed bookmark commits. The expected bookmark moves only on an explicit passage change, or in free mode once the jump has committed.
- **Prior.** The prior only adds a score bonus to its passage and the nearby passages. It never forces a location.
- **Confidence.** Confidence is a calibrated margin between the top two candidates. The thresholds for `locked` and `ambiguous` are options with defaults. The defaults are later tuned on the hand-checked tlog sample and the QUD benchmark.
- **Determinism.** Given the same input, the tracker always produces the same output:
  - no randomness;
  - no wall-clock dependence (time comes from frame indices);
  - stable tie-breaking by canonical word order.
- **Tracing.** When `trace` is on, events carry the shortlist, scores, and state transitions, for replay and debugging.

## Testing Decisions

- **One seam.** Tests go only through the public API: `createTracker` → `push`/`end` → emitted events. They assert observable behaviour: emitted states, positions, spans, and confidence ordering. They never assert internal scores, data structures, or index layout. The index builder is exercised indirectly by building a real index in test setup.
- **Synthetic sequence fixtures.** Take reference phoneme sequences from the index and apply edits:
  - exact recitation;
  - a mid-ayah start;
  - a repeated word or phrase;
  - a backward restart;
  - a skip within a surah;
  - a distant jump;
  - a single wrong word that matches another verse;
  - a shared phrase occurring in several surahs, which should give `ambiguous`;
  - noise or random phonemes, which should give `lost` and then recovery;
  - a long silent gap.
- **Real replay fixtures.** These come from the first tlog teacher pass, stored as small JSON evidence files:
  - The 100:6 clip containing 100:6 plus 100:7 should give a span across both ayahs.
  - The 100:7 clip that actually contains 100:8 should relabel to 100:8.
  - The 100:1 clip with ض→د should lock at 100:1 with a nonzero `per`.
  - A clean exact match.
- **Streaming equivalence.** The same input frames, delivered in different chunk sizes, must produce equivalent results at the same evidence boundaries. Early streaming decisions are never compared with an offline result that has seen future audio.
- **Real-clip labels.** A real-clip fixture's expected label counts as ground truth only after someone has independently checked the audio. Until then it's a regression snapshot.
- **Benchmark harness, separate from the unit tests.** These are tracked numbers, not pass/fail gates, run against:
  - the QUD alignment benchmark;
  - `nurlingo/quran-recitation-detect-split`.
  
  It reports correct-ayah-within-5-seconds rate, false-jump rate, shortlist recall, and time to first lock.
- **Performance check.** Use a fixed-size replay through the same API and record its time against a budget. On the device floor, a 30-second clip must process in real time.
- **Prior art.** The repo has no existing tests. Use Vitest for the core package.

## Out of Scope

- Checker events (wrong, omitted, or added word; pronunciation classes) and the feedback lifecycle.
- The listener and ONNX runtime integration, and the student model.
- The session layer (`startMemorization` and `startFreeRecitation`) and platform SDK packages.
- Bucketing tlog into clean-match, deviant, ambiguous, and reject. That's a script built on this API that applies thresholds.
- Recitation traditions other than Hafs, and madd-length-aware matching beyond the fixed `MoshafAttributes` profile.
- A Rust or WASM port.

## Further Notes

- The QUD universal aligner Space is the reference design for stage 1. Read it before implementing.
- tlog pass-one output lives in the Modal volume `tilawa-tlog` under `pass1/`. Each row has `surah`, `ayah`, `hyp`, `hyp_conf`, `top5_ids` (uint8, n_frames×5), `top5_p` (fp16, n_frames×5), and `prior_per`.
- Learner madd lengths vary. The reference uses a single profile, so substitution and insertion costs for madd extension phonemes should be low, to avoid penalising legitimate length choices during location.
- `PLAN.md` holds the acceptance bar: correct ayah within 5 seconds in at least 95% of starts, and at most 2% false jumps.
