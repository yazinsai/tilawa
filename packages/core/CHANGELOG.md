# Changelog

## 0.4.0

A better default model, and correction mode now catches skipped ayahs and look-alike-verse slips.

- **Default model: `a0w-ep1-a0.5` int8** (`zipformer_a0w_ep1_a05.int8.onnx`, release [`zipformer-a0w-ep1-a0.5`](https://github.com/yazinsai/tilawa/releases/tag/zipformer-a0w-ep1-a0.5), NPL-1.2). It is a 50/50 blend of Quran-Lab v3 with a fine-tune on multi-ayah windows. Phoneme error rate on the 600-clip q-lab benchmark is 3.59% (Quran-Lab v3: 4.45%). Tracking is 53/53 on v1 and 56/58 on held-out multi-ayah windows. Same size (66 MB), same I/O. It also exposes encoder frames, so the opt-in slip head can run. `DEFAULT_ZIPFORMER_IO` is its manifest. The previous default, `zipformer_interp_gentle_a05.int8.onnx` (release v0.3.0), still works unchanged.
- **Structural correction rules, on by default in correction mode** (`DEFAULT_STRUCTURAL`). Tracking mode never runs them, and `structural: false` turns them off.
  - **Ayah order** raises `possible_skipped_ayah` when the reciter jumps over a whole ayah. It needs no expected passage.
  - **Similar verse** raises `possible_substitution` when a word from a look-alike ayah is read, and `possible_omission` when a word is dropped. The substitutions need `setExpected`.
  - Both run once over the take at `stop()` (`timing: "stop"`), and their issues come one per `correct()` call. `timing: "pause"` runs them at pauses too; it is opt-in.
  - Issues they raise carry `source: "ayah_order" | "similar_verse"`. Similar-verse flags on or next to a skipped ayah are dropped.
  - On a held-out synthetic set (EveryAyah dev reciters, passages unseen in tuning), skipped-ayah recall without a passage goes from 91/120 to 103/120. Sister-ayah substitution recall with a passage goes from 11/100 to 54/100.
  - They add 0 flags on 220 uncut controls and on 211 clean minutes.
  - They are text rules with no learned weights. They add nothing on letter-level mispronunciations.
  - New API: `ZipformerSessionOptions.structural`, `ZipformerSession.setStructural()`, `StructuralOptions`, `DEFAULT_STRUCTURAL`, `CorrectionIssue.source`, and the rule internals (`StructuralRules`, `AYAH_ORDER_RULE`, `SIMILAR_VERSE_RULE`, …).
  - The look-alike index (`structural-index.json`, 56 KB gzipped, word indices only) is a dynamic import, so bundlers put it in its own chunk; it loads when a session with the rules on is created.
- **Slip head** (`slipHead: "strict" | "high"`, off by default) over the exposed encoder frames. Its recall gain was on memorised words, so it stays opt-in.

### Also in 0.4.0

Correction mode now tracks the words people get wrong. Tracking mode is unchanged.

- **`ZipformerSession.setExpected({ surah, ayah, ayahEnd? } | null)`** (new type `ExpectedPassage`, opt-in). Pass the passage the reciter is about to read, for example the ayahs on screen. In correction mode the tracker then locks onto the passage's first word as soon as any isti'adha or basmala is past. Later searches only lock inside the passage, and the tracker never relocates to another surah. Without a passage, short takes of repeated or near-identical ayahs (55:13, 37:80, 26:36, …) locked onto the wrong copy or never locked, and the mistaken word was never judged. The passage is kept across `reset()`, and tracking mode ignores it.
- **Skipped ayah, read directly.** With a passage, `possible_skipped_ayah` is also raised when the audio the tracker put on an interior ayah A reads as ayah A+1 (`skipMinChars` 10, `skipMaxDistance` 0.35, `skipMargin` 0.25, `skipMaxHead` 0.1). Before, the aligner bent A+1's audio onto A's text, so A+1 was never matched and the ayah-gap rule could not fire. With a passage, the ayah-gap rule also runs while `stop()` flushes the tail.
- **End-of-ayah anchoring** (`anchorAyahEnd` 2). When the cursor stops within 2 words of an ayah end, settled verdicts realign the last stretch through that end. A word dropped just before the last word is now `skipped`, instead of taking the last word's audio and leaving the last word unheard.
- **No passage:** a take that never locked is aligned to its best search hit (`stopAlignDistance` 0.35) for the final word check. A mid-ayah lock in that check starts at the ayah's first word (`backfillRatio` 1.5). Neither feeds verse tallies.
- **New `EngineConfig` knobs** (correction mode only): `backfillRatio`, `stopAlignDistance`, `outsideJumpCost` (0, off), `skipMinChars`, `skipMaxDistance`, `skipMargin`, `skipMaxHead`, `anchorAyahEnd`. New `CorrectionThresholds.substitutionDistance` (default 0.6, unchanged behaviour).
- Acted-mistake test half, shipped model, paired replay, default path (no passage, as in the main app): F1 goes from 0.267 to 0.335 and recall from 0.155 to 0.203. With `setExpected` (opt-in, for apps that know the passage), F1 is 0.436. Clean false flags are no higher than the shipped engine on any set. Numbers are in `lab/EXPERIMENTS.md`, "Tracker in correction mode".

Correction-mode recall on real slips, at the same false-flag rate on clean recitation.

- **`WordVerdict.gop` / `gopNone` / `gopTwice` / `repGain` / `pairGop`.** Goodness-of-pronunciation scores from the CTC posteriors (per-token log ratio of forcing the expected word against the unconstrained best path over its window). Correction mode only; set on interior words whose neighbours were heard. Optional, so tracking-mode verdicts are unchanged.
- **GOP rule.** A word the aligner already marks `wrong` or `skipped` is flagged when its GOP is ≤ `gopFlag` (−3), it is the local minimum, and both neighbours fit (clear, or GOP ≥ `gopAnchor`, −2). Kind is `possible_omission` when silence explains the window better than the word, else `possible_substitution`. `ok`, `unsure` and `pending` words are never flagged by GOP.
- **`CorrectionIssue.kind: "possible_repetition"`.** A word that fits once but gains ≥ `repetitionGain` (5 nats/token) from a second copy, with a clear left neighbour. A neighbour that also repeats vetoes it (a phrase restart is accepted practice). A retry that repeats the word again is not a correction.
- **Partial omissions.** A `skipped` word with some heard audio (heard ratio ≤ `omissionMaxHeard`, default 1) counts as an omission. Before, only heard ratio 0 did.
- **`CorrectionController.settle(verdicts, cursor)`.** When the tracker is dropped (`stop()`, surah completed, silent idle), the session checks the words `observe()` never saw with settled context once. Vowel flags stay observe-only. `settle: false` turns it off.
- **`CorrectionThresholds`.** New optional `omissionMaxHeard`, `gopFlag`, `gopAnchor`, `settle`, `repetitionGain`. Missing fields take the defaults, so existing thresholds objects keep working.
- Verified on real recordings only (no synthetic or acted mistakes). False flags per clean minute did not rise on any clean set. Numbers are in `lab/EXPERIMENTS.md`, "Correction rules v2 (real recordings)".
- **Defaults re-tuned on acted recitation mistakes** (private set, speaker-disjoint dev/test; numbers in `lab/EXPERIMENTS.md`, "Correction rules on acted help mistakes"). The GOP rule is **off** by default (`gopFlag: -Infinity`): on clean takes it still raised false flags that no other rule did, and it added no recall on the shipped model. Opt in with `gopFlag: -5`. Its default gating is now wrong words only (`gopOnSkipped: false`), substitutions only (`gopOmission: false`), not inside `settle()` (`settleGop: false`), and `gopAnchor: -1`. `vowelWordMargin` goes from 0.5 to 0.8. On the shipped model, test F1 went from 0.150 to 0.259, with clean false flags no higher on any set.
- **New GOP knobs:** `gopOnWrong`, `gopOnSkipped`, `gopOmission`, `gopSubstitution`, `gopNoneMargin`, `gopNoneMin`, `gopLocalMin`, `gopPersistFrames`, `settleGop`.
- **`repetitionMode: 'off' | 'note' | 'flag'`** (default `note`). A single repeated word is a soft note: the session emits `{ type: "correction_note", issue }` (new `WorkerOutbound` member) and recitation is not interrupted. `CorrectionController.takeNotes()` drains them. `flag` keeps the old interrupting behaviour.

## 0.3.1

Correction mode no longer stays silent when a whole ayah is missed.

- **`CorrectionIssue.kind: "possible_skipped_ayah" | "unclear_ayah"`.** When ayah N+2 is matched right after ayah N in the same surah and N+1 was never matched, `ZipformerSession` raises one issue for N+1 at `word: 0`. `possible_skipped_ayah` when the aligner heard almost nothing of N+1 (mean heard ratio below `AYAH_HEARD_FRACTION`, 0.5); `unclear_ayah` when audio was heard but the model could not follow it. The word-level rules (`possibleWordIssues`) are unchanged; they could not see this case because a whole bad ayah has no clear neighbours.
- **`CorrectionIssue.words`.** Number of words the issue covers from `word` (default 1). Ayah-level issues set it to the ayah length, so a retry must produce a clear prefix through the whole ayah.
- **`CorrectionController.raise(issue, cursor)`.** Raises a session-inferred issue with the same gates as a word flag (correction mode, idle, not dismissed/deferred earlier). Ayah-level issues reuse retry / dismiss / review_later and fire once per ayah per session.
- Never fires in tracking mode, during `stop()`, across a tracker re-locate (`located` / `relocated` / idle restart), or across a surah change. A transient `lost` inside one surah does not break the chain — that is the unclear-ayah case.
- Verified: still 0 flags on the 53 clean calibration clips and on the four `correction-audio` recordings; `unclear_ayah` on 104:2 for the user clip that motivated this.

## 0.3.0

Correction mode can now flag harakah (short-vowel) errors.

- **`CorrectionIssue.kind: "possible_vowel"`.** A word whose consonant skeleton matches (distance ≤ 0.15) but whose aligned short vowel differs from the expected one, with the same clear-anchor and 12-frame persistence rules as omissions and substitutions. The word-final vowel (case ending) is never used as evidence: waqf drops it and the model's Quranic prior confidently rewrites it on clean audio. A retry that repeats a confident vowel error does not count as corrected.
- **`WordVerdict.vowelErrors` / `vowelMargin`.** Count of aligned vowel substitutions in the word and `p(heard vowel) − p(expected vowel)` at the token's peak frame (min over mismatches). Both are `0` when none.
- **`CtcToken.vowels` / `HeardChar.vowels`.** For tokens ending in a short vowel, the probabilities of the same token spelled with fatha, damma, kasra at its peak frame. The vocab is consonant(+shadda)+vowel, so `ببُ` has siblings `ببَ`, `ببِ`.
- **`CorrectionThresholds`, `DEFAULT_CORRECTION_THRESHOLDS`, `CorrectionController.thresholds`.** `vowelMargin` (default `0.05`) and `vowelWordMargin` (default `0.5`). Calibrated on 309 clean clips / 106 min: professional recitations produce zero vowel mismatches; crowd-sourced TLOG clips produce 2 flags that look like genuine reciter errors.
- **`ZipformerSession.verdicts()`.** Public read of the active tracker's latest word verdicts, for debug bundles.
- **`vowelMismatches()`** exported for tests and tooling.

## 0.2.1

Browser hang fix, packaging, and README that an outside consumer can follow.

- **EP default.** `{ ort, model }` now picks `["wasm"]` under onnxruntime-web and `["cpu"]` under onnxruntime-node (`listSupportedBackends` when present, else `ort.env.wasm`). Override with `executionProviders`. On web, `ort.env.wasm.numThreads` defaults to `1` unless the caller set it — pthread init hangs in workers without COOP/COEP.
- **FastConformer validation.** `createTilawaSession` / `createRecognitionSession({ engine: "fastconformer" })` throw `Error("fastconformer engine requires assets: vocab, ctcTokens, quran ...")` listing the missing keys before touching them.
- **Packaging.** `files` is `dist/`, `README.md`, `CHANGELOG.md`, `LICENSE`, `NOTICE.md`. Source maps and `src/` are no longer packed.
- **README.** Browser quick start uses `onnxruntime-web`, documents EP / `numThreads` / `wasmPaths` (Vite copies wasm by default), GitHub release asset URLs, licence split, and a Node snippet that matches the 300 ms / `stop()` / events path.

## 0.2.0

### Zipformer is now the default engine

The streaming Zipformer phoneme engine, previously only in the web demo, is now
part of the SDK and is what you get by default.

- **`createRecognitionSession(options)`** — new top-level factory with an
  `engine: "zipformer" | "fastconformer"` selector. Defaults to `"zipformer"`
  (`DEFAULT_ENGINE`). Returns a `RecognitionSession`: `feed()`, `stop()` /
  `flush()`, `reset()`, plus the underlying engine session.
- **`createZipformerSession(options)` / `ZipformerSession`** — the default engine
  directly. 16 kHz PCM → Kaldi fbank → streaming Zipformer2-CTC over 251
  tajweed-phoneme tokens → whole-Quran phoneme n-gram search → per-surah online DP
  tracker → per-word verdicts → the same `WorkerOutbound` verse events the
  FastConformer path emits. Exposes `transcript`, `tallies`, `verses`,
  `engineState`, and `config`.
- **ONNX injection, two shapes.** `{ ort, model }` for web and node (model bytes or
  a loader); `{ session, Tensor }` for React Native, where
  `InferenceSession.create()` takes a file path. No runtime is imported by the
  package.
- **`DEFAULT_ZIPFORMER_IO`** — the shipped model's I/O manifest is bundled, so only
  custom exports need an `io` override.
- The whole engine is exported from the package root (`src/recitation/`): engine,
  fbank, CTC decoder, corpus, search, alignment, phoneme cost table, and the
  event-emission helpers.

### Unchanged

`createTilawaSession(runner, assets, options?)`, `SessionRunner`, `TilawaAssets`,
`QuranDB`, `TextCTCDecoder`, `RecitationTracker`, and the `StreamingConfig`
presets keep their names and behaviour. Existing code runs untouched.

### Verified

- Zipformer streaming, median of 3 repeats: v1 53/53, v2 43/43 — 100% recall,
  precision, and sequence accuracy on both.
- Python lab harness on v1: 100% / 100% / 100%.
- 110 TypeScript test cases across `packages/core` (60) and `web/frontend` (50).

### React Native

No DOM, `Worker`, `fetch`, `TextDecoder`, or top-level `await` in the package;
typed arrays throughout. `BigInt64Array` (for the model's `int64` cache states) is
the only runtime requirement beyond ES2020, and a missing one now fails with an
explicit error. Setup walkthrough: `examples/react-native.md`.

## 0.1.0

Initial release. FastConformer text-CTC pipeline behind `createTilawaSession()`,
with the `SessionRunner` injection seam, `QuranDB` matching, and the streaming
`RecitationTracker`.
