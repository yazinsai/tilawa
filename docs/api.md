# API reference

`@tilawa/core` 1.0.0-rc.0 (Preview, npm tag `next`).

Start with the [quickstart](quickstart.md). This page lists every public export, by import path.

| Import | What it holds | Needs |
|---|---|---|
| [`@tilawa/core`](#tilawacore) | The runtime-neutral engine, event types, errors, model registry | nothing |
| [`@tilawa/core/web`](#tilawacoreweb) | `createTilawa` for browsers, microphone capture | `onnxruntime-web` ≥ 1.19 |
| [`@tilawa/core/node`](#tilawacorenode) | `createTilawa` for Node, WAV input | `onnxruntime-node` ≥ 1.20, Node ≥ 20.12 |
| [`@tilawa/core/react-native`](#tilawacorereact-native) | `createTilawa` for React Native. **Experimental, not yet validated on devices** | `onnxruntime-react-native` ≥ 1.20 |
| [`@tilawa/core/report`](#tilawacorereport) | Opt-in problem reports | nothing |
| [`@tilawa/core/assets/*`](#tilawacoreassets) | `ref-v2.json`, `phrases-v1.json` | nothing |

The package is ESM only. Every subpath ships its own type declarations.

Contents: [createTilawa](#createtilawa) · [Tilawa](#tilawa) · [Sessions](#sessions) · [Events](#events-schema-v1) · [Errors](#errors) · [Version policy](#version-policy)

---

## `createTilawa`

```ts
function createTilawa(options?: TilawaOptions): Promise<Tilawa>
```

Exported by `/web`, `/node` and `/react-native`. It loads the model and the Quran reference, warms the model up, and resolves with a ready [`Tilawa`](#tilawa). The model and the reference load in parallel.

### `TilawaOptions` (all platforms)

| Option | Type | Default | Meaning |
|---|---|---|---|
| `tier` | `"full" \| "lite" \| "auto"` | `"auto"` | Which model. See [tiers](#model-tiers). |
| `assetBase` | `string` | `"https://tilawa.dev/v1/"` | Where models, the reference and (web) the ONNX Runtime files are fetched from. Point it at a mirror to [self-host](self-hosting.md). A trailing `/` is added if missing. |
| `reference` | `ReferenceAsset` | fetched (web, RN) or bundled (Node) | The reference, for offline use: import `@tilawa/core/assets/ref-v2.json`. |
| `phrases` | `PhraseAsset` | the bundled `phrases-v1` | Non-Quran phrases said around a recitation (isti'adha, takbir, amin…). |
| `model` | `LoadedModel` | loaded from the registry | A model you loaded yourself (`loadWebModel`, `loadNodeModel`, `loadNativeModel`). You keep ownership: `tilawa.dispose()` won't release it. |
| `storage` | `ModelStorage` | Cache API (web), `fsStorage(cacheDir)` (Node), in memory (RN) | Where downloaded model bytes (and, on web and RN, the reference) are kept. |
| `providers` | `string[]` | web `["wasm"]`; RN `["cpu"]` | ONNX execution providers, tried in order on web. Ignored on Node (CPU). |
| `threads` | `number` | `1` | ONNX intra-op threads. On web, more than 1 only takes effect when the page is cross-origin isolated. |
| `wasmPaths` | `string` | `<assetBase>runtime/ort/<onnxruntime-web version>/` | Web only. Where onnxruntime-web's `.wasm`/`.mjs` files are served. Ignored when you pass `model`; give it to `loadWebModel` instead. |
| `onProgress` | `(p: LoadProgress) => void` | none | Load progress, see below. |
| `signal` | `AbortSignal` | none | Cancels loading. `createTilawa` then rejects with an `AbortError`. |
| `recordEvidence` | `boolean` | `false` | Keep each session's phoneme evidence for `session.evidence()`. Use it for bug reports and benchmarks. It also keeps the session's 16 kHz audio in memory, and at `end()` re-decodes gaps, so `evidence()` after `end()` can differ from what streamed. |

`LoadProgress` is `{ stage: "model" | "reference" | "warmup", loaded: number, total: number, fraction: number, fromCache: boolean }`. `fraction` is 0..1 within the stage. Byte counts are bytes for `model` and `reference`; `warmup` goes 0 → 1.

### Platform options

| Option | Platform | Meaning |
|---|---|---|
| `providers?: WebProvider[]` | web | `"webgpu"` and/or `"wasm"`. A provider that fails to load or run its first window falls through to the next. |
| `cacheDir?: string` | Node | Model cache directory. Default `$TILAWA_CACHE_DIR`, else `~/.cache/tilawa`. |
| `modelDir?: string` | Node | Load `manifest.json` and its `.onnx` file from this directory instead of the registry. Mirrors `/v1/models/<id>/<version>/`. |
| `pathOf?: (key: string) => string` | RN | The file path where your `storage` keeps a key. onnxruntime-react-native loads a file path faster than bytes across the bridge. |

The option types are exported as `WebTilawaOptions`, `NodeTilawaOptions` and `NativeTilawaOptions`.

### Model tiers

| Tier | Model | Download | `tier: "auto"` picks it |
|---|---|---|---|
| `full` | `spliced-clean@1-int8` | 67,759,744 bytes (67.8 MB) | Otherwise. Always on Node. |
| `lite` | `spliced-clean-lite@1-int8` | 24,650,054 bytes (24.7 MB) | Web: `navigator.deviceMemory` under 4, iPhone and iPad, or another mobile browser without WebGPU (`navigator.gpu`). React Native: always. |

Both detect the same four error classes. See the [model card](model-card.md).

---

## `Tilawa`

```ts
interface Tilawa {
  startFreeRecitation(policy?: SessionPolicy): TilawaSession;
  startMemorization(passage: MemorizationPassage, policy?: SessionPolicy): TilawaSession;
  listen(session: TilawaSession, options?: MicrophoneOptions): Promise<Microphone>;
  readonly capabilities: CapabilityEvent;
  readonly model: TilawaModelInfo;
  dispose(): Promise<void>;
}
```

- **`startFreeRecitation(policy?)`**: a session for recitation from anywhere in the Quran. The tracker finds the place.
- **`startMemorization({ surah, fromAyah, toAyah }, policy?)`**: a session for a declared passage. The tracker stays in it, `position.expected` is the word the reader should be on, and mistakes are judged against the passage. Throws a `RangeError` for an empty or reversed passage.
- **`listen(session, options?)`**: web only. Opens the microphone and streams it into the session. On Node and React Native it rejects with `MicrophoneError("unavailable")`: push audio yourself.
- **`capabilities`**: the `capabilities` event every session sends first.
- **`model`**: `TilawaModelInfo` `{ id, version, sha256, tier, backend, fallbacks }`. `version`, `sha256` and `tier` are null for a passed-in model that isn't in the registry. `backend` is, for example, `onnxruntime-web/webgpu`. `fallbacks` says why each earlier web provider was skipped.
- **`dispose()`**: disposes every open session and releases the ONNX model (unless you passed `model`). Idempotent. Starting a session afterwards throws.

### `SessionPolicy`

| Field | Default | Meaning |
|---|---|---|
| `confirm` | `{ after: "delay", ms: 2500 }` | When a suspected mistake is confirmed: a fixed delay after it, or `{ after: "next-word" }`, once the next word has begun. |
| `minConfidence` | `0.6` | Below this mean decoder confidence over a mistake's phonemes, it ends `uncertain` instead of `confirmed`. |
| `countSelfCorrections` | `false` | Whether `resolved` mistakes (self-corrections, recovered skips) count in `session.mistakes()`. |
| `evidenceMs` | one model frame (40 ms) | The tracker and checker read evidence in cells of this much audio on a fixed grid, so events don't depend on how you chunk audio. Leave it alone unless you know why. With `refinements.pace` on, which push delivers a position can depend on the push size (the events and their order don't). |
| `refinements` | every field `true` but `markWhenSure` | Refinements of the 1.0.0-rc.1 engine's rules (`Refinements`, below). Set a field `false` to restore the rule it replaces, or `markWhenSure` `true` to turn it on; with every field `false` a session is rc.1's. |

`session.policy` is the resolved policy, with every default filled in (`ResolvedSessionPolicy`).

#### `Refinements`

Each field is on unless set `false`, but `markWhenSure`, which is off unless set `true`: it still gets cases wrong that it cannot yet tell from the ones it is for (in its row). A field that refines another (`declaredLeadIn` and `declaredMove` refine `declaredStart`, `leftOnNewAudio` and `begunByConsonants` refine `withdrawBegun`) is read only while that one is on; set `false`, it restores that field's rule without it. The source docs (`CheckerRefinements`, `TrackerRefinements` and `Refinements`) give the exact rules.

| Field | What it does | Known risk |
|---|---|---|
| `readNotSkipped` | A repeat that rejoins is not a skip: an omission is not raised when every word of it was read right earlier in the current pass and the reading after it rejoins: it resumes where reading on from the furthest word read goes, or at an ayah's first word after the reciter broke off (a pause or a phrase that is not the Quran's between the speech either side, or no speech before it in the window). A pause alone excuses nothing: a resumption inside an ayah (forget, hesitate, skip) stays an omission, in a re-reading too, and so does one at an ayah's first word in continuous speech. Nor is an ayah's end a break: whole ayahs left out stay omissions, a re-reading's as a first reading's, unless the reading goes on where the reciter had read to (the ayah after the furthest one read to its end, or the ayah they broke off in, started again). A pass is set by the recitation, the same in free and declared sessions: a new one starts at a takbir or a basmala, at a jump or after the place was lost, and where the reciter goes back to an earlier ayah 30 words or more behind the furthest word read (three ayahs of the median length; or to a surah's first ayah) and reads that ayah through. Coming forward again to where they had been goes on with the pass they left. | A reciter who goes back fewer than 30 words, reads an ayah again and breaks off inside it (a pause, then the next ayah's first word) is not marked for the rest of that ayah, read the first time; one who broke off inside an ayah, went back and starts that ayah again is not marked for the ayahs the run-up left out. |
| `reanchorBack` | A tracker lock of `JUMP_WORDS` or more words, inside the passage and behind the alignment window, restarts the alignment there, unless the window's own text holds the same words (a refrain's later copy). In free recitation the passage is what has been read so far, so a reciter who starts again from an earlier ayah is followed. In a declared passage, once followed back, a lock past the window's far end is followed forward too: landing back on the furthest ayah read is a rejoin and marks nothing, landing further on leaves out the words between, and landing short of it after a return that started a new pass leaves out what that pass skipped. A slide into an earlier copy of the ayah just finished that the reciter corrects alone is not marked (self-correction is no error; it cannot be told from a deliberate revision). While the tracker, gone back to an earlier ayah, is still before an omission, the omission waits; a tracker that only holds or lags does not delay it. | A lock behind the window on words the reciter did not go back to say ends the window's open suspicions `uncertain`. |
| `stutter` | A word heard with a block of sounds said twice in a row, which without one copy has the written word's consonants in order, is not a wrong word. | It compares consonants, so a restarted word whose short vowels were also changed is let through. |
| `ledgerPartial` | The ayah ledger raises no whole-ayah skip for an ayah the alignment read in part; the word-level omissions cover its missing words. | An ayah left out whose words were matched in part by chance (a phrase shared with its neighbours) can be marked as a shorter skip, or not at all. |
| `omissionDue` | Memorization: a sure omission is sent `suspected` again with `due: true` once the reciter has read on past it (the confirmation delay passed, two later words heard right, both begun after the suspicion), ahead of its confirmation. `confirmed` keeps rc.1's rule (two ayahs on, or the end) and stays final: saying the word before then ends it `resolved`. | An app that shows `due` as a mistake shows one the reciter may still correct. |
| `confirmAdvance` | A later word entered on a risky step (a sound not heard as written, or the first sound after a pause heard below the tracker's own `BREATH_CONFIDENCE`, 0.6, apart from `minConfidence`: maybe a breath or a stray sound) shows only once two of its sounds (all of a shorter word's) were heard as written. A word entered on a clean step shows on its first sound, as in rc.1, after a pause too, so a clean ayah opening shows when rc.1 shows it. Every lock follows the same rule. | A correct word whose first sound was heard not as written, or faintly after a pause, shows one sound later. |
| `onsetRun` | A word opening with a doubled sound (a shadda: sun letters, idgham, ٱللَّه) shows once a sound of the doubling's second half is heard (the second of two, the third of three or four): its first half closes the word before, which stays shown until then. | Such a word shows a sound or two later than in rc.1; a held doubled sound heard as more sounds than written does not delay it further. |
| `declaredStart` | Memorization: before the first lock, the speech heard from each onset (the take's start, or the first sound after a pause) is aligned against the passage's opening, which may lock when it fits as a landing must (over five sounds, or the whole first word when shorter) and no phrase fits as well. Every ayah opening of the mushaf is a rival, judged by how well the speech heard so far fits it: a rival is in play while its fit is less than half an edit worse than the declared opening's, and told apart once it is worse by more. While rivals are in play, the declared opening's words show only as far as every rival in play shares them: the same word as written (the orthography's marks set aside: تَبَٰرَكَ is تَبَارَكَ), the same sounds, or the same sounds split into other words that end together (وَإِن مَّا and وَإِمَّا). So of ayahs that open with the same words (113:1 and 114:1, 26:68 and 26:9), the declared one, the next in order, lights as they are said, and the page moves at the first word they do not share; while a rival whose first word differs, however alike it sounds (يس and طسم, ٱلَّذِينَ and وَٱلَّذِينَ), is in play, nothing locks, and the opening shows once the speech tells them apart. The lock shows the word heard (or the word before it, when the sound that told them apart is the next word's first), and each later word shows as it is said. With `declaredLeadIn`, every word of the two ayahs before the declared one is a rival too. If a rival fits better by an edit, or the speech no longer fits the declared opening, the lock is let go and the tracker's search finds the reciter, as after any wrong lock, or, for a rival near the passage, the lock moves there (`declaredMove`). Off, a take's first words show once the tracker's search finds the reciter, as in rc.1. | An opening that a rival's differs from only in a held length (إِنَّ and إِنَّآ), or that is the start of a longer rival (الٓمٓ and الٓمٓر, طسٓ and طسٓمٓ), shows its first word with the first sound of the next word, so a pause after it delays it by the pause. A declared passage of one word whose pausal form opens longer ayahs (55:1 alone) is not shown by its opening, only once the tracker's search finds it, as in rc.1. A reciter who says an ayah that opens with the same words as the declared one is shown the declared words they share while saying them, and the page moves when the words part. A false start of the declared opening lights the words said in it. A declared opening that also opens a phrase (ٱللَّهُ, the takbir's first word) shows once the speech leaves the phrase (the second word's first sound). A reciter who starts inside an ayah elsewhere, not at its opening, with words that fit the declared opening, is shown on it until the tracker loses them. Sounds said straight into the opening, with no pause, delay the lock as in rc.1. Once the lock is let go for a rival far from the passage (another surah's ayah that opens alike), the tracker's search finds the reciter only from what it heard after the lock. |
| `declaredLeadIn` | Memorization, with `declaredStart`: a reciter who leads in from just before the passage is not shown its first word before saying it. Every word of the lead-in tail (the declared ayah's words before the declared word, and every word of the two ayahs before it) is a rival of the declared opening as an ayah's opening is: while one whose first word differs is in play nothing locks, and once one fits the speech better by an edit the lock is let go (or moves to it, `declaredMove`). A tail word inside its ayah shares no word with the declared opening: the declared ayah is the one next in order only to an ayah that opens alike, while a reciter who leads in says the tail's words first. Off, only ayah openings are rivals: a lead-in from inside the ayah before (its last words, or going on after a breath inside it) whose sounds fit the declared opening as a landing must is shown the declared first word before it is said. | A reciter who leads in from further back is shown as with the field off. A declared opening whose first words recur inside the tail shows only once the speech parts from the tail: a word or two later. |
| `declaredMove` | Memorization, with `declaredStart`: when the lock on the declared opening is let go because another ayah's opening (or a word of the lead-in tail, `declaredLeadIn`) fits the speech better by an edit, and that rival lies near the passage (from the lead-in tail to two ayahs past the passage's last, in its surah), the reciter started there: the lock moves to it, on all the speech heard since the onset the declared lock was made on, and the words said there show. A declared opening that no longer fits as a landing must yields to a near rival that does, the same way. While two or more near rivals fit alike (within half an edit), the speech has not told them apart and the mushaf's order does not either (a lead-in from before the passage and a later start open alike): the contest stays open, the declared words showing only as far as every rival in play shares them, until one stands apart; past what a fit reads, the last before the declared ayah, else the first after it. The declared words the page showed that the new run does not cover are named in its `withdrawn`. The first onset after a pause that follows a move (or that ends a contest held open) is judged again against the declared opening and its rivals, as the take's first is: when the declared opening locks there and the run shown does not already cover its first word, the lock goes back to it, and the declared ayah lights from its first word. A rival far from the passage lets go to the tracker's search, as with the field off. Off, every let-go goes to the tracker's search, which reads only the speech heard after the lock: words said can stay dark, a lost-place prompt can come while the reciter recites correctly, and words of another place can light. | The declared words shown before the move show, then go dark (an app that shows `withdrawn`). While near rivals fit alike, the reciter's words past what they share wait for the words to part, or for a pause. A pause that ends a contest held open leaves the place to the declared run and the tracker's search. |
| `noLostBeforeLock` | Memorization: before the place is first found, an assessment that fits nowhere is reported `acquiring`, not `lost`. A take that never finds its place still ends with `lost`. | An app that shows a "lost" prompt shows none until the place was found once, or the take ends. |
| `commitAfterJump` | When the alignment window grows long, its older part is committed at a word boundary and the latest sounds are aligned again from there, in a window that starts an ayah before. If those sounds hold a restart from further back than that window reaches (the reciter started again from an earlier ayah), the cut moves to the first word boundary after the restart's landing whose path stays in its window, or waits (up to twice the window's length), so the restart is not read against the words in the window as wrong words and omissions. A restart reads again the text it lands on: when the first speech placed there does not fit that text, the path reached back only to place speech from elsewhere (a reciter gone back further than the window, onto a refrain's earlier copy), and the cut stays. A wrong word that the restart then reads right is a self-correction; one said wrong again after the restart is still a wrong word. | While it waits, one alignment covers up to twice as many sounds. |
| `firstAyahOut` | Memorization: a take that starts later than the declared passage, in a later ayah with three words or more of the passage before it, marks those words left out (one `skipped_passage` from the passage's first word), as any other skip is marked. A start further in than the alignment's first window reaches (two ayahs) starts the alignment there, as a jump does, so nothing is marked on the words between. A take that starts inside the first ayah, or two words or fewer in (a microphone that opened late, a word cut off), or outside the passage (another surah, before or after it), marks no skip, and speech before the passage that is not the Quran's is set aside as before. | A reciter who means to start later than the passage they declared is marked for the ayahs before. |
| `marksInPassage` | Memorization: no word mark outside the declared passage. The alignment window reaches past the passage's ends (an ayah before, two after the furthest word read), so it can raise a mark on a word that was not asked for; such a mark is dropped, and one that reaches into the passage stays. Free recitation, whose passage is what has been read so far, keeps its marks. | A reciter who reads on past the passage's end is not marked there. |
| `repeatNotWrong` | A restart is not a wrong word. The alignment can charge a restart (the reciter going back a little to take a run at a word again) to words whose first reading has left the alignment window, as one long wrong word whose heard text is the stretch read again: an ayah read whole and started again two or three times before reading on, with a checker commit falling inside the restarts, has its last restart charged to the ayah's next word. A wrong word whose heard text is at least 12 sounds and twice the word's length, and fits the text before it (from two ayahs back, not before a declared passage) as a lock's words fit a passage, charged to words read right earlier in the pass, where the reading after them rejoins as `readNotSkipped` judges a repeat's, was that restart, and is not marked. Any other stays a wrong word: words never read, or a reading that goes on inside the ayah (earlier text said in place of a word, then reading on). It acts only with `readNotSkipped` on, whose record of the words read it uses. | A reciter who goes back fewer than 30 words, says earlier text in place of words they had read right and goes on where they had read to is not marked for them. |
| `consonantConfirm` | One syllable heard on weak evidence is not yet the word: a later word of the tracker's run entered on a sound not heard as written (among the last three up to its first) or on a risky step, whose sounds heard so far another word a skip forward lands on (a later word of its ayah, or the next ayah's first word) opens with as well, shows once two of its consonants (all of a word with fewer) were heard as written, a doubled consonant counting once, or once its last sound is reached: the reciter may have left it out. A word entered on sounds heard as written shows on its first sound, as in rc.1. Never an ayah's first word, which shows as before, and never a word with no consonant (a nasal alone). A held word is never kept dark, nor shown with the next: when the run goes on into the next word first, it shows then, and the next word waits for its own consonants. A lock shows its last word by the same rule, so the word shown does not depend on how the audio was split into pushes. | A correct word entered on a sound the decoder misheard, whose opening a word nearby shares, shows when its second consonant is heard, a sound or a few later than in rc.1; one whose consonants are misheard shows at its last sound, or as the next word begins. |
| `rivalHold` | A misheard opening that fits a word further on does not advance the run: while what was heard of a later word of the run is not all as written, and the opening of a word a skip forward lands on (a later word of its ayah, or the next ayah's first word; inside the declared passage) fits it as well or better, the word waits, until four of its sounds (all of a shorter word's) were heard as written or its last sound is reached. A word whose sounds are heard as written shows on its first sound, as in rc.1, even where another word opens alike: by the owner's rule for look-alikes the order the reciter reads in decides. Never an ayah's first word, for the same reason. A held word shows alone, as the run goes on past it, and a lock judges its last word the same way. | A correct word whose first sound the decoder misheard as a word nearby opens shows a few sounds later than in rc.1. A word left out whose next word opens with its sound still shows on that sound, as in rc.1. |
| `passageWindow` | Memorization: while the tracker's local search is inside the declared passage, none of its jumps lands past the passage's end, so a look-alike after it (a phrase of the next ayahs that fits a sound or two of the passage's) takes no lock while the reciter is still reading the passage, and a global lock past the end is a jump, not the local search's to follow. A reciter who reads on past the passage's last word is followed there in order, as without it (the shown word never moves back into the passage, and the app's lost-place prompt says they left it); a return before the passage's start is followed as without it. | A reciter who leaves the passage by a skip forward, not by reading on, is followed there only once the tracker's search commits the jump, a few sounds later than in rc.1. |
| `restartNoEntry` | A restart is not a surah entry: a jump back to the first word of the surah the tracker is in costs a plain jump, so a reciter who starts the surah again is followed there sooner. Only entering a surah from outside it (reading on from the one before, or jumping in from another) pays the surah entry, which keeps the end of a surah from sliding into the next one's opening. | None known; entering a surah in order costs what it did. |
| `withdrawBegun` | A word only begun goes dark again once the reciter has clearly moved past it, so a word left out never looks said. When the run that shows a word leaves it by a jump forward while that run had only begun it (not said by the tracker's test of a word said, `begunByConsonants`), the locked `position` that first shows the new run names it in `withdrawn`. The word is judged as the run's path left it, before the sounds the new run reads as its own; a later word the old run showed on those sounds is withdrawn too (`leftOnNewAudio`). So too an ayah's first word shown on the onset of a restart, once the run goes back before it. Only a word the run that leaves it showed first is withdrawn: a word an earlier run showed (the reciter said it before going back over it) stays shown. A run is its own: a lock after a position that held no place (a lock let go, a lost place) starts a run, and a word the run showed with no step of its path in it is kept. A word the new run covers is never withdrawn, and a later run that covers a withdrawn word (the reciter says it after all) shows it again. Words the run read on past in order, and words heard in full, stay shown. Off, a word stays shown once shown. | A short word said whose consonant the decoder misheard, after which the run jumps forward, goes dark although said, until a run covers it again. An ayah's first word guessed on two clean consonants of a restart stays shown. A word begun at the take's end, with no run after it, stays shown. A word begun, gone back over and begun again before the skip stays shown, as an earlier run showed it. An app that shows `withdrawn` makes a word lit then darkened visible: the flicker of a word begun and left. |
| `leftOnNewAudio` | With `withdrawBegun`: a later word the run that leaves it showed, which its path reached only after the first sound the new run reads as its own, goes dark whatever was heard of it: what lit it was the new run's speech read against the old run's words. So when two ayahs open alike and the page moves from the one next in order to the one said, the words lit for the first go dark. Off, such a word goes dark only when only begun there. | A word said just before a jump, whose sounds the new run's search also takes as its own, goes dark until a run covers it again. |
| `begunByConsonants` | With `withdrawBegun`: the word a jump leaves was only begun unless the tracker's test of a word said (`showSaid`'s) passes: the path has heard it through its last consonant, every consonant (a doubled one once) or all but one in a word of three or more. Judging a word the run has left, it gives the doubt to the reciter: a consonant heard as its engine-confusable partner (ح and ه, ع and ء, ص and س, ض and د, ط and ت, ق and ك, ذ and ز, ث and س) counts as heard, and the one a longer word may miss may be its last. So a word left after its first syllable goes dark, and a word said in full with a consonant misheard or dropped stays shown. Off, only a word's onset counts as begun: no more than its first consonant and vowel, and fewer than two of its consonants, heard as written, so a word whose first syllable and next consonant were said before the skip stays shown. | A word of one or two consonants said with one misheard as other than its confusable partner goes dark when the run then jumps forward, until a run covers it again. A longer word left just before its last consonant, or with all but one of its consonants said, stays shown. |
| `markWhenSure` | **Off by default.** It can mark a reciter who leaves the declared passage for another surah on the word where they left, before the lost-place prompt that should be all they see (the other surah's opening read past that word, its next word begun, passes every test a confirmation asks but the wait). Memorization: a mark as soon as the checker is sure, and none lost at the take's end. A suspected wrong or added word that passes all its confirmation asks but the wait (read past, the next word begun, heard clearly, and the posteriors favouring what was heard over the written words by the confirmation's margin: twice that for a change heard unclearly or spelling no word), its evidence (class, words, heard text) standing unchanged for 0.8 s in place of the wait, is sent `suspected` again with `due: true`, ahead of its confirmation, which keeps rc.1's time and confirms it. When the session ends, a mark already due is decided on that evidence, and a mistake where the reading stopped (a wrong word after a word recited right, with nothing heard after it, or an added word between two words recited right) on its evidence, by the same test, instead of always `uncertain`, so a slip on the last word before a stop is marked; speech the alignment could not place on the passage (other speech, a hold) ends `uncertain`, as in rc.1. A take that ends one or two words short of the declared passage's end, inside its last ayah, marks them left out. Free recitation keeps rc.1's rule. | A `due` mark can still end `resolved` when the reading after it explains the word (an app takes the mark down); a slip of two words or more just before a stop is not marked; a reciter who means to say a word again after a stop, or who stops a word or two before the passage's end on purpose, is marked. |
| `showSaid` | A word shows once it has been said, not on its first sound: the session's `position` events are the positions to show (`Tracker.shown()`). A word inside an ayah shows once the next word is under way (three of its sounds heard, all of a shorter word's): the decoder places a sound at its start, and a word's last sounds run into the next word's first (a doubled onset, a merged nasal, a joined hamzat al-wasl). An ayah's last word, which a pause follows, shows once heard to its end as written (a final held vowel's first sound: the hold is not waited for). At a pause (0.48 s with no sound decoded, longer than the decoder's gap since a held sound decodes as its start and then quiet; a session tells the tracker the silence its listener decodes), the word stopped on shows once heard as written to its end (a final short vowel aside, which a stop drops); at the take's end and when the place moves to another run, once its consonants were heard; otherwise the word before it shows once heard as written. A word only begun, then left, never shows. Every other rule of what shows still holds; the checker reads the tracker's own positions, unchanged. Off, the events are rc.1's. | A word shows about a syllable after it ends (before a pause, 0.48 s after it; one heard otherwise than as written only once the next word is under way); a reciter who stops on a word after its last consonant (a cut final vowel) is shown it at the pause. |
| `pace` | Display smoothing: words completed by one model answer appear in the rhythm they were said: a later word arriving in the same push as the one before it is held, on the audio clock, by the time between them, and shown at the last push that ends by then, taking the next push to be as long as the longest of the latest eight. It changes only when events are delivered, never what is followed or marked, and never shows a word earlier; an app that draws words in its own layer (a native rebuild) may carry it there and set this `false`. An ayah's first word is never held, events are never reordered, `mistakes()` lists a confirmation once its event is shown, and `end()` shows everything held. | A word shows up to the time between it and the word before it later (at most the audio one push decodes); pushes that grow can show one later than that. Pushes longer than that time hold nothing, so what pace separates depends on the push size. Held events wait for a push or `end()`: an app that stops pushing audio should call `end()`. |

### `MicrophoneOptions` and `Microphone` (web)

| Option | Meaning |
|---|---|
| `vad?: boolean \| VadOptions` | Gate the audio with an energy voice-activity detector. **Default off.** With it on, silence is dropped before the model, so the session clock runs behind the wall clock. `VadOptions`: `marginDb` (default 12 dB above the learnt noise floor), `hangMs`, `preRollMs`. Keep the hangover longer than the confirm delay. |
| `onLevel?: (db) => void` | Raw input level in dBFS, about every 20 ms, for a meter. |
| `onEvents?: (events) => void` | The events each push produced. `subscribe` is usually simpler. |
| `onError?: (e) => void` | A push failed (an inference error). Default: logged to the console, except `SessionEndedError`. |
| `prepared?: PreparedMicrophone` | From `prepareMicrophone()`, see below. `listen()` takes ownership. |

The microphone is opened mono, with echo cancellation, noise suppression and auto gain control **off**. It runs at the device's rate and is resampled to 16 kHz.

`Microphone` is `{ sampleRate, speaking, stop() }`. `sampleRate` is the device rate. `speaking` is true while the VAD hears speech (always true without the VAD). `stop()` stops the tracks, closes the audio context, waits for queued audio, then calls `session.end()` and resolves with its events. It is idempotent.

`listen()` rejects with `MicrophoneError` (`denied`, `unavailable`, `insecure-context`), or with `SessionEndedError` if the session isn't `active`.

---

## Sessions

`TilawaSession` (alias `Session`) is one take.

```ts
interface TilawaSession {
  readonly id: string;                      // uuid v4
  readonly mode: "memorization" | "free";
  readonly passage: DeclaredPassage | null; // {surah, fromAyah, toAyah, from: WordRef, to: WordRef}
  readonly policy: Readonly<ResolvedSessionPolicy>;
  readonly capabilities: CapabilityEvent;
  readonly state: SessionState;             // "active" | "ending" | "ended" | "disposed"
  readonly startedAt: string | null;        // ISO wall-clock time of the first sample
  readonly vad: boolean;                    // true while listen() gates this session with the VAD
  now(): number;                            // session clock, ms
  subscribe(listener: (event: EngineEvent) => void): () => void;
  onAudio(fn: AudioTap): () => void;
  pushAudio(samples: Float32Array, sampleRate: number): Promise<EngineEvent[]>;
  end(): Promise<EngineEvent[]>;
  dispose(): void;
  idle(): Promise<void>;
  onEnded(fn: () => void): () => void;
  pushEvidence(evidence: PhonemeEvidence): EngineEvent[];
  endEvidence(): EngineEvent[];
  mistakes(): CheckerEvent[];
  evidence(): PhonemeEvidence[];
}
```

### Lifecycle

```
active ──end()──▶ ending ──(drained)──▶ ended
   └──────────────── dispose() from any state ───────────▶ disposed
```

- **`pushAudio(samples, sampleRate)`**: mono float samples at any rate. Calls run one at a time in call order; you don't need to await them. Resolves with the events this audio produced (subscribers get the same events; the return value is a convenience for tests). Rejects with `SessionEndedError` once `end()` has started or after `dispose()`.
- **`end()`**: drains the audio still in the model and decides every open mistake, then the state becomes `ended`. Idempotent: every call returns the same promise. Rejects with `SessionEndedError` after `dispose()`.
- **`dispose()`**: drops subscribers, audio taps and buffers, and rejects queued pushes with `SessionEndedError`. It doesn't release the model. `mistakes()` and `evidence()` return `[]` afterwards.
- **Inference failure**: the failing `pushAudio` rejects with the runtime's error, and the session moves to `ended`. Later queued pushes reject with `SessionEndedError`.
- **`idle()`**: resolves once every queued `pushAudio`/`end` has run, so all events for the audio received so far are out.
- **`onEnded(fn)`**: calls `fn` once when the session ends or is disposed, or at once if that already happened. Returns an unsubscribe function.
- **`subscribe(fn)`**: calls `fn` with the `capabilities` event at once, then with every event. Returns an unsubscribe function. A subscriber that throws doesn't stop the others; its error is rethrown asynchronously. Subscribing to a disposed session does nothing.
- **`onAudio(fn)`**: taps the 16 kHz samples the model receives, with the session-clock time of the first one: `(pcm16k: Float32Array, startMs: number) => void`. It never changes them. The reporting recorder uses it. Don't mutate the array.
- **`mistakes()`**: the `confirmed` checker events, plus `resolved` ones when `countSelfCorrections` is on.
- **`evidence()`**: the phoneme evidence so far, when `createTilawa({ recordEvidence: true })`; otherwise `[]`.
- **`pushEvidence(evidence)` / `endEvidence()`**: feed evidence from another listener, bypassing the model (tests, replays). Don't mix with `pushAudio`. Synchronous; throw `SessionEndedError` when the session isn't active.

### The session clock

`timeMs` on `position` and `checker` events, `session.now()`, and the report recorder all use one clock: **milliseconds of 16 kHz audio delivered to the session since its first sample**. It is not the wall clock.

- It advances only as audio is pushed. A paused stream pauses the clock.
- With the VAD on, dropped silence doesn't count, so the clock runs behind the wall clock.
- Inference can lag behind capture. An event's `timeMs` is where it happened in the audio, not when it was emitted.
- `session.startedAt` is the wall-clock time of the first sample, for display.

---

## Events (schema v1)

```ts
type EngineEvent = CapabilityEvent | PositionEvent | CheckerEvent;
```

Every event has `type` and `v: 1` (`EVENT_SCHEMA_VERSION`). Events are plain JSON-serialisable objects.

### `capabilities`

Sent first to every subscriber.

| Field | Type | Meaning |
|---|---|---|
| `modelVersion` | `string` | The model's id, for example `spliced-clean` (name kept from schema v1). |
| `modelRelease` | `string?` | The model's registry version, for example `1-int8`. Absent for an injected model outside the registry. |
| `referenceVersion` | `string` | The reference, for example `ref-v2`. |
| `supported` | `ErrorClass[]` | Error classes this model checks. In 1.0: `wrong_word`, `omitted_word`, `added_word`, `skipped_passage`. |
| `unsupported` | `ErrorClass[]` | Every other class. Never emitted. |
| `sdkVersion?` | `string` | The SDK version, `SDK_VERSION`. Added in 1.0. |
| `modelSha256?` | `string` | The model's sha256. Added in 1.0. |
| `tier?` | `string` | `full` or `lite`. Added in 1.0. |
| `backend?` | `string` | The ONNX backend, for example `onnxruntime-web/wasm`. Added in 1.0. |

### `position`

Where the reader is.

| Field | Type | Meaning |
|---|---|---|
| `timeMs` | `number` | Session clock. |
| `referenceVersion` | `string` | |
| `state` | `TrackerState` | `acquiring` (no place yet), `locked` (following), `ambiguous` (several places fit, for example repeated ayahs), `lost` (the place no longer fits). |
| `observed` | `PositionCandidate[]` | Candidate places, best first, each a distinct location. Can be empty. |
| `expected` | `WordRef \| null` | Memorization sessions: the passage word the reader should be on. Null in free recitation. |
| `hold?` | `"start"` | Set on the locked event that starts a misfit hold: the heard words stopped fitting and the tracker is waiting before it moves. |
| `relocated?` | `{ from: WordRef; to: WordRef }` | A passage that appears in near-identical copies resolved to one copy. `from` equal to `to` means the current copy was confirmed. |
| `spanHold?` | `true` | The first lock is inside such a passage and is still waiting for a word that tells the copies apart. |
| `withdrawn?` | `WordRef[]` | Words shown earlier (on the page: `shown()` names only words it showed, each once) that the reader only began and then left (`refinements.withdrawBegun`), or said as another place, when a take's start moves from the declared opening to the ayah the reader started at (`refinements.declaredMove`): show them dark again. Set on the locked event that first shows the run the reader moved to, which never covers them. A later locked event whose run covers one of them (from its span's `start` to its `word`) shows it again. |
| `trace?` | `TrackerTrace` | Tracker internals. Not set by `createTilawa` sessions. |

`PositionCandidate`:

| Field | Meaning |
|---|---|
| `word: WordRef` | The most recently recited word. |
| `span: WordSpan` | `{ start, end }`: the words covered by the evidence so far. It can cross ayahs. |
| `confidence: number` | 0..1. |
| `cost: number` | Weighted alignment cost per phoneme, for ranking. 0 is an exact match. |
| `per: number` | Phoneme error rate against the span. |
| `prefix?: "basmala"` | A leading basmala was recited before the span. |

`WordRef` is `{ surah, ayah, word }`. All three are 1-based. `word` indexes the words of the ayah as the reference lists them (Uthmani text). The engine returns no display text.

`ayahsRecited(events)` turns a list of events into the ayahs recited, in order: the best word of every `locked` position, with consecutive repeats of one ayah collapsed.

### `checker`

A possible mistake.

| Field | Type | Meaning |
|---|---|---|
| `id` | `string` | Stable across the mistake's states. |
| `timeMs` | `number` | Session clock. |
| `state` | `CheckerState` | `suspected`, then one of `confirmed`, `resolved` (the reader corrected themselves, or a skip was recovered) or `uncertain` (too unclear to judge), each final. A suspected mistake can be sent `suspected` again with `due` first. `suspected` is not for display: it is the checker's working state, raised as soon as the alignment sees a possible mistake and withdrawn as often as not while the reciter reads on or corrects themselves. Show a mistake once it is `confirmed`, or once it is `due` (an omission as a word still owed), taken down if it then ends `resolved` or `uncertain`. |
| `errorClass` | `ErrorClass` | One of the `supported` classes. |
| `location` | `WordRef` | The word the mistake is at. For `skipped_passage`, the first skipped word. |
| `expected?` | `string` | The phonemes expected there. |
| `observed?` | `string` | The phonemes heard. |
| `confidence` | `number` | Mean decoder confidence over the mistake's phonemes. |
| `due?` | `true` | On a `suspected` mistake in a declared passage sent again once it is likely to be confirmed: an omission once the reciter has read on past it (`refinements.omissionDue`), a wrong or added word once it passes all its confirmation asks but the wait, its evidence standing unchanged (`refinements.markWhenSure`). Not yet decided: confirmation still waits for the usual rule, and the mistake can still end `resolved`. Provisional: show it (an omission as a word still owed) and take it down when the same id ends `resolved` or `uncertain`. |

Error classes (`ErrorClass` = `MemorizationErrorClass | PronunciationErrorClass`):

- **Supported in 1.0** (`MEMORIZATION_CLASSES`): `wrong_word`, `omitted_word`, `added_word`, `skipped_passage`.
- **Not supported**: every `PronunciationErrorClass` (`short_vowel`, `missing_shadda`, `madd_natural` and the `consonant_*` pairs). They are in the type so the schema can grow without a breaking change. 1.0 never emits them, and it does not judge pronunciation or tajweed.

`ALL_CLASSES` lists every class.

---

## Errors

All exported from `@tilawa/core`.

| Class | Fields | Thrown by |
|---|---|---|
| `TilawaLoadError` | `stage: "model" \| "reference" \| "warmup"`, `cause` | `createTilawa` |
| `ModelIntegrityError` | | `createTilawa`, `fetchModel`, loaders: truncated, oversized or sha256 mismatch |
| `ModelBudgetError` | | `createTilawa`, `fetchModel`: over the 200 MiB budget (`MODEL_BUDGET_BYTES`) |
| `MicrophoneError` | `reason: "denied" \| "unavailable" \| "insecure-context"` | `listen`, `prepareMicrophone`, `startMicrophone` |
| `SessionEndedError` | | `pushAudio`, `end`, `pushEvidence`, `endEvidence`, `listen` on an ended or disposed session |

`createTilawa` also rejects with an `AbortError` (a `DOMException`, or `signal.reason` if that is an Error) when you abort `signal`. `LoadStage` and `MicrophoneErrorReason` are exported as types.

---

## `@tilawa/core`

Runtime-neutral: no Node built-ins, no ONNX Runtime import. Use it for types, or to build an engine around your own runtime.

**Engine**

- `createEngine(options: EngineOptions): Engine`: sessions from a loaded reference and model, without `createTilawa`'s loading. `EngineOptions` is `{ reference: ReferenceIndex, model: LoadedModel, phrases?, listener?: ListenerOptions, recordEvidence?, info?: { tier?, backend?, modelSha256? } }`. `Engine` has `capabilities`, `startFreeRecitation` and `startMemorization`.
- Types: `Engine`, `EngineOptions`, `Session`, `TilawaSession`, `SessionState`, `SessionPolicy`, `ResolvedSessionPolicy`, `Refinements`, `MemorizationPassage`, `DeclaredPassage`, `AudioTap`, `CapabilityEvent`, `EngineEvent`, `ListenerOptions`, `VadOptions`, and the `createTilawa` types `Tilawa`, `TilawaOptions`, `TilawaModelInfo`, `LoadProgress`, `Microphone`, `MicrophoneOptions`.

**Events**: `EVENT_SCHEMA_VERSION`, `ALL_CLASSES`, `MEMORIZATION_CLASSES`, `ayahsRecited`; types `CheckerEvent`, `CheckerState`, `ErrorClass`, `MemorizationErrorClass`, `PronunciationErrorClass`, `PositionEvent`, `PositionCandidate`, `TrackerState`, `WordRef`, `WordSpan`.

**Evidence** (types): `PhonemeEvidence` `{ inventory, frameMs, phonemes: DecodedPhoneme[], posteriors?: TopKPosteriors, decodedTo? }`, `DecodedPhoneme`, `TopKPosteriors`.

**Reference**: `loadReferenceIndex(asset: ReferenceAsset): ReferenceIndex`; types `ReferenceAsset`, `ReferenceIndex`, `PhraseAsset`, `AyahRef` (`{ surah, ayah }`).

**Models**

- `MODEL_REGISTRY: ModelRegistry`: the shipped tiers, with relative `base` paths, sizes, sha256 and manifests.
- `withBase(registry, base)`: the registry with every relative `base` resolved against `base`.
- `fetchModel(options: FetchModelOptions): Promise<FetchedModel>`: downloads (or reads from `storage`) one tier's bytes. Checks length and sha256 on download; a cache hit is only length-checked. Deletes older versions of the same model id from `storage` once the new one is stored. Throws `ModelBudgetError` before downloading when the tier plus other cached tiers exceed `budgetBytes` (default `MODEL_BUDGET_BYTES`, 200 MiB), unless `allowOverBudget`.
- `memoryStorage(): ModelStorage`: an in-memory store.
- `ortBackend(ort, model, name, sessionOptions?)`: wraps any ONNX Runtime module as an `InferenceBackend`.
- `MODEL_BUDGET_BYTES`.
- Types: `ModelRegistry`, `ModelTier`, `TierName` (`"full" | "lite"`), `ModelStorage` (`get`, `put`, `delete`, `keys`, all async), `FetchModelOptions`, `FetchedModel`, `LoadedModel`, `ModelManifest`, `InferenceBackend`, `Tensor`, `StateSpec`, `OrtModule`.

**Constants**: `DEFAULTS` (the defaults in the tables above, frozen), `REFERENCE` (`{ version: "ref-v2", path, bytes, sha256 }`), `SDK_VERSION`.

**Errors**: see [Errors](#errors).

## `@tilawa/core/web`

- `createTilawa(options?: WebTilawaOptions)`.
- `prepareMicrophone(): PreparedMicrophone`: call synchronously inside a tap handler. It creates and resumes an `AudioContext` and requests the microphone while the user gesture is fresh, which iOS Safari needs. Pass the result to `listen(session, { prepared })`. `PreparedMicrophone` is `{ context, stream, release() }`; call `release()` if you never call `listen()`. Throws `MicrophoneError` synchronously when there is no microphone API.
- `startMicrophone(session, options?)`: what `tilawa.listen` calls.
- `loadWebModel(options?: WebModelOptions)`: loads a tier with onnxruntime-web and warms it up. Options: `tier` (default `"full"`), `registry` (default `MODEL_REGISTRY`, whose relative paths resolve against the page; use `withBase`), `storage` (default `cacheStorage()`), `providers`, `threads`, `wasmPaths`, `allowOverBudget`, `onProgress(loaded, total)`. Resolves with the `LoadedModel` plus `fromCache` and `fallbacks`.
- `cacheStorage(name = "tilawa-models"): ModelStorage`: the Cache API store `createTilawa` uses.
- `decodeWav(bytes: Uint8Array): { sampleRate, samples }`: PCM WAV (8/16/24/32-bit integer or 32-bit float) to mono floats.
- Types: `WebTilawaOptions`, `WebModelOptions`, `WebProvider`, `PreparedMicrophone`, and the shared `Tilawa`, `TilawaOptions`, `TilawaModelInfo`, `LoadProgress`, `Microphone`, `MicrophoneOptions`.

## `@tilawa/core/node`

- `createTilawa(options?: NodeTilawaOptions)`. The reference is read from the package; only the model is downloaded.
- `readWav(path): Promise<{ sampleRate, samples }>` and `decodeWav(bytes)`.
- `loadNodeModel(dir, { threads? }): Promise<LoadedModel>`: loads `manifest.json` and its ONNX file from a directory.
- `loadNodeTier({ tier, cacheDir, registry?, threads?, ... })`: downloads a registry tier into `cacheDir` on first use, then loads it.
- `fsStorage(dir): ModelStorage`: files under `dir`, one per versioned key (`<id>@<version>/model.onnx`).
- Types: `NodeTilawaOptions`, and the shared `Tilawa`, `TilawaOptions`, `TilawaModelInfo`, `LoadProgress`, `Microphone`, `MicrophoneOptions`.

## `@tilawa/core/react-native`

**Experimental: not yet validated on devices.** The API may change in a 1.x minor release.

- `createTilawa(options?: NativeTilawaOptions)`. There is no microphone helper: record with your own module and call `session.pushAudio(samples, sampleRate)`. The default storage is in memory, so pass a persistent `storage` (and `pathOf`) or the model downloads on every launch. Without WebCrypto, downloads are length-checked only.
- `loadNativeModel(options: NativeModelOptions)`: `{ storage, tier?, registry?, pathOf?, providers?, threads?, allowOverBudget?, onProgress? }`. `providers` defaults to `["cpu"]`; for example `["coreml", "cpu"]` on iOS or `["nnapi", "cpu"]` on Android.
- Types: `NativeTilawaOptions`, `NativeModelOptions`, and the shared `Tilawa`, `TilawaOptions`, `TilawaModelInfo`, `LoadProgress`.

## `@tilawa/core/report`

Opt-in report capture. Nothing is kept without a recorder, and nothing is sent without `sendReport`. The full guide is [reporting.md](reporting.md).

- `createRecorder(session, options): Recorder`.
- `Recorder`: `state`, `error`, `mark(input)`, `report(input)`, `contribution(input?)`, `incidents()`, `stats()`, `dispose()`.
- `encodeReport(report): { json, wav }`, `decodeReport(json, wav)`, `reportAudio(report)`, `validateReport(json, wav)`, `encodeWav(pcm: Int16Array)`.
- `sendReport(report, options): Promise<ReportReceipt>`.
- Errors: `ConsentError`, `RecorderError` (`reason`: `disposed`, `expired`, `failed`, `empty`, `limit`, `misuse`), `ReportValidationError` (`errors: string[]`), `ReportSendError` (`status`, `body`).
- Constants: `REPORT_SCHEMA` (`"tilawa.report/1"`), `REPORT_LIMITS`.
- Types: `TilawaReport`, `Incident`, `IncidentKind`, `IncidentClaim`, `ClaimAnswer`, `ReportAudio`, `ReportConsent`, `ReportEngine`, `ReportPassage`, `RecorderOptions`, `RetainOptions`, `MarkInput`, `ReportInput`, `ContributionInput`, `RecorderState`, `RecorderStats`, `RecordableSession`, `SessionState`, `EncodedReport`, `ReportValidation`, `ReportReceipt`, `SendReportOptions`.

## `@tilawa/core/assets/*`

Plain JSON files, for bundling and self-hosting.

| File | Size | sha256 |
|---|---|---|
| `@tilawa/core/assets/ref-v2.json` | 4,749,281 bytes | `f236384ea91e81ad554ba2777bc4073c3ff41e34120343a8a474571c577b7727` |
| `@tilawa/core/assets/phrases-v1.json` | 1,452 bytes | `0277d8dd78d6042795c1332b5a322d08eb71e262a79ce8887b6d66b08c53f688` |

`ref-v2.json` is the Quran reference (Hafs, Uthmani text from Tanzil, phonemized with quran-transcript 0.6.4). `phrases-v1.json` is bundled into the SDK already.

---

## Version policy

Each part is versioned on its own:

| Thing | Versioned by | Rule |
|---|---|---|
| SDK | npm semver | 1.x keeps this API. Changes are additive only. |
| Model | `<id>@<version>` plus sha256 in the bundled registry | The default model changes only in an SDK minor release, with a changelog line. Old model URLs stay up. |
| Reference | `ref-vN` | A new reference is a new file name. A model's phoneme inventory must match its reference. |
| Events | the `v` field (`EVENT_SCHEMA_VERSION`) | Stays `1` for all of 1.x. New optional fields may be added. |
| Report | `schema: "tilawa.report/1"` | Readers reject an unknown major. New optional fields may be added (`tilawa.report/1.<minor>` is accepted). |
| Asset URLs | the `/v1/` path prefix | Files are immutable and never overwritten: `Cache-Control: public, max-age=31536000, immutable`. |

What counts as additive: new exports, new optional options, new optional event or report fields, new values in `unsupported`. Handle unknown fields and unknown future error classes gracefully.

Exceptions: `@tilawa/core/react-native` is experimental until it is validated on devices. Pre-release versions (`1.0.0-rc.x`) may still change before 1.0.0.
