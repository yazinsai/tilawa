# Listener: stable seam spellings across streaming grids

## Prediction

Experiment `listener-grid-stable-seam-spellings`, listener/explore, starts at
`2df59d0e` (main engine `41a161a7`). The card predicts three of five target
additions retained, about three losses, listener −2 and overall −1 wrong calls.
The premise requires at least three target words to survive the unchanged spell
test over the same candidate frames on a half-chunk-shifted full-session pass.

## Mechanism implemented

Re-landed the final `listener-excise-certify-seam-waqf-adds` candidate generator,
its surplus-adjacent nasal conversion, neighbouring-text fragment exclusion,
and excise-then-forward-sum neighbour certification. The enumeration searches
whole pause-bounded utterances at ayah and marked inner-waqf seams regardless
of ownership. Both neighbours must beat omission by the existing
`PHRASE_SWITCH`; no calibration constant changed.

The shipped model's 45-frame fbank window advances by 32 frames and emits eight
40 ms rows. An independent second stream begins 16 fbank frames later and
carries its own attention, convolution, embedding and processed-length states
through the full session. All shifted rows are retained on the original frame
clock, including the shifted chunk's second half. The first four uncovered
frames are unknown. Original posteriors and greedy phones remain the main
listener evidence; no grids are merged.

Only after the parent selects and caches its candidate does the gate inspect
the shifted greedy token onsets inside that exact half-open frame interval.
It applies the same edge strip, nasal conversion, soft-cost whole-word spell
test and text/phrase exclusions, and requires the same lexicon word. Missing
rows fail closed. A failed candidate does not trigger a search for another word.

Argmax rows are carried as optional token metadata so ordinary evidence
re-chunking retains them. The folded cold-gap path uses the original single
stream; if its token replacement removes a metadata carrier, the original
full-stream rows move to the next surviving token. No crop supplies agreement
evidence. The additional stream doubles model calls/cache storage and adds
160 ms of lookahead buffering; device performance was not measured.

Implementation is recorded in `a004f69d` and scope fix `8f896d58`.

## Validation method

The retained probe receives the five session/time pairs as diagnostic arguments,
never as engine constants. After full-session inference it attaches the shifted
rows to the supplied main evidence, preserving the folded gap phones, and runs
both the production engine and an ungated parent control with the same free
tracker. The control is generated only inside the remote diagnostic run by
removing the final agreement condition. Target confirmation events and fixed
frame spelling comparisons are retained separately from practice scoring.

Modal run 1 found a metadata type mismatch in the gap-preservation helper.
Runs 3 and 4 found that the phrase exclusion is local to the checker judge;
the agreement helper was moved into that scope. None of these attempts reached
acoustic inference or yielded target measurements. Run 2 verified the supplied
practice paths and model location. Runs 5 and 6 validate the corrected build.

Unit coverage includes shifted feature offsets, independent cache progression,
primary posterior/phone parity, greedy-run invariance to push size, terminal
padding, missing-row rejection, exact frame/onset clipping, positive ayah and
inner-waqf cases with blurred neighbours, absent/disagreeing second grids, and
preservation of full-stream rows during cold-gap replacement. The parent CTC
excision and lexical boundary tests are retained unchanged apart from providing
the second-grid evidence required by the new gate.

Modal run 5 passed the TypeScript builds and all six test shards: 749 core tests
and 19 SDK tests passed; one existing SDK end-to-end test was skipped. The
repository harness accepted the documented Vitest worker RPC deadline after
all tests on that shard had passed. No test failure was accepted or suppressed.

## Acoustic and engine result

Modal run 6 completed inference on all five full phone sessions. Both the
ungated parent control and the gated production engine confirmed four targets.
No target spelling changed on the shifted grid, including the target that does
not get an emission from the parent.

| Target | Main / shifted greedy spelling in the labelled span | Parent / gated confirmation |
| --- | --- | --- |
| p19, 51:33:1, 102.06 s | وَمِںںں / وَمِںںں | yes / yes |
| p50, 37:99:1, 146.47 s | وَقِفُۥۥهُم / وَقِفُۥۥهُم | yes / yes |
| p31, 4:157:24, 55.47 s | وَءِںںں / وَءِںںں | yes / yes |
| p66, 9:5:15, 112.20 s | وَءِںں / وَءِںں | no / no |
| p49, 26:123:1, 152.25 s | كَرِۦۦم / كَرِۦۦم | yes / yes |

The confirmations occur at 106.36, 153.24, 59.40 and 158.48 seconds respectively,
with observations وَمِن, وَقِفُۥۥهُم, وَءِن and كَرِۦۦم. The 9:5 target also has
same-word exact-frame agreements, but no parent emission; agreement alone does
not create a site. Four confirmations clear the three-target premise threshold,
so practice score 1 was requested before considering any narrowing.

The retained results JSON contains the events from both arms and the labelled
span/candidate diagnostics. Labelled-span strings in the table are descriptive;
the production gate uses the parent's actual candidate cut, not truth timings.
The target probe establishes recall and frame-grid agreement, not precision,
studio behaviour, release performance or a full practice delta.

A separate baseline audit (Modal run 10) reconstructed the starting checker by
removing only this experiment's additions and verified its complete source
SHA-256 against commit `2df59d0e` before running it. The reconstructed source is
byte-identical to that commit. With the same supplied main evidence and free
tracker, the baseline confirmed **0 of 5** targets. Thus the four gated target
confirmations are new wins, rather than warnings already present on this
worktree's starting engine.

The baseline audit initially exceeded the command transport limit; temporary
root text files were also absent from remote snapshots. The final retained
baseline probe reconstructs and hashes the source inside the remote run,
requiring neither a large embedded source nor an exported baseline file.
Temporary local baseline files were removed. These transport attempts yielded
no measurements. The successful audit's events are retained in the baseline
results JSON.

## Requested practice score and its limit

Practice score 1 returned for engine commit `8f896d58`:

| Attribution | Main | This requested replay | Change |
| --- | ---: | ---: | ---: |
| Overall | 30 (20 missed, 10 false) | 30 (20 missed, 10 false) | 0 |
| Listener | 29 | 29 | 0 |
| Tracker | 1 | 1 | 0 |
| Checker | 0 | 0 | 0 |

The receipt's verdict is noise. It does **not** measure the fresh shifted-grid
listener. Read-only inspection of the existing supervisor flow establishes the
reason: `serve_practice` invokes the lab replay directly with the card's cached
listener evidence, without first decoding this branch's SDK. That evidence
predates the newly introduced `seamGrid` metadata, so the agreement gate fails
closed and creates no excise-certify emissions. The earlier target replay
explicitly supplied the full-session shifted rows and produced four new
confirmations; the cached replay cannot adjudicate their precision.

The supervisor's normal validation flow, after this executor finishes, runs the
SDK decode with this worktree as `LAB_SRC`, then scores its newly named listener
evidence. The existing SDK decoder records phoneme objects directly, retaining
the metadata; the cold-gap preservation helper retains the rows through that
path. Neither the scorer, its tooling, the card, the test data nor manifests
were changed to force a different practice replay. A second cached practice
request would repeat the same limitation, so only one of the three requests was
used. The practice receipt is retained alongside this write-up.

## Final state and limits

The premise passes: **four new target confirmations**, with identical spellings
on both full-session grids. Builds, all 749 core tests and all 19 SDK tests pass,
with one existing SDK test skipped. The implementation and all diagnostic
results are committed; no engine change was narrowed after the target result.

Leave the change in place for the supervisor's fresh decode and fold gate.
Precision on clean seams, incident losses, studio behaviour, reader exclusion
and a full practice delta for the actual two-grid listener remain unestablished.
The cached +0 receipt supplies no evidence for either improvement or regression
of this new evidence path. No model export, training, manual fold or discard was
performed, and no premise-kill outcome is warranted.
