# Word-ear identity confirmed by undistilled SSL tokens

Status: implemented, committed and validated on three target slots; awaiting the supervisor's fresh-decode fold gate. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −2 and overall −1: two of the five registered additions won
with about one loss. The required premise is at least three registered targets
with an eligible word-ear extra word that the independent SSL ear confirms.

Reuse the saved short-crop Whisper-small int8 word decoder and the completed
plain-CTC-v4 undistilled w2v-BERT-2.0 int8 ear, without training or calibration.
The word ear proposes unmatched whole words in its sequence diff against the
next two written words over a detector-pause/tracker-step seam span. Existing
whole-mushaf-word, neighbour, prefix, local-fragment, back-run and formula guards
exclude ineligible identities. Its autoregressive confidence is ignored.

The SSL decoder retains its native 20 ms clock, original full-session chunk
phase, actual bounded left history and lookahead. It does not average its
short consonant spikes onto the primary 40 ms clock. A closed identity check
aligns each word-ear proposal to the SSL tokens using the existing substitution,
insertion, deletion and held-sound costs. The complete proposed word must fit
with at most one hard edit and strictly beat all local word, prefix and
contiguous back-run explanations on the same observed interval. A tie abstains.
The primary string, tracker and folded teacher insertion gate are unchanged.

## Validation and limits

The preliminary probe discovers seams from the card's primary evidence using
the actual free tracker. Labels select registered sites only after discovery;
they never supply crop bounds or candidate identities. It tests every contiguous
SSL token interval within each discovered span, a generous feasibility bound
before any emission, pause-boundary or certificate-transport restriction.
Failure to reach three targets even under that bound kills the premise.

Modal run 1 confirmed that both recorded exports remain available in supervisor
scratch. Word asset sizes match the short-crop parent; run 9 also verified all hashes;
the SSL export is the completed eligible plain-CTC-v4 model, not its collapsed
optimization predecessors. The auxiliary word and SSL models total about
1.23 GB; this probe does not establish on-device memory or latency suitability.

Unit coverage checks hard versus soft costs, full-word alignment, held-sound
normalization, strict competitor ranking and ties, local prefix/back-run
construction, native SSL timing and context dispatch, word reconstruction,
seam discovery and lexical proposal guards. No scorer, lab tooling, test data,
supplied manifest, evidence or experiment card is changed.

Practice score calls: 1 of 3. The first call completed with overall and
listener changes of zero on the card's stored primary evidence.

## Acoustic premise probe (Modal run 6)

The probe used the saved short-crop word decoder and native SSL decoder on
the generic spans discovered from the primary stored evidence. The audio
path is `session.audio`, which the session generator sets to the phone rendering;
the probe did not substitute studio audio. It found:

| Registered site | Word ear and SSL result |
| --- | --- |
| p07, 37:36:1 | Word ear proposes `سليم` or `سبيل`; SSL `تَلِم` cannot confirm either within one hard edit. |
| p48, 12:76:1 | Word ear proposes `كذب`; SSL `فَدرفَبَدَءَبِءَ` cannot confirm it. |
| p56, 32:21:1 | Planted `طين` fails the hard-edit budget. Another word-ear proposal, `عنهم`, agrees with SSL `عننَهُم` at 111.94–113.10 s. |
| p57, 51:18:1 | Proposed `قبل` agrees with SSL subinterval `َبڇل` at 95.36–95.76 s: cost 10, one hard edit, best competitor 15. |
| p57, 51:20:1 | Proposed `قبل` agrees with SSL `قَبڇل` at 112.34–112.74 s: cost 0, zero hard edits, best competitor 20. |

Thus three slots have an eligible proposal/SSL agreement, but only two
agreements identify the planted word itself. In p56 the agreeing material is
later than the planted interval (106.76–109.075 s) and may be a fragment of
the written continuation. It must be evaluated through the unchanged warning
matching and fold gate rather than credited as an acoustic identity recovery.
The subinterval search remains unchanged for emission verification.

## Built emission verification (Modal run 8)

A second diagnostic fed the actual saved word/SSL acoustic outputs through
`addWordSslEvidence`, the core checker and the unchanged session-event matching
rules. It covered the four target sessions only, not the full practice set.
Primary tokens and posterior rows were retained; labels selected only the
reported targets, never an engine identity or bound.

The built change recovered 3 of 5 registered misses: p57 at 51:18:1 (warning
95.76 s), p57 at 51:20:1 (112.76 s), and p56 at 32:21:1 (113.12 s). p07 and p48
stayed missed. The p56 warning is credited by the existing location/occurrence
and six-second confirmation window, even though it names the later `عنهم`
fragment rather than the planted `طين`. No clean-seam precision is established
by this selected-site diagnostic.

The SDK now carries dedicated word-agreement certificates with the measured
edit count and strict rank advantage; it does not fabricate a teacher marginal.
The checker rechecks the folded lexical, fragment, phrase and seam guards,
then emits through the same end-of-session seam path with duplicate
suppression. The normal Node directory loader installs both declared ears
lazily and releases their backends with the primary model. The existing
teacher file pins and its insertion lattice code are unchanged.

## Final code and bundle checks

Modal runs 7/9 exercised the unchanged supplied six-shard test harness: all
764 core tests passed, as did all SDK tests except an assertion in the new
loader test that checked the wrapped release function as a spy. That assertion
was corrected to retain and check the original primary release spy. Run 10
built core and SDK and passed all three tests in that file. Together the
checks pass all 47 SDK tests, with one optional help end-to-end test skipped.
No production change followed these checks.

Run 9 verified the SHA-256 of every primary, teacher, word and SSL model file
and copied the original files into this experiment's scratch bundle. The
primary model remains `714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b`;
the teacher and word/SSL hashes match their existing registry pins. The SSL
export's original geometry is chunk 16, left 128, right 7 on its 20 ms clock.
The primary manifest and SSL export metadata were copied byte for byte.
`.autolab/model.json` points to `bundle` for supervisor decoding.

The paired ears add 1,229,042,497 model/tokenizer bytes before runtime memory,
in addition to the existing primary and teacher. This is a lab inference
experiment, not an established device-budget or latency improvement. No
training, numerical tuning, scorer change or data edit was performed.

## Practice score 1 and its evidence-path limit

The authorized `practice-score .` call completed on commit `a10bd14e-dirty`:

| Wrong calls | Main line | This change | Delta |
| --- | --- | --- | --- |
| Overall | 24 (13 missed + 11 false) | 24 (13 missed + 11 false) | 0 |
| Listener | 23 (13 missed + 10 false) | 23 (13 missed + 10 false) | 0 |
| Tracker | 1 (0 missed + 1 false) | 1 (0 missed + 1 false) | 0 |
| Checker | 0 | 0 | 0 |

The score is unchanged, below the improvement requirement. This call does
not exercise the auxiliary models: `serve_practice` invokes the normal lab
run, which selects the experiment card's stored listener evidence. It does
not read `.autolab/model.json` or perform a fresh decode. Those stored rows
contain no new word-agreement certificates, so the checker has no new votes
to consume. Neither the experiment card nor this scoring path was changed.

The supervisor's listener validation after implementation uses a different
existing path: it copies the complete directory selected by `.autolab/model.json`,
decodes phone and studio audio with the branch's SDK, then scores the resulting
listener evidence. The new Node loader and session enrichment are wired on
that path. The final four-session diagnostic ran the installed bundle
through the normal SDK audio API, with 20 ms pushes, all seam sites and both
auxiliary ears plus the unchanged teacher. It established three actual SDK
target warnings, so the implementation is retained for the supervisor's gate. No fresh
full-practice improvement or precision result is claimed from the cached
score.

## Normal SDK audio verification (Modal run 11)

All four sessions loaded the installed primary bundle through `loadNodeModel`
and reported teacher, word ear and SSL ear present. Phone WAVs were pushed
in 20 ms chunks into a normal free-recitation session with recording enabled.
No seam, candidate, bound or model output was supplied from labels or the
earlier cached probe. All generic seam sites ran, and the session's `end()`
published its new certificates and warnings. Each model backend released
without error.

| Session | Agreement certificates | Certificate-supported warnings | Registered targets caught |
| --- | --- | --- | --- |
| p07 | 20 | 5 | 0 |
| p48 | 1 | 1 | 0 |
| p56 | 8 | 2 | 1 |
| p57 | 53 | 6 | 2 |

The actual SDK path caught the same three registered target slots as the
cached acoustic replay, at 95.76 s, 112.76 s and 113.12 s. The third still
names later `عنهم`, not the planted `طين`. Eleven other certificate-supported
warnings appeared elsewhere in these planted sessions; this selected-session
run does not establish full-practice precision, clean-session silence or a
fold-gate gain. Those losses are not used to kill a change that reaches three
target slots: the supervisor's fresh full-set decode and gate must judge it.

Comparison of the new primary token tuples with the older stored listener
artifact matched exactly for p57 and differed for p07, p48 and p56. This is
a comparison across decode artifacts, not a before/after test of the new
metadata function. Unit tests directly verify that enrichment preserves all
primary token values, posterior rows and existing teacher certificates. The
primary streaming decoder, tracker and teacher lattice source files are
unchanged. Run 12 confirmed that the supplied primary manifest and the bundle
manifest are exactly equal. Both declare the primary SHA-256 verified in
run 9; the experiment preserves the supplied main model.

## Outcome

Leave the engine change in place: the normal audio SDK path meets the
three-target prerequisite. The registered physical word is independently
identified at only the two `قبل` sites; the p56 slot is caught by a later
continuation fragment. Extra warnings elsewhere and the 1.23 GB auxiliary
asset cost remain material limits. The stored-evidence practice score is
unchanged; it does not show a gain or establish the fresh-evidence fold gate.

The supervisor must perform its existing fresh phone/studio decode from the
selected bundle and let its gate weigh wins, losses, readers and fully clean
sessions. No fold, numeric tuning, scoring-path edit, experiment-card edit,
data edit or manifest mutation was made. One of the three permitted practice
scores was used; the second and third were not requested because the same
stored rows cannot carry new model judgments.

Implementation commits: `69ec12fb` (the acoustic proposal/agreement prototype)
and `a10bd14e` (SDK assets, certificate transport and emission). Validation
results are in Modal runs 6, 8, 9, 10, 11 and 12 and practice result 1.
