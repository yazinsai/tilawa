# Teacher phone-loop seam additions

Status: implemented; six of six targets recovered through the native SDK. Stored-evidence practice replay was unchanged; fresh supervisor decoding and population validation remain pending. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −2 and overall −2 on practice, recovering two of the six
registered listener-charged seam additions. The card's premise requires at least
three targets with an optional teacher phone-loop marginal strictly above 0.5;
the built change must also confirm at least three targets before practice scoring.

The folded exact-word teacher gate needs a student spelling before it constructs
an insertion arc. This experiment discovers the same ayah and marked inner-waqf
seams before spelling and excision. Visits bound a whole seam, with the actual
previous-two-word prefix onset and the next two words. No zero-edit word visit
or insertion identity is required. An inner seam still needs the existing
run-end pause test.

The original fp32 KD teacher supplies warm-crop, full-inventory posteriors on
its native 40 ms clock. Its closed CTC lattice retains free boundaries and the
existing word and contiguous-side repeat transitions. The optional arc becomes
a phone loop with a consonant-onset count capped at two, entered once at the
existing PHRASE_SWITCH cost. Holding one consonant does not advance its count;
a blank followed by the same consonant is a new CTC onset. Blank, vowels and
nasal hold variants cannot supply the two consonants. Forward-backward computes
the total optional-arc marginal. Viterbi conditioned on taking the insertion provides its phones and frame
bounds. An individual no-word timing can be the global best path even when
all insertion timings together hold a majority; it must not erase the accepted
arc's phone sequence.

A strict majority permits a certificate, subject to the existing nearby-text,
prefix, back-run, substring and non-Quran phrase exclusions. The checker can
confirm the certificate at the next written word even when the student has no
extra greedy tokens. This path uses the teacher's sequence probability in place
of student confidence and local SUPPORT. Exact-word certificates retain their
existing path. Primary tokens, posteriors and tracker input are preserved.
No numerical calibration or practice-specific engine code is added.

## Validation and limits

The probability implementation shares the folded exact-word graph builder and
forward-backward recurrence. All original exact-word lattice tests passed in
Modal run 2. A small exhaustive path enumeration also matched the phone-loop
marginal, including free blank timing and repeated consonant onsets. The first
held-consonant test incorrectly required a low insertion marginal relative to
an impossible no-word read: soft posterior tails can split a strong held run
with a low-probability blank. It was corrected to test the onset constraint on
deterministic rows directly, with separate blank/vowel controls; no decision
threshold or engine behavior was changed to satisfy that test.

New checker and SDK tests exercise spelling-free seam discovery, two-consonant
Viterbi spans, JSON evidence transport, strict 0.5 selection, confirmation without
student surplus, and neighbour/prefix exclusions.

The acoustic probe uses the card's declared listener evidence and free tracker
and checker. Labels select and report the six cases only; the engine determines
seams and crop bounds. It uses the original teacher checkpoint and the SDK's
front end, with full fp32 log probabilities, rather than reconstructing top-k
rows. This is an acoustic diagnostic, not an independent listening judgment or
a population practice score.

Run 3 stopped before inference because tsx is installed in workspace packages,
not at the repository root. Run 4 then established that the Modal snapshot omits the scratch diagnostic
sources. Run 6 places those temporary sources under packages/sdk and modal,
invokes the SDK's tsx, and retains all original seam discovery and teacher
settings. Diagnostic sources were removed from the source tree and archived under the
ignored .autolab/probes directory before practice scoring.

The inherited teacher path buffers audio and publishes auxiliary warnings at
session end. Its 2.43 GB fp32 asset and CPU inference do not meet the device or
immediate-streaming requirements. This experiment tests the proposed evidence
mechanism within that existing implementation.

Modal run 5 passed the supplied six-shard build/test harness: 764 core tests
and 26 SDK tests, with one optional SDK integration test skipped. Every test
passed; the harness accepted its documented Vitest worker RPC timeout on the
checker shard. No timeout, test budget or validation enforcement was edited.

Run 8 passed the final build and focused checks after conditional Viterbi
extraction: 16 core tests and four SDK tests. Its added ambiguous-timing test
checks that a majority insertion still provides a constrained phone sequence
when the no-insertion hypothesis has the best individual timing.


The original teacher checkpoint retained optional-arc mass above 0.5 on all six
registered target seams (run 7). Run 10 replays the final implementation over the
latest complete seam window per prefix, reconstructing teacher posteriors with
the same probability floor as the SDK and checking warnings through the lab's
existing occurrence-matching function. This is target analysis only, not a
practice-set score. No scorer or label was changed.


Run 10 confirms six of six target recoveries with the final seam selection and
unchanged occurrence matching. The last completed window per prefix yielded:

| Target | Teacher loop phones | Marginal | Recovered |
| --- | --- | --- | --- |
| lab-v1.5-p07, 37:36:1 | سَرِۦۦم | 1 | yes |
| lab-v1.5-p48, 12:76:1 | جَبِر | 0.9999999975 | yes |
| lab-v1.5-p56, 32:21:1 | ۥنطِۦۦۦۦن | 1 | yes |
| lab-v1.5-p57, 51:18:1 | ۥۥۥنقَبل | 0.999999998519 | yes |
| lab-v1.5-p57, 51:20:1 | ۥمقَبل | 0.999999996964 | yes |
| lab-v1.5-p79, 20:40:25 | اقَصصَفَاا | 1 | yes |

The phone strings make no lexical identity claim. Some Viterbi spans include
pausal neighbour tails. Target recall alone therefore does not establish clean
seam precision; fresh practice decoding and scoring must determine population behavior.

Run 11 exported the original fp32 teacher with the folded export script and
dependency versions (torch/torchaudio 2.8.0 CPU, quran-muaalem 0.2.2,
transformers 4.55.2, onnx 1.23.2, onnxscript 0.7.2). Its external weights are
byte-identical to the folded teacher: 2,422,885,632 bytes, SHA-256
`d3f24efba836219f0c556ad25e3a1046f9caf36b2ed639cd283f795c78da58b4`.
The re-exported graph is 10,559,069 bytes, SHA-256
`4e1d2ca950e17a952dd15f3494d0791084e588eaac119360e337eca3a4d244ed`;
the SDK asset declaration records that actual artifact. No training, supplied
manifest or primary model change was made. The bundle contains unmodified
copies of `spliced-clean-int8/model.onnx` and `manifest.json`, selected by the
existing `.autolab/model.json` (`{"path":"bundle"}`). Native SDK validation
uses this bundle through the normal `loadNodeModel` path.

Run 12 passed native Node SDK loading, teacher inference, posterior transport,
default checker replay and unchanged occurrence matching on all six targets.
Each was missed in the baseline replay and confirmed with its teacher loop
certificate. The tracker event sequence remained byte-identical. The exported
teacher's selected phone strings and bounds matched the original checkpoint
diagnostic. This run enriched only the selected target windows.

Run 13 exercised the actual `addInsertionEvidence` function across each complete
target session, including every discovered seam and the existing exact-word gate.
All six target additions were still missed in the baseline and confirmed after
enrichment under unchanged occurrence matching. Primary tokens, confidence,
frame bounds and posterior values compared byte-identical after stripping only
auxiliary certificates; the entire tracker event sequence also compared equal.
Each shard passed its build and assertions, taking 268–350 seconds in the
container. The two p57 shards independently replayed the same session, reporting
its two target locations.

This wider replay produced 7, 6, 3, 15 and 5 confirmed added-word warnings in
p07, p48, p56, p57 and p79 respectively. These are total warning counts, including
baseline additions, not measured false-warning counts. None of these selected
sessions is a fully clean population control. Qualifying target mass and six
recoveries satisfy the premise, but the extra emissions leave precision
unresolved. No threshold or guard was adjusted from these diagnostics.

One official practice score completed at implementation commit `74a842c1`;
two scores remain available. Overall stayed at 24 wrong calls (13 missed,
11 false), listener at 23 (13 missed, 10 false), tracker at 1 (one false), and
checker at 0. All changes were zero, verdict noise.

This request replayed the card's existing listener evidence,
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. The supervisor's
`serve_practice` runs `lab.py run`, which scores that selected evidence without
fresh SDK decoding. Those files predate phone-loop certificates; the new path
has no certificate to act on. This compatibility result therefore does not
measure the new ear's population gain, false warnings on fully clean sessions,
studio regression or lost/won tradeoff. Repeating it would not resolve those
questions, so no additional practice score was requested and no narrowing was
justified from this replay.

The normal post-implementation supervisor path reads `.autolab/model.json`,
copies this experiment's bundle, and decodes phone and studio practice audio
with the modified SDK before scoring the resulting listener evidence. Leave
the implementation for that fresh decode and the fold gate. Six native
full-session recoveries pass the premise; aggregate precision remains unproved.
No killed outcome or fold was issued. No scorer, lab tooling, supplied manifest,
test data, evidence or experiment-card edit was made.
