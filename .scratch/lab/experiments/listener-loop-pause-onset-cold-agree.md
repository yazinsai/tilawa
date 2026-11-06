# Pause-onset cold agreement for the teacher phone loop

Status: implemented; acoustic agreement and SDK recovery pass on 3/6 targets.
Full core/SDK tests passed. Official practice result: overall +0, listener +0
(noise). The required overall fold gain is not met. Experiment card unchanged.

## Prediction and mechanism

Predict listener −2 and overall −1 on the practice set. Reuse the identity-free
teacher seam phone loop from `exp/listener-teacher-phone-loop-seam-adds`:
at least two consonant onsets, existing `PHRASE_SWITCH` entry price, free CTC
boundaries and contiguous repeat arcs, marginal strictly above 0.5, and the
existing local-text, back-run, prefix and non-Quran guards. The warm lattice and
its conditional insertion Viterbi extraction are unchanged from that parent.

The new condition is agreement with a second teacher pass. Its audio starts
exactly at the first detector-labelled run-end pause in the seam, after the
preceding phone's held argmax run. It has no left padding. Its right edge is
300 ms after the tracker's committed word-2 entry, on the existing 40 ms clock.
The cold lattice is `[optional loop | word 1, word 2]`, using the same loop
construction, repeat arcs, price and majority cut. A loop certificate is carried
only if both passes exceed 0.5 and their half-open Viterbi spans overlap.
Missing pause, tracker step or cold evidence means abstention. No new cut is
calibrated. The folded exact-word candidate and teacher gate are retained.
Primary phoneme identity, confidence, bounds and posterior rows are preserved;
`tracker.ts` is untouched.

The premise stop is fewer than three of six registered targets retaining cold
majority mass with overlap, or fewer than three recovered target warnings after
integration. If that happens, revert the engine and test changes, record the
killed outcome, and do not request a practice score. Otherwise request the
required practice score and leave the final fold decision to the gate.

## Validation and limits

Modal run 2 uses the supplied six-shard build and core/SDK test harness. Tests
cover the inherited warm-loop lattice, an exhaustive CTC enumeration for the
cold lattice with an empty left side, real extra consonants versus a blank
pause, overlapping versus disjoint warm/cold spans, warm-only and cold-only
evidence, missing pause or word-2 step, and preservation of primary evidence.
The existing exact-word teacher test checks that it retains its original crop
and certificate behavior.

Modal run 7 is a six-target acoustic premise probe using the experiment card's
declared primary listener evidence. It discovers every seam through the free
tracker and checker, then selects the registered incident only for reporting.
The teacher receives the SDK's exact `teacherFeatures` tensors from the warm
and cold audio crops, and its original fp32 checkpoint produces full-inventory
posteriors. Those rows are reconstructed with the SDK's probability floor and
scored by the implemented loop lattice on absolute native frame indices.
No oracle timing is used to choose either crop edge. This is target analysis,
not a practice score or an independent listening review. It does not establish
population precision or release readiness.

The first harness run exposed a new test-fixture error: its exhaustive cold
example named phone 4 while providing only IDs 0–3. The fixture vocabulary was
corrected. All existing core tests passed, and focused run 4 passed all 18 core
loop/certificate tests and all six SDK agreement/transport tests. The suite's
documented worker RPC timeout was accepted only after that shard's tests
passed; no timeout was edited. Early probes that lacked transferred sources or
the card exited before acoustic measurement; run 7 passed the declared listener
ID explicitly and received the diagnostic sources under the SDK package.

Run 7 passed on all six shards (155–171 seconds each), with the original
`quran-muaalem==0.2.2`, `transformers==4.55.2`, torch/torchaudio 2.8.0 CPU teacher.
The warm majority holds on all six targets. The proposed cold agreement retains
exactly three, so the acoustic stop criterion does not fire:

| Target | Cold marginal | Warm loop span (s) | Cold loop span (s) | Pass |
| --- | ---: | --- | --- | --- |
| p07, 37:36:1 | 1 | 144.12–144.96 | 144.12–144.96 | yes |
| p33, 26:208:1 | 0.00000163 | 156.16–157.00 | 157.40–157.52 | no |
| p48, 12:76:1 | 0.0838024 | 40.04–40.56 | 37.44–37.64 | no |
| p56, 32:21:1 | approximately 1 | 104.80–109.08 | 106.28–109.08 | yes |
| p57, 51:18:1 | approximately 1 | 92.36–95.76 | 94.56–95.76 | yes |
| p57, 51:20:1 | 0.00154101 | 110.40–112.76 | 111.68–111.84 | no |

The p33 pause onset is 156.96 s, after the warm loop already used most of the
planted span; the crop intentionally excludes that material. At p48 the cold
loop moves to a different interval and loses majority mass. The second p57
loop overlaps but is well below 0.5. No crop boundary or parameter was adjusted
in response. These are mechanism limits, not conclusions about label validity.

Run 8 tests the resulting certificates through JSON transport, native frame
cells, the default free tracker/checker and unchanged occurrence matching.
It compares target confirmations with baseline and asserts primary token,
posterior and tracker-event identity. The initial diagnostic reported matches
for p07 and p56 (confirmations at 148.96 s and 113.84 s respectively), but
matched only confirmation timestamps. That was stricter than the scorer,
which matches the first warning state and then requires eventual confirmation.
Its initial two-target count cannot be used for the stop decision. These replays
also inject only each selected certificate, rather than invoking the full SDK
producer.

Run 9 exports the original fp32 teacher through the existing exporter into the
experiment's persistent Modal scratch bundle, alongside unmodified copies of
the shipped student graph and manifest. Runs 10 and 11 invoke the actual SDK
`addInsertionEvidence` producer with that teacher over all five target sessions,
then replay the resulting evidence through the free engine. All five sessions
completed successfully, with primary tokens, posteriors and tracker events
unchanged. The producer retains the p57 51:18:1 certificate and the checker
confirms it at 102.12 s. Run 12 replays the saved production certificates,
collects every warning state, and invokes the unchanged session scorer's
assignment logic to establish the correct target count. It confirms three
target gains: p07, p56, and p57 51:18:1. The latter is suspected at 98.44 s,
inside its occurrence window, and confirmed at 102.12 s. The other three
targets remain missed. The stop criterion therefore does not fire. The
five-session diagnostic also reports 18 more wrong calls in total; this is
target-session analysis, not a full practice score, and does not justify
stopping before the required score.

Run 13 repeats the supplied full core/SDK test harness on the final candidate,
including the corrected teacher graph declaration. All six shards pass:
767 core tests and 28 SDK tests, with one optional help end-to-end test skipped.
The documented Vitest worker RPC deadline occurs after all affected core tests
pass; the supplied harness accepts that error. No timeout or harness is changed.
Temporary diagnostic sources have been removed from the reviewable worktree
and archived under ignored `.autolab/probes`. `.autolab/model.json` requests the
persistent scratch `bundle` for practice scoring; its primary graph and
manifest are unchanged copies of the shipped student. The teacher declaration
names the actual exported graph bytes and checksum.

## Official practice result

Practice score 1 of 3 was run on implementation commit `056fcfa2` and returned
`noise`. No parameters were tuned and no further score was requested.

| Charged component | Main | Experiment | Change |
| --- | ---: | ---: | ---: |
| Overall | 24 (13 missed, 11 false) | 24 (13 missed, 11 false) | 0 |
| Listener | 23 (13 missed, 10 false) | 23 (13 missed, 10 false) | 0 |
| Tracker | 1 | 1 | 0 |
| Checker | 0 | 0 | 0 |

This does not meet the required overall gain of three. It also does not show
the fresh SDK replay's three target gains or its added wrong calls in the
aggregate practice counts. The returned result contains aggregate counts and
the implementation SHA, without a fresh evidence trace or per-incident
assignments. It therefore does not establish that the new teacher certificates
reached the scored replay, or that cold agreement improves practice precision.
The SDK recovery count is three, so the instructed premise stop does not fire.
The implementation is left on the experiment branch for the supervisor's gate.

No scorer, lab tooling, supplied labels, manifests or evidence are modified.
Raw artifacts remain under this experiment's Modal scratch directory.


The inherited teacher path buffers session audio and publishes auxiliary seam
judgments at session end. Its fp32 asset is about 2.43 GB. Adding a cold pass does
not resolve those device-size and warning-latency limits; this experiment tests
the acoustic agreement mechanism within the existing implementation.
