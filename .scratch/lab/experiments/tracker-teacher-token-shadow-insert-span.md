# Teacher-token shadow insertion intervals

Status: implemented. Three of six targets retained in the SDK replay; practice scoring pending.

Base: `4ead514893350e880bba91d34b0b26395da49cbd`.
Card: `tracker-teacher-token-shadow-insert-span`, tracker, explore. The card is unchanged.

## Prediction and mechanism

Predict tracker 0, overall −1 on lab-v1.5u with the card's
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0` listener evidence.
The premise requires at least three of the six named targets to have an
insertion hypothesis on teacher greedy tokens that beats continuation by the
existing commit margin, with an interval overlapping the planted word.

The trial runs the ordinary global tracker search on successive teacher-token
prefixes, bounded by a seam pause and the primary tracker's step onto the next
written word. It retains the existing 48-phone history, shortlist generation,
confidence-aware alignment paths, destination word grouping, word edit costs
and commit margin. One leading heard word is priced with the existing
`insertionCost("", 0, phone)`; the following units read the written continuation.
The comparison is read-only and uses separate tracker instances.

A winning half-open frame interval permits an identity-free teacher phone loop
with at least two consonant onsets in the existing closed seam lattice. Loop
phones must stay inside the interval; boundary blanks may wait for the next
written word. The written words, pronunciation variants, repeat transitions and
`PHRASE_SWITCH` entry price are unchanged. A marginal strictly above 0.5 passes
through the existing neighbour, prefix, back-run and non-Quran exclusions.
The SDK carries passing judgments through the folded insertion-certificate
route, and the checker replays them. The primary greedy string and tracker
events remain independent of the shadow.

## Boundary checks and comparison control

Supervisor runs 5–7 replayed all six complete primary phone sessions against
their supplied stored teacher greedy tokens on the same 40 ms inventory and
frame clock. Labels selected reports and measured overlap; they did not supply
word boundaries, spellings or hypotheses.

An initial fixed-entry comparison found no qualifying target. A broader
destination-unit control found p07 and p56 when started at the end of a student
blank run. Expanding the preceding-word window did not add a target. Starting
at the beginning of the same blank run added p57 at 51:18:1: teacher speech
can occur inside frames the student calls blank, so trimming through the whole
student blank run discards the evidence being tested.

p33's preceding-word entries were checked explicitly. Its teacher hears
`كَرِۦۦم` immediately after the previous word's final consonant, without the
existing minimum pause before it. The detected pause starts at frame 3924,
after most of the planted word. Moving the preceding tracker boundary earlier
does not supply a qualifying pre-insertion pause.

With the broad per-alignment control, run 8's bounded loops produced confirmed
checker warnings on p07, p56 and p57 at 51:18:1. On p57 the interval includes a
previous-word tail, but the lattice leaves that tail with the written word and
extracts only `قبل`. These are provisional control results.

The confidence-aware path implementation exposed a comparison bug in a clean
unit test: a poor candidate's segmentation gave clean `ءَ` an apparent gain of
6.5 edits against that same candidate's 10.5-edit continuation. An insertion
must beat the cheapest continuation across the same candidate paths. The
corrected comparison minimizes continuation first, then compares every
insertion with that minimum. No cost, threshold or practice-tuned constant
changes. The final result below uses this corrected comparison.

## Corrected target result

Supervisor run 12 passed all six shards and the complete core/SDK test harness.
The corrected tracker comparison and ordinary SDK certificate route confirm
three targets. The premise therefore survives. These are confirmed checker
events after JSON serialization and native frame-cell rechunking, rather than
counts of candidate intervals or lattice wins alone.

| Target | Representative tracker interval | Gain / commit margin, edits | Bounded-loop phones | SDK warning |
| --- | --- | --- | --- | --- |
| p07, 37:36:1, 144.12 s | frames 3603–3624 | 9.0 / 1.364 | `سَرِۦۦم` | confirmed |
| p48, 12:76:1, 39.98 s | none | no winning hypothesis | none | absent |
| p56, 32:21:1, 106.76 s | frames 2669–2726 | 13.5 / 1.111 | `قِۦۦۦۦن` | confirmed |
| p57, 51:18:1, 95.26 s | frames 2313–2394 | 11.0 / 1.364 | `قبل` | confirmed |
| p57, 51:20:1, 112.26 s | none | no winning hypothesis | none | absent |
| p33, 26:208:1, 156.16 s | none | no winning hypothesis | none | absent |

The representative bounded marginals are approximately 1 on p07 and p56,
and 0.9999792 on p57. All six full-session SDK replays compared equal on
primary phone identities, confidences, frame bounds, posterior rows and the
entire primary tracker event sequence. Detailed intervals, loop results and
emissions are retained in
[tracker-teacher-token-shadow-insert-span-results.json](tracker-teacher-token-shadow-insert-span-results.json).

Run 12 passed 768 core tests and 28 SDK tests, with one optional SDK integration
test skipped. The core count includes three fixed-entry trial tests removed
with their unused helper after validation. The retained tests cover clean
continuation, unchanged tracker events, missing shadow evidence, exhaustive
CTC path mass, interval confinement, blank waiting, held consonants, missing
intervals and SDK certificate transport. Run 13 checks the cleaned final tree.
The exact-word lattice retains its original floating-point evaluation order;
the diagnostic source is archived under ignored `.autolab/probes/`.

## Validation limits

The supplied teacher evidence is a stored full-session decode, with top five
classes per frame out of 42 phonemes plus blank. SDK target
replays supply native-clock slices of those stored tokens and posterior rows
through the ordinary insertion-teacher interface. This tests shadow discovery,
bounded lattice scoring, certificate transport, JSON/frame rechunking and
checker emission. It does not validate a fresh warm-crop fp32 teacher inference.
The supplied model directories contain no teacher ONNX asset; the attempted
fresh-inference diagnostic stopped at that prerequisite.

Teacher evidence is correlated acoustic evidence, not an independent listening
opinion. No labels, audio, supplied evidence, model manifests, scorer or lab
tooling were edited. Target replay does not measure population precision,
studio regressions, fully clean sessions or leave-one-reader behavior. Those
remain practice-score and fold-gate checks.
