# Cold-crop agreement for ownership-free seam spellings

Status: acoustic premise passes; normal SDK evidence retains four targets and live sessions retain three. Practice pending.

## Prediction

Experiment `listener-cold-crop-agrees-seam-spell`, listener, explore, based on
`ae32bb4d` (main parent `a2bc1970`). The unchanged card predicts listener −2,
overall −1 on practice `lab-v1.5u`, retaining at least three of the four named
added-word targets while reducing excise-certify's false additions.

## Mechanism

Re-land the final `listener-excise-certify-seam-waqf-adds` engine patch. Its
ownership-free generator selects entire pause-bounded greedy utterances at
ayah and marked inner-waqf seams, keeps surplus-adjacent held nasals as nūn,
requires the unchanged exact whole non-neighbour spelling, and certifies both
neighbours by read-versus-omit forward sums after cutting the selected frames.
Nearby fragments, prefixes, back-runs and non-Quran phrases remain excluded.

After selection, cold-decode the candidate's own half-open frame range with the
shipped model, fresh state and the gap path's existing 300 ms padding. Padding
supplies edge context; only tokens with centres in the original frame range
contribute to the confirmation string. Apply the same nasal handling, edge
strip and lexical spell test. Require the same lexical word on both views.
The gate may remove the parent's selection; it must not search for a substitute
candidate. No model weights, thresholds, main phones or tracker input change.

## Validation plan and limits

The temporary diagnostic replays each target using the card's supplied main
listener evidence, first through the ungated parent and then through the binary
cold-agreement gate. It records the parent's actual selected frame ranges in
both free-engine and true-position modes; labels select diagnostic targets only.
No truth timing supplies an engine candidate.

The parent's historical final write-up reports that 9:5:15 lacks a right
bounding pause. Current evidence differs: run 5 found an eligible parent span
and a gated confirmation there, as well as at the other three targets.

Unit tests cover fresh state, exact padding, frame mapping, exclusion of padding
phones, blank crops, invalid audio bounds, and removal of a parent emission.
Existing CTC/excision and lexical tests are retained during the probe.

Modal run 1 established the supplied audio paths and listener evidence names.
Run 2 supplied no acoustic measurement: a pre-existing TypeScript error in
`reference.ts` stopped the build. The failed build also left the diagnostic
shell's target array unset. Run 4 retries with fail-fast shell execution,
correct audio resolution and the card's listener evidence. A temporary explicit
string type predicate preserves the reference loader's runtime behaviour while
allowing validation to build. Run 3 requests the full six-shard test harness.

Run 4 built successfully but produced no acoustic measurement because the
transport excludes the root experiment card. Run 5 supplies the card's exact
listener ID as a diagnostic environment variable instead of reading that file.
Run 3 passed all 749 core tests. Its SDK portion found a one-sample rounding
mismatch in one new padding test; the test was corrected to use a crop endpoint
without that floating-point ambiguity, preserving the production crop code.
Run 6 requests the complete SDK suite with the corrected test. The documented
Vitest worker RPC deadline occurred only after the affected core shard's tests
had all passed and is handled by the existing test harness.

Practice score 1 was requested after the normal SDK path cleared the target
floor. Acoustic agreement alone cannot establish precision, studio behaviour,
reader exclusion or a release result.

## Target acoustic result

Modal run 5 successfully rebuilt and replayed all four targets using the card's
`exp-listener-stack-be988c-6a17f391` phone evidence. Every target had the same
parent-selected frame range and cold result in free-engine and true-position
modes. All four confirmed after agreement. These are exact engine cuts, not
oracle-timed crops. Frame indices are half-open, at 40 ms per frame.

| Target | Candidate frames | Warm observation | Cold greedy string | Same lexical word |
| --- | --- | --- | --- | --- |
| p19, 51:33:1 | [2552, 2570) | وَمِن | وَمِںں | yes |
| p50, 37:99:1 | [3662, 3693) | وَقِفُۥۥهُم | وَقِفُۥۥهُم | yes |
| p31, 4:157:24 | [1386, 1404) | وَءِن | وَءِںں | yes |
| p66, 9:5:15 | [2805, 2825) | وَءِن | وَءِںں | yes |

The ungated parent and cold-gated replays both confirm all four. Run 5's
successful shards took 104–182 seconds. No inference ran on the Mac.

Run 6 rebuilt and passed all 19 SDK tests, with the existing end-to-end test
skipped. Production wiring adds further tests for incremental run-end pause
selection, the same span across chunk boundaries, blank ownership, metadata
serialization and preservation through the existing cold-gap splice.

## Normal engine wiring

The SDK computes an acoustic superset of the parent's whole-word utterances
using the identical run-end pause test. It cold-encodes only those utterances
whose warm string spells some whole mushaf word. This preliminary filter has
no neighbour or visit requirements and cannot omit a parent candidate. It
does not emit anything. Cold strings are attached to the next warm token as
`seamCold` side evidence, so the normal serializer and rechunker preserve them.

The checker chooses and caches the parent's span unchanged, then requires a
cold string with exactly the same frame bounds and the same lexical identity,
including the unchanged phrase and nearby-fragment exclusions. Missing and
blank evidence fail closed. No external acceptance override remains.

Audio sessions certify the side evidence before the checker receives each
warm listener chunk. Recordings preserve it through the already-folded gap
splice and certify any changed final utterance bounds against the same raw
audio. The cache is indexed only by frame bounds, not labels or identities.
The raw audio is retained for crops even when evidence recording is disabled.

Run 7 requests the full build/test harness for this wiring. Run 8 performs fresh
normal SDK audio decodes of the four targets and replays the resulting evidence
through both engine arms, checking that removing the side evidence leaves the
main phonemes, posteriors and tracker unchanged.

Runs 7–8 stopped at an inventory type mismatch before inference. The pause
selector was corrected to receive the reference set's ordered token array.
Run 9 then passed builds, all 753 core tests and all 19 SDK tests, with one
existing SDK test skipped. The existing harness accepted its documented
worker RPC timeout only after that core shard's tests all passed.

Run 10 decoded the normal SDK audio path successfully. Recorded evidence
confirmed all four targets through both free-engine and true-position replays.
Live sessions confirmed p19, p50 and 4:157:24; 9:5:15 was recovered only after
the existing end-of-session recorded-evidence gap splice. Thus the built
normal path clears the three-target premise floor. Removing `seamCold`
metadata removed all four added-word confirmations in free-engine replay,
and tracker events were identical with and without the side evidence.

Comparison against the older supplied listener cache found identical warm
phones/posteriors only on p31. Run 11 compares against fresh decodes using the
starting commit's SDK session and gap code, to distinguish older cached data
from an unintended change in this experiment. Temporary baseline copies and
executable probes are removed before the final diff.


## Fresh baseline comparison

Run 11 completed successfully on all four target recordings. Against fresh
SDK decodes using the starting commit's session and gap code, every main
phoneme (with only `seamCold` removed) and every posterior entry is identical.
Tracker events also match with and without the side evidence. The differences
from the older supplied listener cache therefore do not come from this patch.
The four recorded-evidence target assignments are confirmed; the same three
live targets remain confirmed. The temporary probe and baseline source copies
were removed after this check.

This is an experiment, not an on-device cost validation. Live sessions now
retain their input audio to supply cold crops even without evidence recording,
and they perform additional short encodes for pause-bounded lexical utterances.
Peak memory, latency and total inference cost have not been benchmarked. The
fourth target requires the existing recorded-evidence gap splice; it is not a
live-stream win.
